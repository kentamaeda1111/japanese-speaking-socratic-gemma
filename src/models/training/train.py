# 1. Environment Setup and Imports
# 1.1 Import Dependencies
import torch
import gc
import psutil
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import json
import logging
from datetime import datetime
import os
import numpy as np
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils.config import get_api_keys

# 1.2 Global Constants and Environment Variables
# Define global constants
DIALOGUE_JSON_PATH = "data/dialogue/processed/kaggle_model_50.json"  # Path to dialogue JSON file
MAX_SEQUENCE_LENGTH = 256  # Maximum number of tokens per dialogue
MAX_TOKENIZE_LENGTH = 256  # Maximum token length during tokenization

# Setup output directory paths
BASE_OUTPUT_DIR = "models/attention-tuned"  
MODEL_OUTPUT_DIR = f"{BASE_OUTPUT_DIR}/model"
LOG_OUTPUT_DIR = f"{BASE_OUTPUT_DIR}/logs"

# Create directories
for dir_path in [BASE_OUTPUT_DIR, MODEL_OUTPUT_DIR, LOG_OUTPUT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["WANDB_DISABLED"] = "true"

# API key
try:
    # Get API key
    api_keys = get_api_keys()
    huggingface_token = api_keys['huggingface_api_key']
    
    # Set Hugging Face API key
    os.environ["HUGGINGFACE_TOKEN"] = huggingface_token
    
    logging.info("Successfully loaded Hugging Face API key")
except Exception as e:
    logging.error(f"Error loading API keys: {str(e)}")
    raise

# 1.3 Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{LOG_OUTPUT_DIR}/training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

# Output settings
logging.info(f"Using dialogue file: {DIALOGUE_JSON_PATH}")
logging.info(f"Max sequence length: {MAX_SEQUENCE_LENGTH}")
logging.info(f"Max tokenize length: {MAX_TOKENIZE_LENGTH}")

# 2. Data Preprocessing Pipeline
# 2.1 Tokenizer Setup and Initialization
model_name = "google/gemma-2-2b-jpn-it"
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    token=huggingface_token 
)

# 2.2 Data Validation Functions
def validate_message_format(message):
    """Validate message format"""
    if not isinstance(message, dict):
        return False
    if 'role' not in message or 'content' not in message:
        return False
    if message['role'] not in ['user', 'model']:
        return False
    if not isinstance(message['content'], str):
        return False
    return True

def validate_dataset(dataset):
    """Validate dataset structure"""
    first_item = dataset[0]
    print("Validated first item structure:")
    print(f"Keys: {first_item.keys()}")
    print(f"input_ids type: {type(first_item['input_ids'])}")
    print(f"input_ids length: {len(first_item['input_ids'])}")
    return dataset

# 2.3 Data Set Preparation Function
def prepare_dataset():
    conversations = []
    
    try:
        with open(DIALOGUE_JSON_PATH, 'r', encoding='utf-8') as f:
            dialogue_data = json.load(f)
            
        for dialogue in dialogue_data:
            messages = dialogue.get('messages', [])
            
            # Validate message format
            if not all(validate_message_format(msg) for msg in messages):
                logging.warning(f"Skipped dialogue due to invalid message format")
                continue
                
            # Construct conversation in user->model order
            current_conversation = []
            valid_sequence = True
            
            for i in range(0, len(messages)-1, 2):
                if (i+1 < len(messages) and 
                    messages[i]['role'] == 'user' and 
                    messages[i+1]['role'] == 'model'):
                    current_conversation.extend([messages[i], messages[i+1]])
                else:
                    valid_sequence = False
                    break
            
            # Add only valid conversations
            if valid_sequence and current_conversation:
                # Apply Gemma chat template
                formatted_text = tokenizer.apply_chat_template(
                    current_conversation,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                # Check token count
                tokens = tokenizer.encode(formatted_text)
                if len(tokens) <= MAX_SEQUENCE_LENGTH:
                    conversations.append({"text": formatted_text})
                else:
                    logging.warning(f"Skipped conversation due to length: {len(tokens)} tokens")
            
    except Exception as e:
        logging.error(f"Error processing dialogue file: {str(e)}")
        raise
    
    if not conversations:
        raise ValueError("No valid conversations found in the dialogue file")
        
    logging.info(f"Processed {len(conversations)} valid conversations")
    return Dataset.from_list(conversations)

# 2.4 Data Processing Pipeline Function
def tokenize_function(examples):
    result = tokenizer(
        examples['text'],
        truncation=True,
        max_length=MAX_TOKENIZE_LENGTH,      # Use global setting
        padding='max_length',
        add_special_tokens=True,
        return_tensors=None,
    )
    return result

def preprocess_function(examples):
    # Focus on Socratic tone and inquiry patterns
    socratic_patterns = [
        # Question patterns
        "かね", "だろうか", "のかね", "ではないかね",
        # Question introduction
        "では", "について",
        # Second person (characteristic of mature tone)
        "君は", "君が", "君の"
    ]
    
    # Get tokenized text
    texts = tokenizer.batch_decode(examples['input_ids'])
    new_attention_masks = []
    
    for text, mask in zip(texts, examples['attention_mask']):
        if not isinstance(mask, list):
            mask = mask.tolist()

        new_mask = mask.copy() 
        
        # Split text
        sentences = text.split('。')
        current_pos = 0
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            # Detect and highlight Socratic patterns
            for pattern in socratic_patterns:
                if pattern in sentence:
                    # Identify pattern position
                    pattern_tokens = tokenizer.encode(pattern, add_special_tokens=False)
                    pattern_len = len(pattern_tokens)
                    
                    # Highlight tokens containing the pattern and its surroundings
                    pattern_start = current_pos + len(tokenizer.encode(sentence, add_special_tokens=False)) - pattern_len
                    for i in range(max(0, pattern_start - 2), min(len(mask), pattern_start + pattern_len + 2)):
                        new_mask[i] = 1.0  # Max attention to pattern part
            
            # Update position for each sentence segment
            current_pos += len(tokenizer.encode(sentence + '。', add_special_tokens=False))
        
        # Special token masks are set to 1.0
        if tokenizer.bos_token_id is not None:
            new_mask[0] = 1.0  # BOS token
        if tokenizer.eos_token_id is not None:
            new_mask[-1] = 1.0  # EOS token
            
        new_attention_masks.append(new_mask)

    examples['attention_mask'] = new_attention_masks
    return examples

# Add special tokens to tokenizer
tokenizer.add_special_tokens({
    'additional_special_tokens': [
        '。', '、', '！', '？',  # Punctuation marks
    ]
})

# 2.5 Data Set Preparation
# Prepare base dataset
dataset = prepare_dataset()
logging.info(f"Total dataset size: {len(dataset)}")

# Validate dataset structure
print("Dataset structure:")
print(dataset[0])  # Display first element
print("\nDataset features:")
print(dataset.features)

# Optimize dataset batch processing
dataset = dataset.select(range(len(dataset))).shuffle(seed=42)

# Optimize dataset processing
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    batch_size=32,  
    num_proc=4,     
    load_from_cache_file=True,
    desc="Tokenizing datasets",
    remove_columns=dataset.column_names,
)

# Apply preprocessing
tokenized_dataset = tokenized_dataset.map(
    preprocess_function,
    batched=True,
    desc="Applying attention masking"
)

# Final dataset validation
tokenized_dataset = validate_dataset(tokenized_dataset)

# 3. Model Architecture
# 3.1 Quantization Setup (BitsAndBytes)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_storage=torch.uint8,
)

# 3.2 Basic Model Initialization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation='eager',
    token=os.environ["HUGGINGFACE_TOKEN"],
    max_memory={0: "4GiB", 1: "4GiB", "cpu": "24GB"}
)

# 3.3 LoRA Setup and Application
# LoRA parameter setup
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

# Create and initialize LoRA model
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# 3.4 Model Optimization Setup
# Optimize cache setup
model.config.use_cache = False

# Data collator setup
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8
)

# 4. Training Framework
# 4.1 System Resource Monitoring
def log_memory_usage():
    """Log memory usage"""
    import psutil
    import torch
    
    # CPU memory
    process = psutil.Process()
    cpu_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # GPU memory
    gpu_memory = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_memory.append({
                'device': i,
                'allocated': torch.cuda.memory_allocated(i) / 1024 / 1024,  # MB
                'reserved': torch.cuda.memory_reserved(i) / 1024 / 1024,    # MB
                'max_allocated': torch.cuda.max_memory_allocated(i) / 1024 / 1024  # MB
            })
    
    logging.info(f"CPU Memory usage: {cpu_memory:.2f} MB")
    for gpu in gpu_memory:
        logging.info(f"GPU {gpu['device']} Memory:")
        logging.info(f"  - Allocated: {gpu['allocated']:.2f} MB")
        logging.info(f"  - Reserved: {gpu['reserved']:.2f} MB")
        logging.info(f"  - Max Allocated: {gpu['max_allocated']:.2f} MB")

def clear_memory():
    """Memory release function during training"""
    import gc
    gc.collect()
    torch.cuda.empty_cache()

# 4.2 Metrics Calculation System
def compute_metrics(eval_preds):
    return None


# 4.3 Training Callbacks
class TrainingMonitorCallback(TrainerCallback):
    def __init__(self):
        self.train_start_time = None
        self.metrics_history = {
            'step': [],
            'loss': [],
            'learning_rate': [],
            'epoch': [],
            'cpu_ram_usage': [],
            'gpu_vram_usage': [],
            'gpu_utilization': [],
            'batch_size': [],
            'moving_avg_loss': [],
            'lr_schedule': [],
            'batch_metrics': [],
            'gpu_metrics': [],
            'grad_norm': []
        }
        self.peak_metrics = {
            'cpu_ram': 0,
            'gpu_vram': 0,
            'gpu_util': 0
        }
        self.output_dir = Path(f"{BASE_OUTPUT_DIR}/training_progress")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_df = pd.DataFrame(columns=['step', 'loss', 'learning_rate', 'epoch'])
        
    def _record_resource_usage(self):
        """Record current resource usage with timestamp"""
        import psutil
        import torch
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # CPU RAM
        cpu_ram = psutil.Process().memory_info().rss / (1024 * 1024 * 1024)  # GB
        self.peak_metrics['cpu_ram'] = max(self.peak_metrics['cpu_ram'], cpu_ram)
        
        # GPU metrics with timestamp
        if torch.cuda.is_available():
            gpu_metrics = []
            for i in range(torch.cuda.device_count()):
                vram_used = torch.cuda.memory_allocated(i) / (1024 * 1024 * 1024)  # GB
                self.peak_metrics['gpu_vram'] = max(self.peak_metrics['gpu_vram'], vram_used)
                
                # GPU utilization (requires nvidia-smi)
                try:
                    import subprocess
                    result = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'])
                    gpu_util = float(result.decode('utf-8').strip())
                    self.peak_metrics['gpu_util'] = max(self.peak_metrics['gpu_util'], gpu_util)
                except:
                    gpu_util = 0
                
                gpu_metrics.append({
                    'device': i,
                    'vram_used': vram_used,
                    'utilization': gpu_util
                })
                
            # Verify directory exists before saving
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save as time series data
            self.metrics_history['gpu_metrics'].append({
                'timestamp': current_time,
                'metrics': gpu_metrics
            })
                
            self.metrics_history['cpu_ram_usage'].append(cpu_ram)
            self.metrics_history['gpu_vram_usage'].append(vram_used)
            self.metrics_history['gpu_utilization'].append(gpu_util)

    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start_time = datetime.now()
        logging.info("Training started at: %s", self.train_start_time)
        self._record_resource_usage()
        
    def _save_training_metrics(self):
        """Save training metrics to CSV and create visualization"""
        # Save metrics to CSV
        metrics_path = self.output_dir / 'training_metrics.csv'
        self.metrics_df.to_csv(metrics_path, index=False)
        
        # Verify data exists
        if len(self.metrics_df) == 0:
            logging.warning("No metrics data available for plotting")
            return
        
        # Create training progress visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Training Loss plot - Exclude NaN values
        loss_data = self.metrics_df[['step', 'loss']].dropna()
        if len(loss_data) > 0:
            ax1.plot(loss_data['step'], 
                    loss_data['loss'], 
                    label='Training Loss',
                    color='blue')
            ax1.set_xlabel('Training Steps')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Loss Over Time')
            ax1.legend()
            ax1.grid(True)
        
        # Learning rate plot - Exclude NaN values
        lr_data = self.metrics_df[['step', 'learning_rate']].dropna()
        if len(lr_data) > 0:
            ax2.plot(lr_data['step'], 
                    lr_data['learning_rate'],
                    color='green')
            ax2.set_xlabel('Training Steps')
            ax2.set_ylabel('Learning Rate')
            ax2.set_title('Learning Rate Schedule')
            ax2.grid(True)
        
        plt.tight_layout()
        
        # Save training progress visualization
        plt.savefig(self.output_dir / 'training_progress.png')
        plt.close()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Record loss and learning rate
            if 'loss' in logs:
                loss = logs['loss']
                if isinstance(loss, torch.Tensor):
                    loss = loss.item()
                self.metrics_history['loss'].append(loss)
            
            if 'learning_rate' in logs:
                lr = logs['learning_rate']
                self.metrics_history['learning_rate'].append(lr)
            
            # Add metrics to DataFrame
            new_row = {
                'step': state.global_step,
                'loss': logs.get('loss', None),
                'learning_rate': logs.get('learning_rate', None),
                'epoch': state.epoch
            }
            self.metrics_df = pd.concat([self.metrics_df, 
                                       pd.DataFrame([new_row])], 
                                       ignore_index=True)
            
            # Save metrics every 50 steps
            if state.global_step % 50 == 0:
                self._save_training_metrics()
            
            self._record_resource_usage()
        
    def on_train_end(self, args, state, control, **kwargs):
        training_duration = datetime.now() - self.train_start_time
        
        # Save detailed training history
        training_history = {
            'lr_schedule': self.metrics_history['lr_schedule'],
            'batch_metrics': self.metrics_history['batch_metrics'],
            'gpu_metrics': self.metrics_history['gpu_metrics'],
            'moving_avg_loss': self.metrics_history['moving_avg_loss']
        }
        
        # Save training history to JSON file
        history_file = self.output_dir / 'training_history.json'
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(training_history, f, indent=2, ensure_ascii=False)
        
        # Log basic metrics
        logging.info(f"Training completed. Total duration: {training_duration}")
        logging.info(f"Peak CPU RAM usage: {self.peak_metrics['cpu_ram']:.2f} GB")
        logging.info(f"Peak GPU VRAM usage: {self.peak_metrics['gpu_vram']:.2f} GB")
        logging.info(f"Peak GPU utilization: {self.peak_metrics['gpu_util']:.1f}%")
        
        # Create and save final summary
        summary = {
            'training_duration': str(training_duration),
            'final_loss': self.metrics_history['loss'][-1] if self.metrics_history['loss'] else None,
            'total_steps': len(self.metrics_history['step']),
            'final_epoch': self.metrics_history['epoch'][-1] if self.metrics_history['epoch'] else None,
            'learning_rate_summary': {
                'initial': self.metrics_history['learning_rate'][0] if self.metrics_history['learning_rate'] else None,
                'final': self.metrics_history['learning_rate'][-1] if self.metrics_history['learning_rate'] else None,
                'schedule_type': args.lr_scheduler_type
            },
            'loss_summary': {
                'final_moving_avg': self.metrics_history['moving_avg_loss'][-1] if self.metrics_history['moving_avg_loss'] else None,
                'best_loss': min(self.metrics_history['loss']) if self.metrics_history['loss'] else None
            },
            'resource_usage': {
                'peak_cpu_ram_gb': self.peak_metrics['cpu_ram'],
                'peak_gpu_vram_gb': self.peak_metrics['gpu_vram'],
                'peak_gpu_utilization': self.peak_metrics['gpu_util']
            },
            'hardware_info': {
                'cpu_info': self._get_cpu_info(),
                'gpu_info': self._get_gpu_info(),
                'total_ram': self._get_total_ram()
            }
        }
        
        with open(self.output_dir / 'training_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logging.info("Training Complete!")
        logging.info(f"Training duration: {summary['training_duration']}")
        
        # None check added
        if summary['loss_summary']['final_moving_avg'] is not None:
            logging.info(f"Final moving average loss: {summary['loss_summary']['final_moving_avg']:.4f}")
        if summary['loss_summary']['best_loss'] is not None:
            logging.info(f"Best loss achieved: {summary['loss_summary']['best_loss']:.4f}")
        
        logging.info(f"Peak CPU RAM usage: {summary['resource_usage']['peak_cpu_ram_gb']:.2f} GB")
        logging.info(f"Peak GPU VRAM usage: {summary['resource_usage']['peak_gpu_vram_gb']:.2f} GB")
        logging.info(f"Peak GPU utilization: {summary['resource_usage']['peak_gpu_utilization']:.1f}%")

        # Final save of metrics and visualization
        self._save_training_metrics()

    def _get_cpu_info(self):
        import cpuinfo
        try:
            info = cpuinfo.get_cpu_info()
            return {
                'model': info.get('brand_raw', 'Unknown'),
                'cores': psutil.cpu_count(logical=False),
                'threads': psutil.cpu_count(logical=True)
            }
        except:
            return "Failed to get CPU info"
            
    def _get_gpu_info(self):
        if not torch.cuda.is_available():
            return "No GPU available"
        try:
            import subprocess
            result = subprocess.check_output(['nvidia-smi', '--query-gpu=gpu_name,memory.total', '--format=csv,noheader,nounits'])
            gpus = result.decode('utf-8').strip().split('\n')
            return [{'model': g.split(',')[0], 'memory': float(g.split(',')[1])/1024} for g in gpus]
        except:
            return "Failed to get GPU info"
            
    def _get_total_ram(self):
        return psutil.virtual_memory().total / (1024 * 1024 * 1024)  # GB



# 4.4 Custom Trainer Implementation
class CustomTrainer(Trainer):
    def training_step(self, *args, **kwargs):
        loss = super().training_step(*args, **kwargs)
        if self.state.global_step % 50 == 0:
            clear_memory()
            gc.collect()
            torch.cuda.empty_cache()
        return loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is not None:
            # Limit evaluation dataset to 100 samples
            eval_dataset = eval_dataset.select(range(min(100, len(eval_dataset))))
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

# 4.5 Training Setup
training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT_DIR,
    num_train_epochs=30,
    learning_rate=8e-5,
    weight_decay=0.06,
    warmup_ratio=0.25,
    lr_scheduler_type="cosine_with_restarts",
    evaluation_strategy="steps",
    eval_steps=20,
    save_strategy="steps",
    save_steps=100,
    gradient_accumulation_steps=8,
    max_steps=-1,
    disable_tqdm=False,
    logging_dir=LOG_OUTPUT_DIR,
    logging_strategy="steps",
    logging_steps=50,
    no_cuda=False,
    dataloader_num_workers=2,
    report_to=[],
    run_name=None,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=2,
    gradient_checkpointing=True,
    max_grad_norm=0.5,
    dataloader_pin_memory=True,
    save_total_limit=20,
    fp16=True,
    optim="adamw_torch_fused",
    eval_accumulation_steps=8,
    load_best_model_at_end=False,
)

# 5. Execution and Model Management
# 5.1 Data Set Split and Validation
# Data set split
dataset_size = len(tokenized_dataset)
indices = np.random.permutation(dataset_size)
split_idx = int(dataset_size * 0.8)

# Create training and test datasets
train_dataset = tokenized_dataset.select(indices[:split_idx])
eval_dataset = tokenized_dataset.select(indices[split_idx:split_idx+50]) 

# Record dataset size
logging.info(f"Training dataset size: {len(train_dataset)}")
logging.info(f"Evaluation dataset size: {len(eval_dataset)}")

# 5.2 Training Execution Control
# Trainer initialization
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    callbacks=[TrainingMonitorCallback()],
)

# Check memory state
log_memory_usage()

# Training execution
logging.info("Starting training...")
try:
    checkpoint_dir = MODEL_OUTPUT_DIR
    resume_from_checkpoint = None
    
    # Checkpoint status and processing modification
    if os.path.exists(checkpoint_dir):
        logging.info("\nChecking checkpoint status...")
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint-")]
        if checkpoints:
            # Get latest checkpoint
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
            logging.info(f"Found latest checkpoint: {latest_checkpoint}")
            
            # Check checkpoint status
            state_path = os.path.join(checkpoint_path, "trainer_state.json")
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    state = json.load(f)
                current_epoch = state.get('epoch', 0)
                logging.info(f"\nCurrent training status:")
                logging.info(f"Current epoch: {current_epoch}")
                logging.info(f"Target epochs: {training_args.num_train_epochs}")
                
                # Exit safely if completed
                if current_epoch >= training_args.num_train_epochs - 0.1:
                    logging.info("\n" + "="*50)
                    logging.info("IMPORTANT NOTICE:")
                    logging.info(f"Training has already been completed at epoch {current_epoch}!")
                    logging.info(f"Target epochs was {training_args.num_train_epochs}")
                    logging.info(f"Trained model is available at: {checkpoint_dir}")
                    logging.info("="*50 + "\n")
                    exit(0)
            else:
                logging.warning("Invalid checkpoint state found. Proceeding with training...")
                logging.warning(f"Checkpoint directory: {checkpoint_dir}")
        else:
            logging.warning("Checkpoint directory exists but no checkpoints found. Proceeding with training...")

    # Start learning (or resume)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    logging.info("Training completed successfully!")
    
    # Save settings (as JSON)
    import json

    def convert_to_serializable(obj):
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(x) for x in obj]
        return obj

    # Convert all settings
    training_args_dict = convert_to_serializable(training_args.to_dict())
    lora_config_dict = convert_to_serializable(lora_config.to_dict())

    config_dict = {
        "model_name": model_name,
        "training_args": training_args_dict,
        "lora_config": lora_config_dict,
        "bnb_config": {
            "load_in_4bit": bnb_config.load_in_4bit,
            "bnb_4bit_use_double_quant": bnb_config.bnb_4bit_use_double_quant,
            "bnb_4bit_quant_type": bnb_config.bnb_4bit_quant_type,
            "bnb_4bit_compute_dtype": str(bnb_config.bnb_4bit_compute_dtype),
        }
    }
    
    with open(os.path.join(training_args.output_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    # Save model
    trainer.save_model()
    # Save settings
    model.config.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    logging.info("Model and configuration saved successfully!")

except Exception as e:
    logging.error(f"An error occurred: {str(e)}")
    # Checkpoint is also kept even on error
    raise 