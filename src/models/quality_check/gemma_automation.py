from src.utils.config import get_api_keys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os
from queue import Queue
import logging
from huggingface_hub import login
from peft import PeftModel
from anthropic import Anthropic
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import gc

# Global Settings
MAX_HISTORY = 5  # Number of turns of dialogue Gemma remembers (state maintenance)
MAX_UTTERANCES = 5    # Total number of utterances in one dialogue
BASE_MODEL = "google/gemma-2-2b-jpn-it"
SAVE_DIR = "data/dialogue/raw_gemma"  # Save location for dialogue results

# Add new constants for data files
CSV_CONFIG_PATH = "data/config/automation_gemma.csv"
QUESTIONS_JSON_PATH = "src/models/quality_check/questions.json"

# Claude model settings
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"  # Update model name

# Claude's system prompt
SYSTEM_PROMPT = """
あなたはソクラテスと哲学的な対話を行う対話者です。
ソクラテスの質問に対して、以下のように振る舞ってください：
- 哲学的な概念を持ち出しすぎず、一般の人の視点で答えてください
- 質問の意図や意味が分からない場合は無理にあわせず、分からない旨を素直に伝えてください
- 返答は文字数制限を越える可能性があるため、絶対端的にしてください。
"""

# Get API keys using config utility
api_keys = get_api_keys()
HF_TOKEN = api_keys['huggingface_api_key']
CLAUDE_API_KEY = api_keys['claude_api_key_quality2']

if not HF_TOKEN or not CLAUDE_API_KEY:
    logger.warning("Required API keys not found in environment variables")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
SAVE_PATH = os.path.join(ROOT_DIR, SAVE_DIR)

# Create save directory if it doesn't exist
os.makedirs(SAVE_PATH, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TensorFlow warning suppression

class ChatAI:
    """
    ChatAI class
    - Loads the model and tokenizer
    - Manages message history
    - Generates responses
    """
    def __init__(
        self,
        model_path: str = None,
        base_model: str = "google/gemma-2-2b-jpn-it",
        max_history: int = 5,
        hf_token: str = None
    ):
        """
        Constructor
        Args:
            model_path (str, optional): Path to the fine-tuned model. If None, uses base model
            base_model (str): Base model path on Hugging Face
            max_history (int): Number of turns to store in the history
            hf_token (str): Hugging Face access token
        """
        self.max_history = max_history
        self.message_history = Queue(maxsize=max_history)
        self.hf_token = hf_token
        
        try:
            logger.info("Loading model and tokenizer...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model,
                token=hf_token,
                trust_remote_code=True
            )
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            # Load configuration modified
            load_config = {
                "trust_remote_code": True,
                "token": hf_token,
                "device_map": "auto",
                "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
            }
            
            if device == "cuda":
                # GPU memory setting addition
                load_config.update({
                    "max_memory": {0: "10GB"},  # Allocate 10GB to GPU 0
                    "offload_folder": None  # Disable offloading
                })
            else:
                load_config["offload_folder"] = "offload_folder"
                os.makedirs("offload_folder", exist_ok=True)

            # Load model modified
            if model_path:
                base_model_obj = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    **load_config
                ).to(device)
                
                self.model = PeftModel.from_pretrained(
                    base_model_obj,
                    model_path
                ).to(device)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    **load_config
                ).to(device)
            
            logger.info(f"Model loaded successfully on {device}")
            
            self.generation_config = {
                "max_new_tokens": 256,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
            }
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    def __del__(self):
        """Destructor: Release model resources"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        torch.cuda.empty_cache()  # Clear GPU memory

    def _initialize_model(self, base_model, model_path, hf_token):
        """Initialize the model with appropriate device and memory settings"""
        try:
            # Check available memory and GPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Configure model loading parameters based on available resources
            load_config = {
                "torch_dtype": torch.bfloat16,
                "trust_remote_code": True,
                "token": hf_token
            }
            
            # If running on CPU or limited memory, adjust loading strategy
            if device == "cpu":
                load_config["device_map"] = "auto"
                load_config["offload_folder"] = "offload_folder"  # Add offload directory
                os.makedirs("offload_folder", exist_ok=True)
            else:
                load_config["device_map"] = "balanced"

            # Load base model with configured parameters
            base_model_obj = AutoModelForCausalLM.from_pretrained(
                base_model,
                **load_config
            )

            # Load fine-tuned model
            self.model = PeftModel.from_pretrained(
                base_model_obj,
                model_path,
                **load_config
            )

            logger.info(f"Model loaded successfully on {device}")
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    def _update_history(self, message: dict) -> None:
        """
        Enqueues new user or model messages and manages removal of old messages
        """
        if self.message_history.full():
            removed = self.message_history.get()
            logger.debug(f"Removed message from history: {removed}")
        self.message_history.put(message)
        logger.debug(f"Added message to history: {message}")
        logger.debug(f"Current queue size: {self.message_history.qsize()}")

    def _format_messages(self):
        """
        Ensures that messages alternate correctly between user and model
        Returns a list of messages if valid, or an empty list if invalid
        """
        messages = list(self.message_history.queue)
        
        for i in range(len(messages)):
            expected_role = "user" if i % 2 == 0 else "model"
            if messages[i]["role"] != expected_role:
                logger.warning(f"Invalid message sequence detected at position {i}")
                return [messages[-1]] if messages[-1]["role"] == "user" else []
        
        return messages

    def generate_response(self, user_input: str, add_to_history: bool = True) -> str:

        try:
            if add_to_history:
                if self.message_history.qsize() == 0:  # First input case
                    initial_setting = (
                        "You are Socrates, an experienced philosopher. You started the conversation with the following statement:\n"
                        "\"{{QUESTION}}\"\n"
                        "Your dialogue partner responded as follows. Please respond in the same style, asking questions and ending with \"Nai?\" or \"Ie?\" if you want to stop.\n"
                        f"\"{user_input}\""
                    )
                    contextualized_input = initial_setting
                else:
                    # 2nd and subsequent times, include conversation history
                    conversation_history = "You are Socrates, an experienced philosopher. You started the conversation with the following statement:\n"
                    conversation_history += "\"{{QUESTION}}\"\n"
                    conversation_history += "それに対して以下のように今のところ対話が進んでます。文末に「かね？」や「ないか」等をいれつつ、引き続きソクラテスのような口調を使いながら、問いで返してください。\n"

                    # Add previous conversation history
                    messages = list(self.message_history.queue)
                    for msg in messages:
                        if msg["role"] == "user":
                            conversation_history += f"\nUser: \"{msg['content']}\""
                        else:
                            conversation_history += f"\nModel: {msg['content']}"

                    # Add new input
                    conversation_history += f"\n\nUser: \"{user_input}\""
                    contextualized_input = conversation_history

                self._update_history({"role": "user", "content": contextualized_input})
            
            messages = self._format_messages()
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            logger.debug(f"Generated prompt: {prompt}")
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False
            ).to(self.model.device)
            
            outputs = self.model.generate(
                **inputs,
                **self.generation_config
            )
            
            decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

            response_parts = decoded_output.split("<start_of_turn>model")
            if len(response_parts) > 1:
                last_response = response_parts[-1].split("<end_of_turn>")[0].strip()
                if "model" in last_response:
                    last_response = last_response.split("model", 1)[1].strip()
            else:
                last_response = "Failed to respond"
            
            if add_to_history:
                self._update_history({"role": "model", "content": last_response})
            
            return last_response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "There was an error"

    def _get_model_config(self):
        """Get model configuration based on available hardware"""
        config = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "token": self.hf_token
        }
        
        # GPU check
        if torch.cuda.is_available():
            config["device_map"] = "balanced"
            logger.info("GPU detected, using balanced device map")
        else:
            # CPU environment setting
            config["device_map"] = "auto"
            config["offload_folder"] = "offload_folder"
            os.makedirs("offload_folder", exist_ok=True)
            logger.info("CPU environment detected, using memory offloading")
        
        return config

def create_dialogue_session(chatai: ChatAI, dialogue_id: int):
    """
    Execute automated dialogue between Claude and Gemma, saving the results
    Conduct dialogue up to specified number of utterances (MAX_UTTERANCES) in one session
    """
    # Initialize Claude client
    claude = Anthropic(api_key=CLAUDE_API_KEY)
    
    dialogue_history = []
    utterance_count = 0
    
    try:
        # Initial Gemma message
        initial_gemma = question_content  # This line is defined in the existing code
        
        # Add initial Gemma message to history
        dialogue_history.append({
            "role": "gemma",
            "content": initial_gemma
        })
        
        # Get first Claude response
        try:
            response = claude.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=150,
                temperature=0.7,
                system=SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": initial_gemma
                }]
            )
            
            if not response.content:
                raise ValueError("Empty response from Claude")
                
            claude_message = response.content[0].text
            
            # Add Claude's response to history
            dialogue_history.append({
                "role": "claude",  
                "content": claude_message
            })
            utterance_count += 2
            
            # Continue dialogue while under MAX_UTTERANCES
            while utterance_count < MAX_UTTERANCES:
                # Get Gemma's response
                gemma_response = chatai.generate_response(claude_message)
                
                # Add Gemma's response to history
                dialogue_history.append({
                    "role": "gemma",  
                    "content": gemma_response
                })
                utterance_count += 1
                
                if utterance_count >= MAX_UTTERANCES:
                    break
                
                # Get Claude's next response
                messages = []
                for msg in dialogue_history:
                    if msg["role"] == "gemma":
                        messages.append({
                            "role": "user",
                            "content": msg["content"]
                        })
                    else:
                        messages.append({
                            "role": "assistant",
                            "content": msg["content"]
                        })
                
                response = claude.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=150,
                    temperature=0.7,
                    system=SYSTEM_PROMPT,
                    messages=messages
                )
                
                claude_message = response.content[0].text
                
                # Add Claude's response to history
                dialogue_history.append({
                    "role": "claude", 
                    "content": claude_message
                })
                utterance_count += 1
            
            # Save dialogue history after dialogue ends (passing model_path)
            save_dialogue(dialogue_history, dialogue_id, chatai.model.model_path)
            
        except Exception as claude_error:
            logger.error(f"Error in Claude response: {str(claude_error)}")
            raise
            
    except Exception as e:
        logger.error(f"Error in dialogue {dialogue_id}: {str(e)}")
        logger.error("Failed to complete dialogue")

def save_dialogue(dialogue_history: list, dialogue_id: int, model_path: str = None) -> None:
    """
    Save dialogue history in formatted JSON file
    
    Args:
        dialogue_history (list): List of dialogue history
        dialogue_id (int): Dialogue ID
        model_path (str, optional): Path to the actual used model
    """
    # Create save directory
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Set model version and checkpoint from model path
    if model_path:
        try:
            # Convert path to PosixPath and parse
            path = Path(model_path)
            parts = path.parts
            models_idx = parts.index('models')
            if models_idx + 1 < len(parts):
                model_version = parts[models_idx + 1]
            else:
                model_version = "base"
                
            checkpoint = next((part for part in parts if part.startswith('checkpoint-')), "default")
            
        except (ValueError, IndexError) as e:
            logger.warning(f"Could not extract model version or checkpoint from path: {model_path}. Error: {str(e)}")
            model_version = "base"
            checkpoint = "default"
    else:
        # Base model case
        model_version = "base"
        checkpoint = "default"
    
    logger.info(f"Using model_version: {model_version}, checkpoint: {checkpoint}")
    
    # Format new filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"dialogue_{model_version}_{checkpoint}_{dialogue_id}_{timestamp}.json"
    filepath = os.path.join(SAVE_DIR, filename)
    
    # Build formatted data
    formatted_data = {
        "metadata": {
            "question_id": dialogue_id,
            "timestamp": timestamp,
            "model_version": model_version,
            "checkpoint": checkpoint,
            "topic": dialogue_history[0]["content"]
        },
        "pairs": []
    }
    
    # Build dialogue pairs
    history = dialogue_history[1:]  # Exclude initial Gemma statement (topic)
    for i in range(0, len(history), 2):
        if i + 1 >= len(history):
            break
            
        pair = {
            "pair_id": (i // 2) + 1,
            "claude": {
                "content": history[i]["content"]
            },
            "gemma": {
                "content": history[i + 1]["content"]
            },
            "evaluation": {
                "tone": {"quantitative": "", "qualitative": ""},
                "approach": {"quantitative": "", "qualitative": ""},
                "format": {"quantitative": "", "qualitative": ""},
                "logic": {"quantitative": "", "qualitative": ""}
            }
        }
        formatted_data["pairs"].append(pair)
    
    # Save formatted data
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved formatted dialogue {dialogue_id} to {filepath}")

IS_KAGGLE_SUBMISSION = os.path.exists('/kaggle/working')

def load_questions():
    """Load questions from JSON file"""
    with open(QUESTIONS_JSON_PATH, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)
    # Convert to dictionary for easier lookup
    return {str(q['id']): q['content'] for q in questions_data['prompts']}

def load_config():
    """Load configuration from CSV file"""
    df = pd.read_csv(CSV_CONFIG_PATH)
    # Convert QUESTION_ID to integer type
    df['QUESTION_ID'] = df['QUESTION_ID'].astype(int)
    
    # Check for duplicates (combinations of QUESTION_ID, model_version, checkpoint)
    duplicates = df.duplicated(['QUESTION_ID', 'model_version', 'checkpoint'], keep=False)
    if duplicates.any():
        logger.warning("Found duplicate entries in config file:")
        logger.warning(df[duplicates])
    
    # Remove duplicates (based on QUESTION_ID, model_version, checkpoint combination)
    df = df.drop_duplicates(
        subset=['QUESTION_ID', 'model_version', 'checkpoint'],
        keep='first'
    )
    
    return df

def update_csv(csv_path: str, question_id: str, model_version: str, checkpoint: str):
    """Update CSV file with dialogue filename"""
    # Read CSV file with string type for dialogue column
    df = pd.read_csv(csv_path, dtype={'dialogue': str})
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate correct filename
    if pd.isna(model_version) and pd.isna(checkpoint):
        filename = f"dialogue_base_default_{question_id}_{timestamp}.json"
    else:
        filename = f"dialogue_{model_version}_{checkpoint}_{question_id}_{timestamp}.json"
    
    # Update dialogue column for matching rows
    mask = (df['QUESTION_ID'] == int(question_id)) & \
           (df['model_version'] == model_version) & \
           (df['checkpoint'] == checkpoint)
    df.loc[mask, 'dialogue'] = filename
    df.to_csv(csv_path, index=False)

def main():
    """Main execution function"""
    questions = load_questions()
    config_df = load_config()
    
    for _, row in config_df.iterrows():
        question_id = str(int(row['QUESTION_ID']))
        chatai = None
        
        try:
            # Execute garbage collection explicitly
            gc.collect()
            torch.cuda.empty_cache()
            
            # Skip if dialogue already exists
            if pd.notna(row['dialogue']):
                logger.info(f"Dialogue already exists for question {question_id}, skipping...")
                continue
                
            if question_id not in questions:
                logger.warning(f"Question ID {question_id} not found in questions.json")
                continue
            
            # Check if model_version and checkpoint are specified
            use_base_model = pd.isna(row.get('model_version')) or pd.isna(row.get('checkpoint'))
            
            if use_base_model:
                logger.info(f"Using base model for question {question_id}")
                current_model_path = None
            else:
                model_version = row['model_version']
                checkpoint = row['checkpoint']
                current_model_path = os.path.join(ROOT_DIR, "models", model_version, "model", checkpoint)
                logger.info(f"Using fine-tuned model for question {question_id}: {current_model_path}")
            
            # Load the appropriate model
            chatai = ChatAI(
                model_path=current_model_path,  # None for base model
                base_model=BASE_MODEL,
                max_history=MAX_HISTORY,
                hf_token=HF_TOKEN
            )
            
            # Get question content
            question_content = questions[question_id]
            
            # Create dialogue session with the specific question
            dialogue_history = []
            
            # Initialize Claude client
            claude = Anthropic(api_key=CLAUDE_API_KEY)
            
            # Initial Gemma message
            initial_gemma = question_content
            dialogue_history.append({
                "role": "gemma",
                "content": initial_gemma
            })
            
            # Get first Claude response
            response = claude.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=150,
                temperature=0.7,
                system=SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": initial_gemma
                }]
            )
            
            claude_message = response.content[0].text
            dialogue_history.append({
                "role": "claude",
                "content": claude_message
            })
            
            # Continue dialogue for specified number of utterances
            utterance_count = 2  # Initial Gemma question and Claude response
            
            while utterance_count < MAX_UTTERANCES:
                # Get Gemma's response
                gemma_response = chatai.generate_response(claude_message)
                dialogue_history.append({
                    "role": "gemma",
                    "content": gemma_response
                })
                utterance_count += 1
                
                if utterance_count >= MAX_UTTERANCES:
                    break
                
                # Get Claude's response
                messages = []
                for msg in dialogue_history:
                    if msg["role"] == "gemma":
                        messages.append({
                            "role": "user",
                            "content": msg["content"]
                        })
                    else:
                        messages.append({
                            "role": "assistant",
                            "content": msg["content"]
                        })
                
                response = claude.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=150,
                    temperature=0.7,
                    system=SYSTEM_PROMPT,
                    messages=messages
                )
                
                claude_message = response.content[0].text
                dialogue_history.append({
                    "role": "claude",
                    "content": claude_message
                })
                utterance_count += 1
            
            # Save dialogue history after dialogue ends (passing model_path)
            save_dialogue(dialogue_history, int(question_id), current_model_path)
            
            # Update CSV with the correct filename format
            update_csv(
                CSV_CONFIG_PATH, 
                question_id, 
                row['model_version'], 
                row['checkpoint']
            )
            
        except Exception as e:
            logger.error(f"Error processing question {question_id}: {str(e)}")
            if 'chatai' in locals() and chatai is not None:  # chatai exists only if executed
                del chatai
                torch.cuda.empty_cache()
            continue
        
        finally:
            if 'chatai' in locals() and chatai is not None:  # chatai exists only if executed
                del chatai
            torch.cuda.empty_cache()
            gc.collect()
            
            # Log GPU memory usage (optional)
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(0)
                memory_reserved = torch.cuda.memory_reserved(0)
                logger.info(f"GPU Memory - Allocated: {memory_allocated/1024**2:.2f}MB, Reserved: {memory_reserved/1024**2:.2f}MB")
    
    logger.info("Completed all dialogues")

if __name__ == "__main__":
    main()