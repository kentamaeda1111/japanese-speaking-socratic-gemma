# Japanese-Speaking Socratic Gemma

## Supplementary Repository for Kaggle Competition
This repository provides the complete pipeline and resources necessary to replicate the model training process behind [Japanese-Speaking Socratic Gemma](https://www.kaggle.com/code/kentamaeda/japanese-speaking-socratic-gemma-2), Kaggle notebook submission to the [Google - Unlock Global Communication with Gemma](https://www.kaggle.com/competitions/gemma-language-tuning) competition. While the Kaggle notebook only includes the inference code, this repository contains the full end-to-end training pipeline to enable complete model reproduction, including implementations of dialogue generation, quality assessment, data preparation, model training, evaluation, and analysis.

## Repository Structure
```
project_root/
├── src/                      # Source code for pipeline components
├── data/                     # Configuration, prompts, and dialogue data
└── models/                   # Trained model checkpoints and logs
```

## Development Pipeline
This project implements a seven-stage pipeline using Gemma-2b. Each stage corresponds to implementation files:

### Stage 1: Dialogue Generation
- Implementation: `src/data/generation/automation.py`
- Inputs:
  - Configuration: `data/config/automation.csv`
  - Prompts: `data/prompts/*.json`
- Output: Generated dialogue files in `data/dialogue/raw/`
Note: Detailed documentation about the prompt engineering system and dialogue generation strategy is available in `data/prompts/README.md`

### Stage 2: Quality Assessment
- Implementation: `src/data/quality_check/dialogue_quality_check.py`
- Input: Dialogue files from `data/dialogue/raw/`
- Output: Moves low-rated dialogues to `data/dialogue/low_rated/`

### Stage 3: Data Preparation
- Implementation: `src/data/dataset_preparation/dialogue_extractor.py`
- Input: Dialogue files from `data/dialogue/raw/`
- Output: `data/dialogue/processed/kaggle_model.json`

### Stage 4: Model Training
- Implementation: `src/models/training/train.py`
- Input: `data/dialogue/processed/kaggle_model.json`
- Output: Model artifacts in `models/kaggle_model/`

### Stage 5: Automated Model Evaluation
- Implementation: `src/models/quality_check/gemma_automation.py`
- Input: 
  - Configuration: `data/config/automation_gemma.csv`
  - Questions: `src/models/quality_check/questions.json`
- Output: Generated dialogue files in `data/dialogue/raw_gemma/`

### Stage 6: Quality Assessment of Model Outputs
- Implementation: `src/models/quality_check/gemma_quality_check.py`
- Input: Dialogue files from `data/dialogue/raw_gemma/`
- Output: Quality metrics update in `data/config/automation_gemma.csv`

### Stage 7: Results Analysis
- Implementation: `src/models/quality_check/analyze_quality_results.py`
- Input: `data/config/automation_gemma.csv`
- Output: Analysis results and visualizations in `data/analysis/`

## Implementation Transparency
This repository contains key files that are often excluded from public repositories:

1. `data/config/automation.csv`: Configuration parameters for dialogue generation
   - Temperature and maximum turns configuration
   - Quality metrics thresholds

2. `data/dialogue/processed/kaggle_model.json`: Training dataset
   - Training dataset for result verification and reproduction

3. `models/kaggle_model/`: Model artifacts including:
   - Checkpoints
   - Training logs

4. `data/prompts/*.json`: Dialogue generation prompts
   - `assistant_system_prompt/`: Socrates role prompts
   - `user_system_prompt/`: Student role prompts
   - `questions.json`: Initial philosophical questions

5. `data/dialogue/raw_gemma/`: Model evaluation dialogues
   - Combined JSON files for each checkpoint (e.g., `dialogue_attention-tuned_checkpoint-100_combined.json`)
   - Contains 20 philosophical themes × 2 exchanges per theme
   - Includes Claude's quality assessment scores and detailed evaluations for each model response

6. `src/models/quality_check/questions.json`: Quality assessment themes
   - Philosophical questions used for evaluating model outputs

These resources provide the necessary components for reproducing the model evaluation process, from initial dialogue generation through quality assessment to final analysis.

## Hardware Specifications for Training Environment
- GPU: 2x NVIDIA Tesla T4 (15GB VRAM each) or equivalent
- CPU: Intel Xeon CPU @ 2.00GHz (4 cores)
- RAM: 32GB (30GB available is recommended)
- Storage: 20GB+ free space

Note: The system has been tested successfully on Kaggle's T4 x2 environment. Performance with lower specifications cannot be guaranteed.

## Setup Instructions

### Environment Configuration
Required API keys:
- `CLAUDE_API_KEY_1`, `CLAUDE_API_KEY_2`, `CLAUDE_API_KEY_QUALITY`
- `HUGGINGFACE_API_KEY`

Configure API keys using one of the following methods:
- For local development: Create `.env` file
- For Kaggle: Set up in Secrets/environment variables
- For Colab: Configure in Secure form/environment variables

### Quick Start Guide
```bash
# Clone repository and set up environment
git clone https://github.com/kentamaeda1111/japanese-speaking-socratic-gemma
cd japanese-speaking-socratic-gemma
pip install -r requirements.txt

# Set up API key configuration
cp .env.template .env 

# Execute pipeline stages in order
python -m src.data.generation.automation
python -m src.data.quality_check.dialogue_quality_check
python -m src.data.dataset_preparation.dialogue_extractor
python -m src.models.training.train
python -m src.models.quality_check.gemma_automation
python -m src.models.quality_check.gemma_quality_check
python -m src.models.quality_check.analyze_quality_results


