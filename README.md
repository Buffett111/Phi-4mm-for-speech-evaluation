# LTTC English Pronunciation Scoring Model

This repository contains code for fine-tuning and evaluating multimodal language models for English pronunciation scoring, specifically targeting the Language Training and Testing Center (LTTC) dataset.

## Overview

This project implements a system for automated English pronunciation evaluation using multimodal AI models. It processes audio recordings of English speech, evaluates pronunciation quality according to multiple criteria, and assigns scores on a scale of 0-5.

The system uses Phi-4-multimodal models with audio and text processing capabilities to evaluate:
- Sentence-level accuracy
- Fluency
- Prosody
- Completeness
- Overall pronunciation quality
- Relevance to the given question

## Project Structure

- **fine-tune-lttc.py**: Main script for fine-tuning the model on LTTC dataset
- **eval.py**: Evaluation script for testing model performance
- **common.py**: Shared utilities, dataset classes, and evaluation functions
- **scoring.py**: Script for scoring individual audio files
- **round_down_accuracy.py**: Utility for rounding scores in a specific way
- **finetune.sh/finetuneBase.sh**: Shell scripts for running fine-tuning jobs
- **score.sh**: Shell script for running evaluation jobs
- **uploadmodel.py**: Utility for uploading trained models to Hugging Face Hub

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch
- Transformers library
- Accelerate library
- Datasets library
- Librosa (for audio processing)

### Installation

Clone the repository and install dependencies:

```bash
git clone <repository-url>
cd models
pip install -r requirements.txt  # Note: Create this file with dependencies
```

### Fine-tuning the Model

To fine-tune the model on the LTTC dataset:

```bash
bash finetune.sh
```

Or customize the fine-tuning process:

```bash
python fine-tune-lttc.py \
  --model_name_or_path "microsoft/Phi-4-multimodal-instruct" \
  --dataset_name "ntnu-smil/LTTC-Train-1764-0520" \
  --eval_dataset "ntnu-smil/LTTC-Dev-1764-0520" \
  --output_dir "./output/" \
  --learning_rate 4.0e-5 \
  --num_train_epochs 3 \
  --push_to_hub
```

### Evaluating the Model

To evaluate a trained model:

```bash
bash score.sh
```

Or run the evaluation script directly:

```bash
python eval.py \
  --model_name_or_path "your-fine-tuned-model" \
  --dataset_name "ntnu-smil/LTTC-Dev-1764-0520" \
  --output_dir "./eval_results/"
```

### Scoring Individual Audio Files

```bash
python scoring.py --model_name_or_path "your-fine-tuned-model"
```

## Development Notes

From notes.md:
- Adding QA slightly decreases accuracy
- Training with 2 epochs results in still high loss (0.273), need to use 3 epochs

### To-Do List:
1. Change model from 1-shot to multiple shots: provide image and prompt + question first, then send audio, asking model to give a score
2. Integrate image into the prompt
3. Find approach to reduce misjudging between scores 3.5 ~ 4

## Model Architecture

The project uses the Phi-4-multimodal model architecture, with LoRA adapters for efficient fine-tuning of speech processing capabilities. The model processes both audio input and textual prompts to generate pronunciation scores.

## Dataset

The project primarily uses:
- **ntnu-smil/LTTC-Train-1764-0520**: For training
- **ntnu-smil/LTTC-Dev-1764-0520**: For evaluation


## Acknowledgements

- Microsoft for the Phi-4-multimodal model
- Language Training and Testing Center (LTTC) for the dataset 
