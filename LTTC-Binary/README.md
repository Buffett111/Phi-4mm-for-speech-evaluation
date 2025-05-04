# LTTC Binary Classification Pronunciation Scoring Model

This folder contains specialized code for fine-tuning and evaluating a binary classification model for English pronunciation scoring, targeting the Language Training and Testing Center (LTTC) dataset.

## Overview

This implementation creates a binary classification model that predicts whether a pronunciation passes (score ≥ 4) or fails (score < 4), along with a probability that indicates the model's confidence in the classification. This is particularly useful for borderline cases around the passing threshold.

## Features

- **Binary Classification with Probability**: Converts the original LTTC scores (1-5 scale) to binary labels (0/1) with confidence probability
- **Specialized for Borderline Cases**: Optimized to better predict scores near the passing threshold (3.5-4.0)
- **Pipeline for Multiple Datasets**: Support for fine-tuning on both LTTC-Train-1764-0520 and Unseen_1964 datasets

## Score Conversion Logic

Original scores are converted to binary classification with probability as follows:

| Original Score | Binary Label | Probability | Interpretation |
|----------------|--------------|-------------|----------------|
| 1.0            | 0            | 0.99        | Fail with very high confidence |
| 2.0            | 0            | 0.80        | Fail with high confidence |
| 3.0            | 0            | 0.70        | Fail with moderate confidence |
| 3.5            | 0            | 0.50-0.60   | Borderline case (linear interpolation) |
| 4.0            | 1            | 0.65-0.75   | Pass with moderate confidence |
| 5.0            | 1            | 0.95        | Pass with very high confidence |

## Implementation Files

- **common_binary.py**: Contains core functionality for dataset handling, model loading, and evaluation logic specific to binary classification
- **fine-tune-binary.py**: Script for fine-tuning the model with binary classification targets
- **eval_binary.py**: Specialized evaluation script for binary classification models
- **finetune_binary_1764.sh**: Shell script for fine-tuning on the LTTC-Train-1764-0520 dataset
- **finetune_binary_1964.sh**: Shell script for fine-tuning on the Unseen_1964 dataset
- **evaluate_binary.sh**: Shell script for evaluating the binary classification models

## Training Pipeline

1. Fine-tune on dataset1 (LTTC-Train-1764-0520):
   ```bash
   ./LTTC-Binary/finetune_binary_1764.sh
   ```

2. Fine-tune on dataset2 (Unseen_1964), starting from the model trained in step 1:
   ```bash
   ./LTTC-Binary/finetune_binary_1964.sh
   ```

## Evaluation

Evaluate the model on a specific dataset:
```bash
./LTTC-Binary/evaluate_binary.sh <form_id> [dataset_name]
```

Where:
- `form_id` is either "1764" or "1964"
- `dataset_name` is optional (defaults to the dev dataset for the specified form_id)

Example:
```bash
# Evaluate the 1764 model on its dev dataset
./LTTC-Binary/evaluate_binary.sh 1764

# Evaluate the 1964 model on its dataset
./LTTC-Binary/evaluate_binary.sh 1964

# Evaluate the 1764 model on the 1964 dataset
./LTTC-Binary/evaluate_binary.sh 1764 ntnu-smil/Unseen_1964
```

## Model Outputs

The model outputs scores in the format: `binary_label,probability`

For example:
- `0,0.95` means "fails with 95% confidence"
- `1,0.70` means "passes with 70% confidence"

This format provides both the binary classification decision and the model's confidence, which is especially valuable for borderline cases. 