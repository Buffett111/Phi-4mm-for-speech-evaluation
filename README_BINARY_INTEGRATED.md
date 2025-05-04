# Binary-Integrated Classification Model Pipeline

This project implements a model pipeline that integrates a binary classification model with a traditional classification model for English pronunciation evaluation. The pipeline has two main components:

1. A binary model (`ntnu-smil/Phi-4-mm_Binary_QA_NI_0415_1964`) that predicts whether an audio response passes (1) or fails (0) a minimum standard, along with a confidence score.
2. A classification model (`ntnu-smil/Phi-4-multimodal-instruct_QA_NoImage_0325_1964`) that assigns a score from 1-5 based on the quality of the pronunciation.

The innovation in this pipeline is the integration of the binary model's predictions into the classification model's input, guiding the classification model to make more consistent predictions.

## How It Works

1. First, we evaluate the training datasets using the binary model, producing predictions in the format "0,0.80" (fail with 80% confidence) or "1,0.85" (pass with 85% confidence).
2. Then, we fine-tune the classification model, including the binary predictions in its input prompt, with guidance that:
   - When the binary output is 0, the actual score should be between 1-3
   - When the binary output is 1, the actual score should be between 4-5

This integration helps the classification model make more consistent and accurate predictions by leveraging the binary model's signal.

## Scripts

The project includes the following scripts:

- `evaluate_binary_data.py`: Evaluates datasets using the binary model and saves predictions
- `fine-tune-lttc-binary.py`: Fine-tunes the classification model with binary predictions integrated
- `run_binary_integrated_pipeline.sh`: Runs the complete pipeline end-to-end

## Usage

Run the complete pipeline with:

```bash
./run_binary_integrated_pipeline.sh
```

### Individual Steps

1. **Evaluate with binary model**:
   ```bash
   python evaluate_binary_data.py \
       --binary_model_path ntnu-smil/Phi-4-mm_Binary_QA_NI_0415_1964 \
       --dataset_name ntnu-smil/LTTC-Train-1764-0520 \
       --dataset2 ntnu-smil/Unseen_1964 \
       --output_dir ./binary_eval_results
   ```

2. **Fine-tune classification model**:
   ```bash
   python fine-tune-lttc-binary.py \
       --model_name_or_path ntnu-smil/Phi-4-multimodal-instruct_QA_NoImage_0325_1964 \
       --dataset_name ntnu-smil/LTTC-Train-1764-0520 \
       --binary_preds_path ./binary_eval_results/binary_preds_LTTC-Train-1764-0520_1764.json \
       --form_id "1764" \
       --output_dir ./output_binary_integrated_1764
   ```

## Datasets

The pipeline uses two primary datasets:
- `ntnu-smil/LTTC-Train-1764-0520` (form_id: 1764)
- `ntnu-smil/Unseen_1964` (form_id: 1964)

## Implementation Details

### Binary Model Integration

The core of this implementation is the `BinaryIntegratedDataset` class in `fine-tune-lttc-binary.py`, which:

1. Loads binary predictions from a JSON file
2. Incorporates these predictions into the prompt for the classification model
3. Provides guidance on how to interpret the binary predictions

### Modified Instruction Text

The instruction prompt has been modified to include the binary model output and guidance:

```
"You are responsible for evaluating English pronunciation. The provided audio is a response to the question below.
I will also provide you with a binary model prediction in the format '0,0.80' or '1,0.85', where
the first number (0 or 1) indicates if the audio fails (0) or passes (1) a minimum standard,
and the second number indicates the confidence (probability) of that prediction.
Your task is to objectively assign scores from 1 to 5 based on the following criteria: sentence-level accuracy, fluency, prosody, completeness, and relevance to the question.
When the binary output is 0, the actual score should probably be between 1-3.
When the binary output is 1, the actual score should probably be between 4-5.
Your evaluation should remain neutral, fair, and utilize the binary model output as a guide."
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Accelerate
- Datasets

See `requirements.txt` for full dependencies.

## Notes

- This approach only trains the classification model; the binary model is used as-is without further training.
- The binary model's output serves as a guide rather than a strict constraint for the classification model. 