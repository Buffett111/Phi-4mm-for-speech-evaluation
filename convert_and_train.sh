#!/bin/bash

# Script to convert binary results and then run the binary-integrated training

# Input files (binary results)
BINARY_RESULTS_1764="binary_eval_results/binary_results_LTTC-Train-1764-0520_train.json"
BINARY_RESULTS_1964="binary_eval_results/binary_results_Unseen_1964_train.json"

# Output files (binary predictions)
BINARY_PREDS_1764="binary_eval_results/binary_preds_LTTC-Train-1764-0520_1764.json"
BINARY_PREDS_1964="binary_eval_results/binary_preds_Unseen_1964_1964.json"

# Audio column names
AUDIO_COLUMN_1764="wav_path"
AUDIO_COLUMN_1964="wav_file"

# Step 1: Convert binary results to predictions format
echo "Converting binary results to predictions format..."

# Convert for 1764 dataset
if [ -f "$BINARY_RESULTS_1764" ]; then
    python convert_binary_results.py \
        --results_file "$BINARY_RESULTS_1764" \
        --output_file "$BINARY_PREDS_1764" \
        --audio_column "$AUDIO_COLUMN_1764"
else
    echo "Warning: $BINARY_RESULTS_1764 not found. Skipping conversion."
fi

# Convert for 1964 dataset
if [ -f "$BINARY_RESULTS_1964" ]; then
    python convert_binary_results.py \
        --results_file "$BINARY_RESULTS_1964" \
        --output_file "$BINARY_PREDS_1964" \
        --audio_column "$AUDIO_COLUMN_1964"
else
    echo "Warning: $BINARY_RESULTS_1964 not found. Skipping conversion."
fi

# Step 2: Run the binary-integrated training
echo "Running binary-integrated training..."

# Train with 1764 dataset
if [ -f "$BINARY_PREDS_1764" ]; then
    echo "Training with LTTC-Train-1764-0520 (Form ID: 1764)..."
    python fine-tune-lttc-binary.py \
        --model_name_or_path "ntnu-smil/Phi-4-multimodal-instruct_QA_NoImage_0325_1964" \
        --dataset_name "ntnu-smil/LTTC-Train-1764-0520" \
        --eval_dataset "ntnu-smil/LTTC-Dev-1764-0520" \
        --binary_preds_path "$BINARY_PREDS_1764" \
        --form_id "1764" \
        --audio_column "$AUDIO_COLUMN_1764" \
        --eval_audio_column "$AUDIO_COLUMN_1764" \
        --output_dir "./output_binary_integrated_1764" \
        --learning_rate 4e-5 \
        --num_train_epochs 1 \
        --global_batch_size 16 \
        --batch_size_per_gpu 8 \
        --eval_batch_size_per_gpu 8
else
    echo "Warning: $BINARY_PREDS_1764 not found. Skipping 1764 training."
fi

# Train with 1964 dataset
if [ -f "$BINARY_PREDS_1964" ]; then
    echo "Training with Unseen_1964 (Form ID: 1964)..."
    python fine-tune-lttc-binary.py \
        --model_name_or_path "ntnu-smil/Phi-4-multimodal-instruct_QA_NoImage_0325_1964" \
        --dataset_name "ntnu-smil/Unseen_1964" \
        --eval_dataset "ntnu-smil/LTTC-Dev-1764-0520" \
        --binary_preds_path "$BINARY_PREDS_1964" \
        --form_id "1964" \
        --audio_column "$AUDIO_COLUMN_1964" \
        --eval_audio_column "$AUDIO_COLUMN_1764" \
        --output_dir "./output_binary_integrated_1964" \
        --learning_rate 4e-5 \
        --num_train_epochs 1 \
        --global_batch_size 16 \
        --batch_size_per_gpu 8 \
        --eval_batch_size_per_gpu 8
else
    echo "Warning: $BINARY_PREDS_1964 not found. Skipping 1964 training."
fi

echo "Processing complete!" 