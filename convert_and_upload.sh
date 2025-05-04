#!/bin/bash

# Script to convert binary results and upload models to Hugging Face Hub
# Supports sequential fine-tuning where 1964 model builds on 1764 model

# Check if at least one argument is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <binary_results_file> [output_file] [audio_column] [hub_model_id]"
    echo "Example: $0 binary_eval_results/binary_results_LTTC-Train-1764-0520_train.json binary_eval_results/binary_preds_LTTC-Train-1764-0520_1764.json wav_path ntnu-smil/phi-4-mm-lttc-binary-integrated-1764"
    exit 1
fi

# Helper function to free memory
free_memory() {
    echo "Cleaning up memory..."
    # Kill any lingering Python processes that might be holding GPU memory
    pkill -f python || true
    sleep 5
    # Force Python garbage collection
    python -c "import gc; gc.collect()" || true
    # Clear CUDA cache
    python -c "import torch; torch.cuda.empty_cache()" || true
    sleep 5
    echo "Memory cleaned up"
}

# Input parameters
BINARY_RESULTS_FILE=$1
OUTPUT_FILE=${2:-"${BINARY_RESULTS_FILE/results/preds}"}
AUDIO_COLUMN=${3:-"wav_path"}
HUB_MODEL_ID=${4:-"ntnu-smil/phi-4-mm-lttc-binary-integrated"}

# Step 1: Convert binary results to predictions format
echo "Converting binary results to predictions format..."
echo "Input file: $BINARY_RESULTS_FILE"
echo "Output file: $OUTPUT_FILE"
echo "Audio column: $AUDIO_COLUMN"

free_memory

python convert_binary_results.py \
    --results_file "$BINARY_RESULTS_FILE" \
    --output_file "$OUTPUT_FILE" \
    --audio_column "$AUDIO_COLUMN"

# Check if conversion was successful
if [ $? -ne 0 ]; then
    echo "Error: Conversion failed."
    free_memory
    exit 1
fi

# Step 2: Determine form_id from filename
FORM_ID="1764"
if [[ "$BINARY_RESULTS_FILE" == *"1964"* ]]; then
    FORM_ID="1964"
fi

# Step 3: Run the binary-integrated training with push to hub
if [ -n "$HUB_MODEL_ID" ]; then
    echo "Training model with push to Hugging Face Hub..."
    echo "Form ID: $FORM_ID"
    echo "Hub Model ID: $HUB_MODEL_ID"
    
    # Extract dataset name from filename
    if [[ "$BINARY_RESULTS_FILE" == *"LTTC-Train-1764-0520"* ]]; then
        DATASET_NAME="ntnu-smil/LTTC-Train-1764-0520"
    elif [[ "$BINARY_RESULTS_FILE" == *"Unseen_1964"* ]]; then
        DATASET_NAME="ntnu-smil/Unseen_1964"
    else
        echo "Warning: Unable to determine dataset name from filename. Using default."
        DATASET_NAME="ntnu-smil/LTTC-Train-1764-0520"
    fi
    
    # Determine base model
    BASE_MODEL="ntnu-smil/Phi-4-multimodal-instruct_QA_NoImage_0325_1964"
    LEARNING_RATE="4e-5"
    
    # If working with 1964 dataset, check if 1764 model is available
    if [[ "$FORM_ID" == "1964" ]]; then
        BASE_1764_MODEL="ntnu-smil/phi-4-mm-lttc-binary-integrated-1764"
        
        # Check if 1764 model exists on Hugging Face
        echo "Checking if 1764 model exists at $BASE_1764_MODEL..."
        if python -c "from huggingface_hub import model_info; try: model_info('$BASE_1764_MODEL'); print('Model exists'); exit(0); except: print('Model not found'); exit(1);" 2>/dev/null; then
            echo "Found 1764 model. Using it as base for sequential fine-tuning."
            BASE_MODEL="$BASE_1764_MODEL"
            LEARNING_RATE="2e-5"  # Lower learning rate for continued fine-tuning
        else
            echo "1764 model not found. Using original base model."
        fi
    fi
    
    echo "Using base model: $BASE_MODEL"
    echo "Using learning rate: $LEARNING_RATE"
    
    # Clean memory before training
    free_memory
    
    # Run training with push to hub
    python fine-tune-lttc-binary.py \
        --model_name_or_path "$BASE_MODEL" \
        --dataset_name "$DATASET_NAME" \
        --eval_dataset "ntnu-smil/LTTC-Dev-1764-0520" \
        --binary_preds_path "$OUTPUT_FILE" \
        --form_id "$FORM_ID" \
        --audio_column "$AUDIO_COLUMN" \
        --eval_audio_column "wav_path" \
        --output_dir "./output_binary_integrated_${FORM_ID}" \
        --learning_rate "$LEARNING_RATE" \
        --num_train_epochs 3 \
        --global_batch_size 4 \
        --batch_size_per_gpu 1 \
        --eval_batch_size_per_gpu 1 \
        --push_to_hub \
        --hub_model_id "$HUB_MODEL_ID"
    
    # Clean memory after training
    free_memory
fi

echo "Processing complete!" 