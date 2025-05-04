#!/bin/bash

# Script to run the binary-integrated model pipeline
# This script first evaluates datasets using the binary model
# Then uses those results to train the classification model

# Step 1: Set up environment variables
BINARY_MODEL="ntnu-smil/Phi-4-mm_Binary_QA_NI_0415_1964"
CLASS_MODEL="ntnu-smil/Phi-4-multimodal-instruct_QA_NoImage_0325_1964"
DATASET1="ntnu-smil/LTTC-Train-1764-0520"
DATASET2="ntnu-smil/Unseen_1964"
EVAL_DATASET="ntnu-smil/LTTC-Dev-1764-0520"

# Model IDs for Hugging Face Hub
MODEL_1764_ID="ntnu-smil/phi-4-mm-lttc-binary-integrated-1764"
MODEL_1964_ID="ntnu-smil/phi-4-mm-lttc-binary-integrated-1964"

# Audio column names differ between datasets
AUDIO_COLUMN_1764="wav_path"
AUDIO_COLUMN_1964="wav_file"
EVAL_AUDIO_COLUMN="wav_path"  # Assuming eval dataset uses the same column as dataset1

# Create output directories
BINARY_RESULTS_DIR="./binary_eval_results"
CLASS_OUTPUT_DIR="./output_binary_integrated"

mkdir -p $BINARY_RESULTS_DIR
mkdir -p $CLASS_OUTPUT_DIR

source /root/miniconda3/etc/profile.d/conda.sh
conda activate Phi4

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

# Check if binary results already exist
BINARY_META_PATH="$BINARY_RESULTS_DIR/binary_evaluation_metadata.json"

if [ -f "$BINARY_META_PATH" ]; then
    echo "Binary evaluation results already exist at $BINARY_META_PATH"
    echo "Skipping binary evaluation step..."
    
    # Check if we have all necessary prediction files
    DS1_NAME=$(echo $DATASET1 | cut -d'/' -f2)
    DS2_NAME=$(echo $DATASET2 | cut -d'/' -f2)
    
    PRED_FILE1="$BINARY_RESULTS_DIR/binary_preds_${DS1_NAME}_1764.json"
    PRED_FILE2="$BINARY_RESULTS_DIR/binary_preds_${DS2_NAME}_1964.json"
    
    if [ ! -f "$PRED_FILE1" ] || [ ! -f "$PRED_FILE2" ]; then
        echo "Warning: Some prediction files are missing. Running binary evaluation..."
        DO_BINARY_EVAL=true
    else
        echo "All binary prediction files found:"
        echo "- $PRED_FILE1"
        echo "- $PRED_FILE2"
        DO_BINARY_EVAL=false
    fi
else
    echo "No binary evaluation results found. Running binary evaluation..."
    DO_BINARY_EVAL=true
fi

# Step 2: Evaluate datasets using binary model if needed
if [ "$DO_BINARY_EVAL" = true ]; then
    echo "Step 1: Evaluating datasets using binary model..."
    python evaluate_binary_data.py \
        --binary_model_path $BINARY_MODEL \
        --dataset_name $DATASET1 \
        --dataset2 $DATASET2 \
        --output_dir $BINARY_RESULTS_DIR \
        --batch_size 4 \
        --audio_column_1764 $AUDIO_COLUMN_1764 \
        --audio_column_1964 $AUDIO_COLUMN_1964
    
fi

# Step 3: Train classification model with binary results
echo "Step 2: Training classification model with binary results..."

# Get binary prediction paths from results
if [ -f "$BINARY_META_PATH" ]; then
    # Extract binary prediction paths for each dataset
    BINARY_PREDS_1764=$(python -c "import json; f=open('$BINARY_META_PATH'); data=json.load(f); print(data['$DATASET1']['binary_preds_path'])")
    BINARY_PREDS_1964=$(python -c "import json; f=open('$BINARY_META_PATH'); data=json.load(f); print(data['$DATASET2']['binary_preds_path'])")
    
    # Extract audio column names (in case they were changed)
    AUDIO_COLUMN_1764=$(python -c "import json; f=open('$BINARY_META_PATH'); data=json.load(f); print(data['$DATASET1']['audio_column'])")
    AUDIO_COLUMN_1964=$(python -c "import json; f=open('$BINARY_META_PATH'); data=json.load(f); print(data['$DATASET2']['audio_column'])")
else
    echo "Error: Binary evaluation metadata file not found. Cannot proceed with training."
    exit 1
fi

# Step 3.1: Train first model with dataset 1 (form_id 1764)
echo "Step 3.1: Training with $DATASET1 (Form ID: 1764)..."
echo "Using audio column: $AUDIO_COLUMN_1764"
python fine-tune-lttc-binary.py \
    --model_name_or_path $CLASS_MODEL \
    --dataset_name $DATASET1 \
    --eval_dataset $EVAL_DATASET \
    --binary_preds_path $BINARY_PREDS_1764 \
    --form_id "1764" \
    --audio_column $AUDIO_COLUMN_1764 \
    --eval_audio_column $EVAL_AUDIO_COLUMN \
    --output_dir "${CLASS_OUTPUT_DIR}_1764" \
    --learning_rate 2e-6 \
    --num_train_epochs 1 \
    --global_batch_size 4 \
    --batch_size_per_gpu 4 \
    --eval_batch_size_per_gpu 4 \
    --push_to_hub \
    --use_flash_attention \
    --hub_model_id "$MODEL_1764_ID"

# Check if the training was successful
if [ $? -ne 0 ]; then
    echo "Error: Training on dataset 1 failed."
    exit 1
fi


# Step 3.3: Train second model using the first model as base
echo "Step 3.3: Training with $DATASET2 (Form ID: 1964) using $MODEL_1764_ID as base..."
echo "Using audio column: $AUDIO_COLUMN_1964"
python fine-tune-lttc-binary.py \
    --model_name_or_path "$MODEL_1764_ID" \
    --dataset_name $DATASET2 \
    --eval_dataset $EVAL_DATASET \
    --binary_preds_path $BINARY_PREDS_1964 \
    --form_id "1964" \
    --audio_column $AUDIO_COLUMN_1964 \
    --eval_audio_column $EVAL_AUDIO_COLUMN \
    --output_dir "${CLASS_OUTPUT_DIR}_1964" \
    --learning_rate 2e-6 \
    --num_train_epochs 1 \
    --global_batch_size 4 \
    --batch_size_per_gpu 4 \
    --eval_batch_size_per_gpu 4 \
    --push_to_hub \
    --use_flash_attention \
    --hub_model_id "$MODEL_1964_ID"


echo "Sequential binary-integrated model pipeline completed successfully!" 
echo "SOTA model should be available at: $MODEL_1964_ID" 