#!/bin/bash

# Script to evaluate the SOTA binary-integrated model on unseen1964 fulltest split
# This script follows the binary-integrated pipeline, using binary model first

# Set variables
BINARY_MODEL_PATH="ntnu-smil/Phi-4-mm_Binary_QA_NI_0415_1964"
SOTA_MODEL_PATH="ntnu-smil/phi-4-mm-lttc-binary-integrated-1964"
DATASET="ntnu-smil/Unseen_1964"
SPLIT="test"
OUTPUT_DIR="./sota_eval_results_integrated"
AUDIO_COLUMN="wav_file"  # Note: Unseen_1964 uses "wav_file" instead of "wav_path"

# Create output directory
mkdir -p $OUTPUT_DIR

# Activate the conda environment
source /root/miniconda3/etc/profile.d/conda.sh
conda activate Phi4

# Clean up memory before running evaluation
echo "Cleaning up memory..."
python -c "import gc; gc.collect()" || true
python -c "import torch; torch.cuda.empty_cache()" || true
sleep 2

# Check if we should skip binary evaluation (if it was already done)
BINARY_PREDS_PATH="$OUTPUT_DIR/binary_preds_Unseen_1964_$SPLIT.json"
SKIP_BINARY=""
if [ -f "$BINARY_PREDS_PATH" ]; then
    echo "Found existing binary predictions at $BINARY_PREDS_PATH"
    echo "Use --skip_binary_eval flag to skip binary evaluation? (y/n)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        SKIP_BINARY="--skip_binary_eval"
        echo "Will skip binary evaluation and use existing predictions"
    else
        echo "Will run full binary-integrated pipeline"
    fi
fi

# Run evaluation
echo "Starting binary-integrated evaluation of SOTA model on $DATASET $SPLIT split..."
python eval_sota_unseen1964_binary_integrated.py \
    --binary_model_path $BINARY_MODEL_PATH \
    --model_name_or_path $SOTA_MODEL_PATH \
    --dataset_name $DATASET \
    --split $SPLIT \
    --output_dir $OUTPUT_DIR \
    --audio_column $AUDIO_COLUMN \
    --batch_size 4 \
    --use_flash_attention \
    $SKIP_BINARY

# Report completion
echo "Evaluation complete!"
echo "Binary predictions saved to: $OUTPUT_DIR/binary_preds_Unseen_1964_${SPLIT}.json"
echo "Final results saved to: $OUTPUT_DIR/phi-4-mm-lttc-binary-integrated-1964_Unseen_1964_${SPLIT}_binary_integrated.json" 