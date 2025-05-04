#!/bin/bash

# Script to evaluate the SOTA binary-integrated model on unseen1964 fulltest split

# Set variables
MODEL_PATH="ntnu-smil/phi-4-mm-lttc-binary-integrated-1964"
DATASET="ntnu-smil/Unseen_1964"
SPLIT="fulltest"
OUTPUT_DIR="./sota_eval_results"
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

# Run evaluation
echo "Starting evaluation of SOTA model on unseen1964 fulltest split..."
python eval_sota_unseen1964.py \
    --model_name_or_path $MODEL_PATH \
    --dataset_name $DATASET \
    --split $SPLIT \
    --output_dir $OUTPUT_DIR \
    --audio_column $AUDIO_COLUMN \
    --batch_size 4 \
    --use_flash_attention

# Report completion
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT_DIR/phi-4-mm-lttc-binary-integrated-1964_Unseen_1964_fulltest.json" 