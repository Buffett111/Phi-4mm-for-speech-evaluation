#!/bin/bash

# Shell script to evaluate the binary classification model

# Set environment variables
source /root/miniconda3/etc/profile.d/conda.sh
conda activate Phi4

# Get command line arguments for form_id and dataset
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <form_id> [dataset_name]"
    echo "  form_id: Either 1764 or 1964"
    echo "  dataset_name: Optional dataset name (default: depends on form_id)"
    exit 1
fi

FORM_ID="$1"

# Set default dataset based on form_id if not provided
if [ "$#" -lt 2 ]; then
    if [ "$FORM_ID" == "1764" ]; then
        DATASET="ntnu-smil/LTTC-Dev-1764-0520"
    else
        DATASET="ntnu-smil/Unseen_1964"
    fi
else
    DATASET="$2"
fi

# Configuration parameters
MODEL_TYPE="Phi-4-mm_Binary_QA_NI_0415_1764"
# MODEL_TYPE="Phi-4-mm_Binary_QA_NI_0415_1964"
MODEL="ntnu-smil/${MODEL_TYPE}"
SPLIT="train"
AUDIO_COLUMN="wav_path"

# For Unseen_1964 dataset, audio column is different
if [[ "$DATASET" == *"Unseen_1964"* ]]; then
    AUDIO_COLUMN="wav_file"
    SPLIT="fulltest"
fi

# Output directory
EXP_DIR="./LTTC-Binary/eval_results/IS-${FORM_ID}"

echo "Starting evaluation..."
echo "Form ID: ${FORM_ID}"
echo "Model: ${MODEL}"
echo "Dataset: ${DATASET}"
echo "Split: ${SPLIT}"
echo "Output directory: ${EXP_DIR}"

# Create output directory if it doesn't exist
mkdir -p "${EXP_DIR}"

# Run the evaluation script
CUDA_VISIBLE_DEVICES=0 \
python3 eval_binary.py \
    --model_name_or_path "${MODEL}" \
    --use_flash_attention \
    --output_dir "${EXP_DIR}" \
    --dataset_name "${DATASET}" \
    --split "${SPLIT}" \
    --audio_column "${AUDIO_COLUMN}" \
    --form_id "${FORM_ID}" \
    --metric "both" \
    --batch_size 8

echo "Evaluation completed!" 