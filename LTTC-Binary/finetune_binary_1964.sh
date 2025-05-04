#!/bin/bash

# Shell script to fine-tune the binary classification model on Unseen_1964 dataset

# Set environment variables
source /root/miniconda3/etc/profile.d/conda.sh
conda activate Phi4

# Configuration parameters
FORM_ID='1964'
DATE=$(date +"%m%d")
LEARNING_RATE='5.0e-5'
MODEL_TYPE="Phi-4-mm_Binary_QA_NI_0415_${FORM_ID}" #further fine-tuned on 1964 dataset

# Choose the pretrained model
# Option 1: Start from the base model
# BASE_MODEL="microsoft/Phi-4-multimodal-instruct"

# Option 2: Start from the previously fine-tuned model on 1764 dataset
BASE_MODEL="ntnu-smil/Phi-4-mm_Binary_QA_NI_0415_1764"

# Dataset paths
TRAIN_FILE="ntnu-smil/Unseen_1964"
DEV_FILE="ntnu-smil/Unseen_1964"

# Output directory
EXP_DIR="./LTTC-Binary/output/IS-${FORM_ID}/${MODEL_TYPE}"

echo "Starting fine-tuning on ${TRAIN_FILE}..."
echo "Form ID: ${FORM_ID}"
echo "Model type: ${MODEL_TYPE}"
echo "Base model: ${BASE_MODEL}"
echo "Output directory: ${EXP_DIR}"

# Run the fine-tuning script
CUDA_VISIBLE_DEVICES=0 \
python3 fine-tune-binary.py \
    --model_name_or_path "${BASE_MODEL}" \
    --use_flash_attention \
    --output_dir "${EXP_DIR}" \
    --global_batch_size 16 \
    --num_train_epochs 6 \
    --learning_rate "${LEARNING_RATE}" \
    --dataset_name "${TRAIN_FILE}" \
    --eval_dataset "${DEV_FILE}" \
    --eval_batch_size_per_gpu 2 \
    --push_to_hub \
    --hub_model_id "ntnu-smil/${MODEL_TYPE}" \
    --form_id "${FORM_ID}" \
    --train_split "train" \
    --eval_split "validation" \
    --text_column "grade" \
    --audio_column "wav_file"

echo "Fine-tuning completed!" 