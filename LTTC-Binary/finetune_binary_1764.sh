#!/bin/bash

# Shell script to fine-tune the binary classification model on LTTC-Train-1764-0520 dataset

# Set environment variables
source /root/miniconda3/etc/profile.d/conda.sh
conda activate Phi4

# Configuration parameters
FORM_ID='1764'
DATE=$(date +"%m%d")
LEARNING_RATE='5.0e-5'
MODEL_TYPE="Phi-4-mm_Binary_QA_NI_0415_${FORM_ID}"
BASE_MODEL="microsoft/Phi-4-multimodal-instruct"

# Dataset paths
TRAIN_FILE="ntnu-smil/LTTC-Train-1764-0520"
DEV_FILE="ntnu-smil/LTTC-Dev-1764-0520"

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
    --num_train_epochs 7 \
    --learning_rate "${LEARNING_RATE}" \
    --dataset_name "${TRAIN_FILE}" \
    --eval_dataset "${DEV_FILE}" \
    --global_batch_size 16 \
    --batch_size_per_gpu 4 \
    --eval_batch_size_per_gpu 8 \
    --push_to_hub \
    --hub_model_id "ntnu-smil/${MODEL_TYPE}" \
    --form_id "${FORM_ID}" \
    --train_split "train" \
    --eval_split "train" \
    #--skip_initial_eval

echo "Fine-tuning completed!" 