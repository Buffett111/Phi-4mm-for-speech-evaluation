#shell file to run fine-tune-lttc.py

# source /root/miniconda3/etc/profile.d/conda.sh
# conda activate Phi4
# ======================= Intermediate ======================= 
form_id='1964' # 1572, 1764, 1766, 1964(Main)
date='0323'
LEARNING_RATE='4.0e-5' # 1e-3
module_type="phi-4-multimodal-instruct-lttc"  # Removed space after '='

# module_type= "microsoft/Phi-4-multimodal-instruct"
# module_type= "models/LTTC-Intermediate/IS-1764/Phi-4-multimodal-instruct_0323" 


#train_file="ntnu-smil/LTTC-Train-1764-0520"
#dev_file="ntnu-smil/LTTC-Dev-1764-0520"

train_file="ntnu-smil/LTTC-Train1964-0520"
dev_file="ntnu-smil/LTTC-Dev-1964-0520"

exp_dir="./LTTC-Intermediate/IS-${form_id}/${module_type}_${date}"


# train: train.py, train_softlabel, train_wav2vec, train_subModel, pretrained Model

source /root/miniconda3/etc/profile.d/conda.sh
conda activate Phi4

CUDA_VISIBLE_DEVICES=0 \
python3 fine-tune-lttc.py \
    --model_name_or_path "ntnu-smil/${module_type}" \
    --use_flash_attention \
    --output_dir "${exp_dir}" \
    --global_batch_size 16 \
    --num_train_epochs 3 \
    --learning_rate "${LEARNING_RATE}" \
    --dataset_name "${train_file}" \
    --eval_dataset "${dev_file}" \
    --eval_batch_size_per_gpu 1 \
    --push_to_hub
    # --skip_initial_eval