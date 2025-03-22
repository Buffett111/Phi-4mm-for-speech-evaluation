#shell file to run fine-tune-lttc.py

# source /root/miniconda3/etc/profile.d/conda.sh
# conda activate Phi4
# ======================= Intermediate ======================= 
form_id='1764' # 1572, 1764, 1766, 1964(Main)
date='03203'
LEARNING_RATE='4.0e-5' # 1e-3
#microsoft/Phi-4-multimodal-instruct
module_type="Phi-4-multimodal-instruct" 

train_file="ntnu-smil/LTTC-Train-1764-0520"
dev_file="ntnu-smil/LTTC-Dev-1764-0520"

exp_dir="./LTTC-Intermediate/IS-${form_id}/${module_type}_${date}_${LEARNING_RATE}_roundown"


# train: train.py, train_softlabel, train_wav2vec, train_subModel, pretrained Model

source /root/miniconda3/etc/profile.d/conda.sh
conda activate Phi4

CUDA_VISIBLE_DEVICES=0 
python3 fine-tune-lttc.py \
    --model_name_or_path microsoft/${module_type} \
    --use_flash_attention \
    --output_dir ${exp_dir} \
    --global_batch_size 16 \
    --num_train_epochs 1 \
    --learning_rate ${LEARNING_RATE} \
    --dataset_name ${train_file} \
    --eval_dataset ${dev_file} \
    --eval_batch_size_per_gpu 1 \
    --skip_initial_eval