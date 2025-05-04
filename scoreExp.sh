

source /root/miniconda3/etc/profile.d/conda.sh
conda activate Phi4
date='0505'
form_id='1964'
#module_type="phi-4-multimodal-instruct"
#module_type="phi-4-multimodal-instruct-lttc"
#module_type="Phi-4-multimodal-instruct_QA_NoImage_0325"
module_type="phi-4-mm-lttc-sla-exp-01" 
exp_dir="./LTTC-Intermediate/phi-4-mm-lttc-sla-exp-01"

# watch -n 1 nvidia-smi
# --model_name_or_path "ntnu-smil/${module_type}" \
# ntnu-smil/LTTC-Train1964-0520
python3 eval.py \
    --model_name_or_path "ntnu-smil/${module_type}" \
    --use_flash_attention \
    --output_dir "${exp_dir}" \
    --metric "both" \
    --dataset_name "ntnu-smil/Unseen_1964" \
    --audio_column "wav_file" \
    --split "fulltest" \

# --dataset_name "ntnu-smil/LTTC-Dev-${form_id}-0520" \