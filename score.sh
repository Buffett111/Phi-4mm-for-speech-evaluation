

source /root/miniconda3/etc/profile.d/conda.sh
conda activate Phi4

form_id='1964'
#module_type="phi-4-multimodal-instruct-lttc"
module_type="phi-4-multimodal-instruct-lttc-NoQA-NoImage-0323"
exp_dir="./LTTC-Intermediate/IS-${form_id}/eval_after"

python3 eval.py \
    --model_name_or_path "ntnu-smil/${module_type}" \
    --use_flash_attention \
    --output_dir "${exp_dir}" \
    --metric "both" \
    --dataset_name "ntnu-smil/LTTC-Dev-${form_id}-0520" \