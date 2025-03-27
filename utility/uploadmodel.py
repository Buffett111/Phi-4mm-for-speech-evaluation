from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, HfFolder, Repository
from huggingface_hub import whoami, delete_file, upload_file

# Path to your model directory
model_dir = "./LTTC-Intermediate/IS-1964/phi-4-multimodal-instruct-lttc-NoQA-NoImage_0323/"

# Hugging Face repository ID (e.g., "username/model-name")
repo_id = "ntnu-smil/Phi-4-multimodal-instruct-0323-lttc"

# Push the model to Hugging Face
model = AutoModelForCausalLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Save and push the model and tokenizer
model.push_to_hub(repo_id)
tokenizer.push_to_hub(repo_id)

upload_file(
    path_or_fileobj=json.dumps(preprocessor_config, indent=2).encode(),
    path_in_repo="preprocessor_config.json",
    repo_id=args.hub_model_id,
    repo_type="model",
)
print(f"Model uploaded to https://huggingface.co/{repo_id}")