import argparse
import os
import json
from pathlib import Path

import torch
from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from huggingface_hub import whoami, delete_file, upload_file
from datasets import load_dataset

from common import (
    collate_fn,
    evaluate,
    load_model_and_processor,
)

# Binary-integrated dataset class
class BinaryIntegratedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        processor,
        dataset_name,
        split,
        training,
        binary_preds_path,
        text_column="grade",
        audio_column="wav_path",
        question_column="prompt",
        max_samples=None,
        rank=0,
        world_size=1,
        dataset_subset=None,
        form_id="1764",
    ):
        self.processor = processor
        self.training = training
        self.text_column = text_column
        self.audio_column = audio_column
        self.question_column = question_column
        self.form_id = form_id
        
        # Load the dataset
        self.data = (
            load_dataset(dataset_name, dataset_subset, split=split)
            if dataset_subset
            else load_dataset(dataset_name, split=split)
        )
        
        if max_samples is not None:
            self.data = self.data.select(range(max_samples))
        
        if world_size > 1:
            self.data = self.data.shard(num_shards=world_size, index=rank)
        
        # Load binary predictions
        print(f"Loading binary predictions from: {binary_preds_path}")
        try:
            with open(binary_preds_path, 'r') as f:
                binary_data = json.load(f)
                self.binary_predictions = binary_data.get("binary_predictions", {})
                # Check if the audio column in the binary data matches the one provided
                binary_audio_column = binary_data.get("audio_column")
                if binary_audio_column and binary_audio_column != audio_column:
                    print(f"Warning: Audio column in binary data ({binary_audio_column}) "
                         f"differs from provided audio column ({audio_column})")
        except Exception as e:
            print(f"Error loading binary predictions: {e}")
            self.binary_predictions = {}
        
        print(f"Loaded dataset with {len(self.data)} samples")
        print(f"Using audio column: {self.audio_column}")
        print(f"Loaded {len(self.binary_predictions)} binary predictions")
        
        # Define prompts based on form_id
        # Modified to include binary model guidance
        self.instruction_text = self.get_instruction_text()
    
    def get_instruction_text(self):
        """Get instruction text with binary model integration"""
        
        instruction = """
"role": "system",
"content": "You are responsible for evaluating English pronunciation. The provided audio is a response to the question below. \\
I will also provide you with a binary model prediction in the format '0,0.80' or '1,0.85', where \\
the first number (0 or 1) indicates if the audio fails (0) or passes (1) a minimum standard, \\
and the second number indicates the confidence (probability) of that prediction. \\
Your task is to objectively assign scores from 1 to 5 based on the following criteria: sentence-level accuracy, fluency, prosody, completeness, and relevance to the question. \\
When the binary output is 0, the actual score should probably be between 1-3. \\
When the binary output is 1, the actual score should probably be between 4-5. \\
Your evaluation should remain neutral, fair, and utilize the binary model output as a guide. \\
Assign scores according to the following criteria:"
,
"role": "system",
"content": "Score 5: Pronunciation and intonation are correct and natural. The speaker expresses themselves fluently, with no communication barriers. \\
Score 4: Pronunciation and intonation are mostly correct and natural. There are some errors, but they do not hinder understanding. The speaker expresses themselves fairly fluently without communication barriers. \\
Score 3: Pronunciation and intonation occasionally have errors but remain understandable. The speaking speed is slower, with occasional pauses, but communication is still achievable. \\
Score 2: Pronunciation and intonation frequently have errors, which affect the listener's understanding. The speaking speed is slow, with frequent pauses that impact the delivery. \\
Score 1: Pronunciation and intonation have numerous errors, with many inappropriate pauses, making it difficult for the listener to understand. \\
Please follow these guidelines when evaluating the score provided."
,
"role": "user", 
"content": "You should only return the final score of the following input. The output should be a float number between 1 and 5 with one decimal place. Without any other information or text, format: 3.0"
,
"role": "user", 
"content": "Binary model prediction: {binary_pred} \\
Score according to the criteria above and the audio given to you"
"""
        return instruction
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Each example in the dataset has:
          - '{audio_column}': a dict with keys "array" and "sampling_rate"
          - '{text_column}': the answer score of the audio
          - Adding binary prediction to the prompt
        """
        data = self.data[idx]
        
        # Get binary prediction for this sample if available
        binary_pred = self.binary_predictions.get(str(idx), "No binary prediction")
        
        # Format instruction with binary prediction
        formatted_instruction = self.instruction_text.format(binary_pred=binary_pred)
        
        # Format user message with question and audio
        user_message = {
            "role": "user",
            "content": formatted_instruction + self.question_column + "<|audio_1|>",
        }
        
        prompt = self.processor.tokenizer.apply_chat_template(
            [user_message], tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=prompt,
            audios=[
                (
                    data[self.audio_column]["array"],
                    data[self.audio_column]["sampling_rate"],
                )
            ],
            return_tensors="pt",
        )
        
        answer = f"{data[self.text_column]}<|end|><|endoftext|>"
        answer_ids = self.processor.tokenizer(answer, return_tensors="pt").input_ids

        if self.training:
            input_ids = torch.cat([inputs.input_ids, answer_ids], dim=1)
            labels = torch.full_like(input_ids, -100)  # -100 is ignore index
            labels[:, -answer_ids.shape[1] :] = answer_ids
        else:
            input_ids = inputs.input_ids
            labels = answer_ids

        return {
            "input_ids": input_ids,
            "labels": labels,
            "input_audio_embeds": inputs.input_audio_embeds,
            "audio_embed_sizes": inputs.audio_embed_sizes,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="ntnu-smil/Phi-4-multimodal-instruct_QA_NoImage_0325_1964", #classification model
        help="Model name or path to load from",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ntnu-smil/LTTC-Train-1764-0520",
        help="Dataset name to use for training and evaluation",
    )
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="ntnu-smil/LTTC-Dev-1764-0520",
        help="Dataset split to use for evaluation",
    )
    parser.add_argument(
        "--binary_preds_path",
        type=str,
        required=True,
        help="Path to JSON file with binary model predictions for training data",
    )
    parser.add_argument(
        "--eval_binary_preds_path",
        type=str,
        help="Path to JSON file with binary model predictions for eval data",
    )
    parser.add_argument(
        "--dataset_subset",
        type=str,
        default = None,
        help="Dataset subset to use (e.g., 'zh-TW' for Common Voice 19)",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        help="Dataset split to use for training",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="train",
        help="Dataset split to use for testing",
    )
    
    parser.add_argument(
        "--max_train_samples",
        type=int,
        help="Maximum number of training samples",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        help="Maximum number of evaluation samples",
    )
    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
        help="Use Flash Attention for more efficient training on compatible hardware",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_binary_integrated/",
        help="Output directory for saving model checkpoints and logs",
    )
    parser.add_argument(
        "--global_batch_size",
        type=int,
        default=16,
        help="Total batch size across all GPUs (global_batch_size = batch_size_per_gpu * num_gpus * gradient_accumulation_steps)",
    )
    parser.add_argument(
        "--batch_size_per_gpu",
        type=int,
        default=1,
        help="Training batch size per GPU (decrease this value if you encounter OOM errors)",
    )
    parser.add_argument(
        "--eval_batch_size_per_gpu",
        type=int,
        default=1,
        help="Evaluation batch size per GPU (can typically be larger than training batch size since no gradients are stored)",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of complete passes through the training dataset",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=4.0e-5,
        help="Peak learning rate for optimizer",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay coefficient for regularization",
    )
    parser.add_argument(
        "--eval_metric",
        type=str,
        default="both",
        choices=["binary", "cl", "both"],
        help="Evaluation metric: 'binary' for binary classification, 'cl' for traditional classfication, 'both' for both",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="grade",
        help="Name of the column containing the answer score",
    )
    parser.add_argument(
        "--audio_column",
        type=str,
        default="wav_path",
        help="Name of the column containing the audio data for training dataset",
    )
    parser.add_argument(
        "--eval_audio_column",
        type=str,
        default="wav_path",
        help="Name of the column containing the audio data for evaluation dataset",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model to the Hugging Face Hub after training",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default="ntnu-smil/phi-4-multimodal-instruct-lttc-binary-integrated",
        help="Repository name to push to on the Hugging Face Hub",
    )
    parser.add_argument(
        "--form_id",
        type=str,
        default="1764",
        help="Form ID for the dataset (1764 or 1964)",
    )
    parser.add_argument(
        "--no-tqdm", dest="tqdm", action="store_false", help="Disable tqdm"
    )
    parser.add_argument(
        "--skip_initial_eval",
        action="store_true",
        help="Skip evaluation before training",
    )
    args = parser.parse_args()

    # Clean up memory before starting
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    print("Memory cleaned before starting...")

    if args.model_name_or_path.endswith("/"):
        raise ValueError(
            f"Invalid model_name_or_path: '{args.model_name_or_path}'. It should not end with a '/'."
        )

    if args.push_to_hub:
        try:
            whoami()
        except Exception as e:
            print(
                "You need to be logged in to the Hugging Face Hub to push the model. "
                "Please run `huggingface-cli login`."
            )
            raise e
        if args.hub_model_id is None:
            raise ValueError("Please provide a valid repository name to push to the Hugging Face Hub.")

        print(f"Push model to {args.hub_model_id} after training.")
    accelerator = Accelerator()

    with accelerator.local_main_process_first():
        model, processor = load_model_and_processor(
            args.model_name_or_path,
            use_flash_attention=args.use_flash_attention,
        )

    model.set_lora_adapter("speech")

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    

    print(f"Rank: {rank}, World Size: {world_size}")
    print(f"Model: {model.__class__.__name__}")
    print(f"eval_dataset: {args.eval_dataset}")
    print(f"train_dataset: {args.dataset_name}")
    print(f"Train audio column: {args.audio_column}")
    print(f"Eval audio column: {args.eval_audio_column}")
    
    # Load datasets with binary predictions integrated
    eval_dataset = BinaryIntegratedDataset(
        processor,
        dataset_name=args.eval_dataset,
        split=args.eval_split,
        training=False,
        binary_preds_path=args.eval_binary_preds_path if args.eval_binary_preds_path else args.binary_preds_path,
        text_column=args.text_column,
        audio_column=args.eval_audio_column,
        max_samples=args.max_eval_samples,
        rank=rank,
        world_size=world_size,
        dataset_subset=args.dataset_subset,
        form_id=args.form_id,
    )
    
    train_dataset = BinaryIntegratedDataset(
        processor,
        dataset_name=args.dataset_name,
        split=args.train_split,
        training=True,
        binary_preds_path=args.binary_preds_path,
        text_column=args.text_column,
        audio_column=args.audio_column,
        max_samples=args.max_train_samples,
        rank=rank,
        world_size=world_size,
        dataset_subset=args.dataset_subset,
        form_id=args.form_id,
    )

    # Debugging: Print dataset sizes
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")

    # Check for empty datasets
    if len(train_dataset) == 0:
        raise ValueError("Train dataset is empty. Please check the dataset configuration.")
    if len(eval_dataset) == 0:
        raise ValueError("Eval dataset is empty. Please check the dataset configuration.")

    num_gpus = accelerator.num_processes
    print(f"training on {num_gpus} GPUs")
    assert (
        args.global_batch_size % (num_gpus * args.batch_size_per_gpu) == 0
    ), "Batch size must be divisible by the number of GPUs"
    gradient_accumulation_steps = args.global_batch_size // (
        num_gpus * args.batch_size_per_gpu
    )

    if args.use_flash_attention:
        fp16 = False
        bf16 = True
    else:
        fp16 = True
        bf16 = False

    training_args = TrainingArguments(
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size_per_gpu,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-7,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=1.0,
        lr_scheduler_type="linear",
        warmup_steps=50,
        logging_steps=20,
        output_dir=args.output_dir,
        save_strategy="epoch",
        save_total_limit=3,
        # eval_strategy="epoch",
        # load_best_model_at_end=True,
        # metric_for_best_model="eval_loss",
        # greater_is_better=False,
        save_only_model=True,
        bf16=bf16,
        fp16=fp16,
        per_device_eval_batch_size=args.eval_batch_size_per_gpu,
        dataloader_drop_last=True,  # Drop last incomplete batch to avoid OOM
        dataloader_pin_memory=False,  # Disable pinned memory to reduce memory usage
        remove_unused_columns=False,
        report_to=["tensorboard"],
        deepspeed=None,
        disable_tqdm=not args.tqdm,
        dataloader_num_workers=4,
        ddp_find_unused_parameters=True,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id if args.hub_model_id else None,
    )

    out_path = Path(training_args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Make initial evaluation optional based on the flag
    if not args.skip_initial_eval:
        score = evaluate(
            model,
            processor,
            eval_dataset,
            save_path=out_path / "eval_before.json",
            disable_tqdm=not args.tqdm,
            eval_batch_size=args.eval_batch_size_per_gpu,
            metric=args.eval_metric,
        )
        if accelerator.is_main_process:
            metric_name = args.eval_metric.upper()
            metric_value = score.get("accuracy")
            if metric_value is not None:
                print(
                    f"{metric_name} Score before finetuning: {metric_value:.4f}"
                )
            else:
                print(
                    f"{metric_name} Score before finetuning could not be computed. Please check the evaluation setup."
                )

    # Modify dataloader to ensure memory is cleaned up
    def collate_wrapper(*args, **kwargs):
        try:
            # Call original collate function
            result = collate_fn(*args, **kwargs)
            
            # Ensure tensors are the correct shape
            if isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, torch.Tensor) and value.dim() == 0:
                        print(f"Warning: Found scalar tensor for {key}, expanding dimensions...")
                        result[key] = value.unsqueeze(0)
            
            return result
        except Exception as e:
            print(f"Error in collate function: {e}")
            # On error, try to clean up memory
            gc.collect()
            torch.cuda.empty_cache()
            raise
    
    # Use wrapper instead of original collate_fn
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_wrapper,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    trainer.save_model()
    if accelerator.is_main_process:
        processor.save_pretrained(training_args.output_dir)
        if args.push_to_hub:
            processor.push_to_hub(args.hub_model_id)

    if args.push_to_hub:
        # we need to remove chat_template.json on the hub
        delete_file(
            repo_id=args.hub_model_id,
            path_in_repo="chat_template.json",
            repo_type="model",
        )
        # we need to overwrite preprocessor_config.json on the hub
        preprocessor_config = {
            "auto_map": {
                "AutoFeatureExtractor": "microsoft/Phi-4-multimodal-instruct--processing_phi4mm.Phi4MMAudioFeatureExtractor",
                "AutoImageProcessor": "microsoft/Phi-4-multimodal-instruct--processing_phi4mm.Phi4MMImageProcessor",
                "AutoProcessor": "microsoft/Phi-4-multimodal-instruct--processing_phi4mm.Phi4MMProcessor",
            },
            "feature_extractor_type": "Phi4MMAudioFeatureExtractor",
            "image_processor_type": "Phi4MMImageProcessor",
            "processor_class": "Phi4MMProcessor",
            "audio_compression_rate": 8,
            "audio_downsample_rate": 1,
            "audio_feat_stride": 1,
            "dynamic_hd": 36,
        }
        upload_file(
            path_or_fileobj=json.dumps(preprocessor_config, indent=2).encode(),
            path_in_repo="preprocessor_config.json",
            repo_id=args.hub_model_id,
            repo_type="model",
        )

    accelerator.wait_for_everyone()

    # Cleanup memory before loading new model
    del model
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    print("Memory cleaned after training...")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            training_args.output_dir,
            torch_dtype=torch.bfloat16 if args.use_flash_attention else torch.float32,
            trust_remote_code=True,
            _attn_implementation=(
                "flash_attention_2" if args.use_flash_attention else "sdpa"
            ),
        ).to("cuda")
    except ModuleNotFoundError as e:
        print(f"Error loading model: {e}")
        print("Ensure that the required modules are available and correctly referenced.")
        raise e

    score = evaluate(
        model,
        processor,
        eval_dataset,
        save_path=out_path / "eval_after.json",
        disable_tqdm=not args.tqdm,
        eval_batch_size=args.eval_batch_size_per_gpu,
        metric=args.eval_metric,
    )
    if accelerator.is_main_process:
        metric_name = args.eval_metric.upper()
        metric_value = score.get(args.eval_metric.lower())
        if metric_value is not None:
            print(
                f"{metric_name} Score after finetuning: {metric_value:.4f}"
            )
        else:
            print(
                f"{metric_name} Score after finetuning could not be computed. Please check the evaluation setup."
            )


if __name__ == "__main__":
    main() 