import argparse
import os
from pathlib import Path
import json

import torch
from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from huggingface_hub import whoami, delete_file, upload_file

from common import (
    FinetuneDataset,
    collate_fn,
    evaluate,
    load_model_and_processor,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="microsoft/Phi-4-multimodal-instruct",
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
        default="./output/",
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
        default="cl",
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
        help="Name of the column containing the audio data",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model to the Hugging Face Hub after training",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        help="Repository name to push to on the Hugging Face Hub (e.g., 'username/model-name')",
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

    if args.push_to_hub:
        try:
            whoami()
        except Exception as e:
            print(
                "You need to be logged in to the Hugging Face Hub to push the model. "
                "Please run `huggingface-cli login`."
            )
            raise e

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
    
    eval_dataset = FinetuneDataset(
        processor,
        dataset_name=args.eval_dataset,
        split=args.eval_split,
        training=False,
        text_column=args.text_column,
        audio_column=args.audio_column,
        max_samples=args.max_eval_samples,
        rank=rank,
        world_size=world_size,
        dataset_subset=args.dataset_subset,
    )
    train_dataset = FinetuneDataset(
        processor,
        dataset_name=args.dataset_name,
        split=args.train_split,
        training=True,
        text_column=args.text_column,
        audio_column=args.audio_column,
        max_samples=args.max_train_samples,
        rank=rank,
        world_size=world_size,
        dataset_subset=args.dataset_subset,
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
        logging_steps=10,
        output_dir=args.output_dir,
        save_strategy="epoch",
        save_total_limit=3,
        save_only_model=True,
        bf16=bf16,
        fp16=fp16,
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

    # Make initial evaluation optional based on the new flag
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

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_dataset,
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

    # Evaluate after fine-tuning
    del model
    del trainer
    __import__("gc").collect()
    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        training_args.output_dir,
        torch_dtype=torch.bfloat16 if args.use_flash_attention else torch.float32,
        trust_remote_code=True,
        _attn_implementation=(
            "flash_attention_2" if args.use_flash_attention else "sdpa"
        ),
    ).to("cuda")

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