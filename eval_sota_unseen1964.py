#!/usr/bin/env python
# Evaluation script for the SOTA model on unseen1964 fulltest split

import argparse
import os
import json
from pathlib import Path

import torch
from accelerate import Accelerator
from tqdm import tqdm

from common import (
    EvalDataset,
    evaluate,
    load_model_and_processor,
)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SOTA model on unseen1964 fulltest split"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="ntnu-smil/phi-4-mm-lttc-binary-integrated-1964",
        help="Model name or path to load from",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ntnu-smil/Unseen_1964",
        help="Dataset name to use for evaluation",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="fulltest",
        help="Dataset split to use for evaluation",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        help="Maximum number of evaluation samples",
    )
    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
        help="Use Flash Attention for more efficient inference on compatible hardware",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./sota_eval_results/",
        help="Output directory for saving evaluation results",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for evaluation"
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
        default="wav_file",  # Note: Using wav_file for Unseen_1964
        help="Name of the column containing the audio data",
    )
    parser.add_argument(
        "--no-tqdm", dest="tqdm", action="store_false", help="Disable tqdm progress bar"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="both",
        choices=["binary", "cl", "both"],
        help="Evaluation metric: 'binary' for binary classification, 'cl' for traditional classfication, 'both' for both",
    )
    args = parser.parse_args()

    print(f"Evaluating {args.model_name_or_path} on {args.dataset_name} {args.split}...")
    
    # Initialize accelerator
    accelerator = Accelerator()

    # Load model and processor
    with accelerator.local_main_process_first():
        model, processor = load_model_and_processor(
            args.model_name_or_path,
            use_flash_attention=args.use_flash_attention,
        )

    # Get rank and world size for distributed processing
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Create eval dataset
    eval_dataset = EvalDataset(
        processor,
        dataset_name=args.dataset_name,
        split=args.split,
        text_column=args.text_column,
        audio_column=args.audio_column,
        max_samples=args.max_samples,
        rank=rank,
        world_size=world_size,
    )

    # Create output directory if it doesn't exist
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Generate a descriptive filename for the results
    dataset_name = args.dataset_name.split("/")[-1]
    model_name = args.model_name_or_path.split("/")[-1]
    results_filename = f"{model_name}_{dataset_name}_{args.split}.json"
    save_path = out_path / results_filename

    # Run evaluation
    results = evaluate(
        model,
        processor,
        eval_dataset,
        save_path=save_path,
        disable_tqdm=not args.tqdm,
        eval_batch_size=args.batch_size,
        metric=args.metric,
    )

    if accelerator.is_main_process:
        print(f"Evaluation completed with following results:")
        if "accuracy" in results:
            print(f"Accuracy: {results['accuracy']:.4f}")
        if "absolute_accuracy" in results:
            print(f"Absolute Accuracy: {results['absolute_accuracy']:.4f}")
        if "binary_accuracy" in results:
            print(f"Binary Accuracy: {results['binary_accuracy']:.4f}")
        print(f"Number of samples: {results['num_samples']}")
        print(f"Evaluation results saved to {save_path}")


if __name__ == "__main__":
    main() 