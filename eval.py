import argparse
import os
from pathlib import Path

from accelerate import Accelerator

from common import (
    EvalDataset,
    evaluate,
    load_model_and_processor,
)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Scoring model on LTTC dataset"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="microsoft/Phi-4-multimodal-instruct",
        help="Model name or path to load from",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ntnu-smil/LTTC-Dev-1764-0520",
        help="Dataset name to use for evaluation",
    )
    parser.add_argument(
        "--dataset_subset",
        type=str,
        help="Dataset subset to use (e.g., 'zh-TW' for Common Voice 19)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
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
        default="./eval_results/",
        help="Output directory for saving evaluation results",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for evaluation"
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

    accelerator = Accelerator()

    with accelerator.local_main_process_first():
        model, processor = load_model_and_processor(
            args.model_name_or_path,
            use_flash_attention=args.use_flash_attention,
        )

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    eval_dataset = EvalDataset(
        processor,
        dataset_name=args.dataset_name,
        split=args.split,
        text_column=args.text_column,
        audio_column=args.audio_column,
        max_samples=args.max_samples,
        rank=rank,
        world_size=world_size,
        dataset_subset=args.dataset_subset,
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
    evaluate(
        model,
        processor,
        eval_dataset,
        save_path=save_path,
        disable_tqdm=not args.tqdm,
        eval_batch_size=args.batch_size,
        metric=args.metric,
    )

    if accelerator.is_main_process:
        print(f"Evaluation results saved to {save_path}")


if __name__ == "__main__":
    main()