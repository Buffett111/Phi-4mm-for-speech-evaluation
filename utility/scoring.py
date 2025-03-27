import argparse
from common import load_model_and_processor, scoring_audio
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Score audio using Phi-4-multimodal model"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="microsoft/Phi-4-multimodal-instruct",
        help="Model name or path to load from",
    )
    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
        help="Use Flash Attention for more efficient inference on compatible hardware",
    )
    args = parser.parse_args()

    # Load model and processor
    print(f"Loading model from {args.model_name_or_path}...")
    model, processor = load_model_and_processor(
        args.model_name_or_path, use_flash_attention=args.use_flash_attention
    )

    # Load dataset
    ds = load_dataset("ntnu-smil/LTTC-Train1964-0520")
    ds = ds["train"].select(range(10))  # Load first 10 examples

    for example in ds:
        # Process each audio file
        print(f"Processing audio: {example['wav_path']}")
        print(f"Actual anser: {example['grade']}")
        transcription = scoring_audio(
            model,
            processor,
            audio_path=example["wav_path"],  # Pass wav file path
            question=example["prompt"]
        )

        # Print the transcription
        print("\nScoring Result:")
        print("-" * 40)
        print(transcription)
        print("-" * 40)


if __name__ == "__main__":
    main()