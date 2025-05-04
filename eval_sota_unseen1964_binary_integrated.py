#!/usr/bin/env python
# Evaluation script for the SOTA model on unseen1964 fulltest split
# Following the binary-integrated pipeline

import argparse
import os
import json
import sys
from pathlib import Path

import torch
from accelerate import Accelerator
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

# Add necessary paths for importing
sys.path.append("LTTC-Binary")

from common import (
    evaluate,
    load_model_and_processor,
)
from common_binary import (
    evaluate as evaluate_binary,
    load_model_and_processor as load_binary_model,
    EvalDataset as BinaryEvalDataset,
)


# Binary-integrated dataset class (simplified from fine-tune-lttc-binary.py)
class BinaryIntegratedEvalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        processor,
        dataset_name,
        split,
        binary_preds_path,
        text_column="grade",
        audio_column="wav_file",
        question_column="prompt",
        max_samples=None,
        rank=0,
        world_size=1,
        dataset_subset=None,
        form_id="1964",
    ):
        self.processor = processor
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
            "content": formatted_instruction + "<|audio_1|>",
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

        input_ids = inputs.input_ids
        labels = answer_ids

        return {
            "input_ids": input_ids,
            "labels": labels,
            "input_audio_embeds": inputs.input_audio_embeds,
            "audio_embed_sizes": inputs.audio_embed_sizes,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SOTA model on unseen1964 fulltest split using binary-integrated pipeline"
    )
    # Binary model settings
    parser.add_argument(
        "--binary_model_path",
        type=str,
        default="ntnu-smil/Phi-4-mm_Binary_QA_NI_0415_1964",
        help="Path to binary model",
    )
    # SOTA model settings
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="ntnu-smil/phi-4-mm-lttc-binary-integrated-1964",
        help="Model name or path to load from",
    )
    # Dataset settings
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
    # General settings
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
        "--skip_binary_eval",
        action="store_true",
        help="Skip binary evaluation (use existing binary predictions)",
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

    # Create output directory if it doesn't exist
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Generate descriptive filenames
    dataset_name = args.dataset_name.split("/")[-1]
    model_name = args.model_name_or_path.split("/")[-1]
    
    # Binary evaluation phase
    binary_preds_filename = f"binary_preds_{dataset_name}_{args.split}.json"
    binary_preds_path = out_path / binary_preds_filename
    
    if not args.skip_binary_eval or not os.path.exists(binary_preds_path):
        print(f"Step 1: Running binary evaluation with {args.binary_model_path}...")
        
        # Load binary model and processor
        with accelerator.local_main_process_first():
            binary_model, binary_processor = load_binary_model(
                args.binary_model_path,
                use_flash_attention=args.use_flash_attention,
            )
        
        # Get rank and world size for distributed processing
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        # Create binary eval dataset
        binary_eval_dataset = BinaryEvalDataset(
            binary_processor,
            dataset_name=args.dataset_name,
            split=args.split,
            text_column=args.text_column,
            audio_column=args.audio_column,
            max_samples=args.max_samples,
            rank=rank,
            world_size=world_size,
        )
        
        # Run binary evaluation
        binary_results = evaluate_binary(
            binary_model,
            binary_processor,
            binary_eval_dataset,
            save_path=binary_preds_path,
            disable_tqdm=not args.tqdm,
            eval_batch_size=args.batch_size,
            metric="binary",
        )
        
        if accelerator.is_main_process:
            print(f"Binary evaluation completed:")
            if isinstance(binary_results, dict):
                for metric, value in binary_results.items():
                    if isinstance(value, (int, float)):
                        print(f"{metric}: {value:.4f}")
                print(f"Binary predictions saved to {binary_preds_path}")
            
        # Clean up memory
        del binary_model, binary_processor
        torch.cuda.empty_cache()
    else:
        print(f"Using existing binary predictions from {binary_preds_path}")
    
    # Prepare binary predictions for SOTA model evaluation
    if os.path.exists(binary_preds_path):
        # Convert binary results to the format expected by BinaryIntegratedEvalDataset
        with open(binary_preds_path, 'r') as f:
            binary_data = json.load(f)
        
        binary_predictions = {}
        if "predictions_and_labels" in binary_data:
            for i, item in enumerate(binary_data["predictions_and_labels"]):
                binary_predictions[str(i)] = item["prediction"]
        
        # Create a new JSON file with the expected format
        binary_preds_formatted_path = out_path / f"binary_preds_{dataset_name}_{args.split}_formatted.json"
        with open(binary_preds_formatted_path, 'w') as f:
            json.dump({
                "binary_predictions": binary_predictions,
                "audio_column": args.audio_column
            }, f, indent=2)
        
        print(f"Formatted binary predictions saved to {binary_preds_formatted_path}")
        binary_preds_path = binary_preds_formatted_path
    else:
        print(f"Error: Binary predictions file {binary_preds_path} not found")
        return
    
    # SOTA model evaluation phase
    print(f"Step 2: Evaluating SOTA model with binary integrated approach...")
    
    # Load SOTA model and processor
    with accelerator.local_main_process_first():
        model, processor = load_model_and_processor(
            args.model_name_or_path,
            use_flash_attention=args.use_flash_attention,
        )
    
    # Get rank and world size for distributed processing
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Create binary-integrated evaluation dataset
    eval_dataset = BinaryIntegratedEvalDataset(
        processor,
        dataset_name=args.dataset_name,
        split=args.split,
        binary_preds_path=binary_preds_path,
        text_column=args.text_column,
        audio_column=args.audio_column,
        max_samples=args.max_samples,
        rank=rank,
        world_size=world_size,
    )
    
    # Generate a descriptive filename for the final results
    results_filename = f"{model_name}_{dataset_name}_{args.split}_binary_integrated.json"
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
        print(f"SOTA evaluation completed with following results:")
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