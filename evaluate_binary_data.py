import argparse
import os
import json
import sys
from pathlib import Path
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

# Add LTTC-Binary to path for importing
sys.path.append("LTTC-Binary")
from common_binary import convert_to_binary_prob, load_model_and_processor, EvalDataset, evaluate

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate data using binary model and save results for classification model"
    )
    parser.add_argument(
        "--binary_model_path",
        type=str,
        default="ntnu-smil/Phi-4-mm_Binary_QA_NI_0415_1964",
        help="Path to binary model",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ntnu-smil/LTTC-Train-1764-0520",
        help="Dataset name to use for binary evaluation",
    )
    parser.add_argument(
        "--dataset2",
        type=str,
        default="ntnu-smil/Unseen_1964",
        help="Second dataset to evaluate",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use for evaluation",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./binary_eval_results/",
        help="Output directory for saving evaluation results",
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1, 
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="grade",
        help="Name of the column containing the answer score",
    )
    parser.add_argument(
        "--audio_column_1764",
        type=str,
        default="wav_path",
        help="Name of the audio column for 1764 dataset",
    )
    parser.add_argument(
        "--audio_column_1964",
        type=str,
        default="wave_file",
        help="Name of the audio column for 1964 dataset",
    )
    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
        help="Use Flash Attention for more efficient inference on compatible hardware",
    )
    args = parser.parse_args()

    # Create output directory
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load model and processor
    print(f"Loading binary model: {args.binary_model_path}")
    model, processor = load_model_and_processor(
        args.binary_model_path,
        use_flash_attention=args.use_flash_attention,
    )

    # Process both datasets
    datasets = [
        {"name": args.dataset_name, "form_id": "1764", "audio_column": args.audio_column_1764},
        {"name": args.dataset2, "form_id": "1964", "audio_column": args.audio_column_1964}
    ]

    results = {}
    all_accuracies = {}

    for dataset_info in datasets:
        dataset_name = dataset_info["name"]
        form_id = dataset_info["form_id"]
        audio_column = dataset_info["audio_column"]
        
        print(f"\n{'='*80}")
        print(f"Evaluating {dataset_name} with form_id {form_id}")
        print(f"Using audio column: {audio_column}")
        print(f"{'='*80}")
        
        # Load dataset
        eval_dataset = EvalDataset(
            processor,
            dataset_name=dataset_name,
            split=args.split,
            text_column=args.text_column,
            audio_column=audio_column,
            max_samples=None,
            rank=0,
            world_size=1,
            dataset_subset=None,
        )
        
        # Generate a descriptive filename for the results
        ds_name = dataset_name.split("/")[-1]
        results_filename = f"binary_results_{ds_name}_{args.split}.json"
        save_path = out_path / results_filename
        
        # Run evaluation
        eval_results = evaluate(
            model,
            processor,
            eval_dataset,
            save_path=save_path,
            disable_tqdm=False,
            eval_batch_size=args.batch_size,
            metric="binary",
        )
        
        # Display accuracy metrics
        print(f"\n{'='*30} ACCURACY METRICS {'='*30}")
        if isinstance(eval_results, dict):
            # Store accuracy metrics
            accuracy_metrics = {}
            
            # Extract and display all metrics
            for metric_name, metric_value in eval_results.items():
                if isinstance(metric_value, (int, float)):
                    print(f"{metric_name.upper()}: {metric_value:.4f}")
                    accuracy_metrics[metric_name] = metric_value
            
            # Specifically highlight binary accuracy
            binary_acc = eval_results.get("binary_accuracy")
            if binary_acc is not None:
                print(f"\n>> BINARY ACCURACY: {binary_acc:.4f} <<")
                accuracy_metrics["binary_accuracy"] = binary_acc
            
            prob_acc = eval_results.get("probability_accuracy")
            if prob_acc is not None:
                print(f">> PROBABILITY ACCURACY: {prob_acc:.4f} <<")
                accuracy_metrics["probability_accuracy"] = prob_acc
            
            combined_score = eval_results.get("combined_score")
            if combined_score is not None:
                print(f">> COMBINED SCORE: {combined_score:.4f} <<")
                accuracy_metrics["combined_score"] = combined_score
            
            all_accuracies[dataset_name] = accuracy_metrics
        else:
            print("No accuracy metrics available in evaluation results")
        
        print(f"{'='*80}\n")
        
        # Extract binary predictions
        binary_predictions = {}
        
        if os.path.exists(save_path):
            with open(save_path, 'r') as f:
                eval_data = json.load(f)
                
                # Store predictions in format needed for classification model
                if "predictions_and_labels" in eval_data:
                    # Extract from predictions_and_labels format
                    for i, item in enumerate(eval_data.get("predictions_and_labels", [])):
                        sample_id = str(i)
                        pred = item.get("prediction", "")
                        
                        # Clean prediction to ensure it's in format "0,0.80"
                        pred = pred.strip()
                        if "," in pred:
                            binary_predictions[sample_id] = pred
                        else:
                            print(f"Warning: Invalid prediction format for sample {sample_id}: {pred}")
                elif "samples" in eval_data:
                    # Fallback to samples format if available
                    for i, item in enumerate(eval_data.get("samples", [])):
                        sample_id = str(i)
                        pred = item.get("prediction", "")
                        
                        # Clean prediction to ensure it's in format "0,0.80"
                        pred = pred.strip()
                        if "," in pred:
                            binary_predictions[sample_id] = pred
                        else:
                            print(f"Warning: Invalid prediction format for sample {sample_id}: {pred}")
                else:
                    print(f"Warning: No predictions found in the evaluation results at {save_path}")
        
        # Save the extracted binary predictions
        binary_preds_filename = f"binary_preds_{ds_name}_{form_id}.json"
        binary_preds_path = out_path / binary_preds_filename
        
        with open(binary_preds_path, 'w') as f:
            json.dump({
                "binary_predictions": binary_predictions, 
                "form_id": form_id,
                "audio_column": audio_column  # Save audio column name for reference
            }, f, indent=2)
        
        print(f"Binary predictions saved to {binary_preds_path}")
        
        # Store in results dictionary
        results[dataset_name] = {
            "binary_preds_path": str(binary_preds_path),
            "form_id": form_id,
            "audio_column": audio_column,
            "accuracy_metrics": accuracy_metrics if 'accuracy_metrics' in locals() else {}
        }
    
    # Save metadata about all processed datasets
    metadata_path = out_path / "binary_evaluation_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Display summary of accuracies for all datasets
    print(f"\n{'='*30} ACCURACY SUMMARY {'='*30}")
    for dataset_name, metrics in all_accuracies.items():
        print(f"\nDataset: {dataset_name}")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name.upper()}: {metric_value:.4f}")
    print(f"{'='*80}")
    
    print(f"\nEvaluation complete. Metadata saved to {metadata_path}")

if __name__ == "__main__":
    main() 