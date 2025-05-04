#!/usr/bin/env python3
import json
import os
import argparse
from pathlib import Path

def convert_binary_results(results_file, output_file, audio_column):
    """
    Convert binary results from the results file format to the predictions format
    needed by the classification model.
    
    Args:
        results_file: Path to the binary results file (e.g., binary_results_*.json)
        output_file: Path to save the converted binary predictions file
        audio_column: Name of the audio column to include in the metadata
    """
    print(f"Converting binary results from {results_file} to {output_file}")
    
    # Load the binary results file
    with open(results_file, 'r') as f:
        results_data = json.load(f)
    
    # Extract form_id from filename
    form_id = "1764"  # Default
    if "1964" in results_file:
        form_id = "1964"
    
    # Create the binary predictions dictionary
    binary_predictions = {}
    
    # Extract prediction for each sample
    for i, item in enumerate(results_data.get("predictions_and_labels", [])):
        sample_id = str(i)
        pred = item.get("prediction", "")
        
        # Clean prediction to ensure it's in format "0,0.80"
        pred = pred.strip()
        if "," in pred:
            binary_predictions[sample_id] = pred
        else:
            print(f"Warning: Invalid prediction format for sample {sample_id}: {pred}")
    
    # Create the output data structure
    output_data = {
        "binary_predictions": binary_predictions,
        "form_id": form_id,
        "audio_column": audio_column
    }
    
    # Save the output file
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Conversion complete. Saved {len(binary_predictions)} predictions to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Convert binary results to format needed by classification model")
    parser.add_argument("--results_file", type=str, required=True, help="Path to binary results file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output binary predictions file")
    parser.add_argument("--audio_column", type=str, default="wav_path", help="Name of audio column")
    
    args = parser.parse_args()
    
    convert_binary_results(args.results_file, args.output_file, args.audio_column)

if __name__ == "__main__":
    main() 