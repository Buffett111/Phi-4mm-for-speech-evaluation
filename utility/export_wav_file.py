#!/usr/bin/env python3
"""
Script to export WAV files and their corresponding truth scores from a dataset.
"""

import os
import argparse
import soundfile as sf
from datasets import load_dataset

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Export audio files and scores from a dataset')
    parser.add_argument('--dataset', type=str, default='ntnu-smil/Unseen_1964', 
                        help='Dataset name (default: ntnu-smil/Unseen_1964)')
    parser.add_argument('--split', type=str, default='fulltest', 
                        help='Dataset split (default: fulltest)')
    parser.add_argument('--audio_column', type=str, default='wav_path', 
                        help='Name of the column containing audio data (default: wav_path)')
    parser.add_argument('--score_column', type=str, default='grade', 
                        help='Name of the column containing the score (default: grade)')
    parser.add_argument('--output_dir', type=str, default='exported_audio', 
                        help='Output directory (default: exported_audio)')
    
    # Create a mutually exclusive group for selection method
    selection_group = parser.add_mutually_exclusive_group(required=False)
    selection_group.add_argument('--indices', type=int, nargs='+', default=[42, 43], 
                        help='Indices of samples to export (default: 42 43)')
    selection_group.add_argument('--speaker_ids', type=int, nargs='+',
                        help='Speaker IDs to export all matching samples for')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the dataset
    print(f"Loading dataset {args.dataset}, split {args.split}...")
    ds = load_dataset(args.dataset, split=args.split)
    
    # Print the first sample to debug the structure
    print("\nExamining dataset structure:")
    sample = ds[0]
    print(f"Sample keys: {list(sample.keys())}")
    
    # Check if audio column exists
    if args.audio_column not in sample:
        print(f"Error: Column '{args.audio_column}' not found in dataset. Available columns: {list(sample.keys())}")
        return
    
    # Check if score column exists
    if args.score_column not in sample:
        print(f"Error: Column '{args.score_column}' not found in dataset. Available columns: {list(sample.keys())}")
        return
    
    # Check if speaker_id column exists when needed
    if args.speaker_ids is not None and 'speaker_id' not in sample:
        print(f"Error: You specified speaker_ids but 'speaker_id' column not found in dataset. Available columns: {list(sample.keys())}")
        return
    
    print(f"{args.audio_column} type: {type(sample[args.audio_column])}")
    print(f"{args.audio_column} value: {str(sample[args.audio_column])[:200]}...")  # Print first 200 chars
    
    # Determine which samples to export
    indices_to_export = []
    if args.speaker_ids is not None:
        # Select samples by speaker_id
        print(f"Selecting samples by speaker_id: {args.speaker_ids}")
        for idx, sample in enumerate(ds):
            if 'speaker_id' in sample and sample['speaker_id'] in args.speaker_ids:
                indices_to_export.append(idx)
        
        if not indices_to_export:
            print(f"No samples found with speaker_id in {args.speaker_ids}")
            return
            
        print(f"Found {len(indices_to_export)} samples with the specified speaker_id(s)")
    else:
        # Use direct indices
        indices_to_export = args.indices
    
    # Export the selected samples
    exported_count = 0
    for idx in indices_to_export:
        if idx >= len(ds):
            print(f"Warning: Index {idx} is out of range. Dataset has {len(ds)} samples. Skipping.")
            continue
            
        # Get the sample
        sample = ds[idx]
        
        # Extract audio data
        if isinstance(sample[args.audio_column], dict) and 'array' in sample[args.audio_column]:
            # If audio column is a dictionary with audio data
            audio_data = sample[args.audio_column]
            array = audio_data['array']
            sampling_rate = audio_data['sampling_rate']
            original_filename = audio_data.get('path', f"sample_{idx}.wav")
        else:
            # If audio column has different structure
            print(f"Sample {idx} has unexpected {args.audio_column} structure: {type(sample[args.audio_column])}")
            print(f"Looking for alternative audio sources...")
            
            # Try different approaches based on what's in the dataset
            if 'audio' in sample:
                # Some datasets use 'audio' key
                audio_data = sample['audio']
                if isinstance(audio_data, dict) and 'array' in audio_data:
                    array = audio_data['array']
                    sampling_rate = audio_data['sampling_rate']
                    original_filename = audio_data.get('path', f"sample_{idx}.wav")
                else:
                    print(f"Cannot find audio data for sample {idx}. Skipping.")
                    continue
            else:
                print(f"Cannot find audio data for sample {idx}. Skipping.")
                continue
        
        # Get truth score
        score = sample[args.score_column]
        
        # Include speaker_id in filename if available
        speaker_id_str = ""
        if 'speaker_id' in sample:
            speaker_id_str = f"_speaker_{sample['speaker_id']}"
        
        # Create a descriptive filename
        filename_base = os.path.basename(str(original_filename)).replace('.wav', '')
        output_filename = f"{idx}{speaker_id_str}_{filename_base}_score_{score}.wav"
        output_path = os.path.join(args.output_dir, output_filename)
        
        # Save the audio file
        print(f"Exporting sample {idx} to {output_path}")
        print(f"Truth score: {score}")
        sf.write(output_path, array, sampling_rate)
        exported_count += 1
        
        # Save metadata to a text file
        metadata_path = os.path.join(args.output_dir, f"{idx}_metadata.txt")
        with open(metadata_path, 'w') as f:
            f.write(f"Sample index: {idx}\n")
            if 'speaker_id' in sample:
                f.write(f"Speaker ID: {sample['speaker_id']}\n")
            f.write(f"Original filename: {original_filename}\n")
            f.write(f"Truth score: {score}\n")
            f.write(f"Sampling rate: {sampling_rate} Hz\n")
            
            # Also write additional metadata if available
            for key in ['prompt', 'asr_transcription', 'delivery', 'relevance', 'language']:
                if key in sample:
                    f.write(f"{key.capitalize()}: {sample[key]}\n")
    
    print(f"Exported {exported_count} audio files to {args.output_dir}")
    print("Done!")

if __name__ == "__main__":
    main() 