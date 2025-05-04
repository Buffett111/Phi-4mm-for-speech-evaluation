from huggingface_hub import login, HfApi, hf_hub_download
from datasets import load_dataset, Dataset
import os
import tempfile
import shutil
import argparse
import json
from pathlib import Path
import sys
import requests
import re

def revert_dataset_to_commit(dataset_name, commit_hash, token, force=False):
    """
    Revert a Hugging Face dataset to a specific commit.
    
    Args:
        dataset_name (str): Name of the dataset (e.g., "ntnu-smil/Unseen_1964")
        commit_hash (str): The commit hash to revert to
        token (str): Hugging Face token with write access
        force (bool): Skip confirmation if True
    """
    # Login to Hugging Face Hub
    login(token=token)
    api = HfApi()
    
    # Get current dataset info for confirmation
    try:
        current_info = api.repo_info(
            repo_id=dataset_name,
            repo_type="dataset"
        )
        current_commit = current_info.sha
        
        print(f"Current dataset commit: {current_commit}")
        print(f"Target dataset commit: {commit_hash}")
        
        if not force:
            confirmation = input(f"Are you sure you want to revert {dataset_name} to commit {commit_hash}? This will overwrite the current version. [y/N]: ")
            if confirmation.lower() not in ['y', 'yes']:
                print("Operation cancelled.")
                return
    except Exception as e:
        print(f"Warning: Could not retrieve current dataset info: {e}")
        if not force:
            confirmation = input(f"Could not verify current dataset state. Continue anyway? [y/N]: ")
            if confirmation.lower() not in ['y', 'yes']:
                print("Operation cancelled.")
                return
    
    # Create a temporary directory to save the dataset
    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"Working in temporary directory: {tmp_dir}")
        
        # Method 1: First, try to get all files directly from the commit
        try:
            print(f"Listing files in commit {commit_hash}...")
            files = api.list_repo_files(
                repo_id=dataset_name,
                repo_type="dataset",
                revision=commit_hash
            )
            
            print(f"Found {len(files)} files. Downloading...")
            for file_path in files:
                # Skip .git files
                if file_path.startswith(".git/"):
                    continue
                    
                try:
                    # Create the directory structure
                    file_dir = os.path.dirname(os.path.join(tmp_dir, file_path))
                    os.makedirs(file_dir, exist_ok=True)
                    
                    # Download the file
                    saved_path = hf_hub_download(
                        repo_id=dataset_name,
                        filename=file_path,
                        repo_type="dataset",
                        revision=commit_hash,
                        token=token,
                        local_dir=tmp_dir,
                        local_dir_use_symlinks=False
                    )
                    print(f"Downloaded {file_path} to {saved_path}")
                except Exception as e:
                    print(f"Error downloading {file_path}: {e}")
            
            print("Successfully downloaded all files from the target commit.")
            download_success = True
        except Exception as e:
            print(f"Error downloading files directly: {e}")
            download_success = False
        
        # Method 2: If direct download failed, try loading the dataset and saving it
        if not download_success:
            print("Falling back to loading dataset via datasets API...")
            try:
                dataset = load_dataset(dataset_name, revision=commit_hash)
                
                for split in dataset:
                    print(f"Processing split: {split}")
                    
                    # Create directory for this split
                    split_dir = os.path.join(tmp_dir, split)
                    os.makedirs(split_dir, exist_ok=True)
                    
                    # Save the dataset split
                    dataset[split].save_to_disk(split_dir)
                
                print("Successfully loaded and saved dataset via datasets API")
            except Exception as e:
                print(f"Error loading dataset: {e}")
                print("Both methods failed. Cannot proceed with reversion.")
                return
        
        # Final confirmation before push
        if not force:
            confirmation = input(f"Ready to push changes to {dataset_name}. This is your last chance to cancel. Continue? [y/N]: ")
            if confirmation.lower() not in ['y', 'yes']:
                print("Operation cancelled.")
                return
        
        # Push the dataset back to the Hub (with force=True to overwrite)
        print(f"Pushing dataset back to Hub as {dataset_name}...")
        try:
            api.upload_folder(
                folder_path=tmp_dir,
                repo_id=dataset_name,
                repo_type="dataset",
                commit_message=f"Reverting to commit {commit_hash}",
            )
            print(f"Successfully reverted {dataset_name} to commit {commit_hash}")
        except Exception as e:
            print(f"Error pushing dataset: {e}")
            print(f"Error details: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Revert a Hugging Face dataset to a specific commit.")
    parser.add_argument("--dataset_name", type=str, default="ntnu-smil/Unseen_1964", help="Name of the dataset")
    parser.add_argument("--commit_hash", type=str, default="d6dc7c69f435059c9718e92aa2b4f93211879084", help="Commit hash to revert to")
    parser.add_argument("--token", type=str, required=True, help="Hugging Face token with write access")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompts")
    
    args = parser.parse_args()
    
    revert_dataset_to_commit(args.dataset_name, args.commit_hash, args.token, args.force)
