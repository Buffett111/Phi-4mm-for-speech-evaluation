import os
import json
import re

import torch
# import jiwer
from accelerate.utils import gather_object
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BatchFeature,
    StoppingCriteria,
    StoppingCriteriaList,
)
import librosa

ANSWER_SUFFIX = "<|end|><|endoftext|>"
_IGNORE_INDEX = -100
DEBUG = False
EXA = False
QA = True
def de_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)
       
EXAMPLES = """
"role": "system",
"content": "I will provide 1. question 2. ASR text of students' answer and sub-scores from 3 parts: Language Use, Delivery, and Relevance predicted by other models, each of them ranging from 0 to 5. You should predict the final score based on the relevance of the question, sub-scores, and the text."
,
"role": "developer", 
"content": "Below are 3 sample judgments. You can refer to these samples when you predict scores later."
,
"role": "developer",
"content": "Question: Where was this picture probably taken? What makes you think so? What are some things to pay attention to when participating in activities in this kind of place? Which season do you think is suitable to visit this place? Why? If you still have time, please describe what people are wearing and other details of this picture."
,
"role": "developer",
"content": "1. all: 1.5 langUse: 2.5 delivery 3.5 content 2.5, text= This is the river because there is water a lot. Here be quiet river a lot to Summer because summer is very hot River is cool They are they they are very hot because they they were short and Yeah."
,
"role": "developer",
"content": "Question: Where was this picture probably taken? What makes you think so? What are some things to pay attention to when participating in activities in this kind of place? Which season do you think is suitable to visit this place? Why? If you still have time, please describe what people are wearing and other details of this picture."
,
"role": "developer",
"content": "2. all: 1.0 langUse: 0.5 delivery 0.5 relevance 0.5, text= I like the river because it's very warm. I don't know. I don't know. I don't know. I don't like to talk in my bed. I don't like... I don't like... I don't like... I don't like..."
,
"role": "developer",
"content": "Question: Where was this picture probably taken? What makes you think so? What are some things to pay attention to when participating in activities in this kind of place? Which season do you think is suitable to visit this place? Why? If you still have time, please describe what people are wearing and other details of this picture."
,
"role": "developer",
"content": "3. all: 5.0 langUse: 4.5 delivery 5.0 relevance 5.0, text= I think they might be at a national park because the national parks are usually in nature or outside. They might need to watch out for the river because the river might flow very fast and they might have to wear some hiking shoes because it is very dangerous if you wear slippers. I think it is better if you come here in summer because summer is usually very hot and rivers are usually very cool. You can enjoy it very much and you can cool off too. I see the people are wearing different clothes that have different varieties of colors. Someone is wearing striped t-shirt. Someone's wearing a white t-shirt. And I see two kids that are playing. And some people are standing in the river. And one person is sitting on a rock. And two are on the side."
,
"""
# V1
# InstructionText = f"""
# "role": "system", 
# "content": "You are tasked with evaluating English pronunciation. For the given audio \
# You need to provide scores from 0 to 5, including evaluating sentence-level accuracy, fluency, prosody, and \
# completeness, and provide overall total score from 0 to 5. \
# Your grading should be neutral and objective. \
# Assign a score from 0 to 5 using the following criteria: "

# Function to convert scores to binary format with probability
def convert_to_binary_prob(score):
    """
    Convert original LTTC scores (1-5) to binary classification with probability
    
    Args:
        score (float): Original score between 1-5
        
    Returns:
        tuple: (binary_label, probability)
            - binary_label: 1 if pass (score >= 4), 0 if fail (score < 4)
            - probability: Confidence in the classification
    """
    # Convert score to float to handle string inputs
    score = float(score)
    
    # Determine binary label (pass/fail)
    binary_label = 1 if score >= 4.0 else 0
    
    # Calculate probability based on score
    if score <= 1.0:
        # Score 1: {0, 0.99}
        probability = 0.99
    elif score <= 2.0:
        # Score 2: {0, 0.80}
        probability = 0.80
    elif score <= 3.0:
        # Score 3: {0, 0.70}
        probability = 0.70
    elif score < 4.0:
        # Score 3.5: {0, 0.50}
        # Linear interpolation between 3 and 4
        probability = 0.70 - ((score - 3.0) / (4.0 - 3.0)) * (0.70 - 0.30)
    elif score < 5.0:
        # Score 4: {1, 0.75}
        # Linear interpolation between 4 and 5
        probability = 0.65 + ((score - 4.0) / (5.0 - 4.0)) * (0.20)
    else:
        # Score 5: {1, 0.95}
        probability = 0.95
    
    # If binary_label is 0 (fail), probability represents confidence in failing
    # If binary_label is 1 (pass), probability represents confidence in passing
    
    return (binary_label, probability)
       
# Instruction text for binary classification with probability
InstructionText = f"""
"role": "system",
"content": "You are responsible for evaluating English pronunciation. The provided audio is a response to the question below. \
Your task is to objectively assess if the pronunciation passes the minimum standard. \
Please consider sentence-level accuracy, fluency, prosody, completeness, and relevance to the question. \
Assign a binary score (0 for fail, 1 for pass) along with a probability indicating your confidence. \
For example, '1,0.85' means you believe it passes with 85% confidence, while '0,0.90' means you believe it fails with 90% confidence."
,
"role": "system",
"content": "Scoring criteria: \
- Score 0 (Fail): Pronunciation has significant errors, lacks fluency, or has inappropriate pauses that make it difficult to understand. \
- Score 1 (Pass): Pronunciation and intonation are generally correct, the speaker expresses themselves with reasonable fluency, \
and any errors do not significantly hinder understanding. \
Please consider both the pronunciation quality and the relevance to the question when making your assessment."
,
"role": "user", 
"content": "You should only return the binary score and probability for the following input. The output should be in format: 0,0.75 or 1,0.85 without any other information or text." \
,
"role": "user", 
"content": "Score according to the criteria above and the audio given to you"
"""




class MultipleTokenStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_tokens: torch.LongTensor) -> None:
        self.stop_tokens = stop_tokens
        self.max_stop_tokens = stop_tokens.shape[-1]

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for stop_token_list in self.stop_tokens:
            token_len = stop_token_list.shape[0]
            if token_len <= input_ids.shape[1] and torch.all(
                input_ids[0, -token_len:] == stop_token_list
            ):
                return True
        return False


class MultipleTokenBatchStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_tokens: torch.LongTensor, batch_size: int = 1) -> None:
        self.stop_tokens = stop_tokens
        self.max_stop_tokens = stop_tokens.shape[-1]
        self.stop_tokens_idx = torch.zeros(
            batch_size, dtype=torch.long, device=stop_tokens.device
        )

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        generated_inputs = torch.eq(
            input_ids[:, -self.max_stop_tokens :].unsqueeze(1), self.stop_tokens
        )
        equal_generated_inputs = torch.all(generated_inputs, dim=2)
        sequence_idx = torch.any(equal_generated_inputs, dim=1)
        sequence_set_mask = self.stop_tokens_idx == 0
        self.stop_tokens_idx[sequence_idx & sequence_set_mask] = input_ids.shape[-1]
        return torch.all(self.stop_tokens_idx)


class BaseDataset(Dataset):
    def __init__(
        self,
        processor,
        dataset_name,
        split,
        text_column="grade",
        audio_column="wav_path",
        question_column="prompt",
        max_samples=None,
        rank=0,
        world_size=1,
        dataset_subset=None,
    ):
        self.data = (
            load_dataset(dataset_name, dataset_subset, split=split)
            if dataset_subset
            else load_dataset(dataset_name, split=split)
        )
        if max_samples is not None:
            self.data = self.data.select(range(max_samples))
        
        if world_size is None:
            raise ValueError("world_size must be an integer greater than 0.")
        
        if world_size > 1:
            self.data = self.data.shard(num_shards=self.world_size, index=rank)
        self.processor = processor
        self.instruction = InstructionText
        self.text_column = text_column
        self.audio_column = audio_column
        self.question_column = question_column

    def __len__(self):
        return len(self.data)


class EvalDataset(BaseDataset):
    def __getitem__(self, idx):
        """
        Each example in the dataset is expected to have:
          - '{audio_column}': a dict with keys "array" and "sampling_rate"
          - '{text_column}': the answer score of the audio
        """
        data = self.data[idx]
        #print(data) column names
        # print(self.data.column_names)
        # Get the actual question from the data
        question = data.get(self.question_column, "") if QA else ""
        
        # Place question before audio token to be consistent with FinetuneDataset
        user_message = {
            "role": "user",
            "content": self.instruction + question + "<|audio_1|>",
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
        
        # Convert original score to binary probability format
        original_score = data[self.text_column]
        binary_label, probability = convert_to_binary_prob(original_score)
        
        # Format the answer as "binary_label,probability"
        binary_answer = f"{binary_label},{probability:.2f}"
        
        answer = f"{binary_answer}{ANSWER_SUFFIX}"
        answer_ids = self.processor.tokenizer(answer, return_tensors="pt").input_ids
        input_ids = inputs.input_ids
        labels = answer_ids

        return {
            "input_ids": input_ids,
            "labels": labels,
            "input_audio_embeds": inputs.input_audio_embeds,
            "audio_embed_sizes": inputs.audio_embed_sizes,
        }


class FinetuneDataset(BaseDataset):
    def __init__(
        self,
        processor,
        dataset_name,
        split,
        training,
        text_column="grade",
        audio_column="wav_path",
        question_column="prompt",
        max_samples=None,
        rank=0,
        world_size=1,
        dataset_subset=None,
    ):  
        # print(f"Loading dataset '{dataset_name}' with split '{split}'")
        # print(f"world_size: {world_size}")
        
        # Important! need to pass everything parent class needs, otherwise it won't initialize properly and send no warning
        super().__init__(
            processor,
            dataset_name,
            split,
            text_column,
            audio_column,
            question_column,
            max_samples,
            rank,
            world_size,
            dataset_subset,
        )
        self.training = training

        # Debugging: Print dataset size after loading
        print(f"Loaded dataset '{dataset_name}' with {len(self.data)} samples.")

        # Ensure required columns exist
        required_columns = [self.text_column, self.audio_column]
        for column in required_columns:
            if column not in self.data.column_names:
                raise ValueError(f"Missing required column '{column}' in the dataset.")
        
        # Check if question column exists (it's optional)
        if self.question_column not in self.data.column_names:
            print(f"Warning: Question column '{self.question_column}' not found in dataset. Will use empty string for questions.")

    def __getitem__(self, idx):
        """
        Each example in the dataset is expected to have:
          - '{audio_column}': a dict with keys "array" and "sampling_rate"
          - '{text_column}': the answer score of the audio
        """
        data = self.data[idx]

        # Debugging: Print the raw data sample
        de_print(f"Processing sample {idx}: {data}")

        # Get the actual question from the data
        question = data.get(self.question_column, "") if QA else ""
        
        # Change the order of the prompt: QA before audio to prevent the model 
        # from answering the question rather than giving the score
        user_message = {
            "role": "user",
            "content": self.instruction + question + "<|audio_1|>",
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
        
        # Convert original score to binary probability format
        original_score = data[self.text_column]
        binary_label, probability = convert_to_binary_prob(original_score)
        
        # Format the answer as "binary_label,probability"
        binary_answer = f"{binary_label},{probability:.2f}"
        
        answer = f"{binary_answer}{ANSWER_SUFFIX}"
        answer_ids = self.processor.tokenizer(answer, return_tensors="pt").input_ids

        if self.training:
            input_ids = torch.cat([inputs.input_ids, answer_ids], dim=1)
            labels = torch.full_like(input_ids, _IGNORE_INDEX)
            labels[:, -answer_ids.shape[1] :] = answer_ids
        else:
            input_ids = inputs.input_ids
            labels = answer_ids

        # Ensure labels are not empty
        if labels.size(1) == 0:
            labels = torch.full_like(input_ids, _IGNORE_INDEX)

        # Debugging: Print processed input and labels
        de_print(f"Processed input_ids: {input_ids.shape}, labels: {labels.shape}")

        return {
            "input_ids": input_ids,
            "labels": labels,
            "input_audio_embeds": inputs.input_audio_embeds,
            "audio_embed_sizes": inputs.audio_embed_sizes,
        }
        

# Utility functions for batching
def pad_sequence(sequences, padding_side="right", padding_value=0):
    """
    Pad a list of tensors to the same length.
    sequences: list of tensors with shape [seq_len, ...]
    """
    assert padding_side in ["right", "left"]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(seq.size(0) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == "right":
            output[i, :length] = seq
        else:
            output[i, -length:] = seq
    return output


def cat_with_pad(tensors, dim, padding_value=0):
    """
    Concatenate tensors along a specified dimension, padding to match dimensions.
    """
    ndim = tensors[0].dim()
    out_size = [max(t.shape[i] for t in tensors) for i in range(ndim)]
    out_size[dim] = sum(t.shape[dim] for t in tensors)
    output = tensors[0].new_full(out_size, padding_value)
    index = 0
    for t in tensors:
        slices = [slice(0, t.shape[d]) for d in range(ndim)]
        slices[dim] = slice(index, index + t.shape[dim])
        output[tuple(slices)] = t
        index += t.shape[dim]
    return output


def collate_fn(batch):
    input_ids_list = []
    labels_list = []
    input_audio_embeds_list = []
    audio_embed_sizes_list = []
    audio_attention_mask_list = []
    for sample in batch:
        input_ids_list.append(sample["input_ids"][0])
        labels_list.append(sample["labels"][0])
        input_audio_embeds_list.append(sample["input_audio_embeds"])
        audio_embed_sizes_list.append(sample["audio_embed_sizes"])
        audio_attention_mask_list.append(
            sample["input_audio_embeds"].new_full(
                (sample["input_audio_embeds"].size(1),), True, dtype=torch.bool
            )
        )

    input_ids = pad_sequence(input_ids_list, padding_side="left", padding_value=0)
    labels = pad_sequence(labels_list, padding_side="left", padding_value=0)
    audio_attention_mask = (
        pad_sequence(
            audio_attention_mask_list, padding_side="right", padding_value=False
        )
        if len(audio_attention_mask_list) > 1
        else None
    )
    attention_mask = (input_ids != 0).long()
    input_audio_embeds = cat_with_pad(input_audio_embeds_list, dim=0)
    audio_embed_sizes = torch.cat(audio_embed_sizes_list)

    return BatchFeature(
        {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "input_audio_embeds": input_audio_embeds,
            "audio_embed_sizes": audio_embed_sizes,
            "audio_attention_mask": audio_attention_mask,
            "input_mode": 2,  # speech mode
        }
    )


def load_model_and_processor(model_name_or_path, use_flash_attention=False):
    """Load the model and processor from the specified path or model name."""
    processor = AutoProcessor.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if use_flash_attention else torch.float32,
        _attn_implementation="flash_attention_2" if use_flash_attention else "sdpa",
        trust_remote_code=True,
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    return model, processor

def round_score(score):
    if score - int(score) <= 0.5:
        return int(score)
    else:
        return int(score) + 1

@torch.no_grad()
def evaluate(
    model,
    processor,
    eval_dataset,
    save_path=None,
    disable_tqdm=False,
    eval_batch_size=1,
    metric="both",
):
    """
    Evaluate the model on the dataset and calculate accuracy.
    
    Note: This function automatically cleans prediction outputs to extract just the
    binary classification and probability values in the format "X,Y" (e.g., "1,0.95"),
    removing any trailing newlines or descriptive text.
    """
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    model.eval()
    all_generated_texts = []
    all_labels = []

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=8,
        prefetch_factor=2,
        pin_memory=True,
    )
    stop_tokens = ["<|end|>", processor.tokenizer.eos_token]
    stop_tokens_ids = processor.tokenizer(
        stop_tokens, add_special_tokens=False, padding="longest", return_tensors="pt"
    )["input_ids"]
    stop_tokens_ids = stop_tokens_ids.to(f"cuda:{local_rank}")

    for inputs in tqdm(
        eval_dataloader, disable=(rank != 0) or disable_tqdm, desc="running eval"
    ):
        stopping_criteria = StoppingCriteriaList(
            [
                MultipleTokenBatchStoppingCriteria(
                    stop_tokens_ids, batch_size=inputs.input_ids.size(0)
                )
            ]
        )
        inputs = inputs.to(f"cuda:{local_rank}")
        generate_inputs = {
            k: v for k, v in inputs.items() 
            if k not in ['labels']  # Explicitly exclude labels
        }
        generated_ids = model.generate(
            **generate_inputs,
            eos_token_id=processor.tokenizer.eos_token_id,
            max_new_tokens=320,
            stopping_criteria=stopping_criteria,
        )
        stop_tokens_idx = stopping_criteria[0].stop_tokens_idx.reshape(
            inputs.input_ids.size(0), -1
        )[:, 0]
        stop_tokens_idx = torch.where(
            stop_tokens_idx > 0,
            stop_tokens_idx - stop_tokens_ids.shape[-1],
            generated_ids.shape[-1],
        )
        generated_text = [
            processor.decode(
                _pred_ids[inputs["input_ids"].shape[1] : _stop_tokens_idx],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for _pred_ids, _stop_tokens_idx in zip(generated_ids, stop_tokens_idx)
        ]
        all_generated_texts.extend(generated_text)
        labels = [
            processor.decode(_label_ids[_label_ids != 0]).removesuffix(ANSWER_SUFFIX)
            for _label_ids in inputs["labels"]
        ]
        all_labels.extend(labels)

    all_generated_texts = gather_object(all_generated_texts)
    all_labels = gather_object(all_labels)

    results = {}
    if len(all_labels) == 0:
        print("Error: No samples to evaluate.")
        return results

    if rank == 0:
        assert len(all_generated_texts) == len(all_labels)

        if metric.lower() == "cl" or metric.lower() == "both":
            correct = 0
            # calculate accuracy by comparing the generated text with the labels
            for i in range(len(all_labels)):
                de_print(f"Generated: {all_generated_texts[i]}")
                de_print(f"Label: {all_labels[i]}")
                # try to extract the score from the generated text
                try:
                    tmp = all_generated_texts[i].split(" ")[0]
                    tmp = tmp.strip('"\',\\/')
                    generated_score = float(tmp)
                    label_score = float(all_labels[i].split(" ")[0])
                    if abs(generated_score - label_score) <= 0.5:
                        correct += 1
                except Exception as e:
                    print(f"Error: {e}")
                    
            accuracy = correct / len(all_labels)
            results["accuracy"] = accuracy
            print(f"Accuracy: {accuracy:.4f}")
            
            # We don't need absolute accuracy for binary classification
            # This is only relevant for traditional classification with scores 1-5
            # So removing this calculation

        if metric.lower() == "binary" or metric.lower() == "both":
            binary_correct = 0
            binary_preds = []
            binary_refs = []
            prob_preds = []
            prob_refs = []
            
            # Calculate accuracy by comparing the generated text with the labels
            for i in range(len(all_labels)):
                try:
                    # Parse the prediction - format is "binary_label,probability"
                    pred = all_generated_texts[i].strip()
                    ref = all_labels[i].strip()
                    
                    # Handle the binary format with probability (binary_label,probability)
                    if "," in pred:
                        # Clean the prediction - extract just the "X,Y" part using regex
                        match = re.match(r"(\d+,\d+\.\d+)", pred)
                        if match:
                            pred = match.group(1)
                        
                        # Parse the cleaned prediction
                        pred_parts = pred.split(",", 1)
                        binary_pred = int(pred_parts[0].strip())
                        prob_pred = float(pred_parts[1].strip())
                    else:
                        # Try to handle as legacy format (float score)
                        try:
                            tmp = pred.split(" ")[0]
                            tmp = tmp.strip('"\',\\/')
                            score = float(tmp)
                            binary_pred = 1 if score >= 4.0 else 0
                            prob_pred = 0.75 if binary_pred == 1 else 0.25
                            print(f"Warning: Prediction '{pred}' not in binary,probability format. Converted to {binary_pred},{prob_pred}")
                        except:
                            print(f"Error parsing prediction: '{pred}'")
                            continue
                    
                    # Parse the reference in the same way
                    if "," in ref:
                        # New format: "1,0.85" or "0,0.75"
                        ref_parts = ref.split(",", 1)
                        binary_ref = int(ref_parts[0].strip())
                        prob_ref = float(ref_parts[1].strip())
                    else:
                        # Try to handle as legacy format (float score)
                        try:
                            tmp = ref.split(" ")[0]
                            tmp = tmp.strip('"\',\\/')
                            score = float(tmp)
                            binary_ref = 1 if score >= 4.0 else 0
                            prob_ref = 0.75 if binary_ref == 1 else 0.25
                            print(f"Warning: Reference '{ref}' not in binary,probability format. Converted to {binary_ref},{prob_ref}")
                        except:
                            print(f"Error parsing reference: '{ref}'")
                            continue
                    
                    # Store for metrics calculation
                    binary_preds.append(binary_pred)
                    binary_refs.append(binary_ref)
                    prob_preds.append(prob_pred)
                    prob_refs.append(prob_ref)
                    
                    # Binary accuracy
                    if binary_pred == binary_ref:
                        binary_correct += 1
                        
                except Exception as e:
                    print(f"Error calculating binary accuracy for sample {i}: {e}")
            
            # Binary Accuracy
            if binary_preds:  # Check if we have valid predictions
                binary_acc = binary_correct / len(binary_preds)
                results["binary_accuracy"] = binary_acc
                print(f"Binary Accuracy: {binary_acc:.4f}")
                
                # Probability MSE (Mean Squared Error)
                if prob_preds:
                    prob_mse = sum((p - r) ** 2 for p, r in zip(prob_preds, prob_refs)) / len(prob_preds)
                    results["probability_mse"] = prob_mse
                    print(f"Probability MSE: {prob_mse:.4f}")
                    
                    # Probability accuracy within tolerance
                    tolerance = 0.1
                    prob_correct = sum(1 for p, r in zip(prob_preds, prob_refs) if abs(p - r) <= tolerance)
                    prob_acc = prob_correct / len(prob_preds)
                    results["probability_accuracy"] = prob_acc
                    print(f"Probability Accuracy (±{tolerance}): {prob_acc:.4f}")
                    
                    # Combined score (weighted average of binary_accuracy and prob_accuracy)
                    combined_score = 0.7 * binary_acc + 0.3 * prob_acc
                    results["combined_score"] = combined_score
                    print(f"Combined Score: {combined_score:.4f}")
            else:
                print("No valid binary predictions found.")
                results["binary_accuracy"] = 0

        results["num_samples"] = len(all_labels)
        print(f"Number of samples: {len(all_labels)}")

        if save_path:
            with open(save_path, "w") as f:
                save_dict = {
                    "predictions_and_labels": [
                        {"prediction": pred, "label": label}
                        for pred, label in zip(all_generated_texts, all_labels)
                    ],
                    **results,
                }
                if "accuracy" in results:
                    save_dict["accuracy"] = results["accuracy"]
                if "binary_accuracy" in results:
                    save_dict["binary_accuracy"] = results["binary_accuracy"]
                if "probability_mse" in results:
                    save_dict["probability_mse"] = results["probability_mse"]
                if "probability_accuracy" in results:
                    save_dict["probability_accuracy"] = results["probability_accuracy"]
                if "combined_score" in results:
                    save_dict["combined_score"] = results["combined_score"]
                json.dump(save_dict, f, indent=4, ensure_ascii=False)

    return results


def scoring_audio(model, processor, audio_path, question):
    """Scoring audio using the model and processor."""

    # Load and preprocess audio
    try:
        audio, sr = audio_path["array"], audio_path["sampling_rate"]
    except Exception as e:
        # Try to load from file path
        try:
            import librosa
            audio, sr = librosa.load(audio_path, sr=16000)
        except:
            raise ValueError(f"Error loading audio file {audio_path}: {e}")

    # Prepare input for the model
    user_message = {
        "role": "user",
        "content": InstructionText + (question if question else "") + "<|audio_1|>",
    }
    prompt = processor.tokenizer.apply_chat_template(
        [user_message], tokenize=False, add_generation_prompt=True
    )

    # Process input
    inputs = processor(
        text=prompt,
        audios=[(audio, sr)],
        return_tensors="pt",
    )

    # Move inputs to the same device as the model
    inputs = {
        k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()
    }

    # Set up stopping criteria
    stop_tokens = ["<|end|>", processor.tokenizer.eos_token]
    stop_tokens_ids = processor.tokenizer(
        stop_tokens, add_special_tokens=False, padding="longest", return_tensors="pt"
    )["input_ids"].to(model.device)
    stopping_criteria = StoppingCriteriaList(
        [MultipleTokenStoppingCriteria(stop_tokens_ids)]
    )

    # Generate transcription
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            eos_token_id=processor.tokenizer.eos_token_id,
            max_new_tokens=320,
            stopping_criteria=stopping_criteria,
            do_sample=False,  # Deterministic generation
        )

    # Decode the generated text
    result = processor.decode(
        generated_ids[0, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    # Clean up the result (remove any potential end tokens that weren't caught by the stopping criteria)
    for token in stop_tokens:
        result = result.replace(token, "")
    
    # Parse the binary output
    result = result.strip()
    try:
        if "," in result:
            # Clean the result - extract just the "X,Y" part using regex
            match = re.match(r"(\d+,\d+\.\d+)", result)
            if match:
                result = match.group(1)
                
            # Format is "binary_label,probability"
            binary_label, probability = result.split(",", 1)
            binary_label = int(binary_label.strip())
            probability = float(probability.strip())
            
            # Format for return
            formatted_result = f"{binary_label},{probability:.2f}"
        else:
            # Try to interpret as a legacy score
            score = float(result)
            binary_label = 1 if score >= 4.0 else 0
            probability = 0.75 if binary_label == 1 else 0.25
            formatted_result = f"{binary_label},{probability:.2f}"
    except:
        # If parsing fails, return the raw result
        formatted_result = result
        
    return formatted_result