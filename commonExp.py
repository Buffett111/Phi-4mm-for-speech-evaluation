import os
import json

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
InstructionText = f"""
"role": "system",
"content": "You are responsible for evaluating English pronunciation. The provided audio is a response to the question below. \
Your task is to objectively assign scores from 0 to 5 based on the following criteria: sentence-level accuracy, fluency, prosody, and completeness. \
Please also consider how well the audio answers the given question,that is, the relevance of the question and the answer. \
Finally, provide an overall score from 0 to 5. Your evaluation should remain neutral and fair. \
Assign scores according to the following criteria:"
,
"role": "system",
"content": "Score 5: Pronunciation and intonation are correct and natural. The speaker expresses themselves fluently, with no communication barriers. \
Score 4: Pronunciation and intonation are mostly correct and natural. There are some errors, but they do not hinder understanding. The speaker expresses themselves fairly fluently without communication barriers. \
Score 3: Pronunciation and intonation occasionally have errors but remain understandable. The speaking speed is slower, with occasional pauses, but communication is still achievable. \
Score 2: Pronunciation and intonation frequently have errors, which affect the listener's understanding. The speaking speed is slow, with frequent pauses that impact the delivery. \
Score 1: Pronunciation and intonation have numerous errors, with many inappropriate pauses, making it difficult for the listener to understand. \
Score 0: No response or the response is equivalent to no response. \
Please follow these guidelines when evaluating the score provided."
{EXAMPLES if EXA else ""}
"role": "user", 
"content": "You should only return the final score of the following input. The output should be a float number between 0 and 5 with one decimal place. Without any other information or text, format: 3.0" \
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
        
        # Use a hardcoded prompt instead of chat template since we're having issues with the processor
        prompt = "<|user|><|audio_1|>Score the test-taker's speech.<|end|><|assistant|>"
        
        # First, tokenize the text part only
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
        
        # Combine inputs
        
        # Create answer tokens
        answer = f"{data[self.text_column]}{ANSWER_SUFFIX}"
        answer_ids = self.processor.tokenizer(answer, return_tensors="pt").input_ids
        input_ids = inputs.input_ids
        labels = answer_ids

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
    
    processor = AutoProcessor.from_pretrained("microsoft/Phi-4-multimodal-instruct", trust_remote_code=True)

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
            processor.tokenizer.decode(
                _pred_ids[inputs["input_ids"].shape[1] : _stop_tokens_idx],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for _pred_ids, _stop_tokens_idx in zip(generated_ids, stop_tokens_idx)
        ]
        all_generated_texts.extend(generated_text)
        labels = [
            processor.tokenizer.decode(_label_ids[_label_ids != 0]).removesuffix(ANSWER_SUFFIX)
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
            
            # calculate absolute accuracy
            absolute_acc = 0
            for i in range(len(all_labels)):
                try:
                    tmp = all_generated_texts[i].split(" ")[0]
                    tmp = tmp.strip('"\',\\/')
                    gen_score = float(tmp)
                    label_score = float(all_labels[i].split(" ")[0])
                    
                    # Round scores based on the provided rule
                    gen_rounded = round_score(gen_score)
                    label_rounded = round_score(label_score)
                    
                    if gen_rounded == label_rounded:
                        absolute_acc += 1
                except Exception as e:
                    print(f"Error calculating absolute accuracy for sample {i}: {e}")
            
            absolute_acc = absolute_acc / len(all_labels)
            results["absolute_accuracy"] = absolute_acc
            print(f"Absolute Accuracy: {absolute_acc:.4f}")

        if metric.lower() == "binary" or metric.lower() == "both":
            correct = 0
            # calculate accuracy by comparing the generated text with the labels
            for i in range(len(all_labels)):
                try:
                    tmp = all_generated_texts[i].split(" ")[0]
                    tmp = tmp.strip('"\',\\/')
                    gen_score = float(tmp)
                    label_score = float(all_labels[i].split(" ")[0])
                    
                    # Round scores based on the provided rule
                    gen_rounded = round_score(gen_score)
                    label_rounded = round_score(label_score)
                    
                    # Convert to binary pass/fail
                    gen_binary = 1 if gen_rounded > 3 else 0
                    label_binary = 1 if label_rounded > 3 else 0
                    
                    if gen_binary == label_binary:
                        correct += 1
                except Exception as e:
                    print(f"Error calculating binary accuracy for sample {i}: {e}")
                    
            binary_acc = correct / len(all_labels)
            results["binary_accuracy"] = binary_acc
            print(f"Binary Accuracy: {binary_acc:.4f}")

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
                json.dump(save_dict, f, indent=4, ensure_ascii=False)

    return results


def scoring_audio(model, processor, audio_path, question):
    """Scoring audio using the model and processor."""

    # Load and preprocess audio
    try:
        audio, sr = audio_path["array"], audio_path["sampling_rate"]
    except Exception as e:
        raise ValueError(f"Error loading audio file {audio_path}: {e}")

    # Prepare input for the model
    prompt_text = InstructionText + "<|audio_1|> " + question
    
    # Process text and audio separately
    text_inputs = processor.tokenizer(
        prompt_text,
        return_tensors="pt",
    )
    
    audio_inputs = processor.feature_extractor(
        [audio],
        sampling_rate=sr,
        return_tensors="pt",
    )
    
    # Combine inputs into the format expected by the model
    inputs = {
        "input_ids": text_inputs.input_ids,
        "attention_mask": text_inputs.attention_mask,
        "input_audio_embeds": audio_inputs.input_features,
        "audio_embed_sizes": torch.tensor([audio_inputs.input_features.shape[1]]),
    }

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
    transcription = processor.tokenizer.decode(
        generated_ids[0, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    # Clean up the transcription (remove any potential end tokens that weren't caught by the stopping criteria)
    for token in stop_tokens:
        transcription = transcription.replace(token, "")

    return transcription.strip()