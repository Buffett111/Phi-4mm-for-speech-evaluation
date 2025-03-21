import argparse
import json
import os
from pathlib import Path

import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

import os
import io
from PIL import Image
import soundfile as sf
class MultipleTokenBatchStoppingCriteria(StoppingCriteria):
    """Stopping criteria capable of receiving multiple stop-tokens and handling batched inputs."""

    def __init__(self, stop_tokens: torch.LongTensor, batch_size: int = 1) -> None:
        """Initialize the multiple token batch stopping criteria.
        Args:
            stop_tokens: Stop-tokens.
            batch_size: Batch size.
        """

        self.stop_tokens = stop_tokens
        self.max_stop_tokens = stop_tokens.shape[-1]
        self.stop_tokens_idx = torch.zeros(batch_size, dtype=torch.long, device=stop_tokens.device)


    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Only gather the maximum number of inputs compatible with stop tokens
        # and checks whether generated inputs are equal to `stop_tokens`
        generated_inputs = torch.eq(input_ids[:, -self.max_stop_tokens :].unsqueeze(1), self.stop_tokens)
        equal_generated_inputs = torch.all(generated_inputs, dim=2)

        # Mark the position where a stop token has been produced for each input in the batch,
        # but only if the corresponding entry is not already set
        sequence_idx = torch.any(equal_generated_inputs, dim=1)
        sequence_set_mask = self.stop_tokens_idx == 0
        self.stop_tokens_idx[sequence_idx & sequence_set_mask] = input_ids.shape[-1]

        return torch.all(self.stop_tokens_idx)



# Define prompt structure
user_prompt = '<|user|>'
assistant_prompt = '<|assistant|>'
prompt_suffix = '<|end|>'
ANSWER_SUFFIX = "<|end|><|endoftext|>"
_IGNORE_INDEX = -100

InstructionText = f"""
{{
    "role": "system", 
    "content": "You are an English native speaker and a professional English speaking evaluator. Your task is to evaluate spoken English based on pronunciation, intonation, fluency, and the ability to communicate effectively. \
Your grading should be neutral and objective, do NOT overly praise strengths. \
Assign a score from 0 to 5 using the following criteria: "
}},
{{
    "role": "system",
    "content": "Score 5: Pronunciation and intonation are correct and natural. The speaker expresses themselves fluently, with no communication barriers. \
Score 4: Pronunciation and intonation are mostly correct and natural. There are some errors, but they do not hinder understanding. The speaker expresses themselves fairly fluently without communication barriers. \
Score 3: Pronunciation and intonation occasionally have errors but remain understandable. The speaking speed is slower, with occasional pauses, but communication is still achievable. \
Score 2: Pronunciation and intonation frequently have errors, which affect the listener's understanding. The speaking speed is slow, with frequent pauses that impact the delivery. \
Score 1: Pronunciation and intonation have numerous errors, with many inappropriate pauses, making it difficult for the listener to understand. \
Score 0: No response or the response is equivalent to no response. \
Please follow these guidelines when evaluating the score provided."
}},
{{
    "role": "system",
    "content": "I will provide 1. question 2. ASR text of students' answer and sub-scores from 3 parts: Language Use, Delivery, and Relevance predicted by other models, each of them ranging from 0 to 5. You should predict the final score based on the relevance of the question, sub-scores, and the text."
}},
{{
    "role": "developer", 
    "content": "Below are 3 sample judgments. You can refer to these samples when you predict scores later."
}},
{{
    "role": "developer",
    "content": "Question: Where was this picture probably taken? What makes you think so? What are some things to pay attention to when participating in activities in this kind of place? Which season do you think is suitable to visit this place? Why? If you still have time, please describe what people are wearing and other details of this picture."
}},
{{
    "role": "developer",
    "content": "1. all: 1.5 langUse: 2.5 delivery 3.5 content 2.5, text= This is the river because there is water a lot. Here be quiet river a lot to Summer because summer is very hot River is cool They are they they are very hot because they they were short and Yeah."
}},
{{
    "role": "developer",
    "content": "Question: Where was this picture probably taken? What makes you think so? What are some things to pay attention to when participating in activities in this kind of place? Which season do you think is suitable to visit this place? Why? If you still have time, please describe what people are wearing and other details of this picture."
}},
{{
    "role": "developer",
    "content": "2. all: 1.0 langUse: 0.5 delivery 0.5 relevance 0.5, text= I like the river because it's very warm. I don't know. I don't know. I don't know. I don't like to talk in my bed. I don't like... I don't like... I don't like... I don't like..."
}},
{{
    "role": "developer",
    "content": "Question: Where was this picture probably taken? What makes you think so? What are some things to pay attention to when participating in activities in this kind of place? Which season do you think is suitable to visit this place? Why? If you still have time, please describe what people are wearing and other details of this picture."
}},
{{
    "role": "developer",
    "content": "3. all: 5.0 langUse: 4.5 delivery 5.0 relevance 5.0, text= I think they might be at a national park because the national parks are usually in nature or outside. They might need to watch out for the river because the river might flow very fast and they might have to wear some hiking shoes because it is very dangerous if you wear slippers. I think it is better if you come here in summer because summer is usually very hot and rivers are usually very cool. You can enjoy it very much and you can cool off too. I see the people are wearing different clothes that have different varieties of colors. Someone is wearing striped t-shirt. Someone's wearing a white t-shirt. And I see two kids that are playing. And some people are standing in the river. And one person is sitting on a rock. And two are on the side."
}},
{{
    "role": "user", 
    "content": "You should only return the final score of the following input. The output should be a number between 0 and 5 with one decimal place. Without any other information or text, format: '3.0'"
}},
{{
    "role": "user", 
    "content": "Score according to the picture and the audio given to you"
}}
"""

class LTTC_Dataset(Dataset):
    def __init__(self, processor, dataset, split, instruction, training=True):
        self.data = dataset[split]
        self.processor = processor
        self.instruction = instruction
        self.training = training

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        user_message = {
            'role': 'user',
            'content': '<|audio_1|>\n' + self.instruction,
        }
        prompt = self.processor.tokenizer.apply_chat_template(
            [user_message], tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=prompt,
            audio, samplerate = sf.read(io.BytesIO(data["wav_path"]))
            audios=[(audio, samplerate)]
            return_tensors='pt',
        )

        answer = f"{data['grade']}{ANSWER_SUFFIX}"  # Use the "grade" column
        answer_ids = self.processor.tokenizer(answer, return_tensors='pt').input_ids
        if self.training:
            input_ids = torch.cat([inputs.input_ids, answer_ids], dim=1)
            labels = torch.full_like(input_ids, _IGNORE_INDEX)
            labels[:, -answer_ids.shape[1]:] = answer_ids
        else:
            input_ids = inputs.input_ids
            labels = answer_ids

        return {
            'input_ids': input_ids,
            'labels': labels,
            'input_audio_embeds': inputs.input_audio_embeds,
            'audio_embed_sizes': inputs.audio_embed_sizes,
        }

@torch.no_grad()
def evaluate(model, processor, eval_dataset, save_path=None, eval_batch_size=1):
    model.eval()
    all_generated_texts = []
    all_labels = []

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        collate_fn=lambda x: x[0],  # Simplified collate function
        shuffle=False,
    )

    for inputs in eval_dataloader:
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        generated_ids = model.generate(
            input_ids=inputs['input_ids'],
            max_new_tokens=64,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
        generated_text = [
            processor.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in generated_ids
        ]
        all_generated_texts.extend(generated_text)
        labels = [
            processor.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in inputs['labels']
        ]
        all_labels.extend(labels)

    if save_path:
        with open(save_path, 'w') as f:
            json.dump({'generated_texts': all_generated_texts, 'labels': all_labels}, f)

    return all_generated_texts, all_labels

def create_model(model_name_or_path, use_flash_attention=False):
    """Create and load the model with proper dtype and ensure it is moved to GPU."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if use_flash_attention else torch.float32,  # Specify dtype
        _attn_implementation='flash_attention_2' if use_flash_attention else 'sdpa',
        trust_remote_code=True,
    )
    model = model.to('cuda')  # Explicitly move the model to GPU
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='microsoft/Phi-4-multimodal-instruct', help='Model name or path to load from')
    parser.add_argument('--train_file', type=str, default='ntnu-smil/LTTC-Train1764-0520', help='Hugging Face dataset name for training')
    parser.add_argument('--dev_file', type=str, default='ntnu-smil/LTTC-dev1764-0520', help='Hugging Face dataset name for validation')
    parser.add_argument('--output_dir', type=str, default='./output/', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_train_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=4.0e-5, help='Learning rate')
    parser.add_argument('--use_features', action='store_true', help='Use additional features from the dataset')
    parser.add_argument('--use_flash_attention', action='store_true', help='Use Flash Attention')
    parser.add_argument('--instruction', type=str, default=InstructionText, help='Instruction for the task')
    args = parser.parse_args()

    accelerator = Accelerator()

    with accelerator.local_main_process_first():
        processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        model = create_model(
            args.model_name_or_path,
            use_flash_attention=args.use_flash_attention,
        )

    model.set_lora_adapter('speech')


    train_dataset = load_dataset(args.train_file)
    dev_dataset = load_dataset(args.dev_file)

    train_dataset = LTTC_Dataset(processor, train_dataset, split='train', instruction=args.instruction, training=True)
    eval_dataset = LTTC_Dataset(processor, dev_dataset, split='train', instruction=args.instruction, training=False)

    # Evaluate before fine-tuning
    evaluate(model, processor, eval_dataset, save_path=os.path.join(args.output_dir, 'eval_before.json'))

    if args.use_flash_attention:
        fp16 = False
        bf16 = True
    else:
        fp16 = True
        bf16 = False
    
    training_args = TrainingArguments(
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_dir='./logs',
        logging_steps=10,
        disable_tqdm=not accelerator.is_local_main_process,
        bf16=bf16,
        fp16=fp16,
    )
    
    model.set_lora_adapter('speech')
    
    num_gpus = accelerator.num_processes
    print(f'training on {num_gpus} GPUs')
    
    if args.use_flash_attention:
        fp16 = False
        bf16 = True
    else:
        fp16 = True
        bf16 = False
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    trainer.save_model()

    # Evaluate after fine-tuning
    evaluate(model, processor, eval_dataset, save_path=os.path.join(args.output_dir, 'eval_after.json'))

    if accelerator.is_main_process:
        processor.save_pretrained(args.output_dir)

if __name__ == '__main__':
    main()
