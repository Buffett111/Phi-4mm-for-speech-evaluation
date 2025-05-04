# Binary model on 2025/04/15
## Goal
Alright now I want to use current script and dataset to finetune a new model based on base model:
- base model: `microsoft/Phi-4-multimodal-instruct`
- dataset1: `ntnu-smil/LTTC-Train-1764-0520` `train` split
- dataset2: `ntnu-smil/Unseen_1964` `train` split
- form_id is setted base on current dataset used 1764 or 1964

## Approach:
just like the old approach, but now I want to train a model that
only predict **binary output** with probabiliy
you can do this by

1. load dataset
2. And convert origiral `grade` colume to binary score with probability
- the original `grade` colume contain a floating score between 1~5
- in LTTC standard 4~5 mean you pass the test,otherwise you fail
- you shoud convert it into binary with the fittest way you think 
ex: one way to do it is(just example,you can provide several methods for user)
{boolean,probability}
- 1 =>  {0,0.99} meaning that this audio fail the test most likely
- 2 =>  {0,0.80}
- 3 =>  {0,0.70}
- 3.5=> {0,0.45~0.55} this score appear when two judger make different judgements about the audio,one is 3 and one is 4
the main goal of this model is try to predict this score more accurately, whether it's "closer" to 3 or 4. maybe you can find a way to learn the best embedding that can prdict this score accurately
- 4 =>  {1,0.65~0.85}
- 5 =>  {1,0.85~0.99} 

- use current scirpt to finetune model base on this
    - epoach:5(default) 
    - learning rate: 5.0e-5(default) 

## pipeline
1. finetune on dataset1 ,learning embedding
2. upload model to hf, --hub_model_id "ntnu-smil/Phi-4-mm_Binary_QA_NI_${form_id}"
2. finetune on dataset2 ,maybe used learning embedding(optional to use new embedding)
3. upload model to hf, --hub_model_id "ntnu-smil/Phi-4-mm_Binary_QA_NI_${form_id}"

## Rules
- you should not modify any existing file without my permission
- instead, you should copy a new copy of the whole model, and create a folder for it
- work on new folder
- if you have any question, ask me first before you act.