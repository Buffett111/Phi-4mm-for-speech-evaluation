# Integrated Binary model on 2025/04/17
## Goal
Alright now I want to use current script and dataset to integrate two model to build a new model pipeline based on two base LLM models:
- base model1:binary model: `ntnu-smil/Phi-4-mm_Binary_QA_NI_0415_1964`
- base model2:classification model: `ntnu-smil/Phi-4-multimodal-instruct_QA_NoImage_0325_1964`
- dataset1: `ntnu-smil/LTTC-Train-1764-0520` `train` split
- dataset2: `ntnu-smil/Unseen_1964` `train` split
- form_id is setted base on current dataset used 1764 or 1964

## Approach:
 I want to train a model that
base on binary model to predict **classification output** with probabiliy
you can do this by

1. evaluate data using binary model,receive output in the format:0,0.80
meaning the data fail the test with model confidence(probability)0.8

2. just finetune on the classification model using current script`fine-tune-lttc.py`
but change input to add binary result into it

3. modify the prompt text in classification model to tell LLM that The ideal situation is that the classification model can utilize the info from binary output well, when the binary output is 0 the actual score should probabily between 1~3
and  when the binary output is 1 the actual score should probabily between 4~5

**Note that you only need to train the classification model**


## Rules
- if you have any question, ask me first before you act.