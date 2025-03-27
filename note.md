# parameter change process
1. Adding QA slightly decrease the acc
2. when training with 2 epoach, loss is still high(0.273) , need to use epoach 3

## Todo List:
1. change model from 1 shot to multiple shots: give the image and prompt + question first, 
then send the audio, asking model to give a score
2. integrate image into the prompt
3. find approach to reduce mis judging between 3.5 ~ 4