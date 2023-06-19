# sampleGPT
An easy to understand code for a scaled down GPT model in less than 200 lines. Uses a character level encoding for simplicity, GPT uses BPE encoding.
Save the training data in a text file in the same directory as the code. While building I used one volume of an encyclopedia for training.
To train the model, give the 'train' argument while executing the code.
python3 code.py 'train'
To generate data, give the 'generate' argument followed by a string to condition your generation on.
python3 code.py 'generate' 'conditioning text'

Change the configs as needed to scale the model up or down depending upon the resoources at hand.
