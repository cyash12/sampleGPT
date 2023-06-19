# sampleGPT
An easy-to-understand code for a scaled-down GPT model in less than 200 lines. Uses a character level encoding for simplicity, GPT uses BPE encoding.<br>
Save the training data in a text file in the same directory as the code. While building I used one volume of an encyclopedia for training.<br>
To train the model, give the 'train' argument while executing the code.<br>
python3 code.py 'train'<br>
To generate data, give the 'generate' argument followed by a string to condition your generation on.<br>
python3 code.py 'generate' 'conditioning text'<br>
<br>
Change the configs as needed to scale the model up or down depending on the resources at hand.
