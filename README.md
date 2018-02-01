# hello-world-in-tensorflow-binary-to-decimal
Hands-on, tensorflow hello world program with python.
aim for this repo was not to implement the Recurrent neural network. This is about hello word program of tensorflow, and conversion of decimal to binary, so we will prepare dataset of binary and coresponding decimal value. Therefore, it becomes supervise learning. Then we will train dataset using tensorflow library and save our model after training, further we will pull up saved model and test it using binary data.


## Installation

 
 -> NumPy -3.13.1
 
  '''
  sudo pip3 install numpy
  '''
 
 
 -> tensorflow -1.2.1
 
 for installing tensorflow :-https://www.tensorflow.org/install/

## Usage
 clone repo:- git clone https://github.com/parthvadhadiya/hello-world-in-tensorflow-binary-to-decimal/ 
 
 run :- python3 training_RNN.py 
 
   it will generate dataset using create_dataset.py and train RNN with that dataset.
 
 run :- python3 usemodel.py 
 
   for getting predicated decimal value of givan binary
