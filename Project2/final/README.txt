This is our EE-559 project 2 for the group of Axelle Piguet, Fabrizio Forte and Gaëtan Bossy.

framework.py contains the definitions of the classes Module (Linear, Tanh, ReLu, Sigmoid, Softmax,), Losses (MSE), Sequential and Optimizer (SGD).

test.py includes functions for data generation, network training, and decision boundary plots.
It also contains test functions for Tanh and Softmax.

It first generates 2 datasets with 1000 points each. One is for training, one for testing.
3 models are created with three hidden layers each. (more details in report)

Training : 
The loss used is MSE.
The optimizer is SGD (settable parameters). It was used with :
	 - learning rate of 0.01 (model 1 and 2) or 10 (model 3)
	 - momentum = True 
	 - 100 (model 1) or 50 (model 2 and 3) epochs
	 - batch size of 10
