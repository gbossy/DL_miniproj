This is our EE-559 project 1 for the group of Axelle Piguet, Fabrizio Forte and Gaëtan Bossy.

This project assumes the data directory containing MNIST is already present in the directory as the download sometimes is problematic.

models.py contains the definition of 8 models, including the 5 mentioned in the report.

various_data functions.py contains training, plotting and data loading functions we used throughout the project.

test.py contains plots of the accuracy of our models. 
It first prints the number of parameter in each model, then starts training num_runs (2 as a default) instances of each model and plots the accuracy as a function of the number of epochs (20 as a default). 
It then plots the loss and auxiliary losses of a model (C4L2WS by default), which is the 2nd plot present in the report. 
Finally, it plots the evolution of the accuracy of the various settings of a single model (C4L2 and C4L2WS with auxiliary factors 0 and 0.1 by default, can be easily changed), which is the first plot present in the report.
num_runs can be changed to 20 to obtain plots similar to those in the report.

The checkpoints directory saves the state of the model to make plotting easier by saving runs. If one wants to run new instances of the model, delete every .pth file in checkpoints.