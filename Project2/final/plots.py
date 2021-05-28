from torch import empty, tensor, set_grad_enabled
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from framework import *
set_grad_enabled(False)

####################################################################
#the following parameters are needed only for the decision boudary plots
# define bounds of the domain
min1, max1 = 0, 1.01
min2, max2 = 0, 1.01
# define the x and y scale
x1grid = np.arange(min1, max1, 0.001)
x2grid = np.arange(min2, max2, 0.001)
# create all of the lines and rows of the grid
xx, yy = np.meshgrid(x1grid, x2grid)
# flatten each grid to a vector
r1, r2 = xx.flatten(), yy.flatten()
r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
# horizontal stack vectors to create x1,x2 input for the model
grid = np.hstack((r1,r2))
grid = tensor(grid).float()
###########################################################################

#function that plots the decision boudary for the first two models
def countour(net,plotname=None):
    # make predictions for the grid
    yhat = np.floor(net.forward(grid).numpy())
    # reshape the predictions back into a grid
    zz = yhat.reshape(xx.shape)
    # plot the grid of x, y and z values as a surface
    plt.contourf(xx, yy, zz, cmap='Paired')
    #plot the data
    plt.plot(test_data[:,0][test_target[:,0]==1], test_data[:,1][test_target[:,0]==1],'bo')
    plt.plot(test_data[:,0][test_target[:,0]!=1], test_data[:,1][test_target[:,0]!=1],'go')
    if plotname is not None:
        plt.savefig(plotname,bbox_inches='tight')
    plt.show()
    return yhat

#function that plots the decision boudary for the third model (the one with softmax
def countour_(net,plotname=None):
    # make predictions for the grid
    yhat = net.forward(grid).argmax(1).int().numpy()
    # reshape the predictions back into a grid
    zz = yhat.reshape(xx.shape)
    # plot the grid of x, y and z values as a surface
    plt.contourf(xx, yy, zz, cmap='Paired')
    #plot the data
    plt.plot(test_data[:,0][test_target[:,0]==1], test_data[:,1][test_target[:,0]==1],'bo')
    plt.plot(test_data[:,0][test_target[:,0]!=1], test_data[:,1][test_target[:,0]!=1],'go')
    if plotname is not None:
        plt.savefig(plotname,bbox_inches='tight')
    plt.show()
    return yhat

def training(train_data, train_target, net, optimizer, epochs, batch_size, permute = True):
    
    #total numbner of data points 
    N=train_data.shape[0]
    
    #initialize the list that stores the log of the losses
    losses_log = []
    
    #train the network
    for e in range(epochs):
        
        #set the loss to zero at the beginning of an epoch 
        acc_loss=0
        
        #if requested we shuffle the dataset before creating the batches
        if permute:
            indices = list(range(len(train_data)))
            random.shuffle(indices)
            train_data_shuffled = train_data[indices]
            train_target_shuffled = train_target[indices]
            
        else:
            train_data_shuffled = train_data
            train_target_shuffled = train_target
        
        #update the gradients for each batch
        for b in range(0, N, batch_size):
            
            #compute the forward pass of both the net and the loss on the batch
            predictions = net.forward(train_data_shuffled[b:b+batch_size])
            l= loss.forward(predictions, train_target_shuffled[b:b+batch_size])
            
            #increment the loss
            acc_loss += l
            
            #set the gradient of the parameters to zero 
            optimizer.zero_grad()
            
            #call the backward pass
            net.backward(loss.backward())
            
            #take an optimizer step
            optimizer.step()
        
        #log the loss on screen and save it in a tensor
        print(e, '   MSE loss = ' , acc_loss.item()) 
        losses_log += [acc_loss.item()]
    
    #return the log of the losses
    return tensor(losses_log)

#counting corrects on test
def test_tanh(net, test_data, test_target, verbose = False):
    correct_count=0
    for i in range(test_data.shape[0]):
            x=test_data[i]
            y=test_target[i]

            y_pred = net.forward(x.unsqueeze(0))

            correct = y_pred.sign()*y>0
            if verbose:
                print('prediction output \t', round(y_pred.item(), 2), '\t\treal output\t', round(y.item(), 2),'\t\tcorrect prediction?', correct.item())
            if correct : correct_count += 1

    print('Correct predictions after '+str(epochs)+' training steps: '+str(correct_count/test_data.shape[0]*100)+' %')

#counting corrects on test
def test_softmax(net, test_data, test_target, verbose = False):
    correct_count=0
    for i in range(test_data.shape[0]):
            x=test_data[i]
            y=test_target[i]

            y_pred = net.forward(x.unsqueeze(0))

            correct = y_pred.argmax()==y.argmax()
            if verbose:
                print('prediction output \t', y_pred, y_pred.argmax(), '\t\treal output\t', y, y.argmax(),'\t\tcorrect prediction?', correct)
            if correct : correct_count += 1

    print('Correct predictions after '+str(epochs)+' training steps: '+str(correct_count/test_data.shape[0]*100)+' %')

#function to generate the data
def generate_disc_set(nb):
    input_ = empty(nb, 2).uniform_(0, 1)
    target = (input_-0.5).pow(2).sum(1).sub(1 / (math.pi*2)).sign().add(1).div(2).long()
    target= 1 -target
    target = 2*target -1 
    return input_, target.view(-1, 1)

#function to convert to hot_labels
def convert_to_one_hot_labels(input, target):
    tmp = input.new_zeros(target.size(0), target.max() + 1)
    tmp.scatter_(1, target.view(-1, 1), 1.0)
    return tmp

#data generation
N_data_point = 10**3
train_data, train_target= generate_disc_set(N_data_point)
test_data,  test_target = generate_disc_set(N_data_point)

#Initialize a first test model using linear layers and tanh activation functions
linear1 = Linear(2, 25, True)
linear2 = Linear(25,25,True)
linear3 = Linear(25,25,True)
linear4 = Linear(25,1,True)
sigma1 = Tanh()
sigma2 = Tanh()
sigma3 = Tanh()
sigma4 = Tanh()

loss = MSE()

net = Sequential([
    linear1, 
    sigma1 ,
    linear2,
    sigma2 ,
    linear3,
    sigma3 ,
    linear4,
    sigma4
])

#training the net
print('First test: from 2 input units to 25 to 25 to 25 then to 1, using Linear layers and Tanh activation function')
optimizer = SGD(lr = 0.01,momentum=True, parameters = net.get_parameters())
epochs=100
batch_size = 10
losses_log = training(train_data, train_target, net, optimizer, epochs, batch_size)

#testing the net
test_tanh(net, test_data, test_target, verbose = False)

#plotting the decision boudary 
yhat = countour(net,'Proj2Test1boundary.png')

#plotting the losses
plt.plot(tensor(np.arange(losses_log.shape[0])), losses_log)
plt.xlabel('epochs')
plt.ylabel('accumulated loss')
plt.title('Full Tanh')
plt.savefig('Proj2Test1loss.png',bbox_inches='tight')
plt.show()

###########################################################################################
#Initialize a second test model using linear layers and relu activation functions except for the last layer that's Tanh activation function
linear1 = Linear(2, 25, True)
linear2 = Linear(25,25,True)
linear3 = Linear(25,25,True)
linear4 = Linear(25,1,True)
sigma1 = ReLu()
sigma2 = ReLu()
sigma3 = ReLu()
sigma4 = Tanh()
loss = MSE()

net = Sequential([
    linear1, 
    sigma1 ,
    linear2,
    sigma2 ,
    linear3,
    sigma3 ,
    linear4,
    sigma4
])

#trining the net
print('Second test: from 2 input units to 25 to 25 to 25 then to 1, using Linear layers and Relu activation function except for the last activation where we use Tanh')
optimizer = SGD(lr = 0.01,momentum=True, parameters = net.get_parameters())
epochs=50
batch_size = 10
losses_log = training(train_data, train_target, net, optimizer, epochs, batch_size)

#testing the net
test_tanh(net, test_data, test_target, verbose = False)

#plotting the decision boudary
yhat = countour(net,'Proj2Test2boundary.png')

#plotting the losses
plt.plot(tensor(np.arange(losses_log.shape[0])), losses_log)
plt.xlabel('epochs')
plt.ylabel('accumulated loss')
plt.title('ReLu + Tanh at the end')
plt.savefig('Proj2Test2loss.png',bbox_inches='tight')
plt.show()


#######################################################################################
#Initialize a third test model using linear layers and relu activation functions except for the last layer that's softmax activation function and thus has 2 outputs
linear1 = Linear(2, 25, True)
linear2 = Linear(25,25,True)
linear3 = Linear(25,25,True)
linear4 = Linear(25,2,True)
sigma1 = ReLu()
sigma2 = ReLu()
sigma3 = ReLu()
sigma4 = Softmax()
loss = MSE()

net = Sequential([
    linear1, 
    sigma1 ,
    linear2,
    sigma2 ,
    linear3,
    sigma3 ,
    linear4,
    sigma4
])

#Put the data in the correct format (one hot label needed here
train_target_hot = (train_target + 1).div(2.).long()
test_target_hot = (test_target + 1).div(2.).long()
train_target_hot = convert_to_one_hot_labels( train_data, train_target_hot)
test_target_hot = convert_to_one_hot_labels( test_target, test_target_hot)

#training the net
print('Third test: from 2 input units to 25 to 25 to 25 then to 2, using Linear layers and Relu activation function except for the last activation where we use Softmax')
optimizer = SGD(lr = 10,momentum=True, parameters = net.get_parameters())
epochs=50
batch_size = 10
losses_log = training(train_data, train_target_hot, net, optimizer, epochs, batch_size)

#testing the net
test_softmax(net, test_data, test_target_hot, verbose = False)

#plotting the decision boudary 
countour_(net,'Proj2Test3boundary.png')

#plotting the losses
plt.plot(tensor(np.arange(losses_log.shape[0])), losses_log)
plt.xlabel('epochs')
plt.ylabel('accumulated loss')
plt.title('ReLu + Softmax at the end')
plt.savefig('Proj2Test3loss.png',bbox_inches='tight')
plt.show()








