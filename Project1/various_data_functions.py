
import torch
from torch import nn
from torch import optim
from torch import Tensor
from torch.nn import functional as F
import dlc_practical_prologue as prologue
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

def data( N, normalize, one_hot, loss, anew = True,
    train_input = None, train_target = None, train_classes = None, test_input = None, test_target = None, test_classes = None):
    
    def assertion_cycle():
        assert train_input is not None, 'train_input is None and data is not generated anew'
        assert train_target is not None, 'train_target is None and data is not generated anew'
        assert train_classes is not None, 'train_classes is None and data is not generated anew'
        assert test_input is not None, 'test_input is None and data is not generated anew'
        assert test_target is not None, 'test_target is None and data is not generated anew'
        assert test_classes is not None, 'test_classes is None and data is not generated anew'
    
    if anew: 

        #generate data
        transform = transforms.Compose([transforms.ToTensor()])
        train_data = MNIST(root = './data/mnist/', train=True, download=True, transform=transform)
        val_data = MNIST(root = './data/mnist/', train=False, download=True, transform=transform)

        train_input,train_target,train_classes,test_input,test_target,test_classes=prologue.generate_pair_sets(N)
        train_input = train_input.float()
        test_input = test_input.float()

    if one_hot: 
        assertion_cycle()
    
        #classes to onehot faster way to do it ? 
        temp1 = prologue.convert_to_one_hot_labels(train_classes[:,0], train_classes[:,0])
        temp2 = prologue.convert_to_one_hot_labels(train_classes[:,1], train_classes[:,1])        
        train_classes = torch.cat((temp1.unsqueeze(2), temp2.unsqueeze(2)), dim = 2)
        train_classes = torch.transpose(train_classes, 1, 2)

        temp1 = prologue.convert_to_one_hot_labels(test_classes[:,0], test_classes[:,0])
        temp2 = prologue.convert_to_one_hot_labels(test_classes[:,1], test_classes[:,1])        
        test_classes = torch.cat((temp1.unsqueeze(2), temp2.unsqueeze(2)), dim = 2)
        test_classes = torch.transpose(test_classes, 1, 2)

    if normalize:
        assertion_cycle()
        mu, std = train_input.mean(), train_input.std()
        train_input = train_input.sub_(mu).div_(std)
        test_input = test_input.sub_(mu).div_(std)
        
    if loss == nn.MSELoss: 
        assertion_cycle()
        train_target = train_target.float()
        test_target = test_target.float()

        train_classes = train_classes.float()
        test_classes = test_classes.float()
        

    if loss == nn.CrossEntropyLoss:
        assertion_cycle()
        train_target = train_target.long()
        test_target = test_target.long()

        train_classes = train_classes.long()
        test_classes = test_classes.long()
        
    return train_input, train_target, train_classes, test_input, test_target, test_classes
    
def validation_sub(test_input, test_target, test_classes):
    #splitting test set in 2 equal sets
    div_point = int(test_input.shape[0]/2)

    validation_input = test_input[:div_point]
    validation_target = test_target[:div_point]
    validation_classes = test_classes[:div_point]

    test_input = test_input[div_point:]
    test_target = test_target[div_point:]
    test_classes = test_classes[div_point:]

    return  validation_input, validation_target, validation_classes, test_input, test_target, test_classes

