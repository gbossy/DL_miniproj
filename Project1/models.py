from torch import nn
from torch import optim
from torch import Tensor
from torch.nn import functional as F

#########################################################################################
## Models with auxiliary losses 

#The models names are created as follows:
# 'C'+ #of convolutions + 'L' + number of linear layers
# The auxiliary 'branch' of the network has been put between the convolutions 
# and the final linear layers and uses a linear layer per picture to predict the picture label

#the parameter h is computed like this: 
# (shape of output of final convolutional layer)^2 * number of final channels/2

################

class C1L2(nn.Module):
    def __init__(self):
        super().__init__()
                
        #convolutions.
        self.conv1 = nn.Conv2d(2, 144, kernel_size=3,groups=2)
        
        #define the half lenght of the output (linearized) after the last convolution
        self.h = int(4**2*144/2)
        
        #linear layers for the auxiliary losses
        self.aux_linear1 = nn.Linear(self.h, 10)
        self.aux_linear2 = nn.Linear(self.h, 10)
        
        #linear layers for the final output
        self.fc1 = nn.Linear(2*self.h, 20)
        self.fc2 = nn.Linear(20, 2)
        

    def forward(self, x):
        
        #computing the convolution
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=3, stride=3))
        x = x.view(-1, 2*self.h)
        
        #computing the linear+softmax to make the auxiliary output
        #deviding the tensor 
        a1 = x.narrow(1, 0,      self.h)
        a2 = x.narrow(1, self.h, self.h)
                
        #computing the auxiliary output
        a1 = F.softmax(self.aux_linear1(a1), dim = 1).unsqueeze(1)
        a2 = F.softmax(self.aux_linear2(a2), dim = 1).unsqueeze(1)
        #merging the vector to return the final auxiliary output
        aux_output = torch.cat((a1, a2), 1)
        
        #computing the layers for the final output
        x = F.relu(self.fc1(x))
        output = F.softmax(self.fc2(x), dim=1)
        return output, aux_output

    
#############################################


class C1L3(nn.Module):
    def __init__(self):
        super().__init__()
        
        #convolutions
        self.conv1 = nn.Conv2d(2, 30, kernel_size=3,groups=2)
        
        #define the half lenght of the output (linearized) after the last convolution
        self.h = int(4**2*30/2)
        
        #linear layers for the auxiliary losses
        self.aux_linear1 = nn.Linear(self.h, 10)
        self.aux_linear2 = nn.Linear(self.h, 10)
        
        #linear layers for the final output
        self.fc1 = nn.Linear(2*self.h, 128)
        self.fc2 = nn.Linear(128, 20)
        self.fc3 = nn.Linear(20, 2)

    def forward(self, x):
        
        #computing the convolution
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=3, stride=3))
        x = x.view(-1, 2*self.h)
        
        #computing the linear+softmax to make the auxiliary output
        #deviding the tensor 
        a1 = x.narrow(1, 0,      self.h)
        a2 = x.narrow(1, self.h, self.h)
                
        #computing the output
        a1 = F.softmax(self.aux_linear1(a1), dim = 1).unsqueeze(1)
        a2 = F.softmax(self.aux_linear2(a2), dim = 1).unsqueeze(1)
        #merging the vector to return the final auxiliary output
        aux_output = torch.cat((a1, a2), 1)
        
        #computing the layers for the final output
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = F.softmax(self.fc3(x),dim=1)
        return output, aux_output
    
################################################

class C1L5(nn.Module):
    def __init__(self):
        super().__init__()
        
        #convolutions.
        self.conv1 = nn.Conv2d(2, 20, kernel_size=3,groups=2)
        
        #define the half lenght of the output (linearized) after the last convolution
        self.h = int(4**2*20/2)
        
        #linear layers for the auxiliary losses
        self.aux_linear1 = nn.Linear(self.h, 10)
        self.aux_linear2 = nn.Linear(self.h, 10)
        
        #linear layers for the final output
        self.fc1 = nn.Linear(2*self.h, 160)
        self.fc2 = nn.Linear(160, 80)
        self.fc3 = nn.Linear(80,40)
        self.fc4 = nn.Linear(40,20)
        self.fc5 = nn.Linear(20, 2)

    def forward(self, x):
        
        #computing the convolution
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=3, stride=3))
        x = x.view(-1, 2*self.h)
        
        #computing the linear+softmax to make the auxiliary output
        #deviding the tensor 
        a1 = x.narrow(1, 0,      self.h)
        a2 = x.narrow(1, self.h, self.h)
                
        #computing auxiliarythe output
        a1 = F.softmax(self.aux_linear1(a1), dim = 1).unsqueeze(1)
        a2 = F.softmax(self.aux_linear2(a2), dim = 1).unsqueeze(1)
        #merging the vector to return the final auxiliary output
        aux_output = torch.cat((a1, a2), 1)
        
        #computing the layers for the final output
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        output = F.softmax(self.fc5(x),dim=1)
        return output, aux_output
    
########################################

class C2L2(nn.Module):
    def __init__(self):
        super().__init__()
        
        #convolutions
        self.conv1 = nn.Conv2d(2, 44, kernel_size=3,groups=2)
        self.conv2 = nn.Conv2d(44, 220, kernel_size=3, groups = 2)
        
        #define the half lenght of the output (linearized) after the last convolution
        self.h = int(2**2*220/2)
        
        #linear layers for the auxiliary losses
        self.aux_linear1 = nn.Linear(self.h, 10)
        self.aux_linear2 = nn.Linear(self.h, 10)
        
        #linear layers for the final output
        self.fc1 = nn.Linear(2*self.h, 20)
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        
        #computing the convolution
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = x.view(-1, 2*self.h)
        
        #computing the linear+softmax to make the auxiliary output
        #deviding the tensor 
        a1 = x.narrow(1, 0,      self.h)
        a2 = x.narrow(1, self.h, self.h)
                
        #computing auxiliarythe output
        a1 = F.softmax(self.aux_linear1(a1), dim = 1).unsqueeze(1)
        a2 = F.softmax(self.aux_linear2(a2), dim = 1).unsqueeze(1)
        #merging the vector to return the final auxiliary output
        aux_output = torch.cat((a1, a2), 1)
        
        #computing the layers for the final output
        x = F.relu(self.fc1(x))
        output = F.softmax(self.fc2(x), dim=1)
        return output, aux_output
    
    
class C2L3(nn.Module):
    def __init__(self):
        super().__init__()
        
        #convolutions
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3,groups=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3,groups=2)
        
        #define the half lenght of the output (linearized) after the last convolution
        self.h = int(2**2*128/2)
        
        #linear layers for the auxiliary losses
        self.aux_linear1 = nn.Linear(self.h, 10)
        self.aux_linear2 = nn.Linear(self.h, 10)
        
        #linear layers for the final output
        self.fc1 = nn.Linear(2*self.h, 60)
        self.fc2 = nn.Linear(60, 20)
        self.fc3 = nn.Linear(20, 2)

    def forward(self, x):
        
        #computing the convolution
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = x.view(-1, 2*self.h)
        
        #computing the linear+softmax to make the auxiliary output
        #deviding the tensor 
        a1 = x.narrow(1, 0,      self.h)
        a2 = x.narrow(1, self.h, self.h)
                
        #computing auxiliarythe output
        a1 = F.softmax(self.aux_linear1(a1), dim = 1).unsqueeze(1)
        a2 = F.softmax(self.aux_linear2(a2), dim = 1).unsqueeze(1)
        #merging the vector to return the final auxiliary output
        aux_output = torch.cat((a1, a2), 1)
        
        #computing the layers for the final output
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = F.softmax(self.fc3(x), dim=1)
        return output, aux_output
    
###################################################

class C3L2(nn.Module):
    def __init__(self):
        super().__init__()
        self.p=12
        #convolutions
        self.conv1 = nn.Conv2d(2, 2*self.p, kernel_size=3,groups=2)
        self.conv2 = nn.Conv2d(2*self.p, 4*self.p, kernel_size=3,groups=2)
        self.conv3 = nn.Conv2d(4*self.p, 8*self.p, kernel_size=3,groups=2)
        
        #define the half lenght of the output (linearized) after the last convolution
        self.h = int(4**2*8*self.p/2)
        
        #linear layers for the auxiliary losses
        self.aux_linear1 = nn.Linear(self.h, 10)
        self.aux_linear2 = nn.Linear(self.h, 10)
        
        #linear layers for the final output
        self.fc1 = nn.Linear(2*self.h, 20)
        self.fc2 = nn.Linear(20, 2)

        
    def forward(self, x):
        
        #computing the convolution
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(F.max_pool2d(self.conv3(x), kernel_size=2, stride=2))     
        x = x.view(-1, 2*self.h)
        
        #computing the linear+softmax to make the auxiliary output
        #deviding the tensor 
        a1 = x.narrow(1, 0,      self.h)
        a2 = x.narrow(1, self.h, self.h)
                
        #computing auxiliarythe output
        a1 = F.softmax(self.aux_linear1(a1), dim = 1).unsqueeze(1)
        a2 = F.softmax(self.aux_linear2(a2), dim = 1).unsqueeze(1)
        #merging the vector to return the final auxiliary output
        aux_output = torch.cat((a1, a2), 1)
        
        #computing the layers for the final output
        x = F.relu(self.fc1(x))
        output = F.softmax(self.fc2(x), dim=1)
        return output, aux_output
    
##########################################

class C4L2_bis(nn.Module):#bis got switched, be careful
    def __init__(self):
        super().__init__()
        self.p=7
        #convolutions
        self.conv1 = nn.Conv2d(2, 2*self.p, kernel_size=3,groups=2)
        self.conv2 = nn.Conv2d(2*self.p, 4*self.p, kernel_size=3,groups=2)
        self.conv3 = nn.Conv2d(4*self.p, 8*self.p, kernel_size=3,groups=2)
        self.conv4 = nn.Conv2d(8*self.p, 16*self.p, kernel_size=3,groups=2)
        
        #hidden layer parameter
        self.h = int(3**2*16*self.p/2)
        
        #linear layers for the auxiliary losses
        self.aux_linear1 = nn.Linear(self.h, 10)
        self.aux_linear2 = nn.Linear(self.h, 10)
        
        #linear layers for the final output
        self.fc1 = nn.Linear(2*self.h, 20)
        self.fc2 = nn.Linear(20, 2)

        
    def forward(self, x):
        
        #computing the convolution
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(F.max_pool2d(self.conv4(x), kernel_size=2, stride=2))
        x = x.view(-1, 2*self.h)
        
        #computing the linear+softmax to make the auxiliary output
        #deviding the tensor 
        a1 = x.narrow(1, 0,      self.h)
        a2 = x.narrow(1, self.h, self.h)
                
        #computing auxiliarythe output
        a1 = F.softmax(self.aux_linear1(a1), dim = 1).unsqueeze(1)
        a2 = F.softmax(self.aux_linear2(a2), dim = 1).unsqueeze(1)
        #merging the vector to return the final auxiliary output
        aux_output = torch.cat((a1, a2), 1)
        
        x = F.relu(self.fc1(x))
        output = F.softmax(self.fc2(x), dim=1)
        return output, aux_output
    
    
#################################################


class C4L2(nn.Module):
    def __init__(self):
        super().__init__()
        self.p=9
        #convolutions.
        self.conv1 = nn.Conv2d(2, 2*self.p, kernel_size=3,groups=2)
        self.conv2 = nn.Conv2d(2*self.p, 4*self.p, kernel_size=3,groups=2)
        self.conv3 = nn.Conv2d(4*self.p, 8*self.p, kernel_size=2,groups=2)
        self.conv4 = nn.Conv2d(8*self.p, 16*self.p, kernel_size=2,groups=2)
        
        #hidden layer parameter
        self.h = int(3**2*16*self.p/2)
        
        #linear layers for the auxiliary losses
        self.aux_linear1 = nn.Linear(self.h, 10)
        self.aux_linear2 = nn.Linear(self.h, 10)
        
        #linear layers for the final output
        self.fc1 = nn.Linear(2*self.h, 20)
        self.fc2 = nn.Linear(20, 2)

        
    def forward(self, x):
        
        #convolutional computations
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 2*self.h)
        
        #computing the linear+softmax to make the auxiliary output
        #deviding the tensor 
        a1 = x.narrow(1, 0,      self.h)
        a2 = x.narrow(1, self.h, self.h)
        
        #computing auxiliarythe output
        a1 = F.softmax(self.aux_linear1(a1), dim = 1).unsqueeze(1)
        a2 = F.softmax(self.aux_linear2(a2), dim = 1).unsqueeze(1)
        #merging the vector to return the final auxiliary output
        aux_output = torch.cat((a1, a2), 1)
        
        #computing the layers for the final output
        x = F.relu(self.fc1(x))
        output = F.softmax(self.fc2(x), dim=1)
        return output, aux_output

############################################################################################################################################################################################Model with weight sharing

#The number of units after the final convolutional layer has been computed explicitly as 
# (shape of output of final convolutional layer)^2 * number of final channels/2

#The models names are created as follows:
# 'C'+ #of convolutions + 'L' + number of linear layers + 'WS'
# The auxiliary 'branch' of the network has been put between the convolutions 
# and the final linear layers and uses a linear layer per picture to predict the picture label

###################
class C1L2WS(nn.Module):
    def __init__(self):
        super().__init__()
        
        #convolutions. To be run in parallel on the 2 images 
        self.conv1 = nn.Conv2d(1, 72, kernel_size=3)
        
        #hidden layer parameter
        self.h=4**2*72
        
        #linear layer for the auxiliary losses. To be run in parallel on the 2 images 
        self.aux_linear = nn.Linear(self.h, 10)
        
        #The outputs of the convolutions will be merged in a single vector
        #Then, the following linear layers for the real loss will be used.
        self.fc1 = nn.Linear(2*self.h, 20)
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        
        #crop the images in 2
        picture1 =x.narrow(1, 0, 1)
        picture2 =x.narrow(1, 1, 1)
        
        #computing the convolutions+relu in parallel on the two pictures.
        x1 = F.relu(F.max_pool2d(self.conv1(picture1), kernel_size=3, stride=3))
        x2 = F.relu(F.max_pool2d(self.conv1(picture2), kernel_size=3, stride=3))
        
        #Reshaping to make 2 vectors
        x1 = x1.view(-1, self.h)
        x2 = x2.view(-1, self.h)
        
        #computing the linear+softmax to make the auxiliary output
        a1 = F.softmax(self.aux_linear(x1), dim = 1).unsqueeze(1)
        a2 = F.softmax(self.aux_linear(x2), dim = 1).unsqueeze(1)
        #merging the vector to return the final auxiliary output
        aux_output = torch.cat((a1, a2), 1)
        
        #computing the layers for the final output
        x = torch.cat((x1,x2), 1)
        
        x = F.relu(self.fc1(x))
        output = F.softmax(self.fc2(x), dim=1)
        return output, aux_output
    
#########################################
class C1L3WS(nn.Module):
    def __init__(self):
        super().__init__()
        
        #convolutions. To be run in parallel on the 2 images
        self.conv1 = nn.Conv2d(1, 15, kernel_size=3)
        
        #hidden layer parameter
        self.h=4**2*15
        
        #linear layer for the auxiliary losses. To be run in parallel on the 2 images
        self.aux_linear = nn.Linear(self.h, 10)
        
        #The outputs of the convolutions will be merged in a single vector
        #Then, the following linear layers for the real loss will be used.
        self.fc1 = nn.Linear(2*self.h, 128)
        self.fc2 = nn.Linear(128, 20)
        self.fc3 = nn.Linear(20, 2)
        

    def forward(self, x):
        
        #crop the images in 2
        picture1 =x.narrow(1, 0, 1)
        picture2 =x.narrow(1, 1, 1)
        
        #computing the convolutions+relu in parallel on the two pictures.
        x1 = F.relu(F.max_pool2d(self.conv1(picture1), kernel_size=3, stride=3))
        x2 = F.relu(F.max_pool2d(self.conv1(picture2), kernel_size=3, stride=3))
        
        #Reshaping to make 2 vectors
        x1 = x1.view(-1, self.h)
        x2 = x2.view(-1, self.h)
        
        #computing the linear+softmax to make the auxiliary output
        a1 = F.softmax(self.aux_linear(x1), dim = 1).unsqueeze(1)
        a2 = F.softmax(self.aux_linear(x2), dim = 1).unsqueeze(1)
        #merging the vector to return the final auxiliary output
        aux_output = torch.cat((a1, a2), 1)
        
        #computing the linear layers for the final output
        x = torch.cat((x1,x2), 1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = F.softmax(self.fc3(x),dim=1)
        return output, aux_output
    
################################################

class C1L5WS(nn.Module):
    def __init__(self):
        super().__init__()
        
        #convolutions. To be run in parallel on the 2 images
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        
        #hidden layer parameter
        self.h=4**2*10
        
        #linear layer for the auxiliary losses. To be run in parallel on the 2 images 
        self.aux_linear = nn.Linear(self.h, 20)
        
        #The outputs of the convolutions will be merged in a single vector
        #Then, the following linear layers for the real loss will be used.
        self.fc1 = nn.Linear(2*self.h, 160)
        self.fc2 = nn.Linear(160, 80)
        self.fc3 = nn.Linear(80,40)
        self.fc4 = nn.Linear(40,20)
        self.fc5 = nn.Linear(20, 2)

    def forward(self, x):
        
        
        #crop the images in 2
        picture1 =x.narrow(1, 0, 1)
        picture2 =x.narrow(1, 1, 1)
        
        #computing the convolutions+relu in parallel on the two pictures.
        x1 = F.relu(F.max_pool2d(self.conv1(picture1), kernel_size=3, stride=3))
        x2 = F.relu(F.max_pool2d(self.conv1(picture2), kernel_size=3, stride=3))
        
        #Reshaping to make 2 vectors
        x1 = x1.view(-1, self.h)
        x2 = x2.view(-1, self.h)
        
        #computing the linear+softmax to make the auxiliary output
        a1 = F.softmax(self.aux_linear(x1), dim = 1).unsqueeze(1)
        a2 = F.softmax(self.aux_linear(x2), dim = 1).unsqueeze(1)
        #merging the vector to return the final auxiliary output
        aux_output = torch.cat((a1, a2), 1)
        
        #computing the linear layers for the final output
        x = torch.cat((x1,x2), 1)
        
        x = F.relu(self.fc1(x.view(-1, 2*self.h)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        output = F.softmax(self.fc5(x),dim=1)
        return output, aux_output
    
##############################################################

class C2L2WS(nn.Module):
    def __init__(self):
        super().__init__()
        
        #convolutions. To be run in parallel on the 2 images 
        self.conv1 = nn.Conv2d(1, 22, kernel_size=3)
        self.conv2 = nn.Conv2d(22, 110, kernel_size=3)
        
        #hidden layer parameter
        self.h=2**2*110
        #linear layer for the auxiliary losses. To be run in parallel on the 2 images 
        self.aux_linear = nn.Linear(self.h, 10)
        
        #The outputs of the convolutions will be merged in a single vector
        #Then, the following linear layers for the real loss will be used.
        self.fc1 = nn.Linear(2*self.h, 20)
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        
        #crop the images in 2
        picture1 =x.narrow(1, 0, 1)
        picture2 =x.narrow(1, 1, 1)
        
        #computing the convolutions+relu in parallel on the two pictures.
        #first convolution
        x1 = F.relu(F.max_pool2d(self.conv1(picture1), kernel_size=2, stride=2))
        x2 = F.relu(F.max_pool2d(self.conv1(picture2), kernel_size=2, stride=2))
        #second convolution
        x1 = F.relu(F.max_pool2d(self.conv2(x1), kernel_size=2, stride=2))
        x2 = F.relu(F.max_pool2d(self.conv2(x2), kernel_size=2, stride=2))

        #Reshaping to make 2 vectors
        x1 = x1.view(-1, self.h)
        x2 = x2.view(-1, self.h)
        
        #computing the linear+softmax to make the auxiliary output
        a1 = F.softmax(self.aux_linear(x1), dim = 1).unsqueeze(1)
        a2 = F.softmax(self.aux_linear(x2), dim = 1).unsqueeze(1)
        #merging the vector to return the final auxiliary output
        aux_output = torch.cat((a1, a2), 1)
        
        #computing the linear layers for the final output
        x = torch.cat((x1,x2), 1)

        x = F.relu(self.fc1(x.view(-1, 2*self.h)))
        output = F.softmax(self.fc2(x), dim=1)
        return output, aux_output
    
#################################################

class C2L3WS(nn.Module):
    def __init__(self):
        super().__init__()
        
        #convolutions. To be run in parallel on the 2 images 
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        
        #hidden layer parameter
        self.h=2**2*64
        
        #linear layer for the auxiliary losses. To be run in parallel on the 2 images 
        self.aux_linear = nn.Linear(self.h, 10)
        
        #The outputs of the convolutions will be merged in a single vector
        #Then, the following linear layers for the real loss will be used.
        self.fc1 = nn.Linear(2*self.h, 60)
        self.fc2 = nn.Linear(60, 20)
        self.fc3 = nn.Linear(20, 2)

    def forward(self, x):
        
        #crop the images in 2
        picture1 =x.narrow(1, 0, 1)
        picture2 =x.narrow(1, 1, 1)
        
        #computing the convolutions+relu in parallel on the two pictures.
        #first convolution
        x1 = F.relu(F.max_pool2d(self.conv1(picture1), kernel_size=2, stride=2))
        x2 = F.relu(F.max_pool2d(self.conv1(picture2), kernel_size=2, stride=2))
        #second convolution
        x1 = F.relu(F.max_pool2d(self.conv2(x1), kernel_size=2, stride=2))
        x2 = F.relu(F.max_pool2d(self.conv2(x2), kernel_size=2, stride=2))
        
        #Reshaping to make 2 vectors
        x1 = x1.view(-1, self.h)
        x2 = x2.view(-1, self.h)
        
        #computing the linear+softmax to make the auxiliary output
        a1 = F.softmax(self.aux_linear(x1), dim = 1).unsqueeze(1)
        a2 = F.softmax(self.aux_linear(x2), dim = 1).unsqueeze(1)
        #merging the vector to return the final auxiliary output
        aux_output = torch.cat((a1, a2), 1)
        
        #computing the linear layers for the final output
        x = torch.cat((x1,x2), 1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = F.softmax(self.fc3(x), dim=1)
        return output, aux_output


##############################################################

class C3L2WS(nn.Module):
    def __init__(self):
        super().__init__()
        self.p=12
        #convolutions. To be run in parallel on the 2 images
        self.conv1 = nn.Conv2d(1, self.p, kernel_size=3)
        self.conv2 = nn.Conv2d(self.p, 2*self.p, kernel_size=3)
        self.conv3 = nn.Conv2d(2*self.p, 4*self.p, kernel_size=3)
        
        #hidden layer parameter
        self.h=4**2*4*self.p
        
        #linear layer for the auxiliary losses. To be run in parallel on the 2 images 
        self.aux_linear = nn.Linear(self.h, 10)
        
        #The outputs of the convolutions will be merged in a single vector
        #Then, the following linear layers for the real loss will be used.
        self.fc1 = nn.Linear(2*self.h, 20)
        self.fc2 = nn.Linear(20, 2)

        
    def forward(self, x):
        #crop the images in 2
        picture1 =x.narrow(1, 0, 1)
        picture2 =x.narrow(1, 1, 1)
        
        #computing the convolutions+relu in parallel on the two pictures.
        #first convolution
        x1 = F.relu(self.conv1(picture1))
        x2 = F.relu(self.conv1(picture2))
        #second convolution
        x1 = F.relu(self.conv2(x1))
        x2 = F.relu(self.conv2(x2))
        #third convolution
        x1 = F.relu(F.max_pool2d(self.conv3(x1), kernel_size=2, stride=2))
        x2 = F.relu(F.max_pool2d(self.conv3(x2), kernel_size=2, stride=2))
        
        
        #Reshaping to make 2 vectors
        x1 = x1.view(-1, self.h)
        x2 = x2.view(-1, self.h)
        
        #computing the linear+softmax to make the auxiliary output
        a1 = F.softmax(self.aux_linear(x1), dim = 1).unsqueeze(1)
        a2 = F.softmax(self.aux_linear(x2), dim = 1).unsqueeze(1)
        #merging the vector to return the final auxiliary output
        aux_output = torch.cat((a1, a2), 1)
        
        #computing the linear layers for the final output
        x = torch.cat((x1,x2), 1)
        
        x = F.relu(self.fc1(x))
        output = F.softmax(self.fc2(x), dim=1)
        return output, aux_output

###########################################################################

class C4L2WS_bis(nn.Module):#bis got changed
    def __init__(self):
        super().__init__()
        
        self.p=7
        #convolutions. To be run in parallel on the 2 images
        self.conv1 = nn.Conv2d(1, self.p, kernel_size=3)
        self.conv2 = nn.Conv2d(self.p, 2*self.p, kernel_size=3)
        self.conv3 = nn.Conv2d(2*self.p, 4*self.p, kernel_size=3)
        self.conv4 = nn.Conv2d(4*self.p, 8*self.p, kernel_size=3)
        
        #hidden layer parameter
        self.h=3**2*8*self.p
        
        #linear layer for the auxiliary losses. To be run in parallel on the 2 images 
        self.aux_linear = nn.Linear(self.h, 10)
        
        #The outputs of the convolutions will be merged in a single vector
        #Then, the following linear layers for the real loss will be used.
        self.fc1 = nn.Linear(self.h*2, 20)
        self.fc2 = nn.Linear(20, 2)

        
    def forward(self, x):
        #crop the images in 2
        picture1 =x.narrow(1, 0, 1)
        picture2 =x.narrow(1, 1, 1)
        
        #computing the convolutions+relu in parallel on the two pictures.
        #first convolution        
        x1 = F.relu(self.conv1(picture1))
        x2 = F.relu(self.conv1(picture2))
        #second convolution
        x1 = F.relu(self.conv2(x1))
        x2 = F.relu(self.conv2(x2))
        #third convolution
        x1 = F.relu(self.conv3(x1))
        x2 = F.relu(self.conv3(x2))
        #fourth convolution
        x1 = F.relu(F.max_pool2d(self.conv4(x1), kernel_size=2, stride=2))
        x2 = F.relu(F.max_pool2d(self.conv4(x2), kernel_size=2, stride=2))
        
        #Reshaping to make 2 vectors
        x1 = x1.view(-1, self.h)
        x2 = x2.view(-1, self.h)
        
        #computing the linear+softmax to make the auxiliary output
        a1 = F.softmax(self.aux_linear(x1), dim = 1).unsqueeze(1)
        a2 = F.softmax(self.aux_linear(x2), dim = 1).unsqueeze(1)
        #merging the vector to return the final auxiliary output
        aux_output = torch.cat((a1, a2), 1)
        
        #computing the linear layers for the final output
        x = torch.cat((x1,x2), 1)

        x = F.relu(self.fc1(x))
        output = F.softmax(self.fc2(x), dim=1)
        return output, aux_output


#################################################################

class C4L2WS(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.p=9
        
        #convolutions. To be run in parallel on the 2 images
        self.conv1 = nn.Conv2d(1, self.p, kernel_size=3)
        self.conv2 = nn.Conv2d(self.p, 2*self.p, kernel_size=3)
        self.conv3 = nn.Conv2d(2*self.p, 4*self.p, kernel_size=2)
        self.conv4 = nn.Conv2d(4*self.p, 8*self.p, kernel_size=2)
        
        #hidden layer parameter
        self.h=3**2*8*self.p
        
        #linear layer for the auxiliary losses. To be run in parallel on the 2 images 
        self.aux_linear = nn.Linear(self.h, 10)
        
        #The outputs of the convolutions will be merged in a single vector
        #Then, the following linear layers for the real loss will be used.
        self.fc1 = nn.Linear(2*self.h, 20)
        self.fc2 = nn.Linear(20, 2)

        
    def forward(self, x):
        #crop the images in 2
        picture1 =x.narrow(1, 0, 1)
        picture2 =x.narrow(1, 1, 1)
        
        #computing the convolutions+relu in parallel on the two pictures.
        #first convolution        
        x1 = F.relu(self.conv1(picture1))
        x2 = F.relu(self.conv1(picture2))
        #second convolution
        x1 = F.relu(F.max_pool2d(self.conv2(x1), kernel_size=2, stride=2))
        x2 = F.relu(F.max_pool2d(self.conv2(x2), kernel_size=2, stride=2))
        #third convolution
        x1 = F.relu(self.conv3(x1))
        x2 = F.relu(self.conv3(x2))
        #fourth convolution
        x1 = F.relu(self.conv4(x1))
        x2 = F.relu(self.conv4(x2))
        
        #Reshaping to make 2 vectors
        x1 = x1.view(-1, self.h)
        x2 = x2.view(-1, self.h)
        
        #computing the linear+softmax to make the auxiliary output
        a1 = F.softmax(self.aux_linear(x1), dim = 1).unsqueeze(1)
        a2 = F.softmax(self.aux_linear(x2), dim = 1).unsqueeze(1)
        #merging the vector to return the final auxiliary output
        aux_output = torch.cat((a1, a2), 1)
        
        #computing the linear layers for the final output
        x = torch.cat((x1,x2), 1)

        x = F.relu(self.fc1(x))
        output = F.softmax(self.fc2(x), dim=1)
        return output, aux_output
























