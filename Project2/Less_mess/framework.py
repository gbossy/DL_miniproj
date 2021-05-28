from torch import empty, tensor
import math
#set_grad_enabled(False)
import torch
###########################################################################################
## Superclasses of Modules, Losses and Optmizer. To be extended. 
#

# We need to implement at least forward and backward pass
class Module(object):
    def forward (self, *input):
        raise NotImplementedError
        
    def backward ( self , * gradwrtoutput ) :
        raise NotImplementedError
    
    #Returns the parameters of the module. 
    #If not implemented returns an empty list
    def get_parameters( self ) :
        return []   
    

# We need to implement at least forward and backward pass 
class Losses(object):        
    def forward():
        raise NotImplementedError
    def backward():
        raise NotImplementedError

#We need to implement the step function for the optimizer
class Optimizer(object):
    def step(self):
        raise NotImplementedError
    
    #sets the gradients of the pèarameters to zero.
    #It is shared among all optimizers
    def zero_grad(self):
        for parameter in self.param : 
            parameter.grad = 0
            
    
        
#############################################################################################
#Modlues implementation 

# A parameter is an object with three attributes.
# "name" is the name of the parameter e.g, "weight" or "Linear".
# "data" attribute stores the values of the parameter 
# "grad" attribute stores the values of its gradient.
class Parameter():
    def __init__(self):
        self.name = ''
        self.data = None
        self.grad = None

        
#Linear module extends the Module class and has a weight parameter and, if bias == True, a bias parameter.
#the 
class Linear(Module):
    
    #To be initialized it needs the dimension of the input and the output. If bias == True, it also has bias parameter.
    def __init__(self, input_dim, out_dim, bias = True):
        super().__init__()
        std = 1/math.sqrt(input_dim)
        self.parameters = []
        
        #The weights are initialized with a uniform distribution in the interval [-std, std]
        #Where std depends on the input dimension "input_dim": std = 1/sqrt(input_dim)
        self.weight = Parameter()
        self.weight.data = empty(out_dim, input_dim).uniform_()
        self.weight.data = 2*std*self.weight.data - std
        self.weight.name = 'weight'
        self.parameters += [self.weight]
        
        #If a bias is required it initializes it. The initialization of the bias is similar to thet of the weights
        self.with_bias = bias
        if bias :
            self.bias = Parameter()
            self.bias.data = empty(out_dim).uniform_()
            self.bias.data = 2*std*self.bias.data - std
            self.bias.data = self.bias.data.unsqueeze(0)
            self.bias.name = 'bias'
            self.parameters +=[self.bias]
        
        # x is the input of the linear layer. It is initialized to None and updated when function forward() is called
        self.x = None
    
              
    #Implementation of the forward pass. 
    #The input "x" has to have the shape [batch_size, input_dimension] 
    #The output will have the shape [batch_size, output_dimension]
    def forward(self, x):
        #store the input for its use in the backward pass
        self.x = x
        #store the batch size for its use in the backward pass
        self.batch_size = x.shape[0]
        temp = self.x.mm(self.weight.data.T)
        if self.bias:
            temp += self.bias.data
        return temp 
    
       
    #Implementation of the backward pass.
    #It takes as input the the derivative of the loss w.r.t the following layer
    #It updates the gradient of the parameters and returns the derivative of the loss w.r.t this layer
    def backward(self, prev_grad):   
        
        #controls that the forward pass has been called before
        assert self.x is not None,'Call the forward pass first'
        
        #makes sure the previous gradient is in the proper shape
        prev_grad = prev_grad.view(self.batch_size, -1, 1)
        
        #if the gradient of the weights are not yet initialized, it initializes it 
        if self.weight.grad is None:
            self.weight.grad = empty(self.weight.data.shape)
            self.weight.grad[self.bias.grad!=0]=0

        #computes and updates the mean gradient on the batch 
        grad_on_batch = prev_grad.view(self.batch_size, -1, 1)*self.x.view(self.batch_size, 1, -1)
        self.weight.grad += grad_on_batch.mean(0)
        
        #if a bias is present, repet the previous stages
        if self.with_bias:
            if self.bias.grad is None:
                self.bias.grad = empty(self.bias.data.shape)
                self.bias.grad[self.bias.grad!=0]=0
            grad_on_batch = prev_grad.view(self.batch_size, -1)
            self.bias.grad += grad_on_batch.mean(0)
        
        #computes and returns the gradient of the loss w.r.t. the present layer
        #if the output has dimension one, squeezing creates problems
        if prev_grad.shape[1]>1:
            prev_grad = prev_grad.squeeze()
        next_grad = prev_grad@self.weight.data
        return next_grad.squeeze()
    
    
    #function to get the parameters
    def get_parameters(self):
        return self.parameters    
    
    
###################################    
#Tanh nonlinearity
class Tanh(Module):
    
    #initializes the input to none
    def __init__(self):
        self.x = None
    
    #copmutes the forward pass
    def forward (self, x):
        self.x = x
        return x.tanh()
      
    #the backward pass. Takes as input the gradient of the following layer and computes 
    #the derivative w.r.t. the present layer
    def backward ( self, prev_grad) :
        
        #check that the forward pass was called first and the input has been initialized
        assert self.x is not None,'Call the forward pass first'
        
        #defines the derivative of tanh
        def d(x):
            return 4 * (x.exp() + x.mul(-1).exp()).pow(-2)
        
        return d(self.x)*prev_grad
    

####################################   
#ReLU nonlinearity
class ReLu(Module):
    
    #initializes the input to none
    def __init__(self):
        self.x = None
    
    #copmutes the forward pass
    def forward (self, x):
        self.x = x
        x[x<0]=0
        return x
    
    #the backward pass. Takes as input the gradient of the following layer and computes 
    #the derivative w.r.t. the present layer
    def backward ( self, prev_grad) :
        
        #check that the forward pass was called first and the input has been initialized
        assert self.x is not None,'Call the forward pass first'
            
        #defines the derivative of Relu
        def d(x):
            x[x<0]=0
            x[x>0]=1
            return x
        
        return d(self.x)*prev_grad

    
#########################################   
#sigmoid nonlinearity
class Sigmoid(Module):
    
    #initializes the input to none
    def __init__(self):
        self.x = None
    
    #copmutes the forward pass
    def forward (self, x):
        self.x = x
        return x.sigmoid()
        
    #the backward pass. Takes as input the gradient of the following layer and computes 
    #the derivative w.r.t. the present layer
    def backward ( self, prev_grad) :
        
        #check that the forward pass was called first and the input has been initialized
        assert self.x is not None,'Call the forward pass first'
            
        #defines the derivative of Relu  
        def d(x):
            y=x.sigmoid()
            return y*(1-y)
        
        return d(self.x)*prev_grad
    

########################################
#Softmax nonlinearity
class Softmax(Module):
    
    #initializes the input to none
    def __init__(self):
        self.x = None
    
    #copmutes the forward pass
    def forward (self, x):
        self.x = x
        return x.softmax(1)     
    
    #the backward pass. Takes as input the gradient of the following layer and computes 
    #the derivative w.r.t. the present layer
    def backward ( self, prev_grad) :
        
        #check that the forward pass was called first and the input has been initialized
        assert self.x is not None,'Call the forward pass first'
            
        #defines the derivative of Relu (actually the jacobian, as it returns a matrix)  
        def d(x):
            s = x.softmax(1)
            temp = s.unsqueeze(-1)
            off_diag = temp*torch.transpose(temp, 1, 2)
            diag = torch.diag_embed(torch.diagonal(off_diag, dim1 = 1, dim2 = 2).sqrt()) 
            return diag - off_diag
        #return torch.einsum('b ij, bj -> bi', d(self.x),prev_grad)/ self.x.shape[0]
        return d(self.x).bmm(prev_grad.unsqueeze(-1)).squeeze()  / self.x.shape[0]
    

#######################################
#Sequential module. It takes a list of modules at initiliazation and merge them in a unique module
class Sequential(object):
    
    #initialization
    def __init__(self, modules):
        super().__init__()
        
        #stores the modules of the input 
        self.modules=modules
        
        #stores all the parameters of the modules in a separate list
        self.parameters = []
        for m in self.modules:
            param = m.get_parameters()
            if param:
                self.parameters += param
    
    #the forward pass si computed by calling the forward pass of each module in the proper order
    def forward(self,x):
        for m in self.modules:
            x=m.forward(x)
        return x
    
    #the backward pass si computed by calling the backward pass of each module in the proper order
    def backward(self, loss_grad):
        x = loss_grad
        for m in reversed(self.modules):
            x = m.backward(x)
    
    #function to get the parameters of the module
    def get_parameters(self):
        return self.parameters
    
    #ATTENTION: I think we don't need this
    #def set_parameters(self , params):
    #    self.parameters = params
        
        
##########################################################################################################################################
##########################################################################################################################################
#Implementation of the different losses 

#MSE Loss
class MSE(Losses):
  
    #initializes the input to none
    def __init__(self):
        self.x = None
    
    #defines the forward pass
    def forward(self, x, t):
        self.x = x
        self.t = t
        return (x - t).pow(2).mean()
    
    #defines the backward pass
    def backward(self):
        
        #check that the forward pass was called first and the input and target have been initialized
        assert self.x is not None,'Call the forward pass first'
        assert self.t is not None,'Call the forward pass first'
        
        return 2 * (self.x - self.t)/(self.x.nelement())


###########################################################################################################################################
###########################################################################################################################################
#Implementation of the different optimizers


#SGD Optimizer with momentum
class SGD(Optimizer):

    def __init__(self,lr, parameters, momentum = False ,beta= 0.9) :   
        super().__init__()
        
        #stores the parameters
        self.eta = lr
        self.param = parameters
        self.momentum = momentum
        self.beta = beta
        
        # If momentum is required, initialize v to zero
        if momentum:
            self.v = []
            for p in self.param:
                self.v += [empty(p.data.shape).zero_()]
        
    #updates the weights using the gradient
    def step(self): 
        for parameter, i in zip(self.param, range(len(self.param))): 
            
            #if the moementum is requered updates both v and the parameter
            if self.momentum :
                self.v[i] = self.beta * self.v[i] + self.eta * parameter.grad   
                parameter.data = parameter.data - self.v[i]
                                
            #ATTENTION: remember to change the 100                
            #if the moementum is not required updates using the regular SGD
            else :           
                parameter.data = parameter.data - self.eta * parameter.grad
           




