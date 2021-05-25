import torch
import math
torch.set_grad_enabled(False)

###########################################################################################
##
#Classes that will be extended

class Module(object):
    def forward (self, *input):
        raise NotImplementedError
        
    def backward ( self , * gradwrtoutput ) :
        raise NotImplementedError
    
    def get_parameters( self ) :
        return []   
    
class Losses(object):        
    def forward():
        return NotImplementedError
    def backward():
        NotImplementedError

        
class Optimizer(object):
    def zero_grad(self):
        for parameter in self.param : 
            parameter.grad = 0
            
    def step(self):
        raise NotImplementedError
        
#############################################################################################
#Modlues implementation 

#Parameters implementation
class Parameter():
    def __init__(self):
        self.name = ''
        self.data = None
        self.grad = None
        
#Linear module implementation
class Linear(Module):
    
    def __init__(self, input_dim, out_dim, bias = True):
        super().__init__()
        std = 1/math.sqrt(input_dim)
        self.weight = Parameter()
        self.parameters = []
        
        self.weight.data = torch.rand(out_dim, input_dim)
        self.weight.data = 2*std*self.weight.data - std
        self.weight.name = 'weight'
        self.parameters += [self.weight]
        
        self.with_bias = bias
        if bias :
            self.bias = Parameter()
            self.bias.data = torch.rand(out_dim)
            self.bias.data = 2*std*self.bias.data - std
            self.bias.data = self.bias.data.unsqueeze(0)
            self.bias.name = 'bias'
            self.parameters +=[self.bias]
            
        self.x = None
              
    def forward(self, x):
        self.x = x
        self.batch_size = x.shape[0]
        return self.x.mm(self.weight.data.T) + self.bias.data
        
    def backward(self, prev_grad):
        
        prev_grad = prev_grad.view(self.batch_size, -1, 1)
        if self.x is None:
            raise CallForwardFirst
        
        if self.weight.grad is None:
            self.weight.grad = torch.zeros_like(self.weight.data)
        
        grad_on_batch = prev_grad.view(self.batch_size, -1, 1)*self.x.view(self.batch_size, 1, -1)
        self.weight.grad += grad_on_batch.mean(0)
        
        if self.with_bias:
            if self.bias.grad is None:
                self.bias.grad = torch.zeros_like(self.bias.data)
            grad_on_batch = prev_grad.view(self.batch_size, -1)
            self.bias.grad += grad_on_batch.mean(0)
        
        #if the output has dimension one, squeezing creates problems
        if prev_grad.shape[1]>1:
            prev_grad = prev_grad.squeeze()
        next_grad = prev_grad@self.weight.data
        return next_grad.squeeze()
    
    def get_parameters(self):
        return self.parameters    
    
#Tanh nonlinearity
class Tanh(Module):
    def __init__(self):
        self.x = None
    
    def forward (self, x):
        self.x = x
        return torch.tanh(x)
        
    def backward ( self, prev_grad) :
        if self.x is None:
            raise CallForwardFirst
            
        def d(x):
            return 4 * (x.exp() + x.mul(-1).exp()).pow(-2)
        
        return d(self.x)*prev_grad
    

#ReLU nonlinearity
class ReLu(Module):
    def __init__(self):
        self.x = None
    
    def forward (self, x):
        self.x = x
        x[x<0]=0
        return x
        
    def backward ( self, prev_grad) :
        if self.x is None:
            raise CallForwardFirst
            
        def d(x):
            x[x<0]=0
            x[x>0]=1
            return x
        
        return d(self.x)*prev_grad

#sigmoid nonlinearity
class Sigmoid(Module):
    def __init__(self):
        self.x = None
    
    def forward (self, x):
        self.x = x
        return torch.sigmoid(x)
        
    def backward ( self, prev_grad) :
        if self.x is None:
            raise CallForwardFirst
            
        def d(x):
            y=torch.sigmoid(x)
            return y*(1-y)
        
        return d(self.x)*prev_grad
    

#Softmax nonlinearity
class Softmax(Module):
    def __init__(self):
        self.x = None
    
    def forward (self, x):
        self.x = x
        return torch.softmax(x,1)     
     
    def backward ( self, prev_grad) :
        if self.x is None:
            raise CallForwardFirst
            
        
        def d(x):
            s = x.softmax(1)
            temp = s.unsqueeze(-1)
            off_diag = temp*torch.transpose(temp, 1, 2)
            diag = torch.diag_embed(torch.diagonal(off_diag, dim1 = 1, dim2 = 2).sqrt()) 
            return diag - off_diag
        return torch.einsum('b ij, bj -> bi', d(self.x),prev_grad)/ self.x.shape[0]
    

#Sequential module
class Sequential(object):
    def __init__(self, modules):
        super().__init__()
        self.modules=modules
        self.parameters = []
        for m in self.modules:
            param = m.get_parameters()
            if param:
                self.parameters += param
        
    def forward(self,x):
        for m in self.modules:
            x=m.forward(x)
        return x
    
    def backward(self, loss_grad):
        x = loss_grad
        for m in reversed(self.modules):
            #print(m)
            x = m.backward(x)
            
    def get_parameters(self):
        return self.parameters

    def set_parameters(self , params):
        self.parameters = params
        
        
##########################################################################################
##
#Losses 

#MSE Loss
class MSE(Losses):
    # Attention! Works well only when the vectors provided are of the form [batch_size, vector dimension]
    # Otherwise it doesn know what dimesion to pick for the mean computation
    # I'll fix this later
    def __init__(self):
        self.x = None
    def forward(self, x, t):
        self.x = x
        self.t = t
        return (x - t).pow(2).mean()
    
    def backward(self):
        if self.x == None or self.t == None:
            raise CallForwardFirst
        return 2 * (self.x - self.t)/(x.shape.mul()self.x.shape[1]*self.x.shape[0])


###########################################################################################
##
#Optimizers


#SGD Optimizer with momentum
class SGD(Optimizer):

    def __init__(self,lr, parameters, momentum = False ,beta= 0.9) :   
        super().__init__()
        self.eta = lr   # learning rate is fixed
        self.param = parameters
        # Initialize momentum to zero
        self.v = self.param.copy()
        self.v = []
        for i in range(len(self.param)):
            size_param = self.param[i].data.size()
            v_init= torch.zeros(size_param)
            (self.v).append(v_init)
        self.beta = beta
        self.momentum = momentum

    def step(self): 
        for parameter, V in zip(self.param, self.v): 
            if self.momentum :    # SGD + Momentum
                V = self.beta * V + self.eta * parameter.grad   
                parameter.data = parameter.data - V
            else :                # SGD 
                parameter.data = parameter.data - self.eta * parameter.grad
           
        return self.param






























