
import torch
from torch import nn
from torch import optim
from torch import Tensor
from torch.nn import functional as F
import matplotlib.pyplot as plt
import dlc_practical_prologue as prologue
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

def data( N, normalize, one_hot, loss, anew = True,
    train_input = None, train_target = None, train_classes = None, test_input = None, test_target = None, test_classes = None, shuffle=False):
    
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
        
    if shuffle:
        
        #shuffle data
        permuted_index = torch.randperm(train_input.size()[0])
        train_input_shuffled = train_input[permuted_index]
        train_target_shuffled = train_target[permuted_index]
        train_classes_shuffled = train_classes[permuted_index]

    if one_hot: 
        assertion_cycle()
    
        #classes to onehot
        temp1 = prologue.convert_to_one_hot_labels(train_classes[:,0], train_classes[:,0])
        temp2 = prologue.convert_to_one_hot_labels(train_classes[:,1], train_classes[:,1])        
        train_classes = torch.cat((temp1.unsqueeze(2), temp2.unsqueeze(2)), dim = 2)
        train_classes = torch.transpose(train_classes, 1, 2)

        temp1 = prologue.convert_to_one_hot_labels(test_classes[:,0], test_classes[:,0])
        temp2 = prologue.convert_to_one_hot_labels(test_classes[:,1], test_classes[:,1])        
        test_classes = torch.cat((temp1.unsqueeze(2), temp2.unsqueeze(2)), dim = 2)
        test_classes = torch.transpose(test_classes, 1, 2)

    if normalize:
        
        #normalize data
        assertion_cycle()
        mu, std = train_input.mean(), train_input.std()
        train_input = train_input.sub_(mu).div_(std)
        test_input = test_input.sub_(mu).div_(std)
        
    if loss == nn.CrossEntropyLoss:
        assertion_cycle()
        train_target = train_target.long()
        test_target = test_target.long()

        train_classes = train_classes.long()
        test_classes = test_classes.long()
        
    return train_input, train_target, train_classes, test_input, test_target, test_classes
    
#Base functions adapted from the practicals
def train_model(model, train_input, train_target,train_classes, mini_batch_size, test_input=None, test_target=None, crit=nn.CrossEntropyLoss, eta = 1e-3, nb_epochs = 50,print_=False, store_loss = False, aux_factor=1, store_error=False, checkpoint_name=None):
    #Initializing the loss, the optimizer, and the stored loss and errors for the plots
    criterion = crit()
    optimizer = optim.Adam(model.parameters(), lr=eta)
    stored_loss = []
    stored_error = []
    
    #Retrieving data if a similar model has already been trained or partially trained
    nb_epochs_finished = 0
    if checkpoint_name!=None:
        try:
            checkpoint = torch.load(checkpoint_name)
            nb_epochs_finished = checkpoint['nb_epochs_finished']
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            if print_:
                print(f'Checkpoint loaded with {nb_epochs_finished} epochs finished.')
            stored_loss=checkpoint['stored_loss']
            if len(stored_loss)>nb_epochs*3:
                stored_loss=stored_loss[0:nb_epochs*3]
            stored_error=checkpoint['stored_error']
            if len(stored_error)>nb_epochs:
                stored_error=stored_error[0:nb_epochs]
        except FileNotFoundError:
            if print_:
                print('Starting from scratch.')
        except:
            print('Error when loading the checkpoint.')
            exit(1)
    
    #Training the network if the checkpoint wasn't fully trained
    for e in range(nb_epochs_finished,nb_epochs):
        
        #Initializing accumulated losses
        #loss1 is the loss over the output (identifying which number is greater)
        #loss2 and loss3 are the auxiliary losses trying to identify the numbers in the 2 input pictures
        acc_loss = 0
        acc_loss1 = 0
        acc_loss2 = 0
        acc_loss3 = 0
        
        #permuting the samples
        permuted_index = torch.randperm(train_input.size()[0])
        train_input_shuffled = train_input[permuted_index]
        train_target_shuffled = train_target[permuted_index]
        train_classes_shuffled = train_classes[permuted_index]
        
        
        for b in range(0, train_input.size(0), mini_batch_size):
            
            #forward pass
            output,aux_output = model(train_input_shuffled.narrow(0, b, mini_batch_size))
            if crit==nn.CrossEntropyLoss:
                loss1 = criterion(output, train_target_shuffled.narrow(0, b, mini_batch_size))
                loss2 = criterion(aux_output[:,0,:], train_classes_shuffled[:,0].narrow(0, b, mini_batch_size))
                loss3 = criterion(aux_output[:,1,:], train_classes_shuffled[:,1].narrow(0, b, mini_batch_size))
                loss = loss1 + aux_factor*(loss2 + loss3)
            else:
                print("Loss not implemented")
                exit(1)
            
            #Update the accumulated losses
            acc_loss = acc_loss + loss.item()
            acc_loss1 = acc_loss1 + loss1.item()
            acc_loss2 = acc_loss2 + loss2.item()
            acc_loss3 = acc_loss3 + loss3.item()
            
            #zero the gradients
            model.zero_grad()
            
            #backward pass
            loss.backward()
            
            #optimizer step
            optimizer.step()
            
        #update the stored losses and error if needed
        if store_loss:
            stored_loss += [[acc_loss1], [acc_loss2], [acc_loss3]]
        if store_error:
            stored_error +=[compute_nb_errors(model, test_input, test_target, mini_batch_size)]
            
        #print the different losses if needed
        if print_:
            print(e, 'tot loss', acc_loss, 'loss1', acc_loss1, 'loss2', acc_loss2, 'loss3', acc_loss3)
            
        #save the checkpoint for later if needed
        if checkpoint_name!=None:
                checkpoint = {'nb_epochs_finished': e + 1,'model_state': model.state_dict(),'optimizer_state': optimizer.state_dict(),'stored_loss':stored_loss,'stored_error':stored_error}
                torch.save(checkpoint, checkpoint_name)
        
    #return the stored quantities  
    return torch.tensor(stored_loss),torch.tensor(stored_error)

#error computing function
def compute_nb_errors(model, input, target, mini_batch_size=100):
    nb_errors = 0

    for b in range(0, input.size(0), mini_batch_size):
        output , aux_output = model(input.narrow(0, b, mini_batch_size))
        _, predicted_classes = output.max(1)
        for k in range(mini_batch_size):
            if target[b + k]!=predicted_classes[k]:
                nb_errors = nb_errors + 1

    return nb_errors

#this function lets us do multiple runs more easily
def run_many_times(model,N=10**3,crit=nn.CrossEntropyLoss,mini_batch_size=100,n=10,print_=True,eta=1e-3,nb_epochs=25,aux_factor=0,shuffle=True, store_error=False,checkpoint_name=None,plot_name=None):
    
    #initialize the average error
    average_error=0
    
    #initialize the losses and errors tensors
    losses=torch.empty(0,nb_epochs,3)
    errors=torch.empty(0,nb_epochs)
    
    #run n instances of the model and record errors and losses at each point
    for i in range(n):
        m=model()
        
        #get a new dataset
        train_input,train_target,train_classes,test_input,test_target,test_classes=data(N,True,False,nn.CrossEntropyLoss,shuffle=shuffle)
        
        #initiate thecheckpoint name if needed
        if checkpoint_name!=None:
            checkpoint_name_spec=checkpoint_name+'try_'+str(i)+'.pth'
        else:
            checkpoint_name_spec=None
        
        #train the model
        new_losses,new_errors=train_model(m, train_input, train_target,train_classes,mini_batch_size,test_input=test_input, test_target=test_target,crit=crit,eta=eta,nb_epochs=nb_epochs,aux_factor=aux_factor,store_loss=True,store_error=store_error,checkpoint_name=checkpoint_name_spec)
        
        #reshape the new losses
        new_losses=new_losses.view(1,nb_epochs, 3)
        
        #store the error if needed
        if store_error:
            new_errors= new_errors.view(1,nb_epochs)
            errors = torch.cat((errors,new_errors),0)
            
        #Print the test error and update the average error if it's going to be printed
        if print_:
            losses = torch.cat((losses, new_losses), 0)
            nb_test_errors = compute_nb_errors(m, test_input, test_target, mini_batch_size)
            print('test error Net {:0.2f}% {:d}/{:d}'.format((100 * nb_test_errors) / test_input.size(0),
                                                      nb_test_errors, test_input.size(0)))
            average_error+=(100 * nb_test_errors) / test_input.size(0)
    
    #make, save and show a graph of the mean losses and mean auxiliary losses according to the epochs
    if print_:
        print("Average error: "+str(average_error/n))
        avg_losses=torch.sum(losses,0)/n
        fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
        x_lab=torch.arange(nb_epochs)
        x_labels=x_lab.detach().numpy()
        mean=avg_losses[:,0].detach().numpy()
        std=torch.std(losses[:,:,0],0).detach().numpy()
        ax0.plot(x_labels, mean,label='Loss')
        ax0.fill_between(x_labels,mean-std,mean+std,alpha=0.2)
        mean=avg_losses[:,1].detach().numpy()
        std=torch.std(losses[:,:,1],0).detach().numpy()
        ax1.plot(x_labels, mean,label="Aux. Loss 1")
        ax1.fill_between(x_labels,mean-std,mean+std,alpha=0.2)
        mean=avg_losses[:,2].detach().numpy()
        std=torch.std(losses[:,:,2],0).detach().numpy()
        ax1.plot(x_labels, mean,label="Aux. Loss 2")
        ax1.fill_between(x_labels,mean-std,mean+std,alpha=0.2)
        ax0.set_ylabel('Mean base loss')
        ax0.legend()
        ax1.set_ylabel('Mean aux. losses')
        ax1.legend()
        ax1.set_xlabel('Number of epochs')
        plt.savefig(plot_name,bbox_inches='tight')
        plt.show()
    
    #return the errors if needed
    if store_error:
        return errors

#this function returns the number of parameters of a model
def n_params(model):
    n = 0
    for params in model.parameters():
        n += params.numel()
    return n

#this function uses the previous ones to build a big plot with several models
def big_error_plot(models,model_names,n=10,nb_epochs=20,eta=1e-3,name="big_error_plot.png",aux_factor=1):
    
    #x labels for each epoch in our plot
    x_labels=torch.arange(nb_epochs).detach().numpy()
    
    #for each model, train it n times and plot its accuracy as a function of the number of epochs it has been trained for
    for i in range(len(models)):
        print("Starting model " + model_names[i])
        errors=run_many_times(models[i],crit=nn.CrossEntropyLoss,mini_batch_size=10,n=n,print_=False,eta=eta,nb_epochs=nb_epochs,aux_factor=aux_factor,shuffle=True, store_error=True,checkpoint_name='checkpoints/'+model_names[i]+'_'+str(aux_factor).replace('.','')+'_')
        
        #Plotting
        mean=(errors.mean(dim=0)/10).detach().numpy()
        std=(torch.std(errors,dim=0)/10).detach().numpy()
        plt.plot(x_labels,mean)
        plt.fill_between(x_labels,mean-std,mean+std,alpha=0.2)
        
        #Print accuracy
        print('Mean number of errors after '+str(nb_epochs)+' epochs of training: '+str(float(errors.mean(dim=0)[-1]))+' out of 1000 test samples with a standard deviation of '+str(float(torch.std(errors,dim=0)[-1])))
        
    #Label plot, save it and show it
    plt.ylabel('Accuracy[%]')
    plt.xlabel('Number of epochs')
    plt.legend(model_names)
    plt.savefig(name,bbox_inches='tight')
    plt.show()