from torch import nn
from torch import optim
from torch import Tensor
from torch.nn import functional as F
from models import *
from various_data_functions import *
import torch

#Setting plot parameters
plt.rcParams['figure.figsize'] = [5, 3]

print('Models without weight sharing')
models=[C1L2,C1L3,C1L5,C2L2,C2L3,C3L2,C4L2,C4L2_bis]
model_names=['C1L2','C1L3','C1L5','C2L2','C2L3','C3L2','C4L2','C4L2_bis']

for i in range(len(models)):
    m=models[i]()
    print('Number of parameter in model '+model_names[i]+' : '+str(n_params(m)))
    
print('Models with weight sharing')
models=[C1L2WS,C1L3WS,C1L5WS,C2L2WS,C2L3WS,C3L2WS,C4L2WS,C4L2WS_bis]
model_names=['C1L2WS','C1L3WS','C1L5WS','C2L2WS','C2L3WS','C3L2WS','C4L2WS','C4L2WS_bis']
for i in range(len(models)):
    m=models[i]()
    print('Number of parameter in model '+model_names[i]+' : '+str(n_params(m)))
    
#number of runs of each model for the plots
num_runs=2
print('Number of runs of each model in the plots: '+str(num_runs))

#number of epochs in each run
nb_epochs=20
print('Number of epochs for each run of each model in the plots: '+str(nb_epochs))

#Plotting accuracy for all models in the report WITHOUT weight sharing and auxiliary factor 1
models=[C1L2,C2L2,C2L3,C3L2,C4L2]
model_names=['C1L2','C2L2','C2L3','C3L2','C4L2']
big_error_plot(models,model_names,n=num_runs,nb_epochs=nb_epochs,name='big_plot_aux=1.png')

#Same with auxiliary factor 0.1
big_error_plot(models,model_names,n=num_runs,nb_epochs=nb_epochs,aux_factor=0.1,name='big_plot_aux=01.png')

#Same with auxiliary factor 0
big_error_plot(models,model_names,n=num_runs,nb_epochs=nb_epochs,aux_factor=0,name='big_plot_aux=0.png')

#Plotting accuracy for all models in the report WITH weight sharing and auxiliary factor 1
models=[C1L2WS,C2L2WS,C2L3WS,C3L2WS,C4L2WS]
model_names=['C1L2WS','C2L2WS','C2L3WS','C3L2WS','C4L2WS']
big_error_plot(models,model_names,n=num_runs,nb_epochs=nb_epochs,name='big_plotWS_aux=1.png')

#Same with auxiliary factor 0.1
big_error_plot(models,model_names,n=num_runs,nb_epochs=nb_epochs,aux_factor=0.1,name='big_plotWS_aux=01.png')

#Same with auxiliary factor 0
big_error_plot(models,model_names,n=num_runs,nb_epochs=nb_epochs,aux_factor=0,name='big_plotWS_aux=0.png')

#2ND PLOT IN REPORT
#Plot evolution of loss and auxiliary loss
print('Running instances of C4L2WS to plot losses')
model=C4L2WS
run_many_times(model,crit=nn.CrossEntropyLoss,mini_batch_size=10,n=num_runs,print_=True,eta=1e-3,nb_epochs=nb_epochs,aux_factor=0.1,shuffle=True, store_error=False,checkpoint_name=None,plot_name='losses.png')

#1ST PLOT IN REPORT
#Plot each settings of a single model
eta=1e-3
name="C4L2_plot.png"
model=C4L2
model_name='C4L2'
x_lab=torch.arange(nb_epochs)
x_labels=x_lab.detach().numpy()
print('starting the no weight sharing part')
aux_factors=[0,0.1]
for aux_factor in aux_factors:
    print('aux factor = '+str(aux_factor))
    errors=run_many_times(model,crit=nn.CrossEntropyLoss,mini_batch_size=10,n=num_runs,print_=False,eta=eta,nb_epochs=nb_epochs,aux_factor=aux_factor,shuffle=True, store_error=True,checkpoint_name='checkpoints/'+model_name+'_'+str(aux_factor).replace('.','')+'_')
    mean=(errors.mean(dim=0)/10).detach().numpy()
    std=(torch.std(errors,dim=0)/10).detach().numpy()
    plt.plot(x_labels,mean)
    plt.fill_between(x_labels,mean-std,mean+std,alpha=0.1)
    
print('starting the weight sharing part')
model=C4L2WS
model_name=str(model_name)+'WS'
for aux_factor in aux_factors:
    print('aux factor = '+str(aux_factor))
    errors=run_many_times(model,crit=nn.CrossEntropyLoss,mini_batch_size=10,n=num_runs,print_=False,eta=eta,nb_epochs=nb_epochs,aux_factor=aux_factor,shuffle=True, store_error=True,checkpoint_name='checkpoints/'+model_name+'_'+str(aux_factor).replace('.','')+'_')
    mean=(errors.mean(dim=0)/10).detach().numpy()
    std=(torch.std(errors,dim=0)/10).detach().numpy()
    plt.plot(x_labels,mean)
    plt.fill_between(x_labels,mean-std,mean+std,alpha=0.1)
plt.ylabel('Error percentage [%]')
plt.xlabel('Number of epochs')
plt.legend(['No WS 0','No WS 0.1','WS 0','WS 0.1'])
plt.savefig(name,bbox_inches='tight')
plt.show()