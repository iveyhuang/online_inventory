# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 20:28:25 2023

@author: river
"""
import sys
sys.path.append('../')
import numpy as np
from tqdm import tqdm
from utils.loss import cost
from utils.data import DataGen
from models.model import ModelCompiler
from torch.utils.data import DataLoader
from joblib import Parallel, delayed
import torch.nn as nn
import torch.optim as optim

def train(model, optimizer, data_loader, loss_func=cost, lr_schedule=None):
    '''
    

    Parameters
    ----------
    model : custumed model
          class ModelComier.
    optimizer : torch.optim
        eg. Adam or SGD.
    data_loader : pytorch data_loader
        data_loader will return (x, D).
    loss_func : pytho function, optional
        loss function. The default is cost.
    lr_schedule : TYPE, optional
        learning rate schedule. The default is None.

    Returns
    -------
    result : TYPE
        DESCRIPTION.

    '''
    Y = []
    Regret = []
    Mean = []
    result = {}

    for x, demand, optimal_y in data_loader:
        '''
        b: batch_size = 1
        N: deatures
        
        x shape (b, N)
        demand shape (b,1)
        optimal_y shape (b,1)
        '''
        
        y, loss = model(x, demand)
        loss_optimal = loss_func(optimal_y, demand)
        if model.training:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        if lr_schedule is not None:
            lr_schedule.step()
        
        regret = loss - loss_optimal
        
        Y.append(y.squeeze().item())
        Regret.append(regret.squeeze().item())
        Mean.append(np.array(Regret).mean())
        
        result['y'] = Y
        result['regret'] = Regret
        result['regret_mean'] = Mean
    return result
       
def online_simulate(N, data_type, N_all=20, T=2000, num_sample=200, lr_schedule=None, cpu=-1):
    '''
    

    Parameters
    ----------
    N : int
        the number of feature available.
    data_type : str
        type of the data.
    N_all : int, optional
        total number of the features. The default is 20.
    T : int, optional
        time peirod. The default is 2000.
    num_sample : int, optional
        number of the sample. The default is 200.
    lr_schedule : pytorch learning rate schedule, optional
        learning rate schedule. The default is None.
    cpu : int, optional
        if cpu = -1, use all threads
        if cpu = -2, use all threads except one
        if cpu = 0, use one threads. The default is -1.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    data = DataGen(N=N, T=T, num_sample=num_sample)
    dataset = data.load_data(data_type=data_type)
    
    result_linear = list()
    result_net = list()
    
    if cpu == 0:
        for i in tqdm(range(num_sample)):
            model_linear = ModelCompiler(nn.Linear(in_features=N_all, out_features=1))
            model_net = ModelCompiler(nn.Sequential(
                nn.Linear(in_features=N_all, out_features=10),
                nn.ReLU(),
                nn.Linear(in_features=10, out_features=1)       
                ))
            
            model_linear.train()
            model_net.train()
        
            optimizer_linear = optim.SGD(model_linear.parameters(), lr=1e-2)
            optimizer_net = optim.SGD(model_net.parameters(), lr=1e-2)
            
            result_linear.append(train(model_linear, optimizer_linear,  DataLoader(dataset[i]))['regret_mean'])
            result_net.append(train(model_net, optimizer_net,  DataLoader(dataset[i]))['regret_mean'])
    else:
        def func_linear(i):
            linear = nn.Linear(in_features=N_all, out_features=1)
            # nn.init.uniform_(linear.weight.data,1,10)
            model_linear = ModelCompiler(linear)            
            model_linear.train()    
            optimizer_linear = optim.SGD(model_linear.parameters(), lr=1e-2)
            return train(model_linear, optimizer_linear, DataLoader(dataset[i]))['regret_mean']
        
        def func_net(i):
            model_net = ModelCompiler(nn.Sequential(
                nn.Linear(in_features=N_all, out_features=10),
                nn.ReLU(),
                nn.Linear(in_features=10, out_features=1)       
                ))
            model_net.train()
            optimizer_net = optim.SGD(model_net.parameters(), lr=1e-2)
            return train(model_net, optimizer_net,  DataLoader(dataset[i]))['regret_mean']
        
        
        result_linear = Parallel(n_jobs=cpu)(delayed(func_linear)(index) for index in tqdm(range(num_sample)))
        result_net = Parallel(n_jobs=cpu)(delayed(func_net)(index) for index in tqdm(range(num_sample)))
    result_linear = np.array(result_linear)
    result_net = np.array(result_net)
    return {'linear': result_linear, 'net': result_net}      
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
