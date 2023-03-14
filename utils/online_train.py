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




def model_train(model, optimizer, dataset, lr_schedule=None, loss_func=cost, perishable=False, cpu=0):
    model = ModelCompiler(model, loss_func=loss_func, perishable=perishable)
    model.train()
    if cpu == 0:
        result = np.array([train(model, optimizer,  DataLoader(dataset[i]))['regret_mean'] for i in tqdm(range(len(dataset)))])
    else:
        def func(index):
            return train(model, optimizer,  DataLoader(dataset[index]))['regret_mean']
        result = np.array(Parallel(n_jobs=cpu)(delayed(func)(index) for index in tqdm(range(len(dataset)))))  
    return result
    
def online_simulate(N, data_type, T=2000, num_sample=200, lr_schedule=None, cpu=-1):
    data = DataGen(N=N, T=T, num_sample=num_sample)
    dataset = data.load_data(data_type=data_type)
    
    model_linear = nn.Linear(in_features=N, out_features=1)
    model_net = nn.Sequential(
        nn.Linear(in_features=N, out_features=10),
        nn.ReLU(),
        nn.Linear(in_features=10, out_features=1)       
        )
    
    optimizer_linear = optim.SGD(model_linear.parameters(), lr=1e-2)
    optimizer_net = optim.SGD(model_net.parameters(), lr=1e-2)
    
    result_linear = model_train(model=model_linear, 
                                optimizer=optimizer_linear, 
                                dataset=dataset,
                                lr_schedule=lr_schedule,
                                cpu=cpu)
    
    result_net = model_train(model=model_net, 
                             optimizer=optimizer_net, 
                             dataset=dataset,
                             lr_schedule=lr_schedule,
                             cpu=cpu)
    
    return {'linear': result_linear, 'net': result_net}
       
def online_simulate_2(N, data_type, N_all=20, T=2000, num_sample=200, lr_schedule=None, cpu=-1):
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
            model_linear = ModelCompiler(nn.Linear(in_features=N_all, out_features=1))            
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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
