# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 09:48:56 2023

@author: river
"""
import sys
import numpy as np
sys.path.append('../')

from utils.loss import cost
import torch.nn as nn
import torch.nn.functional as F 
import torch

class ModelCompiler(nn.Module):
    def __init__(self, model, loss_func=cost, perishable=False):
        '''
        

        Parameters
        ----------
        model : pytorch models
            eg: nn.Linear.
        loss_func : loss function, optional
             The default is cost.
        perishable :bool, optional
             perishabe or not. The default is False.

        Returns
        -------
        None.

        '''
        super().__init__()
        self.u = torch.tensor(0.)
        self.model = model
        self.perishable = perishable
        self.loss_func = loss_func
    
    def forward(self, x, demand):
        '''
        

        Parameters
        ----------
        x : features
            shape: (1, N).
        demand : demand
            shape: (1, 1).

        Returns
        -------
        y : predict y
            if perishable, y = y_hat
            if unperishable, y = max(y_hat, u).
        loss : TYPE
            loss function
            h*[y-D]^{+} + b*[D-y]^{+}.

        '''
        y_hat = self.model(x)
        if self.perishable:
            y = y_hat # without contraints 
        else:
            y = torch.maximum(y_hat, self.u) # y = max(y_hat, u)
            self.u = (F.relu(y-demand)).data # update u 
        
        loss = self.loss_func(y, demand)
        return y, loss

def relu(x):
    return(np.maximum(0, x))

class FAI:
    def __init__(self, 
                 x:np.array, 
                 demand:np.array,
                 theta:float,
                 perishable=False,
                 h=1,b=3,
                 z_range=[0, 1],
                 omega=[1,10]):
        '''
        

        Parameters
        ----------
        x : np.array
            x features shape:(T,N).
        demand : np.array
            demand shape:(T,1).
        perishable : TYPE, optional
            perishable or not. The default is False.
        h : TYPE, optional
            holding cost. The default is 1.
        b : TYPE, optional
            lost sales cost. The default is 3.
        theta : TYPE, optional
            a parameter influences learning rate. The default is 1.
        omega : TYPE, optional
            a list contains the upper and lower bound of the omega. The default is [0,1].

        Returns
        -------
        None.

        '''
        
        T,N = x.shape
        z_low, z_high = z_range
        self.omega_low, self.omega_high = omega
        
        self.h = h
        self.b = b
        
        self.t = 0
        self.end = T
        self.perishable = perishable
        
        self.u = 0
        self.y = 0
        self.y_hat = 0
        
        self.x = x # shape: (T,N)
        self.demand = demand # shape: (T,1)
        self.z = np.random.uniform(low=z_low, high=z_high, size=(1,N)) # shape: (1,N)
        self.epsilon = 1/((h+b)*theta)
        
        
    
    def __iter__(self):
        return self
    
    def __next__(self):
        '''
        

        Raises
        ------
        StopIteration
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        if self.t < self.end:
            if self.t == 0:
                # self.z = self.projection(self.z, self.omega_low, self.omega_high) # projection
                self.y_hat = np.dot(self.z,self.x[self.t].reshape(-1,1))
                self.y = self.y_hat
                self.t +=1
                return self.y.flatten()
            else:
                if not self.perishable:
                    self.u = relu(self.y-self.demand[self.t-1])
                if self.y_hat-self.demand[self.t-1] > 0:
                    gradient = self.h
                else:
                    gradient = -self.b
                self.z -= (self.epsilon/self.t)*gradient*self.x[self.t-1] # shape: (1,N)
                # self.z = self.projection(self.z, self.omega_low, self.omega_high) # projection
                self.y_hat = np.dot(self.z, self.x[self.t].T)
                self.y = np.maximum(self.y_hat, self.u)
                self.t +=1
                return self.y.flatten()     
        else:
            raise StopIteration
    
    def projection(self, x, low, high):
        x[x<low] = low
        x[x>high] = high
        return x
        
