# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 09:48:56 2023

@author: river
"""
import sys
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


    
        