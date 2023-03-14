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
        super().__init__()
        self.u = torch.tensor(0.)
        self.model = model
        self.perishable = perishable
        self.loss_func = loss_func
    
    def forward(self, x, demand):
        y_hat = self.model(x)
        if self.perishable:
            y = y_hat
        else:
            y = torch.maximum(y_hat, self.u)
            self.u = (F.relu(y-demand)).data
        
        loss = self.loss_func(y, demand)
        return y, loss


    
        