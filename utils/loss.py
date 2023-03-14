# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 23:26:02 2023

@author: river
"""
import torch
import torch.nn.functional as F

def cost(y:torch.tensor, demand:torch.tensor, h=1, b=3):
    '''
    

    Parameters
    ----------
    y : torch.tensor
        shape (1, 1).
    demand : torch.tensor
        shape (1, 1).
    h : TYPE, optional
        holding cost. The default is 1.
    b : TYPE, optional
        lost sales cost. The default is 3.

    Returns
    -------
    cost : TYPE
        cost of this period.

    '''
    cost = h*F.relu(y-demand) + b*F.relu(demand-y) # shape (1, 1)
    return cost

