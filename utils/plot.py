# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 19:29:53 2023

@author: river
"""
import numpy as np
import matplotlib.pyplot as plt

def log(x:np.array):
    '''
    

    Parameters
    ----------
    x : np.array
        x: regret shape (num_sample, T).

    Returns
    -------
    x : TYPE
        if x > 0, return log(x)
        if x = 0 return 0
        if x < 0 return -log(x).

    '''
    x[x>0] = np.log(x[x>0])
    x[x<0] = -np.log(-x[x<0])
    x[x==0] = 0
    return x
    
def plot_error_bands(y:np.array, label:str):
    '''
    

    Parameters
    ----------
    y : np.array
        axis-y.
    label : str
        label of the plot.

    Returns
    -------
    None.

    '''
    y = log(y)
    mean = y.mean(axis=0)
    std = y.std(axis=0)
    x = np.arange(len(mean))
    plt.plot(x, mean, label=label)
    plt.fill_between(x, mean-0.25*std, mean+0.25*std, alpha=0.2)