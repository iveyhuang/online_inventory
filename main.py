# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 12:03:17 2023

@author: river
"""
import matplotlib.pyplot as plt
from utils.online_train import online_simulate_2
import numpy as np
import scienceplots

T = 720
typ = ['linear', 'polynomial', 'exponential', 'sin']
title = ['$g(x) = x$', '$g(x) = x^2$', '$g(x) = e^x$','$g(x) = sinx$']
#%%
def log(x:np.array):
    x[x>0] = np.log(x[x>0])
    x[x<0] = -np.log(-x[x<0])
    x[x==0] = 0
    return x
    
def plot_error_bands(y, label):
    y = log(y)
    mean = y.mean(axis=0)
    std = y.std(axis=0)
    x = np.arange(len(mean))
    plt.plot(x, mean, label=label)
    plt.fill_between(x, mean-0.25*std, mean+0.25*std, alpha=0.2)

for typ,title in zip(typ,title):
    result_1 = online_simulate_2(N=1, T=T, data_type=typ)
    result_10 = online_simulate_2(N=10, T=T, data_type=typ)
    result_20 = online_simulate_2(N=20, T=T, data_type=typ)

    
    plt.style.use(['science','no-latex','grid'])
    plt.figure(figsize=(7.5, 5.625), dpi=200)
    plot_error_bands(result_1['linear'], label='linear 1')
    plot_error_bands(result_10['linear'], label='linear 10')
    plot_error_bands(result_20['linear'],label='linear 20')
    plot_error_bands(result_1['net'],label='net 1')
    plot_error_bands(result_10['net'], label='net 10')
    plot_error_bands(result_20['net'],label='net 20')
    plt.title(title)
    plt.xlabel('time $t$')
    plt.ylabel('log(regret)')
    plt.legend()
    plt.savefig('picture/{}.png'.format(typ)) 
