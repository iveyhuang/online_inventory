# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 12:03:17 2023

@author: river
"""
import matplotlib.pyplot as plt
from utils.online_train import online_simulate_2
from utils.plot import plot_error_bands
import scienceplots

T = 720 # time peirod
typ = ['linear', 'polynomial', 'exponential', 'sin'] # type of the data
title = ['$g(x) = x$', '$g(x) = x^2$', '$g(x) = e^x$','$g(x) = sinx$'] # title of the picture


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
