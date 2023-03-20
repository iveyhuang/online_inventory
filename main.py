# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 12:03:17 2023

@author: river
"""
import matplotlib.pyplot as plt
from utils.online_train import auto_simulate, manual_simluate
from utils.plot import plot_error_bands
import scienceplots

T = 2000 # time peirod
typ = ['linear', 'polynomial', 'exponential', 'sin'] # type of the data
title = ['$g(x) = x$', '$g(x) = x^2$', '$g(x) = e^x$','$g(x) = sinx$'] # title of the picture

'''
auto_simulate use torch auto_grad

manual_simulate use numpy grad manually
'''

# for typ,title in zip(typ,title):
#     result_1 = auto_simulate(N=1, T=T, data_type=typ)
#     result_10 = auto_simulate(N=10, T=T, data_type=typ)
#     result_20 = auto_simulate(N=20, T=T, data_type=typ)
#     plt.style.use(['science','no-latex','grid'])
#     plt.figure(figsize=(7.5, 5.625), dpi=200)
#     plot_error_bands(result_1['linear'], label='linear 1')
#     plot_error_bands(result_10['linear'], label='linear 10')
#     plot_error_bands(result_20['linear'],label='linear 20')
#     plot_error_bands(result_1['net'],label='net 1')
#     plot_error_bands(result_10['net'], label='net 10')
#     plot_error_bands(result_20['net'],label='net 20')
#     plt.title(title)
#     plt.xlabel('log(time) $t$')
#     plt.ylabel('log(regret)')
#     plt.legend()
#     plt.savefig('picture/{}.png'.format(typ)) 

for typ,title in zip(typ,title):
    result_1 = manual_simluate(N=1, T=T, data_type=typ, z_range=[25, 300], omega=[50, 250])
    result_10 = manual_simluate(N=10, T=T, data_type=typ, z_range=[1, 150], omega=[25, 125])
    result_20 = manual_simluate(N=20, T=T, data_type=typ, z_range=[1, 20], omega=[1, 10])   
    plt.style.use(['science','no-latex','grid'])
    plt.figure(figsize=(7.5, 5.625), dpi=200)
    plot_error_bands(result_1, label='linear 1')
    plot_error_bands(result_10, label='linear 10')
    plot_error_bands(result_20,label='linear 20')
    plt.title(title)
    plt.xlabel('log(time) $t$')
    plt.ylabel('log(regret)')
    plt.legend()
    plt.savefig('picture/{}_1.png'.format(typ)) 