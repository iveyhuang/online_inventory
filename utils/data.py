# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 10:57:50 2023

@author: river
"""
import numpy as np
from tqdm import tqdm
from statistics import NormalDist
from torch.utils.data import Dataset
from scipy.stats import truncnorm
import torch

initial_pram = {
    'h':1,
    'b':3,
    'x_low':1,
    'x_high':2,
    'omega_low':1,
    'omega_high':10,
    'loc': 0, # mean of the delta
    'scale': 40**(0.5), # std of the delta
    'delta_low': -3*(40**(0.5)), # -3*std
    'delta_high': 3*(40**(0.5)) # 3*std
    }



#%%


class DataGen:
    def __init__(self, N:int, T:int, num_sample:int, N_all:int=20):
        self.N = N
        self.N_all = N_all
        self.T = T
        self.num_sample = num_sample
        
    def prepare_data(self, data_type:str='linear', power:int=2, prams=initial_pram):
        '''
        

        Parameters
        ----------
        D_t = g(\omega \times x_t) + delta_t
        
        num_sample : int
            number of sample path.
        data_type : str, optional
            linear, polynomial, exponential. The default is 'linear'.
        power : int, optional
            power of the g. The default is 2.
        prams : TYPE, optional
            prameters. The default is initial_pram.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        dict
            DESCRIPTION.

        '''
        
        print('-----Begin Simulation-----')
        b,h,x_low,x_high,omega_low,omega_high,loc,scale, delta_low, delta_high = prams.values()
        N, N_all, T, num_sample = self.N, self.N_all,self.T, self.num_sample
        
        
        x = np.random.uniform(low=x_low, high=x_high, size=(num_sample,T,N_all)) # shape (num_sample, T, N)
        x[:,:,0] = 1
        omega = np.random.uniform(low=omega_low, high=omega_high, size=(num_sample, 1, N_all)) # shape (num_sample, 1, N)
        omega_x = np.stack([np.dot(x[i], omega[i].T) for i in tqdm(range(num_sample), desc = data_type)], axis=0) # shape (num_sample, T, 1)
        
        g_x_linear = omega_x
        g_x_polynomial = omega_x**power
        g_x_exp = np.exp(omega_x)
        g_x_sin = np.sin(omega_x)
        
        # adjust the mean and std equal to the g_x_linear
        g_x_polynomial *= g_x_linear.std()/g_x_polynomial.std()
        g_x_exp *= g_x_linear.std()/g_x_exp.std()
        g_x_sin *= g_x_linear.std()/g_x_sin.std()
        
        g_x_exp += g_x_linear.mean()-g_x_exp.mean()
        g_x_polynomial += g_x_linear.mean() - g_x_polynomial.mean()
        g_x_sin += g_x_linear.mean() - g_x_sin.mean()
        
        a, b = (delta_low - loc) / scale, (delta_high - loc) / scale
        delta = truncnorm.rvs(a, b, loc, scale, size=(num_sample, T, 1))
        theta = truncnorm.pdf(delta, a, b, loc, scale).min()
        
        if data_type == 'linear':
            g_x = g_x_linear
        elif data_type == 'polynomial':
            g_x = g_x_polynomial
        elif data_type == 'exponential':
            g_x = g_x_exp
        elif data_type == 'sin':
            g_x = g_x_sin
        else:
            raise ValueError("data type must be linear, polynomial, trigonometic and exponential!")
        print('\n-----End Simulation-----\n')
        # g_x += scale
        D = g_x + delta
        optimal_y = g_x + truncnorm.ppf(b/(b+h), a, b, loc, scale)
        
        '''
        if N<N_all, we set x[:,:,N:] = 0
        for example:
            N = 1, N_all = 20,
            x = [1, 0, 0...0]
        '''
        if N < N_all:
            x[:,:,N:] = 0
            
        return {'demand':D, 'optimal_y':optimal_y, 'x':x, 'theta': theta}
    
    def load_data(self, data_type='linear'):
        '''
        

        Parameters
        ----------
        data_type : TYPE, optional
            type of the data. The default is 'linear'.

        Returns
        -------
        dataset : pytorch dataset
            (x, D).

        '''
        data_dict = self.prepare_data(data_type=data_type)
        dataset = []
        for i in range(self.num_sample):
            data = ModelDataset(demand=data_dict['demand'][i], 
                                      x=data_dict['x'][i], 
                                      optimal_y=data_dict['optimal_y'][i])
            dataset.append(data)
        return dataset
    
class ModelDataset(Dataset):
    def __init__(self, demand:np.array, x:np.array, optimal_y:np.array):
        '''
        

        Parameters
        ----------
        demand : np.array
            demand shape (T, 1).
        x : np.array
            x features shape (T, N).
        optimal_y : np.array
            optimal y shape (T, 1).

        Returns
        -------
        None.

        '''
        super().__init__()
        self.demand = demand 
        self.x = x
        self.optimal_y = optimal_y
    
    def __getitem__(self, index):
        demand = torch.tensor(self.demand[index,:], dtype=torch.float32)
        x = torch.tensor(self.x[index,:], dtype=torch.float32)
        optimal_y = torch.tensor(self.optimal_y[index,:], dtype=torch.float32)
        return x, demand, optimal_y
    
    def __len__(self):
        return self.demand.shape[0]
        

def cost(stretagy, demand, h=1, b=3, return_bios = False):
    '''
    Parameters
    ----------
    stretagy : numpy array
       your own inventary strategy.
    demand : numpy array
        optimal array.
    h : float, optional
        holding cost. The default is 1.
    b : float, optional
        lost sales cost. The default is 3.
    return_bios: Boolean, optional
        return bios or not. The default is False.

    Returns
    -------
    dict
        bios: stretagy-demand at t.
        cost of very sample data
    '''
    bios = np.stack([stretagy[i]-demand[i] for i in range(stretagy.shape[0])], axis=0)
    bios[bios>=0] *= h
    bios[bios<0] *= -b
    if return_bios:
        return {'bios':bios, 'cost':bios.mean(axis=1)}
    else:
        return bios.mean(axis=1)
