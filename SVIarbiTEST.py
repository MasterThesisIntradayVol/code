#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 14:24:06 2020

@author: adamtornqvist
"""

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
import numpy as np
import scipy.interpolate as interp
import scipy.optimize as optimization

# =============================================================================
#           Some test data 
# =============================================================================

data = pd.read_csv("/Users/adamtornqvist/Desktop/masterthesis/JONAS_DATA/COPPER_LME_2018.csv")

data['total implied variance'] = ((data['implied volatility']/100.0)**2) * (data['Time to maturity']/365.0)

#timetomat = data['Time to maturity']
#timetomat = np.array(timetomat)
#timetomat = np.unique(timetomat)

data['log-moneyness'] = np.log(data['underlying price']/data['strike']) 

day = data['trade date']
day = np.array(day)
day = np.unique(day)

def SVI(x,a,b,p,m,sigma):
    svi = a + b * (p * (x - m) + np.sqrt((x-m)**2 + sigma**2))
    return svi 

def SVIderivate(x,a,b,p,m,sigma):
    SVIder = b * (((x-m)/np.sqrt((x-m)**2+sigma**2))+p)
    return SVIder


def SVIderivateder(x,a,b,p,m,sigma):
    SVIderder = (b * sigma**2) / ((m**2 - 2*m*x + sigma**2 + x**2)**(3.0/2))
    return SVIderder

def butterfly(x,a,b,p,m,sigma):
    svi = a + b * (p * (x - m) + np.sqrt((x-m)**2 + sigma**2))
    SVIder = b * (((x-m)/np.sqrt((x-m)**2+sigma**2))+p)
    SVIderder = (b * sigma**2) / ((m**2 - 2*m*x + sigma**2 + x**2)**(3.0/2))
    butterfly = (1 - x * SVIder / svi)**2 - 1/4 * svi**2 * SVIder**2 + svi * SVIderder
    return butterfly


idx = 0
idx2 = 0
arbicheck_list = np.zeros(np.size(day))
lb = np.array([-1.0, -1.0, -1.0, -1.0, -1.0])
ub = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    

#for i in np.arange(np.size(day)):
for i in np.arange(1,2):
    
    data1 = data.loc[data['trade date'] == day[i],:]
    timetomat = data1['Time to maturity']
    timetomat = np.array(timetomat)
    timetomat = np.unique(timetomat)
    ivt_list = np.zeros((200,np.size(timetomat)))
    g_list = np.zeros((200,np.size(timetomat)))
    

#    for j in np.arange(np.size(timetomat)):
    for j in np.arange(117,118):
        
        data2 = data1.loc[data1['Time to maturity'] == timetomat[j],:]
        ttm = np.array(data2['Time to maturity'])
        lm = np.array(data2['log-moneyness'])
        iv = np.array(data2['total implied variance'])
        
        x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        sigma = np.ones(np.size(iv))
        opt   = optimization.curve_fit(SVI, lm, iv, x0, sigma, bounds = (lb, ub))
        opt = opt[0]
        x = np.arange(-1,1,0.01)
        ivt = SVI(x,*opt)
        ivt_list[:,idx] = ivt
        g = butterfly(x,*opt)
        g_list[:,idx] = g

        idx +=1 
        
    ivt_list = pd.DataFrame(ivt_list)
    g_list = pd.DataFrame(g_list)
    diff = ivt_list.diff(axis =1)
    diff = np.array(diff)
    diff = diff[:,1:np.size(timetomat)]
    arbicheck = sum((diff < 0) *1)
    arbicheck_list[idx2] = sum(arbicheck)
    idx2 +=1
    idx = 0 
    print(idx2)
    # if idx2 == 240:
    #     ivt_list.plot(figsize=(15,15))
    #     g_list.plot(figsize=(15,15))




# ivt_list = pd.DataFrame(ivt_list)
# g_list = pd.DataFrame(g_list)
# ivt_list.plot(figsize=(15,15))
# g_list.plot(figsize=(15,15))

# diff = ivt_list.diff(axis =1)
# diff = np.array(diff)
# diff = diff[:,1:123]
# hej = sum((diff < 0) *1)
# hej = sum(hej)


# testing = ivt_list.iloc[:,0]
# testing = np.array(testing)

x = np.arange(-1,1,0.01)
plt.plot(x,ivt)
plt.scatter(lm,iv)
