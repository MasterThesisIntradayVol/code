#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 09:49:28 2020

@author: adamtornqvist
"""

import numpy as np
import scipy.stats as si
import pandas as pd

def Read_test():
    File = pd.read_excel('/Users/adamtornqvist/Desktop/masterthesis/code/european_options_nobbo.xlsx',sheet_name='CALL', header = 0)
    return File


def newton_vol_call_div(s, x, t, c, r, d, iv):
    
    d1 = (np.log(s / x) + (r - d + 0.5 * iv ** 2) * t) / (iv * np.sqrt(t))
    d2 = (np.log(s / x) + (r - d - 0.5 * iv ** 2) * t) / (iv * np.sqrt(t))
    
    fx = s * np.exp(-d * t) * si.norm.cdf(d1, 0.0, 1.0) - x * np.exp(-r * t) * si.norm.cdf(d2, 0.0, 1.0) - c
    
    vega = (1 / np.sqrt(2 * np.pi)) * s * np.exp(-d * t) * np.sqrt(t) * np.exp((-si.norm.cdf(d1, 0.0, 1.0) ** 2) * 0.5)
    
    tolerance = 0.000001
    x0 = iv
    xnew  = x0
    xold = x0 - 1
        
    while abs(xnew - xold) > tolerance:
    
        xold = xnew
        xnew = (xnew - fx - c) / vega
        
        return abs(xnew)
    

def newton_vol_put_div(S, K, T, P, r, q, sigma):
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    fx = K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * np.exp(-q * T) * si.norm.cdf(-d1, 0.0, 1.0) -  P
    
    vega = (1 / np.sqrt(2 * np.pi)) * S * np.exp(-q * T) * np.sqrt(T) * np.exp((-si.norm.cdf(d1, 0.0, 1.0) ** 2) * 0.5)
    
    tolerance = 0.000001
    x0 = sigma
    xnew  = x0
    xold = x0 - 1
        
    while abs(xnew - xold) > tolerance:
    
        xold = xnew
        xnew = (xnew - fx - P) / vega
        
        return abs(xnew)
    
    
def newton_vol_call(S, K, T, C, r, sigma):
    
    #S: spot price
    #K: strike price
    #T: time to maturity
    #C: Call value
    #r: interest rate
    #sigma: volatility of underlying asset
    
    d1 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    fx = S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0) - C
    
    vega = (1 / np.sqrt(2 * np.pi)) * S * np.sqrt(T) * np.exp(-(si.norm.cdf(d1, 0.0, 1.0) ** 2) * 0.5)
    
    tolerance = 0.000001
    x0 = sigma
    xnew  = x0
    xold = x0 - 1
        
    while abs(xnew - xold) > tolerance:
    
        xold = xnew
        xnew = (xnew - fx - C) / vega
        
        return abs(xnew)
    
    
    
    
def newton_vol_put(S, K, T, P, r, sigma):
    
    d1 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    fx = K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0) - P
    
    vega = (1 / np.sqrt(2 * np.pi)) * S * np.sqrt(T) * np.exp(-(si.norm.cdf(d1, 0.0, 1.0) ** 2) * 0.5)
    
    tolerance = 0.000001
    x0 = sigma
    xnew  = x0
    xold = x0 - 1
        
    while abs(xnew - xold) > tolerance:
    
        xold = xnew
        xnew = (xnew - fx - P) / vega
        
        return abs(xnew)
    
    
newton_vol_call(25, 20, 1, 7, 0.05, 0.25)

newton_vol_put_div(25, 20, 1, 7, 0.05, 0.10, 0.25)



newton_vol_call_div(25, 20, 1, 7, 0.05, 0.10, 0.25)


File = Read_test()
file = np.array(File)

start = 0
stop = 10
iv = np.zeros(stop)
r = 0.0177
for j in np.arange(start, stop):
    t = file[j,0]
    x = file[j,1]
    c= file[j,2]
    s = file[j,3]
    
    s = 25
    x = 20
    t = 1
    c = 7
    r = 0.05
    d = 0.1

    iv = 0.25
    iv[j] = newton_vol_call_div(s, x, t, c, r, d, iv)



#iv = newton_vol_call_div(s, x, t, r, c, iv)