#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 12:56:37 2020

@author: adamtornqvist
"""


import numpy as np
import scipy.stats as si

def BS_call_div(S, K, T, r, q, sigma):
    
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #q: rate of continuous dividend paying asset 
    #sigma: volatility of underlying asset
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    call = (S * np.exp(-q * T) * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    
    return call, d1, d2


def BS_call_div_impvol(S, K, T, r, q, call):
    
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #q: rate of continuous dividend paying asset 
    #sigma: volatility of underlying asset   
    #call: price of the call option
    
    eps = 0.0000001
    price = call + 1000
    start_vol = np.sqrt(2*np.pi/T) * (call/S)
    i  =0
    
    #Newton Raphson
    while np.abs((call-price)/price) > eps:
        
        price, d1, d2 = BS_call_div(S, K, T, r, q, start_vol)
        N = 1/np.sqrt(2*np.pi) * np.exp(-0.5 * d1**2)
        vega = S * np.sqrt(T) *np.exp(-q*T) * N
        vol = start_vol + (call - price)/vega
        start_vol = vol
        i += 1
        print(i)
        
    return vol

call = 18.34        
q = 0.0097      #yfinance, unsure if it is continious
r = 0.0177      #10 year libor from 15/10/2019
K = 220
T = 17/365
S = 236.35

vol = BS_call_div_impvol(S, K, T, r, q, call)
