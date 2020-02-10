#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 12:56:37 2020

@author: adamtornqvist
"""


import numpy as np
import scipy.stats as si
import pandas as pd

def BS_call_wdiv(S, K, T, r, q, sigma):
    
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


def BS_call_wdiv_impvol(S, K, T, r, q, call):
    
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
        
        price, d1, d2 = BS_call_wdiv(S, K, T, r, q, start_vol)
        dN = 1/np.sqrt(2*np.pi) * np.exp(-0.5 * d1**2)
        V = S * np.sqrt(T) *np.exp(-q*T) * dN
        vol = start_vol + (call - price)/V
        start_vol = vol
        i += 1
        print(i)
        
    return vol

def BS_call_divcalc(S, K, T, r, call, sigma):
    
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #q: rate of continuous dividend paying asset 
    #sigma: volatility of underlying asset   
    #call: price of the call option
    
    eps = 0.0000001
    price = call + 1
    start_q = 0.1 #use real instead and add on say 0.01
    i  =0
    
    #Newton Raphson
    while np.abs((call-price)/price) > eps:
        
        price, d1, d2 = BS_call_wdiv(S, K, T, r, start_q, sigma)
        dN = 1/np.sqrt(2*np.pi) * np.exp(-0.5 * d1**2)
        V = S * np.sqrt(T) *np.exp(-start_q*T) * dN
        q = start_q - (call - price)/V
        start_q = q
        i += 1
        if i == 15:
            return q 
        
    return q

def Read_test():
    File = pd.read_excel('/Users/adamtornqvist/Desktop/masterthesis/code/AAPL_OptionTrades_Raw_IV.xlsx',sheet_name='Call_15_OCT_2019', header = 0)
    return File



#call = 18.34        
##q = 0.0097      #yfinance, unsure if it is continious
#r = 0.0177      #10 year libor from 15/10/2019
#K = 220
#T = 17/365
#S = 236.35
#sigma = 0.377

#call = 18.44       
##q = 0.0097      #yfinance, unsure if it is continious
#r = 0.0177      #10 year libor from 15/10/2019
#K = 220
#T = 0.0465753424657534
#S = 236.4
#sigma = 0.371096
#
#
    
call = 2.3     
#q = 0.0097      #yfinance, unsure if it is continious
r = 0.0177      #10 year libor from 15/10/2019
K = 250
T = 0.0849315068493151
S = 2236.390
sigma = 0.254164
q = BS_call_divcalc(S, K, T, r, call, sigma)

