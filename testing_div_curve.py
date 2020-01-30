#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 10:47:19 2020

@author: adamtornqvist
"""

import defBS as BS
import numpy as np
import scipy.stats as si
import pandas as pd

File = BS.Read_test()

start = 0
stop = 10
q_curve = np.zeros(stop)
r = 0.0177

for j in np.arange(start, stop):
    T = File.loc[j,'T']
    K = File.loc[j,'K']
    price = File.loc[j,'Price']
    S = File.loc[j,'S']
    sigma = File.loc[j,'IV']
    
    q_curve[j] = BS.BS_call_divcalc(S,K,T,r,price,sigma)
