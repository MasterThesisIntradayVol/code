#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 09:56:59 2020

@author: adamtornqvist
"""
import numpy as np


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