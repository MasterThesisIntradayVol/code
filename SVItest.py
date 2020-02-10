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

#data['log-moneyness'] = np.log(data['underlying price']/data['strike']) 


#Test for to TTM
time100 = data.loc[data['Time to maturity'] == 100,:]
time9   = data.loc[data['Time to maturity'] == 9,:]
time65   = data.loc[data['Time to maturity'] == 65,:]


time100 = time100.loc[time100['trade date'] == "2018-05-28",:]
time9   = time9.loc[time9['trade date'] == "2018-05-28",:]
time65   = time65.loc[time65['trade date'] == "2018-05-28",:]

time100['log-moneyness'] = np.log(time100['underlying price']/time100['strike'])
time9['log-moneyness'] = np.log(time9['underlying price']/time9['strike'])
time65['log-moneyness'] = np.log(time65['underlying price']/time65['strike'])

#time910.plot(kind='scatter',x='hej',y='implied volatility',color='red')


# =============================================================================
#           3d - ploting 
# =============================================================================


#fig = plt.figure()
#ax = plt.axes(projection='3d')
#
#ax = plt.axes(projection='3d')

# Data for a three-dimensional line
#y = np.array(time91['Time to maturity'])
#x = np.array(time91['log-moneyness'])
#z = np.array(time91['implied volatility'])

#zline = np.linspace(0, 15, 1000)
#xline = np.sin(zline)
#yline = np.cos(zline)
#ax.plot3D(xline, yline, zline, 'gray')
#
#
## Data for three-dimensional scattered points
#zdata = 15 * np.random.random(100)
#xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
#ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
#ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens');
#
#
#
#X = np.array(time91['Time to maturity'])
#Y = np.array(time91['Moneyness. % away from forward'])
#Z = np.array(time91['implied volatility'])
#
#plotx,ploty, = np.meshgrid(np.linspace(np.min(X),np.max(X),100),\
#                           np.linspace(np.min(Y),np.max(Y),100))
#plotz = interp.griddata((X,Y),Z,(plotx,ploty),method='cubic')
#
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(plotx,ploty,plotz,cstride=1,rstride=1,cmap='viridis')




# =============================================================================
# 
# =============================================================================
# =============================================================================
#               SVI Testing
# =============================================================================




#Data trades for one day with one maturity
ttm9 = np.array(time9['Time to maturity'])
lm9 = np.array(time9['log-moneyness'])
iv9 = np.array(time9['total implied variance'])

ttm100 = np.array(time100['Time to maturity'])
lm100 = np.array(time100['log-moneyness'])
iv100 = np.array(time100['total implied variance'])

ttm65 = np.array(time65['Time to maturity'])
lm65 = np.array(time65['log-moneyness'])
iv65 = np.array(time65['total implied variance'])

#Function
def SVI(x,a,b,p,m,sigma):
    svi = a + b * (p * (x - m) + np.sqrt((x-m)**2 + sigma**2))
    return svi 
    
#Initial guess
x0    = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
#Not sure what this is, not the same as sigma in SVI
sigma = np.ones(np.size(iv9))

#Optimization curve SVI
hej9   = optimization.curve_fit(SVI, lm9, iv9, x0, sigma)
hej100 = optimization.curve_fit(SVI, lm100, iv100, x0, sigma)
hej65 = optimization.curve_fit(SVI, lm65, iv65, x0, sigma)

#Get parameters
hej9 = hej9[0]
hej100 = hej100[0]
hej65 = hej65[0]

#Getting dimensions for plot
imp9 = np.zeros(20)
imp100 = np.zeros(20)
imp65 = np.zeros(20)
idx = 0 
for i in np.arange(-1,1,0.1):
    imp9[idx] = SVI(i, *hej9)
    imp100[idx] = SVI(i, *hej100)
    imp65[idx] = SVI(i, *hej65)
    idx +=1 
    
    
#Plot
x = np.arange(-1,1,0.1)
plt.plot(x,imp9)
plt.scatter(lm9,iv9)
plt.plot(x,imp100)
plt.scatter(lm100,iv100)
plt.plot(x,imp65)
plt.scatter(lm65,iv65)
    
