# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 11:25:54 2017

@author: bened
"""
#Test robustness of the spatial method against noise
import numpy as np
import scipy as sp
import scipy.ndimage
import matplotlib.pyplot as plt
import time
import copy
##Print full list/array
np.set_printoptions(threshold=np.nan)

##step-like peaks of width 0.2
def peak(x):
    return ((x > -0.1) & (x < 0.1)).astype(float)

n = 64
trueshift = 0
noise = np.linspace(0.05,0.6,60)
x = np.linspace(-1,2,n)

def shifterror(noise):
    p1 = peak(x) + peak(x-1) + np.random.normal(0,noise,n)
    p2 = peak(x-trueshift/(n-1)*3) + peak(x-1-trueshift/(n-1)*3) + np.random.normal(0,noise,n)
    R = np.correlate(p1,p2, mode = 'full')
    shift = np.argmax(R)-n+1
    return shift+trueshift

def shifterrormean(noise,k):
    list = [np.abs(shifterror(noise)) for i in range(k)]
    return np.mean(list)
t0 = time.time()
errorlist = [shifterrormean(i,20) for i in noise]
t1 = time.time()
total = t1-t0
plt.plot(noise, errorlist, label='Spatial method')
plt.plot(noise, errorlist2, label='Frequency method')
plt.legend(loc='upper left')
plt.title('Shift error')
plt.xlabel('Gaussian noise (SD)')
plt.ylabel('Error (fine scale pixels)')
plt.show()
print(total)

plt.plot(x,p1)
plt.plot(x,p2)
plt.show()