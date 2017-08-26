# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 17:27:48 2017

@author: bened
"""

#Test robustness of the frequency method against noise
import numpy as np
import scipy as sp
import scipy.ndimage
import matplotlib.pyplot as plt
import copy
import time

##Print full list/array
np.set_printoptions(threshold=np.nan)

##step-like peaks of width 0.2
def peak(x):
    return ((x > -0.1) & (x < 0.1)).astype(float)

n = 64
trueshift = 12
noise = np.linspace(0.05,0.6,10)
x = np.linspace(-1,2,n)
frqcutoff = 64
q = range(20)

def shifterror(noise):
    p1 = peak(x) + peak(x-1) + np.random.normal(0,noise,n)
    p2 = peak(x-trueshift/(n-1)*3) + peak(x-1-trueshift/(n-1)*3) + np.random.normal(0,noise,n)
    p1ft = np.fft.fft(p1)[:frqcutoff]
    p2ft = np.fft.fft(p2)[:frqcutoff]
    #b = sp.optimize.fmin(lambda p: np.linalg.norm(np.array([np.exp(2j*np.pi/n*k*p)*p2ft[k] - p1ft[k] for k in range(frqcutoff)])), 0, disp = False)
    #return np.round(trueshift+b)
    shift = -np.argmin([np.linalg.norm(np.array([np.exp(2j*np.pi/n*k*(i))*p2ft[k] - p1ft[k] for k in range(frqcutoff)])) for i in q])
    return shift+trueshift
    
def shifterrormean(noise,k):
    list = [np.abs(shifterror(noise)) for i in range(k)]
    return np.mean(list)
t0 = time.time()
errorlist = [shifterrormean(i,20) for i in noise]
t1 = time.time()
total = t1-t0
plt.plot(noise, errorlist)
plt.title('Shift error (frequency method)')
plt.xlabel('Gaussian noise (SD)')
plt.ylabel('Error (fine scale pixels)')
plt.show()
print(total)