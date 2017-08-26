# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 22:13:51 2017

@author: bened
"""
import numpy as np
import scipy as sp
import scipy.ndimage
import matplotlib.pyplot as plt
import copy
import random

#no. of pts
n = 256
#superresolution factor
srf = 16.0
x = np.linspace(-1,2,n)
noise = 0.3
frames = 50

##step-like peaks of width 0.2
def peak(x):
    return ((x > -0.1) & (x < 0.1)).astype(float)

puresignal = peak(x) + peak(x-1)

#preallocate space for matrix of frames
M = np.zeros((frames, n))
N = np.zeros((frames, n))
#circulant
C = 1/srf*sp.linalg.circulant(np.append(np.ones(srf), np.zeros(n-srf))).T
                        
#generate frames with random shift and put them into a matrix
np.random.seed(0)
for i in range(frames):
    #true signal
    p = peak(x) + peak(x-1) + np.random.normal(0,noise,n)
    shift = random.randint(-8,8)
    M[i,:] = np.roll(np.repeat(np.dot(C[::srf,:], np.roll(p,shift)), srf),-shift)
    
np.random.seed(0)
for i in range(frames):
    #true signal
    shift = random.randint(-8,8)
    p = peak(x) + peak(x-0.35) + np.random.normal(0,noise,n)
    N[i,:] = np.repeat(np.dot(C[::srf,:], np.roll(p,shift)), srf)
    
averageimage = 1.0/frames*M.sum(axis=0)
averageimage2 = 1.0/frames*N.sum(axis=0)
plt.plot(x,averageimage, label = 'Reconst.')
plt.plot(x,puresignal, label = 'Test signal')
plt.legend(loc='upper right')
#plt.plot(x,p)
plt.show()