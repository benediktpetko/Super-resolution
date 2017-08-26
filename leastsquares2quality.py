# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 10:01:30 2017

@author: bened
"""

import numpy as np
import scipy as sp
import scipy.ndimage
import matplotlib.pyplot as plt
import copy
import random

##Print full list/array
np.set_printoptions(threshold=np.nan)

#no. of pts
n = 256
#superresolution factor
srf = 16
x = np.linspace(-1,2,n)
noise = 0.05   #np.linspace(0.05,0.2,1)
frames = range(5,70,5)

##step-like peaks of width 0.2
def peak(x):
    return ((x > -0.1) & (x < 0.1)).astype(float)

#true signal
puresignal = peak(x) + peak(x-1)




#circulant
C = 1/srf*sp.linalg.circulant(np.append(np.ones(srf), np.zeros(n-srf))).T
                        
#generate frames with random shift and put them into a matrix
def quality(noise, frames):
    M = np.zeros((frames, n))
    A = np.zeros((frames, n,n))
    for i in range(frames):
   #preallocate space for matrix of frames       
       p = peak(x) + peak(x-1) + np.random.normal(0,noise,n)
       shift = np.random.randint(-8,8)
       #M[i,:] = np.roll(np.repeat(np.dot(C[::srf,:], np.roll(p,shift)), srf),-shift)
       M[i,:] = np.roll(np.repeat(np.dot((np.roll(C, shift, axis = 0))[::srf,:], p), srf), -shift)
       A[i,:,:] = np.roll(np.repeat((np.roll(C, shift, axis = 0))[::srf,:], srf, axis = 0), -shift, axis = 0)
    B = np.sum(A, axis = 0)
    #return np.linalg.norm(1/frames*np.sum(M, axis=0))
    f = (np.sum(M, axis=0)).T
    lstsqimage = np.linalg.lstsq(B, f, rcond = 1e-5)
    #return 1 - np.sum((lstsqimage[0]-puresignal)**2)/np.sum(lstsqimage[0]**2)
    return lstsqimage
    
def qualitymean(noise, frames, k):
    list = [quality(noise, frames) for i in range(k)]
    return np.mean(list)
#qualitylist = [qualitymean(noise, i, 20) for i in frames]

#plt.plot(x, quality(noise)[0])
#plt.show()
#plt.plot(x, lstsqimage[0])
#plt.show()
plt.plot(x, quality(noise, 10)[0])
#plt.title('Inside sum quality of reconstruction')
#plt.xlabel('Frames')
#plt.ylabel('Quality')
plt.show()