# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 18:28:10 2017

@author: bened
"""

import numpy as np
import scipy as sp
import scipy.ndimage
import matplotlib.pyplot as plt
import copy
import random
import param
reload(param)
import datagen



def testfun(noise = None, gap = None, randstate = None):
    if gap is None:
        gap = param.gap
    if randstate is None:
        randstate = np.random.RandomState(param.seed)  
    if noise is None:
        noise = param.noise
        
    A,M = datagen.datagen(noise, gap, randstate)
    AA = np.zeros((param.frames, param.n, param.n))
    f = np.zeros((param.frames,param.n))
    for i in range(param.frames):
        AA[i,:,:] = np.dot(A[i,:,:].T, A[i,:,:])
        f[i,:] = np.dot(A[i,:,:], M[i,:].T)
    b = np.sum(f, axis = 0)
    B = np.sum(AA, axis = 0)
    reconstruction = np.linalg.lstsq(B, b.T, rcond = 1e-5)[0]
    res = min(1, reconstruction[param.n/3],reconstruction[int(param.n*(1+gap)/3)])-max(0, min(reconstruction[param.n/3:int(param.n*(1+gap)/3)]))
    print('Resolution factor: %f' %res)
    return reconstruction

plt.plot(param.x, testfun(0.1,0.23))
plt.show()