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
import param
reload(param)
import datagen

       
#generate frames with random shift and put them into a matrix

def testfun(noise = None, gap = None, randstate = None):
    if gap is None:
        gap = param.gap
    if randstate is None:
        randstate = np.random.RandomState(param.seed)  
    if noise is None:
        noise = param.noise
        
    A,M = datagen.datagen(noise, gap, randstate)
    B = np.sum(A, axis = 0)
    f = (np.sum(M, axis=0)).T
    reconstruction = np.linalg.lstsq(B, f, rcond = 1e-5)[0]
    res = min(1, reconstruction[param.n/3],reconstruction[int(param.n*(1+gap)/3)])-max(0, min(reconstruction[param.n/3:int(param.n*(1+gap)/3)]))
    print('Resolution factor: %f' %res)
    return reconstruction
    
plt.plot(param.x, testfun(0.1,0.23))
plt.show()
