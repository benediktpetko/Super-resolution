import numpy as np
import scipy as sp
import scipy.ndimage
import matplotlib.pyplot as plt
import copy
import random
from cvxopt import matrix, solvers
import imp
import param
reload(param)

#generate data
def datagen(noise = None, gap = None, randstate = None):
    if gap is None:
        gap = param.gap
    if randstate is None:
        randstate = np.random.RandomState(param.seed)  
    if noise is None:
        noise = param.noise
    M = np.zeros((param.frames, param.n))
    A = np.zeros((param.frames, param.n,param.n))
    for i in range(param.frames):
    #preallocate space for matrix of frames       
        p = param.puresignal1(gap) + randstate.normal(0,noise,param.n)
        shift = randstate.randint(-int(param.srf/2),int(param.srf/2))
        M[i,:] = np.roll(np.repeat(np.dot((np.roll(param.C, shift, axis = 0))[::param.srf,:], p), param.srf), -shift)
        A[i,:,:] = np.roll(np.repeat((np.roll(param.C, shift, axis = 0))[::param.srf,:], param.srf, axis = 0), -shift, axis = 0)
    return A,M