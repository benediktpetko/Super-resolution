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
             
    G0 = matrix(np.append(np.append(param.D,-param.I,axis=1),np.append(-param.D,-param.I,axis=1),axis = 0))
    epsilon = param.frames*noise
    h1 = [matrix(np.append(np.array([epsilon]),-f))]
    G1 = [matrix(-np.append(np.append([np.zeros(param.n)], B, axis = 0), np.zeros((param.n+1,param.n)), axis = 1))]
    sol = solvers.socp(param.c, Gl = G0, hl = matrix(np.zeros(2*param.n)), Gq=G1, hq=h1)
    reconstruction = sol['x'][0:param.n]
    res = min([1, reconstruction[param.n/3],reconstruction[int(param.n*(1+gap)/3)]])-max(0, min(reconstruction[param.n/3:int(param.n*(1+gap)/3)]))
    print('Resolution factor: %f' %res)
    return res