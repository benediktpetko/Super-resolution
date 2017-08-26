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



def testfun(noise = None, gap = None, randstate = None):
    if gap is None:
        gap = param.gap
    if randstate is None:
        randstate = np.random.RandomState(param.seed)  
    if noise is None:
        noise = param.noise
            
    h=[]
    G=[]
    A,M = datagen.datagen(noise, gap, randstate)
    epsilon = 6*noise
    for i in range(param.frames):         
        h += [matrix(np.append(np.array([epsilon]),-M[i,:]))]
        G += [matrix(-np.append(np.append([np.zeros(param.n)], A[i,:,:], axis = 0), np.zeros((param.n+1,param.n)), axis = 1))]
    
            
    #input for the solver        
    G0 = matrix(np.append(np.append(param.I,-param.I,axis=1),np.append(-param.I,-param.I,axis=1),axis = 0))
    sol = solvers.socp(param.c, Gl = G0, hl = matrix(np.zeros(2*param.n)), Gq=G, hq=h)
    reconstruction = sol['x'][0:param.n]
    res = min(1, max(reconstruction[param.n/3-2:param.n/3+3]), max(reconstruction[int(param.n*(1+gap)/3)-2:int(param.n*(1+gap)/3)+3]))-max(0, min(reconstruction[param.n/3:int(param.n*(1+gap)/3)]))
    print('Resolution factor: %f' %res)
    return reconstruction

plt.plot(param.x, testfun(0.1,0.13))
plt.show()
