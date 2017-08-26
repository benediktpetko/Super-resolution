import numpy as np
import scipy as sp
import scipy.ndimage
import matplotlib.pyplot as plt
import copy
import random
from cvxopt import matrix, solvers
import imp


#no. of pts
n = 256
#superresolution factor
srf = 32.0
x = np.linspace(-1,2,n)
noise = 0.3   #np.linspace(0.05,0.2,1)
frames = range(10,30,2)
reg = 0.1
##step-like peaks of width 0.2
def peak(x):
    return ((x > -0.1) & (x < 0.1)).astype(float)

#true signal
puresignal = peak(x) + peak(x-1)




#circulant
C = 1/srf*sp.linalg.circulant(np.append(np.ones(srf), np.zeros(n-srf))).T
                        
#generate frames with random shift and put them into a matrix
def quality(frames):
    M = np.zeros((frames, n))
    A = np.zeros((frames, n,n))
    for i in range(frames):
    #preallocate space for matrix of frames       
        p = peak(x) + peak(x-1) + np.random.normal(0,noise,n)
        shift = np.random.randint(-16,16)
        M[i,:] = np.roll(np.repeat(np.dot((np.roll(C, shift, axis = 0))[::srf,:], p), srf), -shift)
        A[i,:,:] = np.roll(np.repeat((np.roll(C, shift, axis = 0))[::srf,:], srf, axis = 0), -shift, axis = 0)   
    #input for the solver        
    D = sp.linalg.circulant(np.append(np.array([1.,-1.]),np.zeros(n-2))).T
    I = np.eye(n)
    G0 = matrix(np.append(np.append(D,-I,axis=1),np.append(-D,-I,axis=1),axis = 0))
    epsilon = frames*noise
    h1 = [matrix(np.append(np.array([epsilon]),-M.reshape(frames*n)))]
    G1 = [matrix(-np.append(np.append([np.zeros(n)], A.reshape((frames*n,n)), axis = 0), np.zeros((frames*n+1,n)), axis = 1))]
    c = matrix(np.ones(2*n))
    sol = solvers.socp(c, Gl = G0, hl = matrix(np.zeros(2*n)), Gq=G1, hq=h1)
    reconstruction = sol['x']
    return 1-np.sum((np.squeeze(np.array(reconstruction[0:n]))-puresignal)**2)/np.sum(np.array(reconstruction[0:n])**2)
    
def qualitymean(frames, k):
    return np.mean([quality(frames) for i in range(k)])  
    

qualitylist2 = [qualitymean(p,5) for p in frames]
plt.plot(frames,qualitylist2)
plt.show()