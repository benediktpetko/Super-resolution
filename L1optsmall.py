import numpy as np
import scipy as sp
import scipy.ndimage
import matplotlib.pyplot as plt
import copy
import random
import cvxopt
import imp

l1regls = imp.load_source('l1regls.py', 'C:/Users/bened/Pythonscripts/l1regls.py')


#no. of pts
n = 256
#superresolution factor
srf = 16.0
x = np.linspace(-1,2,n)
noise = 0.5   #np.linspace(0.05,0.2,1)
frames = 30   #range(5,70,5)
reg = 0.1
##step-like peaks of width 0.2
def peak(x):
    return ((x > -0.1) & (x < 0.1)).astype(float)

#true signal
puresignal = peak(x) + peak(x-1)




#circulant
C = 1/srf*sp.linalg.circulant(np.append(np.ones(srf), np.zeros(n-srf))).T
                        
#generate frames with random shift and put them into a matrix

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
    
#return 1 - np.sum((lstsqimage[0]-puresignal)**2)/np.sum(lstsqimage[0]**2)
y = l1regls.l1regls(cvxopt.base.matrix(reg*B), cvxopt.base.matrix(reg*f))
    


plt.plot(x, y)
plt.show()