import numpy as np
import scipy as sp
import scipy.ndimage
import matplotlib.pyplot as plt
import copy
import random
import cvxopt
import imp

l1regls = imp.load_source('l1regls.py', 'C:/Users/bened/Pythonscripts/l1regls.py')

##Print full list/array
np.set_printoptions(threshold=np.nan)

#no. of pts
n = 256
#superresolution factor
srf = 16.0
x = np.linspace(-1,2,n)
noise = 0.4
frames = 30

##step-like peaks of width 0.2
def peak(x):
    return ((x > -0.1) & (x < 0.1)).astype(float)



#preallocate space for matrix of frames
M = np.zeros((frames, n))
N = np.zeros((frames, n))
A = np.zeros((frames, n, n))
AA = np.zeros((frames, n, n))
f = np.zeros((frames,n))
puresignal = peak(x) + peak(x-1)

#circulant
C = 1/srf*sp.linalg.circulant(np.append(np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]), np.zeros(n-srf))).T
                        
#generate frames with random shift and put them into a matrix
for i in range(frames):
#true signal
    p = peak(x) + peak(x-1) + np.random.normal(0,noise,n)
    shift = random.randint(-8,8)
    M[i,:] = np.roll(np.repeat(np.dot((np.roll(C, shift, axis = 0))[::srf,:], p), srf), -shift)
    A[i,:,:] = np.roll(np.repeat((np.roll(C, shift, axis = 0))[::srf,:], srf, axis = 0), -shift, axis = 0)
    AA[i,:,:] = np.dot(A[i,:,:].T, A[i,:,:])
    f[i,:] = np.dot(A[i,:,:], M[i,:].T)
b = np.sum(f, axis = 0)
B = np.sum(AA, axis = 0)
   
y = l1regls.l1regls(cvxopt.base.matrix(B), cvxopt.base.matrix(b))
plt.plot(x, y)
plt.show()