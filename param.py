import numpy as np
import scipy as sp
import scipy.ndimage
from cvxopt import matrix
#no. of pts
n = 256

#superresolution factor
srf = 32.0

#domain
x = np.linspace(-1,2,n)

#noise level
noise = 1.
  
#number of frames
frames = 36

#seed
seed = 4

#gap width
gap = 0.3

##test signals of width 0.2
def peak(x):
    return ((x > -0.1) & (x < 0.1)).astype(float)
    
def spike(x):
    return ((x > -0.1) & (x < 0.1)).astype(float)*(-10*abs(x)+1)
    
#circulant
C = 1/srf*sp.linalg.circulant(np.append(np.ones(int(srf)), np.zeros(n-int(srf)))).T
     
    
#true signal
def puresignal1(gap):
    puresignal1 = peak(x) + peak(x-gap)
    return puresignal1

def puresignal2(gap):
    puresignal2 = spike(x) + spike(x-gap)
    return puresignal2
    
#resolution threshold
resthr = 0.5
    
#these are always fed to the cvxopt solver
D = sp.linalg.circulant(np.append(np.array([1.,-1.]),np.zeros(n-2))).T
I = np.eye(n)
c = matrix(np.ones(2*n))
