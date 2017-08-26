import numpy as np
import scipy as sp
import scipy.ndimage
import matplotlib.pyplot as plt
import copy
import random
from cvxopt import matrix, solvers
import imp
import socpTVL1
import L1userguide
import param
import pickle

#silence the solver
solvers.options['show_progress'] = False

#keep in mind: change the constant in gaplist: 0.1 for L1, 0.2 for TV, change G0 in socpTVL1 accordingly

randstate = np.random.RandomState(2)
noiselist = np.linspace(0.05,1.,16)
gaplist = 0.2 + np.linspace(3.0/param.n,3.0*param.srf/param.n,int(param.srf))
#guide = np.zeros((len(gaplist),len(noiselist)))
resmatrix = np.zeros((3,len(gaplist),len(noiselist)))
#progress count
tmp1 = 0.0
tmp2 = 100.0/(len(noiselist)*len(gaplist))
for k in range(len(gaplist)): 
    for i in range(len(noiselist)): 
        resmatrix[:,k,i] = socpTVL1.testfun(noiselist[i], gaplist[k])
        #guide[k,i] = L1userguide.testfun(noiselist[i], gaplist[k])
        tmp1 += tmp2
        print('Completed: %d'% int(tmp1))
        
pickle.dump( resmatrix, open( "TVresmatSRF32.p", "wb" ) )


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 5))
for i in range(len(axes.flat)):
    im = axes.flat[i].imshow(resmatrix[i,:,:], vmin=0, vmax=1)

fig.colorbar(im, ax=axes.ravel().tolist(), label =r'$\alpha$ value')
axes[0].set_title('Inner sum constraint')
axes[1].set_title('Outer sum constraint')
axes[2].set_title('Many constraints')
axes[0].set_xlabel('Noise (SD)')
axes[1].set_xlabel('Noise (SD)')
axes[2].set_xlabel('Noise (SD)')
fig.text(0.5, 0.04, 'Resolution obtained by TV reconstruction', ha='center', va='center')
plt.setp(axes, xticks=[0,5,10,15], xticklabels = [0,0.37,0.68,1], yticks=np.arange(int(param.srf)),yticklabels=1+np.arange(int(param.srf)))
fig.text(0.06, 0.5, 'Gap (HR pixels)', ha='center', va='center', rotation='vertical')
plt.show()

# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
# im = axes.imshow(guide, vmin=0, vmax=1)
# plt.setp(axes, xticks=[0,5,10,15], xticklabels = [0,0.37,0.68,1], yticks=np.arange(int(param.srf)),yticklabels=1+np.arange(int(param.srf)))
# fig.text(0.06, 0.5, 'Gap (HR pixels)', ha='center', va='center', rotation='vertical')
# plt.show()