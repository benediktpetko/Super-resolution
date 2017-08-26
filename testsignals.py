
import numpy as np
import scipy as sp
import scipy.ndimage
import matplotlib.pyplot as plt
import copy
import random
import param
reload(param)
from param import *


#true signal
puresignal1 = peak(x) + peak(x-1)
puresignal2 = spike(x) + spike(x-1)

plt.subplot(1,2,1)
plt.plot(x, puresignal1)
plt.subplot(1,2,2)
plt.plot(x, puresignal2)
plt.show()