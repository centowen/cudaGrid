import sys
import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift
from matplotlib import pylab as pl


# data = []
# with open('data.dat', 'r') as datafile:
#     for row in datafile:
#         data.append(list(map(complex, row[:-2].split(' '))))
#     data = np.array(data)
# 
# weight = []
# with open('uvcov.dat', 'r') as uvcovf:
#     for row in uvcovf:
#         weight.append(list(map(float, row[:-2].split(' '))))
#     weight = np.array(weight)

data = np.array([])
weight = np.array([])
data = np.load('data.npy')[sys.argv[1],:,:]
weight = np.load('weight.npy')[sys.argv[1],:,:]
    
# pl.ion()

pl.figure(1)
pl.clf()
pl.imshow(np.real(fftshift(fft2(fftshift(data)))).transpose(), origin='lower', interpolation='nearest')
# pl.imshow(np.real(fftshift(fft2(fftshift(data)))).transpose(), origin='lower', interpolation='nearest', vmax=5e-3, vmin=-5e-3)
pl.colorbar()
print('peak flux:', str(np.real(np.sum(data))*1e3)+'mJy')
pl.figure(2)
pl.clf()
pl.imshow(np.real(data).transpose(), origin='lower', interpolation='nearest')
#         vmax=0.5e-4, vmin=-0.1e-4)
pl.colorbar()
pl.figure(3)
pl.clf()
pl.imshow(weight.transpose(), origin='lower', interpolation='nearest')
pl.colorbar()
pl.show()
