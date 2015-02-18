from __future__ import division
from ctypes import cdll, c_double, c_float, c_int, c_char_p, POINTER
import numpy as np
from math import pi
import matplotlib.pylab as pl
import os
import os.path
from numpyctypes import c_ndarray


def main():
    print(os.path.join(__path__[0], 'libgrid.so'))
    lib = cdll.LoadLibrary(os.path.join(__path__[0], 'libgrid.so'))
    N = 256
    arcsec = 1./180/3600*pi
    cell = 0.5*arcsec
    x0 = 0.928122246
    y0 = -27.638/180*pi
#     x0 = 2*arcsec
#     x0 = 1.
    grid = lib.c_grid
    grid.restype = None
#     grid.argtype = [c_char_p, POINTER(c_double), POINTER(c_double), POINTER(c_double), c_int, c_double, c_int]


    vis_real = np.zeros((6,N,N))
    vis_imag = np.zeros((6,N,N))
    weight   = np.zeros((6,N,N))
    pb       = np.zeros((6,N,N))

    c_vis_real = c_ndarray(vis_real, dtype=np.double, ndim=3)
    c_vis_imag = c_ndarray(vis_imag, dtype=np.double, ndim=3)
    c_weight   = c_ndarray(weight, dtype=np.double, ndim=3)
    c_pb       = c_ndarray(pb, dtype=np.double, ndim=3)

    grid(c_char_p(b'/data/ecdfs_raw_sorted.ms'), c_vis_real, c_vis_imag, c_weight, 
            c_pb, c_double(cell), c_float(x0), c_float(y0), c_int(1))
#     grid(c_char_p(b'/data/aless_drg.ms'), c_vis_real, c_vis_imag, c_weight, 
#             c_pb, c_double(cell), c_float(x0), c_float(y0), c_int(0))

#     pl.ion()
# pl.imshow(np.real(vis).transpose(), interpolation='nearest', origin='lower')
    vis = vis_real+1j*vis_imag
#     pl.clf()
#     pl.imshow(weight.transpose(), interpolation='nearest', origin='lower')
#     pl.imshow(weight.transpose(), interpolation='nearest', origin='lower')
#     pl.colorbar()
#     pl.savefig('test.png')
#     pl.show()

    np.save('data.npy', vis)
    np.save('weight.npy', weight)

if __name__  == '__main__':
    __path__ = [os.path.dirname(os.path.realpath(__file__))]
    main()
