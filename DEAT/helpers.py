#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import cm
from matplotlib.widgets import Cursor, MultiCursor
import matplotlib.pyplot as plt
import itertools

class imshow_show_z:
    def __init__(self, ax, z, x, y):
        self.ax = ax
        self.x  = x
        self.y  = y
        self.z  = z
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.numrows, self.numcols = self.z.shape
        self.ax.format_coord = self.format_coord
    def format_coord(self, x, y):
        col = int(x/self.dx+0.5)
        row = int(y/self.dy+0.5)
        #print "Nx, Nf = ", len(self.x), len(self.y), "    x, y =", x, y, "    dx, dy =", self.dx, self.dy, "    col, row =", col, row
        xyz_str = ''
        if ((col>=0) and (col<self.numcols) and (row>=0) and (row<self.numrows)):
            zij = self.z[row,col]
            #print "zij =", zij, '  |zij| =', abs(zij)
            if (np.iscomplex(zij)):
                amp = abs(zij)
                phs = np.angle(zij) / np.pi
                if (zij.imag >= 0.0):
                    signz = '+'
                else:
                    signz = '-'
                xyz_str = 'x=' + str('%.4g' % x) + ', y=' + str('%.4g' % y) + ',' \
                            + ' z=(' + str('%.4g' % zij.real) + signz + str('%.4g' % abs(zij.imag)) + 'j)' \
                            + '=' + str('%.4g' % amp) + r'*exp{' + str('%.4g' % phs) + u' π j})'
            else:
                xyz_str = 'x=' + str('%.4g' % x) + ', y=' + str('%.4g' % y) + ', z=' + str('%.6g' % zij)
        else:
            xyz_str = 'x=%1.4f, y=%1.4f'%(x, y)
        return xyz_str
    
def new_imshow(ax, x, y, z, cmap, origin):
    print len(x), len(y), z.shape
    assert(len(x) == z.shape[0])
    assert(len(y) == z.shape[1])
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    if (np.iscomplex(z).any()):
        zabs = abs(z)
    else:
        zabs = z
    # Use this to center pixel around (x,y) values
    extent = (x[0]-dx/2.0, x[-1]+dx/2.0, y[0]-dy/2.0, y[-1]+dy/2.0)
    # Use this to let (x,y) be the lower-left pixel location (upper-left when origin = 'lower' is not used)
    #extent = (x[0]-dx/2.0, x[-1]+dx/2.0, y[0]-dy/2.0, y[-1]+dy/2.0)
    im = ax.imshow(zabs, interpolation = 'none', cmap = cmap, origin=origin)
    imshow_show_z(ax, z, x, y)
    #ax.set_xlim((x[0], x[-1]))
    #ax.set_ylim((y[0], y[-1]))
    return im

def comp(a, b=0, cmap = cm.bone, origin='upper'):
    '''returns ax1, ax2, multicursor'''
    if b == 0 and a.shape[2] == 2:
        thisa, thisb = a[:,:,0], a[:,:,1]
    else:
        thisa, thisb = a, b
    x, y = np.arange(thisa.shape[0]), np.arange(thisa.shape[1])
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey = True, sharex = True, figsize = (40,40), squeeze=True)
    fig.tight_layout()
    new_imshow(ax1, x, y, thisa, cmap, origin=origin)
    new_imshow(ax2, x, y, thisb, cmap, origin=origin)
    multi = MultiCursor(fig.canvas, (ax1, ax2), color='r', useblit=True, horizOn=True, vertOn= True, lw=2)
 #   cursor = Cursor((ax1, ax2), useblit=True, color='red', linewidth=2 )
    plt.show()
    return ax1, ax2, multi
    
def comp3(a, b, c, cmap = cm.bone, origin='upper'):
    '''returns ax1, ax2, multicursor'''

    x, y = np.arange(a.shape[0]), np.arange(a.shape[1])
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharey = True, sharex = True, figsize = (40,40), squeeze=True)
    fig.tight_layout()
    new_imshow(ax1, x, y, a, cmap, origin=origin)
    new_imshow(ax2, x, y, b, cmap, origin=origin)
    new_imshow(ax3, x, y, c, cmap, origin=origin)
    multi = MultiCursor(fig.canvas, (ax1, ax2, ax3), color='r', useblit=True, horizOn=True, vertOn= True, lw=2)
 #   cursor = Cursor((ax1, ax2), useblit=True, color='red', linewidth=2 )
    plt.show()
    return ax1, ax2, ax3, multi
    
def polyfit2d(x, y, z, order=3, linear=False):
    """Two-dimensional polynomial fit. Based uppon code provided by 
    Joe Kington.
    
    References:
        http://stackoverflow.com/questions/7997152/
            python-3d-polynomial-surface-fit-order-dependent/7997925#7997925

    """
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j
        if linear & (i != 0.) & (j != 0.):
            G[:, k] = 0
    m, _, _, _ = np.linalg.lstsq(G, z)
    return m

def polyval2d(x, y, m):
    """Values to two-dimensional polynomial fit. Based uppon code 
        provided by Joe Kington.
    """
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order+1), range(order+1))
    z = np.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        z += a * x**i * y**j
    return z
