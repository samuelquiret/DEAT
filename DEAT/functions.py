# -*- coding: utf-8 -*-
"""
Created on Tue Dec 04 13:30:06 2018

@author: samuel.quiret
"""

import matplotlib.pyplot as plt
import numpy as np
from pylab import cm
from skimage.morphology import convex_hull_image


def __verbose__(text):
    print('\tverbose: {}'.format(text))
    
    
def ima(I, cmap=cm.afmhot, cut=3, colorbar=False, title='', name=''):
    plt.figure(figsize=(15,15), dpi=200)
    plt.title(title)
    maxval, minval = np.nanpercentile(I*1.0,100-cut), np.nanpercentile(I*1.0,cut)
    plt.imshow(I.clip(minval, maxval), cmap=cmap, interpolation='none')
    if colorbar: plt.colorbar()
    if name<>'': plt.savefig(name)
    
    return True
    


def ima_(I, cmap=cm.jet, cut=3, colorbar=False, title = ''):
    plt.figure(figsize=(15,15), dpi=200)
    plt.title(title)
    maxval, minval = np.nanpercentile(I*1.0,100-cut), np.nanpercentile(I*1.0,cut)
    im = plt.imshow(I.clip(minval, maxval), cmap=cmap, interpolation='none')
    if colorbar: plt.colorbar()
    return im


def vor_sanity(vor, ptsim):
    #HARD don't touch
    # finds senseful neighboors, filters out voronoi regions where a vertice is outside the convex hull
    #plt.imshow(skk, interpolation='none', cmap=cm.bone)

    point_sanity={}#dict of good neighboors and bad neighboors for each point
    sk = ptsim*0
    hull = convex_hull_image(ptsim)
    for i in range(len(vor.points)):
        if i not in point_sanity:
            point_sanity[i]={'good':[],'bad':[]}
        ri = vor.point_region[i]
        cy, cx = vor.points[i]
    #    plt.plot(cx, cy, 'b.')
        regs = vor.regions[ri]
    #    plt.text(cx, cy, '{:d}'.format(i))

        for j in range(len(vor.ridge_points)):
            if i in vor.ridge_points[j]:
                if i == vor.ridge_points[j][0]: thisneighnum = vor.ridge_points[j][1]
                else: thisneighnum = vor.ridge_points[j][0]
                well_defined = False
                if -1 not in vor.ridge_vertices[j]:
                    v1, v2 = vor.ridge_vertices[j]
                    v1y, v1x = vor.vertices[v1]
                    v2y, v2x = vor.vertices[v2]
                    #print v1y, v1x
                    if (v1y>0)*(v1x>0)*(v2y>0)*(v2x>0) and \
                    (v1y<(sk.shape[0]-1))*(v1x<(sk.shape[1]-1))* \
                    (v2y<(sk.shape[0]-1))*(v2x<(sk.shape[1]-1)):
                        v1y, v1x = np.round(v1y).astype(int), np.round(v1x).astype(int)
                        v2y, v2x = np.round(v2y).astype(int), np.round(v2x).astype(int)
                        well_defined = hull[v1y,v1x]*hull[v2y,v2x]
                        if well_defined:
                            p1, p2 = vor.ridge_points[j]
                            p1y, p1x = vor.points[p1]
                            p2y, p2x = vor.points[p2]
     #                       plt.plot([p1x, p2x],[p1y, p2y])
                if well_defined: point_sanity[i]['good'].append(thisneighnum)
                else: point_sanity[i]['bad'].append(thisneighnum)
    return point_sanity



def guy(good_mutuals, newborns):
    bb = []
    oo = good_mutuals[0][:,0]
    gg = deds[0]
    bb.append(np.hstack((oo,gg))*1.0)
    print bb
    #print len(bb), 'tt'
    ove=0
    for i in range(len(good_mutuals)):
        if i%10==0: print i,
        tgm = good_mutuals[i]
        #print i, tgm
        pp = []
#        print len(bb), bb[i].shape
        for bef in bb[-1]:
            if bef == -1: pp.append(-1)
            else:
                idmask = bef==tgm[:,0]
                if np.sum(idmask) ==1:
                    k = np.argwhere(idmask)[0]
                    af = tgm[k,1]
                    pp.append(af[0])
                else: pp.append(-1)
        for k in newborns[i]:
            pp.append(k)
        ove = max(ove, len(pp))
        bb.append(np.array(pp))
    cc = []
    for i in bb:
        cc.append(np.append(i, np.zeros(ove-i.size)-1))
    return np.array(cc).T