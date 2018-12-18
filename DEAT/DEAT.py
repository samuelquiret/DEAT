# -*- coding: utf-8 -*-
"""
Created on Thu Dec 06 11:46:17 2018
@author: samuel.quiret

                Declic Experiment Analysing Tool


This file is the DEAT class that performs the analysis of a set of images.

Usage:
 
    >>> from DeclicAnalisys import DeclicAnalysis
    >>> sequence = DEAT(path = pathToFolderWithImages, start_tick=967938780,\
                                  end_tick = 967950780, moteur = 967570127)
    >>> sequence.visu(0)
    >>> sequence.make_all_analysis(skip = 5)
    
    additional optionnal parameters for initialising sequence:
        * result_path
        * result_path
        * TNSP_file
        * wafer_file
        * cluster_file
        * pixel_size (default = 7.2)
        * moteur (default: 0)
        * start_tick (default: 0)
        * end_tick (default: inf)
        * eps (default: 8)
        * min_sample (default: 4)
        * verbose (default: False)
        * plot (default: False)
        * V (default: 2)
        * k_vel (default: 2)
        
        
"""





###################                                           need numpy > 1.13
#if __name__ == "__main__":
import cPickle as pickle
from PIL import Image as im
import os, re, cv2, time, sys
import numpy as np
from shutil import copyfile

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pylab import cm
from matplotlib import colors

from scipy import ndimage
from scipy.spatial import Voronoi
from scipy.interpolate import UnivariateSpline

from skimage.morphology import medial_axis
from skimage.filters import threshold_adaptive
from skimage import morphology, exposure
from skimage.draw import line_aa, line


from collections import Counter
from matplotlib.path import Path

#Filiate cells:
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

from mpl_toolkits.axes_grid1 import make_axes_locatable

from Image import Image

from functions import __verbose__, vor_sanity
from helpers import *



__all__ = ['DEAT']




class DEAT():
    def __init__(self, path = '',  result_path = '', result_name = '', \
                 TNSP_file = '', wafer_file = '', cluster_file = '', \
                 pixel_size = 7.2, moteur = 0, start_tick = 0,\
                  end_tick = float("inf"), eps = 7, min_sample = 4, \
                verbose = False, plot = False, V = 2, k_vel = 2):
        
        print('\n-------------------------------------------\n')
        print('       Declic Experiment Analysing Tool       \n')
        print('-------------------------------------------\n\n')
        
        self.path = path
        
        if self.path[-1]<>'\\': self.path = self.path + '\\'
        
        self.pixel_size = pixel_size
        self.moteur = moteur
        self.start_tick = start_tick
        self.end_tick = end_tick
        self.verbose = verbose
        self.plot = plot
        self.centers = []
        self.images_fullname = []
        self.T = []
        self.N = []
        self.S = []
        self.P = []
        self.clusters = []
        self.skeletons = []
        
        
        self.path_to_TNSP = TNSP_file
        self.path_to_wafers = wafer_file
        self.path_to_clusters = cluster_file
        
        
        self.neighwafer = []
        self.distancewafer = []
        self.stdwafer = []
        self.velmaginst = []
        self.angle_inst = []
        
        #self.mammothdistances = []
        
        
        self.wafer_loaded = False
        self.cluster_loaded = False
        self.TNSP_loaded = False
        
        
        
        self.image_names = []
        images = [i for i in os.listdir(self.path) if (i[-4:]=='.bmp' and 'HR2' in i and '_2.' not in i)]
        for image in images:
            tick = int(re.findall('[0-9]{9,10}',image)[0])
                if tick >= self.start_tick and tick <= self.end_tick:
                    self.image_names.append(image)
                    
        
        if len(self.image_names)<3: 
            print('\nThere must be at least 3 images in the working directory.\
            Please check your parameters\n')
            return False
            
        
        self.image_names.sort()
        
        
        if result_name <> '': result_name = '{}_'.format(result_name)
        if result_path == '':
            self.result_path = '{}results\\{}{}\\'.format(self.path,result_name,time.strftime("%Y.%m.%d_%H.%M.%S"))
            if not os.path.exists('{}results\\'.format(self.path)):
                os.mkdir('{}results\\'.format(self.path))
        else:
            self.result_path = '{}\\{}{}\\'.format(result_path,result_name,time.strftime("%Y.%m.%d_%H.%M.%S"))
            if not os.path.exists(result_path):
                os.mkdir(result_path)
        os.mkdir(self.result_path)  
        print('The different results will be saved in the folder: {}\n'.format(self.result_path))
        
        ### parameters
        self.sigma = 3.7 ## used for center determination --- #Basic treatment. Good enough for a good number of situations
        self.equmag_region = 61  # kernel_size
        self.black_gen_thresh_percentile = 40
        self.halflength = 25                 #### demi_longueur des traits de gradients tracés
        
        self.eps = eps
        self.min_sample = min_sample ## 15, not 4
        
        self.V = V ## vitesse de solidification
        
        self.k_vel = k_vel ## number of images used for the velocity derivation
        
        
        
    
    
    def plot_image(self, id_image):
        '''plot_image(id_image):
                plot the image number id_image of your studied directory'''
        i = Image(path = self.path, name = self.image_names[id_image])
        i.plot_image()
        
    
    def concentricify(self, id_image):
        i = Image(path = self.path, name = self.image_names[id_image])
        return i.concentricify()
    
    def get_image(self, id_image):
        return Image(path = self.path, name = self.image_names[id_image], result_path = self.result_path)
        

#==============================================================================
# LOAD FILES
#==============================================================================


    def load_image(self, n):
        """
        >> load_image(n)
        
        loads image n from current working directory
        """
        if n < len(self.image_names):
            return np.array(Image(path=self.path, name=self.image_names[n]).In)/255.0
        else:
            print('Can\'t load image, please specify an adequat image number (total number: {})'.format(len(self.image_names)))
            return False
        
    def load_image_equalized(self, n):
        """
        >> load_image_equalized(n)
        
        loads contrast enhanced image n from current working directory 
        """
        if n < len(self.image_names):
            return exposure.equalize_adapthist(self.load_image(n), kernel_size=self.equmag_region, clip_limit=0.99, nbins=1000)
        else:
            print('Can\'t load image, please specify an adequat image number (total number: {})'.format(len(self.image_names)))
            return False
        
        
    
    
    def load_files(self, TNSP_file = '', cluster_file = '', wafer_file = ''):
        
        """
        >> load_files(TNSP_file = '', cluster_file = '', wafer_file = '')
        
        loads the files from the cell centers up to the different wafers used 
        for the analysis. 
        The hierarchy is wafer > clusters > TNSP.
        if a new cluster file is specified, it will erase and recompute the
        already loaded wafers, to make sure the clusters and wafers correspond.
        
        """
        print('Loading files...')
        
        ## if we specify new wafer file
        if wafer_file <> '' and os.path.isfile(wafer_file) and wafer_file<>self.path_to_wafers:
            self.load_wafers(wafer_file=wafer_file)
            self.compute_abs_vel()
            return True
        
        
        ## if we specify new cluster file: we need to reload everything
        if cluster_file <> '' and os.path.isfile(cluster_file) and cluster_file<>self.path_to_clusters:
            self.wafer_loaded = False
            self.path_to_wafers = ''
            self.load_clusters(cluster_file=cluster_file)
            self.load_wafers()
            self.compute_abs_vel()
            return True
                  
                  
        
        if self.load_wafers(wafer_file): 
            self.compute_abs_vel()
            return True
        
        if self.load_clusters(cluster_file): 
            self.make_distance_wafer()
            self.compute_abs_vel()
            return True
        
        if self.load_TNSP(TNSP_file): 
            self.load_clusters()
            self.load_wafers()
            self.compute_abs_vel()
        else:
            return False
#        
#                  
        
        return True
    
    
    
    
    def load_TNSP(self, TNSP_file = ''):
        """
        >> load_TNSP(TNSP_file = '')
        
        loads the files from the cell centers computation.
        
        """
        
        if TNSP_file <> '':
            if self.path_to_TNSP <> TNSP_file:
                print 'Another TNSP file has been specified'
                self.path_to_TNSP = TNSP_file
        
        if self.path_to_TNSP == '':
            print('No TNSP file specified.')
            self.process_images()
            return True
        
        if os.path.isfile(self.path_to_TNSP):
            if self.path_to_TNSP <> '{}TNSP.pkl'.format(self.result_path): copyfile(self.path_to_TNSP, '{}TNSP.pkl'.format(self.result_path))
            self.path, self.image_names, self.T,self.N,self.S,self.P = pickle.load(open(self.path_to_TNSP, 'rb'))
            
            self.I = Image(path = self.path, name=self.image_names[0]).In/255.0
            self.times = self.T
            self.imsize= self.I.shape
            if self.verbose: __verbose__('imsize = {}'.format(self.imsize))
            self.TNSP_loaded = True
            print('\t\tTNSP loaded successfully from {:s}'.format(self.path_to_TNSP))
            
        else:
            print('The path to TNSP file is not correct, reprocessing the images...')
            self.process_images()
            
        return True
            
        
    def load_wafers(self, wafer_file=''): 
        """
        >> load_wafers(wafer_file = '')
        
        loads the wafers.
        
        """        
        
        if wafer_file <> '':
            if self.path_to_wafers <> wafer_file:
                if os.path.isfile(wafer_file):
                    print 'Another wafer file has been specified, loading new wafers...'
                    self.path_to_wafers = wafer_file
                    if self.path_to_wafers <> '{}wafers.pkl'.format(self.result_path): copyfile(self.path_to_wafers, '{}wafers.pkl'.format(self.result_path))
                    self.path, self.image_names, self.distancewafer, self.stdwafer, self.neighwafer, self.clusters, self.labels, self.times = pickle.load(open(self.path_to_wafers, 'rb'))
                    self.wafer_loaded = True
                    self.I = self.load_image(0)
                    self.imsize = self.I.shape
                    print('\t\tWafers loaded successfully from {:s}'.format(self.path_to_wafers))
                    return True 
                else: 
                    print('The wafer file specified does not exists.')
            else:
                print('The specified wafers corresponds to the same wafers as specified originally.')
                   
        if self.path_to_wafers <> '':
            if os.path.isfile(self.path_to_wafers):
                if self.path_to_wafers <> '{}wafers.pkl'.format(self.result_path): copyfile(self.path_to_wafers, '{}wafers.pkl'.format(self.result_path))
                self.path, self.image_names, self.distancewafer, self.stdwafer, self.neighwafer, self.clusters, self.labels, self.times = pickle.load(open(self.path_to_wafers, 'rb'))
                self.wafer_loaded = True
                self.I = self.load_image(0)
                self.imsize = self.I.shape
                print('\t\tWafers loaded successfully from {:s}'.format(self.path_to_wafers))
                return True
            else: 
                print('The wafer file specified does not exists.')
        
        
        
        
        if self.path_to_wafers == '' and wafer_file == '':
            if self.cluster_loaded:
                self.make_distance_wafer()
        return False
    
    
    def load_clusters(self, cluster_file=''):
        """
        >> load_clusters(cluster_file = '')
        
        loads the clusters.
        
        """        
        
        if cluster_file <> '':
            if self.path_to_clusters <> cluster_file:
                print 'Another cluster file has been specified' # TODO : il faut recharger les wafers
            self.path_to_clusters = cluster_file
        
        if self.path_to_clusters == '':
            print('No cluster file specified.')
            self.compute_clusters()
            return True
        
        if os.path.isfile(self.path_to_clusters):
            if self.path_to_clusters <> '{}clusters.pkl'.format(self.result_path): copyfile(self.path_to_clusters, '{}clusters.pkl'.format(self.result_path))
            self.path, self.image_names, self.clusters, self.labels, self.times = pickle.load(open(self.path_to_clusters, 'rb'))
            self.cluster_loaded = True
            self.I = self.load_image(0)
            self.imsize = self.I.shape
            print('\t\tClusters loaded successfully from {:s}'.format(self.path_to_clusters))
            
            return True
        else:
            print('The path to cluster file is not correct. Reprocessing...')
            self.compute_clusters()
            return False
    
    
    def compute_clusters(self):
        """
        >> compute_clusters()
        
        Compute the clusters from the different version of Jorge's code, and 
        erase the previously computed wafers
        
        """        
#        self.compute_clusters_V0()
        self.compute_clusters_V1()
#        self.compute_clusters_V2()
#        self.compute_clusters_V3()

        ## as we have loaded new clusters, the previous wafers can be unrelated
        ## so we unset them
        self.neighwafer = []
        self.distancewafer = []
        self.stdwafer = []
        self.velmaginst = []
        self.angle_inst = []
        
        
#==============================================================================
#     
#==============================================================================
    
    
    
    def process_images(self, method = 0):
        """
        >> process_images(method = 0)
        
        Run the center determination algorithm on the images specified in the 
        current directory.
        
        ## method:
            # 0 = gradient (default)
            # 1 = curvature
            # 2 = watershed
        
        """        
        
        if method == 0:
            print('Processing the images using the concentric features and gradient...')
        elif method == 1: 
            print('Processing the images using the concentric features and curvature...')
        elif method == 2: 
            print('Processing the images using the dual threshold...')
        
        
        self.T = []
        self.N = []
        self.S = []
        self.P = []
        
        for n in range(len(self.image_names)):
            
            try:
                tick = int(re.findall('[0-9]{9,10}',self.image_names[n])[0])
                if tick >= self.start_tick:
                    
                    print('\tProcessing image {}... {}'.format(n, self.image_names[n]))
                    
                    # get the time of the image
                    if self.verbose :  __verbose__('Get times of image {} ({})'.format(n,self.image_names[n]))
                    self.T.append((tick - self.moteur)/23.0)
                    #self.T.append(tick)
                    
                    # get the coordinates of the cells       
                    # TODO: choose which method
                    if self.verbose :  __verbose__('Get cells\'centers of image {} ({})'.format(n,self.image_names[n]))
                    
                    if method == 0:
                        image = Image(name=self.image_names[n], path = self.path, result_path = self.result_path, verbose = self.verbose, plot = self.plot)
                        image.find_centers_concentric()
                        centers = image.centers[1:]
                    if method == 1:
                        image = Image(name=self.image_names[n], path = self.path, result_path = self.result_path, verbose = self.verbose, plot = self.plot)
                        image.find_centers_concentric(with_curvature=True)
                        centers = image.centers_curvature[1:]
                    elif method == 2: 
                        image = Image(name=self.image_names[n], path = self.path, result_path = self.result_path, verbose = self.verbose, plot = self.plot)
                        image.find_centers_watershed_V1()
                        centers = image.centers_dualthreshold
                        
                    if self.plot:
                        image.plot_centers()
                    self.P.append(centers)
            
                    if self.verbose :  __verbose__('Get spacing and neighbors of image {} ({})'.format(n,self.image_names[n]))
                    nneighs, spacing = self.neighboors(centers)
                    spacing = [a*self.pixel_size for a in spacing]                    
                    self.S.append(spacing)
                    self.N.append(nneighs)
                    
                    
                if tick >= self.end_tick: break
            except: continue
        
        self.T = np.array(self.T)
        self.N = np.array(self.N)
        self.S = np.array(self.S)
        self.P = np.array(self.P)
        
        self.path_to_TNSP = '{}TNSP.pkl'.format(self.result_path)
        
        print('\n\t\tdump the TNSP vector to {}'.format(self.path_to_TNSP))
        pickle.dump((self.path, self.image_names, self.T,self.N,self.S,self.P),open(self.path_to_TNSP,'wb'))
        
        self.load_TNSP()
        
        return True
    
        
    def neighboors(self, points):
        vor = Voronoi(points)
        vor.close()
        neighs = {}
        for p1, p2 in vor.ridge_points:
            if p1 not in neighs: neighs[p1] = []
            if p2 not in neighs: neighs[p2] = []
            neighs[p1].append(p2)
            neighs[p2].append(p1)
    
        nneighs, spacing = [], []
        for i in neighs:
            nneighs.append(len(neighs[i]))
            spacing.append(np.mean((np.sum((points[i]-points[neighs[i]])**2, axis=1)**.5)))
        return nneighs, spacing
        
        
    
    
    
    
#==============================================================================
# COMPUTE CLUSTERS
#==============================================================================
    
    def compute_clusters_V0(self, TNSP_file = ''): 
        
        if self.load_TNSP(TNSP_file):
#        
            print('compute clusters V0')
            
            ########photoshop a useful mask
            masku = np.ones((1024, 1024))>0#
            #masku = np.array(Image.open('{:s}mask_molde.bmp'.format(res_path)))>0
                
            cts = self.P[:,:2]
            
            #now turn them into 3D coords
            coords = np.zeros((0,3))
            for i in range(len(cts)):
                ind = np.expand_dims(np.array([i]*cts[i].shape[0]), axis=1)
                theserows = np.hstack((cts[i], ind))
                coords = np.vstack((coords, theserows))
            
            oo = []
            for n in range(self.times.size):
                oo.extend(range(np.sum(coords[:,2]==n)))
            coords = np.hstack((coords, np.expand_dims(np.array(oo),1)))
            coords = np.hstack((coords, np.expand_dims(np.arange(coords.shape[0]),1)))
            
            #if there is a useful region mask, limit coords to thiose in it
            coo = coords[:,:2].astype(int)
            coords = coords[masku[coo[:,0], coo[:,1]]]
            ####
#            
#            counter = 0
#            ran = 50
#            
            good_mutuals = []
            newborns = []
            deds = []
            for n in range(self.times.size-1):
                tom_m = coords[:,2] == n+1
                tod_m = coords[:,2] == n#today mask
            
                tom_c = coords[tom_m] #
                tod_c = coords[tod_m] #
            
                mat_fwd = np.tile(tom_c[:,:2], (tod_c.shape[0], 1, 1))
                mat_bak = np.tile(tod_c[:,:2], (tom_c.shape[0], 1, 1)).transpose(1, 0, 2)
            
                dist_mat = np.sum((mat_fwd-mat_bak)**2, axis=2) 
                aa = np.argmin(dist_mat, axis=0)
                bb = np.argmin(dist_mat, axis=1)
                #todays is linked to tomorrow??
                today_is_linked = bb == bb[aa[bb]]#fucking hard to figure out this index thing.
                today_is_linked = aa[bb] == np.arange(bb.size)
            
                #todays succesful couples
                today_links = np.vstack((np.arange(bb.size), bb[aa[bb]])).T[today_is_linked]
                good_mutuals.append(today_links)
            
                #todays dead
                tods_ded = np.arange(bb.size)[~today_is_linked]
                deds.append(tods_ded)
            
                #tomorrow's newborn
                tom_newborn = np.setdiff1d(np.arange(aa.size), today_links[:,1])
                newborns.append(tom_newborn)
                if n%10==0: print n,
            
            
            #This is the actual filiator
            ooo = guy(good_mutuals, newborns)
                
                
            
            x, y = np.arange(ooo.shape[1]), np.arange(ooo.shape[0])
            y, x = np.meshgrid(x, y)[::-1]
            
            oo = ooo*1
            oo[oo==-1]=np.nan
            domain = x*(oo*0+1)
            minlive, maxlive = np.nanmin(domain, axis=1), np.nanmax(domain, axis=1)
            
            suma = [0]
            for n in range(oo.shape[1]):
                suma.append(np.sum(~np.isnan(oo[:,n])))
            suma = np.cumsum(suma[:-1])
            suma = np.tile(suma, (oo.shape[0], 1))
            suma = suma+oo
            
            bb = np.zeros((0,8))
            print '\n', suma.shape
            for i in np.arange(suma.shape[0]):
                if i%1000==0: print i,
                ss = suma[i]
                ss = ss[~np.isnan(ss)].astype(int)
                thisbout = [i, minlive[i], maxlive[i]]
                thisbout = np.tile(thisbout,(ss.size, 1))
                thisbout = np.hstack((coords[ss], thisbout))
                bb = np.vstack((bb, thisbout))
            
            b = bb*1
            
            ##This marks where an old is probably not really dead
            #skips = [1, 2, 3, 4, 5, 6]
            #margin = 90
            #
            #oldnew = {}
            #for s in skips:
            #    print ''
            #    oldnew[s] = []
            #    for deadin in range(times.size-s):
            #        if deadin%100==0:print deadin,
            #            
            #        dm = b[:,7] == deadin
            #        heads = (b[:,7] == deadin)*(b[:,2] == deadin)
            #
            #        bm = b[:,6] == deadin + s
            #        tails = (b[:,6] == (deadin + s))*(b[:,2] == (deadin+s))
            #
            #
            #        for i in b[heads]:
            #            if (i[0]<margin)+(i[0]>(1024-margin))+(i[1]<margin)+(i[1]>(1024-margin)): continue
            #            for j in b[tails]:
            #                if (j[0]<margin)+(j[0]>(1024-margin))+(j[1]<margin)+(j[1]>(1024-margin)): continue
            #                if np.sum((j[:2]-i[:2])**2) < 10**2:
            #                    oldnew[s].append((i,j))
            #                    break
            #                    
            #for i in oldnew:
            #    oldnew[i] = np.array(oldnew[i])
            
            
            
            #make sure all cells have congruent start_end labels
            #I did something wrong above, because thay don't
            if False:
                for i in np.unique(b[:,5]):
                    mask = b[:,5]==i
                    if np.unique(b[mask][:,6]).shape[0]>1:
                        print i, np.unique(b[mask][:,6])
                    if i >500: break    
                
            celllabeldict = dict(zip(np.unique(b[:,5]), np.arange(np.unique(b[:,5]).size)))
            
            uu = np.zeros((np.unique(b[:,5]).size, self.times.size, 2))*0.0+np.nan
            for i in b:
                y, x, t, _, _ , cellno, _, _ = i
                uu[celllabeldict[cellno], t, :] = y, x
                
            naa = np.isnan(uu[:,:,0])
            missings = []
            for i in range(naa.shape[0]):
                nlb = ndimage.label(naa[i])[0]
                thismissings = []
                for j in np.unique(nlb)[1:]:
                    nbout = np.argwhere(nlb==j)
                    thismissings.append((np.min(nbout), np.max(nbout)))
                missings.append(thismissings)
                
            dd = uu*1
            for i in range(len(missings)):
                for dim in [0,1]:
                    #print '*',i, missings[i]
                    for m in missings[i]:
                        if m[0] != 0 and m[1] != (naa.shape[1]-1):
                            filler = np.linspace(uu[i, m[0]-1, dim], uu[i, m[1]+1, dim], m[1]-m[0]+3).astype(int)[1:-1]
                            #print 'e',m,filler[1:-1]
                            if filler.size < 10:# and i<=lastnormal:
                                #print i,
                                dd[i, m[0]:m[1]+1, dim] = filler
                
                
            
            ee = dd*1
            oo = np.arange(ee.shape[1])
            oo = np.tile(oo, (ee.shape[0],1))*1.0
            
            tt = np.tile(self.times, (ee.shape[0],1))*1.0
            
            oo[np.isnan(ee[:,:,0])] = np.nan
            tt[np.isnan(ee[:,:,0])] = np.nan
            
            left = np.nanmin(oo, axis=1).astype(int)
            right = np.nanmax(oo, axis=1).astype(int)
            duration = right-left
            
            mask = (duration>5)
            ee = ee[mask]
            
            
            self.clusters = ee
        
    
    
    
    
    def compute_clusters_V1(self, TNSP_file = ''): 
        """
        >> compute_clusters_V1(TNSP_file = '')
        
        Compute the cell clusters given the TNSP_file (where the center of all
        cells are stored)
        
        
        """        
        
        if self.load_TNSP(TNSP_file):
            print('compute clusters V1')
    
            coords4D = np.array([]) #coords 4D actually, the 4 being mean spacing                       
            #this is nice but you haven't implemented the fleshout() to handle it
            #so use above in the meantime.
            #when you do, the remove_borders=False in fleshout() will be very useful
            #to distinguish a single cell whose color varies.
            for t in range(len(self.P)):
                coords4D = np.append(coords4D, np.hstack( (self.P[t],
                                                           np.expand_dims(np.repeat(t,self.P[t].shape[0]),1),
                                                           np.expand_dims(self.S[t],1)) ) )
                
            coords4D = np.reshape(coords4D, (coords4D.shape[0]/4 , 4) ).astype(int)


            cordszoomed = coords4D[:,0:3]*1
            cordszoomed[:,2] = cordszoomed[:,2]/2.0
                       
            db = DBSCAN(eps=self.eps, min_samples=self.min_sample).fit(cordszoomed)
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            
            print('Estimated number of clusters: %d' % n_clusters_)
            del(cordszoomed)

            bb = np.zeros((n_clusters_, len(self.P), 3), dtype=int)-2                

            for l in np.unique(labels):
                co = coords4D[labels == l]
                for c in co:
                    bb[l,c[2],:2] = c[:2]
                    bb[l,c[2],2] = c[3]                                             

            

            naa = bb[:,:,0]==-2
            #aaa = naa*0
            
            missings = []
            for i in range(naa.shape[0]):
                nlb = ndimage.label(naa[i])[0]
                thismissings = []
                for j in np.unique(nlb)[1:]:
                    nbout = np.argwhere(nlb==j)
                    thismissings.append((np.min(nbout), np.max(nbout)))
                missings.append(thismissings)
            
            dd = bb*1
            for i in range(len(missings)):
                for dim in [0,1]:
                    #print '*',i, missings[i]
                    for m in missings[i]:
                        if m[0] != 0 and m[1] != (naa.shape[1]-1):
                            filler = np.linspace(bb[i, m[0]-1, dim], bb[i, m[1]+1, dim], m[1]-m[0]+3).astype(int)[1:-1]
                            #print 'e',m,filler[1:-1]
                            dd[i, m[0]:m[1]+1, dim] = filler
            
            

        print('Number of clusters after cleaning: {}'.format(dd.shape[0])) 
        
        
        self.path_to_clusters = '{}clusters.pkl'.format(self.result_path)
        
        print('\n\t\tdump the clusters to {}'.format(self.path_to_clusters))
        pickle.dump((self.path, self.image_names, dd, labels, self.times),open(self.path_to_clusters,'wb'))
        
        self.load_clusters()

        return True

    def compute_clusters_V2(self, TNSP_file = ''):
        #now turn them into 3D coords
        
        if self.load_TNSP(TNSP_file):
            print('compute clusters V2')
            
            cts = self.P
            
            coords = np.zeros((0,3))
            for i in range(len(cts)):
                ind = np.expand_dims(np.array([i]*cts[i].shape[0]), axis=1)
                theserows = np.hstack((cts[i], ind))
                coords = np.vstack((coords, theserows))
        
            oo = []
            for n in range(self.T.size):
                oo.extend(range(np.sum(coords[:,2]==n)))
            coords = np.hstack((coords, np.expand_dims(np.array(oo),1)))
            
            
            thesecords = coords*1
            thesecords[:,2]=thesecords[:,2]/1.0
                      
            db = DBSCAN(eps=self.eps, min_samples=self.min_sample).fit(thesecords)# importanteps=7, min_samples=5
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        
            print('Estimated number of clusters: %d' % n_clusters_)
            
            #Turn the clusters into wafers. ie: flat 3D arrays of idmensions [n_cells, n_times, 2-long for x,y]
            T = self.T.size
            ulb = np.unique(labels)
        
            bb = []
        
            dupes = []
        
            for i in ulb[1:]:
                #print i, np.sum(i==labels)
                x_row = np.zeros(T)-2
                y_row = x_row*1
                entries = coords[labels == i]
                for y, x, t, _ in entries:
                    if x_row[int(t)] != -2: dupes.append([y,x,t,i])
                    x_row[int(t)] = x
                    y_row[int(t)] = y
                bb.append([y_row, x_row])
        
            bb = np.transpose(np.array(bb), (0,2,1))
            bb[bb==-2]=np.nan
            ##bb is the raw array which contains all our information
        
            if False:
                ima(bb[:,:,0])
                plt.title('y coords')
                ima(bb[:,:,1])
                plt.title('x coords')
        
            #Now. There are many things to do. Fill those holes.
            #Duplicates are stored in "dupe". 
            #what to do?
            #this is a question without a perfectly objective answer.
            #It is "artsy", depends on the experiment, and might not even be that important.
            #for the moment, I'll just ignore the dupes 
            #(sorry guys, I might have integrated your doppleganger and left you out to rot)
        
            naa = np.isnan(bb[:,:,0])
            #aaa = naa*0
        
            missings = []
            for i in range(naa.shape[0]):
                nlb = ndimage.label(naa[i])[0]
                thismissings = []
                for j in np.unique(nlb)[1:]:
                    nbout = np.argwhere(nlb==j)
                    thismissings.append((np.min(nbout), np.max(nbout)))
                missings.append(thismissings)
        
            dd = bb*1
            for i in range(len(missings)):
                for dim in [0,1]:
                    #print '*',i, missings[i]
                    for m in missings[i]:
                        if m[0] != 0 and m[1] != (naa.shape[1]-1):
                            filler = np.linspace(bb[i, m[0]-1, dim], bb[i, m[1]+1, dim], m[1]-m[0]+3).astype(int)[1:-1]
                            dd[i, m[0]:m[1]+1, dim] = filler
        
            #now compare your handiwork,
            #ima(bb[:,:,0])
    #        plt.title('before hole-filling')
    #        ima(dd[:,:,0])
    #        plt.title('after hole-filling')
            
            
            self.clusters = dd
            self.bb = bb
            
            self.reunify()                                             #### Need to check if ok
            
        return True
    
    
    
    def prime(self, i, primes):
        for prime in primes:
            if not (i == prime or i % prime):
                return False
        primes.append(i)
        return i
            
    def find_primes(self, n, s=0,app = 0):
        """generates a list of n primes for labelling"""
        primes = list([2])
        i, p = 2, 0
        while True:
            if self.prime(i, primes):
                p += 1
                if p == n:
                    primes = np.array(primes, dtype=int)
                    primes = primes[primes>10]
                    if app != None: primes = np.append(0,primes)
                    return primes
            i += 1
    
    def compute_clusters_V3(self, TNSP_file = ''): 
        """
        >> compute_clusters_V3(TNSP_file = '')
        
        Compute the cell clusters given the TNSP_file (where the center of all
        cells are stored). If no TNSP file specified, the method process_images
        is called.
        
        
        #from fuckup repair 3grain 2um act8 aug30 2016
        
        """    
        #now turn them into 3D coords
        
        if self.load_TNSP(TNSP_file):
    
            print('Compute clusters V3\n')
            #now turn them into 3D coords
            coords = np.zeros((0,3))
            for i in range(len(self.P)):
                ind = np.expand_dims(np.array([i]*self.P[i].shape[0]), axis=1)
                theserows = np.hstack((self.P[i], ind))
                coords = np.vstack((coords, theserows))
            
                
            thesecords = coords*1
            
            #db = DBSCAN(eps=7, min_samples=10).fit(thesecords)
            db = DBSCAN(eps=self.eps, min_samples=self.min_sample).fit(thesecords)
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            
            print('Estimated number of clusters: %d' % n_clusters_)
            
            
            a = self.find_primes(1500)[1:]
            primed_imno = a[thesecords[:,2].astype(int)]
            unique_id = primed_imno * labels
            unid, inv, counts  = np.unique(unique_id, return_counts=True, return_inverse=True)
            
            dustmask = labels==-1
            bastardmask = np.logical_and(~dustmask, counts[inv]>1)
            
            #perfect_cords = thesecords[np.logical_and(~bastardmask, ~dustmask)]
            perfect_labels = labels[np.logical_and(~bastardmask, ~dustmask)]
            
            #duplica_cords = thesecords[bastardmask]
            duplica_labels = labels[bastardmask]
            
                
            #now, we straighten out duplicates.
            #first we see if there are really that many repeats, which would indicate a true splitting.
            #if not, we spline the coords. if yes, I don't know yet, but in act 8, 2um, there seem to be fex real cases
            unique_duplicate_labels = np.unique(duplica_labels)
            corrected_duplicates = {}
            dust = {}
            
            for ulb in unique_duplicate_labels:
                thismask = (labels==ulb)
                tc = thesecords[thismask]
                thisun, thisct = np.unique(tc[:,2], return_counts=True)
                if np.sum(thisct>1)<5: #no more than 5 doublets
                    corrected_duplicates[ulb] = tc
                else:
                    earliest, latest = np.min(tc[:,2]), np.max(tc[:,2])
                    duration = latest-earliest
                    normtime = np.arange(earliest, latest+1)
                    xt, yt, tt = tc[:,1], tc[:,0], tc[:,2]
                    
                    tt += (np.arange(len(tc[:,2]))/latest)*10**(-9)
                    
                    splx = UnivariateSpline(tt, xt, s=2*duration)
                    sply = UnivariateSpline(tt, yt, s=2*duration)
                    newx = splx(normtime)
                    newy = sply(normtime)
                    #plt.figure()
                    #plt.scatter(tt, xt)
                    #plt.plot(normtime, newx)
                    if False:#This whole block is swrewy.sum(np.isnan(newy))>0:
                        newtc = np.vstack((newy, newx, normtime)).T
                        corrected_duplicates[ulb] = newtc
                    else:
                        dust[ulb] = tc
        
    
    
            rowtemplate = np.zeros((self.times.size,2))*np.nan
            bb = []
            for pl in np.unique(perfect_labels):
                tc = thesecords[labels==pl]
                thisrow = rowtemplate*1
                for y,x,n in tc:
                    thisrow[int(n),:2]=y,x
                bb.append(thisrow)
            bb = np.array(bb)
            #cut = len(bb)
            
            deformed_bb = []
            for bl in corrected_duplicates:
                thisrow = rowtemplate*1
                for y,x,n in corrected_duplicates[bl]:
                    thisrow[int(n),:]=y,x
                deformed_bb.append(thisrow)
            deformed_bb = np.array(deformed_bb)    
    
            u = []
            counter = 0
            for i in bb[:,:,0][:]:
                x = hash(str(i))
                counter+=1
                u.append(x)
                #if counter%100==0: print counter,
                
            unhash, inv, counts = np.unique(u, return_counts=True, return_inverse=True)
            #print np.sum(counts==2)
            
            u = []
            counter = 0
            for i in deformed_bb[:,:,0][:]:
                x = hash(str(i))
                counter+=1
                u.append(x)
                #if counter%100==0: print counter,
                
            unhash, inv, counts = np.unique(u, return_counts=True, return_inverse=True)
            #print np.sum(counts==2)
    
    
    
            thisbouttemplate = np.transpose(np.tile(np.expand_dims(np.array([-2]*bb.shape[1]),0),(1,2,1)),(0,2,1))
            
            thisbout = thisbouttemplate*1
            bco = thesecords[labels==-1]
            for y,x,n in bco:
                nn = int(n)
                height, timelength, _ = thisbout.shape
                occupied = np.sum(thisbout[:,nn,0] != -2)
                #print occupied,
                if height <= occupied: thisbout = np.vstack((thisbout, thisbouttemplate))
                thisbout[occupied,nn,:] = [y, x]
            
            for k in dust:
                for y,x,n in dust[k]:
                    nn = int(n)
                    height, timelength, _ = thisbout.shape
                    occupied = np.sum(thisbout[:,nn,0] != -2)
                    #print occupied,
                    if height <= occupied: thisbout = np.vstack((thisbout, thisbouttemplate*1.0))
                    thisbout[occupied,nn,:] = [y, x]
            thisbout[thisbout == -2] = np.nan
            
                
            naa = np.isnan(bb[:,:,0])
            #aaa = naa*0
            
            missings = []
            for i in range(naa.shape[0]):
                nlb = ndimage.label(naa[i])[0]
                thismissings = []
                for j in np.unique(nlb)[1:]:
                    nbout = np.argwhere(nlb==j)
                    thismissings.append((np.min(nbout), np.max(nbout)))
                missings.append(thismissings)
            
            #LAST USEFUL. CAREFUL! Changes every time!
            #lastnormal = 2907
            
                
            dd = bb*1
            for i in range(len(missings)):
                for dim in [0,1]:
                    #print '*',i, missings[i]
                    for m in missings[i]:
                        if m[0] != 0 and m[1] != (naa.shape[1]-1):
                            filler = np.linspace(bb[i, m[0]-1, dim], bb[i, m[1]+1, dim], m[1]-m[0]+3).astype(int)[1:-1]
                            #print 'e',m,filler[1:-1]
                            if filler.size < 6:# and i<=lastnormal:
                                dd[i, m[0]:m[1]+1, dim] = filler
                    

        
            if True:        
                #dust that is redundant is eliminated
                for n in range(thisbout.shape[1]):
                    oo = thisbout[:,n,:]
                    oo = np.tile(oo, (dd.shape[0],1,1))
                    uu = np.tile(dd[:,n,:], (thisbout.shape[0], 1,1))
                    uu = np.transpose(uu, (1,0,2))
                    dist = np.sum((uu-oo)**2, axis=2)**.5
                    wh = np.argwhere(dist<5)
                    for i in range(wh.shape[0]):
                        thisbout[wh[i][1],n,:] = np.nan
    
            #reunify wrongly separated (in time) cells
            oo = np.arange(dd.shape[1])
            oo = np.tile(oo, (dd.shape[0],1))*1.0
            
            oo[np.isnan(dd[:,:,0])] = np.nan
            
            left = np.nanmin(oo, axis=1).astype(int)
            right = np.nanmax(oo, axis=1).astype(int)
            ind = np.arange(dd.shape[0])
            
            ee = dd*1
            
            for n in range(dd.shape[1]-1):
                mother = ee[:,n,:][right==n]
                mind = ind[right==n]
                daughter = ee[:,n+1,:][left==(n+1)]
                dind = ind[left==(n+1)]
            #    print n,
            #    print mother.shape, daughter.shape
                m1 = np.tile(mother, (daughter.shape[0], 1, 1))
                d1 = np.tile(daughter, (mother.shape[0], 1, 1))
                d1 = np.transpose(d1, (1,0,2))
                dist = np.sum((m1-d1)**2, axis=2)**.5
            #    print m1.shape, d1.shape
                for i in np.argwhere(dist<9):
                    if dind[i[0]] < bb.shape[0] and mind[i[1]] < bb.shape[0]:
                        ee[mind[i[1]],n+1:] = ee[dind[i[0]],n+1:]*1
                        ee[dind[i[0]],:] = np.nan
                        
            ee = ee[np.sum(~np.isnan(ee[:,:,0]), axis=1)>0]
            eedust = thisbout
            #newnormal = ee.shape[0]-thisbout.shape[0]
            #    print ' '
            #    if n==30: break
            #eedust = ee[newnormal:]
            #ee = ee[:newnormal]
            #self.clusters = ee  
    
    
            print('Number of clusters after cleaning: {}'.format(ee.shape[0])) 
            
            
            self.path_to_clusters = '{}clusters.pkl'.format(self.result_path)
            
            print('\n\t\tdump the clusters to {}'.format(self.path_to_clusters))
            pickle.dump((self.path, self.image_names, ee, labels, self.times),open(self.path_to_clusters,'wb'))
            
            self.load_clusters()
    
    

    def reunify(self):
                #reunify wrongly separated (in time) cells
                
        print ('reunify wrongly separated (in time) cells')
        
        oo = np.arange(self.clusters.shape[1])
        oo = np.tile(oo, (self.clusters.shape[0],1))*1.0
        
        oo[np.isnan(self.clusters[:,:,0])] = np.nan
        
        left = np.nanmin(oo, axis=1).astype(int)
        right = np.nanmax(oo, axis=1).astype(int)
        ind = np.arange(self.clusters.shape[0])
        
        ee = self.clusters*1
        
        for n in range(self.clusters.shape[1]-1):
            mother = ee[:,n,:][right==n]
            mind = ind[right==n]
            daughter = ee[:,n+1,:][left==(n+1)]
            dind = ind[left==(n+1)]
        #    print n,
        #    print mother.shape, daughter.shape
            m1 = np.tile(mother, (daughter.shape[0], 1, 1))
            d1 = np.tile(daughter, (mother.shape[0], 1, 1))
            d1 = np.transpose(d1, (1,0,2))
            dist = np.sum((m1-d1)**2, axis=2)**.5
        #    print m1.shape, d1.shape
            for i in np.argwhere(dist<9):
                if dind[i[0]] < self.bb.shape[0] and mind[i[1]] < self.bb.shape[0]:
                    ee[mind[i[1]],n+1:] = ee[dind[i[0]],n+1:]*1
                    ee[dind[i[0]],:] = np.nan
                    
        self.ee = ee[np.sum(~np.isnan(ee[:,:,0]), axis=1)>0]
        
        self.clusters = self.ee
        

        return True
    
    
    def distances(self, vor, sanity, skel, toplot=False):
        if toplot:
            plt.figure()
            plt.imshow(skel, interpolation='none', cmap=cm.bone)
        rigorous_distances = {}#we only count cells where all neighs are accounted for
        permissive_distances = {}#we also count cells where we couldn't measure all neighboors
        permissive_neighboors = {}
    
        c = vor.points
        for i in range(len(c)):
            rigorous_distances[i] = []
            permissive_distances[i] = []
            permissive_neighboors[i] = []
    
            cy, cx = c[i]
            if toplot:
                plt.plot(cx, cy, 'b.')
                plt.text(cx, cy, '{:d}'.format(i), color='#555599', size=8, va='top', ha='left')
                plt.text(cx, cy, '(x:{:d}, y:{:d}))'.format(int(cx), int(cy)), color='grey', size=8, va='bottom', ha='center')
    
            for j in sanity[i]['good']:
                y1, x1 = c[i]
                y2, x2 = c[j]
                if toplot: plt.plot([x1, x2],[y1, y2], c='k', linewidth=0.4)
                d = self.pixel_size * ((x1-x2)**2+(y1-y2)**2)**.5
                angle = np.arctan2((x2-x1), (y2-y1))*180/np.pi
                angle = (angle+360)%180 + 90
                midpointx = x1+(x2-x1)/2
                midpointy = y1+(y2-y1)/2
                if toplot:
                    plt.text(midpointx, midpointy, '{:0.2f}'.format(d),color='#444400', va='center', ha='center', rotation=angle)
                permissive_distances[i].append(d)
                permissive_neighboors[i].append(j)
                if len(sanity[i]['bad']) == 0:
                    rigorous_distances[i].append(d)
        return permissive_distances, permissive_neighboors

    
    def make_distance_wafer(self, skip = 1):
        """
        >> make_distance_wafer(skip = 1):
        
        Compute the distance and neigh wafer from the clusters. The skip 
        argument (default:1) allows to treat 1 image out of skip for testing 
        
        
        """        
        print('\nProcessing the wafers...')
        
        I = self.load_image(0)
        distancewafer = self.clusters[:,:,0]*np.nan                             ## ee in original code
        stdwafer = distancewafer*np.nan
        neighwafer = distancewafer*0
        for n in np.arange(self.clusters.shape[1])[::skip]:
            existing_mask = ~np.isnan(self.clusters[:,n,0])
            cents = self.clusters[:,n,:2][existing_mask]
            print n,
            thisI = I*0
            ptsim = thisI*0
            for ct0, ct1 in np.clip(cents, 0, self.imsize[0]):
                ct0, ct1 = int(ct0), int(ct1)
                ptsim[ct0,ct1] = 1
    
            vor = Voronoi(cents)
            sanity = vor_sanity(vor, ptsim)
            jpd, jpn = self.distances(vor, sanity, thisI, toplot=False)
            thisdist = []
            thisstd = []
            thisnb = []
            for k in jpd.keys():
                thisdist.append(np.mean(jpd[k]))
                thisstd.append(np.std(jpd[k]))
                thisnb.append(len(jpd[k]))
                
            thisdist = np.array(thisdist)
            thisstd = np.array(thisstd)
            thisnb = np.array(thisnb)
            
            expandeddist = np.zeros(self.clusters.shape[0])*np.nan
            expandeddist[existing_mask] = thisdist
    
            expandedstd = np.zeros(self.clusters.shape[0])*np.nan
            expandedstd[existing_mask] = thisstd
            
            expandednb = np.zeros(self.clusters.shape[0])*np.nan
            expandednb[existing_mask] = thisnb
            
            #mammothdistances[n] = jpd
            distancewafer[:,n] = expandeddist
            stdwafer[:,n] = expandedstd
            neighwafer[:,n] = expandednb
                      
                      
                      
        
        self.distancewafer = np.array(distancewafer)
        self.stdwafer = np.array(stdwafer)
        self.neighwafer = np.array(neighwafer)
        #self.mammothdistances = np.array(mammothdistances)
        
        self.path_to_wafers = '{}wafers.pkl'.format(self.result_path)
        
        print('\n\t\tdump the wafers to {}'.format(self.path_to_wafers))
        pickle.dump((self.path, self.image_names, self.distancewafer, self.stdwafer, self.neighwafer, self.clusters, self.labels, self.times),open(self.path_to_wafers,'wb'))
        
        self.load_wafers()
        
        return True
    
    
#==============================================================================
#         Velocity
#==============================================================================

    def compute_abs_vel(self, k = -1): ### smoothdd does not work
        """
        >> compute_abs_vel(k = -1)
        
        Compute the velocity and direction wafer from the clusters. The skip 
        argument (default:1) allows to treat 1 image out of skip for testing.
        
        the k parameter is the number of images to take into account for the 
        velocity computation:
            
                    vn = (x[n+k] - x[n-k]) / (2*k*dt)
        
        """      
    
        print('Compute velocity...')
        
        if k == -1: k = self.k_vel
                       
        if len(self.times) < 2*k+1: 
            print('Not enough images studied to computed instant velocity with k={}'.format(k))
            return False
        
                       
        deux_k_dt = self.times[2*k:]-self.times[:-2*k]
        dy_deux_k_dt, dx_deux_k_dt = [], []
        
        for i in np.arange(self.clusters.shape[0]):
            
            if np.sum(~np.isnan(self.clusters[i,:,0])) > 3:           ### TODO find out why this if check is there
                yy, xx = self.smooth_abstract(self.clusters[i,:,0]), self.smooth_abstract(self.clusters[i,:,1])
                dy, dx = yy[2*k:] - yy[:-2*k], xx[2*k:] - xx[:-2*k]
            else:
                print('\tcluster {}: unable to compute velocity'.format(i))
                dy, dx = self.clusters[i,2*k:,0]*np.nan, self.clusters[i,2*k:,0]*np.nan
                                
            dy = np.append(np.zeros(k), np.array(dy))  
            dy = np.append(np.array(dy),np.zeros(k))  
            dx = np.append(np.zeros(k), np.array(dx)) 
            dx = np.append(np.array(dx), np.zeros(k)) 
            b = np.append(np.ones(k), np.array(deux_k_dt))  
            dt = np.append(b, np.ones(k))
                            
                
            dy_deux_k_dt.append(dy/dt)
            dx_deux_k_dt.append(dx/dt)
        
        dy_deux_k_dt = np.array(dy_deux_k_dt)
        dx_deux_k_dt = np.array(dx_deux_k_dt)
        
        self.velmaginst = 1000 * self.pixel_size * (dy_deux_k_dt**2 + dx_deux_k_dt**2)**.5  ### nm/s
        self.angle_inst = np.arctan2(-dy_deux_k_dt, dx_deux_k_dt)
        
        return True



    def smooth_abstract(self, places, extend=0):
        basemask = ~np.isnan(places)
        extendmask = basemask*True
        for i in range(extend):
            extendmask = ndimage.binary_dilation(extendmask)    
        y = places*1
        x, y = self.times[basemask], y[basemask]
    
        #spl.set_smoothing_factor(0.5)
        spl = UnivariateSpline(x, y, s=200.0)
        yy = places*1
    
        xs = self.times[extendmask]
        yy[extendmask]=spl(xs)
    
        return yy
        




#==============================================================================
#==============================================================================
# # Make Images
#==============================================================================
#==============================================================================


    def make_skel(self):
        """
        >> make_skel()
        
        Compute the skeleton mask for all the images under study
        
        """      
        
        if len(self.clusters) == 0: self.compute_clusters();
        
        for n in range(self.clusters.shape[1]):
            I = np.array(Image(path=self.path, name=self.image_names[n]).In)
            sk = (I*0).astype(int)
            for i in range(self.clusters.shape[0]):
                    ycoord, xcoord = self.clusters[i,n,:2]
                    if ~np.isnan(ycoord):
                        ycoord, xcoord = int(round(ycoord)), int(round(xcoord))
                        ycoord, xcoord = np.clip(ycoord, 0, 1023), np.clip(xcoord, 0, 1023)
                        #ax.text(xcoord, ycoord,'{:d}'.format(i), color='#33FFFF', ha='center', va='center')  # print cell number
                        sk[ycoord,xcoord] = 1
            sk = medial_axis(sk == 0) == 0
            self.skeletons.append(sk)


        return True
    
    
        
    def make_carto(self, coordinates, values, border_value=np.nan):#in pixels
    
        if coordinates.shape[0] != values.size:
            print('Coordinates and values have different size')
            
        I = np.zeros(self.imsize)
        for y, x in coordinates:
            if np.isnan(y): continue
            y = int(np.round(y.clip(0,I.shape[0]-1)))
            x = int(np.round(x.clip(0,I.shape[1]-1)))
            I[y, x] =1
        I = ~medial_axis(I==0)
    
        lb = ndimage.label(I)[0]
        I = I*0.0+border_value
        for i in range(values.shape[0]):
            y, x = coordinates[i]
            if np.isnan(y): continue
            y = int(np.round(y.clip(0,I.shape[0]-1)))
            x = int(np.round(x.clip(0,I.shape[1]-1)))
            thislabel = lb[y,x]
            I[lb==thislabel] = values[i]
            
        return I
        
    
    
    
    
    
    
    def map_spacing(self, n):
        """
        >> map_spacing(n)
        
        Produce the spacing map of image n
        
        """      
        if len(self.distancewafer) == 0: self.make_distance_wafer()
        
        J = self.make_carto(self.clusters[:,n,:2], self.distancewafer[:,n])
        #J = self.make_carto(self.clusters[:,n,:2], self.clusters[:,n,2])
        ima(J, colorbar=True)
        plt.title(u'primary spacing (µm)')
        plt.show()
        return True
    
    def map_neigh(self, n):
        """
        >> map_neigh(n)
        
        Produce the neigh map of image n
        
        """      
        
        if len(self.neighwafer) == 0: self.make_distance_wafer()
        
        J = self.make_carto(self.clusters[:,n,:2], self.neighwafer[:,n])
        I = J.clip(4,8)
        cmap = plt.cm.get_cmap('jet',5)
        plt.figure(figsize=(15,15), dpi=200)
        plt.title(u'number of neighbours')
        plt.imshow(I, cmap=cmap, interpolation='none')
        plt.colorbar(spacing='proportional')
        plt.show()
        return True

    def map_abs_vel(self, n):
        """
        >> map_abs_vel(n)
        
        Produce the absolute velocity map of image n
        
        """      
        
        if len(self.velmaginst) == 0: self.compute_abs_vel()
        
        J = self.make_carto(self.clusters[:,n,:2], self.velmaginst[:,n])
        J[J>300] = np.nan
        plt.figure(figsize=(15,15), dpi=200)
        plt.imshow(J, cmap='pink')
        plt.colorbar()
        plt.title(u'apparent velocity (nm/s)')
        plt.show()
        return True
    
    
    def map_dir(self, n):
        """
        >> map_dir(n)
        
        Produce the direction map of image n
        
        """      
        
        if len(self.angle_inst) == 0: self.compute_abs_vel()
        
        J = self.make_carto(self.clusters[:,n,:2], self.angle_inst[:,n])
        plt.figure(figsize=(15,15), dpi=200)
        plt.imshow(J, cmap='jet')
        plt.colorbar()
        plt.title('apparent direction (rad)')
        
        plt.show()
        return True    
            
        
        
        
    def make_maps(self, n):
        """
        >> make_maps(n)
        
        Produce the spacing, neigh, vel, std dev, and direction maps of image n
        
        """      

        self.load_files()
        
        thismask = ~np.isnan(self.clusters[:,n,0])
        thisdd = np.round(np.clip(self.clusters[thismask,n,:2], 0, self.imsize[0]-1)).astype(int)  ##ee in original code
        thisvel = self.velmaginst[thismask, n]
        thisstd = self.stdwafer[thismask,n]
        thisdist = self.distancewafer[thismask,n]
        thisneigh = self.neighwafer[thismask,n]
        thisangle = self.angle_inst[thismask, n]
    
        I = Image(path=self.path, name=self.image_names[n]).In/255.
        sk = I*0
        sk[thisdd[:,0], thisdd[:,1]] = 1
        sk = ~medial_axis(sk==0)
        lb = ndimage.label(sk)[0]
    
        speedmap = I*np.nan
        stdmap = I*np.nan
        distmap = I*np.nan
        anglemap = I*np.nan
        neighmap = I*np.nan
        
        for i in range(thisdd.shape[0]):
            y, x = thisdd[i]
            thislb = lb[y, x]
            patchmask = lb==thislb
            speedmap[patchmask] = thisvel[i]
            stdmap[patchmask] = thisstd[i]/thisdist[i]
            neighmap[patchmask] = thisneigh[i]
            anglemap[patchmask] = thisangle[i]
            distmap[patchmask] = thisdist[i]
                       
                       
                       
        maps = [(distmap, u'primary spacing (um)', cm.jet, 'dist_v{:d}_n{:05d}_t{:0.2f}_L{:0.2f}.png'.format(self.V, n, self.times[n], self.times[n]*self.V), (120, 320)),
               (neighmap, u'number of neighbours', cm.jet, 'neighs_v{:d}_n{:05d}_t{:0.2f}_L{:0.2f}.png'.format(self.V, n, self.times[n], self.times[n]*self.V), (4,8)),
               (speedmap, u'apparent velocity (nm/s)', cm.jet, 'speed_v{:d}_n{:05d}_t{:0.2f}_L{:0.2f}.png'.format(self.V, n, self.times[n], self.times[n]*self.V), (10, 120)),
                (stdmap*100, u'standard deviation (%)', cm.jet, 'std_v{:d}_n{:05d}_t{:0.2f}_L{:0.2f}.png'.format(self.V, n, self.times[n], self.times[n]*self.V), (5, 30)),
                (anglemap*180/np.pi, u'apparent direction (ccw deg)', cm.hsv, 'dir_v{:d}_n{:05d}_t{:0.2f}_L{:0.2f}.png'.format(self.V, n, self.times[n], self.times[n]*self.V), (-180, 180))
               ]
        
        
        for x, label, cmap, savename, clip in maps:
            xx = x*1
            lb = ndimage.label(~np.isnan(xx))[0]
            edges = np.concatenate((lb[:2,:].ravel(), lb[-2:,:].ravel(), lb[:,:2].ravel(),lb[:,-2:].ravel()))
            edges = np.unique(edges)
            xx[np.isin( lb, edges)] = np.nan
            fig, ax = plt.subplots(figsize=(10,10), dpi=150)
            ax.set_ylabel('y (mm)')
            ax.set_xlabel('x (mm)')
            ax.set_xticklabels([])
            cax = ax.imshow(xx, cmap=cmap, extent=(0, self.imsize[1]*self.pixel_size/1000., self.imsize[0]*self.pixel_size/1000.,0), vmin=clip[0], vmax=clip[1])
            cb = fig.colorbar(cax, orientation='horizontal', pad=0.001, fraction=0.04765)
            ax.grid()
            cb.set_label(label)
            fig.savefig('{:s}trash_{:s}'.format(self.result_path, savename))
            plt.close(fig)        
        
        return True



    def make_all_analysis(self, skip = 1): 
        """
        make_all_analysis() or make_all_analysis(skip=10)
            generate analysis of the working directory (every 'skip value' images, 1 by default)
            plot the image with skeleton, the maps for spacing, speed, direction,
            standard deviation, number of neighbors and the histograms for the 
            spacing, speed and directions
        """
        
        self.load_files()
        
        print('\nMaking images...\n')
        
        
        if ~os.path.exists('{:s}basic_maps\\'.format(self.result_path)):
            os.mkdir('{:s}basic_maps\\'.format(self.result_path))
            
        
        fig = plt.figure(figsize=(30, 30), dpi = 100, tight_layout = True)
        gs = GridSpec(3, 3, width_ratios=[ 1, 1, 0.8], bottom=0.02, top=0.98, left=0.05, right = 0.98, wspace=0.2, hspace = 0.09)
            
        
        
        ##### Plot Skelet
        ax0 = plt.subplot(gs[0])
        im0 = ax0.imshow(self.load_image(0), cmap=cm.bone, interpolation="none", vmin=0, vmax=1)
        ax0.set_yticklabels([])
        ax0.set_xticklabels([])
        ax0.set_xlabel(u'Skeleton', fontsize=30)
        divider0 = make_axes_locatable(ax0)
        cax0 = divider0.append_axes("right", "5%", pad="3%")
        plt.colorbar(im0, cax=cax0, orientation='vertical')
        ax0.set_ylim(self.imsize[0],0)
        ax0.set_xlim(0,self.imsize[1])
        
        
        ##### Plot Primary Spacing
        minspac = 120
        maxspac = 320
        ax1 = plt.subplot(gs[1])
        im1 = ax1.imshow(-self.I, interpolation="none", cmap=cm.jet, vmin=minspac, vmax=maxspac)
        ax1.set_yticklabels([])
        ax1.set_xticklabels([])
        ax1.set_xlabel(u'primary spacing (um)', fontsize=30)
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", "5%", pad="3%")
        plt.colorbar(im1, cax=cax1, orientation='vertical')
        ax1.set_ylim(self.imsize[0],0)
        ax1.set_xlim(0,self.imsize[1])
        
        
        ##### Plot standard deviation
        minspac = 5
        maxspac = 30
        ax3 = plt.subplot(gs[3])
        im3 = ax3.imshow(-self.I, interpolation="none", cmap=cm.jet, vmin=minspac, vmax=maxspac)
        ax3.set_yticklabels([])
        ax3.set_xticklabels([])
        ax3.set_xlabel( u'standard deviation (%)', fontsize=30)
        divider3 = make_axes_locatable(ax3)
        cax3 = divider3.append_axes("right", "5%", pad="3%")
        plt.colorbar(im3, cax=cax3, orientation='vertical')
        ax3.set_ylim(self.imsize[0],0)
        ax3.set_xlim(0,self.imsize[1])
        
        
        ##### Plot directions
        minspac = -180
        maxspac = 180
        ax7 = plt.subplot(gs[7])
        im7 = ax7.imshow(-self.I, interpolation="none", cmap=cm.hsv, vmin=minspac, vmax=maxspac)
        ax7.set_yticklabels([])
        ax7.set_xticklabels([])
        ax7.set_xlabel( u'apparent direction (ccw deg)', fontsize=30)
        divider7 = make_axes_locatable(ax7)
        cax7 = divider7.append_axes("right", "5%", pad="3%")
        plt.colorbar(im7, cax=cax7, orientation='vertical')
        ax7.set_ylim(self.imsize[0],0)
        ax7.set_xlim(0,self.imsize[1])
        
        ##### Plot Velocity
        minspac = 10
        maxspac = 120
        ax4 = plt.subplot(gs[4])
        im4 = ax4.imshow(-self.I, interpolation="none", cmap=cm.jet, vmin=minspac, vmax=maxspac)
        ax4.set_yticklabels([])
        ax4.set_xticklabels([])
        ax4.set_xlabel( u'apparent velocity (nm/s)', fontsize=30)
        divider4 = make_axes_locatable(ax4)
        cax4 = divider4.append_axes("right", "5%", pad="3%")
        plt.colorbar(im4, cax=cax4, orientation='vertical')
        ax4.set_ylim(self.imsize[0],0)
        ax4.set_xlim(0,self.imsize[1])
        
        ##### Plot neighbors
        minspac = 2
        maxspac = 8
        ax6 = plt.subplot(gs[6])
        
        
        # make a color map of fixed colors
        #○cmap_neigh = colors.ListedColormap(['0.0', '0.125', '0.25', '0.375', '0.5', '0.675', '0.80', '0.975'])
        cmap_neigh = colors.ListedColormap(['C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'])
        bounds_neigh=[1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5]
        ticks_neigh=[2,3,4,5,6,7,8]
        norm_neigh = colors.BoundaryNorm(bounds_neigh, cmap_neigh.N)  
        

        im6 = ax6.imshow(-self.I, interpolation="none", cmap=cmap_neigh, norm=norm_neigh, vmin=minspac, vmax=maxspac)
        
        ax6.set_yticklabels([])
        ax6.set_xticklabels([])
        ax6.set_xlabel( u'Number of neighbors', fontsize=30)
        divider6 = make_axes_locatable(ax6)
        cax6 = divider6.append_axes("right", "5%", pad="3%")
        
        cbar = plt.colorbar(im6, cax=cax6, cmap=cmap_neigh, norm=norm_neigh, ticks=ticks_neigh,
                                spacing='proportional', orientation='vertical')
        
        
#        cbar.ax.get_yaxis().set_ticks([])
#        for j, lab in enumerate(ticks_neigh):
#            cbar.ax.text(.5, (2 * j + 1) / 8.0, lab, ha='center', va='center')
#        
        #plt.colorbar(im6, cax=cax6, orientation='vertical', spacing = 'proportional')
        ax6.set_ylim(self.imsize[0],0)
        ax6.set_xlim(0,self.imsize[1])
        
        
        ##### Spacing histogram
        ax2 = plt.subplot(gs[2])
        
        ##### Speed histogram
        ax5 = plt.subplot(gs[5])
        
        ##### Orientation histogram
        ax8 = plt.subplot(gs[8])
        
        
        
            
        for n in np.arange(self.clusters.shape[1])[::skip]:
            thismask = ~np.isnan(self.clusters[:,n,0])
            thisdd = np.round(np.clip(self.clusters[thismask,n,:2], 0, self.imsize[0]-1)).astype(int)  ##ee in original code
            thisdist = self.distancewafer[thismask,n]
            thisvel = self.velmaginst[thismask, n]
            thisstd = self.stdwafer[thismask,n]
            thisneigh = self.neighwafer[thismask,n]
            thisangle = self.angle_inst[thismask, n]
        
            I = self.load_image_equalized(n)
            sk = I*0
            sk[thisdd[:,0], thisdd[:,1]] = 1
            sk = ~medial_axis(sk==0)
            lb = ndimage.label(sk)[0]
        
            distmap = I*np.nan
            speedmap = I*np.nan
            stdmap = I*np.nan
            anglemap = I*np.nan
            neighmap = I*np.nan
            
            for i in range(thisdd.shape[0]):
                y, x = thisdd[i]
                thislb = lb[y, x]
                patchmask = lb==thislb
                distmap[patchmask] = thisdist[i]
                speedmap[patchmask] = thisvel[i]
                stdmap[patchmask] = thisstd[i]/thisdist[i]
                neighmap[patchmask] = thisneigh[i]
                anglemap[patchmask] = thisangle[i]
        
            
            
            lb = ndimage.label(~np.isnan(distmap))[0]
            edges = np.concatenate((lb[:2,:].ravel(), lb[-2:,:].ravel(), lb[:,:2].ravel(),lb[:,-2:].ravel()))
            edges = np.unique(edges)
            distmap[np.isin( lb, edges)] = np.nan
                    
            
            lb = ndimage.label(~np.isnan(speedmap))[0]
            edges = np.concatenate((lb[:2,:].ravel(), lb[-2:,:].ravel(), lb[:,:2].ravel(),lb[:,-2:].ravel()))
            edges = np.unique(edges)
            speedmap[np.isin( lb, edges)] = np.nan
                
            
            lb = ndimage.label(~np.isnan(anglemap))[0]
            edges = np.concatenate((lb[:2,:].ravel(), lb[-2:,:].ravel(), lb[:,:2].ravel(),lb[:,-2:].ravel()))
            edges = np.unique(edges)
            anglemap[np.isin( lb, edges)] = np.nan
                     
            
            lb = ndimage.label(~np.isnan(stdmap))[0]
            edges = np.concatenate((lb[:2,:].ravel(), lb[-2:,:].ravel(), lb[:,:2].ravel(),lb[:,-2:].ravel()))
            edges = np.unique(edges)
            stdmap[np.isin( lb, edges)] = np.nan
                   
                   
            lb = ndimage.label(~np.isnan(neighmap))[0]
            edges = np.concatenate((lb[:2,:].ravel(), lb[-2:,:].ravel(), lb[:,:2].ravel(),lb[:,-2:].ravel()))
            edges = np.unique(edges)
            neighmap[np.isin( lb, edges)] = np.nan
                    
                    
            im0.set_data(I+~sk)
            im1.set_data(distmap)
            im3.set_data(stdmap*100)
            im7.set_data(anglemap*180/np.pi)
            im4.set_data(speedmap)
            im6.set_data(neighmap)
            
            
            
            
            ax2.clear()
            hdist, bdist = self.make_hist(thisdist, (120, 300))
            ax2.bar(bdist, hdist, 10)
            ax2.set_xlabel( u'Primary Spacing (um)', fontsize=20)
            
            ax5.clear()
            hvel, bvel = self.make_hist(thisvel, (10, 120))
            ax5.bar(bvel, hvel, 6)
            ax5.set_xlabel( u'apparent velocity (nm/s)', fontsize=20)
            
            ax8.clear()
            hdir, bdir = self.make_hist(thisangle*180/np.pi, (-180, 180))
            ax8.bar(bdir, hdir, 18)
            ax8.set_xlabel( u'apparent direction (ccw deg)', fontsize=20)
#        
#            
#            plt.show()
#            break
        
#        
            plt.suptitle('t={:0.3f}min, L={:0.1f}mm i={:d}, im={:s}'.format(self.times[n]/60., self.times[n]*4/1000.0, 
                                                               n, self.image_names[n].replace('_1.bmp','')[-10:]), fontsize=35)
            
            plt.savefig('{:s}basic_maps\\image_oo{:03d}.png'.format(self.result_path, n))
            
#            
            for ax in [ax0, ax1]:
                for i in ax.get_children():
                    try:
                        if 'collection' in i.get_label():
                            i.remove()
                    except: pass
        
                    try:
                         if 'Bitstream' in i.get_name():
                            i.remove()
                    except: pass  
            print n,
            
        fig.close()    
        return True    
            

        
    def make_hist(self, data, range_, bins = -1):
        """
        make_hist(data, range, bins = -1)
        
        Returns the bins and histogram values for data. 
        
        if bins is set to -1 (default), the number of bins will be automatically
        computed using the Sturge Rule
        """
        
        if bins == -1:
            #Sturge Rule
            bins = int(np.round(1 + 3.322 * np.log10(len(data))))
        
        h, b = np.histogram(data, bins = bins, range= range_, density = True)
        b = (b[1:]+b[:-1])/2
            
        return h, b    
        
        
    
    def hist_spacing(self, n, bins=-1):
        """
        >>> hist_spacing(n, bins = -1)
        
        Plots the spacing histogram of image n 
        
        if bins is set to -1 (default), the number of bins will be automatically
        computed using the Sturge Rule
        """
        
        if len(self.distancewafer) == 0: self.make_distance_wafer()
        
        if bins == -1:
            #Sturge Rule
            bins = int(np.round(1 + 3.322 * np.log10(len(self.clusters[:,n,2]))))
        
        
        h, b = np.histogram(self.clusters[:,n,2], bins = bins, range= (100, 300))
        b = (b[1:]+b[:-1])/2
        plt.figure()
        plt.xlabel(u'primary spacing (µm)')
        plt.bar(b, h)
        plt.show()
        return True


        
    def hist_neigh(self, n, bins=-1):
        """
        >>> hist_neigh(n, bins = -1)
        
        Plots the neigh histogram of image n 
        
        if bins is set to -1 (default), the number of bins will be automatically
        computed using the Sturge Rule
        """
        
        if len(self.neighwafer) == 0: self.make_distance_wafer()
        
        if bins == -1:
            #Sturge Rule
            bins = int(np.round(1 + 3.322 * np.log10(len(self.neighwafer[:,n]))))
        
        
        h, b = np.histogram(self.neighwafer[:,n], bins = bins, range= (2, 10))
        b = (b[1:]+b[:-1])/2
        plt.figure()
        plt.xlabel(u'number of neighbours')
        plt.bar(b, h)
        plt.show()
        return True
    
    
    
    def hist_abs_vel(self, n, bins=-1):
        """
        >>> hist_abs_vel(n, bins = -1)
        
        Plots the absolute velocity histogram of image n 
        
        if bins is set to -1 (default), the number of bins will be automatically
        computed using the Sturge Rule
        """
        
        if len(self.velmaginst) == 0: self.compute_abs_vel()
        
        if bins == -1:
            #Sturge Rule
            bins = int(np.round(1 + 3.322 * np.log10(len(self.velmaginst[:,n]))))
        
        
        #p0, p1 = np.percentile(self.velmaginst[:,n], 1), np.percentile(self.velmaginst[:,n], 99)
        
        h, b = np.histogram(self.velmaginst[:,n], bins = bins, range= (0, 120))
        b = (b[1:]+b[:-1])/2
        plt.figure()
        plt.xlabel(u'apparent velocity (nm/s)')
        plt.bar(b, h)
        plt.show()
        return True
    
    
    
    def hist_dir(self, n, bins=-1):
        """
        >>> hist_dir(n, bins = -1)
        
        Plots the direction histogram of image n 
        
        if bins is set to -1 (default), the number of bins will be automatically
        computed using the Sturge Rule
        """
        
        if len(self.angle_inst) == 0: self.compute_abs_vel()
        
        if bins == -1:
            #Sturge Rule
            bins = int(np.round(1 + 3.322 * np.log10(len(self.angle_inst[:,n]))))
        
        #p0, p1 = np.percentile(self.velmaginst[:,n], 1), np.percentile(self.velmaginst[:,n], 99)
        
        h, b = np.histogram(self.angle_inst[:,n], bins = bins, range= (-np.pi, np.pi))
        b = (b[1:]+b[:-1])/2
        plt.figure()
        plt.xlabel(u'apparent direction (rad)')
        plt.bar(b, h)
        plt.show()
        return True
        
        
                
        
    
    def visu(self, id_image):
        """
        >>> visu(n)
        
        Plots the image n with its skeleton mask
        
        """
        
        if id_image>=len(self.image_names):
            print('The image number spécified must be lower than the actual number of images in the studied folder ({})'.format(len(self.image_names)));
            return False
        else:
            if len(self.skeletons) == 0: self.make_skel()
        
            f, ax = plt.subplots(figsize=(15,15))
        
            I = np.array(Image(path=self.path, name=self.image_names[id_image]).In)/255.0
            J = plt.imshow(I, cmap=cm.gray, interpolation='none')
            
            #plt.scatter(subject[:,id_image,1], subject[:,id_image,0], marker = 'o', color='blue', s=30, label='gradient')
            J.set_data(I + (self.skeletons[id_image]==0))
            
            plt.show()
        
            return True
            
        
        
        
#==============================================================================
#         Grains
#==============================================================================
    
    #Do not erase. Creates images used in hand-marking of cells.
    def create_image_for_hand_marking(self, skip = 1):
        """
        create_image_for_hand_marking() or create_image_for_hand_marking(skip = 10)
        
        generate images (every 'skip value' images, default = 1) used in 
        the hand-marking of cells for the sub-grain identification
        """
               
        self.load_files()
        
        print('create images for hand marking...')
        
        oo = np.arange(self.clusters.shape[1])
        oo = np.tile(oo, (self.clusters.shape[0],1))*1.0
        tt = np.tile(self.times, (self.clusters.shape[0],1))*1.0
        
        oo[np.isnan(self.clusters[:,:,0])] = np.nan
        tt[np.isnan(self.clusters[:,:,0])] = np.nan
        
        left = np.nanmin(oo, axis=1).astype(int)
        right = np.nanmax(oo, axis=1).astype(int)
        #duration = right-left
        
        first_coord = self.clusters[np.arange(self.clusters.shape[0]),left, :2]
        last_coord = self.clusters[np.arange(self.clusters.shape[0]),right, :2]
        
        dr = last_coord-first_coord
        dt = self.times[right] - self.times[left]
        dr_dt = dr/np.tile(dt, (2,1)).T
        
        avgvelmag = np.sum(dr_dt**2, axis=1)**.5
        avgdir = np.arctan2(-dr_dt[:,0], dr_dt[:,1])
    
        
        for n in np.arange(0,self.clusters.shape[1],skip):#[300]:#range(9,529, 10):
            
            print n,
            
            I = self.load_image(n)
            II = self.load_image_equalized(n)
            
            sk = (I*0).astype(int)   
            for i in range(self.clusters.shape[0]):
                ycoord, xcoord = self.clusters[i,n,:2]*1
                if ~np.isnan(ycoord):
                    ycoord, xcoord = int(round(ycoord)), int(round(xcoord))
                    ycoord, xcoord = np.clip(ycoord, 0, self.imsize[0]-1), np.clip(xcoord, 0, self.imsize[1]-1)
                    sk[ycoord,xcoord] = 1
    
            sk = medial_axis(sk == 0)==0
            lb = ndimage.label(sk)[0]
    
            dirmap = lb*np.nan
#            dirmap2 = lb*np.nan
    
            magmap = lb*np.nan
            dr0map = lb*np.nan
            dr1map = lb*np.nan
            horsign = lb*np.nan
    
            for i in range(self.clusters.shape[0]):
                ycoord, xcoord = self.clusters[i,n,:2]*1
                if ~np.isnan(ycoord):# != -2 and durationi[i]>10:
                    ycoord, xcoord = int(round(ycoord)), int(round(xcoord))
                    ycoord, xcoord = np.clip(ycoord, 0, self.imsize[1]-1), np.clip(xcoord, 0, self.imsize[0]-1)
                    thislb = lb[ycoord, xcoord]
                    dirmap[lb==thislb] = avgdir[i]
#                    dirmap2[lb==thislb] = avgdir2[i]    ### ?
    
                    magmap[lb==thislb] = avgvelmag[i]
                    dr0map[lb==thislb] = dr_dt[i,0]
                    dr1map[lb==thislb] = dr_dt[i,1]
                    horsign[lb==thislb] = np.sign(dr_dt[i,1])
    
    
    
            for xxx, nn in ((I, 'image'), (I**2, 'image2'), (dirmap,'dirmap'), ((dirmap+180)%90, 'dirmap2'), 
                            (magmap.clip(0.001, 0.01)**2,'mag2'), (magmap.clip(0.001, 0.01),'mag1'), (dr0map, 'dr0m'), (dr1map, 'dr1map'), (horsign, 'horsign') ):
#            for xxx, nn in ((II, 'image2'), (I, 'image'),
#                            (magmap.clip(0.019, 0.025),'mag2'), (magmap.clip(0.019, 0.045),'mag'), (horsign, 'horsign')):
                oo = xxx*1
                oo = oo-np.nanmin(oo)
                oo = oo*255/np.nanmax(oo)
                oo = oo.astype(np.uint8)
                imaa = im.fromarray(oo)
                fname = '{:s}visual_aids_to_mark_grains_im{:d}_{:s}.bmp'.format(self.result_path,n,nn)
                imaa.save(fname)
    
        
        
        
        
        
        
        
        