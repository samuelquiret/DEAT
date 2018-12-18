# -*- coding: utf-8 -*-
"""
Created on Tue Dec 04 10:45:02 2018
@author: samuel.quiret

This file is the Image class, used by the DeclicAnalysis class. It deals with
the center determination mostly.

 

"""


#from Sequence import Sequence
from PIL import Image as im
import numpy as np
from skimage import exposure, morphology
import skimage
from scipy import ndimage
from skimage.draw import line_aa
import matplotlib.pyplot as plt
from pylab import cm

from functions import __verbose__
from functions import ima


#for dual threshold
from skimage.morphology import disk, medial_axis
from skimage.filters import rank

import time


__all__ = ['Image']


def just_minima(b, s=1):
    c=b*1#ndimage.gaussian_filter(b,1)
    return ((c <= np.roll(c,  s, 0)) &
            (c <= np.roll(c, -s, 0)) &
            (c <= np.roll(c,  s, 1)) &
            (c <= np.roll(c, -s, 1)) &
            (c <= np.roll(np.roll(c,  s, 0),  s, 1)) &
            (c <= np.roll(np.roll(c,  -s, 0),  s, 1)) &
            (c <= np.roll(np.roll(c,  s, 0),  -s, 1)) &
            (c <= np.roll(np.roll(c,  -s, 0),  -s, 1)))
    
def local_minima_8neigh(b, s=1, R=10):
    c = ndimage.gaussian_laplace(b, R)
    return ((c <= np.roll(c,  s, 0)) &
            (c <= np.roll(c, -s, 0)) &
            (c <= np.roll(c,  s, 1)) &
            (c <= np.roll(c, -s, 1)) &
            (c <= np.roll(np.roll(c,  s, 0),  s, 1)) &
            (c <= np.roll(np.roll(c,  -s, 0),  s, 1)) &
            (c <= np.roll(np.roll(c,  s, 0),  -s, 1)) &
            (c <= np.roll(np.roll(c,  -s, 0),  -s, 1)))    
    
    
    
    
    
    
    
def mean_curvature(Z):
    Zy, Zx  = np.gradient(Z,2)
    Zxy, Zxx = np.gradient(Zx)
    Zyy, _ = np.gradient(Zy)
    H = (Zx**2 + 1)*Zyy - 2*Zx*Zy*Zxy + (Zy**2 + 1)*Zxx
    H = -H/(2*(Zx**2 + Zy**2 + 1)**(1.5))
    return H



def myhull(lb): ## hull returns convex envelop surrounding white pixels. So this should give the figure 3.7.e
    res = lb*0
    unlb = np.unique(lb)
    unlb = unlb[unlb != 0]
    for l in unlb:
        #print l,
        foot = morphology.convex_hull_image(lb==l) 
        if np.sum(foot)<5000:
            res[foot>0] = l
    return res



class Image():
    
    def __init__(self, name = '', path = '', result_path = '', verbose = False, plot = False, fullname=''):
        
        self.separator = '\\'
        
        
        
        
        self.name = name
        self.path = path
        self.result_path = result_path
        self.fullname = fullname
        
        if self.fullname == '':
            self.fullname = '{}\\{}'.format(self.path,self.name)
            self.In = im.open('{}\\{}'.format(self.path,self.name))
        else:
            self.In = im.open('{}'.format(self.fullname))
        
        
        self.image_size = self.In.size
        self.In = np.array(self.In)
        
        
        self.centers = []
        self.centers_curvature = []
        self.centers_dualthreshold = []
        self.verbose = verbose
        self.plot = plot
        self.I = []
        
        self.fullname = '{}{}'.format(self.path,self.name)
        
        
        ### parameters
        self.sigma = 3.7 ## used for center determination --- #Basic treatment. Good enough for a good number of situations
        self.equmag_region = 61  # kernel_size
        self.black_gen_thresh_percentile = 40
        self.halflength = 25                 #### demi_longueur des traits de gradients tracés
        
        
        
        
        ### TODO
        #créer classe parametres
        
    def change_grad_thres(self, grad_thres):
        if grad_thres<>self.black_gen_thresh_percentile:
            print('new value of grad_thres: {}\t(old={})'.format(grad_thres,self.black_gen_thresh_percentile))
            self.black_gen_thresh_percentile = grad_thres
        
    def change_sigma(self, sigma):
        if sigma<>self.sigma:
            print('new value of sigma: {}\t(old={})'.format(sigma,self.sigma))
            self.sigma = sigma
        
    def change_kernel_size(self, kernel_size):
        if kernel_size<>self.equmag_region:
            print('new value of kernel_size: {}\t(old={})'.format(kernel_size,self.equmag_region))
            self.equmag_region = kernel_size
        
        
        
        
    def get_size(self):
        return len(self.In)
        
    
    def plot_image(self):
        plt.figure(figsize=(15,15), dpi=200)
        plt.imshow(self.In, interpolation='none', cmap=cm.bone)
        plt.show()
    
    def plot_centers(self):
        if len(self.centers) > 0:
            plt.figure(figsize=(15,15), dpi=200)
            plt.imshow(self.I, interpolation='none', cmap=cm.bone)
            
            if len(self.centers)>0:
                plt.scatter(self.centers[:,1], self.centers[:,0], marker = 'o', color='blue', s=30, label='gradient')
            
            if len(self.centers_curvature)>0:
                plt.scatter(self.centers_curvature[:,1], self.centers_curvature[:,0], marker='x', color='red', s=30, label='curvature')
               
                
            gradient_param = 'contrast enhancement: kernelsize ={}\nbrightness gradient: threshold = {}\nGaussian blur: sigma = {}'.format(self.equmag_region, self.black_gen_thresh_percentile, self.sigma)
            plt.text(0, -100, gradient_param, fontsize=14)
            
            plt.legend(loc='best', bbox_to_anchor=(1,1))
            plt.savefig('{}centers_{}jpeg'.format(self.result_path,self.name[:-3]))#, dpi=100)
            
            
        else:
            print('The centers have not been evaluated yet...')
        

    def find_centers_concentric(self, with_curvature = False, sigma=None, grad_thres = None, kernel_size = None, fourier = False):
        """
        find the centers using the concentric features. Default is with the 
        gradients.
        if with_curvature = True is specified, use the curvature
        
        parameters: 
            sigma = None
            grad_thres = None
            kernel_size = None
        """
        if self.verbose:
            __verbose__('find center concentric with gradients... ')
            __verbose__('This is usually good enough, but if you often miss centers, or get extra false centers. You should try to adjust this_sigma above, and play with the thresholds.')
      
        
        if sigma<>None: 
            self.change_sigma(sigma)
            
        if grad_thres<>None: 
            self.change_grad_thres(grad_thres)
            
        if kernel_size<>None: 
            self.change_kernel_size(kernel_size)
        
            
        In = self.In/255.0
        
        
        #An algorithm for local contrast enhancement, that uses histograms computed over different tile regions of the image. Local details can therefore be enhanced even in regions that are darker or lighter than most of the image.
        #kernel_size: integer or list-like, optional
            #Defines the shape of contextual regions used in the algorithm. If iterable is passed, it must have the same number of elements as image.ndim (without color channel). If integer, it is broadcasted to each image dimension. By default, kernel_size is 1/8 of image height by 1/8 of its width.
        #clip_limit : float, optional
            #Clipping limit, normalized between 0 and 1 (higher values give more contrast).
        #nbins : int, optional
            #Number of gray bins for histogram (“data range”)                                                                                                                                          
        self.I = exposure.equalize_adapthist(In, kernel_size=self.equmag_region, clip_limit=0.99, nbins = 1000)
        
        if self.plot:
            ima(self.I, title='contrast enhancement with kernel_size of {}'.format(self.equmag_region), name='{}gradient_contrast_{}jpeg'.format(self.result_path,self.name[:-3]))
        
        
        
        
        oo = ndimage.gaussian_filter(self.I, 1)#was 2
        dy, dx = np.gradient(oo, 2)#was 1
        dy, dx = ndimage.gaussian_filter(dy, 1), ndimage.gaussian_filter(dx, 1)

        #magnitude of gradient
        mag = (dx**2+dy**2)**.5
        mag = mag-np.nanmin(mag)
        mag = mag/np.nanmax(mag)

        #I equalize the magnitude.
        #equmag2 is a lazy trick to get rid of some rectangular
        #border artigacts that equalize_adapthist() introduces sometimes
        equmag = exposure.equalize_adapthist(mag, kernel_size=self.equmag_region, clip_limit=0.99, nbins = 1000)
        equmag2 = exposure.equalize_adapthist(mag[::-1,::-1], kernel_size=self.equmag_region, clip_limit=0.99, nbins = 1000)[::-1,::-1]
        equmag[equmag2>equmag]=equmag2[equmag2>equmag]
        mag = equmag


        #I only want the opinion of pixels with higher than local average gradient magnitude
        thresh = np.percentile(mag, self.black_gen_thresh_percentile)
        mask  = mag>thresh
        
        if self.plot:
            ima(mask, title='brightness gradient level curves with threshold of {}'.format(self.black_gen_thresh_percentile), name='{}gradient_edges_{}jpeg'.format(self.result_path,self.name[:-3]))
    
        #the np indexing nomenclature simplifies this masking. We get a 1D array, M-long
        #with M being the number of pixels our mask let pass
        #this array conyains the angles of the famous "white" lines, in radians.
        ang = np.arctan2(dy[mask], dx[mask])
        # 1D array with the coordinates of the white pixels. M-long
        #positionally coherent with array above.
        xycoords = np.argwhere(mask).T
        yy, xx = xycoords
        black = self.I*0.0

        #create 4 M-long 1D arrays, with the start and end x and y coords of each line
        x0 = np.round(xx-self.halflength*np.cos(ang)).astype(int)
        x1 = np.round(xx+self.halflength*np.cos(ang)).astype(int)
        y0 = np.round(yy-self.halflength*np.sin(ang)).astype(int)
        y1 = np.round(yy+self.halflength*np.sin(ang)).astype(int)
        #make sure I don't use coordinates that put me outside the image
        ylim, xlim = self.I.shape[0]-1, self.I.shape[1]-1
        #behold, the famous "painting of the lines"
        for i in np.arange(xx.size):
            rr, cc, val = line_aa(y0[i],x0[i],y1[i],x1[i])
            black[rr.clip(0, ylim), cc.clip(0, xlim)] += val
                  
        if self.plot:
            ima(black, title='gradient map', name='{}gradient_stars_{}jpeg'.format(self.result_path,self.name[:-3])) #look at the stars!
    
        #now the tricky, mean, vicious, unreliable part.
        #deciding which maxima correspond to actual cell-centers
    
       
        
        
        #use fourier peak to calculate a radius of structures       
            ##below seems to have nothing to do with what the brillant comment announces...
        #this radius will be used as a sigma to optimize smoothing
#        if fourier:
#            toka = []
#            tuku = []
#            counter=0
#            for rrr in np.linspace(2,7,50):
#                ccd = ndimage.gaussian_filter(black,rrr)
#                ccdd = just_minima(-ccd)
#                suma = np.sum(ccdd)#sk2 = skimage.morphology.medial_axis(~ccdd)
#                toka.append(rrr)
#                tuku.append(suma)
#                print counter,
#                counter+=1
#            this_sigma =  np.interp(grecy, tuku[::-1], toka[::-1])
#        
        
         #turn black im into points
        ccd = ndimage.gaussian_filter(black, self.sigma)
        
        #make a picture with white dots at black's maxima
        ccdd = just_minima(-ccd)
        #ccdd = local_minima_8neigh(-ccd, s=1, R=10)
        
           
        if self.plot:
            ima(ccd, title=r'gradient maxima with $sigma$={}'.format(self.sigma), name='{}gradient_maxima_{}jpeg'.format(self.result_path,self.name[:-3])) #look at the stars!
    
        
        #label each white dot
        lb = ndimage.label(ccdd)[0]
    
        #return a list with the center of mass of each dot.
        #these are the cell centers
        centros = ndimage.center_of_mass(lb, labels=lb, index = np.unique(lb))
        centros = np.array(centros)
    
    
        self.centers = centros
        
        if not with_curvature:
            return True
        
        else:
    
            if self.verbose:
                __verbose__('find center concentric with gradients... ')
                __verbose__('You are welcome to try this rickety method based on the brightness "curvature"\
                i.e. grooves are usually curved "up", and summits are usually curved "down"\
                we eliminate centers seen near grooves, and we add centers for unrepresented areas of "down curvature"')
          
           
            #We will first eliminate dots that are in "upward curving brightness" areas
            #use gaussian curvature to see if I missed something
            curv = mean_curvature(ndimage.gaussian_filter(self.I, self.sigma))
            lima = np.percentile(curv, 47)#42 originally #set a threshold of upwardness
            curv_para_eliminar = curv>lima
            curv_para_eliminar = morphology.remove_small_holes(curv_para_eliminar, 600)
            
            #if self.plot:
             #   __verbose__('this explains what I mean by upward curving, I mean the grooves. Because there are no centers in grooves.')
                #ima(curv_para_eliminar)  #this explains what I mean by upward curving
                                           #I mean the grooves. Because there are no centers in grooves.
          
            #this multiplication is the actuall work vv
            clean_dots1 = ccdd*curv_para_eliminar
            
            #skclean = skimage.morphology.medial_axis(~clean_dots1)
        
            #now I do kind of the opposite. I find "downward brightness curving" areas
            lima = np.percentile(curv, 70)#70 originally
            curv_para_nutrir = curv>lima
            
            #remove artifact holes and white things
            curv_para_nutrir = skimage.morphology.remove_small_holes(curv_para_nutrir, 200)
            curv_para_nutrir = skimage.morphology.remove_small_objects(curv_para_nutrir, 50)
        
        
            #the story here is that I have a picture with white dots. I label my
            #curva_para_nutrir: literally, in autoctonous spanish: "curvature mask to feed"
            #because I'll use it to fatten our point list.
            lb = ndimage.label(curv_para_nutrir)[0]
            lua = lb*1
            #I labelled my dots, and if there is already a dot in my "feeder curvature mask"
            #I erase its trace fromthe mask
            for i in np.unique(lb[clean_dots1]):
                lua[lb==i]=0
        
            #this is a vulgar but +- funcional way to add dots that might not have
            #been seen by the localminima8(). I already removed the 
            #"clean-dot"-represented mask members.
            #I increase the size of the centers representes in clean dots
            fat_dots = ndimage.binary_dilation(clean_dots1, iterations=3)
            #now I take the hulls of the members of the curv mask, while preserving their label
            #I call this lua. don't know why
            lua = myhull(lua)
            fatso = fat_dots*1
            #I take the center of mass of the
            centros = ndimage.center_of_mass(lua, labels=lua, index = np.unique(lua))
            centros = np.array(centros)
            centros = np.round(centros[1:]).clip(0,1023).astype(int)
            luaim = lb*0
            luaim[centros[:,0],centros[:,1]]=1
            luaim = ndimage.binary_dilation(luaim, iterations=3)
            #I turn it into fat dots, and add both
            maga = ndimage.binary_dilation(luaim+fatso, iterations=3)
            #This explains why I fatten the dots:
            #if they are redudant they will usually become one splotch, since the dots are fat and can 
            #get superposed. if they don't superpose, I assume it's a physical cell center
            lb = ndimage.label(maga)[0]
            centros = ndimage.center_of_mass(lb, labels=lb, index = np.unique(lb))
            
            
            
            self.centers_curvature = np.array([list(a) for a in centros])
            return True


    def find_centers_watershed_V0(self, T1):
        
        print('seuil: {}\n'.format(T1))
        
        selem = disk(30)
        I2 = rank.equalize(self.In, selem=selem)
        I2 = ndimage.gaussian_filter(I2,1)
        
        deholed = ~skimage.morphology.remove_small_objects(~(I2>T1), 40)
        
        lb = ndimage.label(deholed)[0]
        
        
        cv = myhull(lb) #### cv pour convex ?
        
        
        foot = cv-lb   ##### ??????????????????
        foot = np.clip(foot,0,3000)
        foot2 = foot>0
        foot2 = skimage.morphology.remove_small_objects(foot2, min_size=15)
        foot3 = foot2*foot
        
        lb2 = lb*1
        ufoot = np.unique(foot3)
        ufoot = ufoot[ufoot>0]
        
        plt.figure(figsize=(10,10))
        plt.imshow(foot2)
        plt.title('threshold={}'.format(T1))
        plt.savefig('N:\\rapport\\images\\watershed\\foot2_seuil={}'.format(T1))
        plt.close()
#        ## superposition
#        print('\nsuperposition\n')
#        newpart = I2>170
#        oldpart = I2>120
#        newlb = ndimage.label(newpart*(cv==ufoot[0]))
#        gonzo = newlb[0]*(cv==ufoot[0])
#        trashpalegonzo=gonzo*0
#        temp = lb*0
#        print(len(ufoot))
#        print('\nwatershed\n')
#        for l in ufoot[:]:
#            print l
#            oldtotsurf = np.sum(lb==l)
#            oldlb = ndimage.label(oldpart*(cv==l))
#            newlb = ndimage.label(newpart*(cv==l))
#            gonzo = newlb[0]*(cv==l)
#            palegonzo = gonzo>0
#        
#            for repet in range(8):
#                if np.sum(palegonzo)>=oldtotsurf: break
#                palegonzo = ndimage.binary_dilation(palegonzo)
#            newnewlabels = morphology.watershed(-ndimage.distance_transform_edt(palegonzo), gonzo, mask=palegonzo)
#            
#            gonzotofill = np.max(lb2)+1+(newnewlabels)
#            gonzotofill[newnewlabels==0]=0
#            trashpalegonzo+=gonzotofill    
#            gonzotofill = myhull(gonzotofill)
#        
#            temp[cv==l] += newnewlabels[cv==l]
#            lb2[cv==l] = gonzotofill[cv==l]
#            #if oldlb[1] != newlb[1]:
#            #    temp = temp +newlb[0]
#        
#        lb=lb2*0
#        unlb = np.unique(lb2)[1:]
#        for i in range(unlb.size):
#            mask = lb2==unlb[i]
#            if 15<np.sum(mask)<5000: lb[mask] = i+1
#        
#        
#        
#        
#        return lb, lb2




    def find_centers_watershed_V1(self, t1 = 140, t2 = 170):
        
        """
        find the centers using the dual threshold technique.
        
        
        parameters: 
            t1: should be about the 50th percentile
            t2: should be about the 75th percentile
        """
        
        start = time.time()
        
        
        print('seuil 1: {}\nseuil 2: {}\n\n'.format(t1,t2))
        
        I = np.array(self.In)/255.0
        
        
        selem = disk(30)
        I2 = rank.equalize(I, selem=selem)
        I2 = ndimage.gaussian_filter(I2,1)
        deholed = ~skimage.morphology.remove_small_objects(~(I2>t1), 40)
        
        lb = ndimage.label(deholed)[0]
        


        foot1 = I2>np.median(I2)
        foot2 = I2>np.median(I2[foot1])
        foot3 = skimage.morphology.remove_small_objects(foot2, 20)
        foot3 = ndimage.binary_erosion(foot3, iterations=1) ### looks like Mq
        
        l1 = ndimage.label(foot1)[0]
        l2 = ndimage.label(foot2)[0]
        
#        h, bins = np.histogram(I2,100)
#        plt.plot(bins[1:], h)
#        
#        kk = I*0
#        unlb1 = np.unique(l1)
#        maxo = l1*l2
#        for u1 in unlb1:
#            thisgonzo1 = l1==u1
#            if np.sum(thisgonzo1)>5000: continue
#            thisgonzo2 = l2*[l1==u1]
#            if u1>400: break
        
        start_hull = time.time()
        cv = myhull(lb)
        end_hull = time.time()
        
        foot = cv-lb
        foot = np.clip(foot,0,3000)
        foot2 = foot>0
        foot2 = skimage.morphology.remove_small_objects(foot2, min_size=15)
        foot3 = foot2*foot   ### tells where there is possibly merging of cells?
        
        
        
#        plt.imshow(cv)
#        plt.title('c_convex_hull')
#        plt.savefig('{}c_convex_hull.png'.format(rpath))
#
#        plt.imshow(foot3)
#        plt.title('d_foot3')
#        plt.savefig('{}d_foot3.png'.format(rpath))
        
        
        lb2 = lb*1
        ufoot = np.unique(foot3)
        ufoot = ufoot[ufoot>0]
        
        newpart = I2>t2
        oldpart = I2>t1
        
        #temp = lb*0
        
        #random.shuffle(ufoot)
        start_watershed = time.time()
        for l in ufoot[:]:
            oldtotsurf = np.sum(lb==l)
            newtotsurf = np.sum((lb==l)*newpart)
            #threshtotsurf = (oldtotsurf+newtotsurf)/2.0
            
            oldlb = ndimage.label(oldpart*(cv==l))
            newlb = ndimage.label(newpart*(cv==l))
            gonzo = newlb[0]*(cv==l)
            palegonzo = gonzo>0
        
            for repet in range(8):
                if np.sum(palegonzo)>=oldtotsurf: break
                palegonzo = ndimage.binary_dilation(palegonzo)
                
            #return palegonzo, gonzo, lb, l, cv
            newnewlabels = morphology.watershed(-ndimage.distance_transform_edt(palegonzo), gonzo, mask=palegonzo)
            
            gonzotofill = np.max(lb2)+1+(newnewlabels) ## make new labels
            
                                
            gonzotofill[newnewlabels==0]=0 
        
        
        
            gonzotofill = myhull(gonzotofill)
        
            #temp[cv==l] += newnewlabels[cv==l]
            lb2[cv==l] = gonzotofill[cv==l]
            #if oldlb[1] != newlb[1]:
            #    temp = temp +newlb[0]
        
        end_watershed = time.time()
        
        lb=lb2*0
        unlb = np.unique(lb2)[1:]
        for i in range(unlb.size):
            mask = lb2==unlb[i]
            if 15<np.sum(mask)<5000: lb[mask] = i+1
        
        
        unlb, lbcts = np.unique(lb, return_counts=True)
        unlb, lbcts = unlb[1:], lbcts[1:]
        c = ndimage.center_of_mass(lb,lb, unlb)
        c = np.array(c)
        
        
        sk = (I*0).astype(int)
        for i in range(len(c)):
                ycoord, xcoord = c[i,0], c[i,1]
                if ~np.isnan(ycoord):
                    ycoord, xcoord = int(round(ycoord)), int(round(xcoord))
                    ycoord, xcoord = np.clip(ycoord, 0, 1023), np.clip(xcoord, 0, 1023)
                    #ax.text(xcoord, ycoord,'{:d}'.format(i), color='#33FFFF', ha='center', va='center')  # print cell number
                    sk[ycoord,xcoord] = 1
        sk = medial_axis(sk == 0) == 0
          
        plt.figure(figsize=(20,20))
        II = exposure.equalize_adapthist(I, kernel_size=61, clip_limit=0.99, nbins = 1000)
        plt.imshow(II + (sk==0).astype(int))
        plt.scatter(c[:,1], c[:,0], color='red', s=5)
        plt.title('dual threshold_t1={}__t2={}'.format(t1,t2))
        plt.savefig('{}dual_threshold_final_t1={}__t2={}.png'.format(rpath,t1,t2))
        
        end = time.time()
        
        print('\ttime hull: {:.4}s\n\ttime watershed: {:.4}s\n\ttime total: {:.4}s\n'.format(end_hull - start_hull, end_watershed - start_watershed, end - start))
        
        self.centers_dualthreshold = c
        
        return end_hull - start_hull, end_watershed - start_watershed, end - start
#