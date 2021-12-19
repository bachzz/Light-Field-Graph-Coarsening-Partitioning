#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 15:52:26 2020

@author: mrizkall
"""

#For Segmentation
from sklearn import mixture,cluster
from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
import time
# Graph coarsening
#from libraries.coarsening_utils import *
#import libraries.graph_utils
import sys
# Numpy, Scipy, Labeling
import math
import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.sparse as ss
import scipy.ndimage.measurements as labeling

#import collections as cl
#import hdf5storage as hdf5
import scipy.io as sio
from skimage import data, segmentation, color
from skimage.future import graph
from matplotlib import pyplot as plt

# open cv
import cv2 as cv2
# PYGSP, Networks
#import networkx as nx
import pygsp as gsp

# For visualization
#import matplotlib as matplot
import matplotlib.pyplot as plt
from matplotlib.pyplot import pause

from mpl_toolkits.mplot3d import Axes3D

gsp.plotting.BACKEND = 'matplotlib'
#%% Global Variables
FREEMAN_DICT = {-90:'4', -45:'3', 0:'2', 45:'1', 90:'0', 135:'7', 180:'6', -135:'5'}
ALLOWED_DIRECTIONS = np.array([0, 45, 90, 135, 180, -45, -90, -135])


def bbox(mask):
    a = np.where(mask != 0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    return bbox

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def ismember(a, b):
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = i
            
    LOCB = [bind.get(itm, np.nan) for itm in a]
    Lia = ~np.isnan(LOCB)  
      
    return Lia,  LOCB  # None can be replaced by any other "not in b" value


def find_nearest(self, array, value):
    '''
    Find the nearest element of array to the given value
    '''
    idx = (np.abs(array-value)).argmin()
    return array[idx]


def extents(f):
  delta = f[1] - f[0]
  return [f[0] - delta/2, f[-1] + delta/2]


#%% Function to project labels after performing Normalized cut
def Projection_labels(Segmentation_ref, mask, Disparity_ref, compute_color = 0, LF_RGB = [np.nan]):
    # segmentation before 4 D
    # mask 4D 
    # disp 2D
    # Segmentation projection from top-left corner view to the whole LF
    
    ##Mref = 0
    ##Nref = 0
    ##Segmentation_first_view = Segmentation_before
    [M,N,X,Y] = mask.shape
    #disparity_first_view = disp

    ### 
    Segmentation_dict = Segmentation_ref # dict
    Disparity_dict = Disparity_ref
    Mref_sequence = [0,4]
    Nref_sequence = [0,4]
    K = 3 # number of views to be projected in a local neighborhood = 4-1
    ###
    
    #LabelList = np.unique(Segmentation_first_view[np.squeeze(mask[0,0,:,:])])
    ###
    LabelList_dict = {}
    DepthSquence = np.zeros(mask.shape) + np.inf
    LabelSquence = np.zeros(mask.shape) + np.nan

    for view_id in Segmentation_dict:
        # e.g: view_id = '0_1'
        i_ref = int(view_id.split('_')[0])
        j_ref = int(view_id.split('_')[1])
        #print(i_ref,j_ref)
        LabelList_dict[view_id] = np.unique(Segmentation_dict[view_id][np.squeeze(mask[i_ref,j_ref,:,:])])
        LabelSquence[i_ref, j_ref, :, :] = Segmentation_dict[view_id]
        DepthSquence[i_ref, j_ref, :, :] = Disparity_dict[view_id]
    ###

    # First view
    ##DepthSquence = np.zeros(mask.shape) + np.inf
    ##LabelSquence = np.zeros(mask.shape) + np.nan
    ##LabelSquence[Mref, Nref, :, :] = Segmentation_first_view
    ##DepthSquence[Mref, Nref, :, :] = disp.astype('float')
     

    # reference depth
    #Depth_ref = disparity_first_view
    ##disp_copy = disp.copy()
    ##disparity_values = []
    # calcul du depth reference pr les super-rayons dans la premiere vue
    ##for LabelInd in LabelList:
    ##    temp_depth = disp_copy[Segmentation_first_view==LabelInd];
    ##    temp_depth = np.median(temp_depth)
    ##    disp_copy[Segmentation_first_view==LabelInd] = temp_depth;
    ##    disparity_values.append(temp_depth)
    
    ##DepthSquence[0,0,:,:] = disp_copy
    
    disparity_values = {};

    for view_id in Segmentation_dict:
        i_ref = int(view_id.split('_')[0])
        j_ref = int(view_id.split('_')[1])

        disparity_values[view_id] = []
        
        disp_med = Disparity_dict[view_id]#disp.copy()
        
        # calcul du depth reference pr les super-rayons dans la premiere vue
        for LabelInd in LabelList_dict[view_id]:
            temp_depth = disp_med[Segmentation_dict[view_id]==LabelInd];
            temp_depth = np.median(temp_depth)
            disp_med[Segmentation_dict[view_id]==LabelInd] = temp_depth;
            disparity_values[view_id].append(temp_depth)
        
        DepthSquence[i_ref,j_ref,:,:] = disp_med


    ###
    for m in range(M):

        # case 1: m in Mref_sequence
        if (m in Mref_sequence):

            # update Mref
            Mref = m

            # for each n_ref, project to K neighbor views
            for n_ref in Nref_sequence:

                # update Nref
                Nref = n_ref

                disp_temp = np.squeeze(DepthSquence[Mref,Nref,:,:]) # median disp in that ref
                view_id_ref = f'{Mref}_{Nref}'

		####
                views_count = K
                if (n_ref == Nref_sequence[-1]):
                    views_count = K+1
		####

                for n in range(n_ref+1, n_ref+views_count+1):
                    # legal indices in view m,n
                    mask_viewmn = np.squeeze(mask[m,n,:,:])
                    legal_ind = np.nonzero(mask_viewmn == True)
                    legal_ind = np.ravel_multi_index([legal_ind[0], legal_ind[1]], (X,Y))

                    for LabelInd in LabelList_dict[view_id_ref]:
                        # temp depth of the superrays
                        temp_depth = disp_temp[Segmentation_dict[view_id_ref]==LabelInd]
                        [rx, ry] = (Segmentation_dict[view_id_ref]==LabelInd).nonzero()
                        temp_depth = np.median(temp_depth)
                        if (np.isnan(temp_depth) or np.isinf(temp_depth)):
                            print("infinite value encountered in 1st column")
                        # projection from first view to the horizontal neighbors next to it
                        temp_LabelSquence = np.squeeze(LabelSquence[m,n,:,:]);
                        temp_DepthSquence = np.squeeze(DepthSquence[m,n,:,:]);
                        # reference
                        ##Mref = m
                        ##Nref = n_ref
                        # target
                        Mtar = m
                        Ntar = n
                        # projection
                        nx = (Mtar-Mref)*temp_depth
                        ny = (Ntar-Nref)*temp_depth
                        nx = np.round(rx + nx).astype('int64')
                        ny = np.round(ny + ry).astype('int64')
                        
                        # Out of range not included in original Super-rays
                        illegal_ind_mask11 = np.logical_or(nx<0,nx>(X-1))
                        illegal_ind_mask12 = np.logical_or(ny<0,ny>(Y-1))
                        illegal_ind_mask1 = np.logical_or(illegal_ind_mask11,illegal_ind_mask12)
                        nx = nx[~illegal_ind_mask1]
                        ny = ny[~illegal_ind_mask1]
                        
                        # occlusion detection
                        occlusion_depth = temp_DepthSquence[nx, ny]
                        illegal_ind_mask2 = occlusion_depth<temp_depth
                        
                        nx = nx[~illegal_ind_mask2]
                        ny = ny[~illegal_ind_mask2]
                        temp_LabelSquence[nx, ny] = LabelInd
                        temp_DepthSquence[nx, ny] = temp_depth

                    Cru_Label = temp_LabelSquence
                    Cru_Depth = temp_DepthSquence

                    print('Horizontal fill\n')
                    # horizontal fill
                    for x in range(X):
                        x_Label = Cru_Label[x, :]
                        x_Depth = Cru_Depth[x, :]
                        x_Binary = np.isinf(x_Depth)
                        x_Binary = labeling.label(x_Binary) 
                        for ind_disocc in range(1, np.max(x_Binary[0])+1):
                            if np.sum(x_Binary[0]==ind_disocc)>(Y-1):
                                continue
                            x_Binary_temp = np.array(x_Binary[0]==ind_disocc, dtype=bool)
                            x_Binary_temp = x_Binary_temp.nonzero()
                            x_Binary_temp = np.array([np.min(x_Binary_temp[0])-1, np.max(x_Binary_temp[0])+1])
                            x_Binary_temp = x_Binary_temp[x_Binary_temp>=0]
                            x_Binary_temp = x_Binary_temp[x_Binary_temp<=Y-1]
                            
                            temp_Depth = x_Depth[x_Binary_temp]
                            temp_Label = x_Label[x_Binary_temp]
                            
                        
                            if temp_Depth.size!=0:
                                temp_Depth = np.max(temp_Depth[~np.isinf(temp_Depth)])
                                i = np.argmax(temp_Depth[~np.isinf(temp_Depth)])
                                temp_Label = temp_Label[i]
                                x_Depth[x_Binary[0]==ind_disocc] = temp_Depth
                                x_Label[x_Binary[0]==ind_disocc]= temp_Label
                            else:
                                # vertical fill if we have disocclusions in the
                                # bottom part of the image
                                x_Depth[x_Binary[0]==ind_disocc] = Cru_Depth[x-1,x_Binary[0]==ind_disocc]
                                x_Label[x_Binary[0]==ind_disocc] = Cru_Label[x-1,x_Binary[0]==ind_disocc]
                                
                                
                        Cru_Label[x,:] = x_Label
                        Cru_Depth[x,:] = x_Depth
                            
                    DepthSquence[m][n][mask_viewmn] = Cru_Depth[mask_viewmn]
                    LabelSquence[m][n][mask_viewmn] = Cru_Label[mask_viewmn]

        # case 2: m not in Mref_sequence
        else:
            for n_ref in Nref_sequence:
                
                # update Nref
                Nref = n_ref

                mask_viewmn = np.squeeze(mask[m,Nref,:,:])

                disp_temp = np.squeeze(DepthSquence[Mref,Nref,:,:]) # median disp in that ref
                view_id_ref = f'{Mref}_{Nref}'

                # project from nearest reference view above
                for LabelInd in LabelList_dict[view_id_ref]:
                    temp_depth = disp_temp[np.squeeze(LabelSquence[Mref,Nref,:,:])==LabelInd]
                    [rx, ry] = (np.squeeze(LabelSquence[Mref,Nref,:,:])==LabelInd).nonzero()
                    temp_depth = np.median(temp_depth)
                    if (np.isnan(temp_depth) or np.isinf(temp_depth)):
                        print("infinite value encountered in vertical projection")
                    #projection on the neighboring views in the vertical axis
                    temp_LabelSquence = np.squeeze(LabelSquence[m,Nref,:,:]);
                    temp_DepthSquence = np.squeeze(DepthSquence[m,Nref,:,:]);
                    
                    ##Mref = 0
                    ##Nref = 0
                    Mtar = m
                    Ntar = Nref

                    nx = (Mtar-Mref)*temp_depth
                    ny = (Ntar-Nref)*temp_depth
                    nx = np.round(rx + nx).astype('int64')
                    ny = np.round(ny + ry).astype('int64')

                    #Out of range not included in original Super-rays
                    illegal_ind_mask11 = np.logical_or(nx<0,nx>(X-1))
                    illegal_ind_mask12 = np.logical_or(ny<0,ny>(Y-1))
                    illegal_ind_mask1 = np.logical_or(illegal_ind_mask11,illegal_ind_mask12)
                    nx = nx[~illegal_ind_mask1]
                    ny = ny[~illegal_ind_mask1]
                    
                    # occlusion detection
                    occlusion_depth = temp_DepthSquence[nx, ny]
                    illegal_ind_mask2 = occlusion_depth<temp_depth
                    nx = nx[~illegal_ind_mask2]
                    ny = ny[~illegal_ind_mask2]
                    temp_LabelSquence[nx, ny] = LabelInd
                    temp_DepthSquence[nx, ny] = temp_depth
                    
                Cru_Label = temp_LabelSquence
                Cru_Depth  = temp_DepthSquence
                #vertical fill
                print('Vertical fill\n')
                for y in range(Y):
                    y_Label = Cru_Label[:,y]
                    y_Depth = Cru_Depth[:,y]
                    y_Binary = np.isinf(y_Depth)
                    y_Binary = labeling.label(y_Binary)
                    
                    for ind_disocc in range(1, np.max(y_Binary[0])+1):
                        if np.sum(y_Binary[0]==ind_disocc)==(X-1):
                            continue
                        y_Binary_temp = np.array(y_Binary[0]==ind_disocc, dtype=bool)
                        y_Binary_temp = y_Binary_temp.nonzero()                     
                        y_Binary_temp = np.array([np.min(y_Binary_temp[0])-1, np.max(y_Binary_temp[0])+1])
                        y_Binary_temp = y_Binary_temp[y_Binary_temp>=0]
                        y_Binary_temp = y_Binary_temp[y_Binary_temp<=X-1]
                        
                        temp_Depth = y_Depth[y_Binary_temp]
                        temp_Label = y_Label[y_Binary_temp]
                        
                        if temp_Depth.size!=0:
                            temp_Depth = np.max(temp_Depth[~np.isinf(temp_Depth)])
                            i = np.argmax(temp_Depth[~np.isinf(temp_Depth)])
                            temp_Label = temp_Label[i]
                            y_Depth[y_Binary[0]==ind_disocc] = temp_Depth
                            y_Label[y_Binary[0]==ind_disocc] = temp_Label
                        else:
                            #Horizontal fill 
                            y_Depth[y_Binary[0]==ind_disocc] = Cru_Depth[y_Binary[0]==ind_disocc,y-1]
                            y_Label[y_Binary[0]==ind_disocc] = Cru_Label[y_Binary[0]==ind_disocc,y-1]
                    Cru_Label[:,y] = y_Label
                    Cru_Depth[:,y] = y_Depth

                DepthSquence[m][Nref][mask_viewmn] = Cru_Depth[mask_viewmn]
                LabelSquence[m][Nref][mask_viewmn] = Cru_Label[mask_viewmn]


                # for each n_ref, project to K neighbor views

                ####
                views_count = K
                if (n_ref == Nref_sequence[-1]):
                    views_count = K+1
                ####
                for n in range(n_ref+1, n_ref+views_count+1):
                    mask_viewmn = np.squeeze(mask[m,n,:,:])
                    disp_temp = np.squeeze(DepthSquence[m, Nref, :, :])
                    for LabelInd in LabelList_dict[view_id_ref]:
                        temp_depth = disp_temp[np.squeeze(LabelSquence[m, Nref, :, :]) == LabelInd]
                        [rx, ry] =(np.squeeze(LabelSquence[m, Nref, :, :])==LabelInd).nonzero()
                        temp_depth = np.median(temp_depth)
                        if (np.isnan(temp_depth) or np.isinf(temp_depth)):
                            print("infinite value encountered in other columns")
                        # projection on the neighboring views in the horizontal axis
                        temp_LabelSquence = np.squeeze(LabelSquence[m,n,:,:]);
                        temp_DepthSquence = np.squeeze(DepthSquence[m,n,:,:]);
                        # reference
                        ##Mref = m
                        ##Nref = 0
                        # target
                        Mtar = Mref ##m
                        Ntar = n
    
                        # projection
                        nx = (Mtar-Mref)*temp_depth
                        ny = (Ntar-Nref)*temp_depth
                        nx = np.round(rx + nx).astype('int64')
                        ny = np.round(ny + ry).astype('int64')
                            
                        # Out of range not included in original Super-rays
                        illegal_ind_mask11 = np.logical_or(nx<0,nx>(X-1))
                        illegal_ind_mask12 = np.logical_or(ny<0,ny>(Y-1))
                        illegal_ind_mask1 = np.logical_or(illegal_ind_mask11,illegal_ind_mask12)
                        nx = nx[~illegal_ind_mask1]
                        ny = ny[~illegal_ind_mask1]
                        
                        # occlusion detection
                        occlusion_depth = temp_DepthSquence[nx, ny]
                        illegal_ind_mask2 = occlusion_depth<temp_depth
                        #illegal_ind_mask2 = occlusion_depth>temp_depth
                        nx = nx[~illegal_ind_mask2]
                        ny = ny[~illegal_ind_mask2]
                        temp_LabelSquence[nx, ny] = LabelInd
                        temp_DepthSquence[nx, ny] = temp_depth
                            
                        LabelSquence[m,n,:,:] = temp_LabelSquence
                        DepthSquence[m,n,:,:] = temp_DepthSquence
                        
                    Cru_Label = temp_LabelSquence
                    Cru_Depth = temp_DepthSquence
                    print('Horizontal fill\n')
                    # horizontal fill
                    for x in range(X):
                        x_Label = Cru_Label[x, :]
                        x_Depth = Cru_Depth[x, :]
                        x_Binary = np.isinf(x_Depth)
                        x_Binary = labeling.label(x_Binary) 
                        for ind_disocc in range(1, np.max(x_Binary[0])+1):
                            if np.sum(x_Binary[0]==ind_disocc)>(Y-1):
                                continue
                            x_Binary_temp = np.array(x_Binary[0]==ind_disocc, dtype=bool)
                            x_Binary_temp = x_Binary_temp.nonzero()
                            x_Binary_temp = np.array([np.min(x_Binary_temp[0])-1, np.max(x_Binary_temp[0])+1])
                            x_Binary_temp = x_Binary_temp[x_Binary_temp>=0]
                            x_Binary_temp = x_Binary_temp[x_Binary_temp<=Y-1]
                            
                            temp_Depth = x_Depth[x_Binary_temp]
                            temp_Label = x_Label[x_Binary_temp]
                            if temp_Depth.size!=0:
                                temp_Depth = np.max(temp_Depth[~np.isinf(temp_Depth)])
                                i = np.argmax(temp_Depth[~np.isinf(temp_Depth)])
                                temp_Label = temp_Label[i]
                                x_Depth[x_Binary[0]==ind_disocc] = temp_Depth
                                x_Label[x_Binary[0]==ind_disocc]= temp_Label
                            else:
                                # vertical fill if we have disocclusions in the
                                # bottom part of the image
                                x_Depth[x_Binary[0]==ind_disocc] = Cru_Depth[x-1,x_Binary[0]==ind_disocc]
                                x_Label[x_Binary[0]==ind_disocc] = Cru_Label[x-1,x_Binary[0]==ind_disocc]
                                
                        Cru_Label[x,:] = x_Label
                        Cru_Depth[x,:] = x_Depth
    
                    DepthSquence[m][n][mask_viewmn] = Cru_Depth[mask_viewmn]
                    LabelSquence[m][n][mask_viewmn] = Cru_Label[mask_viewmn]

                    
    DepthSquence[~mask] = np.nan
    LabelSquence[~mask] = np.nan
    
    ##########################################
    ConMatrixCell = []
    PxlIndCell = []
    
    if compute_color == 1:
        ColorMatrixCell = []
    
    ###
    LabelList = np.array([], dtype=int)
    for view_id in LabelList_dict:
        LabelList = np.append(LabelList, LabelList_dict[view_id])
    LabelList = np.unique(LabelList)
    ###

    for label in LabelList:
        print(['label=%d out of %d\n', label, np.max(LabelList)])
        num = np.sum(LabelSquence==label)
        mask__ = np.array(LabelSquence==label, dtype=bool)
        coords = mask__.nonzero()
        GIndex = (np.array(LabelSquence.ravel()==label, dtype=bool)).nonzero()
        mIndex = coords[0]
        nIndex = coords[1]
        GIndex = GIndex[0]
        if compute_color == 1:
            ColorMatrix = np.zeros((num, 3))
            for ind in range(num):
                [m,n,x,y] = np.unravel_index(GIndex[ind],np.shape(LabelSquence))
                ColorMatrix[ind][:] = np.squeeze(LF_RGB[m,n,x,y,:])
        
        ## Spatial connections
        Block_matrix_sp = []
        t = time.time()
        #%%
        for ind_view in range(M*N):
            [m,n] = np.unravel_index(ind_view,(M,N))
            MASK_sp = np.array(LabelSquence[m][n][:]==label,dtype=bool)
            MASK_coords = MASK_sp.nonzero()
            x_sp = MASK_coords[0]
            y_sp = MASK_coords[1]
            mask_sp = np.array(LabelSquence[m][n][:]==label,dtype=bool)
            print(len(x_sp))
            if len(x_sp)!=0:
                # bounding box computation
                x_sp_min = np.min(x_sp);
                x_sp_max = np.max(x_sp);
                y_sp_min = np.min(y_sp);
                y_sp_max = np.max(y_sp);
                bounding_box = mask_sp[x_sp_min:x_sp_max+1,y_sp_min:y_sp_max+1]
                # Graph 2d-grid using gsp_2dgrid for that add gsp to toolboxes needed
                shape_bbox = np.shape(bounding_box)
                N1 = shape_bbox[0]
                N2 = shape_bbox[1]
                if N1!=1 and N2 !=1:
                    graph_sp = gsp.graphs.Grid2d(N1, N2)
                else:
                    if N1 == 1: graph_sp = gsp.graphs.Path(N2)
                    else: graph_sp = gsp.graphs.Path(N1)
                     
                # adjacency in a view
                Adjacency_sp = graph_sp.A;
                idx_keep = np.nonzero(np.ravel(bounding_box))[0]
                print(len(idx_keep))
                Adjacency_sp = Adjacency_sp[idx_keep,:]
                Adjacency_sp = Adjacency_sp[:,idx_keep]
                # block matrix
                Block_matrix_sp.append(Adjacency_sp);
                print(Adjacency_sp.shape[0])
            else:
                continue
                
        #%%        
        ##############################33
        Adjacency_spatial = ss.block_diag(Block_matrix_sp); 
        elapsed = time.time() - t
        print(['Time taken for blkdiag is ', elapsed]);
        t = time.time()
        
        ## Angular connections
        ConnectionInd1 =[]
        ConnectionInd2 =[]
        
        for ind_view in range(M*N):
            coords_view = np.unravel_index(ind_view,(M,N))
            # m and n for view
            m = coords_view[0]
            n = coords_view[1]   
            depth_view = DepthSquence[m][n][:];    
            MASK_sp = np.array(LabelSquence[m][n][:]==label,dtype=bool)
            MASK_coords = MASK_sp.nonzero()
            # x and y for pixels inside view
            x_sp = MASK_coords[0]
            y_sp = MASK_coords[1]
            Index_GIndex_vue = np.logical_and(mIndex == m, nIndex == n).nonzero();
            # top, bttom, left and right views
            mT=m-1
            mB=m+1
            nL=n-1
            nR=n+1
            mlist=[m, m, mT, mB]
            nlist=[nL, nR, n, n]
            if len(x_sp)!=0:
                # Projections in all directions up, down, left and right
                depth = np.median(depth_view[LabelSquence[m][n][:]==label])
                # Connections in the case where the values exist in the SP and not outside the borders
                mList = np.full((len(x_sp),4),-1)
                nList = np.full((len(x_sp),4),-1)
                dxList = np.full((len(x_sp),4),-1)
                dyList = np.full((len(x_sp),4),-1)
                for counter in range(len(mlist)):
                    mList[:,counter] = mlist[counter]
                    nList[:,counter] = nlist[counter]
                    if ((depth/(M-1)-np.floor(depth/(M-1)))!=0.5):
                        dxList[:,counter] = np.round(x_sp+(mlist[counter]-m)*depth/(M-1))
                        dyList[:,counter] = np.round(y_sp+(nlist[counter]-n)*depth/(N-1))
                    else:
                        offset = 0.001
                        dxList[:,counter] = np.round(x_sp+(mlist[counter]-m)*(depth+offset)/(M-1))
                        dyList[:,counter] = np.round(y_sp+(nlist[counter]-n)*(depth+offset)/(N-1))

                for counter in range(len(mlist)):
                    m_projected = mList[:,counter]
                    n_projected = nList[:,counter]
                    x_projected = dxList[:,counter]
                    y_projected = dyList[:,counter]
                    or_1 = np.logical_or(m_projected<0,m_projected>M-1)
                    or_2 = np.logical_or(n_projected<0,n_projected>N-1)
                    or_3 = np.logical_or(x_projected<0,x_projected>X-1)
                    or_4 = np.logical_or(y_projected<0,y_projected>Y-1)
                    or_or = np.logical_or(or_1,or_2)
                    or_or_2 = np.logical_or(or_3,or_4)
                    or_or_or = np.logical_or(or_or,or_or_2)
                    m_projected = m_projected[or_or_or==False]
                    n_projected = n_projected[or_or_or==False]
                    x_projected = x_projected[or_or_or==False]
                    y_projected = y_projected[or_or_or==False]
                    GIndex_projected = np.ravel_multi_index([m_projected, n_projected, x_projected, y_projected], np.shape(LabelSquence));
                    if len(GIndex_projected)!=0:
                        [Lia, Index_GIndex_projected] = ismember(GIndex_projected, GIndex)
                        Index_GIndex_projected=np.asarray(Index_GIndex_projected)
                        Index_GIndex_vue_ = Index_GIndex_vue[0][Lia]               
                        ConnectionInd1.extend(Index_GIndex_vue_);
                        ConnectionInd2.extend(Index_GIndex_projected[Lia]);
                
            else:
                continue
        ConnectionInd3 = []
        ConnectionInd3.extend(ConnectionInd1)
        ConnectionInd3.extend(ConnectionInd2)
        ConnectionInd4 = []
        ConnectionInd4.extend(ConnectionInd2);
        ConnectionInd4.extend(ConnectionInd1);

        elapsed = time.time() - t
        
        Adjacency_ang = ss.coo_matrix((np.ones(len(ConnectionInd3)), (ConnectionInd3, ConnectionInd4)), shape=np.shape(Adjacency_spatial));
        #Adjacency_ang = np.multiply(Adjacency_ang>0, 1)
        print(['Time taken for angular connections is ' ,elapsed]);
        
        ConMatrix = Adjacency_spatial+Adjacency_ang
        ConMatrix = np.multiply(ConMatrix>0, 1)
        if compute_color == 1:
            ColorMatrixCell.append(ColorMatrix)
        
        ConMatrixCell.append(ConMatrix)        
        PxlIndCell.append(GIndex)
    if compute_color == 0:
        return LabelSquence, disparity_values, ConMatrixCell, PxlIndCell
    else:
        return LabelSquence, disparity_values, ConMatrixCell, ColorMatrixCell, PxlIndCell


## parameters
dataset = sys.argv[1]
num_SR = int(sys.argv[2])
## RGB and disparity
data_root_path = 'Test_datasets/'
#LF_original = sio.loadmat('/temp_dd/igrida-fs1/mrizkall/Dataset_graphs/'+dataset+'/QP0/RGB.mat',variable_names=['RGB'])
#Disp_original = sio.loadmat('/temp_dd/igrida-fs1/mrizkall/Dataset_graphs/'+dataset+'/QP0/Disparity.mat',variable_names=['Disparity'])
LF_original = sio.loadmat(data_root_path+dataset+'/QP0/RGB.mat',variable_names=['RGB'])
Disp_original = sio.loadmat(data_root_path+dataset+'/QP0/Disparity.mat',variable_names=['Disparity'])  #'Disparity'])

#num_SR = 1000

#LF_original = sio.loadmat('/home/mrizkall/Desktop/Project_Graph_Code/Datasets_graphs/Friends/QP0/RGB.mat',variable_names=['RGB'])
#Disp_original = sio.loadmat('/home/mrizkall/Desktop/Project_Graph_Code/Datasets_graphs/Friends/QP0/Disparity.mat',variable_names=['Disparity'])

LF_original = LF_original['RGB']
print(Disp_original)
##Disp = Disp_original['disp']  #['Disparity']
Disp = Disp_original['Disparity'].copy()

M = LF_original.shape[0]
N = LF_original.shape[1]
X = LF_original.shape[2]
Y = LF_original.shape[3]

img_0_0 = np.squeeze(LF_original[0,0,:,:,:])
img_0_4 = np.squeeze(LF_original[0,4,:,:,:])
img_4_0 = np.squeeze(LF_original[4,0,:,:,:])
img_4_4 = np.squeeze(LF_original[4,4,:,:,:])

#disp = np.squeeze(Disp[0,0,:,:]) #puisque maintenant on a juste la disparité de la première vue en entrée
##disp = Disp

labels_0_0 = segmentation.slic(img_0_0, compactness=30, n_segments=num_SR)
labels_0_4 = segmentation.slic(img_0_4, compactness=30, n_segments=num_SR)
labels_4_0 = segmentation.slic(img_4_0, compactness=30, n_segments=num_SR)
labels_4_4 = segmentation.slic(img_4_4, compactness=30, n_segments=num_SR)
#out1 = color.label2rgb(labels1, img_1_1, kind='avg')
#img_1_1 = 255*img_1_1;
#img = img.astype(int);

#filename = '/temp_dd/igrida-fs1/mrizkall/Dataset_graphs/GBT_LF_'+dataset+'_0QP_64views_'+str(num_SR)+'numSR_original.mat'
filename = 'Results/GBT_LF_'+dataset+'_0QP_'+str(M*N)+'views_'+str(num_SR)+'numSR_original.mat'

Segmentation_ref = {
    '0_0': labels_0_0,
    '0_4': labels_0_4,
    '4_0': labels_4_0,
    '4_4': labels_4_4
}

Disparity_ref = {
    '0_0': Disp[0,0,:,:],
    '0_4': Disp[0,4,:,:],
    '4_0': Disp[4,0,:,:],
    '4_4': Disp[4,4,:,:]
}

mask = np.ones((LF_original.shape[0:4])).astype('bool')
LabelSquence, disparity_values, ConMatrixCell, ColorMatrixCell, PxlIndCell = Projection_labels(Segmentation_ref, mask, Disparity_ref, 1, LF_original)

sio.savemat(filename, {'LF_RGB': LF_original, 'Depthset': Disp_original['Disparity'], 'ConMatrixCell':ConMatrixCell,'ColorMatrixCell':ColorMatrixCell, 'DestC':LabelSquence, 'PxlIndCell':PxlIndCell, 'disp_perSR':disparity_values})

