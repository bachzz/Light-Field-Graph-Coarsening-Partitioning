#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 10:48:19 2019

@author: mrizkall
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#"""
#Created on Thu Oct 24 14:39:24 2019
#
#@author: mrizkall
#"""

# Import and setup 



#For Segmentation
from sklearn import mixture,cluster
from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
# Graph coarsening
from libraries.coarsening_utils import *
import libraries.graph_utils
import sys
import time
# Numpy, Scipy, Labeling
import math
import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.sparse as ss
import scipy.ndimage.measurements as labeling
# Plot and collections and HDF5
import h5py
import time
# for multiprocessing
#import multiprocessing as mp
#from pygsp import graphs

#from sklearn.neighbors import kneighbors_graph
#from skimage.segmentation import mark_boundaries as showSR
#from skimage import segmentation, color
#from skimage.segmentation import felzenszwalb, slic
# OpenCV
import cv2 as cv2
# PYGSP, Networks
#import networkx as nx
import pygsp as gsp

#pickle
import pickle

# For visualization
#import matplotlib as matplot
import matplotlib.pyplot as plt
from matplotlib.pyplot import pause
gsp.plotting.BACKEND = 'matplotlib'
import ray
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

def find_nearest(self, array, value):
    '''
    Find the nearest element of array to the given value
    '''
    idx = (np.abs(array-value)).argmin()
    return array[idx]


def extents(f):
  delta = f[1] - f[0]
  return [f[0] - delta/2, f[-1] + delta/2]

def ismember(a, b):
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = i
            
    LOCB = [bind.get(itm, np.nan) for itm in a]
    Lia = ~np.isnan(LOCB)  
      
    return Lia,  LOCB  # None can be replaced by any other "not in b" value


#%% Read the graphs of all super-rays
#LF_graphs = sio.loadmat('/home/mrizkall/Desktop/Project_Graph_Code/Datasets_graphs/GBT_LF_Friends_0QP_64views_200numSR_original_PYTHON.mat');
def run_segmentation_algorithm(filename,filename_model,filename_scaler,n_max=15000,PSNR_min=30,cut_algo='Ncut',lambda_=300,method_rate_coef='total_variation',enable_reduction=1,enable_Ncut=1,weight_flag=0,method_coarsening='variation_edges',k=10,psnr_computation='lifting'):
    f = sio.loadmat(filename)
    model_TV = pickle.load(open(filename_model, 'rb'))
    std_scale = pickle.load(open(filename_scaler, 'rb'))
    # List all groups
    print("Keys: %s" % f.keys())
    LF_RGB_key = list(f.keys())[3]
    depth_key = list(f.keys())[4]
    conmatrix_key = list(f.keys())[5]
    colormatrix_key = list(f.keys())[6]
    DestC_key = list(f.keys())[7]
    
    ColorMatrixCell = list(f[colormatrix_key])
    ConMatrixCell = list(f[conmatrix_key])
    flag = 0
    # Get the data and organize it
    Depthset = np.array(f[depth_key])
    Label_map =  np.array(f[DestC_key])
    LF_RGB = np.array(f[LF_RGB_key])    
    if np.max(LF_RGB) <= 1:
        flag = 1
        LF_RGB = np.round(255*LF_RGB).astype(np.uint8)
    # Pour un super-rayon avoir l'adjacency matrix
    
    [M,N,X,Y,C] = LF_RGB.shape
    Luminance_LF  = np.zeros([M,N,X,Y])
    # Find Luminance LF 
    for m in range(M):
        for n in range(N):
            yuv  = cv2.cvtColor(np.squeeze(LF_RGB[m,n,:,:,:]), cv2.COLOR_BGR2YUV)
            Luminance_LF[m,n,:,:]  = np.squeeze(yuv[:,:,0])

#   Start Ray.
    ray.init(num_cpus=12,object_store_memory=35000*1024*1024)

#   Start 4 tasks in parallel.
    result_ids = []
    img_id = ray.put(np.round(np.squeeze(Luminance_LF[0,0,:,:])))
    depth_id = ray.put(np.squeeze(Depthset[0,0,:,:]))
    label_id = ray.put(Label_map)
  
#    img_id = np.squeeze(Luminance_LF[0,0,:,:])
#    depth_id =np.squeeze(Depthset[0,0,:,:])
#    label_id = Label_map

#    for i_SR in range(20):
#        [Seg_SR, Disp_SR, Coarsening_PSNR_1ststage, Coarsening_all_stages_SR, T_Coeff_SR, Con_SR, Cmtx_SR] = SR_process(i_SR, depth_id, img_id, label_id, np.round(255*ColorMatrixCell[0][i_SR]).astype(np.uint8), ConMatrixCell[0][i_SR], n_max, PSNR_min, cut_algo, lambda_, method_rate_coef,enable_reduction,enable_Ncut, weight_flag, method_coarsening, k, psnr_computation)
#        
    
    for i_SR in range(len(ConMatrixCell[0])):
        if flag == 1:
            result_ids.append(SR_process.remote(i_SR, depth_id, img_id, label_id, np.round(255*ColorMatrixCell[0][i_SR]).astype(np.uint8), ConMatrixCell[0][i_SR], model_TV, std_scale, n_max, PSNR_min, cut_algo, lambda_, method_rate_coef,enable_reduction,enable_Ncut, weight_flag, method_coarsening, k, psnr_computation))
        else:
            result_ids.append(SR_process.remote(i_SR, depth_id, img_id, label_id, np.round(ColorMatrixCell[0][i_SR]).astype(np.uint8), ConMatrixCell[0][i_SR], model_TV, std_scale,n_max, PSNR_min, cut_algo, lambda_, method_rate_coef,enable_reduction,enable_Ncut, weight_flag, method_coarsening, k, psnr_computation))
      
    # Wait for the tasks to complete and retrieve the results.
    results = ray.get(result_ids)  # [0, 1, 2, 3]
    ray.shutdown()
    
    
    # Segmentation after RD optimization
    Segmentation_map_new = [r[0] for r in results]
    Segmentation_map_new = [item for sublist in Segmentation_map_new for item in sublist]
    
    # Disparity values
    Disparity_map_new = [r[1] for r in results]
    Disparity_map_new = [item for sublist in Disparity_map_new for item in sublist]
    
    # PSNR Coarsening
    Coarsening_PSNR_1ststage = [r[2] for r in results]
    Coarsening_PSNR_1ststage = [item for sublist in Coarsening_PSNR_1ststage for item in sublist]
    Coarsening_all_stages = [r[3] for r in results]
    Coarsening_all_stages = [item for sublist in Coarsening_all_stages for item in sublist]
    
    # Transform coefficients 
    T_Coeff = [r[4] for r in results]
    T_Coeff = [item for sublist in T_Coeff for item in sublist]
    
    # Connection matrices 
    Con = [r[5] for r in results]
    Con = [item for sublist in Con for item in sublist]
    
    # Connection matrices 
    Cmtx = [r[6] for r in results]
    Cmtx = [item for sublist in Cmtx for item in sublist]

    # original values for PSNR computation
    Lum_SR = [r[7] for r in results]
    Lum_SR = [item for sublist in Lum_SR for item in sublist]
     
    
    
    Seg_labels = np.zeros(Luminance_LF.size)
    for t in range(len(Segmentation_map_new)):
        Seg_labels[Segmentation_map_new[t]] = t+1
    
    Seg_labels = np.reshape(Seg_labels,Luminance_LF.shape);
    Seg_labels.astype('uint16')
  
    return  T_Coeff, Con, Disparity_map_new, Seg_labels, Luminance_LF, Coarsening_PSNR_1ststage, Coarsening_all_stages,Cmtx,Lum_SR



#%% Independent Super-ray processing 
#def SR_process(i_SR,Depth,img,Label_map,Color,data,ir,jc ,n_max=15000,PSNR_min=30,cut_algo = 'Ncut',lambda_=300,method_rate_coef='total_variation',weight_flag=0,method_coarsening='variation_edges',k=10,psnr_computation='lifting'):
@ray.remote
def SR_process(i_SR,Depth,img,Label_map,Color,A,model_TV,std_scale,n_max=15000,PSNR_min=30,cut_algo = 'Ncut',lambda_=300,method_rate_coef='total_variation',enable_reduction=1,enable_Ncut=1,weight_flag=0,method_coarsening='variation_edges',k=10,psnr_computation='lifting'):
    print(i_SR)
    n = Color.shape[1]
    Coarsening_PSNR_1ststage = []
    #A = ss.csr_matrix((np.ones(ir.shape),ir,jc),shape=(n,n))
    ni = A.shape[0]
    Coarsening_all_stages_SR = []
    if ni <= 10:
        mask = Label_map==(i_SR)
        I = np.ravel_multi_index(mask.nonzero(),mask.shape)
        temp_depth = Depth[Label_map[0,0,:,:]==(i_SR)]
        temp_depth = np.median(temp_depth)
        Seg_SR = [I]
        Disp_SR =[temp_depth]
        Con_SR = [A]
    else:
        # Signal 
        x_color = Color
        x_color = np.reshape(x_color,(1,int(x_color.size/3),3))
        yuv = cv2.cvtColor(x_color, cv2.COLOR_BGR2YUV)
        luminance = yuv[0,:,0]
        x = luminance;
        # graph
        if weight_flag == 0:
            G = gsp.graphs.Graph(A)
        else:
            ni = A.shape[0]
            term1 = A.toarray() * np.transpose(np.tile(x, (ni,1)));
            term2 = np.tile(x,(ni,1)) * A.toarray();
            W_sig = ss.csr_matrix(np.multiply(A.toarray(),np.exp(-np.multiply((term1-term2),(term1-term2)))));
            G = gsp.graphs.Graph(W_sig)
            del term1
            del term2
            del W_sig
        
        # Tant que je peux toujours gagner en rate ou faire du coarsening sans penaliser la distortion
        
        if ni > n_max and ni<100000 and enable_reduction == 1:
        # Essayer le coarsening avec r
            r = 1 - get_level_coarsening(ni, n_max) 
            C, Gc, Call, Gall = coarsen(G, K=k, r=r, method= method_coarsening) 
            
            if psnr_computation == 'no_lifting':
                psnr_sr_2 = compute_distortion_coarsening(C,x)
                distortion = psnr_sr_2
            else:
                # coarsen 
                xC = coarsen_vector(x, C)
                # lift 
                x_Lifted = lift_vector(xC, C)
                xLifted = np.round(x_Lifted)
                x_Lifted[xLifted>255] = 255
                x_Lifted[xLifted<0] = 0
                # compute PSNR
                psnr_sr = psnr(x,x_Lifted)
                distortion = psnr_sr
                Coarsening_PSNR_1ststage.append(distortion)
            #print(distortion)    
            if distortion > PSNR_min:
                 mask = Label_map==i_SR
                 I = np.ravel_multi_index(mask.nonzero(),mask.shape)
                 temp_depth = Depth[Label_map[0,0,:,:]==i_SR]
                 temp_depth = np.median(temp_depth)
                 Seg_SR = [I]
                 Disp_SR = [temp_depth]
                 Cmtx_SR = []
                 T_Coeff_SR =[]
                 Lum_SR = []
                 # compute T_Coeff_SR
                 Cmtx_SR.append([C])
                 Con_SR = [G.A]
                 Gc.compute_fourier_basis()
                 T_Coeff_SR.append(Gc.gft(xC))
                 Lum_SR.append(x)
                 
                 
            else:
                mask = Label_map==i_SR
                [T_Coeff_SR, Con_SR, Seg_SR, Disp_SR, ID_hiearchy, delta_Rate_SR, Coarsening_all_stages_SR,Cmtx_SR,Lum_SR] = perform_normalized_cut(img, Label_map, Depth, x, G, model_TV, std_scale, n_max, i_SR, cut_algo, PSNR_min,lambda_,method_rate_coef,enable_reduction,enable_Ncut)
        else:
            mask = Label_map==i_SR
            [T_Coeff_SR, Con_SR, Seg_SR, Disp_SR, ID_hiearchy ,delta_Rate_SR, Coarsening_all_stages_SR,Cmtx_SR,Lum_SR] = perform_normalized_cut(img, Label_map, Depth, x, G, model_TV, std_scale, n_max, i_SR, cut_algo, PSNR_min,lambda_,method_rate_coef,enable_reduction,enable_Ncut)
    
    return (Seg_SR, Disp_SR, Coarsening_PSNR_1ststage, Coarsening_all_stages_SR, T_Coeff_SR, Con_SR, Cmtx_SR,Lum_SR)


#%% Function to compute the rate of the contour
def compute_rate_contour(region_seg_bef, label_SR):  
    Segmentation = region_seg_bef 
    Segmentation_binary = np.array(Segmentation == label_SR ,dtype=np.uint8)
    image_contours, hierarchy = cv2.findContours(Segmentation_binary.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    image_contour = image_contours[0]
    freeman_code = np.zeros(len(image_contour)-1) 
    for i in range(0,len(image_contour)-1):
        delta_x = image_contour[i+1,0,0] - image_contour[i,0,0] 
        delta_y = image_contour[i+1,0,1] - image_contour[i,0,1]
        
        if delta_x == 0 and delta_y == 0:
            pass
        elif delta_x > 0 and delta_y == 0:
            freeman_code[i] = 2
        elif delta_x < 0 and delta_y == 0:
            freeman_code[i] = 6
        elif delta_x == 0 and delta_y > 0:
            freeman_code[i] = 4
        elif delta_x == 0 and delta_y < 0:
            freeman_code[i] = 0
        elif delta_x > 0 and delta_y > 0:
            freeman_code[i] = 3
        elif delta_x > 0 and delta_y < 0:
            freeman_code[i] = 1
        elif delta_x < 0 and delta_y > 0:
            freeman_code[i] = 5
        elif delta_x < 0 and delta_y < 0:
            freeman_code[i] = 7    
    freeman_code_count = np.histogram(freeman_code,bins=[0,1,2,3,4,5,6,7,8])   
    Boundary_size = len(freeman_code)   
    if Boundary_size == 0:
        freeman_code_entropy = np.inf
    else:
        freeman_code_prob = freeman_code_count[0]/len(freeman_code)
        #freeman_code_entropy = -freeman_code.size*np.sum(freeman_code_prob[freeman_code_prob.nonzero()]*(np.log2(freeman_code_prob[freeman_code_prob.nonzero()])))
        freeman_code_entropy = -Boundary_size*np.sum(freeman_code_prob[freeman_code_prob!=0]*(np.log2(freeman_code_prob[freeman_code_prob!=0])))
        # returns the entropy of my edges to code
    return Boundary_size, freeman_code_entropy , image_contours   
       

#%% Function to compute the rate of the boundary
def compute_rate_boundary(region_seg_bef, region_seg_aft1, region_seg_aft2, label_SR):
    B_size_1, Rate_1, image_contour_1 = compute_rate_contour(region_seg_aft1, label_SR[1])
    B_size_2, Rate_2, image_contour_2 = compute_rate_contour(region_seg_aft2, label_SR[2]) 
    B_size, Rate, image_contour = compute_rate_contour(region_seg_bef, label_SR[0])
    
    if B_size_1 == 0 or B_size_2 == 0 : # case where one cluster is made of one pixel
        return 1, np.inf
    else:
        if B_size_1+B_size_2-B_size == 0: # case where the boundary is one pixel wide
            print(B_size_1)
            print(B_size_2)
            print(B_size)
            #print(image_contour_1) 
            #print(image_contour_2)
            #print(image_contour)
            #im = np.zeros(region_seg_aft1.shape, dtype=np.uint8)
            #cv2.drawContours(im,image_contour_1,0,255)
            #cv2.imwrite("/temp_dd/igrida-fs1/mrizkall/c1.png", im)
            #im = np.zeros(region_seg_aft1.shape, dtype=np.uint8)
            #cv2.drawContours(im,image_contour_2,0,255)
            #cv2.imwrite("/temp_dd/igrida-fs1/mrizkall/c2.png", im)
            #im = np.zeros(region_seg_aft1.shape, dtype=np.uint8)
            #cv2.drawContours(im,image_contour,0,255)
            #cv2.imwrite("/temp_dd/igrida-fs1/mrizkall/c0.png", im)
            if (B_size_1>B_size_2*5 and B_size_1>B_size_2) or (B_size_2>B_size_1*5 and B_size_2>B_size_1):
                return 1, np.inf
            else:
                return 1, 0
        else:
            return (B_size_1+B_size_2-B_size)/2, (Rate_1 + Rate_2 - Rate)/(B_size_1+B_size_2-B_size)

#%% Function to compute the distortion with just coarsening
def compute_distortion_coarsening(C, Signal):
    ni = len(Signal)
    D = sp.sparse.diags(np.array(1/np.sum(C,0))[0])    
    Pinv = (C.dot(D)).T
    MSE = 1/ni*(np.power(np.linalg.norm(Signal-Pinv@C.dot(Signal)),2)) 
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(MSE))    


#%% Function to compute the rate for the coefficients in a super-ray
def compute_rate_coefficients(G, Signal, method_rate_coef):
    #ni = len(Signal)
    rate_coef = 0
    if method_rate_coef == 'total_variationDC':
       #compute total variation
       offset = 0.01
       #T = offset*sp.sparse.eye(G.N, format='csc') + G.L
       U0 = np.full((G.N,1),1/np.sqrt(G.N))
       T = offset * U0 @ np.transpose(U0) + G.L
       TV = np.transpose(Signal) @ T @ Signal
       rate_coef = TV
    if method_rate_coef == 'total_variation':
       #compute total variation
       T = G.L
       TV = np.transpose(Signal) @ T @ Signal
       rate_coef = TV
    if method_rate_coef == 'shiftoperator':
       S2 = np.linalg.norm((G.L)@Signal,2)
       S2 = S2*S2
       rate_coef = S2
    if method_rate_coef == 'normalizedDC':
       G.compute_laplacian('normalized')
       offset = 0.01
       U0 = np.diag(np.sqrt(G.d)) @ np.full((G.N,1),1/np.sqrt(G.N))
       T = offset * U0 @ np.transpose(U0) + G.L
       TV = np.transpose(Signal) @ T @ Signal
       rate_coef = TV
    if method_rate_coef == 'normalized':
       G.compute_laplacian('normalized')
       T = G.L
       TV = np.transpose(Signal) @ T @ Signal
       rate_coef = TV

    
    return rate_coef

#%% Function to define the level of coarsening that we should do for a specific superray for complexity issues
def get_level_coarsening(ni, max_n):
    return max_n/ni

            
#%% Function to perform recursive Ncut on a specific super-ray                                             
def perform_normalized_cut(img, Label_map, disparity_first_view, Luminance_SR, G, model_TV, std_scale, bArea, label_SR, cut_algo, PSNR_min,lambda_,method_rate_coef,enable_reduction=1,enable_Ncut=1,k=10,M=9,N=9):
    #disparity_first_view = np.squeeze(disparity_map[0,0,:,:])
    #print(label_SR)
  
    mask = np.zeros(Label_map.shape)
    mask = (Label_map == label_SR)
    list_= label_SR
    
    # Run normalized cut on the pixels with their 4-nn graph inside the views
    [T_Coeff_SR, Con_SR, Seg, Disp, ID, Rate, Coarsening_all_stages,Cmtx,Lum_SR] = Ncut_Partition(img, mask, disparity_first_view, G, Luminance_SR, model_TV, std_scale, bArea, list_, M, N,cut_algo, PSNR_min,lambda_,method_rate_coef,enable_reduction,enable_Ncut)
    
    # create disparity map new and segmentation map new out of Seg and Disp
    return T_Coeff_SR, Con_SR, Seg, Disp, ID, Rate, Coarsening_all_stages,Cmtx,Lum_SR

#%% Function to perform recursive Ncut on a specific super-ray              
def Ncut_Partition(img, mask, disp, G, Luminance_SR, model_TV, std_scale, bArea, ID, M, N, cut_algo, PSNR_min,lambda_,method_rate_coef,enable_reduction=1,enable_Ncut=1, k=10,method_coarsening='variation_edges',psnr_computation='lifting'):
    # SR : Luminance image, mask to know where is the super-ray 
    # NcutPartition - Partitioning
    
    #% Try coarsening if bArea<ni<120000 if its okay, then return directly the indices, the disparity and 
    ni = np.sum(mask)   
    #print(ni)
    if enable_reduction == 1:
        if ni > bArea and ni<100000:
            # Essayer le coarsening avec r
            x = Luminance_SR
            r = 1 - get_level_coarsening(ni, bArea) 
            C, Gc, Call, Gall = coarsen(G, K=k, r=r, method=method_coarsening) 
            
            if psnr_computation == 'no_lifting':
                psnr_sr_2 = compute_distortion_coarsening(C,x)
                distortion = psnr_sr_2
            else:
                # coarsen 
                xC = coarsen_vector(x, C)
                # lift 
                x_Lifted = lift_vector(xC, C)
                xLifted = np.round(x_Lifted)
                x_Lifted[xLifted>255] = 255
                x_Lifted[xLifted<0] = 0
                # compute PSNR
                psnr_sr = psnr(x,x_Lifted)
                distortion = psnr_sr
                
            # Check distortion 
           # print(distortion)
            if distortion > PSNR_min:
                Coarsening_all_stages = []
                T_Coeff_SR = []
                Coarsening_all_stages.extend([distortion])
                I = np.ravel_multi_index(mask.nonzero(),mask.shape)
                temp_depth = disp*np.squeeze(mask[0,0,:,:])
                temp_depth = np.median(temp_depth)   
                Seg = []
                Lum_SR = []
                Cmtx = []
                
                delta_rate = [] 
                disp_out = []
                Seg.append(I)
                disp_out.extend([temp_depth])
                delta_rate.append(0)
                Cmtx.append([C])
                Con_SR = [G.W]
                Gc.compute_fourier_basis()
                T_Coeff_SR.append(Gc.gft(xC))
                Lum_SR.append(x)
                
                return T_Coeff_SR, Con_SR, Seg, disp_out, ID, delta_rate, Coarsening_all_stages, Cmtx, Lum_SR
        
    if (np.sum(np.squeeze(mask[0,0,:,:])) < 10):    
        Seg = []
        delta_rate = []
        disp_out = []
        Coarsening_all_stages = []
        T_Coeff_SR =[]    
        Cmtx=[]
        Lum_SR = []
        S = Luminance_SR
        mask_copy = np.copy(mask)
        I = mask_copy.flatten().nonzero()
        Seg.append(I[0])
        temp_depth = disp*np.squeeze(mask[0,0,:,:])
        temp_depth = np.median(temp_depth)   
        disp_out.extend([temp_depth])
        delta_rate.append([np.nan])
        Coarsening_all_stages.extend([np.nan])
        G.compute_fourier_basis()
        S_T = G.gft(S)
        Cmtx.append([np.nan])
        T_Coeff_SR.append(S_T)
        Con_SR =[G.W]
        Lum_SR.append(S)
        
        # ajouter sse_reduced
        return T_Coeff_SR, Con_SR, Seg, disp_out, ID, delta_rate, Coarsening_all_stages, Cmtx, Lum_SR
    else:
        # Ncut paritioning
        mask_copy = np.copy(mask)
        I = mask_copy.flatten().nonzero()
        label_im_or = np.full([mask.shape[2], mask.shape[3]], -1.)
        label_im_or[np.squeeze(mask[0,0,:,:])] = 0
        
        if (cut_algo =='Ncut'):
            graph = image.img_to_graph(img, mask=np.squeeze(mask[0,0,:,:]))
            graph2 = image.img_to_graph(disp, mask=np.squeeze(mask[0,0,:,:]))
            # Take an exponential decreasing function of the gradient
            graph.data = np.exp(-graph.data/100)
            graph2.data = np.exp(-graph2.data/graph2.data.std())
            # Force the solver to be arpack, since amg is numerically
            # unstable on this example , number of clusters can be changed
            graph.data = (graph.data)*(graph2.data)
            graph = graph + 0.1*ss.eye(graph.shape[0])
            labels = spectral_clustering(graph, n_clusters=2, eigen_solver='arpack')
            label_im = np.full([mask.shape[2],mask.shape[3]], -1.)
            label_im[np.squeeze(mask[0,0,:,:])] = labels.astype('uint32')   
            
        if (cut_algo =='kmeans'):
            label_im = np.full([mask.shape[2],mask.shape[3]], -1.)
            mask_vue = np.squeeze(mask[0,0,:,:])
            #nvue = np.sum(mask_vue)
            boundingbox = bbox(mask_vue)
            maskvue_cropped = mask_vue[boundingbox[0]:boundingbox[1]+1][:,boundingbox[2]:boundingbox[3]+1]        
            image_cropped = img[boundingbox[0]:boundingbox[1]+1][:,boundingbox[2]:boundingbox[3]+1]
            labels = slic(image_cropped, n_segments=2)
            #g = graph.rag_mean_color(image_cropped, labels)
            #labels2 = graph.cut_normalized(labels, g)
            #labels2 = labels2.astype('uint32')
            labels[maskvue_cropped==0] = -1.
            #label_im[boundingbox[0]:boundingbox[1]+1][:,boundingbox[2]:boundingbox[3]+1] = labels2
            label_im[boundingbox[0]:boundingbox[1]+1][:,boundingbox[2]:boundingbox[3]+1] = labels.astype('uint32') 
            plt.imshow(label_im)
            plt.show()
            pause(1)
          
        if (cut_algo =='gmm'):
            X = img[mask[0,0,:,:].nonzero()]
            X = X.reshape(-1,1)
            gmm = mixture.GaussianMixture(n_components=2, covariance_type='full')
            gmm.fit(X)
            clusters = gmm.predict(X)
            label_im = np.full([mask.shape[2],mask.shape[3]], -1.)
            label_im[np.squeeze(mask[0,0,:,:])] = clusters.astype('uint32')   
            
        if (cut_algo =='agglomerative'):
            xy = mask[0,0,:,:].nonzero()
            x = xy[0]
            y = xy[1]
            x = x.reshape(-1,1)
            y = y.reshape(-1,1)
            xy = np.concatenate((x,y),axis=1)
            graph2 = image.grid_to_graph(mask.shape[2],mask.shape[3],mask = mask[0,0,:,:])
            # Take an exponential decreasing function of the gradient
            #graph2.data = np.exp(-graph2.data/10)
            X = img[mask[0,0,:,:].nonzero()]
            X = X.reshape(-1,1)
            AGLO = cluster.AgglomerativeClustering(n_clusters = 2, connectivity=graph2.tocsr())
            AGLO.fit(X)
            clusters = AGLO.labels_
            label_im = np.full([mask.shape[2],mask.shape[3]], -1.)
            label_im[np.squeeze(mask[0,0,:,:])] = clusters.astype('uint32')
    
        if (cut_algo == 'felzenszwalb'):
             label_im = np.full([mask.shape[2],mask.shape[3]], -1.)
             mask_vue = np.squeeze(mask[0,0,:,:])
             boundingbox = bbox(mask_vue)
             maskvue_cropped = mask_vue[boundingbox[0]:boundingbox[1]][:,boundingbox[2]:boundingbox[3]]
             maskvue_cropped = maskvue_cropped.flatten()         
             labels = felzenszwalb(img[boundingbox[0]:boundingbox[1]][:,boundingbox[2]:boundingbox[3]], scale=100, sigma=0.5, min_size=100)
             labels = labels.astype('uint32')
             label_im[maskvue_cropped.nonzero()] = labels[maskvue_cropped.nonzero()]
             plt.imshow(label_im)
             plt.show()
             pause(1)
             
        # Compute delta_R
        label_SR = [0,0,1]
        # Rate boundary 
        
        Boundary_size, Boundary_rate  = compute_rate_boundary(label_im_or, label_im, label_im, label_SR)
        Boundary_rate = Boundary_rate/2
        # Project subsequent super-rays to be able to compute total variation after projection on two super-rays
        
        [LabelSquence, disparity_values, ConMatrixCell, PxlIndCell] = Projection_labels(label_im, mask, disp)
        
        #masks to enter recursivity
        I_1 = (LabelSquence[mask] == 0).nonzero()
        I_2 = (LabelSquence[mask] == 1).nonzero()
        
        mask1 = np.zeros(mask.shape, dtype=bool)
        mask2 = np.zeros(mask.shape, dtype=bool)
        
        mask1 = (LabelSquence==0)*mask
        mask2 = (LabelSquence==1)*mask
        

        S = Luminance_SR
        A1 = ConMatrixCell[0]
        G1 = gsp.graphs.Graph(A1)
        G1.is_connected()
        A2 = ConMatrixCell[1]
        G2 = gsp.graphs.Graph(A2)
        G2.is_connected()


        # create Gspatial1 Gspatial2 Gangular1 Gangular2
        #Gspatial1 = gsp.graphs.Graph(ConMatrixCellSpatial[0])
        #Gspatial2 = gsp.graphs.Graph(ConMatrixCellSpatial[1])
        #Gangular1 = gsp.graphs.Graph(ConMatrixCellAngular[0])
        #Gangular2 = gsp.graphs.Graph(ConMatrixCellAngular[1])


        # return all those variables
        S1 = Luminance_SR[I_1[0]]
        S2 = Luminance_SR[I_2[0]]
        # Rate of coefficients before and after partitioning

        A12 = G.W[I_1[0],:]
        A12 = A12[:,I_2[0]]
        #nbr_edges_L12 = np.sum(A12)

        #TV_A = compute_rate_coefficients(G1, S1, method_rate_coef)
        #TV_B = compute_rate_coefficients(G2, S2, method_rate_coef)
        #NeA = np.sum(G.W)

        TV_coeff = compute_rate_coefficients(G, S, method_rate_coef)

        # Compute tv angular before and after
        #TV_A_s = compute_rate_coefficients(Gspatial1, S1, method_rate_coef)
        #TV_B_s = compute_rate_coefficients(Gspatial2, S2, method_rate_coef)
        #TV_A_a = compute_rate_coefficients(Gangular1, S1, method_rate_coef)
        #TV_B_a = compute_rate_coefficients(Gangular2, S2, method_rate_coef)
        #TV_s = compute_rate_coefficients(Gspatial, S, method_rate_coef)
        #TV_a = compute_rate_coefficients(Gangular, S, method_rate_coef)


        #  Test with just smaller weights and adding an extra on the DC
        A12 = (G.W).copy()
        A12[np.ix_(I_1[0],I_1[0])] = 0
        A12[np.ix_(I_2[0],I_2[0])] = 0

        Acut = 0.001*A12[np.ix_(np.concatenate((I_1[0],I_2[0])),np.concatenate((I_1[0],I_2[0])))] + ss.block_diag([A1, A2])
        Gcut = gsp.graphs.Graph(Acut)

        TV2_coeff =  compute_rate_coefficients(Gcut, np.concatenate((S1,S2)), method_rate_coef)

        TV_coeff_peredge = TV_coeff/np.sum(G.W)
        #NeB = np.sum(Acut)
        TV2_coeff_peredge = TV2_coeff/np.sum(Acut)
        R_B = Boundary_rate
        #
        if enable_Ncut == 1:
            #delta_R_coeff = compute_rate_coefficients(L, S)/np.sum(S*S)  - compute_rate_coefficients(L1, S1)/np.sum(S1*S1)  - compute_rate_coefficients(L2, S2)/np.sum(S2*S2)
            #delta_R_coeff = compute_rate_coefficients(G, S, method_rate_coef)  - compute_rate_coefficients(G1, S1, method_rate_coef)  - compute_rate_coefficients(G2, S2, method_rate_coef)
            #Total Rate Gain
            #delta_R =  delta_R_coeff/nbr_edges_L12 - lambda_*Boundary_rate
            if np.isinf(R_B) or np.isnan(R_B):
                cut_decision = False
            else:
                q = np.round(255*np.sqrt(12)/(np.sqrt(10**(PSNR_min/10))))
                X_test = np.array([[q,TV_coeff_peredge,TV2_coeff_peredge,R_B]])
                X_test_std = std_scale.transform(X_test)
                cut_decision = np.logical_not(model_TV.predict(X_test_std))
        else:
            cut_decision = False
        #

        # Step 4. Decide if the current partition should be divided
        # to compute the rate of the coefficients we need the segmentation new and the luminance new
        # if we are not gaining in rate and ni is less than bArea, stop recursion.
        #print(cut_decision)
            
        # If no rate gain, then do not partition
        if  (cut_decision == False) and (ni < bArea) : 
            Seg = []
            cut_decision = []
            disp_out = []
            Coarsening_all_stages = []
            T_Coeff_SR = []
            Cmtx = []
            Lum_SR = []
            Seg.append(I[0])
            temp_depth = disp*np.squeeze(mask[0,0,:,:])
            temp_depth = np.median(temp_depth)   
            disp_out.extend([temp_depth])
            cut_decision.append(cut_decision)
            Coarsening_all_stages.extend([np.nan])
            G.compute_fourier_basis()
            S_T = G.gft(S)
            Cmtx.append([np.nan])
            T_Coeff_SR.append(S_T)
            Lum_SR.append(S)
            Con_SR =[G.W]
            # ajouter sse_reduced
            return T_Coeff_SR, Con_SR, Seg, disp_out, ID, cut_decision, Coarsening_all_stages, Cmtx, Lum_SR
        
        # if a partition is too small, stop partitioning  
        # Otherwise necessarily try to segment the subsequent superray
    
        # ajouter sse_reduced
        # Segment A
        [T_Coeff_SRA, ConA, SegA, dispA, IdA, cut_decisionA, Coarsening_all_stagesA, CmtxA, Luminance_SRA] = Ncut_Partition(img, mask1, disp, G1, S1, model_TV, std_scale, bArea, [ID,'A'],M, N,cut_algo,PSNR_min,lambda_,method_rate_coef,enable_reduction,enable_Ncut);
        # Segment B
        [T_Coeff_SRB, ConB, SegB, dispB, IdB, cut_decisionB, Coarsening_all_stagesB, CmtxB, Luminance_SRB] = Ncut_Partition(img, mask2, disp, G2, S2, model_TV, std_scale, bArea, [ID,'B'],M, N,cut_algo,PSNR_min,lambda_,method_rate_coef,enable_reduction,enable_Ncut);
        
        # concatenate lists
        SegA.extend(SegB)
        T_Coeff_SRA.extend(T_Coeff_SRB)
        Luminance_SRA.extend(Luminance_SRB)
        dispA.extend(dispB)
        IdA.extend(IdB)
        cut_decisionA.extend(cut_decisionB)
        Coarsening_all_stagesA.extend(Coarsening_all_stagesB)
        CmtxA.extend(CmtxB)
        ConA.extend(ConB)
        return T_Coeff_SRA, ConA, SegA, dispA, IdA, cut_decisionA, Coarsening_all_stagesA, CmtxA, Luminance_SRA

#%% Function to project labels after performing Normalized cut and graph construction v2 
def Projection_labels(Segmentation_before, mask, disp, compute_color = 0, LF_RGB = [np.nan]):
    # segmentation before 4 D
    # mask 4D 
    # disp 2D
    # Segmentation projection from top-left corner view to the whole LF
    Mref = 0
    Nref = 0
    Segmentation_first_view = Segmentation_before
    [M,N,X,Y] = mask.shape
    #disparity_first_view = disp
    LabelList = np.unique(Segmentation_first_view[np.squeeze(mask[0,0,:,:])])
    
    # First view
    DepthSquence = np.zeros(mask.shape) + np.inf
    LabelSquence = np.zeros(mask.shape) + np.nan
    LabelSquence[Mref, Nref, :, :] = Segmentation_first_view
    DepthSquence[Mref, Nref, :, :] = disp.astype('float')
      
    # reference depth
    #Depth_ref = disparity_first_view
    disp_copy = disp.copy()
    disparity_values = []
    # calcul du depth reference pr les super-rayons dans la premiere vue
    for LabelInd in LabelList:
        temp_depth = disp_copy[Segmentation_first_view==LabelInd];
        temp_depth = np.median(temp_depth)
        disp_copy[Segmentation_first_view==LabelInd] = temp_depth;
        disparity_values.append(temp_depth)
    
    DepthSquence[0,0,:,:] = disp_copy
    
    for m in range(M):
        #print('View m = %d\n', m)
        # pour la premiere ligne
        if (m == 0):
            disp_temp = np.squeeze(DepthSquence[0,0,:,:])
            for n in range(1,N): #toutes les colonnes
                # legal indices in view m,n
                mask_viewmn = np.squeeze(mask[m,n,:,:])
                legal_ind = np.nonzero(mask_viewmn == True)
                legal_ind = np.ravel_multi_index([legal_ind[0], legal_ind[1]], (X,Y))
                #print('View n = %d\n', n)
                for LabelInd in LabelList:
                    # temp depth of the superrays
                    temp_depth = disp_temp[Segmentation_first_view==LabelInd]
                    [rx, ry] = (Segmentation_first_view==LabelInd).nonzero()
                    temp_depth = np.median(temp_depth)
                    #if (np.isnan(temp_depth) or np.isinf(temp_depth)):
                    #    print("infinite value encountered in 1st column")
                    # projection from first view to the horizontal neighbors next to it
                    temp_LabelSquence = np.squeeze(LabelSquence[m,n,:,:]);
                    temp_DepthSquence = np.squeeze(DepthSquence[m,n,:,:]);
                    # reference
                    Mref = 0
                    Nref = 0
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
            
                #print('Horizontal fill\n')
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
                    
        else:
            #print('View m = %d\n',m)
            for n in range(N):
                mask_viewmn = np.squeeze(mask[m,n,:,:])
                #print('View n = %d\n',n)
                # first view : project from view above
                if n == 0:
                    disp_temp = np.squeeze(DepthSquence[0,0,:,:])
                    # project
                    for LabelInd in LabelList:
                        temp_depth = disp_temp[np.squeeze(LabelSquence[0,0,:,:])==LabelInd]
                        [rx, ry] = (np.squeeze(LabelSquence[0,0,:,:])==LabelInd).nonzero()
                        temp_depth = np.median(temp_depth)
                       # if (np.isnan(temp_depth) or np.isinf(temp_depth)):
                           # print("infinite value encountered in vertical projection")
                        #projection on the neighboring views in the horizontal axis
                        temp_LabelSquence = np.squeeze(LabelSquence[m,n,:,:]);
                        temp_DepthSquence = np.squeeze(DepthSquence[m,n,:,:]);
                        
                        Mref = 0
                        Nref = 0
                        Mtar = m
                        Ntar = 0
    
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
                   # print('Vertical fill\n')
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

                    DepthSquence[m][n][mask_viewmn] = Cru_Depth[mask_viewmn]
                    LabelSquence[m][n][mask_viewmn] = Cru_Label[mask_viewmn]
                    
                else:
                    disp_temp = np.squeeze(DepthSquence[m, 0, :, :])
                    for LabelInd in LabelList:
                        temp_depth = disp_temp[np.squeeze(LabelSquence[m, 0, :, :]) == LabelInd]
                        [rx, ry] =(np.squeeze(LabelSquence[m, 0, :, :])==LabelInd).nonzero()
                        temp_depth = np.median(temp_depth)
                       # if (np.isnan(temp_depth) or np.isinf(temp_depth)):
                          #  print("infinite value encountered in other columns")
                        # projection on the neighboring views in the horizontal axis
                        temp_LabelSquence = np.squeeze(LabelSquence[m,n,:,:]);
                        temp_DepthSquence = np.squeeze(DepthSquence[m,n,:,:]);
                        # reference
                        Mref = m
                        Nref = 0
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
                        #illegal_ind_mask2 = occlusion_depth>temp_depth
                        nx = nx[~illegal_ind_mask2]
                        ny = ny[~illegal_ind_mask2]
                        temp_LabelSquence[nx, ny] = LabelInd
                        temp_DepthSquence[nx, ny] = temp_depth
                            
                        LabelSquence[m,n,:,:] = temp_LabelSquence
                        DepthSquence[m,n,:,:] = temp_DepthSquence
                        
                    Cru_Label = temp_LabelSquence
                    Cru_Depth = temp_DepthSquence
                   # print('Horizontal fill\n')
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
        
    for label in LabelList:
        #print(['label=%d out of %d\n', label, np.max(LabelList)])
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
            #print(len(x_sp))
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
               # print(len(idx_keep))
                Adjacency_sp = Adjacency_sp[idx_keep,:]
                Adjacency_sp = Adjacency_sp[:,idx_keep]
                # block matrix
                Block_matrix_sp.append(Adjacency_sp);
               # print(Adjacency_sp.shape[0])
            else:
                continue
                
        #%%        
        ##############################33
        Adjacency_spatial = ss.block_diag(Block_matrix_sp); 
        elapsed = time.time() - t
        #print(['Time taken for blkdiag is ', elapsed]);
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

        #elapsed = time.time() - t
        
        Adjacency_ang = ss.coo_matrix((np.ones(len(ConnectionInd3)), (ConnectionInd3, ConnectionInd4)), shape=np.shape(Adjacency_spatial));
        #Adjacency_ang = np.multiply(Adjacency_ang>0, 1)
        #print(['Time taken for angular connections is ' ,elapsed]);
        
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



#%% Function to project labels after performing Normalized cut and graph construction v2  - with spatial and angular adjacencies
def Projection_labels_v2(Segmentation_before, mask, disp, compute_color = 0, LF_RGB = [np.nan]):
    # segmentation before 4 D
    # mask 4D 
    # disp 2D
    # Segmentation projection from top-left corner view to the whole LF
    Mref = 0
    Nref = 0
    Segmentation_first_view = Segmentation_before
    [M,N,X,Y] = mask.shape
    #disparity_first_view = disp
    LabelList = np.unique(Segmentation_first_view[np.squeeze(mask[0,0,:,:])])
    
    # First view
    DepthSquence = np.zeros(mask.shape) + np.inf
    LabelSquence = np.zeros(mask.shape) + np.nan
    LabelSquence[Mref, Nref, :, :] = Segmentation_first_view
    DepthSquence[Mref, Nref, :, :] = disp.astype('float')
      
    # reference depth
    #Depth_ref = disparity_first_view
    disp_copy = disp.copy()
    disparity_values = []
    # calcul du depth reference pr les super-rayons dans la premiere vue
    for LabelInd in LabelList:
        temp_depth = disp_copy[Segmentation_first_view==LabelInd];
        temp_depth = np.median(temp_depth)
        disp_copy[Segmentation_first_view==LabelInd] = temp_depth;
        disparity_values.append(temp_depth)
    
    DepthSquence[0,0,:,:] = disp_copy
    
    for m in range(M):
        #print('View m = %d\n', m)
        # pour la premiere ligne
        if (m == 0):
            disp_temp = np.squeeze(DepthSquence[0,0,:,:])
            for n in range(1,N): #toutes les colonnes
                # legal indices in view m,n
                mask_viewmn = np.squeeze(mask[m,n,:,:])
                legal_ind = np.nonzero(mask_viewmn == True)
                legal_ind = np.ravel_multi_index([legal_ind[0], legal_ind[1]], (X,Y))
                #print('View n = %d\n', n)
                for LabelInd in LabelList:
                    # temp depth of the superrays
                    temp_depth = disp_temp[Segmentation_first_view==LabelInd]
                    [rx, ry] = (Segmentation_first_view==LabelInd).nonzero()
                    temp_depth = np.median(temp_depth)
                    #if (np.isnan(temp_depth) or np.isinf(temp_depth)):
                     #   print("infinite value encountered in 1st column")
                    # projection from first view to the horizontal neighbors next to it
                    temp_LabelSquence = np.squeeze(LabelSquence[m,n,:,:]);
                    temp_DepthSquence = np.squeeze(DepthSquence[m,n,:,:]);
                    # reference
                    Mref = 0
                    Nref = 0
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
            
                #print('Horizontal fill\n')
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
                    
        else:
            #print('View m = %d\n',m)
            for n in range(N):
                mask_viewmn = np.squeeze(mask[m,n,:,:])
                #print('View n = %d\n',n)
                # first view : project from view above
                if n == 0:
                    disp_temp = np.squeeze(DepthSquence[0,0,:,:])
                    # project
                    for LabelInd in LabelList:
                        temp_depth = disp_temp[np.squeeze(LabelSquence[0,0,:,:])==LabelInd]
                        [rx, ry] = (np.squeeze(LabelSquence[0,0,:,:])==LabelInd).nonzero()
                        temp_depth = np.median(temp_depth)
                        #if (np.isnan(temp_depth) or np.isinf(temp_depth)):
                         #   print("infinite value encountered in vertical projection")
                        #projection on the neighboring views in the horizontal axis
                        temp_LabelSquence = np.squeeze(LabelSquence[m,n,:,:]);
                        temp_DepthSquence = np.squeeze(DepthSquence[m,n,:,:]);
                        
                        Mref = 0
                        Nref = 0
                        Mtar = m
                        Ntar = 0
    
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
                    #print('Vertical fill\n')
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

                    DepthSquence[m][n][mask_viewmn] = Cru_Depth[mask_viewmn]
                    LabelSquence[m][n][mask_viewmn] = Cru_Label[mask_viewmn]
                    
                else:
                    disp_temp = np.squeeze(DepthSquence[m, 0, :, :])
                    for LabelInd in LabelList:
                        temp_depth = disp_temp[np.squeeze(LabelSquence[m, 0, :, :]) == LabelInd]
                        [rx, ry] =(np.squeeze(LabelSquence[m, 0, :, :])==LabelInd).nonzero()
                        temp_depth = np.median(temp_depth)
                        #if (np.isnan(temp_depth) or np.isinf(temp_depth)):
                         #   print("infinite value encountered in other columns")
                        # projection on the neighboring views in the horizontal axis
                        temp_LabelSquence = np.squeeze(LabelSquence[m,n,:,:]);
                        temp_DepthSquence = np.squeeze(DepthSquence[m,n,:,:]);
                        # reference
                        Mref = m
                        Nref = 0
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
                        #illegal_ind_mask2 = occlusion_depth>temp_depth
                        nx = nx[~illegal_ind_mask2]
                        ny = ny[~illegal_ind_mask2]
                        temp_LabelSquence[nx, ny] = LabelInd
                        temp_DepthSquence[nx, ny] = temp_depth
                            
                        LabelSquence[m,n,:,:] = temp_LabelSquence
                        DepthSquence[m,n,:,:] = temp_DepthSquence
                        
                    Cru_Label = temp_LabelSquence
                    Cru_Depth = temp_DepthSquence
                    #print('Horizontal fill\n')
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
    ConMatrixCellSpatial = []
    ConMatrixCellAngular = []
    PxlIndCell = []
    
    if compute_color == 1:
        ColorMatrixCell = []
        
    for label in LabelList:
        #print(['label=%d out of %d\n', label, np.max(LabelList)])
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
               #print(len(idx_keep))
                Adjacency_sp = Adjacency_sp[idx_keep,:]
                Adjacency_sp = Adjacency_sp[:,idx_keep]
                # block matrix
                Block_matrix_sp.append(Adjacency_sp);
                #print(Adjacency_sp.shape[0])
            else:
                continue
                
        #%%        
        ##############################33
        Adjacency_spatial = ss.block_diag(Block_matrix_sp); 
        elapsed = time.time() - t
        #print(['Time taken for blkdiag is ', elapsed]);
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
        #print(['Time taken for angular connections is ' ,elapsed]);
        
        ConMatrix = Adjacency_spatial+Adjacency_ang
        Adjacency_spatial = np.multiply(Adjacency_spatial>0, 1)
        Adjacency_ang = np.multiply(Adjacency_ang>0, 1)
        ConMatrix = np.multiply(ConMatrix>0, 1)
        
        
        
        if compute_color == 1:
            ColorMatrixCell.append(ColorMatrix)
        
        ConMatrixCell.append(ConMatrix)    
        ConMatrixCellSpatial.append(Adjacency_spatial) 
        ConMatrixCellAngular.append(Adjacency_ang) 
        PxlIndCell.append(GIndex)
    if compute_color == 0:
        return LabelSquence, disparity_values, ConMatrixCell,ConMatrixCellSpatial,ConMatrixCellAngular,PxlIndCell
    else:
        return LabelSquence, disparity_values, ConMatrixCell,ConMatrixCellSpatial,ConMatrixCellAngular,ColorMatrixCell, PxlIndCell




#%% MAIN fonction
dataset = sys.argv[1]
nmax = int(sys.argv[2])
psnr_min = int(sys.argv[3])
cut_algo = sys.argv[4]
lambda_ = int(sys.argv[5])
method_ = sys.argv[6]
enable_reduction = int(sys.argv[7])
enable_Ncut = int(sys.argv[8])

#dataset = 'FountainVincent'
#cut_algo = 'Ncut'
#nmax = 15000
#psnr_min = 25
#lambda_ = 1000
#method_ = 'total_variation'
#enable_reduction = 1
#enable_Ncut = 0
print(dataset)
print(nmax)
print(psnr_min)
print(cut_algo)
print(lambda_)
print(method_)
print(enable_reduction)
print(enable_Ncut)

# A revoir les noms des fichiers suivant les graphes construits, et les paths du modelSVM.
filename = '/temp_dd/igrida-fs1/mrizkall/Dataset_graphs/GBT_LF_'+dataset+'_0QP_64views_500numSR_original.mat'
#filename = '/home/mrizkall/Desktop/Project_Graph_Code/Datasets_graphs/GBT_LF_'+dataset+'_0QP_64views_500numSR_original.mat'
filename_model = '/temp_dd/igrida-fs1/mrizkall/Dataset_graphs/Model/model_SVM.sav'
filename_scaler = '/temp_dd/igrida-fs1/mrizkall/Dataset_graphs/Model/scaler.sav'
[T_Coeff, Con, Disparity_map_new, Seg_labels, Luminance_LF, Coarsening_PSNR_1ststage, Coarsening_all_stages, Cmtx, Lum_SR] = run_segmentation_algorithm(filename,filename_model,filename_scaler,nmax,psnr_min,cut_algo,lambda_,method_,enable_reduction,enable_Ncut)

#sio.savemat('/home/mrizkall/Desktop/Project_Graph_Code/Coarsening/graph-coarsening-master/Code_coarseningbasedlightfieldcompression/Results_coarsening_algo_'+cut_algo+'nmax_'+str(nmax)+'psnr_min_'+str(psnr_min)+'dataset_'+dataset+'_coder_output.mat', {'T_Coeff': T_Coeff,'Con': Con,'Cmtx':Cmtx,'label_map':Seg_labels,'Disparity_values_new':Disparity_map_new, 'Reduction_PSNR_1':Coarsening_PSNR_1ststage,'Reduction_PSNR_all':Coarsening_all_stages})

sio.savemat('/temp_dd/igrida-fs1/mrizkall/results_coarsening_segmentation/Results_coarsening_algo_'+cut_algo+'nmax_'+str(nmax)+'psnr_min_'+str(psnr_min)+'_'+str(lambda_)+'_'+method_+'_dataset_'+dataset+'reduction'+str(enable_reduction)+'Ncut'+str(enable_Ncut)+'_coder_output.mat', {'T_Coeff': T_Coeff,'Con': Con,'Cmtx':Cmtx,'label_map':Seg_labels,'Disparity_values_new':Disparity_map_new, 'Reduction_PSNR_1':Coarsening_PSNR_1ststage,'Reduction_PSNR_all':Coarsening_all_stages,'Luminance_LF':Luminance_LF,'Lum_SR':Lum_SR})
