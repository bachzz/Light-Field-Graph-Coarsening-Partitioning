#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 11:32:28 2020

@author: mrizkall
"""

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
# OpenCV
import cv2 as cv2
# PYGSP, Networks
#import networkx as nx
import pygsp as gsp
# For visualization
#import matplotlib as matplot
import matplotlib.pyplot as plt
from matplotlib.pyplot import pause
gsp.plotting.BACKEND = 'matplotlib'
import ray

#%% Function decoder
def decoder_run(filename_coder_output, coeff_reduced_filename):      
    f = sio.loadmat(filename_coder_output)
    # load the decoded values for a specific dataset, dB ...
    f1 = sio.loadmat(coeff_reduced_filename)
    
    # T_coeffkey = list(f.keys())[3]
    Conkey = list(f.keys())[4]
    Cmtxkey = list(f.keys())[5]
    label_mapkey = list(f.keys())[6]
    # Disparitykey = list(f.keys())[7]
    Lum_SRkey = list(f.keys())[11]
    Con_SR = list(f[Conkey])
    Label_map = np.array(f[label_mapkey])
    
    # T_coeff = list(f[T_coeffkey])
    Cmtx = list(f[Cmtxkey])
    Lum_SR = list(f[Lum_SRkey])
    
    # Coeff decoded after de-quantization
    Coeff_reduced_decoded = f1['Coeff_reduced_decode']
    Coeff_reduced_decoded_SR_Q = Coeff_reduced_decoded[0,:]
    
    # Finding idx_reduced
    Idx_reduced = []
    for i in range(len(Cmtx)):
        if np.shape(Cmtx[i][0])[0]!=1:
            Idx_reduced.append(i)
            
    # retrieve Cmtx[reduced], Con_SR[reduced], Lum_SR[reduced]      
    
    Cmtx_reduced = [Cmtx[i][0] for i in Idx_reduced]
    Con_SR_reduced = [Con_SR[0][i] for i in Idx_reduced]
    Lum_SR_reduced = [Lum_SR[0][i] for i in Idx_reduced]
    
    # Start Ray for parallel computing.
    ray.init(num_cpus=12,object_store_memory=20000*1024*1024)
#    ray.init()
    result_ids =[]

    # Start tasks in parallel.
    # retrieve : sse_sr for all Q_idx --> list [sse_sr_q1, sse_sr_q2 ...]
    #    for i_SR in range(2):
    for i_SR in range(len(Con_SR_reduced)):
        Coeff_decoded_SR = [r[0,i_SR] for r in Coeff_reduced_decoded_SR_Q]
        Coeff_decoded_SR_ = [Coeff_decoded_SR[qidx][~np.isnan(Coeff_decoded_SR[qidx])] for qidx in range(len(Coeff_decoded_SR))]
        result_ids.append(SR_decode.remote(Coeff_decoded_SR_, Con_SR_reduced[i_SR], Cmtx_reduced[i_SR], Lum_SR_reduced[i_SR]))
    
    # Wait for the tasks to complete and retrieve the results.
    results = ray.get(result_ids)  # [0, 1, 2, 3]
    
    # shutdown parallel computing
    ray.shutdown()
    

    # R_reconstructed, SR_sse_reduced, SR_mse_reduced
    SR_reconstructed = [r[0] for r in results]
    SR_sse_reduced = [r[1] for r in results]
    SR_mse_reduced = [r[2] for r in results]
    
    # reconstructed LF after lifting
    Rec_reduced_LFlist =[]
    for q_idx in range(len(SR_reconstructed[0])):
        Rec_reduced = np.zeros(Label_map.size)+np.NaN
        for i_SR in range(len(Idx_reduced)):
            Rec_reduced[Label_map.ravel() == Idx_reduced[i_SR]+1] = SR_reconstructed[i_SR][q_idx]
        Rec_reduced_LFlist.append(np.reshape(Rec_reduced, Label_map.shape).astype(np.uint8));
        
 
    return  SR_reconstructed, SR_sse_reduced, SR_mse_reduced, Rec_reduced_LFlist


# %% Inverse transform and lifting to compute the sse for all reduced super-rays
# function ray remote for each super-ray
# for each super-ray 
# return : sse_sr for all Q_idx --> list [sse_sr_q1, sse_sr_q2 ...]

@ray.remote
def SR_decode(T_coeff_decoded, W_SR, Cmtx_SR, Lum_SR):
    # T_coeff_decoded is a list where each object is a vector of decoded transform SR coeffs values for a specific quality
    # W_SR : the original adjacency matrix for the specific super-ray
    # Cmtx_SR : the reduction matrix obtained for the specific super-ray
    # Lum_SR : the original luminance values for the specific super-ray
    # compute W original in the original space
    W = W_SR.astype(np.float64)
    D = ss.diags(np.ravel(W.sum(1)), 0)
    L0 = (D - W).tocsc()
    
    # go down in the reduced space
    L = coarsen_matrix(L0, Cmtx_SR)
    A = L
    A[np.eye(A.shape[0]).astype(np.bool)] = 0
    A = -A
    G = gsp.graphs.Graph(A)
    # compute eigendecomposition of the reduced laplacian
    G.compute_fourier_basis()
    
    # for different Quantization steps 
    # inverse GFT
    x_hat = [G.igft(T_coeff_decoded[qidx].T) for qidx in range(len(T_coeff_decoded))]
    # lift back vectors
    x_tilde = [lift_vector(x_hat[qidx],Cmtx_SR) for qidx in range(len(x_hat))]
    
    
    x_tilde_clipped = [np.clip(x_tilde[qidx], 0, 255).astype(np.uint8) for qidx in range(len(x_tilde))]
    x_original = np.round(Lum_SR).astype(np.float64)
    SR_reconstructed = x_tilde_clipped 
    
    # compute sse and mse and reconstructed list of values in original SR
    
    SR_sse_reduced = [np.sum((x_original[0].T - x_tilde_clipped[qidx][0])*(x_original[0].T - x_tilde_clipped[qidx][0])) for qidx in range(len(x_tilde))]
    SR_mse_reduced = [item/x_original.shape[1] for item in SR_sse_reduced]
    
    return SR_reconstructed, SR_sse_reduced, SR_mse_reduced

#%% function principale
dataset = sys.argv[1]
psnr_min = int(sys.argv[2])
nmax = 5000
cut_algo = 'Ncut'
lambda_ = 1000
method_ = 'total_variation'
enable_reduction = int(sys.argv[3])
enable_Ncut = int(sys.argv[4])

filename = '/temp_dd/igrida-fs1/mrizkall/results_coarsening_segmentation/Results_coarsening_algo_'+cut_algo+'nmax_'+str(nmax)+'psnr_min_'+str(psnr_min)+'_'+str(lambda_)+'_'+method_+'_dataset_'+dataset+'reduction'+str(enable_reduction)+'Ncut'+str(enable_Ncut)+'_coder_output.mat'

#filename = 'Results_coarsening_algo_Ncutnmax_5000psnr_min_35_1000_total_variation_dataset_Friendsreduction1Ncut0_coder_output.mat'
filename_2 = '/temp_dd/igrida-fs1/mrizkall/results_coarsening_segmentation/Coeff_reduced_decoded_'+dataset+'_'+str(psnr_min)+'dB_Reduction.mat'

decoder_output = decoder_run(filename,filename_2)

SR_reconstructed = decoder_output[0]
SR_sse_reduced = decoder_output[1]
SR_mse_reduced = decoder_output[2]
Rec_reduced_LFlist = decoder_output[3]

sio.savemat('/temp_dd/igrida-fs1/mrizkall/results_coarsening_segmentation/decoder_output_'+cut_algo+'nmax_'+str(nmax)+'psnr_min_'+str(psnr_min)+'_'+str(lambda_)+'_'+method_+'_dataset_'+dataset+'reduction'+str(enable_reduction)+'Ncut'+str(enable_Ncut)+'.mat', {'SR_reconstructed': SR_reconstructed,'SR_sse_reduced': SR_sse_reduced,'SR_mse_reduced':SR_mse_reduced,'Rec_reduced_LFlist':Rec_reduced_LFlist})

#i_SR = 80
#Coeff_decoded_SR = [r[0,i_SR] for r in Coeff_reduced_decoded_SR_Q]
#Coeff_decoded_SR_ = [Coeff_decoded_SR[qidx][~np.isnan(Coeff_decoded_SR[qidx])] for qidx in range(len(Coeff_decoded_SR))]
#r = SR_decode(Coeff_decoded_SR_, Con_SR_reduced[i_SR], Cmtx_reduced[i_SR], Lum_SR_reduced[i_SR])
 
