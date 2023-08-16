# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 Vivone Gemine.
All rights reserved. This work should only be used for nonprofit purposes.

@author: Vivone Gemine (website: https://sites.google.com/site/vivonegemine/home )
"""
"""
 Description: 
           Reduced resolution quality indexes. 
 
 Interface:
           [Q2n_index, Q_index, ERGAS_index, SAM_index] = indexes_evaluation(I_F,I_GT,ratio,L,Q_blocks_size,flag_cut_bounds,dim_cut,th_values)

 Inputs:
           I_F:                Fused Image;
           I_GT:               Ground-Truth image;
           ratio:              Scale ratio between MS and PAN. Pre-condition: Integer value;
           L:                  Image radiometric resolution; 
           Q_blocks_size:      Block size of the Q-index locally applied;
           flag_cut_bounds:    Cut the boundaries of the viewed Panchromatic image;
           dim_cut:            Define the dimension of the boundary cut;
           th_values:          Flag. If th_values == 1, apply an hard threshold to the dynamic range.

 Outputs:
           Q2n_index:          Q2n index;
           Q_index:            Q index;
           ERGAS_index:        Erreur Relative Globale Adimensionnelle de Synthèse (ERGAS) index;
           SAM_index:          Spectral Angle Mapper (SAM) index.
 
 References:
         G. Vivone, M. Dalla Mura, A. Garzelli, and F. Pacifici, "A Benchmarking Protocol for Pansharpening: Dataset, Pre-processing, and Quality Assessment", IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 14, pp. 6102-6118, 2021.
         G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M. O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting Pansharpening With Classical and Emerging Pansharpening Methods", IEEE Geoscience and Remote Sensing Magazine, vol. 9, no. 1, pp. 53 - 81, March 2021.
"""
import numpy as np
from .ERGAS import ERGAS 
from .SAM import SAM 
from .Q import Q
from .q2n import q2n

def indexes_evaluation(I_F,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,th_values):
    
    """ cut bounds """
    if (flag_cut_bounds == 1):
        I_GT = I_GT[dim_cut-1:I_GT.shape[0]-dim_cut,dim_cut-1:I_GT.shape[1]-dim_cut,:]
        I_F = I_F[dim_cut-1:I_F.shape[0]-dim_cut,dim_cut-1:I_F.shape[1]-dim_cut,:]
    
    if th_values == 1:
        I_F[I_F > 2**L] = 2**L
        I_F[I_F < 0] = 0
        
    ERGAS_index = ERGAS(I_GT,I_F,ratio)
    SAM_index, SAM_map = SAM(I_GT,I_F)
    
    if np.remainder(Qblocks_size,2) == 0:
        Q_index = Q(I_GT,I_F,Qblocks_size + 1)
    else:
        Q_index = Q(I_GT,I_F,Qblocks_size)
    
    Q2n_index, Q2n_index_map = q2n(I_GT, I_F, Qblocks_size, Qblocks_size)

    return Q2n_index, Q_index, ERGAS_index, SAM_index