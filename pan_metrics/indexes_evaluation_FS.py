# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 Vivone Gemine.
All rights reserved. This work should only be used for nonprofit purposes.

@author: Vivone Gemine (website: https://sites.google.com/site/vivonegemine/home )
"""

"""
 Description: 
           Full resolution quality indexes. 
 
 Interface:
           [HQNR_index, D_lambda, D_S] = indexes_evaluation_FS(I_F,I_MS_LR,I_PAN,L,th_values,I_MS,sensor,ratio,Qblocks_size)

 Inputs:
           I_F:                Fused image;
           I_MS_LR:            Original MS image;
           I_PAN:              Panchromatic image;
           L:                  Image radiometric resolution; 
           th_values:          Flag. If th_values == 1, apply an hard threshold to the dynamic range;
           I_MS:               MS image upsampled to the PAN size;
           sensor:             String for type of sensor (e.g. 'WV2','IKONOS');
           ratio:              Scale ratio between MS and PAN. Pre-condition: Integer value;
           Qblocks_size:       Block size for the Q-index.

 Outputs:
           HQNR_index:         HQNR index;
           D_lambda:           Spectral distortion index;
           D_S:                Spatial distortion index.

 References:
     G. Vivone, M. Dalla Mura, A. Garzelli, and F. Pacifici, "A Benchmarking Protocol for Pansharpening: Dataset, Pre-processing, and Quality Assessment", IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 14, pp. 6102-6118, 2021.
     G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M. O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting Pansharpening With Classical and Emerging Pansharpening Methods", IEEE Geoscience and Remote Sensing Magazine, vol. 9, no. 1, pp. 53 - 81, March 2021.           
"""
from .HQNR import HQNR

def indexes_evaluation_FS(I_F,I_MS_LR,I_PAN,L,th_values,I_MS,sensor,ratio,Qblocks_size):
    
    if th_values == 1:
        I_F[I_F > 2**L] = 2**L
        I_F[I_F < 0] = 0

    HQNR_index, D_lambda, D_S = HQNR(I_F,I_MS_LR,I_MS,I_PAN,Qblocks_size,sensor,ratio)

    return HQNR_index, D_lambda, D_S