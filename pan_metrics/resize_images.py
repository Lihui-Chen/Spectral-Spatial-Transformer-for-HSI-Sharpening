# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 Vivone Gemine.
All rights reserved. This work should only be used for nonprofit purposes.

@author: Vivone Gemine (website: https://sites.google.com/site/vivonegemine/home )
"""

"""
 Description:
            Resize_images generates the low resolution panchromatic (PAN) and multispectral (MS) images according to Wald's protocol. 

 Interface:
           [I_MS_LR, I_PAN_LR] = resize_images(I_MS,I_PAN,ratio,sensor)
 
 Inputs:
           	I_MS:               MS image upsampled at PAN scale;
            I_PAN:              PAN image;
            ratio:              Scale ratio between MS and PAN. Pre-condition: Integer value;
            sensor:             String for type of sensor (e.g. 'WV2', 'IKONOS').
 
 Outputs:
           I_MS_LR:            Low Resolution MS image;
           I_PAN_LR:           Low Resolution PAN image.
 
 References:
         G. Vivone, M. Dalla Mura, A. Garzelli, and F. Pacifici, "A Benchmarking Protocol for Pansharpening: Dataset, Pre-processing, and Quality Assessment", IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 14, pp. 6102-6118, 2021.
         G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M. O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting Pansharpening With Classical and Emerging Pansharpening Methods", IEEE Geoscience and Remote Sensing Magazine, vol. 9, no. 1, pp. 53 - 81, March 2021.           
"""

import numpy as np
from .imresize import imresize
from .MTF import MTF

def resize_images(I_MS,I_PAN,ratio,sensor):

    I_MS = I_MS.astype('float64')
    I_PAN = I_PAN.astype('float64')
    
    if (sensor == 'QB') or (sensor == 'IKONOS') or (sensor == 'GeoEye1') or (sensor == 'WV2') or (sensor == 'WV3') or (sensor == 'WV4'):
        flag_resize_new = 2
    else:
        flag_resize_new = 1

    if (flag_resize_new == 1):
        """ Bicubic Interpolator MS"""
        I_MS_LP = np.zeros((int(round(I_MS.shape[0]/ratio)),int(round(I_MS.shape[1]/ratio)),I_MS.shape[2]))
        
        for idim in range(I_MS.shape[2]):
            I_MS_LP[:,:,idim] = imresize(I_MS[:,:,idim], 1/ratio)
        
        I_MS_LR = I_MS_LP.astype('float64')
        
        """ Bicubic Interpolator PAN"""
        I_PAN_LR = imresize(I_PAN, 1/ratio)
    
    elif (flag_resize_new == 2):
        I_MS_LP = MTF(I_MS,sensor,ratio)
            
        """ Decimation MS"""
        I_MS_LP_D = I_MS_LP[int(ratio/2):-1:int(ratio),int(ratio/2):-1:int(ratio),:]
        
        I_MS_LR = I_MS_LP_D.astype('float64')
        
        I_PAN_LR = imresize(I_PAN, 1/ratio)
    
    return I_MS_LR, I_PAN_LR