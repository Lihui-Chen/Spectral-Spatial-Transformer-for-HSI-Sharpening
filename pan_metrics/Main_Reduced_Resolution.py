# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 Vivone Gemine.
All rights reserved. This work should only be used for nonprofit purposes.

@author: Vivone Gemine (website: https://sites.google.com/site/vivonegemine/home )
"""

import numpy as np
import scipy.io
from .imresize import imresize
from .interp23 import interp23
from .resize_images import resize_images


"""
 Description:
           Simulation of datasets at reduced resolution (reduced resolution assessment)
 
 References:
         G. Vivone, M. Dalla Mura, A. Garzelli, and F. Pacifici, "A Benchmarking Protocol for Pansharpening: Dataset, Pre-processing, and Quality Assessment", IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 14, pp. 6102-6118, 2021.
         G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M. O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting Pansharpening With Classical and Emerging Pansharpening Methods", IEEE Geoscience and Remote Sensing Magazine, vol. 9, no. 1, pp. 53 - 81, March 2021.     
"""

"""
Analyzed image choice
"""
# sensor = 'WV3'
# im_tag = 'NY1'

# sensor = 'GeoEye1'
# im_tag = 'GeoEye1_May'

sensor = 'IKONOS'
im_tag = 'Toulouse'

""" Quality Index Blocks """
Qblocks_size = 32

""" Interpolator """
bicubic = 0

""" Cut Final Image"""
flag_cut_bounds = 1

dim_cut = 21

""" Threshold values out of dynamic range """
thvalues = 0

""" Print Eps """
printEPS = 0

""" Resize Factor """
ratio = 4

""" Radiometric Resolution """
L = 11

""" Dataset load """
if (im_tag == 'NY1'):
    mat = scipy.io.loadmat('NY1_WV3_FR.mat')
    I_MS_LR = mat.get('I_MS_LR')
    I_PAN = mat.get('I_PAN')
    I_MS_LR = I_MS_LR.astype('float64')
    I_PAN = I_PAN.astype('float64')
elif (im_tag == 'GeoEye1_May'):
    mat = scipy.io.loadmat('Collazzone_GeoEye_May_FR.mat')
    I_MS_LR = mat.get('I_MS_LR')
    I_PAN = mat.get('I_PAN')
    I_MS_LR = I_MS_LR.astype('float64')
    I_PAN = I_PAN.astype('float64')
elif (im_tag == 'Toulouse'):   
    mat = scipy.io.loadmat('Toulouse_IKONOS_FR.mat')
    I_MS_LR = mat.get('I_MS_LR')
    I_PAN = mat.get('I_PAN')
    I_MS_LR = I_MS_LR.astype('float64')
    I_PAN = I_PAN.astype('float64')
else:
    print("Error in loading the data")
    I_MS_LR = []
    I_PAN = []
    
    
""" GT"""
I_GT = I_MS_LR

"""  Preparation of image to fuse """
I_MS_LR, I_PAN = resize_images(I_MS_LR,I_PAN,ratio,sensor)

""" Upsampling"""
if (bicubic == 1):
    H = np.zeros((I_PAN.shape[0],I_PAN.shape[1],I_MS_LR.shape[2]))   
    for idim in range(I_MS_LR.shape[2]):
        H[:,:,idim] = imresize(I_MS_LR[:,:,idim],ratio)
    I_MS = H
else:
    I_MS = interp23(I_MS_LR,ratio)