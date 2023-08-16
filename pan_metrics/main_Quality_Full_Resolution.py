# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 Vivone Gemine.
All rights reserved. This work should only be used for nonprofit purposes.

@author: Vivone Gemine (website: https://sites.google.com/site/vivonegemine/home )
"""
"""
 Description:
           Main for full resolution tests comparing with MATLAB results
 
 Notes:
     Results very close to the MATLAB toolbox's ones. In particular, the results of D_S are more accurate than the MATLAB toolbox's ones
     because the Q-index is applied in a sliding window way. Instead, for computational reasons, the MATLAB toolbox uses a distinct block implementation
     of the Q-index.

 References:
         G. Vivone, M. Dalla Mura, A. Garzelli, and F. Pacifici, "A Benchmarking Protocol for Pansharpening: Dataset, Pre-processing, and Quality Assessment", IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 14, pp. 6102-6118, 2021.
         G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M. O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting Pansharpening With Classical and Emerging Pansharpening Methods", IEEE Geoscience and Remote Sensing Magazine, vol. 9, no. 1, pp. 53 - 81, March 2021.     
"""

import scipy.io
import numpy as np
from .indexes_evaluation_FS import indexes_evaluation_FS

#dataset = "Toulouse_IKONOS_FR"
#dataset = "NY1_WV3_FR"
dataset = "Collazzone_GeoEye_July_FR"

""" Load dataset """
mat = scipy.io.loadmat(dataset + ".mat")

I_MS = mat.get('I_MS')
I_MS_LR = mat.get('I_MS_LR')
I_PAN = mat.get('I_PAN')
dim_cut = np.squeeze(mat.get('dim_cut'))
L = np.squeeze(mat.get('L'))
th_values = np.squeeze(mat.get('th_values'))
flag_cut_bounds = np.squeeze(mat.get('flag_cut_bounds'))
ratio = np.squeeze(mat.get('ratio'))
Qblocks_size = np.int32(np.squeeze(mat.get('Qblocks_size')))
sensor = mat.get('sensor')

I_MS = I_MS.astype('float64')
I_PAN = I_PAN.astype('float64')
I_MS_LR = I_MS_LR.astype('float64')


""" Load MATLAB results """
matRes = scipy.io.loadmat("results_" + dataset + ".mat")



""" Let's take some examples ..."""


""" EXP """
I_MS_LR = mat.get('I_MS_LR')
I_MS_LR = I_MS_LR.astype('float64')

HQNR_index, D_lambda, D_S = indexes_evaluation_FS(I_MS,I_MS_LR,I_PAN,L,th_values,I_MS,sensor,ratio,Qblocks_size)

""" Load MATLAB results """
QNR_EXP_MATLAB = matRes.get('QNRI_EXP')
D_S_EXP_MATLAB = matRes.get('D_S_EXP')
D_lambda_EXP_MATLAB = matRes.get('D_lambda_EXP')

""" Print results """
print("EXP")
print("DS: %.4f, DS MATLAB: %.4f, Error: %.4f" % (D_S, D_S_EXP_MATLAB, np.abs(D_S - D_S_EXP_MATLAB)))
print("Dl: %.4f, Dl MATLAB: %.4f, Error: %.4f" % (D_lambda, D_lambda_EXP_MATLAB, np.abs(D_lambda - D_lambda_EXP_MATLAB)))
print("HQNR: %.4f, HQNR MATLAB: %.4f, Error: %.4f" % (HQNR_index, QNR_EXP_MATLAB, np.abs(HQNR_index - QNR_EXP_MATLAB)))



""" BDSD """

""" Load fusion result """
I_BDSD = matRes.get('I_BDSD')
I_BDSD = I_BDSD.astype('float64')
I_MS_LR = mat.get('I_MS_LR')
I_MS_LR = I_MS_LR.astype('float64')

HQNR_index, D_lambda, D_S = indexes_evaluation_FS(I_BDSD,I_MS_LR,I_PAN,L,th_values,I_MS,sensor,ratio,Qblocks_size)

""" Load MATLAB results """
QNR_BDSD_MATLAB = matRes.get('QNRI_BDSD')
D_S_BDSD_MATLAB = matRes.get('D_S_BDSD')
D_lambda_BDSD_MATLAB = matRes.get('D_lambda_BDSD')

""" Print results """
print("BDSD")
print("DS: %.4f, DS MATLAB: %.4f, Error: %.4f" % (D_S, D_S_BDSD_MATLAB, np.abs(D_S - D_S_BDSD_MATLAB)))
print("Dl: %.4f, Dl MATLAB: %.4f, Error: %.4f" % (D_lambda, D_lambda_BDSD_MATLAB, np.abs(D_lambda - D_lambda_BDSD_MATLAB)))
print("HQNR: %.4f, HQNR MATLAB: %.4f, Error: %.4f" % (HQNR_index, QNR_BDSD_MATLAB, np.abs(HQNR_index - QNR_BDSD_MATLAB)))



""" MTF-GLP """

""" Load fusion result """
I_MTF_GLP = matRes.get('I_MTF_GLP')
I_MTF_GLP = I_MTF_GLP.astype('float64')
I_MS_LR = mat.get('I_MS_LR')
I_MS_LR = I_MS_LR.astype('float64')

HQNR_index, D_lambda, D_S = indexes_evaluation_FS(I_MTF_GLP,I_MS_LR,I_PAN,L,th_values,I_MS,sensor,ratio,Qblocks_size)

""" Load MATLAB results """
QNR_MTF_GLP_MATLAB = matRes.get('QNRI_MTF_GLP')
D_S_MTF_GLP_MATLAB = matRes.get('D_S_MTF_GLP')
D_lambda_MTF_GLP_MATLAB = matRes.get('D_lambda_MTF_GLP')

""" Print results """
print("MTF-GLP")
print("DS: %.4f, DS MATLAB: %.4f, Error: %.4f" % (D_S, D_S_MTF_GLP_MATLAB, np.abs(D_S - D_S_MTF_GLP_MATLAB)))
print("Dl: %.4f, Dl MATLAB: %.4f, Error: %.4f" % (D_lambda, D_lambda_MTF_GLP_MATLAB, np.abs(D_lambda - D_lambda_MTF_GLP_MATLAB)))
print("HQNR: %.4f, HQNR MATLAB: %.4f, Error: %.4f" % (HQNR_index, QNR_MTF_GLP_MATLAB, np.abs(HQNR_index - QNR_MTF_GLP_MATLAB)))