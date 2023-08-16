# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 Vivone Gemine.
All rights reserved. This work should only be used for nonprofit purposes.

@author: Vivone Gemine (website: https://sites.google.com/site/vivonegemine/home )
"""
"""
 Description:
           Main for reduced resolution tests comparing with MATLAB results
 
 Notes:
     Results very close to the MATLAB toolbox's ones. Only Q_avg differs from the MATLAB implementation when S is even (when S is odd the results are identical).
     Indeed, when S is even ssim cannot work and S is modified to the closest odd number getting very similar results
     with respect to the MATLAB toolbox's ones.
     
 References:
         G. Vivone, M. Dalla Mura, A. Garzelli, and F. Pacifici, "A Benchmarking Protocol for Pansharpening: Dataset, Pre-processing, and Quality Assessment", IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 14, pp. 6102-6118, 2021.
         G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M. O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting Pansharpening With Classical and Emerging Pansharpening Methods", IEEE Geoscience and Remote Sensing Magazine, vol. 9, no. 1, pp. 53 - 81, March 2021.    
"""

import scipy.io
import numpy as np
from .indexes_evaluation import indexes_evaluation 

#dataset = "Toulouse_IKONOS_RR"
#dataset = "NY1_WV3_RR"
dataset = "Collazzone_GeoEye_July_RR"

""" Load dataset """
mat = scipy.io.loadmat(dataset + ".mat")

I_GT = mat.get('I_GT')
I_MS = mat.get('I_MS')
dim_cut = np.squeeze(mat.get('dim_cut'))
L = np.squeeze(mat.get('L'))
th_values = np.squeeze(mat.get('th_values'))
flag_cut_bounds = np.squeeze(mat.get('flag_cut_bounds'))
ratio = np.squeeze(mat.get('ratio'))
Qblocks_size = np.int32(np.squeeze(mat.get('Qblocks_size')))
sensor = mat.get('sensor')

I_GT = I_GT.astype('float64')
I_MS = I_MS.astype('float64')

""" Load MATLAB results """
matRes = scipy.io.loadmat("results_" + dataset + ".mat")



""" Take some examples ..."""


""" EXP """
Q2n_value, Q_value, ERGAS_value, SAM_value = indexes_evaluation(I_MS,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,th_values)

""" Load MATLAB results """
Q_EXP_MATLAB = matRes.get('Q_EXP')
ERGAS_EXP_MATLAB = matRes.get('ERGAS_EXP')
SAM_EXP_MATLAB = matRes.get('SAM_EXP')
Q_avg_EXP_MATLAB = matRes.get('Q_avg_EXP')

""" Print results """
print("EXP")
print("ERGAS: %.4f, ERGAS MATLAB: %.4f, Error: %.4f" % (ERGAS_value, ERGAS_EXP_MATLAB, np.abs(ERGAS_value - ERGAS_EXP_MATLAB)))
print("SAM: %.4f, SAM MATLAB: %.4f, Error: %.4f" % (SAM_value, SAM_EXP_MATLAB, np.abs(SAM_value - SAM_EXP_MATLAB)))
print("Qavg: %.4f, Qavg MATLAB: %.4f, Error: %.4f" % (Q_value, Q_avg_EXP_MATLAB, np.abs(Q_value - Q_avg_EXP_MATLAB)))
print("Q2n: %.4f, Q2n MATLAB: %.4f, Error: %.4f" % (Q2n_value, Q_EXP_MATLAB, np.abs(Q2n_value - Q_EXP_MATLAB)))



""" BDSD """

""" Load fusion result """
I_BDSD = matRes.get('I_BDSD')
I_BDSD = I_BDSD.astype('float64')

Q2n_value, Q_value, ERGAS_value, SAM_value = indexes_evaluation(I_BDSD,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,th_values)

""" Load MATLAB results """
Q_BDSD_MATLAB = matRes.get('Q_BDSD')
ERGAS_BDSD_MATLAB = matRes.get('ERGAS_BDSD')
SAM_BDSD_MATLAB = matRes.get('SAM_BDSD')
Q_avg_BDSD_MATLAB = matRes.get('Q_avg_BDSD')

""" Print results """
print("BDSD")
print("ERGAS: %.4f, ERGAS MATLAB: %.4f, Error: %.4f" % (ERGAS_value, ERGAS_BDSD_MATLAB, np.abs(ERGAS_value - ERGAS_BDSD_MATLAB)))
print("SAM: %.4f, SAM MATLAB: %.4f, Error: %.4f" % (SAM_value, SAM_BDSD_MATLAB, np.abs(SAM_value - SAM_BDSD_MATLAB)))
print("Qavg: %.4f, Qavg MATLAB: %.4f, Error: %.4f" % (Q_value, Q_avg_BDSD_MATLAB, np.abs(Q_value - Q_avg_BDSD_MATLAB)))
print("Q2n: %.4f, Q2n MATLAB: %.4f, Error: %.4f" % (Q2n_value, Q_BDSD_MATLAB, np.abs(Q2n_value - Q_BDSD_MATLAB)))



""" MTF-GLP """

""" Load fusion result """
I_MTF_GLP = matRes.get('I_MTF_GLP')
I_MTF_GLP = I_MTF_GLP.astype('float64')

Q2n_value, Q_value, ERGAS_value, SAM_value = indexes_evaluation(I_MTF_GLP,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,th_values)

""" Load MATLAB results """
Q_MTF_GLP_MATLAB = matRes.get('Q_MTF_GLP')
ERGAS_MTF_GLP_MATLAB = matRes.get('ERGAS_MTF_GLP')
SAM_MTF_GLP_MATLAB = matRes.get('SAM_MTF_GLP')
Q_avg_MTF_GLP_MATLAB = matRes.get('Q_avg_MTF_GLP')

""" Print results """
print("MTF-GLP")
print("ERGAS: %.4f, ERGAS MATLAB: %.4f, Error: %.4f" % (ERGAS_value, ERGAS_MTF_GLP_MATLAB, np.abs(ERGAS_value - ERGAS_MTF_GLP_MATLAB)))
print("SAM: %.4f, SAM MATLAB: %.4f, Error: %.4f" % (SAM_value, SAM_MTF_GLP_MATLAB, np.abs(SAM_value - SAM_MTF_GLP_MATLAB)))
print("Qavg: %.4f, Qavg MATLAB: %.4f, Error: %.4f" % (Q_value, Q_avg_MTF_GLP_MATLAB, np.abs(Q_value - Q_avg_MTF_GLP_MATLAB)))
print("Q2n: %.4f, Q2n MATLAB: %.4f, Error: %.4f" % (Q2n_value, Q_MTF_GLP_MATLAB, np.abs(Q2n_value - Q_MTF_GLP_MATLAB)))



"""SR-D """

""" Load fusion result """
I_SR_D = matRes.get('I_SR_D')
I_SR_D = I_SR_D.astype('float64')

Q2n_value, Q_value, ERGAS_value, SAM_value = indexes_evaluation(I_SR_D,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,th_values)

""" Load MATLAB results """
Q_SR_D_MATLAB = matRes.get('Q_SR_D')
ERGAS_SR_D_MATLAB = matRes.get('ERGAS_SR_D')
SAM_SR_D_MATLAB = matRes.get('SAM_SR_D')
Q_avg_SR_D_MATLAB = matRes.get('Q_avg_SR_D')

""" Print results """
print("SR-D")
print("ERGAS: %.4f, ERGAS MATLAB: %.4f, Error: %.4f" % (ERGAS_value, ERGAS_SR_D_MATLAB, np.abs(ERGAS_value - ERGAS_SR_D_MATLAB)))
print("SAM: %.4f, SAM MATLAB: %.4f, Error: %.4f" % (SAM_value, SAM_SR_D_MATLAB, np.abs(SAM_value - SAM_SR_D_MATLAB)))
print("Qavg: %.4f, Qavg MATLAB: %.4f, Error: %.4f" % (Q_value, Q_avg_SR_D_MATLAB, np.abs(Q_value - Q_avg_SR_D_MATLAB)))
print("Q2n: %.4f, Q2n MATLAB: %.4f, Error: %.4f" % (Q2n_value, Q_SR_D_MATLAB, np.abs(Q2n_value - Q_SR_D_MATLAB)))