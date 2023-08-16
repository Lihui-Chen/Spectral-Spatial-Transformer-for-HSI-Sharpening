import matplotlib.pyplot as plt
import numpy as np
import torch

import numpy as np
from skimage.exposure import histogram


def hist_line_stretch(img, nbins, bound=[0.01, 0.99]):
    def _line_strectch(img):
        # img = img.astype(np.uint16)
        ori = img
        img = img.reshape(-1)
        hist1, bins1 = histogram(img, nbins=nbins, normalize=True)
        cumhist = np.cumsum(hist1)
        lowThreshold= np.where(cumhist>=bound[0])[0][0]
        highThreshold = np.where(cumhist>=bound[1])[0][0]
        lowThreshold = bins1[lowThreshold]
        highThreshold = bins1[highThreshold]
        ori[np.where(ori<lowThreshold)] = lowThreshold
        ori[np.where(ori>highThreshold)] = highThreshold
        ori = (ori-lowThreshold)/(highThreshold-lowThreshold+np.finfo(np.float).eps)
        return ori, lowThreshold, highThreshold
    if img.ndim>2:
        lowThreshold=np.zeros(img.shape[2])
        highThreshold=np.zeros(img.shape[2])
        for i in range(img.shape[2]):
            img[:,:,i],lowThreshold[i], highThreshold[i]  = _line_strectch(img[:,:,i].squeeze())
    else:
        img, lowThreshold, highThreshold = _line_strectch(img)
    return img, lowThreshold, highThreshold


def vis_weights_curves(input_tensor, ):
    in_ch, out_ch, w, h = input_tensor.shape
    input_tensor = input_tensor.view(in_ch, out_ch)


    # fig, ax = plt.subplots(nrows=2, ncols=(out_ch+1)//2)
    fig = plt.figure() 
    # # for i in range(out_ch):
    # #     tmp_weights = input_tensor[:,i]
    #     tmp_weights = tmp_weights.cpu().numpy()
    #     ax[0].plot(tmp_weights)
    tmp_weights = input_tensor[0]
    tmp_weights = tmp_weights.cpu().numpy()
    plt.plot(tmp_weights)
    plt.show()
    plt.close()
    return fig
        
def linstretch(hr, sr, ):
    c, h, w = hr.shape
    for i in range(c):
        max_pix, min_pix = hr[i].max(), hr[i].min()
        data = torch.stack((hr[i], sr[i]), dim=0)
        data[data<min_pix] = min_pix
        data[data>max_pix] = max_pix
        data = (data-min_pix)/(max_pix-min_pix)
        hr[i], sr[i] = torch.split(data, 1, dim=0)

    return hr, sr