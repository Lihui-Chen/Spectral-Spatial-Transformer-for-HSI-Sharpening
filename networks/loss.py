import torch
import torch.nn as nn


class S3Loss(nn.Module):
    '''
    Choi, Jae-Seok; Kim, Yongwoo; Kim, Munchurl (2020): S3: A Spectral-Spatial Structure Loss for Pan-Sharpening Networks. 
    In IEEE Geosci. Remote Sensing Lett. 17 (5), pp. 829–833. DOI: 10.1109/LGRS.2019.2934493.
    '''
    def __init__(self, scale, device):
        super(S3Loss, self).__init__()
        self.device = device
        self.scale = scale
        self.L1loss = nn.L1Loss().to(device)
        
        

    def forward(self, lr_ms, hr_pan, hr_ms, fuse_ms):
        down_hr_pan = torch.nn.functional.upsample(hr_pan, scale_factor=1/self.scale, mode='area')

        M_ = torch.zeros(hr_pan.shape).cuda()
        G_ = torch.zeros(hr_pan.shape).cuda()
        for j in range(lr_ms.shape[0]):
            down_hr_pan_array = down_hr_pan[j].view(-1)
            down_hr_pan_array = fromDlpack(to_dlpack(down_hr_pan_array))
            for i in range(lr_ms.shape[1]):
                lr_ms_band = lr_ms[j,i,:,:].squeeze(0)
                lr_ms_band_array = lr_ms_band.view(-1)
                lr_ms_band_array = fromDlpack(to_dlpack(lr_ms_band_array))
                if i == 0:
                    A = cp.vstack([lr_ms_band_array**0, lr_ms_band_array**1])
                else:
                    A = cp.vstack([A,  lr_ms_band_array**1])
            
            sol, r, rank, s = cp.linalg.lstsq(A.T,down_hr_pan_array)
          
            sol = from_dlpack(toDlpack(sol))
            
            for i in range(hr_ms.shape[1]):
                M_[j] += hr_ms[j,i,:,:] * sol[i+1]
                G_[j] += fuse_ms[j,i,:,:] * sol[i+1]
            M_[j] += sol[0]
            G_[j] += sol[0] 
        
        mean_filter = kornia.filters.BoxBlur((31,31))
        e = torch.Tensor([1e-10]).cuda()
        r = 4
        a = 1 
        mean_M_ = mean_filter(M_)
        mean_P =  mean_filter(hr_pan) 
        mean_M_xP = mean_filter(M_*hr_pan)
        cov_M_xP = mean_M_xP - mean_M_*mean_P
        mean_M_xM_ = mean_filter(M_*M_)
        std_M_ = torch.sqrt(torch.abs(mean_M_xM_ - mean_M_*mean_M_) + e)
        mean_PxP = mean_filter(hr_pan*hr_pan)
        std_P = torch.sqrt(torch.abs(mean_PxP - mean_P*mean_P) + e)
        corr_M_xP = cov_M_xP / (std_M_*std_P)
        S = corr_M_xP**r
        loss_c = self.L1loss(fuse_ms*S, hr_ms*S)

        grad_P = (hr_pan - mean_P) / std_P
        mean_G_ = mean_filter(G_)
        mean_G_xG_ = mean_filter(G_*G_)
        std_G_ = torch.sqrt(torch.abs(mean_G_xG_ - mean_G_*mean_G_) + e)
        grad_G_ = (G_ - mean_G_) / std_G_
        loss_a = self.L1loss(grad_G_*(2-S), grad_P*(2-S))

        loss = loss_c + a*loss_a
        return loss