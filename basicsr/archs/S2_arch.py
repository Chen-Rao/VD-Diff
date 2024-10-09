import torch
import time
from torch import nn as nn
from basicsr.utils.registry import ARCH_REGISTRY
from einops import rearrange
from basicsr.archs.common import *
from basicsr.archs.ldm.ddpm import DDPM
from .S1_arch import LE_arch, WADT_arch

class CE(nn.Module):
    def __init__(self,n_feats = 64, n_encoder_res = 6):
        super(CE, self).__init__()
        E1=[nn.Conv2d(48, n_feats, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True)]
        E2=[
            ResBlock(
                default_conv, n_feats, kernel_size=3
            ) for _ in range(n_encoder_res)
        ]
        E3=[
            nn.Conv2d(n_feats, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        ]
        E=E1+E2+E3
        self.E = nn.Sequential(
            *E
        )
        self.mlp = nn.Sequential(
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.LeakyReLU(0.1, True)
        )
        self.pixel_unshuffle = nn.PixelUnshuffle(4)
    def forward(self, x):
        x = self.pixel_unshuffle(x)
        fea = self.E(x).squeeze(-1).squeeze(-1)
        fea1 = self.mlp(fea)
        return fea1
    
class ResMLP(nn.Module):
    def __init__(self,n_feats = 512):
        super(ResMLP, self).__init__()
        self.resmlp = nn.Sequential(
            nn.Linear(n_feats , n_feats ),
            nn.LeakyReLU(0.1, True),
        )
    def forward(self, x):
        res=self.resmlp(x)
        return res

class denoise(nn.Module):
    def __init__(self,n_feats = 64, n_denoise_res = 5,timesteps=5):
        super(denoise, self).__init__()
        self.max_period=timesteps*10
        n_featsx4=4*n_feats
        resmlp = [
            nn.Linear(n_featsx4*2+1, n_featsx4),
            nn.LeakyReLU(0.1, True),
        ]
        for _ in range(n_denoise_res):
            resmlp.append(ResMLP(n_featsx4))
        self.resmlp=nn.Sequential(*resmlp)

    def forward(self,x, t,c):
        t=t.float()
        t =t/self.max_period
        t=t.view(-1,1)
        c = torch.cat([c,t,x],dim=1)
        fea = self.resmlp(c)

        return fea 
    
class DM_arch(nn.Module):
    def __init__(self, num_feat=64, n_denoise_res = 1, linear_start= 0.1, linear_end= 0.99, timesteps = 4):
        super().__init__()
        self.num_feat = num_feat
        self.condition = CE(n_feats=64) # n_encoder_res=n_encoder_res
        self.denoise= denoise(n_feats=64, n_denoise_res=n_denoise_res,timesteps=timesteps)

        self.diffusion = DDPM(denoise=self.denoise, condition=self.condition ,n_feats=64,linear_start= linear_start,
                              linear_end= linear_end, timesteps = timesteps)
        
    def forward(self, lq, IPR_S1=None):
        if self.training:
            IPR, IPR_list = self.diffusion(rearrange(lq, 'b t c h w -> (b t) c h w'),IPR_S1)
            return IPR, IPR_list
        else:
            IPR = self.diffusion(rearrange(lq, 'b t c h w -> (b t) c h w'))
        return IPR    
    
@ARCH_REGISTRY.register()
class S2_arch(nn.Module):
    def __init__(self, num_feat=64, propagation_blocks = [4,2,1], num_blocks= [2,3,3,4], heads= [1,2,4,8], 
                 use_cross_attention=False, cross_attention_blocks=1, cross_attention_heads=8, bias=False,
                 n_denoise_res = 1, linear_start= 0.1, linear_end= 0.99, timesteps = 4):
        super().__init__()
        self.num_feat = num_feat
        # extractor & reconstruction
        
        self.LE = LE_arch(n_feats=64)
        self.DM = DM_arch(num_feat=num_feat, n_denoise_res = n_denoise_res, linear_start= linear_start, linear_end= linear_end, timesteps = timesteps)
        self.WADT = WADT_arch(num_feat=num_feat, propagation_blocks = propagation_blocks, num_blocks= num_blocks, heads= heads, 
                 use_cross_attention=use_cross_attention, cross_attention_blocks=cross_attention_blocks, cross_attention_heads=cross_attention_heads, bias=bias)
        
    def forward(self, lq, gt=None):
        if self.training:
            IPR_S1 = self.LE(rearrange(lq, 'b t c h w -> (b t) c h w'), rearrange(gt, 'b t c h w -> (b t) c h w'))
            IPR_DM, IPR_list = self.DM(lq, IPR_S1)
            return IPR_S1, IPR_DM
        else:
            IPR = self.DM(lq)
            output = self.WADT(lq, IPR)
            return output.contiguous() + lq
        


