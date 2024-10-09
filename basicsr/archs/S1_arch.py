import torch
import time
from torch import nn as nn
from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBN, flow_warp, make_layer
from basicsr.archs.transformer_z import TransformerBlock, WADTL
import torch.nn.functional as F
from basicsr.archs.wave_tf import DWT, IWT
from basicsr.archs.wave_tf import HaarDownsampling
from basicsr.archs.kpn_pixel import IDynamicDWConv
from einops import rearrange
from basicsr.archs.common import *
from basicsr.archs.cross_block import CrossAttention
from basicsr.archs.propagation import manual_conv3d_propagation_backward, manual_conv3d_propagation_forward
class LE_arch(nn.Module):
    def __init__(self,n_feats = 64, n_encoder_res = 6):
        super(LE_arch, self).__init__()
        E1=[nn.Conv2d(96, n_feats, kernel_size=3, padding=1),
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
    def forward(self, x,gt):
        gt0 = self.pixel_unshuffle(gt)
        x0 = self.pixel_unshuffle(x)
        x = torch.cat([x0, gt0], dim=1)
        fea = self.E(x).squeeze(-1).squeeze(-1)
        fea1 = self.mlp(fea)
        return fea1



class WADT_arch(nn.Module):
    def __init__(self, num_feat=64, propagation_blocks = [4,2,1], num_blocks= [2,3,3,4], heads= [1,2,4,8], 
                 use_cross_attention=False, cross_attention_blocks=1, cross_attention_heads=8, bias=False):
        super().__init__()
        self.num_feat = num_feat

        self.feat_extractor = nn.Conv3d(3, num_feat, (1, 3, 3), 1, (0, 1, 1), bias=True)
        self.recons = nn.Conv3d(num_feat, 3, (1, 3, 3), 1, (0, 1, 1), bias=True)

        # wave tf
        self.wave = HaarDownsampling(num_feat)
        self.x_wave_1_conv1 = nn.Conv2d(num_feat * 3, num_feat * 3, 1, 1, 0, groups=3)
        self.x_wave_1_conv2 = nn.Conv2d(num_feat * 3, num_feat * 3, 1, 1, 0, groups=3)
        # wave pro
        self.x_wave_2_conv1 = nn.Conv2d(num_feat * 3, num_feat * 3, 1, 1, 0, groups=3)
        self.x_wave_2_conv2 = nn.Conv2d(num_feat * 3, num_feat * 3, 1, 1, 0, groups=3)

        self.transformer_scale4 = WADTL(dim=num_feat,num_blocks=num_blocks,heads=heads, 
                                            num_block = propagation_blocks, use_cross_attention=use_cross_attention, 
                                            cross_attention_heads = cross_attention_heads, bias = bias)
        self.backward_propagation_2 = manual_conv3d_propagation_backward(num_feat, propagation_blocks, use_cross_attention, cross_attention_heads, bias)
        self.forward_propagation_2 = manual_conv3d_propagation_forward(num_feat, propagation_blocks, use_cross_attention, cross_attention_heads, bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, lq, IPR):
        b, t, c, h, w = lq.size()
        lrs_feature = self.feat_extractor(rearrange(lq, 'b t c h w -> b c t h w'))     # b c t h w
        # scale1
        tf_input_feature = rearrange(lrs_feature, 'b c t h w -> (b t) c h w')
        tf_wave1_l, tf_wave1_h = self.wave(tf_input_feature)
        tf_wave1_h = self.x_wave_1_conv2(self.lrelu(self.x_wave_1_conv1(tf_wave1_h)))
        tf_wave2_l, tf_wave2_h = self.wave(tf_wave1_l)
        tf_wave2_l = rearrange(self.transformer_scale4(rearrange(tf_wave2_l, '(b t) c h w -> b t c h w', b=b), IPR), 'b t c h w -> (b t) c h w')
        tf_wave2_h = self.x_wave_2_conv2(self.lrelu(self.x_wave_2_conv1(tf_wave2_h)))
        # scale1
        tf_wave1_l = self.wave(torch.cat([tf_wave2_l, tf_wave2_h], dim=1), rev=True)
        tf_wave1_l = rearrange(self.forward_propagation_2(self.backward_propagation_2(rearrange(tf_wave1_l, '(b t) c h w -> b t c h w', b=b))), 'b t c h w -> (b t) c h w')

        pro_feat = rearrange(self.wave(torch.cat([tf_wave1_l, tf_wave1_h], dim=1), rev=True), '(b t) c h w -> b t c h w', b=b)
        out = rearrange(self.recons(rearrange(pro_feat, 'b t c h w -> b c t h w')), 'b c t h w -> b t c h w')
        return out.contiguous() + lq
    
@ARCH_REGISTRY.register()
class S1_arch(nn.Module):
    def __init__(self, num_feat=64, propagation_blocks = [4,2,1], num_blocks= [2,3,3,4], heads= [1,2,4,8], 
                 use_cross_attention=False, cross_attention_blocks=1, cross_attention_heads=8, bias=False):
        super().__init__()
        self.num_feat = num_feat
        # extractor & reconstruction
        self.LE = LE_arch(n_feats=64)
        self.WADT = WADT_arch(num_feat=num_feat, propagation_blocks = propagation_blocks, num_blocks= num_blocks, heads= heads, 
                 use_cross_attention=use_cross_attention, cross_attention_blocks=cross_attention_blocks, cross_attention_heads=cross_attention_heads, bias=bias)
        

    def forward(self, lq, gt):
        # time_start = time.time()
        # print(lrs.size())
        IPR = self.LE(rearrange(lq, 'b t c h w -> (b t) c h w'), rearrange(gt, 'b t c h w -> (b t) c h w'))
        output = self.WADT(lq, IPR)
        return output.contiguous() + lq, IPR


