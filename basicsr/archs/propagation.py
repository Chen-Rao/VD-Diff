import torch
import time
from torch import nn as nn
from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBN, flow_warp, make_layer
# from basicsr.archs.ChanDynamic_GMLP import TransformerBlock
from basicsr.archs.kpn_pixel import IDynamicDWConv
from einops import rearrange
from basicsr.archs.common import *
from basicsr.archs.cross_block import CrossAttention

class ResidualBlocks2D(nn.Module):
    def __init__(self, num_feat=64, num_block=30):
        super().__init__()
        self.main = nn.Sequential(
            make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat))

    def forward(self, fea):
        return self.main(fea)
class manual_conv3d_propagation_backward(nn.Module):
    def __init__(self, num_feat=64, num_block=15, use_cross_attention=False, cross_attention_heads=8, bias=False):
        super().__init__()
        self.num_feat = num_feat
        if use_cross_attention:
            self.cross_attention = CrossAttention(num_feat, cross_attention_heads, bias)
        else:
            self.conv1 = nn.Conv2d(num_feat * 2, num_feat * 2, 3, 1, 1, bias=True)
            self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
            self.conv3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
            
            self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.kernel_conv_pixel = IDynamicDWConv(num_feat, 3, 1, 4, 1)
        self.resblock_bcakward2d = ResidualBlocks2D(num_feat, num_block)
        self.use_cross_attention = use_cross_attention

    def forward(self, feature):
        # predefine
        b, t, c, h, w = feature.size()                           # b t 64 256 256
        backward_list = []
        if self.use_cross_attention:
            feat_prop = feature[:, -1, :, :, :]
        else:
            feat_prop = feature.new_zeros(b, c, h, w)
        # propagation
        for i in range(t - 1, -1, -1):
            x_feat = feature[:, i, :, :, :]
            # fusion propagation
            if self.use_cross_attention:
                feat_prop = self.cross_attention(x_feat, feat_prop)
            else:
                feat_fusion = torch.cat([x_feat, feat_prop], dim=1)  # b 128 256 256
                feat_fusion = self.lrelu(self.conv1(feat_fusion))  # b 128 256 256
                feat_prop1, feat_prop2 = torch.split(feat_fusion, self.num_feat, dim=1)
                feat_prop1 = feat_prop1 * torch.sigmoid(self.conv2(feat_prop1))
                feat_prop2 = feat_prop2 * torch.sigmoid(self.conv3(feat_prop2))
                feat_prop = feat_prop1 + feat_prop2
            # dynamic conv
            feat_prop = self.kernel_conv_pixel(feat_prop)
            # resblock2D
            feat_prop = self.resblock_bcakward2d(feat_prop)
            backward_list.append(feat_prop)

        backward_list = backward_list[::-1]
        conv3d_feature = torch.stack(backward_list, dim=1)      # b 64 t 256 256
        return conv3d_feature


class manual_conv3d_propagation_forward(nn.Module):
    def __init__(self, num_feat=64, num_block=15, use_cross_attention=False, cross_attention_heads=8, bias=False):
        super().__init__()
        self.num_feat = num_feat
        if use_cross_attention:
            self.cross_attention = CrossAttention(num_feat, cross_attention_heads, bias)
        else:
            self.conv1 = nn.Conv2d(num_feat * 2, num_feat * 2, 3, 1, 1, bias=True)
            self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
            self.conv3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
            
            self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.kernel_conv_pixel = IDynamicDWConv(num_feat, 3, 1, 4, 1)
        self.resblock_bcakward2d = ResidualBlocks2D(num_feat, num_block)
        self.use_cross_attention = use_cross_attention
    def forward(self, feature):
        # predefine
        b, t, c, h, w = feature.size()                          # b t 64 256 256
        forward_list = []
        if self.use_cross_attention:
            feat_prop = feature[:, 0, :, :, :]
        else:
            feat_prop = feature.new_zeros(b, c, h, w)
        for i in range(0, t):
            x_feat = feature[:, i, :, :, :]
            # fusion propagation
            if self.use_cross_attention:
                feat_prop = self.cross_attention(x_feat, feat_prop)
            else:
                feat_fusion = torch.cat([x_feat, feat_prop], dim=1)  # b 128 256 256
                feat_fusion = self.lrelu(self.conv1(feat_fusion))  # b 128 256 256
                feat_prop1, feat_prop2 = torch.split(feat_fusion, self.num_feat, dim=1)
                feat_prop1 = feat_prop1 * torch.sigmoid(self.conv2(feat_prop1))
                feat_prop2 = feat_prop2 * torch.sigmoid(self.conv3(feat_prop2))
                feat_prop = feat_prop1 + feat_prop2
            # dynamic conv
            feat_prop = self.kernel_conv_pixel(feat_prop)
            # resblock2D
            feat_prop = self.resblock_bcakward2d(feat_prop)
            forward_list.append(feat_prop)

        conv3d_feature = torch.stack(forward_list, dim=1)      # b 64 t 256 256
        return conv3d_feature