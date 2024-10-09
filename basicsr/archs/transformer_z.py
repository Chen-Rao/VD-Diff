from einops import rearrange
import torch
import torch.nn as nn
import numbers
import torch.nn.functional as F
from basicsr.archs.propagation import manual_conv3d_propagation_backward, manual_conv3d_propagation_forward
from torchvision import transforms
import matplotlib.pyplot as plt
import scipy
import numpy as np
import os


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.kernel = nn.Sequential(
            nn.Linear(256, dim*2, bias=False),
        )
        self.qkv = nn.Conv3d(dim, dim*3, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=bias)
        


    def forward(self, x,k_v):
        b,t,c,h,w = x.shape
        k_v=self.kernel(k_v).view(b,t,c*2,1,1)
        k_v1,k_v2=k_v.chunk(2, dim=2)
        x = x*k_v1+k_v2  
        
        x = rearrange(x, 'b t c h w -> b c t h w')
        qkv = self.qkv_dwconv(rearrange(self.qkv(x), 'b c t h w -> (b t) c h w'))
        q,k,v = qkv.chunk(3, dim=1)   
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        
        out = rearrange(self.project_out(rearrange(out, '(b t) c h w -> b c t h w',b=b)), 'b c t h w -> b t c h w')
        return out
    
class DynamicDWConv(nn.Module):
    def __init__(self, channels, kernel_size, stride=1, groups=1):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2
        self.groups = groups

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        Block1 = [nn.Conv2d(channels, channels, kernel_size, 1, kernel_size//2, groups=channels)
                  for _ in range(3)]
        Block2 = [nn.Conv2d(channels, channels, kernel_size, 1, kernel_size//2, groups=channels)
                  for _ in range(3)]
        self.tokernel = nn.Conv2d(channels, kernel_size ** 2 * self.channels, 1, 1, 0)
        self.bias = nn.Parameter(torch.zeros(channels))
        self.Block1 = nn.Sequential(*Block1)
        self.Block2 = nn.Sequential(*Block2)

    def forward(self, x):
        b, c, h, w = x.shape
        weight = self.tokernel(self.pool(self.Block2(self.maxpool(self.Block1(self.avgpool(x))))))
        weight = weight.view(b * self.channels, 1, self.kernel_size, self.kernel_size)
        x = F.conv2d(x.reshape(1, -1, h, w), weight, self.bias.repeat(b), stride=self.stride, padding=self.padding, groups=b * self.groups)
        x = x.view(b, c, x.shape[-2], x.shape[-1])
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv3d(dim, hidden_features*2, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=bias)
        self.project_out = nn.Conv3d(hidden_features, dim, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=bias)
        # self.kerner_conv_channel = DynamicDWConv(hidden_features, 3, 1, hidden_features)
        self.kerner_conv_channel = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.kernel = nn.Sequential(
            nn.Linear(256, dim*2, bias=False),
        )
    def forward(self, x, k_v):
        b,c,t,h,w = x.shape
        k_v=self.kernel(k_v).view(-1,c*2,1,1)
        k_v1,k_v2=k_v.chunk(2, dim=1)
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = x*k_v1+k_v2  
        x = rearrange(x, '(b t) c h w -> b c t h w', b=b)
        x = self.project_in(x)
        x1, x2 = rearrange(self.kerner_conv_channel(rearrange(x, 'b c t h w -> (b t) c h w')), '(b t) c h w -> b c t h w', b=b).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
    

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        b=x.shape[0]
        return rearrange(self.body(rearrange(x, 'b t c h w -> (b t) c h w')), '(b t) c h w -> b t c h w', b=b)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        b=x.shape[0]
        return rearrange(self.body(rearrange(x, 'b t c h w -> (b t) c h w')), '(b t) c h w -> b t c h w', b=b)
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, y):
        x = y[0]
        k_v=y[1]
        b = x.shape[0]
        x = x + self.attn(rearrange(self.norm1(rearrange(x, 'b t c h w -> (b t) c h w')),'(b t) c h w -> b t c h w', b=b),k_v)
        x = x + rearrange(self.ffn(rearrange(rearrange(self.norm2(rearrange(x, 'b t c h w -> (b t) c h w')), '(b t) c h w -> b t c h w', b=b), 'b t c h w -> b c t h w'), k_v), 
                          'b c t h w -> b t c h w')
        return [x,k_v]
class WADTL(nn.Module):
    def __init__(self, 
        size = 512,
        inp_channels=3, 
        out_channels=3, 
        dim = 64,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',
        use_cross_attention=False, cross_attention_heads=8,num_block=15
    ):

        super(WADTL, self).__init__()
        
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.down1_2 = Downsample(dim) 
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.up2_1 = Upsample(int(dim*2**1)) 
        self.reduce_chan_level1 = nn.Conv2d(int(dim*2), int(dim), kernel_size=1, bias=bias)
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.output = Upsample(int(dim))

    def forward(self, inp_img, k_v):
        b = inp_img.shape[0]
        out_enc_level1,*_ = self.encoder_level1([inp_img,k_v])
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        latent,*_ = self.latent([inp_enc_level2,k_v])       

        inp_dec_level1 = self.up2_1(latent)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 2)
        inp_dec_level1 = rearrange(self.reduce_chan_level1(rearrange(inp_dec_level1, 'b t c h w -> (b t) c h w')), '(b t) c h w -> b t c h w', b=b)
        out_dec_level1,_ = self.decoder_level1([inp_dec_level1,k_v])
        ouput = out_dec_level1 + inp_img
        
        return ouput
        


