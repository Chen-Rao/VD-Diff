import torch
import time
from torch import nn as nn
from basicsr.utils.registry import ARCH_REGISTRY
from einops import rearrange
from .S2_arch import S2_arch

@ARCH_REGISTRY.register()
class S3_arch(S2_arch):
    def forward(self, lq, gt=None):
        if self.training:
            IPR_S2 = self.LE(rearrange(lq, 'b t c h w -> (b t) c h w'), rearrange(gt, 'b t c h w -> (b t) c h w'))
            IPR_DM, IPR_list = self.DM(lq, IPR_S2)
            output = self.WADT(lq, IPR_S2)
            return output.contiguous() + lq, IPR_S2, IPR_DM
        else:
            IPR = self.DM(lq)
            output = self.WADT(lq, IPR)
            return output.contiguous() + lq
        


