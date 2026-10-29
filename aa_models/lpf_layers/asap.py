'''
ASAP module
can be used and distributed under the MIT license
Reference:
[1] Grabinski, J., Jung, S., Keuper, J., & Keuper, M. (2022). 
    "FrequencyLowCut Pooling--Plug & Play against Catastrophic Overfitting."
    European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2022.
[2] Grabinski, J., Keuper, J. and Keuper, M. 
    "Fix your downsampling ASAP! Be natively more robust via Aliasing and Spectral Artifact free Pooling." 
    arXiv preprint arXiv:2307.09804 (2023).
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# ---------------------------------------------------------------------------
# CHANGED (AMP compatibility) -- only modification compared to the reference
# implementation by Grabinski et al. (github.com/GeJulia/flc_pooling, asap.py).
# Change was needed to avoid runtime errors when using automatic mixed precision 
# (torch.autocast, --amp) for speed ups. FFTs in FP16 only support power-of-two signal.
# The other methods here do not support FP16 FFTs. For consistency all is FP32.
#
# Problem: Under automatic mixed precision (torch.autocast, --amp) the input of
# ASAP arrives as FP16 (output of the preceding Conv/BN/ReLU). torch.fft.fft2 on
# FP16 CUDA tensors only supports signal lengths that are powers of two
# (PyTorch docs, torch.fft.fft2). ASAP_padding_one pads 64 -> 65 (33, 17) and
# ASAP_padding_large pads to 127 (63, 31) -> runtime error. ASAP (stbl) uses
# 64/32/16 and would run, but with the experimental, less precise complex-half FFT.
#
# Fix: each forward() now disables autocast locally and casts the input to FP32;
# the original computation is unchanged and lives in _forward_fp32(). The output
# is FP32; the following convolution is cast back to FP16 by autocast as usual.
#
# Effect: Without --amp nothing changes (autocast is off anyway and x is already
# FP32) -> bitwise identical to the reference. With --amp, ASAP is computed in
# FP32 exactly as in the reference implementation.
#
# THESIS: "All models were trained with automatic mixed precision (Micikevicius
# et al., 2018). Numerically sensitive operations -- the FFT in ASAP and the
# DWT -- are computed in FP32, since FFTs in FP16 are only supported for
# power-of-two signal lengths. All evaluations are performed in FP32."
# THESIS: "Apart from this precision handling, ASAP is used unchanged from the
# reference implementation of Grabinski et al."
# ---------------------------------------------------------------------------


class ASAP(nn.Module):
    # pooling trough selecting only the low frequent part in the fourier domain and only using this part to go back into the spatial domain
    # save computations as we do not need to do the downsampling trough conv with stride 2
    # using a hamming window to prevent sinc-interpolation artifacts
    def __init__(self, transpose=True):
        self.transpose = transpose
        self.window2d = None
        super(ASAP, self).__init__()

    def forward(self, x):
        # CHANGED (AMP): run ASAP in FP32 -- FFT in FP16 only supports power-of-two
        # sizes and is less precise. No effect without --amp (bitwise identical).
        with torch.autocast(device_type="cuda", enabled=False):
            return self._forward_fp32(x.float())

    def _forward_fp32(self, x):  # CHANGED: renamed from forward(); body unchanged (reference code)

        if self.transpose:
            x = x.transpose(2,3)
        if self.window2d is None:
            window1d = np.abs(np.hamming(x.size(2)))
            window2nd = np.abs(np.hamming(x.size(3)))
            window2d = np.sqrt(np.outer(window1d,window2nd))
            self.window2d = torch.Tensor(window2d).cuda()
            del window1d
            del window2d
            del window2nd
        
        low_part = torch.fft.fftshift(torch.fft.fft2(x, norm='forward'))
        low_part = low_part*self.window2d.unsqueeze(0).unsqueeze(0)
        low_part = low_part[:,:,int(x.size(2)/4):int(x.size(2)/4*3),int(x.size(3)/4):int(x.size(3)/4*3)]
        
        return torch.fft.ifft2(torch.fft.ifftshift(low_part), norm='forward').real


class ASAP_padding_one(nn.Module):
    # pooling trough selecting only the low frequent part in the fourier domain and only using this part to go back into the spatial domain
    # save computations as we do not need to do the downsampling trough conv with stride 2
    # using a hamming window to prevent sinc-interpolation artifacts
    def __init__(self):
        self.window2d = None
        super(ASAP_padding_one, self).__init__()

    def forward(self, x):
        # CHANGED (AMP): run ASAP in FP32 -- FFT in FP16 only supports power-of-two
        # sizes and is less precise. No effect without --amp (bitwise identical).
        with torch.autocast(device_type="cuda", enabled=False):
            return self._forward_fp32(x.float())

    def _forward_fp32(self, x):  # CHANGED: renamed from forward(); body unchanged (reference code)
        
        x = F.pad(x, (0, 1, 0, 1), "constant", 0)
        if not torch.is_tensor(self.window2d):
            window1d = np.abs(np.hamming(x.size(2)))
            window2nd = np.abs(np.hamming(x.size(3)))
            window2d = np.sqrt(np.outer(window1d,window2nd))
            self.window2d = torch.Tensor(window2d).cuda()
            del window1d
            del window2d
            del window2nd
            
        low_part = torch.fft.fftshift(torch.fft.fft2(x, norm='forward'))
        low_part = low_part*self.window2d.unsqueeze(0).unsqueeze(0)
        low_part = low_part[:,:,int(x.size(2)/4):int(x.size(2)/4*3),int(x.size(3)/4):int(x.size(3)/4*3)]
        
        fc = torch.fft.ifft2(torch.fft.ifftshift(low_part), norm='forward').real
        return fc


class ASAP_padding_large(nn.Module):
    # pooling trough selecting only the low frequent part in the fourier domain and only using this part to go back into the spatial domain
    # save computations as we do not need to do the downsampling trough conv with stride 2
    # using a hamming window to prevent sinc-interpolation artifacts
    def __init__(self):
        self.window2d = None
        super(ASAP_padding_large, self).__init__()

    def forward(self, x):
        # CHANGED (AMP): run ASAP in FP32 -- FFT in FP16 only supports power-of-two
        # sizes and is less precise. No effect without --amp (bitwise identical).
        with torch.autocast(device_type="cuda", enabled=False):
            return self._forward_fp32(x.float())

    def _forward_fp32(self, x):  # CHANGED: renamed from forward(); body unchanged (reference code)
        
        x = F.pad(x, (int(x.size(3)/2-1), int(x.size(3)/2), int(x.size(2)/2-1), int(x.size(2)/2)), "constant", 0)
        if not torch.is_tensor(self.window2d):
            window1d = np.abs(np.hamming(x.size(2)))
            window2nd = np.abs(np.hamming(x.size(3)))
            window2d = np.sqrt(np.outer(window1d,window2nd))
            self.window2d = torch.Tensor(window2d).cuda()
            del window1d
            del window2d
            del window2nd
            
        low_part = torch.fft.fftshift(torch.fft.fft2(x, norm='forward'))
        low_part = low_part*self.window2d.unsqueeze(0).unsqueeze(0)
        low_part = low_part[:,:,int(x.size(2)/4):int(x.size(2)/4*3),int(x.size(3)/4):int(x.size(3)/4*3)]
        
        fc = torch.fft.ifft2(torch.fft.ifftshift(low_part), norm='forward').real
        fc = fc[:,:,int(fc.size(2)/4):int(3*fc.size(2)/4),int(fc.size(3)/4): int(3*fc.size(3)/4)]
        return fc