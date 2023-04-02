import torch
import torch.nn as nn
"""Encapsulated: two convolutions, each followed by ReLU"""
class Conv3x3_Relu_2x(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.go = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU()
        )
    def forward(self, x):
        return self.go(x)
"""Concatenate"""
# we have big image from contracting step,
# need to concatenate it to small image from expansive step
# UNet's implementation makes small image bigger, using padding
# Bug UNet paper says we are cropping bigger image to fit to smaller
# I assume image are squares (should be fine because, MRI image has black background)
def Concatenate(big, small):
    big_len = big.size()[0]
    small_len = small.size()[0]
    start = big_len//2 - small_len//2
    end = start + small_len
    big = big[start:end, start:end]
    return torch.cat([big, small], dim=1)

