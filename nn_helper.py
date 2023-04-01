import torch.nn as nn
#############################
# channels = height of matrix
#############################
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

