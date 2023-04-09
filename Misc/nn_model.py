import torch
import torch.nn as nn
from nn_helper import * # 'import nn_helper' NO work!
"""Neural Network"""
# Reference:
# https://arxiv.org/pdf/1505.04597.pdf
# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

class Me_Net(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Me_Net, self).__init__()
        # n_channels = 2 for MRI image (real & complex part for ea pixel)
        self.n_channels = n_channels
        # n_classes: probability distribution for ea pixel. =2 for MRI image
        self.n_classes = n_classes
        # (?) No need flatten, because I did it in preprocessing
        # (?) cannot use custom function inside nn.Sequential ;-;
        # two 3x3 convolutions, unpadded, each followed by rectified linear unit
        # in_channels = 2, because MRI image has real and complex parts
        # (?) out_channels = 2, because output and input images have same format
        # Question: stride = 1 ?
        # (?) no padding?
        self.Contract_1 = (Conv3x3_Relu_2x(n_channels, 64))
        # followed by 2x2 max pooling, stride = 2, for downsampling
        self.c1_c2 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.Contract_2 = (Conv3x3_Relu_2x(64, 128))
        self.c2_c3 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.Contract_3 = (Conv3x3_Relu_2x(128, 256))
        self.c3_c4 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.Contract_4 = (Conv3x3_Relu_2x(256, 512))
        self.c4_c5 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.Bottom = (Conv3x3_Relu_2x(512, 1024))
        # upsampling of the feature map,
        # followed by a 2x2 (up-)convolution that halves the feature channels 
        # (?) transpose convolution OR upsampling
        # seems like transpose convolution because I can't find way to reduce depth with upsampling
        # (?) how many steps for transpose convolution?
        # seems like 2, because in contracting step, 2x2 filter + stride 2 = 1/2 image dimensions
        # (?) a concatenation with the correspondingly cropped feature map from the contracting path
        # and two 3x3 convolutions, each followed by a ReLU
        # (?) align_corners = False
        # Reference: https://github.com/pytorch/vision/issues/1708
        self.c5_c4 = nn.ConvTranspose2d(1024, 512, kernel_size=(10,10), stride=2)
        self.Expand_4 = (Conv3x3_Relu_2x(1024, 512))
        self.c4_c3 = nn.ConvTranspose2d(512, 256, kernel_size=(18,18), stride=2)
        self.Expand_3 = (Conv3x3_Relu_2x(512, 256))
        self.c3_c2 = nn.ConvTranspose2d(256, 128, kernel_size=(18,18), stride=2)
        self.Expand_2 = (Conv3x3_Relu_2x(512, 128))
        self.c2_c1 = nn.ConvTranspose2d(128, 64, kernel_size=(26,26), stride=2)
        # upsampling: just operator, NO param
        # convTransPose: has param, can train
        self.Expand_1 = (Conv3x3_Relu_2x(128, 64))
        self.out = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=(1, 1), stride=1)
    def forward(self, x):
        Cont_1 = self.Contract_1(x) # take 1/2 to concatenate
        c1_c2 = self.c1_c2(Cont_1)
        Cont_2 = self.Contract_2(c1_c2) # take 1/2 to concatenate
        c2_c3 = self.c2_c3(Cont_2)
        Cont_3 = self.Contract_3(c2_c3) # take 1/2 to concatenate
        c3_c4 = self.c3_c4(Cont_3)
        Cont_4 = self.Contract_4(c3_c4) # take 1/2 to concatenate
        c4_c5 = self.c4_c5(Cont_4)
        bottom = self.Bottom(c4_c5)
        c5_c4_half = self.c5_c4(bottom) # add other 1/2
        c5_c4_full = torch.cat([Cont_4, c5_c4_half])
        Expa_4 = self.Expand_4(c5_c4_full) 
        c4_c3_half = self.c4_c3(Expa_4) # add other 1/2
        c4_c3_full = torch.cat([Cont_3, c4_c3_half])
        Expa_3 = self.Expand_3(c4_c3_full)
        c3_c2_half = self.c3_c2(Expa_3) # add other 1/2
        c3_c2_full = torch.cat([Cont_2, c3_c2_half])
        Expa_2 = self.Expand_2(c3_c2_full)
        c2_c1_half = self.c3_c2(Expa_2) # add other 1/2
        c2_c1_full = torch.cat([Cont_1, c2_c1_half])
        Expa_1 = self.Expand_1(c2_c1_full)
        logits = self.out(Expa_1)
        return logits
    
"""train()"""
def train(dataloader, model, loss_fn, optimizer):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

"""test()"""
def test(dataloader, model, loss_fn):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")