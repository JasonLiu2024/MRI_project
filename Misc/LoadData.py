import torch
import numpy as np
import h5py
from torchvision.transforms import ToTensor
# Reference: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
# (?) 1 complex channel -> 1 real channel + 1 imaginary channel? (before converting to Tensor)
"""partition: dictionary of {set of IDs} for train, partition, validation"""
"""labels: dictionary, contain index of labels y, given index of input X"""
"""Dataset class, derived from: PyTorch's Dataset"""
def Load_h5py(DATA):
    # load data
    with h5py.File(DATA, 'r') as hdf:
        keys = list(hdf.keys())
        trnOrg = hdf.get('trnOrg')
        x_GroundTruth = np.array(trnOrg) # <- this is our y
        trnMask = hdf.get('trnMask')
        P_SamplingOperator = np.array(trnMask)
    # Formula: y = P*Fourier(x) + e <- e = 0 here!
    y_NoisyMeasurement = P_SamplingOperator * np.fft.fft2(x_GroundTruth) + 0
    # Formula: x_hat = InverseFourier(y)
    x_hat_ZeroFilledImg = np.fft.ifft2(y_NoisyMeasurement) # <- this is our X
    return x_hat_ZeroFilledImg, x_GroundTruth

class h5py_Dataset(torch.utils.data.Dataset):
    # initialize:
    def __init__(self, subset_X_IDs, y_IDs, DATA):
        # subset_X_IDs: e.g. only use mth to nth examples to train
        self.subset_X_IDs = subset_X_IDs
        self.y_IDs = y_IDs
        self.X, self.y = Load_h5py(DATA=DATA)
    # length: number of elements in subset of X_IDs
    def __len__(self):
        return len(self.subset_X_IDs)
    # get item by index in [m, n]
    def __getitem__(self, index):
        ID = self.subset_X_IDs[index]
        # pick out X & y
        X_pick = self.X[ID]
        y_pick = self.y[ID]
        return Complex_to_2_Channel(X_pick), Complex_to_2_Channel(y_pick)
    
def Complex_to_2_Channel(image):
    # image = 2d numpy array, shape = (m, n) <- 1 channel
    real = image.real
    imag = image.imag
    comb = np.dstack([real, imag]) # <- shape = (m, n, 2) <- 2 channel
    comb_t = ToTensor(comb) # <- shape (2, m, n) <- 2 channel, reformat to Tensor
    return comb_t