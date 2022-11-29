from importlib.resources import path
import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import math
import h5py

class Test_set(Dataset):
    def __init__(self,scaling=1.):
        hf = h5py.File("DATA_SET.hdf5",'r')
        self.data = hf.get("TEST_SET")
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        return torch.from_numpy(self.data[index,:,:,:]).float()
    
    
class Train_set(Dataset):
    def __init__(self,scaling=1.):
        hf = h5py.File("DATA_SET.hdf5",'r')
        self.data = hf.get("TRAIN_SET")
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        return torch.from_numpy(self.data[index,:,:,:]).float()
   
def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    
    if(mse == 0):
        return 100
    max_pixel = 1.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    
    return psnr 

if __name__=='__main__':
    a = Test_set()
    for i in range(5):
        b = a.__getitem__(i)
        b = np.array(b).transpose((1,2,0))
        cv2.imwrite(f'test_storage/img_{i}.png',255*b)