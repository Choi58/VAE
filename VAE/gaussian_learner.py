import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import *
import cv2
import torch.nn.functional as F
import math
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def _PSNR(original, compressed,scaling):
    original = tensor2np2y(original)
    compressed = tensor2np2y(compressed)
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):
        return 100 
    max_pixel = 255.0/scaling
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr  
 
def tensor2np2y(tensor):
    img = np.array(tensor.detach().to('cpu')).transpose((0,2,3,1))       
    return img

mse = nn.MSELoss()
kl_loss = nn.KLDivLoss()
bce_loss = nn.BCELoss()
def train_loop(dataloader, model, optimizer,scaling,type='gauss'):
    size = len(dataloader.dataset)
    for batch, X in enumerate(dataloader):
        X = X.to(device)
        # 예측 오류 계산
        [appro_x,x,mu,sigma] = model(X)
        reconst_error = mse(appro_x,X)
        regularization = torch.mean(-0.5*torch.sum(1+sigma-mu**2-
                                    sigma.exp(),dim=1),dim=0)
        loss = reconst_error + 0.00024*regularization
        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 500 == 0:
            psnr = _PSNR(x,appro_x,scaling)
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} psnr:{psnr:>7f} [{current:>5d}/{size:>5d}]")

def test(dataloader, model,scaling):
    size = len(dataloader.dataset)
    model.eval()
    with torch.no_grad():
        likeli_mean,count = 1,0
        for idx, U in enumerate(dataloader):
            X = U.to(device)
            [appro_x,x,mu,sigma] = model(X)
            reconst_error = -torch.log(torch.tensor(0.4))+mse(appro_x,X)
            regularization = torch.mean(-0.5*torch.sum(1+sigma-mu**2-
                                    sigma.exp(),dim=1),dim=0)
            ELBO = reconst_error + 0.00024* np.array(regularization.detach().to('cpu'))
            
            print(f"ELBO : {ELBO:>7f}")
            count += 1
            if count < 11:
                img_save(appro_x,idx,'pred',scaling)
                img_save(X,idx,'original',scaling=scaling)
            if count < 11:
                return likeli_mean/10
            if count < 11:
                break
            likeli_mean += ELBO
    print(likeli_mean/size)
            
            
def img_save(pred,idx,nm,scaling):
    pred = torch.clamp(scaling*pred,min=0,max=255)
    y = np.array(pred.detach().to('cpu'))
    y = y.squeeze()
    y = np.transpose(y,(1,2,0))
    cv2.imwrite(f'test_storage/{nm}_{idx}_pred.png',y)
