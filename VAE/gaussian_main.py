from pickle import TRUE
import torch
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from dataset import *
from gaussian_vae import *
from gaussian_learner import *
import time
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#training_detail
learning_rate = 0.01
scaling = 255.
batch_size = 64
epochs = 0
n_feats = 64
train = True
pretraied = True
start_time = time.time()
patch_size = 64

model = VAE().to(device)
if pretraied==True:
    model.load_state_dict(torch.load('gaussian_model.pth'))
test_data = Test_set()
test_dataloader= DataLoader(test_data, batch_size=1)

training_data = Train_set()
train_dataloader = DataLoader(training_data, batch_size=batch_size,shuffle=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                             betas=[0.9,0.96],eps=1e-8)

count,maximum = 0, 0
for t in range(epochs):
    print(f"Epoch {t+1} Learning Rate {learning_rate:7f}\n-------------------------------")
    train_loop(train_dataloader,model,optimizer,scaling,type)
    torch.save(model.state_dict(), "gaussian_model.pth")
    result = test(test_dataloader, model,scaling=scaling)
    if maximum < result:
        maximum = result
    else:
        count += 1
    if (count+1) % 10 == 0:
        learning_rate /= 10
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                             betas=[0.9,0.96],eps=1e-8)
        
test(test_dataloader, model,scaling=scaling)
    
print(f"Done! time : {time.time()-start_time:5f}")