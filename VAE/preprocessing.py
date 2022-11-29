import h5py
import numpy as np
import cv2
import os
import time

start_time = time.time()
data_path = 'img_align_celeba/'
percentage = 0.7

data_list = os.listdir(data_path)
data_length = len(data_list)
train_length = int(data_length*percentage)
test_length = data_length-train_length
count = 0
TR_D,TT_D =[],[]
for nm in data_list:
    img = cv2.imread(data_path+nm)/255.
    img = cv2.resize(img,dsize=(64,64),interpolation=cv2.INTER_CUBIC )
    img = np.transpose(img,(2,0,1))
    if count < train_length:
        TR_D.append(img)
    else:
        TT_D.append(img)
    count += 1
TR_D,TT_D = np.array(TR_D),np.array(TT_D)
with h5py.File('DATA_SET.hdf5', 'w') as f:
    f.create_dataset("TRAIN_SET",data=TR_D)
    f.create_dataset("TEST_SET",data=TT_D)
print(start_time-time.time())