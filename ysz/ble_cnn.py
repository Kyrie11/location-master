######description#######
#蓝牙数据格式
# ap1 ap2 ... apn x y classification timestamp
#将蓝牙强度低于阈值或者没有信号的部分记作100
import os

import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

EPOCH = 30
BATCH_SIZE = 30
LR = 0.001
DROP_RATE = 0.5

root_dir = '../data/fusion/MyFingerprint/'
file_list = os.listdir(root_dir)
data = 1
for i in file_list:
    file = open(root_dir+""+i)
    new_data = numpy.genfromtxt(root_dir+""+i,dtype=str,delimiter=',')
    new_data = new_data[1:,2:27]
    if(data == 1):
        data=new_data
    else:
        data = numpy.concatenate((data, new_data), axis=0)
print(data.shape)
print(data)
label_ble = 'classification'


# transform = transforms.Compose([transforms.Resize(12,12), transforms.ToTensor()])


