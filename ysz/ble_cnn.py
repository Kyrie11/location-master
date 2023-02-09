######description#######
#蓝牙数据格式
# ap1 ap2 ... apn x y classification timestamp
#将蓝牙强度低于阈值或者没有信号的部分记作100
#参照论文 https://ieeexplore.ieee.org/abstract/document/9237969
import os

import numpy
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)
EPOCH = 50
BATCH_SIZE = 5
LR = 0.001
DROP_RATE = 0.5



def get_rawdata(root_dir):

    file_list = os.listdir(root_dir)

    raw_data = None
    label_data = None

    for i in file_list:
        new_data = numpy.genfromtxt(root_dir + "" + i, dtype=str, delimiter=',')
        pd_data = pd.read_csv(root_dir+""+i)
        # new_label_data = new_data[1:, 28].astype(np.float64)
        new_label_data= pd_data['classification'].astype(np.float64)
        new_data = new_data[1:, 2:27].astype(np.float64)
        new_data = abs(new_data-100) #这一步主要是把蓝牙强度信号转换成灰度值范围

        if(label_data is None):
            label_data = new_label_data

        else:
            label_data = np.concatenate((label_data, new_label_data), axis=0)

        if (raw_data is None):
            raw_data = new_data
        else:
            raw_data = numpy.concatenate((raw_data, new_data), axis=0)
    cols = raw_data.shape[0]
    raw_data = np.resize(raw_data, (cols, 1, 5, 5))

    return raw_data, label_data

class GetLoader(Dataset):
    def __init__(self, raw_data, data_label):
        # self.data = torch.from_numpy(raw_data).float()
        self.data = raw_data
        self.label = data_label

    def __getitem__(self, item):
        data = self.data[item]
        labels = self.label[item]
        return data, labels

    def __len__(self):
        return len(self.data)

class BLE_CNN(nn.Module):
    def __init__(self):
        super(BLE_CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, #像素通道数
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding=1 #con2d出来的图片尺寸不变
                       ),#output shape(16,5,5)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1),#output shape(16,3,3)
        )
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=16,
                                             out_channels=32,
                                             kernel_size=2,
                                             stride=1,
                                             padding=0), #output shape:(32,2,2)
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=1, stride=1)) #output shape:(32,2,2)
        # self.conv3 = nn.Sequential(nn.Conv2d(in_chan))
        self.out = nn.Linear(32*2*2, 60)

    def forward(self, x):
        x = x.type(torch.DoubleTensor)

        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

def train_network(network, train_data):



    optimizer = torch.optim.Adam(network.parameters(), lr=LR)

    loss_func = nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        for i, (data, label) in enumerate(train_data):
            batch_data = Variable(data)
            batch_label = Variable(label)
            output = network(batch_data)

            loss = loss_func(output, batch_label.long())
            print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__=="__main__":
    root_dir = '../data/fusion/MyFingerprint/train/'
    raw_data, label_data = get_rawdata(root_dir)
    print(raw_data.shape)
    print(label_data.shape)
    # torch_data = GetLoader(raw_data, label_data)
    data_tensor = torch.from_numpy(raw_data).type(torch.LongTensor)
    label_tensor = torch.from_numpy(label_data)
    torch_dataset = TensorDataset(data_tensor, label_tensor)
    train_data = DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=2)
    network = BLE_CNN()
    network = network.double()
    train_network(network, train_data)

    root_dir = "../data/fusion/MyFingerprint/test/"
    raw_data, label_data = get_rawdata(root_dir)
    data_tensor = torch.from_numpy(raw_data).type(torch.LongTensor)
    label_tensor = torch.from_numpy(label_data)
    torch_dataset = TensorDataset(data_tensor, label_tensor)
    test_output = network(data_tensor)
    pred = torch.max(test_output,1)[1].data.numpy().squeeze()
    print(pred, 'prediction number')
    print(label_tensor)
    # for i, (data, label) in enumerate(train_data):
    #     print("第 {} 个Batch \n{}".format(i, data))






