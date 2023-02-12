######description#######
#蓝牙数据格式
# ap1 ap2 ... apn x y classification timestamp
#将蓝牙强度低于阈值或者没有信号的部分记作100
#参照论文 https://ieeexplore.ieee.org/abstract/document/9237969
import os

import numpy
import pandas
import pandas as pd
import torch
import torch.nn as nn
import torchvision.datasets
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as data

torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)
EPOCH = 1000
BATCH_SIZE = 20
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
        new_data = abs(new_data-100)/100.0 #这一步主要是把蓝牙强度信号转换成灰度值范围

        if(label_data is None):
            label_data = new_label_data

        else:
            label_data = np.concatenate((label_data, new_label_data), axis=0)

        if (raw_data is None):
            raw_data = new_data
        else:
            raw_data = numpy.concatenate((raw_data, new_data), axis=0)
    rows = raw_data.shape[0]
    raw_data = np.resize(raw_data, (rows, 1, 5, 5))

    return raw_data, label_data

class GetLoader(Dataset):
    def __init__(self, file_path):
        # self.data = torch.from_numpy(raw_data).float()
        self.rssi_data = pandas.read_csv(file_path)
        self.label_data = self.rssi_data['classification']
        print(self.label_data)
        self.rssi_data = self.rssi_data.iloc[0:, 1:26]

        self.rssi_data = torch.from_numpy(self.rssi_data.values)
        self.rssi_data = self.rssi_data.resize(self.rssi_data.shape[0],1,5,5)
        self.label_data = torch.from_numpy(self.label_data.values)



    def __getitem__(self, item):
        data = self.rssi_data[item]
        label = self.label_data[item]
        data = torch.tensor(data)
        # data = self.transforms(data)
        label = torch.tensor(label)
        return data, label

    def __len__(self):
        return self.rssi_data.shape[0]

class BLE_CNN(nn.Module):
    def __init__(self):
        super(BLE_CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,  # 像素通道数
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding=1  # con2d出来的图片尺寸不变
                      ),  # output shape(16,5,5)
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=3, stride=1),#output shape(16,3,3)
        )
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=16,
                                             out_channels=32,
                                             kernel_size=2,
                                             stride=1,
                                             padding=0),  # output shape:(32,4,4)
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=1, stride=1))  # output shape:(32,4,4)
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=32,
                                             out_channels=64,
                                             kernel_size=2,
                                             stride=1,
                                             padding=0),  # output shape:(64,3,3)
                                   nn.ReLU(),
                                   )
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=64,
                                             out_channels=64,
                                             kernel_size=2,
                                             padding=1,
                                             stride=1), #output shape:(64,4,4)
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=1)) #output shape:(64,2,2)

        self.out = nn.Linear(64 * 2 * 2, 60)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.type(torch.DoubleTensor)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x = x.view(x.size(0), -1)
        # output = self.out(x)
        # return output
        x = torch.flatten(x, 1)
        logits = self.out(x)
        prob = self.softmax(logits)
        return logits, prob
def train_network(network, train_data, device):



    optimizer = torch.optim.Adam(network.parameters(), lr=LR)

    loss_func = nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        for i, (data, label) in enumerate(train_data):
            batch_data = Variable(data)
            batch_data = batch_data.to(device)
            batch_label = Variable(label)
            batch_label = batch_label.to(device)
            train_logits, train_prob = network(batch_data)
            loss = loss_func.forward(train_logits, batch_label)
            # loss = loss_func(output, batch_label.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_pred = torch.argmax(train_prob, dim=1)
            train_acc = (train_pred==batch_label).float()
            train_acc = torch.mean(train_acc)
            print("loss:",loss.item(), 'acc:',train_acc.item())


if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_dir = '../data/fusion/MyFingerprint/train/'
    raw_data, label_data = get_rawdata(root_dir)
    print(raw_data.shape)
    print(label_data.shape)
    # torch_data = GetLoader(raw_data, label_data)
    data_tensor = torch.from_numpy(raw_data).type(torch.LongTensor)
    label_tensor = torch.from_numpy(label_data).type(torch.LongTensor)
    torch_dataset = GetLoader('all.csv')
    train_data = DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=2)
    network = BLE_CNN()
    network.to(device)
    network = network.double()
    train_network(network, train_data, device)

    root_dir = "../data/fusion/MyFingerprint/test/"
    raw_data, label_data = get_rawdata(root_dir)
    data_tensor = torch.from_numpy(raw_data).type(torch.LongTensor)
    label_tensor = torch.from_numpy(label_data)
    torch_dataset = TensorDataset(data_tensor, label_tensor)
    test_output = network(data_tensor)
    pred = torch.max(test_output,1)[1].data.numpy().squeeze()
    print(pred, 'prediction number')
    print(label_tensor)
    count = 0
    for i in range(0, len(pred)):
        if(pred[i]==label_tensor[i]):
            count+=1
    print("准确率为:{}".format(count/len(label_tensor)))
