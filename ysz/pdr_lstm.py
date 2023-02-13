#输入 加速度、角速度
#输出 移动步长
#参考论文 https://ceur-ws.org/Vol-2498/short60.pdf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy
import random

torch.manual_seed(200)
random.seed(200)

'''
    Neural Network: CNN_LSTM
'''

class CNN_LSTM(nn.Module):

    def __init__(self, args):
        super(CNN_LSTM, self).__init__()
        self.args = args
        self.hidden_dim = args.lstm_hidden_dim
        self.num_layers = args.lstm_num_layers
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes
        self.C = C
        self.embed = nn.Embedding(V, D, padding_idx=args.paddingId)
        if(args.word_Embedding):
            self.embed.weight.data.copy_(args.pretrained_weight)

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=16,
                      kernel_size=12,
                      stride=1,
                      padding=0),
            nn.Dropout(p=0.5),
            nn.ReLU()
        )

        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=12,
                      stride=1,
                      padding=0),
            nn.Dropout(p=0.5),
            nn.ReLU()
        )

        self.conv1_3 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=12,
                      stride=1,
                      padding=0),
            nn.Dropout(p=0.5),
            nn.ReLU()
        )

        self.conv2_2 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=16,
                      kernel_size=24,
                      stride=1,
                      padding=0),
            nn.Dropout(p=0.5),
            nn.ReLU()
        )

        self.conv2_3 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=24,
                      stride=1,
                      padding=0),
            nn.Dropout(p=0.5),
            nn.ReLU()
        )
        if(self.args.cuda is True):
            for conv in self.convs1:
                conv = conv.cuda()


        self.lstm = nn.LSTM(16, 64, dropout=args.dropout, num_layers=1)

        L = len(Ks) * Co + self.hidden_dim
        self.hidden2label1 = nn.Linear(L, L // 2)
        self.hidden2label2 = nn.Linear(L // 2, C)

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=64,
                      out_features=128),
            nn.BatchNorm1d(),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=128,
                      out_features=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.convs(x)
        x1 = self.conv1_2(x)
        x1 = self.conv1_3(x1)
        x2 = self.conv2_2(x)
        x2 = self.conv2_3(x2)
        x = torch.cat([x1, x2], 0)
        x = self.lstm(x)
        x = self.fc1(x)
        x = self.fc2(x)
