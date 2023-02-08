import os
import numpy as np
import pandas as pd
import torch
import torchvision
import torch.utils.data as Data
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# def NN_test():


def load_data(data_dir='./data/fusion/MyFingerprint'):
    filelist = os.listdir(data_dir)
    cols = ['RFstar_51DE',	'RFstar_040D',	'RFstar_1B7B',	'RFstar_6891',	'RFstar_EFBC',	'RFstar_2037',
    	    'RFstar_0259',	'RFstar_4EE6',	'RFstar_9E56',	'RFstar_B613',		'RFstar_B571',
        	'RFstar_8FBE',	'RFstar_78B2',	'RFstar_FA6E',	'RFstar_02C5',	'RFstar_D4E4',	'RFstar_FB47',
            'RFstar_54CE',	'RFstar_3D4D',	'RFstar_6F5F',	'RFstar_D8FC',	'RFstar_9F01',	'RFstar_30C2',
            'RFstar_3D81',	'RFstar_42D5']
    """'RFstar_9120',"""

    cols = ['RFstar_51DE','RFstar_040D','RFstar_1B7B','RFstar_6891','RFstar_EFBC','RFstar_2037','RFstar_0259',
            'RFstar_4EE6','RFstar_9E56','RFstar_B613','RFstar_B571','RFstar_8FBE','RFstar_78B2',
            'RFstar_FA6E','RFstar_02C5','RFstar_D4E4','RFstar_FB47','RFstar_54CE','RFstar_3D4D','RFstar_6F5F',
            'RFstar_D8FC','RFstar_9F01','RFstar_30C2','RFstar_3D81','RFstar_42D5']
    label = 0
    label2filename = {}
    x = np.zeros((1, len(cols)))
    y = np.zeros((1, 1))
    error_file_list = []
    for i, file in enumerate(filelist):
        try:
            filepath = os.path.join(data_dir, file)
            print(filepath)
            df = pd.read_csv(filepath, index_col=0)
            # if 'RFstar_9120' not in df.columns.tolist():
            #     df.insert(12, 'RFstar_9120', 100)
            # print(df.head())
            df.replace(100, -100, inplace=True)
            df.fillna(-100, inplace=True)
            label2filename[label] = file



            # _x =(df.loc[:, cols].values+60)/40
            _x = df.loc[:, cols].values+80
            _y = np.zeros((1, len(_x))) + label
            # _y = np.zeros((1, len(_x))) + [xx,yy]

            x = np.append(x, _x, axis=0)
            y = np.append(y, _y)

            print(i, filepath, _x.shape, _y.shape, x.shape, y.shape)

            label += 1
        except Exception as e:
            error_file_list.append(file)
            print('error in file:', filepath)
            print(e)
        # if i > 5: break
    if len(error_file_list):
        print('error files:', error_file_list)

    x = x[1:, :]
    y = y[1:]
    # print(x)
    print('x shape:', x.shape, 'y shape:',  y.shape, 'label count:', label)

    x_train, x_test,  y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 7)
    x_train = torch.tensor(x_train, dtype=torch.float)
    x_test = torch.tensor(x_test, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    # x = torch.normal(x)
    # print(x)

    torch_dataset = Data.TensorDataset(x_train, y_train)
    loader_train = Data.DataLoader(
        # 从数据库中每次抽出batch size个样本
        dataset=torch_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=2,
    )

    torch_datasettest = Data.TensorDataset(x_test, y_test)
    loader_test = Data.DataLoader(
        # 从数据库中每次抽出batch size个样本
        dataset=torch_datasettest,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=2,
    )

    return x_train, x_test,  y_train, y_test, loader_train,loader_test, label

class Net(nn.Module):
    def __init__(self, in_feas=26, out_feas=100):
        super(Net, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_feas, 64),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.out = nn.Sequential(
            nn.Linear(32, out_feas),
            # nn.Softmax(dim=1),
        )
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        return x


def train(network, train_loader, optimizer, epochs=10, loss_func=nn.CrossEntropyLoss()):
    print('training')
    train_losses = []
    network.train()
    # SummaryWriter压缩（包括）了所有内容
    writer = SummaryWriter('runs/train-1')
    # 创建 writer object，log会被存入'runs/train-1'
  
    for epoch in range(epochs):
        loss_list = []
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = network(data.to(device))
            # print(output, target)
            # print(output.sum(), len(target))
            loss = loss_func(output, target.to(device))
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        loss = np.mean(loss_list)
        train_losses.append(loss)
        writer.add_scalar("name",loss,epoch)
        if epoch % log_interval == 0:
            print('epoch:{:0>5d}/{}, loss:{:.3f}'.format(epoch, epochs, loss))
        if epoch % logtest_interval == 0:
            test(network,testloader)
    
    torch.save(network.state_dict(), './model.pth')
    torch.save(optimizer.state_dict(), './optimizer.pth')
    writer.close()



test_losses=[]
def test(network,test_loader,loss_func=nn.CrossEntropyLoss()):
    print('testing')
    network.eval()
    test_loss = 0
    correct = 0
    # writer = SummaryWriter('runs/test-1')
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = network(data)
            # test_loss +=loss_func(output, target, size_average=False).item()
            test_loss +=loss_func(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def testnext(network, test_loader, loss_func=nn.CrossEntropyLoss()):
    print('testing')
    network.eval()
    test_loss = 0
    correct = 0
    # writer = SummaryWriter('runs/test-1')
    with torch.no_grad():
        for data in test_loader:
           #  print(data[0].shape)
            # print(len(data))
            # data = torch.tensor(data[:])
            output = network(data[0])
            # print('output.shape', output.shape)
            # test_loss +=loss_func(output, target, size_average=False).item()
            # test_loss +=loss_func(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            # correct += pred.eq(target.data.view_as(pred)).sum()
    # test_loss /= len(test_loader.dataset)
    # test_losses.append(test_loss)
    # print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))
    # writer.close()
    return pred


if __name__ == '__main__':

    # print(torch.__version__)  # 查看torch当前版本号
    # print(torch.version.cuda)  # 编译当前版本的torch使用的cuda版本号
    # print(torch.cuda.is_available())  # 查看当前cuda是否可用于当前版本的Torch，如果输出True，则表示可用

    data_dir='./data/fusion/MyFingerprint'
    learning_rate = 1e-3
    momentum = 0.5
    batch_size_train = 256
    batch_size_test = 1000
    log_interval = 10
    logtest_interval = 100
    random_seed = 1
    torch.manual_seed(random_seed)
    # path = os.path.abspath(os.path.join(os.getcwd(), "./data"))
    # fingerprint_path = path + '/fusion/MyFingerprint'
    x_train, x_test,  y_train, y_test, trainloader,testloader, label_count=load_data(data_dir)
    #x, y, loader, label_count = load_data(data_dir)
    network = Net(in_feas=x_train.shape[1], out_feas=label_count).to(device)
    print(next(network.parameters()).device)

    # print(network)
    optimizer = optim.SGD(network.parameters(), lr=learning_rate)
    train(network, trainloader, optimizer, epochs=2000)
    #test(network,testloader)

   
   










































































































































































































































































































































































































