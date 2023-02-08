'''
1.Model参数列表（1个参数）：
WIFI信号强度矩阵

2.Model参数类型：
numpy.ndarray
'''

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import ensemble
import matplotlib.ticker as ticker
from scipy.stats import norm
from sklearn import neighbors, svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import model_selection
import torch
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import torch.utils.data as Data

from network import testnext, Net

class Model(object):
    def __init__(self, rssi):
        self.rssi = rssi

    def create_fingerprint(self, path):
        RSSI = None
        X = None
        Y = None
        # path为采集的指纹数据目录，每个坐标为一个文件，文件的命名格式为：x_y
        directory = os.walk(path)  
        for _, _, file_list in directory:
            for file_name in file_list:
                # position = file_name.split('.')[0].split('-') # 获取文件记录的坐标
                # x = np.array([[int(position[0])]])
                # y = np.array([[int(position[1])]])
                position = file_name.split('.')[0].split('_')  # 获取文件记录的坐标
                x = np.array([[int(position[2])]])
                y = np.array([[int(position[1])]])
                # x = (x + 50) / 65
                # y = y / 40
                df = pd.read_csv(path + "/" + file_name)

                # columns = [col for col in df.columns if 'rssi' in col]
                if 'RFstar_9120' not in df.columns.tolist():
                    df.insert(12,'RFstar_9120',100)
                # columns = [col for col in df.columns if 'RFstar' in col]
                columns=['RFstar_51DE','RFstar_040D','RFstar_1B7B','RFstar_6891','RFstar_EFBC','RFstar_2037','RFstar_0259','RFstar_4EE6','RFstar_9E56',
                         'RFstar_B613','RFstar_B571','RFstar_8FBE','RFstar_78B2','RFstar_FA6E','RFstar_02C5','RFstar_D4E4','RFstar_FB47',
                         'RFstar_54CE','RFstar_3D4D','RFstar_6F5F','RFstar_D8FC','RFstar_9F01','RFstar_30C2','RFstar_3D81','RFstar_42D5']
                ''''RFstar_9120','''

                # rssi = df[columns].values
                # rssi = rssi[300:] # 视前300行数据为无效数据

                # rssi_mean = np.mean(rssi, axis=0).reshape(1, rssi.shape[1])# 求每一列的平均

                # print(len(columns),y,x)

                # rssi_mean = np.zeros((1, len(columns)))
                rssi_mean = np.full((1, len(columns)),-100)
                for i, col in enumerate(columns):
                    mask = df[col] != 100
                    if len(df.loc[mask, col]) > 0:
                        mean = df.loc[mask, col].mean()
                        rssi_mean[0,i] = mean
                # rssi_mean = np.array(rssi_m).reshape(1,len(rssi_m))
                rssi_mean = (rssi_mean+60)/40

                if RSSI is None:
                    RSSI = rssi_mean
                    X = x
                    Y = y
                else:
                    RSSI = np.concatenate((RSSI,rssi_mean), axis=0)#axis=0纵向拼接
                    X = np.concatenate((X,x), axis=0)
                    Y = np.concatenate((Y,y), axis=0)
        # fingerprint = np.concatenate((RSSI, X, Y), axis=1)#axis=1横向拼接
        # fingerprint = pd.DataFrame(fingerprint, index=None, columns = columns+['x', 'y'])
        # # rssi = fingerprint[[col for col in fingerprint.columns if 'rssi' in col]].values
        # rssi = fingerprint[[col for col in fingerprint.columns if 'RFstar' in col]].values
        #
        # position = fingerprint[['x', 'y']].values
        # print(rssi.shape)
        # print(position)

        fingerprint = np.concatenate(( X, Y), axis=1)  # axis=1横向拼接
        return RSSI, fingerprint

    # 标准差
    def square_accuracy(self, predictions, labels):
        accuracy = np.sqrt(np.mean(np.sum((predictions - labels)**2, 1)))
        return round(accuracy, 3)

    def ml_limited_reg(self, type, offline_rss, offline_location, online_rss, online_location):
        if type == 'knn':
            k = 3
            ml_reg = neighbors.KNeighborsRegressor(n_neighbors=k, weights='uniform', metric='euclidean')
        elif type == 'rf':
            ml_reg = RandomForestRegressor(n_estimators=10)

        init_x = 0
        init_y = 0
        predict = np.array([[init_x, init_y]])
        limited_rss = None
        limited_location = None
        offset = 2 # m

        for k, v in enumerate(online_rss):
            if k == 0:
                continue
            for v1, v2 in zip(offline_rss, offline_location):
                if (v2[0] >= init_x-offset and v2[0] <= init_x+offset) and (v2[1] >= init_y-offset and v2[1] <= init_y+offset):
                    v1 = v1.reshape(1, v1.size)
                    v2 = v2.reshape(1, v2.size)
                    if limited_rss is None:
                        limited_rss = v1
                        limited_location = v2
                    else:
                        limited_rss = np.concatenate((limited_rss, v1), axis=0)
                        limited_location = np.concatenate((limited_location, v2), axis=0)
            v = v.reshape(1, v.size)
            predict_point = ml_reg.fit(limited_rss, limited_location).predict(v)
            predict = np.concatenate((predict, predict_point), axis=0)
            init_x = predict_point[0][0]
            init_y = predict_point[0][1]
            limited_rss = None
            limited_location = None
        
        accuracy = self.square_accuracy(predict, online_location)
        return predict, accuracy

    # 选取信号最强的num个rssi作为匹配
    def wknn_strong_signal_reg(self, offline_rss, offline_location, online_rss, online_location):
        num = 8
        k = 3
        rssi_length = offline_rss.shape[1]
        knn_reg = neighbors.KNeighborsRegressor(n_neighbors=k, weights='distance', metric='euclidean')

        limited_location = None

        for rssi in online_rss:
            keys = np.argsort(rssi)[(rssi_length - num):]
            # keys = np.argsort(rssi)[:num]
            rssi = rssi.reshape(1, rssi_length)
            limited_online_rssi = rssi[:,keys] # from small to big
            limited_offline_rssi = offline_rss[:,keys]
            predict_point = knn_reg.fit(limited_offline_rssi, offline_location).predict(limited_online_rssi)
            if limited_location is None:
                limited_location = predict_point
            else:
                limited_location = np.concatenate((limited_location, predict_point), axis=0)

        predict = limited_location
        accuracy = self.square_accuracy(predict, online_location)
        return predict, accuracy

    # knn regression
    def knn_reg(self, offline_rss, offline_location, online_rss, online_location):
        k = 3
        knn_reg = neighbors.KNeighborsRegressor(n_neighbors=k, weights='uniform', metric='euclidean')
        predict = knn_reg.fit(offline_rss, offline_location).predict(online_rss)
        print(predict.shape)
        accuracy = self.square_accuracy(predict, online_location)
        return predict, accuracy
    
    # wknn regression
    def wknn_reg(self, offline_rss, offline_location, online_rss, online_location):
        k = 2
        wknn_reg = neighbors.KNeighborsRegressor(n_neighbors=k, weights='distance', metric='euclidean')
        predict = wknn_reg.fit(offline_rss, offline_location).predict(online_rss)
        accuracy = self.square_accuracy(predict, online_location)
        return predict, accuracy
    
    # 支持向量机
    def svm_reg(self, offline_rss, offline_location, online_rss, online_location):
        # kernel = ['rbf', 'linear', 'poly', 'sigmoid']
        # C = [0.1, 1,25,50,100, 200,300,400,500,600,700,800,1000]
        # parameters = {'kernel': kernel, 'C': C}
        # grid_svc_x = model_selection.GridSearchCV(estimator=svm.SVC(max_iter=10000), param_grid=parameters,
        #                                         scoring='accuracy', cv=5, verbose=1)
        # grid_svc_y = model_selection.GridSearchCV(estimator=svm.SVC(max_iter=10000), param_grid=parameters,
        #                                           scoring='accuracy', cv=5, verbose=1)
        # # 模型在训练数据集上的拟合
        # grid_svc_x.fit(offline_rss, offline_location[:, 0])
        # grid_svc_y.fit(offline_rss, offline_location[:, 1])
        # # 返回交叉验证后的最佳参数值
        # print(grid_svc_x.best_params_, grid_svc_x.best_score_)
        # print(grid_svc_y.best_params_, grid_svc_y.best_score_)


        clf_x = svm.SVR(C=100, gamma=0.01)
        clf_y = svm.SVR(C=100, gamma=0.01)
        clf_x = svm.SVR(kernel='poly', C=50, gamma=0.01)
        clf_y = svm.SVR(kernel='poly', C=50, gamma=0.01)
        clf_x.fit(offline_rss, offline_location[:, 0])
        clf_y.fit(offline_rss, offline_location[:, 1])
        # print(clf_x.best_score_)
        # print(clf_y.best_score_)
        x = clf_x.predict(online_rss)
        y = clf_y.predict(online_rss)
        predict = np.column_stack((x, y))
        accuracy = self.square_accuracy(predict, online_location)
        return predict, accuracy

    def nuralnetwork(self, online_rss_file_path, online_location,fingerprint_file_path):
        cols = ['RFstar_51DE','RFstar_040D','RFstar_1B7B','RFstar_6891','RFstar_EFBC','RFstar_2037','RFstar_0259','RFstar_4EE6',
               'RFstar_9E56','RFstar_B613','RFstar_B571','RFstar_8FBE','RFstar_78B2','RFstar_FA6E','RFstar_02C5',
               'RFstar_D4E4','RFstar_FB47','RFstar_54CE','RFstar_3D4D','RFstar_6F5F','RFstar_D8FC','RFstar_9F01','RFstar_30C2',
               'RFstar_3D81','RFstar_42D5']
        ''''RFstar_9120', '''
        df = pd.read_csv(online_rss_file_path)
        x = df.loc[:, cols].values
        # x = (x+60)/40
        x = x + 80

        XX=None
        YY=None
        # print(x.shape)

        ##################################
        # for i in range(x.shape[0]):
        #     x_test = torch.tensor(x[i, ...], dtype=torch.float)
        #     print(x_test.shape)
        #     print(x_test)
        #     # print(x_test.type)
        #     torch_datasettest = Data.TensorDataset(x_test)
        #     loader_test = Data.DataLoader(
        #         # 从数据库中每次抽出batch size个样本
        #         dataset=torch_datasettest,
        #         batch_size=26,
        #         shuffle=False,
        #         num_workers=2,
        #     )
        #     model = Net(in_feas=x.shape[1], out_feas=234)
        #     # model.load_state_dict(torch.load('./model.pth'))
        #     model.eval()
        #     pred = test(model, loader_test)
        ##################################

        x_test = torch.tensor(x, dtype=torch.float)
        torch_datasettest = Data.TensorDataset(x_test)
        loader_test = Data.DataLoader(
            # 从数据库中每次抽出batch size个样本
            dataset=torch_datasettest,
            batch_size=55,
            shuffle=True,
            # num_workers=2,
        )
        model = Net(in_feas=x.shape[1], out_feas=234)
        # model.load_state_dict(torch.load('./model.pth'))
        model.eval()
        # predict=np.zeros((1,1))

        pred = testnext(model, loader_test).cpu().numpy()  # 一共55个数据
        for cls in pred:
            idx = int(cls)
            position = os.listdir(fingerprint_file_path)[idx].split('.')[0].split('_')  # 获取文件记录的坐标
            xx = np.array([[int(position[2])]])
            yy = np.array([[int(position[1])]])
            if XX is None and YY is None:
                XX=xx
                YY=yy
            else:
                XX= np.concatenate((XX, xx), axis=0)
                YY = np.concatenate((YY, yy), axis=0)
            # predict_ = np.column_stack((x, y))
            # predict = np.append(predict,predict_)

        predict = np.concatenate((XX, YY), axis=1)
        # predict=predict[1:]
        # position = file_name.split('.')[0].split('_')  # 获取文件记录的坐标
        # x = np.array([[int(position[2])]])
        # y = np.array([[int(position[1])]])
        # predict = np.column_stack((x, y))
        accuracy = self.square_accuracy(predict, online_location)
        return predict, accuracy

    # 随机森林
    def rf_reg(self, offline_rss, offline_location, online_rss, online_location):
        estimator = RandomForestRegressor(n_estimators=150)
        estimator.fit(offline_rss, offline_location)
        predict = estimator.predict(online_rss)
        accuracy = self.square_accuracy(predict, online_location)
        return predict, accuracy

    # 梯度提升
    def dbdt(self, offline_rss, offline_location, online_rss, online_location):
        clf = MultiOutputRegressor(ensemble.GradientBoostingRegressor(n_estimators=100, max_depth=10))
        clf.fit(offline_rss, offline_location)
        predict = clf.predict(online_rss)
        accuracy = self.square_accuracy(predict, online_location)
        return predict, accuracy
    
    # 多层感知机
    def nn(self, offline_rss, offline_location, online_rss, online_location):
        clf = MLPRegressor(hidden_layer_sizes=(100, 100))
        clf.fit(offline_rss, offline_location)
        predict = clf.predict(online_rss)
        accuracy = self.square_accuracy(predict, online_location)
        return predict, accuracy

    '''
        data为np.array类型
    '''
    def determineGaussian(self, data, merge, interval=1, wipeRange=300):
        offset = wipeRange
        data = data[offset:]

        minValue = np.min(data)
        maxValue = np.max(data)
        meanValue = np.mean(data)

        length = math.ceil((maxValue-minValue)/interval)
        counterArr = length * [0]
        valueRange = length * [0]

        textstr = '\n'.join((
                r'$max=%.2f$' % (maxValue, ),
                r'$min=%.2f$' % (minValue, ),
                r'$mean=%.2f$' % (meanValue, )))

        if merge==True:
            # 区间分段样本点
            result = []
            temp_data = data[0]
            for i in range(0, len(data)):
                if temp_data == data[i]:
                    continue
                else:
                    result.append(temp_data)
                    temp_data = data[i]
            data = result

        for index in range(len(counterArr)):
            valueRange[index] = minValue + interval*index

        for value in data:
            key = int((value - minValue) / interval)
            if key >=0 and key <length:
                counterArr[key] += 1
        
        if merge==True:
            print('Wi-Fi Scan Times:', len(data))

        probability = np.array(counterArr) / np.sum(counterArr)
        normal_mean = np.mean(data)
        normal_sigma = np.std(data)
        normal_x = np.linspace(minValue, maxValue, 100)
        normal_y = norm.pdf(normal_x, normal_mean, normal_sigma)
        normal_y = normal_y * np.max(probability) / np.max(normal_y)

        _, ax = plt.subplots()

        # Be sure to only pick integer tick locations.
        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_major_locator(ticker.MaxNLocator(integer=True))

        ax.bar(valueRange, probability, label='distribution')
        ax.plot(normal_x, normal_y, 'r-', label='fitting')
        plt.xlabel('rssi value')
        plt.ylabel('probability')
        plt.title('信号强度数据的高斯拟合')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
        plt.legend()
        plt.show()
    
    def rssi_fluctuation(self, merge, wipeRange=300):
        # wipeRange=300表示前300行数据中包含了无效数据，可以直接去除
        offset = wipeRange
        rssi = self.rssi[offset:]
        rows = rssi.shape[0]
        columns = rssi.shape[1]
        lines = [0]*(columns+1)
        labels = [0]*(columns+1)

        filename = ''

        if merge == False:
            for i in range(0, columns):
                lines[i], = plt.plot(range(0, rows), rssi[:, i])
                labels[i] = 'rssi' + str(i+1)
            
            plt.title('指纹库文件'+filename+'数据波动情况')
            plt.legend(handles=lines, labels=labels, loc='best')
            plt.xlabel('样本点数目/个')
            plt.ylabel('WiFi信号强度/dBm')
            plt.show()

        elif merge == True:
            # 采集周期（以一定样本点数目为一个周期）
            indexs = []
            results = []
            
            # 区间分段样本点
            for i in range(0, columns):
                counter = 0
                intervals = []
                result = []
                temp_rssi = rssi[0, i]
                for j in range(0, rows):
                    if temp_rssi == rssi[j, i]:
                        counter = counter +1
                    else:
                        intervals.append(counter)
                        result.append(temp_rssi)
                        temp_rssi = rssi[j, i]
                indexs.append(intervals)
                results.append(result)
                intervals = []
            
            # 确定最小长度
            length = 0
            for i in range(0, columns):
                if length==0:
                    length = len(results[i])
                else:
                    if len(results[i]) < length:
                        length = len(results[i])
            
            # 显示图像
            for i in range(0, columns):
                lines[i], = plt.plot(range(0, length), results[i][:length])
                labels[i] = 'rssi' + str(i+1)
            
            plt.title('指纹库文件'+filename+'数据波动情况')
            plt.legend(handles=lines, labels=labels, loc='best')
            plt.xlabel('WiFi扫描次数/次')
            plt.ylabel('WiFi信号强度/dBm')
            plt.xticks(range(0, length, int(length/5))) # 保证刻度为整数
            plt.show()

    # 显示运动轨迹图
    def show_trace(self, predict_trace, **kw):
        plt.grid()
        handles = []
        labels = []
        if 'real_trace' in kw:
            real_trace = kw['real_trace'].T
            trace_x = real_trace[0]
            trace_y = real_trace[1]
            l1, = plt.plot(trace_x, trace_y, 'o-')
            handles.append(l1)
            labels.append('real tracks')
            for k in range(0, len(trace_x)):
                plt.annotate(k, xy=(trace_x[k], trace_y[k]), xytext=(trace_x[k]+0.1,trace_y[k]+0.1), color='green')

        predict = predict_trace.T
        x = predict[0]
        y = predict[1]

        for k in range(0, len(x)):
            plt.annotate(k, xy=(x[k], y[k]), xytext=(x[k]+0.1,y[k]+0.1))
        
        l2, = plt.plot(x, y, 'o-')
        handles.append(l2)
        labels.append('wifi predicting')
        plt.scatter(x, y, c='red')
        plt.legend(handles=handles ,labels=labels, loc='best')
        plt.show()

#
# if __name__ == '__main__':
#     model = Model(rssi=None)
#     model.nuralnetwork(online_rss_file_path=r'C:\Users\admin\Desktop\location\location-master\data\fusion\Mydata\BLEdata_12.1.2.csv', online_location='/fusion/Mydata/RealTrace12.1.2.csv')
















