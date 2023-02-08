import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import location.pdr as pdr
import location.wifi as wifi
import location.fusion as fusion

'''
真实实验
注意假如有n个状态，那么就有n-1次状态转换
'''

frequency = 50 # 数据采集频率
sigma_wifi = 3
sigma_pdr = .1
sigma_yaw = 15/360
# 初始状态
X = np.matrix('3; 4.5; 0')
# X = np.matrix('2; 2; 0') # 对初始状态进行验证
print(X)

path = os.path.abspath(os.path.join(os.getcwd(), "./data"))
# real_trace_file = path + '/fusion/LType/RealTrace.csv'
# walking_data_file = path + '/fusion/LType/LType-05.csv'
# fingerprint_path = path + '/fusion/Fingerprint'
#
# df_walking = pd.read_csv(walking_data_file) # 实验数据
# real_trace = pd.read_csv(real_trace_file).values # 真实轨迹
#
# # 主要特征参数
# rssi = df_walking[[col for col in df_walking.columns if 'rssi' in col]].values
# linear = df_walking[[col for col in df_walking.columns if 'linear' in col]].values
# gravity = df_walking[[col for col in df_walking.columns if 'gravity' in col]].values
# rotation = df_walking[[col for col in df_walking.columns if 'rotation' in col]].values


real_trace_file = path + '/fusion/Mydata/RealTrace12.1.2.csv'
walking_data_file = path + '/fusion/Mydata/sensordata_12.1.2.csv'
bluetooth_data_file = path + '/fusion/Mydata/BLEdata_12.1.2.csv'

fingerprint_path = path + '/fusion/MyFingerprint'
df_walking = pd.read_csv(walking_data_file) # 实验数据
real_trace = pd.read_csv(real_trace_file).values # 真实轨迹
df_BLE = pd.read_csv(bluetooth_data_file)
df_BLE = df_BLE.replace(100,-100)
rssi = df_BLE[['RFstar_51DE','RFstar_040D','RFstar_1B7B','RFstar_6891','RFstar_EFBC','RFstar_2037','RFstar_0259',
               'RFstar_4EE6','RFstar_9E56','RFstar_B613','RFstar_B571','RFstar_8FBE','RFstar_78B2',
               'RFstar_FA6E','RFstar_02C5','RFstar_D4E4','RFstar_FB47','RFstar_54CE','RFstar_3D4D','RFstar_6F5F',
               'RFstar_D8FC','RFstar_9F01','RFstar_30C2','RFstar_3D81','RFstar_42D5']].values
''''RFstar_9120','''
linear = df_walking[[col for col in df_walking.columns if 'lac' in col]].values
gravity = df_walking[[col for col in df_walking.columns if 'gac' in col]].values
rotation = df_walking[[col for col in df_walking.columns if 'rv' in col]].values
rssi = (rssi+60)/40




pdr = pdr.Model(linear, gravity, rotation)
wifi = wifi.Model(rssi)
fusion = fusion.Model()

# 指纹数据
fingerprint_rssi, fingerprint_position = wifi.create_fingerprint(fingerprint_path)

# 找到峰值出的rssi值
steps = pdr.step_counter(frequency=frequency, walkType='fusion')
print('steps:', len(steps))
# result = fingerprint_rssi[0].reshape(1, rssi.shape[1])
# for k, v in enumerate(steps):
#     index = v['index']
#     value = rssi[index]
#     value = value.reshape(1, len(value))
#     result = np.concatenate((result,value),axis=0)
result = rssi[0].reshape(1, rssi.shape[1])
for k, v in enumerate(steps):
    value = rssi[k+1]
    value = value.reshape(1, len(value))
    result = np.concatenate((result,value),axis=0)
# result = rssi.reshape(rssi.shape[0], rssi.shape[1])
# print(result)

# knn算法
predict, accuracy = wifi.knn_reg(fingerprint_rssi, fingerprint_position, result, real_trace)
print('knn accuracy:', accuracy, 'm')

# svm算法
# predict, accuracy = wifi.svm_reg(fingerprint_rssi, fingerprint_position, result, real_trace)
# print('svm accuracy:', accuracy, 'm')

# rf算法
# predict, accuracy = wifi.rf_reg(fingerprint_rssi, fingerprint_position, result, real_trace)
# print('rf accuracy:', accuracy, 'm')

# gdbt算法
# predict, accuracy = wifi.dbdt(fingerprint_rssi, fingerprint_position, result, real_trace)
# print('gdbt accuracy:', accuracy, 'm')

# 多层感知机
# predict, accuracy = wifi.nn(fingerprint_rssi, fingerprint_position, result, real_trace)
# print('nn accuracy:', accuracy, 'm')

# 神经网络
# predict, accuracy = wifi.nuralnetwork(online_rss_file_path=bluetooth_data_file, online_location=real_trace,fingerprint_file_path=fingerprint_path)
# print('nn accuracy:', accuracy, 'm')

predict = np.array(predict)

init_x = X[0, 0]

init_y = X[1, 0]
print(init_x)
print(init_y)
init_angle = X[2, 0]
x_pdr, y_pdr, strides, angle = pdr.pdr_position(frequency=frequency, walkType='fusion', offset=init_angle, initPosition=(init_x, init_y))
print(len(angle))
# ekf
X_real = real_trace[:,0]
Y_real = real_trace[:,1]
X_wifi = predict[:,0]
Y_wifi = predict[:,1]
X_pdr = x_pdr
Y_pdr = y_pdr
L = strides

theta_counter = -1
def state_conv(parameters_arr):
    global theta_counter
    theta_counter = theta_counter+1
    x = parameters_arr[0]
    y = parameters_arr[1]
    theta = parameters_arr[2]
    return x+L[theta_counter]*np.sin(theta), y+L[theta_counter]*np.cos(theta), angle[theta_counter]

# 目前不考虑起始点（设定为0，0），因此wifi数组长度比实际位置长度少1
observation_states = []

for i in range(len(angle)):
    x = X_wifi[i]
    y = Y_wifi[i]
    observation_states.append(np.matrix([
        [x], [y], [L[i]], [angle[i]]
    ]))

transition_states = []
for k, v in enumerate(angle):
    x = X_pdr[k]
    y = Y_pdr[k]
    theta = angle[k]
    V = np.matrix([[x],[y],[theta]])
    transition_states.append(V)

# 状态协方差矩阵（初始状态不是非常重要，经过迭代会逼近真实状态）
P = np.matrix([[1, 0, 0],
               [0, 1, 0],
               [0, 0, 1]])
# 观测矩阵
H = np.matrix([[1, 0, 0],
               [0, 1, 0],
               [0, 0, 0],
               [0, 0, 1]])
# 状态转移协方差矩阵
Q = np.matrix([[sigma_pdr**2, 0, 0],
               [0, sigma_pdr**2, 0],
               [0, 0, sigma_yaw**2]])
# 观测噪声方差
R = np.matrix([[sigma_wifi**2, 0, 0, 0],
               [0, sigma_wifi**2, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, sigma_yaw**2]])

def jacobF_func(i):
    return np.matrix([[1, 0, L[i]*np.cos(angle[i])],
                      [0, 1, -L[i]*np.sin(angle[i])],
                      [0, 0, 1]])

S = fusion.ekf2d(
    transition_states = transition_states # 状态数组
   ,observation_states = observation_states # 观测数组
   ,transition_func = state_conv # 状态预测函数（传入参数为数组格式，该数组包含了用到的状态转换遇到的数据）
   ,jacobF_func = jacobF_func # 一阶线性的状态转换公式
   ,initial_state_covariance = P
   ,observation_matrices = H
   ,transition_covariance = Q
   ,observation_covariance = R
)

X_ekf = []
Y_ekf = []

for v in S:
    X_ekf.append(v[0, 0])
    Y_ekf.append(v[1, 0])

ekf_predict = np.concatenate((np.array(X_ekf).reshape(len(X_ekf), 1),
                              np.array(Y_ekf).reshape(len(Y_ekf), 1)), axis=1)
accuracy = fusion.square_accuracy(ekf_predict, real_trace)
print('ekf accuracy:', accuracy, 'm')

x = X_ekf
y = Y_ekf
for k in range(0, len(x)):
    plt.annotate(k, xy=(x[k], y[k]), xytext=(x[k]+0.1,y[k]+0.1))

plt.grid()
plt.plot(X_real, Y_real, 'o-', label='real tracks')
plt.plot(X_wifi, Y_wifi, 'r.', label='observation points')
plt.plot(X_pdr, Y_pdr, 'o-', label='dead reckoning')
plt.plot(X_ekf, Y_ekf, 'o-', label='ekf positioning')
plt.legend()
plt.show()