import location.pdr as pdr
import location.wifi as wifi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


path = os.path.abspath(os.path.join(os.getcwd(), "./data"))

# real_trace_file = path + '/fusion/LType/RealTrace.csv'
# walking_data_file = path + '/fusion/LType/LType-02.csv'
# fingerprint_path = path + '/fusion/Fingerprint'

real_trace_file = path + '/fusion/Mydata/RealTrace12.1.2.csv'
walking_data_file = path + '/fusion/Mydata/sensordata_12.1.2.csv'
bluetooth_data_file = path + '/fusion/Mydata/BLEdata_12.1.2.0.csv'

fingerprint_path = path + '/fusion/MyFingerprint'
df_walking = pd.read_csv(walking_data_file) # 实验数据
real_trace = pd.read_csv(real_trace_file).values # 真实轨迹
df_BLE = pd.read_csv(bluetooth_data_file)
df_BLE = df_BLE.replace(100,-100)
# df_BLE = (df_BLE+60)/40
# 主要特征参数
# rssi = df_walking[[col for col in df_walking.columns if 'rssi' in col]].values
# linear = df_walking[[col for col in df_walking.columns if 'linear' in col]].values
# gravity = df_walking[[col for col in df_walking.columns if 'gravity' in col]].values
# rotation = df_walking[[col for col in df_walking.columns if 'rotation' in col]].values

# rssi = df_BLE[[col for col in df_BLE.columns if 'RFstar' in col]].values
rssi = df_BLE[['RFstar_51DE','RFstar_040D','RFstar_1B7B','RFstar_6891','RFstar_EFBC','RFstar_2037','RFstar_0259','RFstar_4EE6',
               'RFstar_9E56','RFstar_B613','RFstar_B571','RFstar_8FBE','RFstar_78B2','RFstar_FA6E','RFstar_02C5',
               'RFstar_D4E4','RFstar_FB47','RFstar_54CE','RFstar_3D4D','RFstar_6F5F','RFstar_D8FC','RFstar_9F01','RFstar_30C2',
               'RFstar_3D81','RFstar_42D5']].values
''''RFstar_9120','''
linear = df_walking[[col for col in df_walking.columns if 'lac' in col]].values
gravity = df_walking[[col for col in df_walking.columns if 'gac' in col]].values
rotation = df_walking[[col for col in df_walking.columns if 'rv' in col]].values
rssi = (rssi+60)/40
pdr = pdr.Model(linear, gravity, rotation)
wifi = wifi.Model(rssi)

# 指纹数据
fingerprint_rssi, fingerprint_position = wifi.create_fingerprint(fingerprint_path)
# print(fingerprint_rssi)
# 找到峰值出的rssi值
# steps = pdr.step_counter(frequency=70, walkType='abnormal')
steps = pdr.step_counter(frequency=100, walkType='normal')
print('steps:', len(steps))
# print(fingerprint_rssi[0])
# result = fingerprint_rssi[0].reshape(1, rssi.shape[1])
# result = rssi[0].reshape(1, rssi.shape[1])
# for k, v in enumerate(steps):
#     # index = v['index']
#     # value = rssi[index]
#     value = rssi[k+1]
#     value = value.reshape(1, len(value))
#     result = np.concatenate((result,value),axis=0)
result = rssi[0].reshape(1, rssi.shape[1])
for k, v in enumerate(steps):
    value = rssi[k+1]
    value = value.reshape(1, len(value))
    result = np.concatenate((result,value),axis=0)
# result = rssi.reshape(rssi.shape[0], rssi.shape[1])
# print(result)
# result = np.array(rssi).reshape(1, rssi.shape[1])

# knn算法
# predict, accuracy = wifi.knn_reg(fingerprint_rssi, fingerprint_position, result, real_trace)
# print('knn accuracy:', accuracy, 'm')

# 添加区域限制的knn回归
# predict, accuracy = wifi.ml_limited_reg('knn', fingerprint_rssi, fingerprint_position, result, real_trace)
# print('knn_limited accuracy:', accuracy, 'm')

# svm算法
# predict, accuracy = wifi.svm_reg(fingerprint_rssi, fingerprint_position, result, real_trace)
# print('svm accuracy:', accuracy, 'm')

# rf算法
# predict, accuracy = wifi.rf_reg(fingerprint_rssi, fingerprint_position, result, real_trace)
# print('rf accuracy:', accuracy, 'm')

# 添加区域限制rf的rf算法
# predict, accuracy = wifi.ml_limited_reg('rf', fingerprint_rssi, fingerprint_position, result, real_trace)
# print('rf_limited accuracy:', accuracy, 'm')

# gdbt算法
# predict, accuracy = wifi.dbdt(fingerprint_rssi, fingerprint_position, result, real_trace)
# print('gdbt accuracy:', accuracy, 'm')

# 多层感知机
predict, accuracy = wifi.nn(fingerprint_rssi, fingerprint_position, result, real_trace)
print('nn accuracy:', accuracy, 'm')


# 神经网络
# predict, accuracy = wifi.nuralnetwork(online_rss_file_path=bluetooth_data_file, online_location=real_trace,fingerprint_file_path=fingerprint_path)
# print('nn accuracy:', accuracy, 'm')

wifi.show_trace(predict, real_trace=real_trace)