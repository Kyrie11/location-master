import location.pdr as pdr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

path = os.path.abspath(os.path.join(os.getcwd(), "./data"))
# walking_data_file = path + '/linear_8m/linear01.csv'
# walking_data_file = path + '/still/still02.csv'
# walking_data_file = path + '/fusion/LType/LType-06.csv'
walking_data_file = path + '/fusion/Mydata/sensordata_11.19.2.csv'
real_trace_file = path + '/fusion/Mydata/RealTrace.csv'
real_trace = pd.read_csv(real_trace_file).values # 真实轨迹

df_walking = pd.read_csv(walking_data_file)

# 获得线性加速度、重力加速度、姿态仰角的numpy.ndarray数据
linear = df_walking[[col for col in df_walking.columns if 'lac' in col]].values
gravity = df_walking[[col for col in df_walking.columns if 'gac' in col]].values
rotation = df_walking[[col for col in df_walking.columns if 'rv' in col]].values

pdr = pdr.Model(linear, gravity, rotation)

# # Demo1：显示垂直方向合加速度与步伐波峰分布
# # frequency：数据采集频率
# # walkType：行走方式（normal为正常走路模式，abnormal为做融合定位实验时走路模式）
# pdr.show_steps(frequency=70, walkType='abnormal')

# # Demo2：显示数据在一定范围内的分布情况，用来判断静止数据呈现高斯分布
# # 传入参数为静止状态x（y或z）轴线性加速度
# acc_z = linear[:,2]
# pdr.show_gaussian(acc_z, True)

# # Demo3：显示三轴线性加速度分布情况
# pdr.show_data('rotation')

# # Demo4：获取步伐信息
# # 返回值steps为字典类型，index为样本序号，acceleration为步伐加速度峰值
steps = pdr.step_counter(frequency=100, walkType='normal')
valleyarray = pdr.find_valley(frequency=100, walkType='normal')
print('steps:', len(steps))
stride = pdr.step_stride_Weinberg # 步长推算函数
# # 计算步长推算的平均误差
accuracy = []
for i,v in enumerate(steps):
    # a = v['acceleration']
    print(stride(v['acceleration'],valleyarray[i]['acceleration']))
    accuracy.append(
        np.abs(stride(v['acceleration'],valleyarray[i]['acceleration'])-0.65)
    )
square_sum = 0
for v in accuracy:
    square_sum += v*v
acc_mean = (square_sum/len(steps))**(1/2)
print("mean: %f" % acc_mean) # 平均误差
print("min: %f" % np.min(accuracy)) # 最小误差
print("max: %f" % np.max(accuracy)) # 最大误差
print("sum: %f" % np.sum(accuracy)) # 累积误差

# # Demo5：获取航向角
# theta = pdr.step_heading()[:10]
# temp = theta[0]
# for i,v in enumerate(theta):
#     theta[i] = np.abs(v-temp)*360/(2*np.pi)
#     print(theta[i])
# print("mean: %f" % np.mean(theta))

# Demo6：显示PDR预测轨迹
# 注意：PDR不清楚初始位置与初始航向角
# pdr.show_trace(frequency=70, walkType='normal')

#  offset=-0.25
pdr.show_trace(frequency=100, walkType='normal',  initPosition=(3, 4.5), real_trace=real_trace, offset=0)
# pdr.show_steps(frequency=100, walkType='normal')
# pdr.show_data(dataType='linear')