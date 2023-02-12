import csv
import os

import pandas

with open('all.csv' , 'a', encoding='utf-8') as write_obj:
    root_dir = '../data/fusion/MyFingerprint/train/'
    file_list = os.listdir(root_dir)
    count = 0
    for file in file_list:
        with open(root_dir+""+file, 'r') as read_obj:
            writer = csv.writer(write_obj)
            data = pandas.read_csv(root_dir+""+file)
            rssi_data = data.iloc[1:, 2:27]
            print(rssi_data)
            location_data = data['classification']
            rssi_data.insert(25,column='classification', value=location_data)
            if(count==0):
                rssi_data.to_csv('all.csv',sep=",",mode='a')
            else:
                rssi_data.to_csv('all.csv', sep=",", mode='a',header=False)

            count+=1