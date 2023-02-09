import csv
import os

root_dir = ''
file_list = os.listdir(root_dir)
for i in file_list:
    with open(root_dir + "" + i, 'w', encoding="utf-8") as obj:
        writer = csv.writer(obj)

