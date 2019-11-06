import matplotlib.pyplot as plt
import pandas as pd
import heapq
import tensorflow as tf
import numpy as np
from statistics import mean
from scipy.signal import find_peaks
from scipy.signal import argrelmin

import xlsxwriter
import random

dataset_paths = ["data/Squat 1.xlsx", "data/Squat 2.xlsx", "data/Squat 3.xlsx"]
Allsets = [[0]*8]*27

for j in range(len(dataset_paths)):
    dataset_path = dataset_paths[j]
    
    column_names = ['Time', 'X', 'Y', 'Z']
    raw_dataset = pd.read_excel(dataset_path, names=column_names, comment='#')
    dataset = raw_dataset.copy()
    dataset.tail()

    Time_dataset = dataset.get('Time').values
    X_dataset = dataset.get('X').values
    Y_dataset = dataset.get('Y').values
    Z_dataset = dataset.get('Z').values
    Y_data_mean = mean(Y_dataset)

    minimals = argrelmin(Y_dataset)
    minimals = minimals[0]
    i = 0
    while i < len(minimals)-1:
        if(abs(minimals[i+1]-minimals[i])<3):
            if(minimals[i+1]>minimals[i]):
                minimals = np.delete(minimals,i+1)
            else:
                minimals = np.delete(minimals,i)
            i=0
        else:
            i = i + 1
    minimals = np.delete(minimals,0)
    deletion = []
    for i in range(len(minimals)):
        if(Y_dataset[minimals[i]]>9.5):
            if Y_dataset[minimals[i]]<10.5:
                deletion.append(i)
    minimals = np.delete(minimals,deletion)

    if minimals.size > 18:
        deletion = []
        for i in range(len(minimals)):
            if (Y_dataset[minimals[i]] > 9.3):
                deletion.append(i)
        minimals = np.delete(minimals, deletion)
    minimals = minimals[:18]
    i = 0
    sets = [[0]*8]*9
    while i < len(minimals)-1:
        sets[int(i/2)] = X_dataset[minimals[i]:minimals[i]+8]
        sets[int(i/2)] = np.append(sets[int(i/2)], Y_dataset[minimals[i]:minimals[i]+8])
        sets[int(i/2)] = np.append(sets[int(i/2)], Z_dataset[minimals[i]:minimals[i] + 8])
        i = i+2

    Allsets[9*j] = sets[0]
    Allsets[9*j] = np.append(Allsets[9*j], [0, 0, 0, 0, 1, 0])
    Allsets[9*j+1] = sets[1]
    Allsets[9 * j+1] = np.append(Allsets[9 * j+1], [0, 0, 0, 0, 1, 0])
    Allsets[9*j+2] = sets[2]
    Allsets[9 * j+2] = np.append(Allsets[9 * j+2], [0, 0, 0, 0, 1, 0])
    Allsets[9*j+3] = sets[3]
    Allsets[9 * j+3] = np.append(Allsets[9 * j+3], [0, 0, 0, 0, 0, 1])
    Allsets[9*j+4] = sets[4]
    Allsets[9 * j+4] = np.append(Allsets[9 * j+4], [0, 0, 0, 0, 0, 1])
    Allsets[9*j+5] = sets[5]
    Allsets[9 * j+5] = np.append(Allsets[9 * j+5], [0, 0, 0, 0, 0, 1])
    Allsets[9*j+6] = sets[6]
    Allsets[9 * j+6] = np.append(Allsets[9 * j+6], [0, 0, 0, 0, 0, 1])
    Allsets[9*j + 7] = sets[7]
    Allsets[9 * j+7] = np.append(Allsets[9 * j+7], [0, 0, 0, 0, 0, 1])
    Allsets[9*j + 8] = sets[8]
    Allsets[9 * j+8] = np.append(Allsets[9 * j+8], [0, 0, 0, 0, 0, 1])

workbook = xlsxwriter.Workbook('data/squat sets.xlsx')
worksheet = workbook.add_worksheet("All sets")
col = 0
single_set = 0
while single_set < len(Allsets):
    row = 0
    for element in Allsets[single_set]:
        worksheet.write(row, col, element)
        row += 1
    col += 1
    single_set += 1


workbook.close()

#plt.plot(Time_dataset, Y_dataset)
#plt.plot(Time_dataset[minimals], Y_dataset[minimals], "x")
#plt.show()
