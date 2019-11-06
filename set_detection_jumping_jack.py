from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import pandas as pd
# import tensorflow as tf
import numpy as np
from statistics import mean
from scipy.signal import find_peaks
import xlsxwriter
import random

dataset_paths = ["data/Jumping Jacks 1.xlsx","data/Jumping Jacks 2.xlsx","data/Jumping Jacks 3.xlsx"]
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

    peaks, _ = find_peaks(Y_dataset, threshold=Y_data_mean*0.5, height=10)
    X_dataset = X_dataset[peaks[0] - 5: peaks[len(peaks) - 1] + 4]
    Y_dataset = Y_dataset[peaks[0] - 5: peaks[len(peaks)-1]+4]
    Z_dataset = Z_dataset[peaks[0] - 5: peaks[len(peaks) - 1] + 4]
    Time_dataset = Time_dataset[peaks[0] - 5: peaks[len(peaks)-1]+4]

    deletion = []
    peaks = [x-(peaks[0] - 5) for x in peaks]
    for i in range(len(peaks)-1):
        if abs(peaks[i]-peaks[i+1]) > 10:
            for i in range(peaks[i] + 3,peaks[i+1] - 1):
                deletion.append(i)

    X_dataset = np.delete(X_dataset, deletion)
    Y_dataset = np.delete(Y_dataset, deletion)
    Z_dataset = np.delete(Z_dataset, deletion)
    Time_dataset = np.delete(Time_dataset,deletion)
    peaks, _ = find_peaks(Y_dataset, distance=2)

    # finds the sets
    sets = [[0]*5]*9
    current_peak = 0
    current_end = 0
    for i in range(9):
        sets[i] = X_dataset[peaks[current_peak]-1: peaks[current_peak] + 4]
        sets[i] = np.append(sets[i],[0,0,0])
        sets[i] = np.append(sets[i], Y_dataset[peaks[current_peak] - 1: peaks[current_peak] + 4])
        sets[i] = np.append(sets[i], [0, 0, 0])
        sets[i] = np.append(sets[i], Z_dataset[peaks[current_peak] - 1: peaks[current_peak] + 4])
        sets[i] = np.append(sets[i], [0, 0, 0])
        current_end = peaks[current_peak] + 3
        if i != 8:
            while peaks[current_peak] < current_end:
                current_peak = current_peak + 1

    Allsets[9*j] = sets[0]
    Allsets[9*j] = np.append(Allsets[9*j], [1, 0, 0, 0, 0, 0])
    Allsets[9*j+1] = sets[1]
    Allsets[9 * j+1] = np.append(Allsets[9 * j+1], [1, 0, 0, 0, 0, 0])
    Allsets[9*j+2] = sets[2]
    Allsets[9 * j+2] = np.append(Allsets[9 * j+2], [1, 0, 0, 0, 0, 0])
    Allsets[9*j+3] = sets[3]
    Allsets[9 * j+3] = np.append(Allsets[9 * j+3], [0, 1, 0, 0, 0, 0])
    Allsets[9*j+4] = sets[4]
    Allsets[9 * j+4] = np.append(Allsets[9 * j+4], [0, 1, 0, 0, 0, 0])
    Allsets[9*j+5] = sets[5]
    Allsets[9 * j+5] = np.append(Allsets[9 * j+5], [0, 1, 0, 0, 0, 0])
    Allsets[9*j+6] = sets[6]
    Allsets[9 * j+6] = np.append(Allsets[9 * j+6], [0, 1, 0, 0, 0, 0])
    Allsets[9*j + 7] = sets[7]
    Allsets[9 * j+7] = np.append(Allsets[9 * j+7], [0, 1, 0, 0, 0, 0])
    Allsets[9*j + 8] = sets[8]
    Allsets[9 * j+8] = np.append(Allsets[9 * j+8], [0, 1, 0, 0, 0, 0])


workbook = xlsxwriter.Workbook('data/jumping jacks sets.xlsx')
worksheet = workbook.add_worksheet("Allsets")
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

# print the plot
plt.plot(Time_dataset, Y_dataset)
plt.plot(Time_dataset[peaks], Y_dataset[peaks], "x")
plt.show()
