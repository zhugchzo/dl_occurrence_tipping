import pandas as pd
import numpy as np
import os
import ruptures as rpt

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)

# Get the directory containing the current file
current_dir = os.path.dirname(current_file_path)

# Change the working directory to the directory of the current file
os.chdir(current_dir)

# 20mv
txt = open('../empirical_data/thermoacoustic_20mv.txt')
datalines = txt.readlines()
dataset = []
for data in datalines:
    data = data.strip().split()
    dataset.append(data)

for data in dataset:
    data[0] = (float(data[0])-3.65630914691268160E+9)*0.02
    data[1] = float(data[1])*1000/0.2175

dataset = np.array(dataset[int(12E5/(2.4/1.65)) : int(12E5/(2.4/1.75))])[::50]

dataset_detection = dataset[:,1]

algo_20 = rpt.Pelt(model="rbf").fit(dataset_detection)
location_20 = algo_20.predict(pen=10)

tipping_20 = dataset[location_20[0],0]

# 40mv
txt = open('../empirical_data/thermoacoustic_40mv.txt')
datalines = txt.readlines()
dataset = []
for data in datalines:
    data = data.strip().split()
    dataset.append(data)

for data in dataset:
    data[0] = (float(data[0])-3.65629963087196300E+9)*0.04
    data[1] = float(data[1])*1000/0.2175

dataset = np.array(dataset[int(6E5/(2.4/1.7)) : int(6E5/(2.4/1.8))])[::25]

dataset_detection = dataset[:,1]

algo_40 = rpt.Pelt(model="rbf").fit(dataset_detection)
location_40 = algo_40.predict(pen=10)

tipping_40 = dataset[location_40[0],0]

# 60mv
txt = open('../empirical_data/thermoacoustic_60mv.txt')
datalines = txt.readlines()
dataset = []
for data in datalines:
    data = data.strip().split()
    dataset.append(data)

for data in dataset:
    data[0] = (float(data[0])-3.65630991104181150E+9)*0.06
    data[1] = float(data[1])*1000/0.2175

dataset = np.array(dataset[int(4E5/(2.4/1.8)) : int(4E5/(2.4/1.9))])[::20]

dataset_detection = dataset[:,1]

algo_60 = rpt.Pelt(model="rbf").fit(dataset_detection)
location_60 = algo_60.predict(pen=10)

tipping_60 = dataset[location_60[0],0]

print(' 20mv:{}, 40mv:{}, 60mv:{}'.format(tipping_20,tipping_40,tipping_60))


