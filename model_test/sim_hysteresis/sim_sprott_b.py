#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on: March 22, 2024

@author: Chengzuo Zhuge

Sampling irregularly on Sprott B model 
Compute residual time series

"""

# import python libraries
import numpy as np
import pandas as pd
import ewstools
import os

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)

# Get the directory containing the current file
current_dir = os.path.dirname(current_file_path)

# Change the working directory to the directory of the current file
os.chdir(current_dir)

# Simulation parameters
tmax = 400

# EWS parameters
dt2 = 1 # spacing between time-series for EWS computation
rw = 0.25 # rolling window
span = 0.2 # bandwidth
lags = [1] # autocorrelation lag times
ews = ['var','ac']


over_time = 30000 #两个方向采样的终止步长，参数序列增大或减小0.3*pi

data_f = pd.read_csv('../data_hysteresis/sprott_b_white/original series/sprott_b_forward.csv')
data_r = pd.read_csv('../data_hysteresis/sprott_b_white/original series/sprott_b_reverse.csv')


ts_f = np.arange(0,over_time)
s_f = np.linspace(1,len(ts_f)-2,len(ts_f)-2)
np.random.shuffle(s_f)
s_f = list(np.sort(s_f[0:tmax-2]))
s_f.insert(0,0)
s_f.append(len(ts_f)-1)

ts_r = np.arange(0,over_time)
s_r = np.linspace(1,len(ts_r)-2,len(ts_r)-2)
np.random.shuffle(s_r)
s_r = list(np.sort(s_r[0:tmax-2]))
s_r.insert(0,0)
s_r.append(len(ts_r)-1)

f_sample = data_f.iloc[s_f].copy()
r_sample = data_r.iloc[s_r].copy()


f_sample['Time'] = np.arange(0,tmax)
f_sample.set_index('Time', inplace=True)
r_sample['Time'] = np.arange(0,tmax)
r_sample.set_index('Time', inplace=True)


f_traj = f_sample.copy()
r_traj = r_sample.copy()

# compute resids for forward time series

var = 'z'

f_ews_dic = ewstools.core.ews_compute(f_traj[var], 
                roll_window = rw,
                smooth='Lowess',
                span = span,
                lag_times = lags, 
                ews = ews)

# The DataFrame of EWS
f_ews = f_ews_dic['EWS metrics']

# Include a column in the DataFrames for realisation number and variable
f_ews['Variable'] = var
f_ews['k'] = f_traj['k']

# compute resids for reverse time series

r_ews_dic = ewstools.core.ews_compute(r_traj[var], 
                roll_window = rw,
                smooth='Lowess',
                span = span,
                lag_times = lags, 
                ews = ews)

# The DataFrame of EWS
r_ews = r_ews_dic['EWS metrics']

# Include a column in the DataFrames for realisation number and variable
r_ews['Variable'] = var
r_ews['k'] = r_traj['k']

f_ews = f_ews[['Residuals','k']]
r_ews = r_ews[['Residuals','k']]

f_ews.to_csv('../data_hysteresis/sprott_b_white/sprott_b_forward_{}.csv'.format(tmax))
r_ews.to_csv('../data_hysteresis/sprott_b_white/sprott_b_reverse_{}.csv'.format(tmax))