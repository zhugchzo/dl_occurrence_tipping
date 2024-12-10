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
import math
import ewstools
import os

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)

# Get the directory containing the current file
current_dir = os.path.dirname(current_file_path)

# Change the working directory to the directory of the current file
os.chdir(current_dir)

trans = pd.read_csv('../data_hysteresis/sprott_b_white/original series/sprott_b_trans.csv')
trans_f = trans['trans_time1'].iloc[0]
trans_r = trans['trans_time2'].iloc[0]

data_f = pd.read_csv('../data_hysteresis/sprott_b_white/original series/sprott_b_forward.csv',index_col=0)
data_r = pd.read_csv('../data_hysteresis/sprott_b_white/original series/sprott_b_reverse.csv',index_col=0)

start_f = data_f['k'].iloc[0]
over_f = trans['trans_value1'].iloc[0]
start_r = data_r['k'].iloc[0]
over_r = trans['trans_value2'].iloc[0]

for bl in np.linspace(1,1.4,5):

    bl = round(bl,2)
    bl_pi = bl*math.pi

    ts_start = int((bl_pi-start_f)*trans_f/(over_f-start_f))

    tmax = 400 # sequence length
    n = np.random.uniform(4,400) # length of the randomly selected sequence to the bifurcation
    series_len = tmax + int(n)

    ts_f = np.arange(ts_start,trans_f)
    s_f = np.linspace(ts_start+1,ts_start+len(ts_f)-2,len(ts_f)-2)
    np.random.shuffle(s_f)
    s_f = list(np.sort(s_f[0:series_len-2]))
    s_f.insert(0,ts_start)
    s_f.append(ts_start+len(ts_f)-1)

    f_sample = data_f.iloc[s_f].copy()
    f_tseries = f_sample.iloc[0:tmax].copy()

    state_tseries = f_tseries['x']
    ts = ewstools.TimeSeries(data=state_tseries)
    ts.detrend(method='Lowess', span=0.2)

    f_tseries['residuals_x'] = ts.state['residuals']
    f_tseries['Time'] = np.arange(0,tmax)
    f_tseries.set_index('Time', inplace=True)

    f_tseries.to_csv('../data_hysteresis/sprott_b_white/sprott_b_forward_{}.csv'.format(bl))

for bl in np.linspace(2,1.6,5):

    bl = round(bl,2)
    bl_pi = bl*math.pi

    ts_start = int(abs((bl_pi-start_r)*trans_r/(over_r-start_r)))

    tmax = 400 # sequence length
    n = np.random.uniform(4,400) # length of the randomly selected sequence to the bifurcation
    series_len = tmax + int(n)

    ts_r = np.arange(ts_start,trans_r)
    s_r = np.linspace(ts_start+1,ts_start+len(ts_r)-2,len(ts_r)-2)
    np.random.shuffle(s_r)
    s_r = list(np.sort(s_r[0:series_len-2]))
    s_r.insert(0,ts_start)
    s_r.append(ts_start+len(ts_r)-1)

    r_sample = data_r.iloc[s_r].copy()
    r_tseries = r_sample.iloc[0:tmax].copy()

    state_tseries = r_tseries['x']
    ts = ewstools.TimeSeries(data=state_tseries)
    ts.detrend(method='Lowess', span=0.2)

    r_tseries['residuals_x'] = ts.state['residuals']
    r_tseries['Time'] = np.arange(0,tmax)
    r_tseries.set_index('Time', inplace=True)

    r_tseries.to_csv('../data_hysteresis/sprott_b_white/sprott_b_reverse_{}.csv'.format(bl))