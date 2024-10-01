#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 19:29:10 2019

@author: tbury

Compute residual dynamics from Lowess smoothing for each time series
generated for training data

"""

import numpy as np
import pandas as pd

import ewstools
import os
import sys


bif_max = int(sys.argv[1])
batch_num = int(sys.argv[2])


# Create directories for output
if not os.path.exists('hopf/output_resids'):
    os.makedirs('hopf/output_resids')
if not os.path.exists('fold/output_resids'):
    os.makedirs('fold/output_resids')
if not os.path.exists('branch/output_resids'):
    os.makedirs('branch/output_resids')

# Loop through each time-series and compute residuals

# Counter
i = (batch_num-1)*bif_max + 1
# While the file exists
#hopf
while os.path.isfile('hopf/output_sims/white_tseries'+str(i)+'.csv'):
    df_traj_white = pd.read_csv('hopf/output_sims/white_tseries'+str(i)+'.csv')
    df_traj_red = pd.read_csv('hopf/output_sims/red_tseries'+str(i)+'.csv')
    
    # Compute EWS
    dic_ews_white = ewstools.TimeSeries(data = df_traj_white['xw'])
    dic_ews_white.detrend(method = 'Lowess', span = 0.2)
    df_resids_white = pd.DataFrame(columns=['residuals', 'b'])
    df_resids_white['residuals'] = dic_ews_white.state['residuals']
    df_resids_white['b'] = df_traj_white['b']

    dic_ews_red = ewstools.TimeSeries(data = df_traj_red['xr'])
    dic_ews_red.detrend(method = 'Lowess', span = 0.2)
    df_resids_red = pd.DataFrame(columns=['residuals', 'b'])
    df_resids_red['residuals'] = dic_ews_red.state['residuals']
    df_resids_red['b'] = df_traj_red['b']
    
    # Output residual time-series
    df_resids_white.to_csv('hopf/output_resids/white_resids'+str(i)+'.csv')
    df_resids_red.to_csv('hopf/output_resids/red_resids'+str(i)+'.csv')
    
    if np.mod(i,100) == 0:
        print('Residuals for hopf trajectory {} complete'.format(i))
        
    # Increment
    i+=1

i = (batch_num-1)*bif_max + 1
#fold
while os.path.isfile('fold/output_sims/white_tseries'+str(i)+'.csv'):
    df_traj_white = pd.read_csv('fold/output_sims/white_tseries'+str(i)+'.csv')
    df_traj_red = pd.read_csv('fold/output_sims/red_tseries'+str(i)+'.csv')
    
    # Compute EWS
    dic_ews_white = ewstools.TimeSeries(data = df_traj_white['xw'])
    dic_ews_white.detrend(method = 'Lowess', span = 0.2)
    df_resids_white = pd.DataFrame(columns=['residuals', 'b'])
    df_resids_white['residuals'] = dic_ews_white.state['residuals']
    df_resids_white['b'] = df_traj_white['b']

    dic_ews_red = ewstools.TimeSeries(data = df_traj_red['xr'])
    dic_ews_red.detrend(method = 'Lowess', span = 0.2)
    df_resids_red = pd.DataFrame(columns=['residuals', 'b'])
    df_resids_red['residuals'] = dic_ews_red.state['residuals']
    df_resids_red['b'] = df_traj_red['b']
    
    # Output residual time-series
    df_resids_white.to_csv('fold/output_resids/white_resids'+str(i)+'.csv')
    df_resids_red.to_csv('fold/output_resids/red_resids'+str(i)+'.csv')
       
    if np.mod(i,100) == 0:
        print('Residuals for fold trajectory {} complete'.format(i))
        
    # Increment
    i+=1

i = (batch_num-1)*bif_max + 1
#branch
while os.path.isfile('branch/output_sims/white_tseries'+str(i)+'.csv'):
    df_traj_white = pd.read_csv('branch/output_sims/white_tseries'+str(i)+'.csv')
    df_traj_red = pd.read_csv('branch/output_sims/red_tseries'+str(i)+'.csv')
    
    # Compute EWS
    dic_ews_white = ewstools.TimeSeries(data = df_traj_white['xw'])
    dic_ews_white.detrend(method = 'Lowess', span = 0.2)
    df_resids_white = pd.DataFrame(columns=['residuals', 'b'])
    df_resids_white['residuals'] = dic_ews_white.state['residuals']
    df_resids_white['b'] = df_traj_white['b']

    dic_ews_red = ewstools.TimeSeries(data = df_traj_red['xr'])
    dic_ews_red.detrend(method = 'Lowess', span = 0.2)
    df_resids_red = pd.DataFrame(columns=['residuals', 'b'])
    df_resids_red['residuals'] = dic_ews_red.state['residuals']
    df_resids_red['b'] = df_traj_red['b']
    
    # Output residual time-series
    df_resids_white.to_csv('branch/output_resids/white_resids'+str(i)+'.csv')
    df_resids_red.to_csv('branch/output_resids/red_resids'+str(i)+'.csv')

    if np.mod(i,100) == 0:
        print('Residuals for branch trajectory {} complete'.format(i))
        
    # Increment
    i+=1