#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
import train_funs as funs

import sys

batch_num = int(sys.argv[1])
nsims = int(sys.argv[2])

tburn = 100
ts_len = 500 #  length of time series to store for training
max_order = 10 # Max polynomial degree

# Noise amplitude distribution parameters
sigma = 0.01

sup_value = []
sub_value = []

print('Run forced PF simulations')
j = 1

while j <= nsims:

    n = np.random.uniform(5,500)
    series_len = ts_len + int(n)
    
    # Draw starting value of bifurcaiton parameter at random
    # Eigenvalue lambda = 1+mu
    # Lower bound csp to lambda=-0.8
    # Upper bound csp to lambda=0.8
    bl = np.random.uniform(-1.8,-0.2)
    bh = 0.2
    
    # Run simulation
    simulation_results = funs.simulate_pf(bl, bh, ts_len, series_len, tburn, sigma, max_order)
    
    stop = simulation_results[0]
    sup_df_traj = simulation_results[1]
    sub_df_traj = simulation_results[2]
    sup_trans_point = simulation_results[3]
    sub_trans_point = simulation_results[4]

    if stop == 1:
        continue

    else:
        # Export
        sup_df_traj[['x','b']].to_csv('sup_pitchfork/output_sims/sup_tseries'+str((batch_num-1)*nsims + j)+'.csv')
        sub_df_traj[['x','b']].to_csv('sub_pitchfork/output_sims/sub_tseries'+str((batch_num-1)*nsims + j)+'.csv')
 
        sup_df_value = pd.DataFrame([series_len,sup_trans_point])
        sup_df_value.to_csv('sup_pitchfork/value'+str((batch_num-1)*nsims + j)+'.csv',header=False, index=False)
        sub_df_value = pd.DataFrame([series_len,sub_trans_point])
        sub_df_value.to_csv('sub_pitchfork/value'+str((batch_num-1)*nsims + j)+'.csv',header=False, index=False)

        print('Simulation '+str(j)+' complete')

        j += 1

