#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on: March 22, 2024

@author: Zhuge Chengzuo

Test DL algorithm on sprott B model 

"""

import pandas
import numpy as np
from tensorflow.keras.models import load_model
import os

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)

# Get the directory containing the current file
current_dir = os.path.dirname(current_file_path)

# Change the working directory to the directory of the current file
os.chdir(current_dir)

if not os.path.exists('../../results'):
    os.makedirs('../../results')

kk = 10

seq_len = 500

f_trans = 4.58982161367064
r_trans = 4.83495634709873

ssf_list = []
ssr_list = []

preds_f_list = []
preds_r_list = []
relative_ef_list = []
relative_er_list = []

ssf_vals = np.linspace(1,1.4,11)
ssr_vals = np.linspace(2,1.6,11)

for ssf,ssr in zip(ssf_vals,ssr_vals):

    ssf = round(ssf,2)
    ssr = round(ssr,2)

    f_resids = pandas.read_csv('../data_hysteresis/sprott_b_white/sprott_b_forward_{}.csv'.format(ssf))
    r_resids = pandas.read_csv('../data_hysteresis/sprott_b_white/sprott_b_reverse_{}.csv'.format(ssr))
    keep_col_resids = ['residuals_x','k']
    new_f_resids = f_resids[keep_col_resids]
    new_r_resids = r_resids[keep_col_resids]

    values_fs = new_f_resids['k'].iloc[0]
    values_fo = new_f_resids['k'].iloc[-1]

    values_rs = new_r_resids['k'].iloc[0]
    values_ro = new_r_resids['k'].iloc[-1]

    f_resids = new_f_resids.values
    r_resids = new_r_resids.values

    for i in range(seq_len-len(f_resids)):
        f_resids=np.insert(f_resids,0,[[0,0]],axis= 0)
        r_resids=np.insert(r_resids,0,[[0,0]],axis= 0)

    # normalizing input time series by the average. 
    values_avg_f = 0.0
    values_avg_r = 0.0
    count_avg_f = 0
    count_avg_r = 0

    for i in range (0,seq_len):
        if f_resids[i][0]!= 0 or f_resids[i][1]!= 0:
            values_avg_f = values_avg_f + abs(f_resids[i][0])                                       
            count_avg_f = count_avg_f + 1

        if r_resids[i][0]!= 0 or r_resids[i][1]!= 0:
            values_avg_r = values_avg_r + abs(r_resids[i][0])                                       
            count_avg_r = count_avg_r + 1
    
    if count_avg_f != 0:
        values_avg_f = values_avg_f/count_avg_f
        for i in range (0,seq_len):
            if f_resids[i][0]!= 0 or f_resids[i][1]!= 0:
                f_resids[i][0]= f_resids[i][0]/values_avg_f
                f_resids[i][1]= (f_resids[i][1]-values_fs)/(values_fo-values_fs)

    if count_avg_r != 0:
        values_avg_r = values_avg_r/count_avg_r
        for i in range (0,seq_len):
            if r_resids[i][0]!= 0 or r_resids[i][1]!= 0:
                r_resids[i][0]= r_resids[i][0]/values_avg_r
                r_resids[i][1]= (r_resids[i][1]-values_rs)/(values_ro-values_rs)

    test_f = f_resids.reshape(-1,seq_len,2,1)
    test_r = r_resids.reshape(-1,seq_len,2,1)

    test_preds_record_f = []
    test_preds_record_r = []

    for i in range(1,kk+1):

        model_name = '../../dl_model/best_model_{}.keras'.format(i)

        model = load_model(model_name)

        test_preds_f = model.predict(test_f)
        test_preds_r = model.predict(test_r)

        test_preds_record_f.append(test_preds_f)
        test_preds_record_r.append(test_preds_r)

    test_preds_record_f = np.array(test_preds_record_f)
    test_preds_record_r = np.array(test_preds_record_r)

    preds_f = np.mean(test_preds_record_f) * (values_fo-values_fs) + values_fs
    preds_r = np.mean(test_preds_record_r) * (values_ro-values_rs) + values_rs

    distance_f = abs(f_trans - values_fo)
    distance_r = abs(r_trans - values_ro)
    error_f = abs(preds_f - f_trans)
    error_r = abs(preds_r - r_trans)

    relative_ef = error_f/distance_f
    relative_er = error_r/distance_r

    preds_f_list.append(preds_f)
    preds_r_list.append(preds_r)
    relative_ef_list.append(relative_ef)
    relative_er_list.append(relative_er)

    ssf_list.append(ssf)
    ssr_list.append(ssr)

preds_results = {'ssf_list':ssf_list, 'preds_f_list':preds_f_list, 'relative_ef_list':relative_ef_list,
                    'ssr_list':ssr_list, 'preds_r_list':preds_r_list, 'relative_er_list':relative_er_list}

preds_results = pandas.DataFrame(preds_results)
preds_results.to_csv('../../results/sprott_b.csv',header = True)








