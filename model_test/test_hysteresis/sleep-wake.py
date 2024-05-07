#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on: March 22, 2024

@author: Zhuge Chengzuo

Test DL algorithm on sleep-wake model 

"""

import pandas
import numpy as np
from tensorflow.keras.models import load_model
import sys
import os

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)

# Get the directory containing the current file
current_dir = os.path.dirname(current_file_path)

# Change the working directory to the directory of the current file
os.chdir(current_dir)

if not os.path.exists('../../model_results'):
    os.makedirs('../../model_results')

kk = 5

seq_len = 500
test_len = 450

f_trans = 1.15282788241984
r_trans = 0.883226316248411

f_start = 0.1
f_over = 0.7
r_start = 1.9
r_over = 1.3

f_resids = pandas.read_csv('../data_hysteresis/sleep-wake_white/sleep-wake_forward_{}.csv'.format(test_len))
r_resids = pandas.read_csv('../data_hysteresis/sleep-wake_white/sleep-wake_reverse_{}.csv'.format(test_len))
keep_col_resids = ['Residuals','D']
new_f_resids = f_resids[keep_col_resids]
new_r_resids = r_resids[keep_col_resids]
f_resids = new_f_resids.values
r_resids = new_r_resids.values

for i in range(seq_len-len(f_resids)):
    f_resids=np.insert(f_resids,0,[[0,0]],axis= 0)
    r_resids=np.insert(r_resids,0,[[0,0]],axis= 0)

values_avg_f = 0.0
values_avg_r = 0.0
count_avg_f = 0
count_avg_r = 0

for i in range(0,seq_len):
    if f_resids[i][0]!= 0 or f_resids[i][1]!= 0:
        values_avg_f = values_avg_f + abs(f_resids[i][0])                                       
        count_avg_f = count_avg_f + 1
    if r_resids[i][0]!= 0 or r_resids[i][1]!= 0:
        values_avg_r = values_avg_r + abs(r_resids[i][0])                                       
        count_avg_r = count_avg_r + 1
if count_avg_f != 0:
    values_avg_f = values_avg_f/count_avg_f
    for i in range(0,seq_len):
        if f_resids[i][0]!= 0 or f_resids[i][1]!= 0:
            f_resids[i][0]= f_resids[i][0]/values_avg_f
            f_resids[i][1]= (f_resids[i][1]-f_start)/(f_over-f_start)
if count_avg_r != 0:
    values_avg_r = values_avg_r/count_avg_r
    for i in range(0,seq_len):
        if r_resids[i][0]!= 0 or r_resids[i][1]!= 0:
            r_resids[i][0]= r_resids[i][0]/values_avg_r
            r_resids[i][1]= (r_resids[i][1]-r_start)/(r_over-r_start)

test_f = f_resids.reshape(-1,seq_len,2,1)
test_r = r_resids.reshape(-1,seq_len,2,1)

test_preds_record_f = []
test_preds_record_r = []

for i in range(1,kk+1):

    model_name = '../../dl_model/best_model_fold_white_{}.pkl'.format(i)

    model = load_model(model_name)

    test_preds_f = model.predict(test_f)
    test_preds_f = test_preds_f * (f_over-f_start) + f_start

    test_preds_r = model.predict(test_r)
    test_preds_r = test_preds_r * (r_over-r_start) + r_start

    test_preds_record_f.append(test_preds_f)
    test_preds_record_r.append(test_preds_r)

test_preds_record_f = np.array(test_preds_record_f)
test_preds_record_r = np.array(test_preds_record_r)

preds_f = np.mean(test_preds_record_f)
preds_r = np.mean(test_preds_record_r)

distance_f = abs(f_trans - f_over)
distance_r = abs(r_trans - r_over)
error_f = abs(preds_f - f_trans)
error_r = abs(preds_r - r_trans)

relative_ef = error_f/distance_f
relative_er = error_r/distance_r

preds_results = {'preds_f':preds_f,'preds_r':preds_r,'relative_ef':relative_ef,'relative_er':relative_er,
                 'f_start':f_start,'f_over':f_over,'r_start':r_start,'r_over':r_over}
preds_results = pandas.DataFrame(preds_results,index=[0])
preds_results.to_csv('../../model_results/sleep-wake.csv',header = True)








