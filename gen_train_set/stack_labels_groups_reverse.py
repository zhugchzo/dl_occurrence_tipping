# -*- coding: utf-8 -*-
"""
Created on May 4 2024

@author: Chengzuo Zhuge

Modified by: Thomas M. Bury

Script to stack group and label data from across batches (parameter decreasing)

"""


import numpy as np
import pandas as pd
import os
import csv
import sys


# Command line parameters
num_batches=int(sys.argv[1]) # number of batches generated
ts_len = int(sys.argv[2]) # time series length

# List of batch numbers
batch_nums=range(1,num_batches+1)

#----------------------------
# Concatenate label data
#-----------------------------

hopf_list_df_values = []
fold_list_df_values = []
branch_list_df_values = []
# Import label dataframes
#hopf
for i in batch_nums:
    filepath = 'output_reverse/ts_{}/batch{}/hopf/output_values/out_values.csv'.format(ts_len,i)
    df_values = pd.read_csv(filepath)
    hopf_list_df_values.append(df_values)

hopf_df_values = pd.concat(hopf_list_df_values).set_index('sequence_ID')

# Export
filepath='output_reverse/ts_{}/hopf/combined/'.format(ts_len)
if not os.path.exists('output_reverse/ts_{}/hopf/'.format(ts_len)):
    os.mkdir('output_reverse/ts_{}/hopf/'.format(ts_len))
if not os.path.exists(filepath):
    os.mkdir(filepath)
hopf_df_values.to_csv(filepath+'values.csv')

#fold
for i in batch_nums:
    filepath = 'output_reverse/ts_{}/batch{}/fold/output_values/out_values.csv'.format(ts_len,i)
    df_values = pd.read_csv(filepath)
    fold_list_df_values.append(df_values)

fold_df_values = pd.concat(fold_list_df_values).set_index('sequence_ID')

# Export
filepath='output_reverse/ts_{}/fold/combined/'.format(ts_len)
if not os.path.exists('output_reverse/ts_{}/fold/'.format(ts_len)):
    os.mkdir('output_reverse/ts_{}/fold/'.format(ts_len))
if not os.path.exists(filepath):
    os.mkdir(filepath)
fold_df_values.to_csv(filepath+'values.csv')

#branch
for i in batch_nums:
    filepath = 'output_reverse/ts_{}/batch{}/branch/output_values/out_values.csv'.format(ts_len,i)
    df_values = pd.read_csv(filepath)
    branch_list_df_values.append(df_values)

branch_df_values = pd.concat(branch_list_df_values).set_index('sequence_ID')

# Export
filepath='output_reverse/ts_{}/branch/combined/'.format(ts_len)
if not os.path.exists('output_reverse/ts_{}/branch/'.format(ts_len)):
    os.mkdir('output_reverse/ts_{}/branch/'.format(ts_len))
if not os.path.exists(filepath):
    os.mkdir(filepath)
branch_df_values.to_csv(filepath+'values.csv')



#----------------------------
# Concatenate group data
#-----------------------------

hopf_list_df_groups = []
fold_list_df_groups = []
branch_list_df_groups = []
# Import label dataframes
#hopf
for i in batch_nums:
    filepath = 'output_reverse/ts_{}/batch{}/hopf/output_groups/groups.csv'.format(ts_len,i)
    df_groups_temp = pd.read_csv(filepath)
    hopf_list_df_groups.append(df_groups_temp)

hopf_df_groups = pd.concat(hopf_list_df_groups).set_index('sequence_ID')

# Export  
filepath='output_reverse/ts_{}/hopf/combined/'.format(ts_len)
hopf_df_groups.to_csv(filepath+'groups.csv')

#fold
for i in batch_nums:
    filepath = 'output_reverse/ts_{}/batch{}/fold/output_groups/groups.csv'.format(ts_len,i)
    df_groups_temp = pd.read_csv(filepath)
    fold_list_df_groups.append(df_groups_temp)

fold_df_groups = pd.concat(fold_list_df_groups).set_index('sequence_ID')

# Export  
filepath='output_reverse/ts_{}/fold/combined/'.format(ts_len)
fold_df_groups.to_csv(filepath+'groups.csv')

#branch
for i in batch_nums:
    filepath = 'output_reverse/ts_{}/batch{}/branch/output_groups/groups.csv'.format(ts_len,i)
    df_groups_temp = pd.read_csv(filepath)
    branch_list_df_groups.append(df_groups_temp)

branch_df_groups = pd.concat(branch_list_df_groups).set_index('sequence_ID')

# Export  
filepath='output_reverse/ts_{}/branch/combined/'.format(ts_len)
branch_df_groups.to_csv(filepath+'groups.csv')









