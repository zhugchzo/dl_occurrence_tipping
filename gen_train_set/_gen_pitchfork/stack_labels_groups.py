# -*- coding: utf-8 -*-
"""
Created on May 4 2024

@author: Chengzuo Zhuge

Modified by: Thomas M. Bury

Script to stack group and label data from across batches (parameter increasing)

"""


import numpy as np
import pandas as pd
import os
import csv
import sys


# Command line parameters
num_batches=int(sys.argv[1]) # number of batches generated

# List of batch numbers
batch_nums=range(1,num_batches+1)

#----------------------------
# Concatenate label data
#-----------------------------

sup_pitchfork_list_df_values = []
sub_pitchfork_list_df_values = []

# Import label dataframes
#sup_pitchfork
for i in batch_nums:
    filepath = 'output/batch{}/sup_pitchfork/out_values.csv'.format(i)
    df_values = pd.read_csv(filepath)
    sup_pitchfork_list_df_values.append(df_values)

sup_pitchfork_df_values = pd.concat(sup_pitchfork_list_df_values).set_index('sequence_ID')

# Export
filepath='output/sup_pitchfork/combined/'
if not os.path.exists('output/sup_pitchfork/'):
    os.mkdir('output/sup_pitchfork/')
if not os.path.exists(filepath):
    os.mkdir(filepath)
sup_pitchfork_df_values.to_csv(filepath+'values.csv')

#sub_pitchfork
for i in batch_nums:
    filepath = 'output/batch{}/sub_pitchfork/out_values.csv'.format(i)
    df_values = pd.read_csv(filepath)
    sub_pitchfork_list_df_values.append(df_values)

sub_pitchfork_df_values = pd.concat(sub_pitchfork_list_df_values).set_index('sequence_ID')

# Export
filepath='output/sub_pitchfork/combined/'
if not os.path.exists('output/sub_pitchfork/'):
    os.mkdir('output/sub_pitchfork/')
if not os.path.exists(filepath):
    os.mkdir(filepath)
sub_pitchfork_df_values.to_csv(filepath+'values.csv')




#----------------------------
# Concatenate group data
#-----------------------------

sup_pitchfork_list_df_groups = []
sub_pitchfork_list_df_groups = []

# Import label dataframes
#sup_pitchfork
for i in batch_nums:
    filepath = 'output/batch{}/sup_pitchfork/out_groups.csv'.format(i)
    df_groups_temp = pd.read_csv(filepath)
    sup_pitchfork_list_df_groups.append(df_groups_temp)

sup_pitchfork_df_groups = pd.concat(sup_pitchfork_list_df_groups).set_index('sequence_ID')

# Export  
filepath='output/sup_pitchfork/combined/'
sup_pitchfork_df_groups.to_csv(filepath+'groups.csv')

#sub_pitchfork
for i in batch_nums:
    filepath = 'output/batch{}/sub_pitchfork/out_groups.csv'.format(i)
    df_groups_temp = pd.read_csv(filepath)
    sub_pitchfork_list_df_groups.append(df_groups_temp)

sub_pitchfork_df_groups = pd.concat(sub_pitchfork_list_df_groups).set_index('sequence_ID')

# Export  
filepath='output/sub_pitchfork/combined/'
sub_pitchfork_df_groups.to_csv(filepath+'groups.csv')









