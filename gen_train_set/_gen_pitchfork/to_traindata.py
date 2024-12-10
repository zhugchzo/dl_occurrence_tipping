# -*- coding: utf-8 -*-
"""
Created on May 4 2024

@author: Chengzuo Zhuge

Modified by: Thomas M. Bury

Choose a ratio for training/validation/testing here.

Script to:
    Take in all label files and output single list of labels
    Split data into a ratio for training/validation/testing

"""


import numpy as np
import pandas as pd
import os
import csv
import sys


# External arguments
bif_max = int(sys.argv[1])
batch_num = int(sys.argv[2])

#----------------------------
# Convert value files into single csv file
#-----------------------------

sup_pitchfork_list_times = []
sup_pitchfork_list_labels = []
sup_pitchfork_list_values = []

sub_pitchfork_list_times = []
sub_pitchfork_list_labels = []
sub_pitchfork_list_values = []

# Import all value files
# sup_pitchfork
for i in np.arange(bif_max)+1 + bif_max*(batch_num-1):
    filename = 'sup_pitchfork/value'+str(i)+'.csv'
    # Import value
    with open(filename) as csvfile:
        label_value = list(csv.reader(csvfile))
        time_int = int(float(label_value[0][0]))
        value_float = float(label_value[1][0])

    # Add label to list
    sup_pitchfork_list_values.append(value_float)
    sup_pitchfork_list_times.append(time_int)


# Make an array of the values
sup_pitchfork_ar_values = np.array(sup_pitchfork_list_values)
sup_pitchfork_ar_times = np.array(sup_pitchfork_list_times)

# Make an array for the indices
sup_pitchfork_ar_index = np.arange(bif_max)+1 + bif_max*(batch_num-1)

# Combine into a DataFrame for export
sup_pitchfork_df_values = pd.DataFrame({'sequence_ID':sup_pitchfork_ar_index, 'series_len':sup_pitchfork_ar_times, 'values':sup_pitchfork_ar_values})

# Export to csv
sup_pitchfork_df_values.to_csv('sup_pitchfork/out_values.csv', header=True,index=False)

# sub_pitchfork
for i in np.arange(bif_max)+1 + bif_max*(batch_num-1):
    filename = 'sub_pitchfork/value'+str(i)+'.csv'
    # Import value
    with open(filename) as csvfile:
        label_value = list(csv.reader(csvfile))
        time_int = int(float(label_value[0][0]))
        value_float = float(label_value[1][0])

    # Add label to list
    sub_pitchfork_list_values.append(value_float)
    sub_pitchfork_list_times.append(time_int)


# Make an array of the values
sub_pitchfork_ar_values = np.array(sub_pitchfork_list_values)
sub_pitchfork_ar_times = np.array(sub_pitchfork_list_times)

# Make an array for the indices
sub_pitchfork_ar_index = np.arange(bif_max)+1 + bif_max*(batch_num-1)

# Combine into a DataFrame for export
sub_pitchfork_df_values = pd.DataFrame({'sequence_ID':sub_pitchfork_ar_index, 'series_len':sub_pitchfork_ar_times, 'values':sub_pitchfork_ar_values})

# Export to csv
sub_pitchfork_df_values.to_csv('sub_pitchfork/out_values.csv', header=True,index=False)


#----------------------------
# Create groups file in ratio for training:validation:testing
#-----------------------------

# Create the file groups.csv with headers (sequence_ID, dataset_ID)
# Use numbers 1 for training, 2 for validation and 3 for testing
# Use raito 38:1:1


# Compute number of bifurcations for each group
num_valid = int(np.floor(bif_max*0.04))
num_test = int(np.floor(bif_max*0.01))
num_train = bif_max - num_valid - num_test

# Create list of group numbers
group_nums = [1]*num_train + [2]*num_valid + [3]*num_test

# Assign group numbers to each bifurcation category
sup_pitchfork_df_values['dataset_ID'] = group_nums
sub_pitchfork_df_values['dataset_ID'] = group_nums

'''
# Concatenate dataframes and select relevant columns
df_groups = pd.concat([df_sub_pitchfork,df_sup_pitchfork,df_branch])[['sequence_ID','dataset_ID']]
'''
sup_pitchfork_df_groups = sup_pitchfork_df_values[['sequence_ID','dataset_ID']].copy()
sub_pitchfork_df_groups = sub_pitchfork_df_values[['sequence_ID','dataset_ID']].copy()

# Sort rows by sequence_ID
sup_pitchfork_df_groups.sort_values(by=['sequence_ID'], inplace=True)
sub_pitchfork_df_groups.sort_values(by=['sequence_ID'], inplace=True)

# Export to csv
sup_pitchfork_df_groups.to_csv('sup_pitchfork/out_groups.csv', header=True,index=False)
sub_pitchfork_df_groups.to_csv('sub_pitchfork/out_groups.csv', header=True,index=False)





