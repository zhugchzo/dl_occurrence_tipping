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

hopf_list_times = []
hopf_list_labels = []
hopf_list_values_1 = []
hopf_list_values_2 = []

fold_list_times = []
fold_list_labels = []
fold_list_values_1 = []
fold_list_values_2 = []

branch_list_times = []
branch_list_labels = []
branch_list_values_1 = []
branch_list_values_2 = []
# Import all value files
# hopf
for i in np.arange(bif_max)+1 + bif_max*(batch_num-1):
    filename = 'hopf/output_values/value'+str(i)+'.csv'
    # Import value
    with open(filename) as csvfile:
        label_value = list(csv.reader(csvfile))
        label_int = int(float(label_value[0][0]))
        time_int = int(float(label_value[1][0]))
        value_float_1 = float(label_value[2][0])
        value_float_2 = float(label_value[3][0])

    # Add label to list
    hopf_list_values_1.append(value_float_1)
    hopf_list_values_2.append(value_float_2)
    hopf_list_labels.append(label_int)
    hopf_list_times.append(time_int)


# Make an array of the values
hopf_ar_values_1 = np.array(hopf_list_values_1)
hopf_ar_times = np.array(hopf_list_times)
hopf_ar_values_2 = np.array(hopf_list_values_2)
hopf_ar_labels = np.array(hopf_list_labels)

# Make an array for the indices
hopf_ar_index = np.arange(bif_max)+1 + bif_max*(batch_num-1)

# Combine into a DataFrame for export
hopf_df_values = pd.DataFrame({'sequence_ID':hopf_ar_index, 'series_len':hopf_ar_times, 'values':hopf_ar_values_1,
                                                             'bifvalues':hopf_ar_values_2, 'class_label':hopf_ar_labels})

# Export to csv
hopf_df_values.to_csv('hopf/output_values/out_values.csv', header=True,index=False)

#fold
for i in np.arange(bif_max)+1 + bif_max*(batch_num-1):
    filename = 'fold/output_values/value'+str(i)+'.csv'
    # Import value
    with open(filename) as csvfile:
        label_value = list(csv.reader(csvfile))
        label_int = int(float(label_value[0][0]))
        time_int = int(float(label_value[1][0]))
        value_float_1 = float(label_value[2][0])
        value_float_2 = float(label_value[3][0])

    # Add label to list
    fold_list_values_1.append(value_float_1)
    fold_list_values_2.append(value_float_2)
    fold_list_labels.append(label_int)
    fold_list_times.append(time_int)

# Make an array of the values
fold_ar_values_1 = np.array(fold_list_values_1)
fold_ar_values_2 = np.array(fold_list_values_2)
fold_ar_times = np.array(fold_list_times)
fold_ar_labels = np.array(fold_list_labels)

# Make an array for the indices
fold_ar_index = np.arange(bif_max)+1 + bif_max*(batch_num-1)

# Combine into a DataFrame for export
fold_df_values = pd.DataFrame({'sequence_ID':fold_ar_index, 'series_len':fold_ar_times, 'values':fold_ar_values_1,
                                                             'bifvalues':fold_ar_values_2, 'class_label':fold_ar_labels})

# Export to csv
fold_df_values.to_csv('fold/output_values/out_values.csv', header=True,index=False)

#branch
for i in np.arange(bif_max)+1 + bif_max*(batch_num-1):
    filename = 'branch/output_values/value'+str(i)+'.csv'
    # Import value
    with open(filename) as csvfile:
        label_value = list(csv.reader(csvfile))
        label_int = int(float(label_value[0][0]))
        time_int = int(float(label_value[1][0]))
        value_float_1 = float(label_value[2][0])
        value_float_2 = float(label_value[3][0])

    # Add label to list
    branch_list_values_1.append(value_float_1)
    branch_list_values_2.append(value_float_2)
    branch_list_labels.append(label_int)
    branch_list_times.append(time_int)


# Make an array of the values
branch_ar_values_1 = np.array(branch_list_values_1)
branch_ar_values_2 = np.array(branch_list_values_2)
branch_ar_times = np.array(branch_list_times)
branch_ar_labels = np.array(branch_list_labels)

# Make an array for the indices
branch_ar_index = np.arange(bif_max)+1 + bif_max*(batch_num-1)

# Combine into a DataFrame for export
branch_df_values = pd.DataFrame({'sequence_ID':branch_ar_index, 'series_len':branch_ar_times, 'values':branch_ar_values_1,
                                                                 'bifvalues':branch_ar_values_2, 'class_label':branch_ar_labels})

# Export to csv
branch_df_values.to_csv('branch/output_values/out_values.csv', header=True,index=False)


#----------------------------
# Create groups file in ratio for training:validation:testing
#-----------------------------

# Create the file groups.csv with headers (sequence_ID, dataset_ID)
# Use numbers 1 for training, 2 for validation and 3 for testing
# Use raito 38:1:1

# Make output folder to split
if not os.path.exists('hopf/output_groups'):
    os.makedirs('hopf/output_groups')
if not os.path.exists('fold/output_groups'):
    os.makedirs('fold/output_groups')
if not os.path.exists('branch/output_groups'):
    os.makedirs('branch/output_groups')
    
'''
# Collect Fold bifurcations (label 0)
df_fold = df_values[df_values['class_label']==0].copy()
# Collect Hopf bifurcations (label 1)
df_hopf = df_values[df_values['class_label']==1].copy()
# Collect Branch points (label 2)
df_branch = df_values[df_values['class_label']==2].copy()


# Check they all have the same length
assert len(df_fold) == len(df_hopf)
assert len(df_hopf) == len(df_branch)
assert len(df_branch) == len(df_fold)
'''


# Compute number of bifurcations for each group
num_valid = int(np.floor(bif_max*0.04))
num_test = int(np.floor(bif_max*0.01))
num_train = bif_max - num_valid - num_test

# Create list of group numbers
group_nums = [1]*num_train + [2]*num_valid + [3]*num_test

# Assign group numbers to each bifurcation category
hopf_df_values['dataset_ID'] = group_nums
fold_df_values['dataset_ID'] = group_nums
branch_df_values['dataset_ID'] = group_nums

'''
# Concatenate dataframes and select relevant columns
df_groups = pd.concat([df_fold,df_hopf,df_branch])[['sequence_ID','dataset_ID']]
'''
hopf_df_groups = hopf_df_values[['sequence_ID','dataset_ID']].copy()
fold_df_groups = fold_df_values[['sequence_ID','dataset_ID']].copy()
branch_df_groups = branch_df_values[['sequence_ID','dataset_ID']].copy()
# Sort rows by sequence_ID
hopf_df_groups.sort_values(by=['sequence_ID'], inplace=True)
fold_df_groups.sort_values(by=['sequence_ID'], inplace=True)
branch_df_groups.sort_values(by=['sequence_ID'], inplace=True)
# Export to csv
hopf_df_groups.to_csv('hopf/output_groups/groups.csv', header=True,index=False)
fold_df_groups.to_csv('fold/output_groups/groups.csv', header=True,index=False)
branch_df_groups.to_csv('branch/output_groups/groups.csv', header=True,index=False)




