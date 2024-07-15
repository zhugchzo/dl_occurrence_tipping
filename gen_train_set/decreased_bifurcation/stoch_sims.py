#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on May 4 2024

@author: Chengzuo Zhuge

Modified by: Thomas M. Bury

Script to:
    Get info from b.out files (AUTO files)
    Run stochastic simulations up to bifurcation points and irregularly sample Uniform(600,1000) time units
    Ouptut time-series of 500 time units prior to transition

"""

import numpy as np
import pandas as pd
import csv
import os

# Function to convert b.out files into readable form
from convert_bifdata import convert_bifdata 
    
# Function to simulate model
from sim_model import sim_model

# Create model class
class Model():
    pass

# Get command line variables
import sys
hopf_count = int(sys.argv[1])
fold_count = int(sys.argv[2])
branch_count = int(sys.argv[3])
bif_max = int(sys.argv[4])
batch_num = int(sys.argv[5])
ts_len = int(sys.argv[6])
same_num = 5


# Create directory if does not exist
if not os.path.exists('hopf/output_sims'):
    os.makedirs('hopf/output_sims')
if not os.path.exists('hopf/output_values'):
    os.makedirs('hopf/output_values')
if not os.path.exists('fold/output_sims'):
    os.makedirs('fold/output_sims')
if not os.path.exists('fold/output_values'):
    os.makedirs('fold/output_values')
if not os.path.exists('branch/output_sims'):
    os.makedirs('branch/output_sims')
if not os.path.exists('branch/output_values'):
    os.makedirs('branch/output_values')
if not os.path.exists('output_counts'):
    os.makedirs('output_counts')



# Noise amplitude
sigma_tilde = 0.01
print('Using sigma_tilde value of {}'.format(sigma_tilde))


# Total count of bifurcations in this batch
total_count = hopf_count + fold_count + branch_count + 1    
# Corresponding ID for naming time series file
hopf_seq_id = hopf_count + (batch_num-1)*bif_max + 1
fold_seq_id = fold_count + (batch_num-1)*bif_max + 1
branch_seq_id = branch_count + (batch_num-1)*bif_max + 1

# Parameter labels
parlabels_a = ['a'+str(i) for i in np.arange(1,11)]
parlabels_b = ['b'+str(i) for i in np.arange(1,11)]
parlabels = parlabels_a + parlabels_b


#----------
# Extract info from b.out files
#–-------------

# Initiate list of models
list_models = []

# Assign attributes to model objects from b.out files

for j in range(len(parlabels)):
    # Check to see if file exists
    bool_exists = os.path.exists('output_auto/b.out'+parlabels[j])
    if not bool_exists:
        continue

    model_temp = Model()
    
    out = convert_bifdata('output_auto/b.out'+parlabels[j])
    # The `convert_bifdata` function converts a model run with a bifurcation parameter using AUTO into a Python dictionary. The structure of this dictionary is as follows:
    # {
    #     'type': bif_type,       # Type of bifurcation
    #     'value': bif_val,       # Bifurcation value
    #     'bif_param': bif_param, # Bifurcation parameter, such as 'a1'
    #     'branch_vals': xVals    # Values along the equilibrium branches
    # }

        
    # Assign bifurcation properties to model object
    model_temp.bif_param = out['bif_param']
    model_temp.bif_type = out['type']
    model_temp.bif_value = out['value']
    model_temp.branch_vals = out['branch_vals']

    
    # Import parameter values for the model
    with open('output_model/pars.csv') as csvfile:
        pars_raw = list(csv.reader(csvfile))#20个参数的list
    par_list = [float(p[0]) for p in pars_raw]
    par_dict = dict(zip(parlabels,par_list))
    # Assign parameters to model object
    model_temp.pars = par_dict
    
    
    # Import equilibrium data as an array
    with open('output_model/equi.csv') as csvfile:
        equi_raw = list(csv.reader(csvfile))   
           
    equi_list = [float(e[0]) for e in equi_raw]
    equi_array = np.array(equi_list)
    # Assign equilibria to model object
    model_temp.equi_init = equi_array
 
    # Add model to list
    list_models.append(model_temp)
    

# Separate models into their bifurcation types
hb_models = [model for model in list_models if model.bif_type == 'HB']
bp_models = [model for model in list_models if model.bif_type == 'BP']
lp_models = [model for model in list_models if model.bif_type == 'LP']


#-------------------
## Simulate models
#------------------
    
# Construct noise as in Methods
rv_tri = np.random.triangular(0.75,1,1.25)
# rv_tri = 1 # temporary
sigma = sigma_tilde * rv_tri


# Only simulate bifurcation types that have count below bif_max

# Create booleans
[hopf_sim, fold_sim, branch_sim] = np.array([hopf_count, fold_count, branch_count]) < bif_max

print('Begin simulating model up to bifurcation points')
# Loop through model configurations (different bifurcation params)
for i in range(len(list_models)):
    '''
    In the `list_models`, each `model` is an object with the following attributes:
    1. `bif_param` - bifurcation parameter
    2. `bif_type` - type of bifurcation
    3. `bif_value` - bifurcation value
    4. `branch_vals` - equilibrium branches
    5. `pars` - 20 parameters of the model
    6. `equi_init` - initial equilibrium point of the model
    '''
    model = list_models[i]
        
    # Simulate a Hopf trajectory
    if hopf_sim and (model.bif_type == 'HB'):
        print('Simulating a Hopf trajectory')
        
        j = 0
        loop_count = 0
        while j < same_num:
            loop_count += 1
            # Pick relative rate randomly from [0.1,0.2,...,1]
            relative_rate = np.random.choice(np.arange(1,11)/10) #relative_rate : relative rate of change of the bifurcation parameter
            n = np.random.uniform(5,500)
            series_len = ts_len + int(n)

            df_out,trans_point_auto,stop1,stop2,stop3 = sim_model(model, relative_rate=relative_rate, series_len=series_len,
                            sigma=sigma)
            
            if loop_count == 2*same_num + 1 and j == 0:
                break

            if loop_count == 10*same_num + 1:
                hopf_count -= j
                total_count -= j
                hopf_seq_id -= j
                break  

            if stop1:
                print('   rrate Nan appears')
                hopf_count -= j
                total_count -= j
                hopf_seq_id -= j
                break

            if stop2:
                print('   Cant find rrate = 0')
                hopf_count -= j
                total_count -= j
                hopf_seq_id -= j
                break

            if stop3:
                print('   sim Nan appears')
                continue

            trans_point_rrate = df_out.loc[series_len-1,'b']
            df_cut = df_out.loc[0:ts_len-1].reset_index()
            df_cut.set_index('Time', inplace=True)
            # Export
            df_cut[['xw','b']].to_csv('hopf/output_sims/white_tseries'+str(hopf_seq_id)+'.csv')
            df_cut[['xr','b']].to_csv('hopf/output_sims/red_tseries'+str(hopf_seq_id)+'.csv')
            df_value = pd.DataFrame([1,series_len,trans_point_rrate,trans_point_auto])
            df_value.to_csv('hopf/output_values/value'+str(hopf_seq_id)+'.csv',
                            header=False, index=False)
            
            hopf_count += 1
            total_count += 1
            hopf_seq_id += 1
            j += 1
            # Allow a maximum of one Hopf bifurcation for model
            hopf_sim = False
            print('   Achieved {} steps - exporting'.format(ts_len))             

            if hopf_count == bif_max:
                break

            
    # Simulate a Fold trajectory
    if fold_sim and (model.bif_type == 'LP'):
        print('Simulating a Fold trajectory')
        
        j = 0
        loop_count = 0
        while j < same_num:
            loop_count += 1
            # Pick relative rate randomly from [0.1,0.2,...,1]
            relative_rate = np.random.choice(np.arange(1,11)/10) #relative_rate : relative rate of change of the bifurcation parameter
            n = np.random.uniform(5,500)
            series_len = ts_len + int(n)

            df_out,trans_point_auto,stop1,stop2,stop3 = sim_model(model, relative_rate=relative_rate, series_len=series_len,
                            sigma=sigma)

            if loop_count == 2*same_num + 1 and j == 0:
                break

            if loop_count == 10*same_num + 1:
                fold_count -= j
                total_count -= j
                fold_seq_id -= j
                break  

            if stop1:
                print('   rrate Nan appears')
                fold_count -= j
                total_count -= j
                fold_seq_id -= j
                break

            if stop2:
                print('   Cant find rrate = 0')
                fold_count -= j
                total_count -= j
                fold_seq_id -= j
                break

            if stop3:
                print('   sim Nan appears')
                continue         

            trans_point_rrate = df_out.loc[series_len-1,'b']
            df_cut = df_out.loc[0:ts_len-1].reset_index()
            df_cut.set_index('Time', inplace=True)
            # Export
            df_cut[['xw','b']].to_csv('fold/output_sims/white_tseries'+str(fold_seq_id)+'.csv')
            df_cut[['xr','b']].to_csv('fold/output_sims/red_tseries'+str(fold_seq_id)+'.csv')
            df_value = pd.DataFrame([0,series_len,trans_point_rrate,trans_point_auto])
            df_value.to_csv('fold/output_values/value'+str(fold_seq_id)+'.csv',
                            header=False, index=False)
            
            fold_count += 1
            total_count += 1
            fold_seq_id += 1
            j += 1
            # Allow a maximum of one fold bifurcation from model
            fold_sim = False
            print('   Achieved {} steps - exporting'.format(ts_len))

            if fold_count == bif_max:
                break       


    # Simulate a Branch point trajectory
    if branch_sim and (model.bif_type == 'BP'):
        print('Simulating a branch point')

        j = 0
        loop_count = 0
        while j < same_num:
            loop_count += 1
            # Pick relative rate randomly from [0.1,0.2,...,1]
            relative_rate = np.random.choice(np.arange(1,11)/10) #relative_rate : relative rate of change of the bifurcation parameter
            n = np.random.uniform(5,500)
            series_len = ts_len + int(n)

            df_out,trans_point_auto,stop1,stop2,stop3 = sim_model(model, relative_rate=relative_rate, series_len=series_len,
                            sigma=sigma)

            if loop_count == 2*same_num + 1 and j == 0:
                break

            if loop_count == 10*same_num + 1:
                branch_count -= j
                total_count -= j
                branch_seq_id -= j
                break  

            if stop1:
                print('   rrate Nan appears')
                branch_count -= j
                total_count -= j
                branch_seq_id -= j
                break

            if stop2:
                print('   Cant find rrate = 0')
                branch_count -= j
                total_count -= j
                branch_seq_id -= j
                break

            if stop3:
                print('   sim Nan appears')
                continue  

            trans_point_rrate = df_out.loc[series_len-1,'b']
            df_cut = df_out.loc[0:ts_len-1].reset_index()
            df_cut.set_index('Time', inplace=True)
            # Export
            df_cut[['xw','b']].to_csv('branch/output_sims/white_tseries'+str(branch_seq_id)+'.csv')
            df_cut[['xr','b']].to_csv('branch/output_sims/red_tseries'+str(branch_seq_id)+'.csv')
            df_value = pd.DataFrame([2,series_len,trans_point_rrate,trans_point_auto])
            df_value.to_csv('branch/output_values/value'+str(branch_seq_id)+'.csv',
                            header=False, index=False)
            
            branch_count += 1
            total_count += 1
            branch_seq_id += 1
            j += 1
            # Allow a maximum of one transcritical bifurcation from model
            branch_sim = False
            print('   Achieved {} steps - exporting'.format(ts_len)) 

            if branch_count == bif_max:
                break  
            

print('Simulations finished\n')
# Export updated counts of bifurcations for the bash script
list_counts = np.array([hopf_count, fold_count, branch_count])
np.savetxt('output_counts/list_counts.txt',list_counts, fmt='%i')
    









    
