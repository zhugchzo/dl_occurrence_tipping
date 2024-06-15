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

kk = 5

seq_len = 500

preds_l_1 = []
preds_l_20 = []
preds_l_21 = []
preds_l_22 = []
preds_l_3 = []
preds_l_4 = []

microcosm_par_range_list = ['0-15','1-16','2-17','3-17.5']

for p in range(len(microcosm_par_range_list)):

    par_range = microcosm_par_range_list[p]

    df_dev = pandas.read_csv('../data_nus/microcosm_fold/microcosm_fold_400_dev_{}.csv'.format(par_range))
    # fit curve for AC
    k,b = np.polyfit(df_dev['b'].values,df_dev['DEV'].values,1)
    pred_dev = (1-b)/k
    preds_l_1.append(pred_dev)

thermoacoustic_par_range_list = ['0-1','0.05-1.05','0.1-1.1','0.15-1.15']

for p in range(len(thermoacoustic_par_range_list)):

    par_range = thermoacoustic_par_range_list[p]
 
    df_dev_0 = pandas.read_csv('../data_nus/thermoacoustic_hopf/thermoacoustic_20_hopf/thermoacoustic_20_hopf_400_dev_{}.csv'.format(par_range))
    # fit curve for AC
    k_0,b_0 = np.polyfit(df_dev_0['b'].values,df_dev_0['DEV'].values,1)
    pred_dev_0 = (1-b_0)/k_0
    preds_l_20.append(pred_dev_0)
 
    df_dev_1 = pandas.read_csv('../data_nus/thermoacoustic_hopf/thermoacoustic_40_hopf/thermoacoustic_40_hopf_400_dev_{}.csv'.format(par_range))
    # fit curve for AC
    k_1,b_1 = np.polyfit(df_dev_1['b'].values,df_dev_1['DEV'].values,1)
    pred_dev_1 = (1-b_1)/k_1
    preds_l_21.append(pred_dev_1)

    df_dev_2 = pandas.read_csv('../data_nus/thermoacoustic_hopf/thermoacoustic_60_hopf/thermoacoustic_60_hopf_400_dev_{}.csv'.format(par_range))
    # fit curve for AC
    k_2,b_2 = np.polyfit(df_dev_2['b'].values,df_dev_2['DEV'].values,1)
    pred_dev_2 = (1-b_2)/k_2
    preds_l_22.append(pred_dev_2)

Mo_par_range_list = ['160-142','159-141','158-140','157-139']

for p in range(len(Mo_par_range_list)):

    par_range = Mo_par_range_list[p]

    df_dev = pandas.read_csv('../data_nus/hypoxia_64PE_Mo_fold/hypoxia_64PE_Mo_fold_400_dev_{}.csv'.format(par_range))
    # fit curve for AC
    k,b = np.polyfit(df_dev['b'].values,df_dev['DEV'].values,1)
    pred_dev = (1-b)/k
    preds_l_3.append(-pred_dev)

U_par_range_list = ['300-268','298-266','296-264','294-262']

for p in range(len(U_par_range_list)):

    par_range = U_par_range_list[p]

    df_dev = pandas.read_csv('../data_nus/hypoxia_64PE_U_branch/hypoxia_64PE_U_branch_400_dev_{}.csv'.format(par_range))
    # fit curve for AC
    k,b = np.polyfit(df_dev['b'].values,df_dev['DEV'].values,1)
    pred_dev = (1-b)/k
    preds_l_4.append(-pred_dev)

dic_preds = {'preds_1':preds_l_1,
             'preds_20':preds_l_20,
             'preds_21':preds_l_21,
             'preds_22':preds_l_22,
             'preds_3':preds_l_3,
             'preds_4':preds_l_4}

csv_out = pandas.DataFrame(dic_preds)
csv_out.to_csv('../../results/empirical_nus_results_dev_1.csv',header = True)


preds_l_1 = []
preds_l_20 = []
preds_l_21 = []
preds_l_22 = []
preds_l_3 = []
preds_l_4 = []

microcosm_par_range_list = ['0-15','1-16','2-17','3-17.5']

for p in range(len(microcosm_par_range_list)):

    par_range = microcosm_par_range_list[p]

    df_dev = pandas.read_csv('../data_nus/microcosm_fold/microcosm_fold_400_dev_{}.csv'.format(par_range))
    # fit curve for AC
    a,b,c = np.polyfit(df_dev['b'].values,df_dev['DEV'].values,2)

    if b**2-4*a*(c-1)<0:
        pred_dev = -b/(2*a)
    else:
        pred_dev = (-b + np.sqrt(b**2-4*a*(c-1)))/(2*a)

    preds_l_1.append(pred_dev)

thermoacoustic_par_range_list = ['0-1','0.05-1.05','0.1-1.1','0.15-1.15']

for p in range(len(thermoacoustic_par_range_list)):

    par_range = thermoacoustic_par_range_list[p]
 
    df_dev_0 = pandas.read_csv('../data_nus/thermoacoustic_hopf/thermoacoustic_20_hopf/thermoacoustic_20_hopf_400_dev_{}.csv'.format(par_range))
    # fit curve for AC
    a_0,b_0,c_0 = np.polyfit(df_dev_0['b'].values,df_dev_0['DEV'].values,2)

    if b_0**2-4*a_0*(c_0-1)<0:
        pred_dev_0 = -b_0/(2*a_0)
    else:
        pred_dev_0 = (-b_0 + np.sqrt(b_0**2-4*a_0*(c_0-1)))/(2*a_0)

    preds_l_20.append(pred_dev_0)
 
    df_dev_1 = pandas.read_csv('../data_nus/thermoacoustic_hopf/thermoacoustic_40_hopf/thermoacoustic_40_hopf_400_dev_{}.csv'.format(par_range))
    # fit curve for AC
    a_1,b_1,c_1 = np.polyfit(df_dev_1['b'].values,df_dev_1['DEV'].values,2)

    if b_1**2-4*a_1*(c_1-1)<0:
        pred_dev_1 = -b_1/(2*a_1)
    else:
        pred_dev_1 = (-b_1 + np.sqrt(b_1**2-4*a_1*(c_1-1)))/(2*a_1)

    preds_l_21.append(pred_dev_1)

    df_dev_2 = pandas.read_csv('../data_nus/thermoacoustic_hopf/thermoacoustic_60_hopf/thermoacoustic_60_hopf_400_dev_{}.csv'.format(par_range))
    # fit curve for AC
    a_2,b_2,c_2 = np.polyfit(df_dev_2['b'].values,df_dev_2['DEV'].values,2)

    if b_2**2-4*a_2*(c_2-1)<0:
        pred_dev_2 = -b_2/(2*a_2)
    else:
        pred_dev_2 = (-b_2 + np.sqrt(b_2**2-4*a_2*(c_2-1)))/(2*a_2)

    preds_l_22.append(pred_dev_2)

Mo_par_range_list = ['160-142','159-141','158-140','157-139']

for p in range(len(Mo_par_range_list)):

    par_range = Mo_par_range_list[p]

    df_dev = pandas.read_csv('../data_nus/hypoxia_64PE_Mo_fold/hypoxia_64PE_Mo_fold_400_dev_{}.csv'.format(par_range))
    # fit curve for AC
    a,b,c = np.polyfit(df_dev['b'].values,df_dev['DEV'].values,2)

    if b**2-4*a*(c-1)<0:
        pred_dev = -b/(2*a)
    else:
        pred_dev = (-b + np.sqrt(b**2-4*a*(c-1)))/(2*a)
    
    preds_l_3.append(-pred_dev)

U_par_range_list = ['300-268','298-266','296-264','294-262']

for p in range(len(U_par_range_list)):

    par_range = U_par_range_list[p]

    df_dev = pandas.read_csv('../data_nus/hypoxia_64PE_U_branch/hypoxia_64PE_U_branch_400_dev_{}.csv'.format(par_range))
    # fit curve for AC
    a,b,c = np.polyfit(df_dev['b'].values,df_dev['DEV'].values,2)

    if b**2-4*a*(c-1)<0:
        pred_dev = -b/(2*a)
    else:
        pred_dev = (-b + np.sqrt(b**2-4*a*(c-1)))/(2*a)
        
    preds_l_4.append(-pred_dev)

dic_preds = {'preds_1':preds_l_1,
             'preds_20':preds_l_20,
             'preds_21':preds_l_21,
             'preds_22':preds_l_22,
             'preds_3':preds_l_3,
             'preds_4':preds_l_4}

csv_out = pandas.DataFrame(dic_preds)
csv_out.to_csv('../../results/empirical_nus_results_dev_2.csv',header = True)