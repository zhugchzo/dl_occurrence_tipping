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

microcosm_par_range_list = ['0-14','1-15','2-16','3-17']

for p in range(len(microcosm_par_range_list)):

    par_range = microcosm_par_range_list[p]

    df_resids = pandas.read_csv('../data_nus/microcosm_fold/microcosm_fold_400_resids_{}.csv'.format(par_range))
    keep_col_resids = ['Residuals','b']
    new_f_resids = df_resids[keep_col_resids]
    values_resids = new_f_resids.values

    values_s = df_resids['b'].iloc[0]
    values_o = df_resids['b'].iloc[-1]

    # Padding input sequences for DL
    for j in range(seq_len-len(values_resids)):
        values_resids=np.insert(values_resids,0,[[0,0]],axis= 0)
            

    # normalizing input time series by the average. 
    values_avg = 0.0
    count_avg = 0
    for i in range (0,seq_len):
        if values_resids[i][0]!= 0 or values_resids[i][1]!= 0:
            values_avg = values_avg + abs(values_resids[i][0])                                       
            count_avg = count_avg + 1
    if count_avg != 0:
        values_avg = values_avg/count_avg
        for j in range (0,seq_len):
            if values_resids[j][0]!= 0 or values_resids[j][1]!= 0:
                values_resids[j][0]= values_resids[j][0]/values_avg
                values_resids[j][1]= (values_resids[j][1]-values_s)/(values_o-values_s)

    test = values_resids
    test = np.array(test)
    test = test.reshape(-1,seq_len,2,1)

    test_preds_record = []

    for i in range(1,kk+1):

        model_name = '../../dl_combined_model/best_model_combined_white_{}.pkl'.format(i)

        model = load_model(model_name)

        test_preds = model.predict(test)
        test_preds = test_preds * (values_o-values_s) + values_s

        test_preds_record.append(test_preds)


    test_preds_record = np.array(test_preds_record)

    preds = np.mean(test_preds_record)
    preds_l_1.append(preds)

thermoacoustic_par_range_list = ['0-1','0.05-1.05','0.1-1.1','0.15-1.15']

for p in range(len(thermoacoustic_par_range_list)):

    par_range = thermoacoustic_par_range_list[p]
 
    df_resids_0 = pandas.read_csv('../data_nus/thermoacoustic_hopf/thermoacoustic_20_hopf/thermoacoustic_20_hopf_400_resids_{}.csv'.format(par_range))
    df_resids_1 = pandas.read_csv('../data_nus/thermoacoustic_hopf/thermoacoustic_40_hopf/thermoacoustic_40_hopf_400_resids_{}.csv'.format(par_range))
    df_resids_2 = pandas.read_csv('../data_nus/thermoacoustic_hopf/thermoacoustic_60_hopf/thermoacoustic_60_hopf_400_resids_{}.csv'.format(par_range))
    keep_col_resids = ['Residuals','b']
    new_f_resids_0 = df_resids_0[keep_col_resids]
    values_resids_0 = new_f_resids_0.values
    new_f_resids_1 = df_resids_1[keep_col_resids]
    values_resids_1 = new_f_resids_1.values
    new_f_resids_2 = df_resids_2[keep_col_resids]
    values_resids_2 = new_f_resids_2.values

    values_s_0 = df_resids_0['b'].iloc[0]
    values_o_0 = df_resids_0['b'].iloc[-1]
    values_s_1 = df_resids_1['b'].iloc[0]
    values_o_1 = df_resids_1['b'].iloc[-1]
    values_s_2 = df_resids_2['b'].iloc[0]
    values_o_2 = df_resids_2['b'].iloc[-1]

    # Padding input sequences for DL
    for j in range(seq_len-len(values_resids_1)):
        values_resids_0=np.insert(values_resids_0,0,[[0,0]],axis= 0)
        values_resids_1=np.insert(values_resids_1,0,[[0,0]],axis= 0)
        values_resids_2=np.insert(values_resids_2,0,[[0,0]],axis= 0)
            

    # normalizing input time series by the average. 
    values_avg_0 = 0.0
    count_avg_0 = 0
    for i in range (0,seq_len):
        if values_resids_0[i][0]!= 0 or values_resids_0[i][1]!= 0:
            values_avg_0 = values_avg_0 + abs(values_resids_0[i][0])                                       
            count_avg_0 = count_avg_0 + 1
    if count_avg_0 != 0:
        values_avg_0 = values_avg_0/count_avg_0
        for j in range (0,seq_len):
            if values_resids_0[j][0]!= 0 or values_resids_0[j][1]!= 0:
                values_resids_0[j][0]= values_resids_0[j][0]/values_avg_0
                values_resids_0[j][1]= (values_resids_0[j][1]-values_s_0)/(values_o_0-values_s_0)


    values_avg_1 = 0.0
    count_avg_1 = 0
    for i in range (0,seq_len):
        if values_resids_1[i][0]!= 0 or values_resids_1[i][1]!= 0:
            values_avg_1 = values_avg_1 + abs(values_resids_1[i][0])                                       
            count_avg_1 = count_avg_1 + 1
    if count_avg_1 != 0:
        values_avg_1 = values_avg_1/count_avg_1
        for j in range (0,seq_len):
            if values_resids_1[j][0]!= 0 or values_resids_1[j][1]!= 0:
                values_resids_1[j][0]= values_resids_1[j][0]/values_avg_1
                values_resids_1[j][1]= (values_resids_1[j][1]-values_s_1)/(values_o_1-values_s_1)


    values_avg_2 = 0.0
    count_avg_2 = 0
    for i in range (0,seq_len):
        if values_resids_2[i][0]!= 0 or values_resids_2[i][1]!= 0:
            values_avg_2 = values_avg_2 + abs(values_resids_2[i][0])                                       
            count_avg_2 = count_avg_2 + 1
    if count_avg_2 != 0:
        values_avg_2 = values_avg_2/count_avg_2
        for j in range (0,seq_len):
            if values_resids_2[j][0]!= 0 or values_resids_2[j][1]!= 0:
                values_resids_2[j][0]= values_resids_2[j][0]/values_avg_2
                values_resids_2[j][1]= (values_resids_2[j][1]-values_s_2)/(values_o_2-values_s_2)

    test_0 = values_resids_0
    test_0 = np.array(test_0)
    test_0 = test_0.reshape(-1,seq_len,2,1)

    test_1 = values_resids_1
    test_1 = np.array(test_1)
    test_1 = test_1.reshape(-1,seq_len,2,1)

    test_2 = values_resids_2
    test_2 = np.array(test_2)
    test_2 = test_2.reshape(-1,seq_len,2,1)

    test_preds_record_0 = []
    test_preds_record_1 = []
    test_preds_record_2 = []

    for i in range(1,kk+1):

        model_name = '../../dl_combined_model/best_model_combined_white_{}.pkl'.format(i)

        model = load_model(model_name)

        test_preds_0 = model.predict(test_0)
        test_preds_1 = model.predict(test_1)
        test_preds_2 = model.predict(test_2)

        test_preds_0 = test_preds_0 * (values_o_0-values_s_0) + values_s_0
        test_preds_1 = test_preds_1 * (values_o_1-values_s_1) + values_s_1
        test_preds_2 = test_preds_2 * (values_o_2-values_s_2) + values_s_2

        test_preds_record_0.append(test_preds_0)
        test_preds_record_1.append(test_preds_1)
        test_preds_record_2.append(test_preds_2)

    test_preds_record_0 = np.array(test_preds_record_0)
    test_preds_record_1 = np.array(test_preds_record_1)
    test_preds_record_2 = np.array(test_preds_record_2)

    preds_0 = np.mean(test_preds_record_0)
    preds_1 = np.mean(test_preds_record_1)
    preds_2 = np.mean(test_preds_record_2)

    preds_l_20.append(preds_0)
    preds_l_21.append(preds_1)
    preds_l_22.append(preds_2)

Mo_par_range_list = ['160-142','159-141','158-140','157-139']

for p in range(len(Mo_par_range_list)):

    par_range = Mo_par_range_list[p]

    df_resids = pandas.read_csv('../data_nus/hypoxia_64PE_Mo_fold/hypoxia_64PE_Mo_fold_400_resids_{}.csv'.format(par_range))
    keep_col_resids = ['Residuals','b']
    new_f_resids = df_resids[keep_col_resids]
    values_resids = new_f_resids.values

    values_s = df_resids['b'].iloc[0]
    values_o = df_resids['b'].iloc[-1]

    # Padding input sequences for DL
    for j in range(seq_len-len(values_resids)):
        values_resids=np.insert(values_resids,0,[[0,0]],axis= 0)
            

    # normalizing input time series by the average. 
    values_avg = 0.0
    count_avg = 0
    for i in range (0,seq_len):
        if values_resids[i][0]!= 0 or values_resids[i][1]!= 0:
            values_avg = values_avg + abs(values_resids[i][0])                                       
            count_avg = count_avg + 1
    if count_avg != 0:
        values_avg = values_avg/count_avg
        for j in range (0,seq_len):
            if values_resids[j][0]!= 0 or values_resids[j][1]!= 0:
                values_resids[j][0]= values_resids[j][0]/values_avg
                values_resids[j][1]= (values_resids[j][1]-values_s)/(values_o-values_s)

    test = values_resids
    test = np.array(test)
    test = test.reshape(-1,seq_len,2,1)

    test_preds_record = []

    for i in range(1,kk+1):

        model_name = '../../dl_combined_model/best_model_combined_red_{}.pkl'.format(i)

        model = load_model(model_name)

        test_preds = model.predict(test)
        test_preds = test_preds * (values_o-values_s) + values_s

        test_preds_record.append(test_preds)


    test_preds_record = np.array(test_preds_record)

    preds = -np.mean(test_preds_record)
    preds_l_3.append(preds)

U_par_range_list = ['300-268','298-266','296-264','294-262']

for p in range(len(U_par_range_list)):

    par_range = U_par_range_list[p]

    df_resids = pandas.read_csv('../data_nus/hypoxia_64PE_U_branch/hypoxia_64PE_U_branch_400_resids_{}.csv'.format(par_range))
    keep_col_resids = ['Residuals','b']
    new_f_resids = df_resids[keep_col_resids]
    values_resids = new_f_resids.values

    values_s = df_resids['b'].iloc[0]
    values_o = df_resids['b'].iloc[-1]

    # Padding input sequences for DL
    for j in range(seq_len-len(values_resids)):
        values_resids=np.insert(values_resids,0,[[0,0]],axis= 0)
            

    # normalizing input time series by the average. 
    values_avg = 0.0
    count_avg = 0
    for i in range (0,seq_len):
        if values_resids[i][0]!= 0 or values_resids[i][1]!= 0:
            values_avg = values_avg + abs(values_resids[i][0])                                       
            count_avg = count_avg + 1
    if count_avg != 0:
        values_avg = values_avg/count_avg
        for j in range (0,seq_len):
            if values_resids[j][0]!= 0 or values_resids[j][1]!= 0:
                values_resids[j][0]= values_resids[j][0]/values_avg
                values_resids[j][1]= (values_resids[j][1]-values_s)/(values_o-values_s)

    test = values_resids
    test = np.array(test)
    test = test.reshape(-1,seq_len,2,1)

    test_preds_record = []

    for i in range(1,kk+1):

        model_name = '../../dl_combined_model/best_model_combined_red_{}.pkl'.format(i)

        model = load_model(model_name)

        test_preds = model.predict(test)
        test_preds = test_preds * (values_o-values_s) + values_s

        test_preds_record.append(test_preds)


    test_preds_record = np.array(test_preds_record)

    preds = -np.mean(test_preds_record)
    preds_l_4.append(preds)

dic_preds = {'preds_1':preds_l_1,
             'preds_20':preds_l_20,
             'preds_21':preds_l_21,
             'preds_22':preds_l_22,
             'preds_3':preds_l_3,
             'preds_4':preds_l_4}

csv_out = pandas.DataFrame(dic_preds)
csv_out.to_csv('../../results/empirical_nus_results_combined.csv',header = True)