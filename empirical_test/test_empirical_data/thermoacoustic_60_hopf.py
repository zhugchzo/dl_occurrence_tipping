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

kk = 10
seq_len = 500
tp_point = 1.87

sample_start = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8]

ss_list = []

preds_dl_list = []
preds_ac_list = []
preds_dev_list = []

relative_dl_list = []
relative_ac_list = []
relative_dev_list = []

for ss in sample_start:

    df_resids = pandas.read_csv('../data_nus/thermoacoustic_60_hopf/{}/thermoacoustic_60_hopf_resids.csv'.format(ss))
    keep_col_resids = ['residuals','b']
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
        for i in range (0,seq_len):
            if values_resids[i][0]!= 0 or values_resids[i][1]!= 0:
                values_resids[i][0]= values_resids[i][0]/values_avg
                values_resids[i][1]= (values_resids[i][1]-values_s)/(values_o-values_s)

    test = values_resids.reshape(-1,seq_len,2,1)

    test_preds_record = []

    for i in range(1,kk+1):
        
        model_name = '../../dl_model/best_model_{}.keras'.format(i)

        model = load_model(model_name)

        test_preds = model.predict(test)

        test_preds_record.append(test_preds)

    test_preds_record = np.array(test_preds_record)

    preds_dl = np.mean(test_preds_record) * (values_o-values_s) + values_s
    relative_dl = abs((preds_dl - tp_point) / (values_o - tp_point))

    df_ac = pandas.read_csv('../data_nus/thermoacoustic_60_hopf/{}/thermoacoustic_60_hopf_ac.csv'.format(ss))

    # fit linear curve for AC
    k,b = np.polyfit(df_ac['b'].values,df_ac['ac1'].values,1)
    preds_ac_1 = (1-b)/k

    # fit curve for AC
    a,b,c = np.polyfit(df_ac['b'].values,df_ac['ac1'].values,2)

    if b**2-4*a*(c-1)<0:
        preds_ac = preds_ac_1
    else:
        preds_ac_2 = (-b + np.sqrt(b**2-4*a*(c-1)))/(2*a)

        if abs(preds_ac_1 - tp_point) >= abs(preds_ac_2 - tp_point):
            preds_ac = preds_ac_2
        else:
            preds_ac = preds_ac_1

    relative_ac = abs((preds_ac - tp_point) / (values_o - tp_point))

    df_dev = pandas.read_csv('../data_nus/thermoacoustic_60_hopf/{}/thermoacoustic_60_hopf_dev.csv'.format(ss))

    # fit linear curve for dev
    k,b = np.polyfit(df_dev['b'].values,df_dev['DEV'].values,1)
    preds_dev_1 = (1-b)/k

    # fit curve for dev
    a,b,c = np.polyfit(df_dev['b'].values,df_dev['DEV'].values,2)

    if b**2-4*a*(c-1)<0:
        preds_dev = preds_dev_1
    else:
        preds_dev_2 = (-b + np.sqrt(b**2-4*a*(c-1)))/(2*a)

        if abs(preds_dev_1 - tp_point) >= abs(preds_dev_2 - tp_point):
            preds_dev = preds_dev_2
        else:
            preds_dev = preds_dev_1

    relative_dev = abs((preds_dev - tp_point) / (values_o - tp_point))

    ss_list.append(ss)

    preds_dl_list.append(preds_dl)
    preds_ac_list.append(preds_ac)
    preds_dev_list.append(preds_dev)

    relative_dl_list.append(relative_dl)
    relative_ac_list.append(relative_ac)
    relative_dev_list.append(relative_dev)

preds_results = {'ss_list':ss_list, 'preds_dl_list':preds_dl_list, 'relative_dl_list':relative_dl_list,
                 'preds_ac_list':preds_ac_list, 'relative_ac_list':relative_ac_list,
                 'preds_dev_list':preds_dev_list, 'relative_dev_list':relative_dev_list}

preds_results = pandas.DataFrame(preds_results)
preds_results.to_csv('../../results/thermoacoustic_60_hopf.csv',header = True)




    






