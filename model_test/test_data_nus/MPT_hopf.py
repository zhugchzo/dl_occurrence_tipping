import pandas
import numpy as np
from tensorflow.keras.models import load_model
from confidence_calculate import confidence_mean
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
numtest = 20

ul_list = []

error_list_dl = []
error_list_dl_min = []
error_list_dl_max = []

error_list_dl_combined = []
error_list_dl_combined_min = []
error_list_dl_combined_max = []

error_list_dl_null = []
error_list_dl_null_min = []
error_list_dl_null_max = []

for i in np.linspace(0,0.3,11):

    ul = round(i,2)

    sequences_resids = list()
    normalization = list()

    for i in range (1,numtest+1):
        df_resids = pandas.read_csv('../data_nus/MPT_hopf_red/{}/resids/MPT_hopf_500_resids_'.format(ul)+str(i)+'.csv')
        keep_col_resids = ['Residuals','u']
        new_f_resids = df_resids[keep_col_resids]
        values_resids = new_f_resids.values

        values_s = df_resids['u'].iloc[0]
        values_o = df_resids['u'].iloc[-1]
        
        # Padding input sequences for DL
        for j in range(seq_len-len(values_resids)):
            values_resids=np.insert(values_resids,0,[[0,0]],axis= 0)
        
        sequences_resids.append(values_resids)
        normalization.append([values_s,values_o-values_s])

    sequences_resids = np.array(sequences_resids)

    # normalizing input time series by the average. 
    # 按平均值规范化输入时间串行。
    for i in range(numtest):
        values_avg = 0.0
        count_avg = 0
        for j in range (0,seq_len):
            if sequences_resids[i,j][0]!= 0 or sequences_resids[i,j][1]!= 0:
                values_avg = values_avg + abs(sequences_resids[i,j][0])                                       
                count_avg = count_avg + 1
        if count_avg != 0:
            values_avg = values_avg/count_avg
            for j in range (0,seq_len):
                if sequences_resids[i,j][0]!= 0 or sequences_resids[i,j][1]!= 0:
                    sequences_resids[i,j][0]= sequences_resids[i,j][0]/values_avg
                    sequences_resids[i,j][1]= (sequences_resids[i,j][1]-normalization[i][0])/normalization[i][1]

    test = sequences_resids
    test = np.array(test)
    test = test.reshape(-1,seq_len,2,1)

    result = pandas.read_csv('../data_nus/MPT_hopf_red/{}/MPT_hopf_500_result.csv'.format(ul),header=0)
    trans_point = result['trans point'].values
    distance = result['distance'].values
    trans_point = list(trans_point)
    distance = list(distance)

    normalization = np.array(normalization)

    test_preds_record = []
    test_preds_combined_record = []
    test_preds_null_record = []

    error_record_dl = []
    error_record_dl_combined = []
    error_record_dl_null = []

    preds = np.zeros([kk,numtest])
    preds_combined = np.zeros([kk,numtest])
    preds_null = np.zeros([kk,numtest])
    
    for i in range(1,kk+1):

        model_name = '../../dl_model/best_model_hopf_red_{}.pkl'.format(i)

        model = load_model(model_name)

        test_preds = model.predict(test)
        test_preds = test_preds.reshape(numtest)
        test_preds = test_preds * normalization[:,1] + normalization[:,0]

        test_preds_record.append(test_preds)

    for i in range(1,kk+1):

        combined_model_name = '../../dl_combined_model/best_model_combined_red_{}.pkl'.format(i)

        combined_model = load_model(combined_model_name)

        test_preds_combined = combined_model.predict(test)
        test_preds_combined = test_preds_combined.reshape(numtest)
        test_preds_combined = test_preds_combined * normalization[:,1] + normalization[:,0]

        test_preds_combined_record.append(test_preds_combined)

    for i in range(1,kk+1):

        null_model_name = '../../dl_null_model/best_model_hopf_red_null_{}.pkl'.format(i)

        null_model = load_model(null_model_name)

        test_preds_null = null_model.predict(test)
        test_preds_null = test_preds_null.reshape(numtest)
        test_preds_null = test_preds_null * normalization[:,1] + normalization[:,0]

        test_preds_null_record.append(test_preds_null)  
    
    for i in range(kk):
        recordi = test_preds_record[i]
        recordi_combined = test_preds_combined_record[i]
        recordi_null = test_preds_null_record[i]
        for j in range(numtest):
            preds[i][j] = recordi[j]
            preds_combined[i][j] = recordi_combined[j]
            preds_null[i][j] = recordi_null[j]

    preds = preds.mean(axis=0) #每个测试样本在所有模型上的平均预测结果
    preds_combined = preds_combined.mean(axis=0)
    preds_null = preds_null.mean(axis=0)

    for j in range(numtest):
        e_1 = abs(preds[j]-trans_point[j])/distance[j]
        error_record_dl.append(e_1)
        e_2 = abs(preds_combined[j]-trans_point[j])/distance[j]
        error_record_dl_combined.append(e_2)
        e_3 = abs(preds_null[j]-trans_point[j])/distance[j]
        error_record_dl_null.append(e_3)

    error_record_dl = np.array(error_record_dl)
    mean_error_dl = np.mean(error_record_dl)
    confidence_dl = confidence_mean(error_record_dl,0.05)[1]
    min_dl = min(confidence_dl)
    max_dl = max(confidence_dl)

    error_record_dl_combined = np.array(error_record_dl_combined)
    mean_error_dl_combined = np.mean(error_record_dl_combined)
    confidence_dl_combined = confidence_mean(error_record_dl_combined,0.05)[1]
    min_dl_combined = min(confidence_dl_combined)
    max_dl_combined = max(confidence_dl_combined)

    error_record_dl_null = np.array(error_record_dl_null)
    mean_error_dl_null = np.mean(error_record_dl_null)
    confidence_dl_null = confidence_mean(error_record_dl_null,0.05)[1]
    min_dl_null = min(confidence_dl_null)
    max_dl_null = max(confidence_dl_null)

    ul_list.append(ul)

    error_list_dl.append(mean_error_dl)
    error_list_dl_min.append(min_dl)
    error_list_dl_max.append(max_dl)

    error_list_dl_combined.append(mean_error_dl_combined)
    error_list_dl_combined_min.append(min_dl_combined)
    error_list_dl_combined_max.append(max_dl_combined)

    error_list_dl_null.append(mean_error_dl_null)
    error_list_dl_null_min.append(min_dl_null)
    error_list_dl_null_max.append(max_dl_null)

dic_error = {'ul':ul_list,
             'error_dl':error_list_dl,
             'min_dl':error_list_dl_min,
             'max_dl':error_list_dl_max,            
             'error_dl_combined':error_list_dl_combined,
             'min_dl_combined':error_list_dl_combined_min,
             'max_dl_combined':error_list_dl_combined_max,
             'error_dl_null':error_list_dl_null,
             'min_dl_null':error_list_dl_null_min,
             'max_dl_null':error_list_dl_null_max}

csv_out = pandas.DataFrame(dic_error)
csv_out.to_csv('../../results/MPT_hopf_nus.csv',header = True)