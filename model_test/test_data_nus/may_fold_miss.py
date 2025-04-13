import pandas
import numpy as np
from tensorflow.keras.models import load_model
from confidence_calculate import confidence_mean
import os

kk = 10
seq_len = 500
numtest = 10

if not os.path.exists('../../results/data_miss'):
    os.makedirs('../../results/data_miss')

dr_list = []

dr_error_list_dl = []
dr_error_list_dl_min = []
dr_error_list_dl_max = []

for delete_rate in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:

    error_record_dl = list()

    for c_rate in [1e-5,2e-5,3e-5,4e-5,5e-5]:

        sequences_resids = list()
        sequences_ac = list()
        sequences_bac = list()
        sequences_dev = list()
        sequences_bdev = list()
        normalization = list()

        for n in range(1,numtest+1):

            df_resids = pandas.read_csv('../data_nus/may_fold_white/{}/{}/resids/may_fold_resids_'.format(0.0,c_rate)+str(n)+'.csv')

            keep_col_resids = ['residuals','b']
            new_f_resids = df_resids[keep_col_resids]

            values_resids_full = new_f_resids.values

            num_rows_to_delete = int(delete_rate * values_resids_full.shape[0])

            delete_indices = np.random.choice(values_resids_full.shape[0], size=num_rows_to_delete, replace=False)

            values_resids = np.delete(values_resids_full, delete_indices, axis=0)

            values_s = values_resids[0,1]
            values_o = values_resids[-1,1]

            # Padding input sequences for DL
            for j in range(seq_len-len(values_resids)):
                values_resids=np.insert(values_resids,0,[[0,0]],axis= 0)
            
            sequences_resids.append(values_resids)
            normalization.append([values_s,values_o-values_s])

        sequences_resids = np.array(sequences_resids)

        # normalizing input time series by the average
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

        result = pandas.read_csv('../data_nus/may_fold_white/{}/{}/may_fold_result.csv'.format(0.0,c_rate),header=0)
        trans_point = result['trans point'].values
        distance = result['distance'].values
        trans_point = list(trans_point)
        distance = list(distance)

        normalization = np.array(normalization)

        test_preds_record = list()

        preds = np.zeros([kk,numtest])
        
        for i in range(1,kk+1):

            model_name = '../../dl_model/best_model_{}.keras'.format(i)

            model = load_model(model_name)

            test_preds = model.predict(test)
            test_preds = test_preds.reshape(numtest)
            test_preds = test_preds * normalization[:,1] + normalization[:,0]

            test_preds_record.append(test_preds)

        for i in range(kk):
            recordi = test_preds_record[i]
            for j in range(numtest):
                preds[i][j] = recordi[j]

        preds = preds.mean(axis=0)

        for j in range(numtest):
            e_1 = abs(preds[j]-trans_point[j])/distance[j]
            error_record_dl.append(e_1)

    error_record_dl = np.array(error_record_dl)
    mean_error_dl = np.mean(error_record_dl)
    confidence_dl = confidence_mean(error_record_dl,0.05)[1]
    min_dl = min(confidence_dl)
    max_dl = max(confidence_dl)

    dr_list.append(delete_rate)

    dr_error_list_dl.append(mean_error_dl)
    dr_error_list_dl_min.append(min_dl)
    dr_error_list_dl_max.append(max_dl)

dic_dr_error = {'dr':dr_list,
        'error_dl':dr_error_list_dl,
        'min_dl':dr_error_list_dl_min,
        'max_dl':dr_error_list_dl_max}

csv_dr_out = pandas.DataFrame(dic_dr_error)
csv_dr_out.to_csv('../../results/data_miss/may_fold_dr.csv',header = True)