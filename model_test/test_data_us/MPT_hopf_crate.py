import pandas
import numpy as np
from tensorflow.keras.models import load_model
from confidence_calculate import confidence_mean
import os

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)

# Get the directory containing the current file
current_dir = os.path.dirname(current_file_path)

# Change the working directory to the directory of the current file
os.chdir(current_dir)

kk = 10
seq_len = 500
numtest = 10

dic_error = {}

for c_rate in [1e-5,2e-5,3e-5,4e-5,5e-5]:

    ul_list = []

    error_list_dl = []
    error_list_dl_min = []
    error_list_dl_max = []

    for u in np.linspace(0,0.3,11):

        ul = round(u,2)

        sequences_resids = list()
        normalization = list()

        for n in range (1,numtest+1):
            df_resids = pandas.read_csv('../data_us/MPT_hopf_red/{}/{}/resids/MPT_hopf_resids_'.format(ul,c_rate)+str(n)+'.csv')
            keep_col_resids = ['residuals','u']
            new_f_resids = df_resids[keep_col_resids]

            values_s = df_resids['u'].iloc[0]
            values_o = df_resids['u'].iloc[-1]

            values_resids = new_f_resids.values
            
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

        result = pandas.read_csv('../data_us/MPT_hopf_red/{}/{}/MPT_hopf_result.csv'.format(ul,c_rate),header=0)
        trans_point = result['trans point'].values
        distance = result['distance'].values
        trans_point = list(trans_point)
        distance = list(distance)

        normalization = np.array(normalization)

        test_preds_record = []

        error_record_dl = []

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

        ul_list.append(ul)

        error_list_dl.append(mean_error_dl)
        error_list_dl_min.append(min_dl)
        error_list_dl_max.append(max_dl)

    dic_error['error_dl_{}'.format(c_rate)] = error_list_dl
    dic_error['min_dl_{}'.format(c_rate)] = error_list_dl_min
    dic_error['max_dl_{}'.format(c_rate)] = error_list_dl_max

ul_list = ul_list[:11]
dic_error['ul'] = ul_list

csv_out = pandas.DataFrame(dic_error)
csv_out.to_csv('../../results/MPT_hopf/MPT_hopf_crate_us.csv',header = True)