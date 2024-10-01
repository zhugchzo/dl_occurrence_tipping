import pandas
import numpy as np
from tensorflow.keras.models import load_model
from confidence_calculate import confidence_mean

kk = 10
seq_len = 500
numtest = 50

rl_list = []

error_list_dl_sup = []
error_list_dl_sup_min = []
error_list_dl_sup_max = []

error_list_dl_sub = []
error_list_dl_sub_min = []
error_list_dl_sub_max = []

for i in np.linspace(0,0.2,11):

    rl = round(i,2)

    sequences_resids = list()
    normalization = list()

    for i in range (1,numtest+1):
        df_resids = pandas.read_csv('../data_nus/pitchfork_white/{}/resids/pitchfork_resids_'.format(rl)+str(i)+'.csv')
        keep_col_resids = ['residuals','r']
        new_f_resids = df_resids[keep_col_resids]

        values_s = df_resids['r'].iloc[0]
        values_o = df_resids['r'].iloc[-1]

        values_resids = new_f_resids.values
        
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

    result = pandas.read_csv('../data_nus/pitchfork_white/{}/pitchfork_result.csv'.format(rl),header=0)
    trans_point = result['trans point'].values
    distance = result['distance'].values
    trans_point = list(trans_point)
    distance = list(distance)

    normalization = np.array(normalization)

    test_preds_sup_record = []
    test_preds_sub_record = []

    error_record_dl_sup = []
    error_record_dl_sub = []

    preds_sup = np.zeros([kk,numtest])
    preds_sub = np.zeros([kk,numtest])
    
    for i in range(1,kk+1):

        sup_model_name = '../../dl_model_SI/best_model_suppf_{}.keras'.format(i)

        sup_model = load_model(sup_model_name)

        test_preds_sup = sup_model.predict(test)
        test_preds_sup = test_preds_sup.reshape(numtest)
        test_preds_sup = test_preds_sup * normalization[:,1] + normalization[:,0]

        test_preds_sup_record.append(test_preds_sup)

    for i in range(1,kk+1):

        sub_model_name = '../../dl_model_SI/best_model_subpf_{}.keras'.format(i)

        sub_model = load_model(sub_model_name)

        test_preds_sub = sub_model.predict(test)
        test_preds_sub = test_preds_sub.reshape(numtest)
        test_preds_sub = test_preds_sub * normalization[:,1] + normalization[:,0]

        test_preds_sub_record.append(test_preds_sub)  
    
    for i in range(kk):
        recordi_sup = test_preds_sup_record[i]
        recordi_sub = test_preds_sub_record[i]
        for j in range(numtest):
            preds_sup[i][j] = recordi_sup[j]
            preds_sub[i][j] = recordi_sub[j]

    preds_sup = preds_sup.mean(axis=0)
    preds_sub = preds_sub.mean(axis=0)

    for j in range(numtest):
        e_1 = abs(preds_sup[j]-trans_point[j])/distance[j]
        error_record_dl_sup.append(e_1)

        e_2 = abs(preds_sub[j]-trans_point[j])/distance[j]
        error_record_dl_sub.append(e_2)

    error_record_dl_sup = np.array(error_record_dl_sup)
    mean_error_dl_sup = np.mean(error_record_dl_sup)
    confidence_dl_sup = confidence_mean(error_record_dl_sup,0.05)[1]
    min_dl_sup = min(confidence_dl_sup)
    max_dl_sup = max(confidence_dl_sup)

    error_record_dl_sub = np.array(error_record_dl_sub)
    mean_error_dl_sub = np.mean(error_record_dl_sub)
    confidence_dl_sub = confidence_mean(error_record_dl_sub,0.05)[1]
    min_dl_sub = min(confidence_dl_sub)
    max_dl_sub = max(confidence_dl_sub)

    rl_list.append(rl)

    error_list_dl_sup.append(mean_error_dl_sup)
    error_list_dl_sup_min.append(min_dl_sup)
    error_list_dl_sup_max.append(max_dl_sup)

    error_list_dl_sub.append(mean_error_dl_sub)
    error_list_dl_sub_min.append(min_dl_sub)
    error_list_dl_sub_max.append(max_dl_sub)

dic_error = {'rl':rl_list,
             'error_dl_sup':error_list_dl_sup,
             'min_dl_sup':error_list_dl_sup_min,
             'max_dl_sup':error_list_dl_sup_max,          
             'error_dl_sub':error_list_dl_sub,
             'min_dl_sub':error_list_dl_sub_min,
             'max_dl_sub':error_list_dl_sub_max}

csv_out = pandas.DataFrame(dic_error)
csv_out.to_csv('../../results/pitchfork_nus.csv',header = True)