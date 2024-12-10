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

if not os.path.exists('../../results/global_fold'):
    os.makedirs('../../results/global_fold')

ul_list = []

ul_error_list_dl = []
ul_error_list_dl_min = []
ul_error_list_dl_max = []

ul_error_list_ac = []
ul_error_list_ac_min = []
ul_error_list_ac_max = []

ul_error_list_dev = []
ul_error_list_dev_min = []
ul_error_list_dev_max = []

ul_error_list_lstm = []
ul_error_list_lstm_min = []
ul_error_list_lstm_max = []

uh_error_list_dl_sum = []
uh_error_list_ac_sum = []
uh_error_list_dev_sum = []
uh_error_list_lstm_sum = []

uh_error_list_dl = []
uh_error_list_dl_min = []
uh_error_list_dl_max = []

uh_error_list_ac = []
uh_error_list_ac_min = []
uh_error_list_ac_max = []

uh_error_list_dev = []
uh_error_list_dev_min = []
uh_error_list_dev_max = []

uh_error_list_lstm = []
uh_error_list_lstm_min = []
uh_error_list_lstm_max = []

uh_error_dic_dl_temp = {}
uh_error_dic_ac_temp = {}
uh_error_dic_dev_temp = {}
uh_error_dic_lstm_temp = {}

for u in np.linspace(1.4,1.2,11):

    ul = round(u,2)

    error_record_dl = list()
    error_record_ac = list()
    error_record_dev = list()
    error_record_lstm = list()

    for c_rate in [-5e-7,-6e-7,-7e-7,-8e-7,-9e-7]:

        sequences_resids = list()
        sequences_ac = list()
        sequences_uac = list()
        sequences_dev = list()
        sequences_udev = list()
        normalization = list()

        for n in range (1,numtest+1):
            df_resids = pandas.read_csv('../data_nus/global_fold_red/{}/{}/resids/global_fold_resids_'.format(ul,c_rate)+str(n)+'.csv')
            df_ac = pandas.read_csv('../data_nus/global_fold_red/{}/{}/ac/global_fold_ac_'.format(ul,c_rate)+str(n)+'.csv')
            df_dev = pandas.read_csv('../data_nus/global_fold_red/{}/{}/dev/global_fold_dev_'.format(ul,c_rate)+str(n)+'.csv')
            keep_col_resids = ['residuals','u']
            new_f_resids = df_resids[keep_col_resids]

            values_s = df_resids['u'].iloc[0]
            values_o = df_resids['u'].iloc[-1]

            new_f_ac = df_ac['AC red']
            new_f_uac = df_ac['u']
            new_f_dev = df_dev['DEV']
            new_f_udev = df_dev['u']
            values_resids = new_f_resids.values
            values_ac = np.array(new_f_ac.values)
            values_uac = np.array(new_f_uac.values)
            values_dev = np.array(new_f_dev.values)
            values_udev = np.array(new_f_udev.values)
            
            # Padding input sequences for DL
            for j in range(seq_len-len(values_resids)):
                values_resids=np.insert(values_resids,0,[[0,0]],axis= 0)
            
            sequences_resids.append(values_resids)
            normalization.append([values_s,values_o-values_s])

            sequences_ac.append(values_ac)
            sequences_uac.append(values_uac)
            sequences_dev.append(values_dev)
            sequences_udev.append(values_udev)

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

        result = pandas.read_csv('../data_nus/global_fold_red/{}/{}/global_fold_result.csv'.format(ul,c_rate),header=0)
        trans_point = result['trans point'].values
        distance = result['distance'].values
        trans_point = list(trans_point)
        distance = list(distance)

        normalization = np.array(normalization)

        test_preds_record = []
        test_preds_lstm_record = []

        preds = np.zeros([kk,numtest])
        preds_lstm = np.zeros([kk,numtest])
        
        for i in range(1,kk+1):

            model_name = '../../dl_model/best_model_{}.keras'.format(i)

            model = load_model(model_name)

            test_preds = model.predict(test)
            test_preds = test_preds.reshape(numtest)
            test_preds = test_preds * normalization[:,1] + normalization[:,0]

            test_preds_record.append(test_preds)

        for i in range(1,kk+1):

            lstm_model_name = '../../dl_model/best_lstm_model_{}.keras'.format(i)

            lstm_model = load_model(lstm_model_name)

            test_preds_lstm = lstm_model.predict(test)
            test_preds_lstm = test_preds_lstm.reshape(numtest)
            test_preds_lstm = test_preds_lstm * normalization[:,1] + normalization[:,0]

            test_preds_lstm_record.append(test_preds_lstm)  
        
        for i in range(kk):
            recordi = test_preds_record[i]
            recordi_lstm = test_preds_lstm_record[i]
            for j in range(numtest):
                preds[i][j] = recordi[j]
                preds_lstm[i][j] = recordi_lstm[j]

        preds = preds.mean(axis=0)
        preds_lstm = preds_lstm.mean(axis=0)

        for j in range(numtest):
            e_1 = abs(preds[j]-trans_point[j])/distance[j]
            error_record_dl.append(e_1)
            uh_error_list_dl_sum.append([round(distance[j],3),e_1])

            #fit curve for AC
            param_ac_1 = np.polyfit(sequences_ac[j],sequences_uac[j],1)
            p_ac_1 = np.poly1d(param_ac_1)
            ac_1 = p_ac_1(1)

            param_ac_2 = np.polyfit(sequences_ac[j],sequences_uac[j],2)
            p_ac_2 = np.poly1d(param_ac_2)
            ac_2 = p_ac_2(1)

            if abs(ac_1 - trans_point[j]) >= abs(ac_2 - trans_point[j]):
                ac = ac_2
            else:
                ac = ac_1

            e_2 = abs(ac-trans_point[j])/distance[j]
            error_record_ac.append(e_2)
            uh_error_list_ac_sum.append([round(distance[j],3),e_2])

            #fit curve for DEV
            param_dev_1 = np.polyfit(sequences_dev[j],sequences_udev[j],1)
            p_dev_1 = np.poly1d(param_dev_1)
            dev_1 = p_dev_1(1)

            param_dev_2 = np.polyfit(sequences_dev[j],sequences_udev[j],2)
            p_dev_2 = np.poly1d(param_dev_2)
            dev_2 = p_dev_2(1)

            if abs(dev_1 - trans_point[j]) >= abs(dev_2 - trans_point[j]):
                dev = dev_2
            else:
                dev = dev_1

            e_3 = abs(dev-trans_point[j])/distance[j]
            error_record_dev.append(e_3)
            uh_error_list_dev_sum.append([round(distance[j],3),e_3])

            e_4 = abs(preds_lstm[j]-trans_point[j])/distance[j]
            error_record_lstm.append(e_4)
            uh_error_list_lstm_sum.append([round(distance[j],3),e_4])

        for key, value in uh_error_list_dl_sum:
            if key in uh_error_dic_dl_temp:
                uh_error_dic_dl_temp[key].append(value)
            else:
                uh_error_dic_dl_temp[key] = [value]

        for key, value in uh_error_list_ac_sum:
            if key in uh_error_dic_ac_temp:
                uh_error_dic_ac_temp[key].append(value)
            else:
                uh_error_dic_ac_temp[key] = [value]

        for key, value in uh_error_list_dev_sum:
            if key in uh_error_dic_dev_temp:
                uh_error_dic_dev_temp[key].append(value)
            else:
                uh_error_dic_dev_temp[key] = [value]

        for key, value in uh_error_list_lstm_sum:
            if key in uh_error_dic_lstm_temp:
                uh_error_dic_lstm_temp[key].append(value)
            else:
                uh_error_dic_lstm_temp[key] = [value]

    error_record_dl = np.array(error_record_dl)
    mean_error_dl = np.mean(error_record_dl)
    confidence_dl = confidence_mean(error_record_dl,0.05)[1]
    min_dl = min(confidence_dl)
    max_dl = max(confidence_dl)

    error_record_ac = np.array(error_record_ac)
    mean_error_ac = np.mean(error_record_ac)
    confidence_ac = confidence_mean(error_record_ac,0.05)[1]
    min_ac = min(confidence_ac)
    max_ac = max(confidence_ac)

    error_record_dev = np.array(error_record_dev)
    mean_error_dev = np.mean(error_record_dev)
    confidence_dev = confidence_mean(error_record_dev,0.05)[1]
    min_dev = min(confidence_dev)
    max_dev = max(confidence_dev)

    error_record_lstm = np.array(error_record_lstm)
    mean_error_lstm = np.mean(error_record_lstm)
    confidence_lstm = confidence_mean(error_record_lstm,0.05)[1]
    min_lstm = min(confidence_lstm)
    max_lstm = max(confidence_lstm)

    ul_list.append(ul)

    ul_error_list_dl.append(mean_error_dl)
    ul_error_list_dl_min.append(min_dl)
    ul_error_list_dl_max.append(max_dl)

    ul_error_list_ac.append(mean_error_ac)
    ul_error_list_ac_min.append(min_ac)
    ul_error_list_ac_max.append(max_ac)

    ul_error_list_dev.append(mean_error_dev)
    ul_error_list_dev_min.append(min_dev)
    ul_error_list_dev_max.append(max_dev)

    ul_error_list_lstm.append(mean_error_lstm)
    ul_error_list_lstm_min.append(min_lstm)
    ul_error_list_lstm_max.append(max_lstm)

dic_ul_error = {'ul':ul_list,
            'error_dl':ul_error_list_dl,
            'min_dl':ul_error_list_dl_min,
            'max_dl':ul_error_list_dl_max,
            'error_ac':ul_error_list_ac,
            'min_ac':ul_error_list_ac_min,
            'max_ac':ul_error_list_ac_max,
            'error_dev':ul_error_list_dev,
            'min_dev':ul_error_list_dev_min,
            'max_dev':ul_error_list_dev_max,            
            'error_lstm':ul_error_list_lstm,
            'min_lstm':ul_error_list_lstm_min,
            'max_lstm':ul_error_list_lstm_max}

csv_ul_out = pandas.DataFrame(dic_ul_error)
csv_ul_out.to_csv('../../results/global_fold/global_fold_ul_nus.csv',header = True)

sorted_keys = sorted(uh_error_dic_dl_temp.keys())
uh_error_dic_dl = {key: uh_error_dic_dl_temp[key] for key in sorted_keys}

sorted_keys = sorted(uh_error_dic_ac_temp.keys())
uh_error_dic_ac = {key: uh_error_dic_ac_temp[key] for key in sorted_keys}

sorted_keys = sorted(uh_error_dic_dev_temp.keys())
uh_error_dic_dev = {key: uh_error_dic_dev_temp[key] for key in sorted_keys}

sorted_keys = sorted(uh_error_dic_lstm_temp.keys())
uh_error_dic_lstm = {key: uh_error_dic_lstm_temp[key] for key in sorted_keys}

uh_list = list(uh_error_dic_dl.keys())

for uh in uh_list:

    # dl
    error_record_dl = uh_error_dic_dl[uh]
    mean_error_dl = np.mean(error_record_dl)
    confidence_dl = confidence_mean(error_record_dl,0.05)[1]
    min_dl = min(confidence_dl)
    max_dl = max(confidence_dl)

    uh_error_list_dl.append(mean_error_dl)
    uh_error_list_dl_min.append(min_dl)
    uh_error_list_dl_max.append(max_dl)
    
    #ac
    error_record_ac = uh_error_dic_ac[uh]
    mean_error_ac = np.mean(error_record_ac)
    confidence_ac = confidence_mean(error_record_ac,0.05)[1]
    min_ac = min(confidence_ac)
    max_ac = max(confidence_ac)

    uh_error_list_ac.append(mean_error_ac)
    uh_error_list_ac_min.append(min_ac)
    uh_error_list_ac_max.append(max_ac)

    #dev
    error_record_dev = uh_error_dic_dev[uh]
    mean_error_dev = np.mean(error_record_dev)
    confidence_dev = confidence_mean(error_record_dev,0.05)[1]
    min_dev = min(confidence_dev)
    max_dev = max(confidence_dev)

    uh_error_list_dev.append(mean_error_dev)
    uh_error_list_dev_min.append(min_dev)
    uh_error_list_dev_max.append(max_dev)

    #lstm
    error_record_lstm = uh_error_dic_lstm[uh]
    mean_error_lstm = np.mean(error_record_lstm)
    confidence_lstm = confidence_mean(error_record_lstm,0.05)[1]
    min_lstm = min(confidence_lstm)
    max_lstm = max(confidence_lstm)

    uh_error_list_lstm.append(mean_error_lstm)
    uh_error_list_lstm_min.append(min_lstm)
    uh_error_list_lstm_max.append(max_lstm)

dic_uh_error = {'uh':uh_list,
            'error_dl':uh_error_list_dl,
            'min_dl':uh_error_list_dl_min,
            'max_dl':uh_error_list_dl_max,
            'error_ac':uh_error_list_ac,
            'min_ac':uh_error_list_ac_min,
            'max_ac':uh_error_list_ac_max,
            'error_dev':uh_error_list_dev,
            'min_dev':uh_error_list_dev_min,
            'max_dev':uh_error_list_dev_max,            
            'error_lstm':uh_error_list_lstm,
            'min_lstm':uh_error_list_lstm_min,
            'max_lstm':uh_error_list_lstm_max}

csv_uh_out = pandas.DataFrame(dic_uh_error)
csv_uh_out.to_csv('../../results/global_fold/global_fold_uh_nus.csv',header = True)