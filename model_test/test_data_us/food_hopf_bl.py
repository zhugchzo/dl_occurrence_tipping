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

if not os.path.exists('../../results/food_hopf'):
    os.makedirs('../../results/food_hopf')

kl_list = []

kl_error_list_dl = []
kl_error_list_dl_min = []
kl_error_list_dl_max = []

kl_error_list_ac = []
kl_error_list_ac_min = []
kl_error_list_ac_max = []

kl_error_list_dev = []
kl_error_list_dev_min = []
kl_error_list_dev_max = []

kl_error_list_lstm = []
kl_error_list_lstm_min = []
kl_error_list_lstm_max = []

kh_error_list_dl_sum = []
kh_error_list_ac_sum = []
kh_error_list_dev_sum = []
kh_error_list_lstm_sum = []

kh_error_list_dl = []
kh_error_list_dl_min = []
kh_error_list_dl_max = []

kh_error_list_ac = []
kh_error_list_ac_min = []
kh_error_list_ac_max = []

kh_error_list_dev = []
kh_error_list_dev_min = []
kh_error_list_dev_max = []

kh_error_list_lstm = []
kh_error_list_lstm_min = []
kh_error_list_lstm_max = []

kh_error_dic_dl_temp = {}
kh_error_dic_ac_temp = {}
kh_error_dic_dev_temp = {}
kh_error_dic_lstm_temp = {}

for k in np.linspace(0.2,0.4,11):

    kl = round(k,2)

    error_record_dl = list()
    error_record_ac = list()
    error_record_dev = list()
    error_record_lstm = list()

    for c_rate in [1e-5,2e-5,3e-5,4e-5,5e-5]:

        sequences_resids = list()
        sequences_ac = list()
        sequences_kac = list()
        sequences_dev = list()
        sequences_kdev = list()
        normalization = list()

        for n in range (1,numtest+1):
            df_resids = pandas.read_csv('../data_us/food_hopf_white/{}/{}/resids/food_hopf_resids_'.format(kl,c_rate)+str(n)+'.csv')
            df_ac = pandas.read_csv('../data_us/food_hopf_white/{}/{}/ac/food_hopf_ac_'.format(kl,c_rate)+str(n)+'.csv')
            df_dev = pandas.read_csv('../data_us/food_hopf_white/{}/{}/dev/food_hopf_dev_'.format(kl,c_rate)+str(n)+'.csv')
            keep_col_resids = ['residuals','k']
            new_f_resids = df_resids[keep_col_resids]

            values_s = df_resids['k'].iloc[0]
            values_o = df_resids['k'].iloc[-1]

            new_f_ac = df_ac['ac1']
            new_f_kac = df_ac['k']
            new_f_dev = df_dev['DEV']
            new_f_kdev = df_dev['k']
            values_resids = new_f_resids.values
            values_ac = np.array(new_f_ac.values)
            values_kac = np.array(new_f_kac.values)
            values_dev = np.array(new_f_dev.values)
            values_kdev = np.array(new_f_kdev.values)
            
            # Padding input sequences for DL
            for j in range(seq_len-len(values_resids)):
                values_resids=np.insert(values_resids,0,[[0,0]],axis= 0)
            
            sequences_resids.append(values_resids)
            normalization.append([values_s,values_o-values_s])

            sequences_ac.append(values_ac)
            sequences_kac.append(values_kac)
            sequences_dev.append(values_dev)
            sequences_kdev.append(values_kdev)

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

        result = pandas.read_csv('../data_us/food_hopf_white/{}/{}/food_hopf_result.csv'.format(kl,c_rate),header=0)
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
            kh_error_list_dl_sum.append([round(distance[j],3),e_1])

            #fit curve for AC
            param_ac_1 = np.polyfit(sequences_ac[j],sequences_kac[j],1)
            p_ac_1 = np.poly1d(param_ac_1)
            ac_1 = p_ac_1(1)

            param_ac_2 = np.polyfit(sequences_ac[j],sequences_kac[j],2)
            p_ac_2 = np.poly1d(param_ac_2)
            ac_2 = p_ac_2(1)

            if abs(ac_1 - trans_point[j]) >= abs(ac_2 - trans_point[j]):
                ac = ac_2
            else:
                ac = ac_1

            e_2 = abs(ac-trans_point[j])/distance[j]
            error_record_ac.append(e_2)
            kh_error_list_ac_sum.append([round(distance[j],3),e_2])

            #fit curve for DEV
            param_dev_1 = np.polyfit(sequences_dev[j],sequences_kdev[j],1)
            p_dev_1 = np.poly1d(param_dev_1)
            dev_1 = p_dev_1(1)

            param_dev_2 = np.polyfit(sequences_dev[j],sequences_kdev[j],2)
            p_dev_2 = np.poly1d(param_dev_2)
            dev_2 = p_dev_2(1)

            if abs(dev_1 - trans_point[j]) >= abs(dev_2 - trans_point[j]):
                dev = dev_2
            else:
                dev = dev_1

            e_3 = abs(dev-trans_point[j])/distance[j]
            error_record_dev.append(e_3)
            kh_error_list_dev_sum.append([round(distance[j],3),e_3])

            e_4 = abs(preds_lstm[j]-trans_point[j])/distance[j]
            error_record_lstm.append(e_4)
            kh_error_list_lstm_sum.append([round(distance[j],3),e_4])

        for key, value in kh_error_list_dl_sum:
            if key in kh_error_dic_dl_temp:
                kh_error_dic_dl_temp[key].append(value)
            else:
                kh_error_dic_dl_temp[key] = [value]

        for key, value in kh_error_list_ac_sum:
            if key in kh_error_dic_ac_temp:
                kh_error_dic_ac_temp[key].append(value)
            else:
                kh_error_dic_ac_temp[key] = [value]

        for key, value in kh_error_list_dev_sum:
            if key in kh_error_dic_dev_temp:
                kh_error_dic_dev_temp[key].append(value)
            else:
                kh_error_dic_dev_temp[key] = [value]

        for key, value in kh_error_list_lstm_sum:
            if key in kh_error_dic_lstm_temp:
                kh_error_dic_lstm_temp[key].append(value)
            else:
                kh_error_dic_lstm_temp[key] = [value]

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

    kl_list.append(kl)

    kl_error_list_dl.append(mean_error_dl)
    kl_error_list_dl_min.append(min_dl)
    kl_error_list_dl_max.append(max_dl)

    kl_error_list_ac.append(mean_error_ac)
    kl_error_list_ac_min.append(min_ac)
    kl_error_list_ac_max.append(max_ac)

    kl_error_list_dev.append(mean_error_dev)
    kl_error_list_dev_min.append(min_dev)
    kl_error_list_dev_max.append(max_dev)

    kl_error_list_lstm.append(mean_error_lstm)
    kl_error_list_lstm_min.append(min_lstm)
    kl_error_list_lstm_max.append(max_lstm)

dic_kl_error = {'kl':kl_list,
            'error_dl':kl_error_list_dl,
            'min_dl':kl_error_list_dl_min,
            'max_dl':kl_error_list_dl_max,
            'error_ac':kl_error_list_ac,
            'min_ac':kl_error_list_ac_min,
            'max_ac':kl_error_list_ac_max,
            'error_dev':kl_error_list_dev,
            'min_dev':kl_error_list_dev_min,
            'max_dev':kl_error_list_dev_max,            
            'error_lstm':kl_error_list_lstm,
            'min_lstm':kl_error_list_lstm_min,
            'max_lstm':kl_error_list_lstm_max}

csv_kl_out = pandas.DataFrame(dic_kl_error)
csv_kl_out.to_csv('../../results/food_hopf/food_hopf_kl_us.csv',header = True)

sorted_keys = sorted(kh_error_dic_dl_temp.keys())
kh_error_dic_dl = {key: kh_error_dic_dl_temp[key] for key in sorted_keys}

sorted_keys = sorted(kh_error_dic_ac_temp.keys())
kh_error_dic_ac = {key: kh_error_dic_ac_temp[key] for key in sorted_keys}

sorted_keys = sorted(kh_error_dic_dev_temp.keys())
kh_error_dic_dev = {key: kh_error_dic_dev_temp[key] for key in sorted_keys}

sorted_keys = sorted(kh_error_dic_lstm_temp.keys())
kh_error_dic_lstm = {key: kh_error_dic_lstm_temp[key] for key in sorted_keys}

kh_list = list(kh_error_dic_dl.keys())

for kh in kh_list:

    # dl
    error_record_dl = kh_error_dic_dl[kh]
    mean_error_dl = np.mean(error_record_dl)
    confidence_dl = confidence_mean(error_record_dl,0.05)[1]
    min_dl = min(confidence_dl)
    max_dl = max(confidence_dl)

    kh_error_list_dl.append(mean_error_dl)
    kh_error_list_dl_min.append(min_dl)
    kh_error_list_dl_max.append(max_dl)
    
    #ac
    error_record_ac = kh_error_dic_ac[kh]
    mean_error_ac = np.mean(error_record_ac)
    confidence_ac = confidence_mean(error_record_ac,0.05)[1]
    min_ac = min(confidence_ac)
    max_ac = max(confidence_ac)

    kh_error_list_ac.append(mean_error_ac)
    kh_error_list_ac_min.append(min_ac)
    kh_error_list_ac_max.append(max_ac)

    #dev
    error_record_dev = kh_error_dic_dev[kh]
    mean_error_dev = np.mean(error_record_dev)
    confidence_dev = confidence_mean(error_record_dev,0.05)[1]
    min_dev = min(confidence_dev)
    max_dev = max(confidence_dev)

    kh_error_list_dev.append(mean_error_dev)
    kh_error_list_dev_min.append(min_dev)
    kh_error_list_dev_max.append(max_dev)

    #lstm
    error_record_lstm = kh_error_dic_lstm[kh]
    mean_error_lstm = np.mean(error_record_lstm)
    confidence_lstm = confidence_mean(error_record_lstm,0.05)[1]
    min_lstm = min(confidence_lstm)
    max_lstm = max(confidence_lstm)

    kh_error_list_lstm.append(mean_error_lstm)
    kh_error_list_lstm_min.append(min_lstm)
    kh_error_list_lstm_max.append(max_lstm)

dic_kh_error = {'kh':kh_list,
            'error_dl':kh_error_list_dl,
            'min_dl':kh_error_list_dl_min,
            'max_dl':kh_error_list_dl_max,
            'error_ac':kh_error_list_ac,
            'min_ac':kh_error_list_ac_min,
            'max_ac':kh_error_list_ac_max,
            'error_dev':kh_error_list_dev,
            'min_dev':kh_error_list_dev_min,
            'max_dev':kh_error_list_dev_max,            
            'error_lstm':kh_error_list_lstm,
            'min_lstm':kh_error_list_lstm_min,
            'max_lstm':kh_error_list_lstm_max}

csv_kh_out = pandas.DataFrame(dic_kh_error)
csv_kh_out.to_csv('../../results/food_hopf/food_hopf_kh_us.csv',header = True)