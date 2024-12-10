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

if not os.path.exists('../../results/may_fold'):
    os.makedirs('../../results/may_fold')

bl_list = []

bl_error_list_dl = []
bl_error_list_dl_min = []
bl_error_list_dl_max = []

bl_error_list_ac = []
bl_error_list_ac_min = []
bl_error_list_ac_max = []

bl_error_list_dev = []
bl_error_list_dev_min = []
bl_error_list_dev_max = []

bl_error_list_lstm = []
bl_error_list_lstm_min = []
bl_error_list_lstm_max = []

bh_error_list_dl_sum = []
bh_error_list_ac_sum = []
bh_error_list_dev_sum = []
bh_error_list_lstm_sum = []

bh_error_list_dl = []
bh_error_list_dl_min = []
bh_error_list_dl_max = []

bh_error_list_ac = []
bh_error_list_ac_min = []
bh_error_list_ac_max = []

bh_error_list_dev = []
bh_error_list_dev_min = []
bh_error_list_dev_max = []

bh_error_list_lstm = []
bh_error_list_lstm_min = []
bh_error_list_lstm_max = []

bh_error_dic_dl_temp = {}
bh_error_dic_ac_temp = {}
bh_error_dic_dev_temp = {}
bh_error_dic_lstm_temp = {}

for b in np.linspace(0,0.2,11):

    bl = round(b,2)

    error_record_dl = list()
    error_record_ac = list()
    error_record_dev = list()
    error_record_lstm = list()

    for c_rate in [1e-5,2e-5,3e-5,4e-5,5e-5]:

        sequences_resids = list()
        sequences_ac = list()
        sequences_bac = list()
        sequences_dev = list()
        sequences_bdev = list()
        normalization = list()

        for n in range (1,numtest+1):
            df_resids = pandas.read_csv('../data_us/may_fold_white/{}/{}/resids/may_fold_resids_'.format(bl,c_rate)+str(n)+'.csv')
            df_ac = pandas.read_csv('../data_us/may_fold_white/{}/{}/ac/may_fold_ac_'.format(bl,c_rate)+str(n)+'.csv')
            df_dev = pandas.read_csv('../data_us/may_fold_white/{}/{}/dev/may_fold_dev_'.format(bl,c_rate)+str(n)+'.csv')
            keep_col_resids = ['residuals','b']
            new_f_resids = df_resids[keep_col_resids]

            values_s = df_resids['b'].iloc[0]
            values_o = df_resids['b'].iloc[-1]

            new_f_ac = df_ac['ac1']
            new_f_bac = df_ac['b']
            new_f_dev = df_dev['DEV']
            new_f_bdev = df_dev['b']
            values_resids = new_f_resids.values
            values_ac = np.array(new_f_ac.values)
            values_bac = np.array(new_f_bac.values)
            values_dev = np.array(new_f_dev.values)
            values_bdev = np.array(new_f_bdev.values)
            
            # Padding input sequences for DL
            for j in range(seq_len-len(values_resids)):
                values_resids=np.insert(values_resids,0,[[0,0]],axis= 0)
            
            sequences_resids.append(values_resids)
            normalization.append([values_s,values_o-values_s])

            sequences_ac.append(values_ac)
            sequences_bac.append(values_bac)
            sequences_dev.append(values_dev)
            sequences_bdev.append(values_bdev)

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

        result = pandas.read_csv('../data_us/may_fold_white/{}/{}/may_fold_result.csv'.format(bl,c_rate),header=0)
        trans_point = result['trans point'].values
        distance = result['distance'].values
        trans_point = list(trans_point)
        distance = list(distance)

        normalization = np.array(normalization)

        test_preds_record = list()
        test_preds_lstm_record = list()

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
            bh_error_list_dl_sum.append([round(distance[j],3),e_1])

            #fit curve for AC
            param_ac_1 = np.polyfit(sequences_ac[j],sequences_bac[j],1)
            p_ac_1 = np.poly1d(param_ac_1)
            ac_1 = p_ac_1(1)

            param_ac_2 = np.polyfit(sequences_ac[j],sequences_bac[j],2)
            p_ac_2 = np.poly1d(param_ac_2)
            ac_2 = p_ac_2(1)

            if abs(ac_1 - trans_point[j]) >= abs(ac_2 - trans_point[j]):
                ac = ac_2
            else:
                ac = ac_1

            e_2 = abs(ac-trans_point[j])/distance[j]
            error_record_ac.append(e_2)
            bh_error_list_ac_sum.append([round(distance[j],3),e_2])

            #fit curve for DEV
            param_dev_1 = np.polyfit(sequences_dev[j],sequences_bdev[j],1)
            p_dev_1 = np.poly1d(param_dev_1)
            dev_1 = p_dev_1(1)

            param_dev_2 = np.polyfit(sequences_dev[j],sequences_bdev[j],2)
            p_dev_2 = np.poly1d(param_dev_2)
            dev_2 = p_dev_2(1)

            if abs(dev_1 - trans_point[j]) >= abs(dev_2 - trans_point[j]):
                dev = dev_2
            else:
                dev = dev_1

            e_3 = abs(dev-trans_point[j])/distance[j]
            error_record_dev.append(e_3)
            bh_error_list_dev_sum.append([round(distance[j],3),e_3])

            e_4 = abs(preds_lstm[j]-trans_point[j])/distance[j]
            error_record_lstm.append(e_4)
            bh_error_list_lstm_sum.append([round(distance[j],3),e_4])

        for key, value in bh_error_list_dl_sum:
            if key in bh_error_dic_dl_temp:
                bh_error_dic_dl_temp[key].append(value)
            else:
                bh_error_dic_dl_temp[key] = [value]

        for key, value in bh_error_list_ac_sum:
            if key in bh_error_dic_ac_temp:
                bh_error_dic_ac_temp[key].append(value)
            else:
                bh_error_dic_ac_temp[key] = [value]

        for key, value in bh_error_list_dev_sum:
            if key in bh_error_dic_dev_temp:
                bh_error_dic_dev_temp[key].append(value)
            else:
                bh_error_dic_dev_temp[key] = [value]

        for key, value in bh_error_list_lstm_sum:
            if key in bh_error_dic_lstm_temp:
                bh_error_dic_lstm_temp[key].append(value)
            else:
                bh_error_dic_lstm_temp[key] = [value]

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

    bl_list.append(bl)

    bl_error_list_dl.append(mean_error_dl)
    bl_error_list_dl_min.append(min_dl)
    bl_error_list_dl_max.append(max_dl)

    bl_error_list_ac.append(mean_error_ac)
    bl_error_list_ac_min.append(min_ac)
    bl_error_list_ac_max.append(max_ac)

    bl_error_list_dev.append(mean_error_dev)
    bl_error_list_dev_min.append(min_dev)
    bl_error_list_dev_max.append(max_dev)

    bl_error_list_lstm.append(mean_error_lstm)
    bl_error_list_lstm_min.append(min_lstm)
    bl_error_list_lstm_max.append(max_lstm)

dic_bl_error = {'bl':bl_list,
            'error_dl':bl_error_list_dl,
            'min_dl':bl_error_list_dl_min,
            'max_dl':bl_error_list_dl_max,
            'error_ac':bl_error_list_ac,
            'min_ac':bl_error_list_ac_min,
            'max_ac':bl_error_list_ac_max,
            'error_dev':bl_error_list_dev,
            'min_dev':bl_error_list_dev_min,
            'max_dev':bl_error_list_dev_max,            
            'error_lstm':bl_error_list_lstm,
            'min_lstm':bl_error_list_lstm_min,
            'max_lstm':bl_error_list_lstm_max}

csv_bl_out = pandas.DataFrame(dic_bl_error)
csv_bl_out.to_csv('../../results/may_fold/may_fold_bl_us.csv',header = True)

sorted_keys = sorted(bh_error_dic_dl_temp.keys())
bh_error_dic_dl = {key: bh_error_dic_dl_temp[key] for key in sorted_keys}

sorted_keys = sorted(bh_error_dic_ac_temp.keys())
bh_error_dic_ac = {key: bh_error_dic_ac_temp[key] for key in sorted_keys}

sorted_keys = sorted(bh_error_dic_dev_temp.keys())
bh_error_dic_dev = {key: bh_error_dic_dev_temp[key] for key in sorted_keys}

sorted_keys = sorted(bh_error_dic_lstm_temp.keys())
bh_error_dic_lstm = {key: bh_error_dic_lstm_temp[key] for key in sorted_keys}

bh_list = list(bh_error_dic_dl.keys())

for bh in bh_list:

    # dl
    error_record_dl = bh_error_dic_dl[bh]
    mean_error_dl = np.mean(error_record_dl)
    confidence_dl = confidence_mean(error_record_dl,0.05)[1]
    min_dl = min(confidence_dl)
    max_dl = max(confidence_dl)

    bh_error_list_dl.append(mean_error_dl)
    bh_error_list_dl_min.append(min_dl)
    bh_error_list_dl_max.append(max_dl)
    
    #ac
    error_record_ac = bh_error_dic_ac[bh]
    mean_error_ac = np.mean(error_record_ac)
    confidence_ac = confidence_mean(error_record_ac,0.05)[1]
    min_ac = min(confidence_ac)
    max_ac = max(confidence_ac)

    bh_error_list_ac.append(mean_error_ac)
    bh_error_list_ac_min.append(min_ac)
    bh_error_list_ac_max.append(max_ac)

    #dev
    error_record_dev = bh_error_dic_dev[bh]
    mean_error_dev = np.mean(error_record_dev)
    confidence_dev = confidence_mean(error_record_dev,0.05)[1]
    min_dev = min(confidence_dev)
    max_dev = max(confidence_dev)

    bh_error_list_dev.append(mean_error_dev)
    bh_error_list_dev_min.append(min_dev)
    bh_error_list_dev_max.append(max_dev)

    #lstm
    error_record_lstm = bh_error_dic_lstm[bh]
    mean_error_lstm = np.mean(error_record_lstm)
    confidence_lstm = confidence_mean(error_record_lstm,0.05)[1]
    min_lstm = min(confidence_lstm)
    max_lstm = max(confidence_lstm)

    bh_error_list_lstm.append(mean_error_lstm)
    bh_error_list_lstm_min.append(min_lstm)
    bh_error_list_lstm_max.append(max_lstm)

dic_bh_error = {'bh':bh_list,
            'error_dl':bh_error_list_dl,
            'min_dl':bh_error_list_dl_min,
            'max_dl':bh_error_list_dl_max,
            'error_ac':bh_error_list_ac,
            'min_ac':bh_error_list_ac_min,
            'max_ac':bh_error_list_ac_max,
            'error_dev':bh_error_list_dev,
            'min_dev':bh_error_list_dev_min,
            'max_dev':bh_error_list_dev_max,            
            'error_lstm':bh_error_list_lstm,
            'min_lstm':bh_error_list_lstm_min,
            'max_lstm':bh_error_list_lstm_max}

csv_bh_out = pandas.DataFrame(dic_bh_error)
csv_bh_out.to_csv('../../results/may_fold/may_fold_bh_us.csv',header = True)