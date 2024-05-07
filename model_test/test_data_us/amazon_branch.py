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

kk = 5
seq_len = 500
numtest = 20

pl_list = []

error_list_dl = []
error_list_dl_min = []
error_list_dl_max = []

error_list_ac = []
error_list_ac_min = []
error_list_ac_max = []

error_list_dev = []
error_list_dev_min = []
error_list_dev_max = []

for i in np.linspace(0.9,0.1,11):
    
    pl = round(i,2)

    sequences_resids = list()
    sequences_ac = list()
    sequences_pac = list()
    sequences_dev = list()
    sequences_pdev = list()
    normalization = list()

    for i in range (1,numtest+1):
        df_resids = pandas.read_csv('../data_us/amazon_branch_red/{}/resids/amazon_branch_500_resids_'.format(pl)+str(i)+'.csv')
        df_ac = pandas.read_csv('../data_us/amazon_branch_red/{}/ac/amazon_branch_500_ac_'.format(pl)+str(i)+'.csv')
        df_dev = pandas.read_csv('../data_us/amazon_branch_red/{}/dev/amazon_branch_500_dev_'.format(pl)+str(i)+'.csv')
        keep_col_resids = ['Residuals','p']
        new_f_resids = df_resids[keep_col_resids]

        values_s = df_resids['p'].iloc[0]
        values_o = df_resids['p'].iloc[-1]

        new_f_ac = df_ac['AC red']
        new_f_pac = df_ac['p']
        new_f_dev = df_dev['DEV']
        new_f_pdev = df_dev['p']
        values_resids = new_f_resids.values
        values_ac = np.array(new_f_ac.values)
        values_pac = np.array(new_f_pac.values)
        values_dev = np.array(new_f_dev.values)
        values_pdev = np.array(new_f_pdev.values)
        
        # Padding input sequences for DL
        for j in range(seq_len-len(values_resids)):
            values_resids=np.insert(values_resids,0,[[0,0]],axis= 0)
        
        sequences_resids.append(values_resids)
        normalization.append([values_s,values_o-values_s])

        sequences_ac.append(values_ac)
        sequences_pac.append(values_pac)
        sequences_dev.append(values_dev)
        sequences_pdev.append(values_pdev)

    sequences_resids = np.array(sequences_resids)

    # normalizing input time series by the average. 
    # 按平均值规范化输入时间串行。
    for i in range(numtest):
        values_avg = 0.0
        count_avg = 0
        for j in range (0,seq_len):
            if sequences_resids[i,j][0]!= 0 and sequences_resids[i,j][1]!= 0:
                values_avg = values_avg + abs(sequences_resids[i,j][0])                                       
                count_avg = count_avg + 1
        if count_avg != 0:
            values_avg = values_avg/count_avg
            for j in range (0,seq_len):
                if sequences_resids[i,j][0]!= 0 and sequences_resids[i,j][1]!= 0:
                    sequences_resids[i,j][0]= sequences_resids[i,j][0]/values_avg
                    sequences_resids[i,j][1]= (sequences_resids[i,j][1]-normalization[i][0])/normalization[i][1]

    test = sequences_resids
    test = np.array(test)
    test = test.reshape(-1,seq_len,2,1)

    result = pandas.read_csv('../data_us/amazon_branch_red/{}/amazon_branch_500_result.csv'.format(pl),header=0)
    trans_point = result['trans point'].values
    distance = result['distance'].values
    trans_point = list(trans_point)
    distance = list(distance)

    normalization = np.array(normalization)

    test_preds_record = []
    error_record_dl = []
    error_record_ac = []
    error_record_dev = []
    preds = np.zeros([kk,numtest])
    
    for i in range(1,kk+1):

        model_name = '../../dl_model/best_model_branch_red_{}.pkl'.format(i)

        model = load_model(model_name)

        test_preds = model.predict(test)
        test_preds = test_preds.reshape(numtest)
        test_preds = test_preds * normalization[:,1] + normalization[:,0]

        test_preds_record.append(test_preds)

    
    for i in range(kk):
        recordi = test_preds_record[i]
        for j in range(numtest):
            preds[i][j] = recordi[j]

    preds = preds.mean(axis=0) #每个测试样本在所有模型上的平均预测结果

    for j in range(numtest):
        e_1 = abs(preds[j]-trans_point[j])/distance[j]
        error_record_dl.append(e_1)
        #fit curve for AC
        param_ac = np.polyfit(sequences_ac[j],sequences_pac[j],2)
        p_ac = np.poly1d(param_ac)
        ac = p_ac(1)
        e_2 = abs(ac-trans_point[j])/distance[j]
        error_record_ac.append(e_2)
        #fit curve for DEV
        param_dev = np.polyfit(sequences_dev[j],sequences_pdev[j],2)
        p_dev = np.poly1d(param_dev)
        dev = p_dev(1)
        e_3 = abs(dev-trans_point[j])/distance[j]
        error_record_dev.append(e_3)
    
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
    

    pl_list.append(pl)

    error_list_dl.append(mean_error_dl)
    error_list_dl_min.append(min_dl)
    error_list_dl_max.append(max_dl)

    error_list_ac.append(mean_error_ac)
    error_list_ac_min.append(min_ac)
    error_list_ac_max.append(max_ac)

    error_list_dev.append(mean_error_dev)
    error_list_dev_min.append(min_dev)
    error_list_dev_max.append(max_dev)

dic_error = {'pl':pl_list,
             'error_dl':error_list_dl,
             'min_dl':error_list_dl_min,
             'max_dl':error_list_dl_max,
             'error_ac':error_list_ac,
             'min_ac':error_list_ac_min,
             'max_ac':error_list_ac_max,
             'error_dev':error_list_dev,
             'min_dev':error_list_dev_min,
             'max_dev':error_list_dev_max}

csv_out = pandas.DataFrame(dic_error)
csv_out.to_csv('../../results/amazon_branch_us.csv',header = True)