import pandas
import numpy as np
import os
from tensorflow.keras.models import load_model
from confidence_calculate import confidence_mean

kk = 10
seq_len = 500
numtest = 10

if not os.path.exists('../../results/single_DL_model'):
    os.makedirs('../../results/single_DL_model')

kk = 10
seq_len = 500
numtest = 10

bl_list = []
kl_list = []
al_list = []

ul_list1 = []
ul_list2 = []
pl_list = []

error_list_fold_white = []
error_list_fold_white_min = []
error_list_fold_white_max = []

error_list_hopf_white = []
error_list_hopf_white_min = []
error_list_hopf_white_max = []

error_list_branch_white = []
error_list_branch_white_min = []
error_list_branch_white_max = []

error_list_fold_red = []
error_list_fold_red_min = []
error_list_fold_red_max = []

error_list_hopf_red = []
error_list_hopf_red_min = []
error_list_hopf_red_max = []

error_list_branch_red = []
error_list_branch_red_min = []
error_list_branch_red_max = []

# may fold
for b in np.linspace(0,0.2,11):

    bl = round(b,2)

    for c_rate in [1e-5,2e-5,3e-5,4e-5,5e-5]:

        sequences_resids = list()
        normalization = list()

        for n in range (1,numtest+1):
            df_resids = pandas.read_csv('../data_nus/may_fold_white/{}/{}/resids/may_fold_resids_'.format(bl,c_rate)+str(n)+'.csv')
            keep_col_resids = ['residuals','b']
            new_f_resids = df_resids[keep_col_resids]

            values_s = df_resids['b'].iloc[0]
            values_o = df_resids['b'].iloc[-1]

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

        result = pandas.read_csv('../data_nus/may_fold_white/{}/{}/may_fold_result.csv'.format(bl,c_rate),header=0)
        trans_point = result['trans point'].values
        distance = result['distance'].values
        trans_point = list(trans_point)
        distance = list(distance)

        normalization = np.array(normalization)

        test_preds_record = []

        error_record_dl = []

        preds = np.zeros([kk,numtest])
        
        for i in range(1,kk+1):

            model_name = '../../dl_model_SI/best_model_fold_{}.keras'.format(i)

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
            e = abs(preds[j]-trans_point[j])/distance[j]
            error_record_dl.append(e)

    error_record_dl = np.array(error_record_dl)
    mean_error_dl = np.mean(error_record_dl)
    confidence_dl = confidence_mean(error_record_dl,0.05)[1]
    min_dl = min(confidence_dl)
    max_dl = max(confidence_dl)

    bl_list.append(bl)

    error_list_fold_white.append(mean_error_dl)
    error_list_fold_white_min.append(min_dl)
    error_list_fold_white_max.append(max_dl)

# food hopf
for k in np.linspace(0.2,0.4,11):

    kl = round(k,2)

    for c_rate in [1e-5,2e-5,3e-5,4e-5,5e-5]:

        sequences_resids = list()
        normalization = list()

        for n in range (1,numtest+1):
            df_resids = pandas.read_csv('../data_nus/food_hopf_white/{}/{}/resids/food_hopf_resids_'.format(kl,c_rate)+str(n)+'.csv')
            keep_col_resids = ['residuals','k']
            new_f_resids = df_resids[keep_col_resids]

            values_s = df_resids['k'].iloc[0]
            values_o = df_resids['k'].iloc[-1]

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

        result = pandas.read_csv('../data_nus/food_hopf_white/{}/{}/food_hopf_result.csv'.format(kl,c_rate),header=0)
        trans_point = result['trans point'].values
        distance = result['distance'].values
        trans_point = list(trans_point)
        distance = list(distance)

        normalization = np.array(normalization)

        test_preds_record = []

        error_record_dl = []

        preds = np.zeros([kk,numtest])
        
        for i in range(1,kk+1):

            model_name = '../../dl_model_SI/best_model_fold_{}.keras'.format(i)

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
            e = abs(preds[j]-trans_point[j])/distance[j]
            error_record_dl.append(e)

    error_record_dl = np.array(error_record_dl)
    mean_error_dl = np.mean(error_record_dl)
    confidence_dl = confidence_mean(error_record_dl,0.05)[1]
    min_dl = min(confidence_dl)
    max_dl = max(confidence_dl)

    kl_list.append(kl)

    error_list_hopf_white.append(mean_error_dl)
    error_list_hopf_white_min.append(min_dl)
    error_list_hopf_white_max.append(max_dl)

# cr branch
for a in np.linspace(0,5,11):

    al = round(a,2)

    for c_rate in [1e-4,2e-4,3e-4,4e-4,5e-4]:

        sequences_resids = list()
        normalization = list()

        for n in range (1,numtest+1):
            df_resids = pandas.read_csv('../data_nus/cr_branch_white/{}/{}/resids/cr_branch_resids_'.format(al,c_rate)+str(n)+'.csv')
            keep_col_resids = ['residuals','a']
            new_f_resids = df_resids[keep_col_resids]

            values_s = df_resids['a'].iloc[0]
            values_o = df_resids['a'].iloc[-1]

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

        result = pandas.read_csv('../data_nus/cr_branch_white/{}/{}/cr_branch_result.csv'.format(al,c_rate),header=0)
        trans_point = result['trans point'].values
        distance = result['distance'].values
        trans_point = list(trans_point)
        distance = list(distance)

        normalization = np.array(normalization)

        test_preds_record = []

        error_record_dl = []

        preds = np.zeros([kk,numtest])
        
        for i in range(1,kk+1):

            model_name = '../../dl_model_SI/best_model_fold_{}.keras'.format(i)

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
            e = abs(preds[j]-trans_point[j])/distance[j]
            error_record_dl.append(e)

    error_record_dl = np.array(error_record_dl)
    mean_error_dl = np.mean(error_record_dl)
    confidence_dl = confidence_mean(error_record_dl,0.05)[1]
    min_dl = min(confidence_dl)
    max_dl = max(confidence_dl)

    al_list.append(al)

    error_list_branch_white.append(mean_error_dl)
    error_list_branch_white_min.append(min_dl)
    error_list_branch_white_max.append(max_dl)

# global fold
for u in np.linspace(1.4,1.2,11):

    ul = round(u,2)

    for c_rate in [-5e-7,-6e-7,-7e-7,-8e-7,-9e-7]:

        sequences_resids = list()
        normalization = list()

        for n in range (1,numtest+1):
            df_resids = pandas.read_csv('../data_nus/global_fold_red/{}/{}/resids/global_fold_resids_'.format(ul,c_rate)+str(n)+'.csv')
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

        result = pandas.read_csv('../data_nus/global_fold_red/{}/{}/global_fold_result.csv'.format(ul,c_rate),header=0)
        trans_point = result['trans point'].values
        distance = result['distance'].values
        trans_point = list(trans_point)
        distance = list(distance)

        normalization = np.array(normalization)

        test_preds_record = []

        error_record_dl = []

        preds = np.zeros([kk,numtest])
        
        for i in range(1,kk+1):

            model_name = '../../dl_model_SI/best_model_fold_{}.keras'.format(i)

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
            e = abs(preds[j]-trans_point[j])/distance[j]
            error_record_dl.append(e)

    error_record_dl = np.array(error_record_dl)
    mean_error_dl = np.mean(error_record_dl)
    confidence_dl = confidence_mean(error_record_dl,0.05)[1]
    min_dl = min(confidence_dl)
    max_dl = max(confidence_dl)

    ul_list1.append(ul)

    error_list_fold_red.append(mean_error_dl)
    error_list_fold_red_min.append(min_dl)
    error_list_fold_red_max.append(max_dl)

# MPT hopf
for u in np.linspace(0,0.3,11):

    ul = round(u,2)

    for c_rate in [1e-5,2e-5,3e-5,4e-5,5e-5]:

        sequences_resids = list()
        normalization = list()

        for n in range (1,numtest+1):
            df_resids = pandas.read_csv('../data_nus/MPT_hopf_red/{}/{}/resids/MPT_hopf_resids_'.format(ul,c_rate)+str(n)+'.csv')
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

        result = pandas.read_csv('../data_nus/MPT_hopf_red/{}/{}/MPT_hopf_result.csv'.format(ul,c_rate),header=0)
        trans_point = result['trans point'].values
        distance = result['distance'].values
        trans_point = list(trans_point)
        distance = list(distance)

        normalization = np.array(normalization)

        test_preds_record = []

        error_record_dl = []

        preds = np.zeros([kk,numtest])
        
        for i in range(1,kk+1):

            model_name = '../../dl_model_SI/best_model_fold_{}.keras'.format(i)

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
            e = abs(preds[j]-trans_point[j])/distance[j]
            error_record_dl.append(e)

    error_record_dl = np.array(error_record_dl)
    mean_error_dl = np.mean(error_record_dl)
    confidence_dl = confidence_mean(error_record_dl,0.05)[1]
    min_dl = min(confidence_dl)
    max_dl = max(confidence_dl)

    ul_list2.append(ul)

    error_list_hopf_red.append(mean_error_dl)
    error_list_hopf_red_min.append(min_dl)
    error_list_hopf_red_max.append(max_dl)

# amazon branch
for p in np.linspace(0.9,0.1,11):

    pl = round(p,2)

    for c_rate in [-1e-5,-2e-5,-3e-5,-4e-5,-5e-5]:

        sequences_resids = list()
        normalization = list()

        for n in range (1,numtest+1):
            df_resids = pandas.read_csv('../data_nus/amazon_branch_red/{}/{}/resids/amazon_branch_resids_'.format(pl,c_rate)+str(n)+'.csv')
            keep_col_resids = ['residuals','p']
            new_f_resids = df_resids[keep_col_resids]

            values_s = df_resids['p'].iloc[0]
            values_o = df_resids['p'].iloc[-1]

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

        result = pandas.read_csv('../data_nus/amazon_branch_red/{}/{}/amazon_branch_result.csv'.format(pl,c_rate),header=0)
        trans_point = result['trans point'].values
        distance = result['distance'].values
        trans_point = list(trans_point)
        distance = list(distance)

        normalization = np.array(normalization)

        test_preds_record = []

        error_record_dl = []

        preds = np.zeros([kk,numtest])
        
        for i in range(1,kk+1):

            model_name = '../../dl_model_SI/best_model_fold_{}.keras'.format(i)

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
            e = abs(preds[j]-trans_point[j])/distance[j]
            error_record_dl.append(e)

    error_record_dl = np.array(error_record_dl)
    mean_error_dl = np.mean(error_record_dl)
    confidence_dl = confidence_mean(error_record_dl,0.05)[1]
    min_dl = min(confidence_dl)
    max_dl = max(confidence_dl)

    pl_list.append(pl)

    error_list_branch_red.append(mean_error_dl)
    error_list_branch_red_min.append(min_dl)
    error_list_branch_red_max.append(max_dl)

dic_error = {'bl':bl_list,
             'error_fold_white':error_list_fold_white,
             'min_fold_white':error_list_fold_white_min,
             'max_fold_white':error_list_fold_white_max,

             'kl':kl_list,          
             'error_hopf_white':error_list_hopf_white,
             'min_hopf_white':error_list_hopf_white_min,
             'max_hopf_white':error_list_hopf_white_max,

             'al':al_list,          
             'error_branch_white':error_list_branch_white,
             'min_branch_white':error_list_branch_white_min,
             'max_branch_white':error_list_branch_white_max,

             'ul1':ul_list1,
             'error_fold_red':error_list_fold_red,
             'min_fold_red':error_list_fold_red_min,
             'max_fold_red':error_list_fold_red_max,

             'ul2':ul_list2,          
             'error_hopf_red':error_list_hopf_red,
             'min_hopf_red':error_list_hopf_red_min,
             'max_hopf_red':error_list_hopf_red_max,

             'pl':pl_list,          
             'error_branch_red':error_list_branch_red,
             'min_branch_red':error_list_branch_red_min,
             'max_branch_red':error_list_branch_red_max}

csv_out = pandas.DataFrame(dic_error)
csv_out.to_csv('../../results/single_DL_model/test_DL_fold.csv',header = True)