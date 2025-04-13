import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas
import os

kk = 10
seq_len = 500
numtest = 10

if not os.path.exists('../../results/saliency/global_fold'):
    os.makedirs('../../results/saliency/global_fold')

for c_rate in [-5e-7,-6e-7,-7e-7,-8e-7,-9e-7]:

    for n in range(1,numtest+1):

        df_resids = pandas.read_csv('../data_nus/global_fold_red/{}/{}/resids/global_fold_resids_'.format(1.4,c_rate)+str(n)+'.csv')

        keep_col_resids = ['residuals','u']
        new_f_resids = df_resids[keep_col_resids]

        values_s = df_resids['u'].iloc[0]
        values_o = df_resids['u'].iloc[-1]

        values_resids = new_f_resids.values

        k = seq_len-len(values_resids)

        # Padding input sequences for DL
        for j in range(k):
            values_resids = np.insert(values_resids,0,[[0,0]],axis=0)

        values_avg = 0.0
        count_avg = 0
        for j in range(0,seq_len):
            if values_resids[j][0]!= 0 or values_resids[j][1]!= 0:
                values_avg = values_avg + abs(values_resids[j][0])                                       
                count_avg = count_avg + 1
        if count_avg != 0:
            values_avg = values_avg/count_avg
            for j in range(0,seq_len):
                if values_resids[j][0]!= 0 or values_resids[j][1]!= 0:
                    values_resids[j][0]= values_resids[j][0]/values_avg
                    values_resids[j][1]= (values_resids[j][1]-values_s)/(values_o-values_s)

        saliency_feature_x = np.zeros(seq_len-k)
        saliency_feature_b = np.zeros(seq_len-k)

        for i in range(1,kk+1):

            model_name = '../../dl_model/best_model_{}.keras'.format(i)

            model = load_model(model_name)

            input_data = tf.convert_to_tensor(values_resids, dtype=tf.float32)
            input_data = tf.expand_dims(input_data, axis=0)

            with tf.GradientTape() as tape:
                tape.watch(input_data)
                predictions = model(input_data)

            gradients = tape.gradient(predictions, input_data)

            saliency_map_feature_1 = np.abs(gradients[0, :, 0])[k:]
            saliency_map_feature_2 = np.abs(gradients[0, :, 1])[k:]

            saliency_feature_x += saliency_map_feature_1
            saliency_feature_b += saliency_map_feature_2

        saliency_feature_x = saliency_feature_x / kk
        saliency_feature_b = saliency_feature_b / kk

        t = np.linspace(k+1, seq_len, num = seq_len-k)

        dic_saliency = {'Time':t, 'saliency_feature_x':saliency_feature_x, 'saliency_feature_b':saliency_feature_b}

        csv_saliency_out = pandas.DataFrame(dic_saliency)
        csv_saliency_out.to_csv('../../results/saliency/global_fold/saliency_feature_{}_{}.csv'.format(c_rate,n),header = True)