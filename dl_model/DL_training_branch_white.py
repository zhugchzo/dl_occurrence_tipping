# Python script to train a neural network using Keras library.

import os
import zipfile

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import pandas
import numpy as np
import random
import sys

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Reshape

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime

from sklearn.metrics import r2_score

random.seed(datetime.now())

# Get command line parameters
kk_num = int(sys.argv[1]) # index for neutral network

single_train_numbers = 100000 # volume of training data for each changing direction of parameter (increase / decrease)
seq_len = 500  # length of input time series
pad_left = 250 # padding length of input time series

set_size = int(single_train_numbers)+1  # set to size of time series library plus 1
train_numbers = int(single_train_numbers)*2 # volume of training data 100000*2

zf_branch_f = zipfile.ZipFile('/output_forward/ts_500/branch/combined/output_resids.zip')
zf_branch_r = zipfile.ZipFile('/output_reverse/ts_500/branch/combined/output_resids.zip')
sequences = list()
normalization = list()

#branch
for i in range (1,set_size):
    df_f = pandas.read_csv(zf_branch_f.open('output_resids/white_resids'+str(i)+'.csv'))
    df_r = pandas.read_csv(zf_branch_r.open('output_resids/white_resids'+str(i)+'.csv'))
    keep_col = ['Residuals','b']
    new_f = df_f[keep_col]
    new_r = df_r[keep_col]
    values_f = new_f.values
    values_r = new_r.values

    values_fo = df_f['b'].iloc[-1]
    values_ro = df_r['b'].iloc[-1]

    sequences.append(values_f)
    sequences.append(values_r)
    normalization.append([0,values_fo])
    normalization.append([0,values_ro])

sequences = np.array(sequences)

#Padding input sequences
for i in range(train_numbers):
    pad_length = int(pad_left*random.uniform(0, 1))
    for j in range(0,pad_length):     
        sequences[i,j][0] = 0
        sequences[i,j][1] = 0
    values_s = sequences[i,pad_length][1]
    normalization[i][0] = values_s
    normalization[i][1] = normalization[i][1] - values_s

# training labels file
label_f='/output_forward/ts_500/branch/combined/values.csv'
label_r='/output_reverse/ts_500/branch/combined/values.csv'
targets_f = pandas.read_csv(label_f,header = 0)####
targets_r = pandas.read_csv(label_r,header = 0)
targets_value = []
for i in range(single_train_numbers):

    targets_f_value = targets_f.iloc[i,2]
    targets_r_value = targets_r.iloc[i,2]

    targets_f_value = (targets_f_value - normalization[2*i][0])/normalization[2*i][1]
    targets_r_value = (targets_r_value - normalization[2*i+1][0])/normalization[2*i+1][1]

    targets_value.append(targets_f_value)
    targets_value.append(targets_r_value)

targets_value = np.array(targets_value)

# train/validation/test split denotations
groups_f_name='/output_forward/ts_500/branch/combined/groups.csv'
#groups_r_name='/output_reverse/ts_500/branch/combined/groups.csv'
groups_f = pandas.read_csv(groups_f_name,header = 0)
#groups_r = pandas.read_csv(groups_f_name,header = 0)
groups_f = groups_f.values[:,1]
#groups_r = groups_r.values[:,1]
groups = np.concatenate([groups_f,groups_f])

# normalizing input time series by the average. 
for i in range(train_numbers):
    values_avg = 0.0
    count_avg = 0
    for j in range(0,seq_len):
        if sequences[i,j][0]!= 0 or sequences[i,j][1]!= 0:
            values_avg = values_avg + abs(sequences[i,j][0])                                       
            count_avg = count_avg + 1
        
    if count_avg != 0:
        values_avg = values_avg/count_avg
        for j in range (0,seq_len):
            if sequences[i,j][0]!= 0 or sequences[i,j][1]!= 0:
                sequences[i,j][0]= sequences[i,j][0]/values_avg
                sequences[i,j][1]= (sequences[i,j][1]-normalization[i][0])/normalization[i][1]

final_seq = sequences

# apply train/test/validation labels
train = [final_seq[i] for i in range(len(groups)) if (groups[i]==1)]
validation = [final_seq[i] for i in range(len(groups)) if groups[i]==2]
test = [final_seq[i] for i in range(len(groups)) if groups[i]==3]

train_target = [targets_value[i] for i in range(len(groups)) if (groups[i]==1)]
validation_target = [targets_value[i] for i in range(len(groups)) if groups[i]==2]
test_target = [targets_value[i] for i in range(len(groups)) if groups[i]==3]


train = np.array(train)
validation = np.array(validation)
test = np.array(test)


train_target = np.array(train_target)
validation_target = np.array(validation_target)
test_target = np.array(test_target)


train = train.reshape(-1,500,2,1)
validation = validation.reshape(-1,500,2,1)
test = test.reshape(-1,500,2,1)

# hyperparameter settings
pool_size_param = (3,1)
filters_param = 30
learning_rate_param = 0.005
dropout_percent = 0.1
mem_cells = 40
mem_cells2 = 30
kernel_size_param = (8,2)
epoch_param = 500
batch_param = 1024
initializer_param = 'lecun_normal'

model = Sequential()

# add layers
model.add(Conv2D(filters=filters_param, kernel_size=kernel_size_param, activation='relu', padding='same',input_shape=(seq_len, 2, 1),
kernel_initializer = initializer_param))
    
model.add(Dropout(dropout_percent))
model.add(MaxPooling2D(pool_size=pool_size_param))
model.add(Reshape((-1,filters_param)))
# LSTM1
model.add(LSTM(mem_cells, return_sequences=True, kernel_initializer = initializer_param))
model.add(Dropout(dropout_percent))
# LSTM2
model.add(LSTM(mem_cells2, kernel_initializer = initializer_param))
model.add(Dropout(dropout_percent))

model.add(Dense(units=1, activation=None,kernel_initializer = initializer_param))

a_name="best_model_branch_white_"
b_name = kk_num
c_name = ".pkl"
model_name = a_name+str(b_name)+c_name

# Set up optimiser
adam = Adam(learning_rate=learning_rate_param)
chk = ModelCheckpoint(model_name, monitor='val_mape', save_best_only=True, mode='min', verbose=1)
model.compile(loss='mse', optimizer=adam, metrics=['mape'])
model.fit(train, train_target, epochs=epoch_param, batch_size=batch_param, callbacks=[chk], validation_data=(validation,validation_target))

test_target = pandas.DataFrame(test_target,columns=['times'])

model_name = "best_model_branch_white_"+str(kk_num)+".pkl"

model = load_model(model_name)

test_preds = model.predict(test)

test_preds = pandas.DataFrame(test_preds,columns=['times'])

pred_acc = r2_score(test_target,test_preds)

result_acc = open('branch_white_result.txt','a+')
result_acc.write(str(pred_acc)+'\n')