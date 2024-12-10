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
from tensorflow.keras.layers import Input

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime

random.seed(datetime.now())

# Get command line parameters
kk_num = int(sys.argv[1]) # index for neutral network
single_train_numbers = 25000 # volume of training data for each changing direction of parameter (increase / decrease), noise type (white / red) and bifurcation type (fold / hopf / transcritical (branch))
seq_len = 500  # length of input time series
pad_left = 250 # padding length of input time series

set_size = int(single_train_numbers)+1  # set to size of time series library plus 1
train_numbers = int(single_train_numbers)*12 # volume of training data 25000*12

zf_fold_f = zipfile.ZipFile('/increased_bifurcation/output/ts_500/fold/combined/output_resids.zip')
zf_fold_r = zipfile.ZipFile('/decreased_bifurcation/output/ts_500/fold/combined/output_resids.zip')
zf_hopf_f = zipfile.ZipFile('/increased_bifurcation/output/ts_500/hopf/combined/output_resids.zip')
zf_hopf_r = zipfile.ZipFile('/decreased_bifurcation/output/ts_500/hopf/combined/output_resids.zip')
zf_branch_f = zipfile.ZipFile('/increased_bifurcation/output/ts_500/branch/combined/output_resids.zip')
zf_branch_r = zipfile.ZipFile('/decreased_bifurcation/output/ts_500/branch/combined/output_resids.zip')

sequences = list()
normalization = list()

for i in range (1,set_size):

    df_fold_f_white = pandas.read_csv(zf_fold_f.open('output_resids/white_resids'+str(i)+'.csv'))
    df_fold_f_red = pandas.read_csv(zf_fold_f.open('output_resids/red_resids'+str(i)+'.csv'))
    df_fold_r_white = pandas.read_csv(zf_fold_r.open('output_resids/white_resids'+str(i)+'.csv'))
    df_fold_r_red = pandas.read_csv(zf_fold_r.open('output_resids/red_resids'+str(i)+'.csv'))

    df_hopf_f_white = pandas.read_csv(zf_hopf_f.open('output_resids/white_resids'+str(i)+'.csv'))
    df_hopf_f_red = pandas.read_csv(zf_hopf_f.open('output_resids/red_resids'+str(i)+'.csv'))
    df_hopf_r_white = pandas.read_csv(zf_hopf_r.open('output_resids/white_resids'+str(i)+'.csv'))
    df_hopf_r_red = pandas.read_csv(zf_hopf_r.open('output_resids/red_resids'+str(i)+'.csv'))

    df_branch_f_white = pandas.read_csv(zf_branch_f.open('output_resids/white_resids'+str(i)+'.csv'))
    df_branch_f_red = pandas.read_csv(zf_branch_f.open('output_resids/red_resids'+str(i)+'.csv'))
    df_branch_r_white = pandas.read_csv(zf_branch_r.open('output_resids/white_resids'+str(i)+'.csv'))
    df_branch_r_red = pandas.read_csv(zf_branch_r.open('output_resids/red_resids'+str(i)+'.csv'))

    keep_col_white = ['residuals','b']
    keep_col_red = ['residuals','b']

    new_fold_f_white = df_fold_f_white[keep_col_white]
    new_fold_f_red = df_fold_f_red[keep_col_red]
    new_fold_r_white = df_fold_r_white[keep_col_white]
    new_fold_r_red = df_fold_r_red[keep_col_red]

    new_hopf_f_white = df_hopf_f_white[keep_col_white]
    new_hopf_f_red = df_hopf_f_red[keep_col_red]
    new_hopf_r_white = df_hopf_r_white[keep_col_white]
    new_hopf_r_red = df_hopf_r_red[keep_col_red]

    new_branch_f_white = df_branch_f_white[keep_col_white]
    new_branch_f_red = df_branch_f_red[keep_col_red]
    new_branch_r_white = df_branch_r_white[keep_col_white]
    new_branch_r_red = df_branch_r_red[keep_col_red]

    values_fold_f_white = new_fold_f_white.values
    values_fold_f_red = new_fold_f_red.values
    values_fold_r_white = new_fold_r_white.values
    values_fold_r_red = new_fold_r_red.values

    values_hopf_f_white = new_hopf_f_white.values
    values_hopf_f_red = new_hopf_f_red.values
    values_hopf_r_white = new_hopf_r_white.values
    values_hopf_r_red = new_hopf_r_red.values

    values_branch_f_white = new_branch_f_white.values
    values_branch_f_red = new_branch_f_red.values
    values_branch_r_white = new_branch_r_white.values
    values_branch_r_red = new_branch_r_red.values

    values_fold_f_white_over = df_fold_f_white['b'].iloc[-1]
    values_fold_f_red_over = df_fold_f_red['b'].iloc[-1]
    values_fold_r_white_over = df_fold_r_white['b'].iloc[-1]
    values_fold_r_red_over = df_fold_r_red['b'].iloc[-1]

    values_hopf_f_white_over = df_hopf_f_white['b'].iloc[-1]
    values_hopf_f_red_over = df_hopf_f_red['b'].iloc[-1]
    values_hopf_r_white_over = df_hopf_r_white['b'].iloc[-1]
    values_hopf_r_red_over = df_hopf_r_red['b'].iloc[-1]

    values_branch_f_white_over = df_branch_f_white['b'].iloc[-1]
    values_branch_f_red_over = df_branch_f_red['b'].iloc[-1]
    values_branch_r_white_over = df_branch_r_white['b'].iloc[-1]
    values_branch_r_red_over = df_branch_r_red['b'].iloc[-1]

    sequences.append(values_fold_f_white)
    sequences.append(values_fold_f_red)
    sequences.append(values_fold_r_white)
    sequences.append(values_fold_r_red)

    sequences.append(values_hopf_f_white)
    sequences.append(values_hopf_f_red)
    sequences.append(values_hopf_r_white)
    sequences.append(values_hopf_r_red)

    sequences.append(values_branch_f_white)
    sequences.append(values_branch_f_red)
    sequences.append(values_branch_r_white)
    sequences.append(values_branch_r_red)

    normalization.append([0,values_fold_f_white_over])
    normalization.append([0,values_fold_f_red_over])
    normalization.append([0,values_fold_r_white_over])
    normalization.append([0,values_fold_r_red_over])

    normalization.append([0,values_hopf_f_white_over])
    normalization.append([0,values_hopf_f_red_over])
    normalization.append([0,values_hopf_r_white_over])
    normalization.append([0,values_hopf_r_red_over])

    normalization.append([0,values_branch_f_white_over])
    normalization.append([0,values_branch_f_red_over])
    normalization.append([0,values_branch_r_white_over])
    normalization.append([0,values_branch_r_red_over])

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
label_fold_f='/increased_bifurcation/output/ts_500/fold/combined/values.csv'
label_fold_r='/decreased_bifurcation/output/ts_500/fold/combined/values.csv'
label_hopf_f='/increased_bifurcation/output/ts_500/hopf/combined/values.csv'
label_hopf_r='/decreased_bifurcation/output/ts_500/hopf/combined/values.csv'
label_branch_f='/increased_bifurcation/output/ts_500/branch/combined/values.csv'
label_branch_r='/decreased_bifurcation/output/ts_500/branch/combined/values.csv'
targets_fold_f = pandas.read_csv(label_fold_f,header = 0)
targets_fold_r = pandas.read_csv(label_fold_r,header = 0)
targets_hopf_f = pandas.read_csv(label_hopf_f,header = 0)
targets_hopf_r = pandas.read_csv(label_hopf_r,header = 0)
targets_branch_f = pandas.read_csv(label_branch_f,header = 0)
targets_branch_r = pandas.read_csv(label_branch_r,header = 0)
targets_value = []
for i in range(single_train_numbers):

    targets_fold_f_value = targets_fold_f.iloc[i,2]
    targets_fold_r_value = targets_fold_r.iloc[i,2]
    targets_hopf_f_value = targets_hopf_f.iloc[i,2]
    targets_hopf_r_value = targets_hopf_r.iloc[i,2]
    targets_branch_f_value = targets_branch_f.iloc[i,2]
    targets_branch_r_value = targets_branch_r.iloc[i,2]

    targets_fold_f_white_value = (targets_fold_f_value - normalization[12*i][0])/normalization[12*i][1]
    targets_fold_f_red_value = (targets_fold_f_value - normalization[12*i+1][0])/normalization[12*i+1][1]
    targets_fold_r_white_value = (targets_fold_r_value - normalization[12*i+2][0])/normalization[12*i+2][1]
    targets_fold_r_red_value = (targets_fold_r_value - normalization[12*i+3][0])/normalization[12*i+3][1]

    targets_hopf_f_white_value = (targets_hopf_f_value - normalization[12*i+4][0])/normalization[12*i+4][1]
    targets_hopf_f_red_value = (targets_hopf_f_value - normalization[12*i+5][0])/normalization[12*i+5][1]
    targets_hopf_r_white_value = (targets_hopf_r_value - normalization[12*i+6][0])/normalization[12*i+6][1]
    targets_hopf_r_red_value = (targets_hopf_r_value - normalization[12*i+7][0])/normalization[12*i+7][1]

    targets_branch_f_white_value = (targets_branch_f_value - normalization[12*i+8][0])/normalization[12*i+8][1]
    targets_branch_f_red_value = (targets_branch_f_value - normalization[12*i+9][0])/normalization[12*i+9][1]
    targets_branch_r_white_value = (targets_branch_r_value - normalization[12*i+10][0])/normalization[12*i+10][1]
    targets_branch_r_red_value = (targets_branch_r_value - normalization[12*i+11][0])/normalization[12*i+11][1]

    targets_value.append(targets_fold_f_white_value)
    targets_value.append(targets_fold_f_red_value)
    targets_value.append(targets_fold_r_white_value)
    targets_value.append(targets_fold_r_red_value)

    targets_value.append(targets_hopf_f_white_value)
    targets_value.append(targets_hopf_f_red_value)
    targets_value.append(targets_hopf_r_white_value)
    targets_value.append(targets_hopf_r_red_value)

    targets_value.append(targets_branch_f_white_value)
    targets_value.append(targets_branch_f_red_value)
    targets_value.append(targets_branch_r_white_value)
    targets_value.append(targets_branch_r_red_value)

targets_value = np.array(targets_value)

# train/validation/test split denotations
groups_name='/increased_bifurcation/output/ts_500/fold/combined/groups.csv'
groups = pandas.read_csv(groups_name,header = 0)
groups = groups.values[:single_train_numbers,1]
groups = np.concatenate([groups,groups,groups,groups,groups,groups,groups,groups,groups,groups,groups,groups])

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
train = [final_seq[i] for i in range(train_numbers) if (groups[i]==1)]
validation = [final_seq[i] for i in range(train_numbers) if groups[i]==2]
test = [final_seq[i] for i in range(train_numbers) if groups[i]==3]

train_target = [targets_value[i] for i in range(train_numbers) if (groups[i]==1)]
validation_target = [targets_value[i] for i in range(train_numbers) if groups[i]==2]
test_target = [targets_value[i] for i in range(train_numbers) if groups[i]==3]


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
pool_size_param = (4,1)
filters_param = 60
learning_rate_param = 0.01
dropout_percent = 0.1
mem_cells = 40
mem_cells2 = 60
kernel_size_param = (10,2)
epoch_param = 200
batch_param = 1024
initializer_param = 'lecun_normal'

model = Sequential()

model.add(Input(shape=(seq_len, 2, 1)))

# add layers
model.add(Conv2D(filters=filters_param, kernel_size=kernel_size_param, activation='relu', padding='same',kernel_initializer = initializer_param))
    
model.add(Dropout(dropout_percent))
model.add(MaxPooling2D(pool_size=pool_size_param))
model.add(Reshape((-1,filters_param)))
# LSTM1
model.add(LSTM(mem_cells, return_sequences=True, kernel_initializer = initializer_param))
model.add(Dropout(dropout_percent))
# LSTM2
model.add(LSTM(mem_cells2, kernel_initializer = initializer_param))
model.add(Dropout(dropout_percent))

model.add(Dense(units=1, activation=None, kernel_initializer = initializer_param))

model_name = "best_model_"+str(kk_num)+".keras"

# Set up optimiser
adam = Adam(learning_rate=learning_rate_param)
chk = ModelCheckpoint(model_name, monitor='val_mape', save_best_only=True, mode='min', verbose=1)
model.compile(loss='mse', optimizer=adam, metrics=['mape'])
model.fit(train, train_target, epochs=epoch_param, batch_size=batch_param, callbacks=[chk], validation_data=(validation,validation_target))