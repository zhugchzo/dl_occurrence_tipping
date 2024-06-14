import pandas as pd
import numpy as np
import ewstools
import os

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)

# Get the directory containing the current file
current_dir = os.path.dirname(current_file_path)

# Change the working directory to the directory of the current file
os.chdir(current_dir)

series_len = 400
numSims = 4
# EWS parameters
dt2 = 1 # spacing between time-series for EWS computation
rw = 0.25 # rolling window
span = 0.2 # bandwidth
lags = [1] # autocorrelation lag times
ews = ['var','ac']

txt = open('../empirical_data/hypoxia_64PE_Mo_fold.txt')
datalines = txt.readlines()
dataset_orgin = []
for data in datalines:
    data = data.strip().split('\t')
    dataset_orgin.append(data)

dataset = []
for data in dataset_orgin:
    data[0] = float(data[0])
    data[4] = float(data[4])
    dataset.append([data[0],data[4]])

dataset_1 = dataset[575:] # 160-142
dataset_2 = dataset[543:1114] # 159-141
dataset_3 = dataset[511:1079] # 158-140
dataset_4 = dataset[478:1046] # 157-139

dataset_1.reverse()
dataset_2.reverse()
dataset_3.reverse()
dataset_4.reverse()

dataset_sum = [dataset_1,dataset_2,dataset_3,dataset_4]
par_range_sum = ['160-142','159-141','158-140','157-139']

appended_ews = []

for c in range(numSims):

    dataset = np.array(dataset_sum[c])

    d = np.arange(0,len(dataset))
    np.random.shuffle(d)
    d = list(np.sort(d[0:series_len]))
    dataset_dl = dataset[d]

    df_mic = pd.DataFrame(data=None,columns=['x','b'])
    df_mic['x'] = dataset_dl[:,1]
    df_mic['b'] = dataset_dl[:,0]

    ews_dic = ewstools.core.ews_compute(df_mic['x'], 
                roll_window = rw,
                smooth='Lowess',
                span = span,
                lag_times = lags, 
                ews = ews)
    
    # The DataFrame of EWS
    df_ews_temp = ews_dic['EWS metrics']
    # Include a column in the DataFrames for realisation number and variable
    df_ews_temp['tsid'] = c+1
    df_ews_temp['b'] = df_mic['b']
        
    # Add DataFrames to list
    appended_ews.append(df_ews_temp)

    print('EWS for realisation '+str(c+1)+' complete')

# Concatenate EWS DataFrames
df_ews = pd.concat(appended_ews).reset_index()

#------------------------------------
# Export data 
#-----------------------------------

# Create directories for output
if not os.path.exists('../data_nus'):
    os.makedirs('../data_nus')

if not os.path.exists('../data_nus/hypoxia_64PE_Mo_fold'):
    os.makedirs('../data_nus/hypoxia_64PE_Mo_fold')

for i in np.arange(numSims)+1:
    df_resids = df_ews[df_ews['tsid'] == i][['Time','Residuals','b']]
    filepath_resids='../data_nus/hypoxia_64PE_Mo_fold/hypoxia_64PE_Mo_fold_400_resids_{}.csv'.format(par_range_sum[i-1])
    df_resids.to_csv(filepath_resids,index=False)

    df_state = df_ews[df_ews['tsid'] == i][['Time','State variable','b']]

    # Get the minimum and maximum values of the original 'b' column
    b_min = df_state['b'].min()
    b_max = df_state['b'].max()

    # Create a new uniformly distributed 'b' column
    new_b_values = np.linspace(b_min, b_max, series_len)

    # Interpolate the 'State variable'
    df_state_reverse = df_state.sort_values(by='b', ascending=True)
    interpolated_values = np.interp(new_b_values, df_state_reverse['b'], df_state_reverse['State variable'])

    # Create a new DataFrame for interpolated state variable
    interpolated_df = pd.DataFrame({
        'Time': np.arange(series_len),
        'State variable': interpolated_values[::-1],
        'b': new_b_values[::-1]
    })

    filepath_state='../data_nus/hypoxia_64PE_Mo_fold/hypoxia_64PE_Mo_fold_400_state_{}.csv'.format(par_range_sum[i-1])
    df_state.to_csv(filepath_state,index=False)

    filepath_interpolate='../data_nus/hypoxia_64PE_Mo_fold/hypoxia_64PE_Mo_fold_400_interpolate_{}.csv'.format(par_range_sum[i-1])
    interpolated_df.to_csv(filepath_interpolate,index=False)