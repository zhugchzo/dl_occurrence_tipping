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

txt = open('../empirical_data/microcosm.txt')
datalines = txt.readlines()
dataset = []
for data in datalines:
    data = data.strip().split('\t')
    dataset.append(data)

clean_NA = []
for i in range(1,len(dataset)-1):
    if dataset[i][1] != 'NA' and dataset[i+1][1] == 'NA':
        clean_NA.append(i)
    if dataset[i-1][1] == 'NA' and dataset[i][1] != 'NA':
        clean_NA.append(i)

for i in range(0,len(clean_NA),2):
    indexmin = clean_NA[i]
    indexmax = clean_NA[i+1]
    NA_approximate = np.linspace(float(dataset[indexmin][1]),float(dataset[indexmax][1]),indexmax-indexmin+1)
    for j in range(indexmin,indexmax+1):
        dataset[j][1] = NA_approximate[j-indexmin]

for data in dataset:
    data[0] = 500 + float(data[0])*23
    data[1] = float(data[1])

dataset_1 = dataset[:3949] # 0-14
dataset_2 = dataset[282:4231] # 1-15
dataset_3 = dataset[564:4512] # 2-16
dataset_4 = dataset[846:4794] # 3-17

dataset_sum = [dataset_1,dataset_2,dataset_3,dataset_4]
par_range_sum = ['0-14','1-15','2-16','3-17']

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

if not os.path.exists('../data_nus/microcosm_fold'):
    os.makedirs('../data_nus/microcosm_fold')

for i in np.arange(numSims)+1:
    df_resids = df_ews[df_ews['tsid'] == i][['Time','Residuals','b']]
    filepath_resids='../data_nus/microcosm_fold/microcosm_fold_400_resids_{}.csv'.format(par_range_sum[i-1])
    df_resids.to_csv(filepath_resids,index=False)

    df_state = df_ews[df_ews['tsid'] == i][['Time','State variable','b']]

    # Get the minimum and maximum values of the original 'b' column
    b_min = df_state['b'].min()
    b_max = df_state['b'].max()

    # Create a new uniformly distributed 'b' column
    new_b_values = np.linspace(b_min, b_max, series_len)

    # Interpolate the 'State variable'
    interpolated_values = np.interp(new_b_values, df_state['b'], df_state['State variable'])

    # Create a new DataFrame for interpolated state variable
    interpolated_df = pd.DataFrame({
        'Time': np.arange(series_len),
        'State variable': interpolated_values,
        'b': new_b_values
    })

    filepath_state='../data_nus/microcosm_fold/microcosm_fold_400_state_{}.csv'.format(par_range_sum[i-1])
    df_state.to_csv(filepath_state,index=False)

    filepath_interpolate='../data_nus/microcosm_fold/microcosm_fold_400_interpolate_{}.csv'.format(par_range_sum[i-1])
    interpolated_df.to_csv(filepath_interpolate,index=False)