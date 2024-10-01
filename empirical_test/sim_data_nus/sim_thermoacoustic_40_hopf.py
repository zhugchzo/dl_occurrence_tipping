import pandas as pd
import numpy as np
import os
import ewstools

from rpy2.robjects import r
from rpy2.robjects.packages import importr
from rpy2.robjects import globalenv
from rpy2.robjects import pandas2ri

pandas2ri.activate()
importr("rEDM")

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)

# Get the directory containing the current file
current_dir = os.path.dirname(current_file_path)

# Change the working directory to the directory of the current file
os.chdir(current_dir)

# EWS parameters
auto_lag = 1 # autocorrelation lag times
ews = ['ac1']
rw = 0.1

txt = open('../empirical_data/thermoacoustic_40mv.txt')
datalines = txt.readlines()
dataset = []
for data in datalines:
    data = data.strip().split()
    dataset.append(data)

for data in dataset:
    data[0] = (float(data[0])-3.65629963087196300E+9)*0.04
    data[1] = float(data[1])*1000/0.2175

# dataset[0:int(6E5/(2.4/1.76))]  0-1.76 (the tipping point)

sample_start = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6]

for ss in sample_start:

    if not os.path.exists('../data_nus/thermoacoustic_40_hopf/{}'.format(ss)):
        os.makedirs('../data_nus/thermoacoustic_40_hopf/{}'.format(ss))

    if ss == 0:

        dataset_s = np.array(dataset[ss : int(6E5/(2.4/1.76))])

    else:

        dataset_s = np.array(dataset[int(6E5/(2.4/ss)) : int(6E5/(2.4/1.76))])

    tmax = 400 # sequence length
    n = np.random.uniform(4,400) # length of the randomly selected sequence to the bifurcation
    series_len = tmax + int(n)

    d = np.arange(0,len(dataset_s))
    np.random.shuffle(d)
    d = list(np.sort(d[0:series_len]))
    dataset_dl = dataset_s[d]
    dataset_dl = dataset_dl[0:tmax]

    df_resids = pd.DataFrame(data=None,columns=['x','b'])
    df_resids['x'] = dataset_dl[:,1]
    df_resids['b'] = dataset_dl[:,0]

    state_tseries = df_resids['x']
    ts = ewstools.TimeSeries(data=state_tseries)
    ts.detrend(method='Lowess', span=0.2)

    df_resids['residuals'] = ts.state['residuals']
    df_resids['Time'] = np.arange(0,tmax)
    df_resids.set_index('Time', inplace=True)

    filepath_resids='../data_nus/thermoacoustic_40_hopf/{}/thermoacoustic_40_hopf_resids.csv'.format(ss)
    df_resids.to_csv(filepath_resids)

    # Get the minimum and maximum values of the original 'b' column
    b_min = df_resids['b'].min()
    b_max = df_resids['b'].max()

    # Create a new uniformly distributed 'b' column
    new_b_values = np.linspace(b_min, b_max, tmax)

    # Interpolate the 'State variable'
    interpolated_values = np.interp(new_b_values, df_resids['b'], df_resids['x'])

    # Create a new DataFrame for interpolated state variable
    df_interpolated = pd.DataFrame({
        'Time': np.arange(tmax),
        'x': interpolated_values,
        'b': new_b_values
    })

    filepath_resids='../data_nus/thermoacoustic_40_hopf/{}/thermoacoustic_40_hopf_resids.csv'.format(ss)
    df_resids.to_csv(filepath_resids,index=False)

    filepath_interpolate='../data_nus/thermoacoustic_40_hopf/{}/thermoacoustic_40_hopf_interpolate.csv'.format(ss)
    df_interpolated.to_csv(filepath_interpolate,index=False)

    # Compute AC
    df_traj_temp = df_interpolated[['x','b']]
    df_traj_x = df_traj_temp['x']
    ews_dic = ewstools.TimeSeries(data = df_traj_x)
    ews_dic.compute_auto(lag = auto_lag, rolling_window = rw)

    df_ac = pd.DataFrame(data=None,columns=['ac1','b'])
    df_ac['ac1'] = ews_dic.ews['ac1']
    df_ac['b'] = df_traj_temp['b']
    df_ac = df_ac[df_ac['ac1'].notna()]

    filepath_ac = '../data_nus/thermoacoustic_40_hopf/{}/thermoacoustic_40_hopf_ac.csv'.format(ss)
    df_ac.to_csv(filepath_ac)

    # Compute DEV
    df_traj_x = df_interpolated['x']
    df_traj_b = df_interpolated['b']
    rdf_x = pandas2ri.py2rpy(df_traj_x)
    rdf_b = pandas2ri.py2rpy(df_traj_b)
    globalenv['x_time_series'] = rdf_x
    globalenv['b_time_series'] = rdf_b
    rscript = '''
    E <- 5
    tau <- -2
    theta <- seq(0,2.5,by=0.5)
    window_size <- 100
    step_size <- 20

    window_indices <- seq(window_size, NROW(x_time_series), step_size)
    matrix_result <- matrix(NaN, nrow = length(window_indices), ncol = 3)
    index <- 0

    for(j in window_indices)
    {
        index <- index + 1
        rolling_window <- x_time_series[(j-window_size+1):j]
        b <- b_time_series[j]
        
        norm_rolling_window <- (rolling_window - mean(rolling_window, na.rm=TRUE))/sd(rolling_window, na.rm=TRUE)
        
        smap <- s_map(norm_rolling_window, E=E, tau=tau, theta=theta, silent=TRUE)

        best <- which.max(smap$rho)
        theta_best <- smap[best,]$theta
        
        smap <- s_map(norm_rolling_window, E=E, tau=tau, theta=theta_best, silent=TRUE, save_smap_coefficients=TRUE)
        
        smap_co <- smap$smap_coefficients[[1]]
        
        matrix_eigen <- matrix(NA, nrow = NROW(smap_co), ncol = 1)
        
        for(i in 1:NROW(smap_co))
        {
            if(!is.na(smap_co[i,2]))
            {
                M <- rbind(as.numeric(smap_co[i, 3:(E+2)]), cbind(diag(E - 1), rep(0, E - 1)))
                M_eigen <- eigen(M)$values
                lambda1 <- M_eigen[order(abs(M_eigen))[E]]
                
                matrix_eigen[i,1] <- abs(lambda1)
            }
        }
        
        matrix_result[index,1] <- j
        matrix_result[index,2] <- b
        matrix_result[index,3] <- mean(matrix_eigen[,1],na.rm=TRUE)
    }

    result <- matrix_result
    return(result)
    '''
    dev_b = r(rscript)[:,1]
    dev_x = r(rscript)[:,2]

    df_dev = pd.DataFrame(data=None,columns=['DEV','b'])
    df_dev['DEV'] = dev_x
    df_dev['b'] = dev_b

    filepath_dev = '../data_nus/thermoacoustic_40_hopf/{}/thermoacoustic_40_hopf_dev.csv'.format(ss)
    df_dev.to_csv(filepath_dev)