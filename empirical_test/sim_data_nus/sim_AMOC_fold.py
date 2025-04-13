import pandas as pd
import numpy as np
import os
import ewstools

from BB import ac_red
from rpy2.robjects import r
from rpy2.robjects.packages import importr
from rpy2.robjects import globalenv
from rpy2.robjects import pandas2ri

pandas2ri.activate()
importr("rEDM")

# EWS parameters
rw_bb = 0.25

csv = pd.read_csv('../empirical_data/AMOC_data.csv') # 1758

dataset = csv.to_numpy()

sample_start = [0, 200, 400, 600, 800, 1000]

for ss in sample_start:

    if not os.path.exists('../data_nus/AMOC_fold/{}'.format(ss)):
        os.makedirs('../data_nus/AMOC_fold/{}'.format(ss))

    dataset_s = np.array(dataset[ss : 1758])

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

    filepath_resids='../data_nus/AMOC_fold/{}/AMOC_fold_resids_{}.csv'.format(ss)
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

    df_interpolated.set_index('Time', inplace=True)

    filepath_interpolate='../data_nus/AMOC_fold/{}/AMOC_fold_interpolate_{}.csv'.format(ss)
    df_interpolated.to_csv(filepath_interpolate)

    # Compute AC
    df_traj_temp = df_interpolated[['x','b']]
    x = df_traj_temp['x']
    b = df_traj_temp['b']
    ac_r,b_ac = ac_red(x,b,rw_bb)
    df_ac = pd.DataFrame(data=None,columns=['AC red','b'])
    df_ac['AC red'] = ac_r
    df_ac['b'] = b_ac

    filepath_ac = '../data_nus/AMOC_fold/{}/AMOC_fold_ac_{}.csv'.format(ss)
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

    filepath_dev = '../data_nus/AMOC_fold/{}/AMOC_fold_dev_{}.csv'.format(ss)
    df_dev.to_csv(filepath_dev)


