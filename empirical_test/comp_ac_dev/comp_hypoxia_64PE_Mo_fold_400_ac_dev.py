import pandas as pd
import numpy as np
import ewstools
import os
from BB import ac_red

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

series_len = 400
numSims = 4
# EWS parameters
dt2 = 1 # spacing between time-series for EWS computation
rw = 0.5 # rolling window
span = 0.2 # bandwidth
lags = [1] # autocorrelation lag times
ews = ['var','ac']

# Create directories for output
if not os.path.exists('../data_nus'):
    os.makedirs('../data_nus')

if not os.path.exists('../data_nus/hypoxia_64PE_Mo_fold'):
    os.makedirs('../data_nus/hypoxia_64PE_Mo_fold')

par_range_sum = ['160-140','159-139','158-138','157-137']

for c in range(numSims):

    interpolated_df = pd.read_csv('../data_nus/hypoxia_64PE_Mo_fold/hypoxia_64PE_Mo_fold_400_interpolate_{}.csv'.format(par_range_sum[c]))

    df_state = interpolated_df[['State variable','b']]

    x = df_state['State variable']
    b = df_state['b']
    ac_r,b_ac = ac_red(x,b,rw)

    df_ac = pd.DataFrame(data=None,columns=['AC red','b'])
    df_ac['AC red'] = ac_r
    df_ac['b'] = b_ac

    # Compute DEV
    df_traj_x = df_state['State variable']
    df_traj_b = df_state['b']
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
    '''
    dev_b = r(rscript)[:,1]
    dev_x = r(rscript)[:,2]

    df_dev = pd.DataFrame(data=None,columns=['DEV','b'])
    df_dev['DEV'] = dev_x
    df_dev['b'] = dev_b

    print('EWS for realisation '+str(c+1)+' complete')

    # Export DEV as individual files for dynamical eigenvalue
    filepath_dev = '../data_nus/hypoxia_64PE_Mo_fold/hypoxia_64PE_Mo_fold_400_dev_{}.csv'.format(par_range_sum[c])
    df_dev.to_csv(filepath_dev,index=False)

    # Export lag-1 autocorrelation red as individual files for BB method
    filepath_ac = '../data_nus/hypoxia_64PE_Mo_fold/hypoxia_64PE_Mo_fold_400_ac_{}.csv'.format(par_range_sum[c])
    df_ac.to_csv(filepath_ac,index=False)



    