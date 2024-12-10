#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 16 2024

@author: Chengzuo Zhuge

Modified by: Thomas M. Bury

Simulate May's harvesting model 
Simulations going through Fold bifurcation
Compute lag-1 autocorrelation by degenerate fingerprinting
Compute DEV

"""

# import python libraries
import numpy as np
import pandas as pd
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

#--------------------------------
# Global parameters
#–-----------------------------


# Simulation parameters
dt = 0.01
t0 = 0
tburn = 500 # burn-in period
numSims = 10
sigma = 0.01 # noise intensity

# degenerate fingerprinting parameters
auto_lag = 1 # autocorrelation lag times
ews = ['ac1']
rw_degf = 0.1

#----------------------------------
# Simulate model
#----------------------------------

# Model

def de_fun(x,g,k,h,s):
    return g*x*(1-x/k) - h*x**2/(s**2 + x**2)

def recov_fun(x,g,k,h,s):
    rrate = g - 2*g*x/k - 2*s**2*h*x/(s**2 + x**2)**2
    return rrate
    
# Model parameters
g = 1 # growth rate
k = 1 # carrying capacity
s = 0.1 # half-saturation constant of harvesting function

bcrit = 0.260437 # bifurcation point (computed in Mathematica)
bh = 1.2*bcrit # bifurcation parameter high

for i in np.linspace(0,0.2,11):

    bl = round(i,2) # bifurcation parameter low

    for c_rate in [1e-5,2e-5,3e-5,4e-5,5e-5]:

        sim_len = int((bh-bl)*dt/c_rate)

        # Initialise a list to collect trajectories
        list_traj1_append = []
        list_distance_1 = []
        list_traj2_append = []
        list_distance_2 = []

        list_trans_point = []

        j = 0
        # loop over simulations
        print('\nBegin simulations \n')
        while j < numSims:    

            x0 = 0.8197 # intial condition (equilibrium value computed in Mathematica)
            equi0 = 0.8197

            tmax = int(np.random.uniform(250,500)) # randomly selected sequence length
            n = np.random.uniform(5,500) # length of the randomly selected sequence to the bifurcation
            series_len = tmax + int(n)

            t = np.arange(t0,sim_len,dt)

            x = np.zeros(len(t))
            equi = np.zeros(len(t))
            rrate = np.zeros(len(t))

            b = pd.Series(np.linspace(bl,bh,len(t)),index=t)

            # Create brownian increments (s.d. sqrt(dt))
            dW_burn = np.random.normal(loc=0, scale=sigma*np.sqrt(dt), size = int(tburn/dt))
            dW = np.random.normal(loc=0, scale=sigma*np.sqrt(dt), size = len(t))

            # Run burn-in period on x0
            for i in range(int(tburn/dt)):
                x0 = x0 + de_fun(x0,g,k,b[0],s)*dt + dW_burn[i]
                equi0 = equi0 + de_fun(equi0,g,k,b[0],s)*dt
            
            rrate0 = recov_fun(equi0,g,k,b[0],s)

            # Initial condition post burn-in period
            x[0]=x0
            equi[0]=equi0
            rrate[0]=rrate0
            
            rrate_record = []

            # Run simulation
            for i in range(len(t)-1):
                x[i+1] = x[i] + de_fun(x[i],g,k,b.iloc[i],s)*dt + dW[i]
                equi[i+1] = equi[i] + de_fun(equi[i],g,k,b.iloc[i],s)*dt
                rrate[i+1] = recov_fun(equi[i+1],g,k,b.iloc[i+1],s)
                # make sure that state variable remains >= 0
                if x[i+1] < 0:
                    x[i+1] = 0
                # Determine the tipping point by the recovery rate changing from negative to positive
                if rrate[i] < 0 and rrate[i+1] > 0:
                    rrate_record.append(i+1)
                    break

            if len(rrate_record) != 0:

                trans_time = rrate_record[0] # the time of tipping point

                ts = np.arange(0,trans_time)

                c1 = np.linspace(1,len(ts)-2,len(ts)-2)
                np.random.shuffle(c1)
                c1 = list(np.sort(c1[0:series_len-2]))
                c1.insert(0,0)
                c1.append(len(ts)-1)

                c2 = np.linspace(0,trans_time,series_len)
                c2 = list(map(int,c2))

            else:
                print('Cant find rrate = 0')
                continue
                    
            # Store series data in a temporary DataFrame
            data = {'tsid': (j+1)*np.ones(len(t)),'Time': t,'x': x,'b': b.values}
            df_temp = pd.DataFrame(data)
            trans_point = df_temp.loc[trans_time-1,'b']

            df_temp_1 = df_temp.iloc[c1].copy() # irregularly-sampled time series
            df_temp_1['Time'] = np.arange(0,series_len)
            df_temp_1.set_index('Time', inplace=True)
            df_cut_1 = df_temp_1.iloc[0:tmax].copy()

            # Get the minimum and maximum values of the original 'b' column
            b_min = df_cut_1['b'].min()
            b_max = df_cut_1['b'].max()

            # Create a new uniformly distributed 'b' column
            new_b_values = np.linspace(b_min, b_max, tmax)

            # Interpolate the 'State variable'
            interpolated_values = np.interp(new_b_values, df_cut_1['b'], df_cut_1['x'])

            df_cut_1['interpolated_x'] = interpolated_values
            df_cut_1['interpolated_b'] = new_b_values

            df_temp_2 = df_temp.iloc[c2].copy() # regularly-sampled time series
            df_temp_2['Time'] = np.arange(0,series_len)
            df_temp_2.set_index('Time', inplace=True)
            df_cut_2 = df_temp_2.iloc[0:tmax].copy()

            nearest_point_1 = df_cut_1.loc[tmax-1,'b']
            nearest_point_2 = df_cut_2.loc[tmax-1,'b']
            distance_1 = trans_point - nearest_point_1
            distance_2 = trans_point - nearest_point_2

            # Append to list
            list_traj1_append.append(df_cut_1)
            list_distance_1.append(distance_1)
            list_traj2_append.append(df_cut_2)
            list_distance_2.append(distance_2)

            list_trans_point.append(trans_point)

            print('Simulation '+str(j+1)+' complete')

            j += 1

        #  Concatenate DataFrame from each tsid
        df_traj_1 = pd.concat(list_traj1_append) # irregularly-sampled time series
        df_traj_2 = pd.concat(list_traj2_append) # regularly-sampled time series
        df_result_1 = pd.DataFrame(data=None,columns=['trans point','distance'])
        df_result_2 = pd.DataFrame(data=None,columns=['trans point','distance'])

        df_result_1['trans point'] = list_trans_point
        df_result_1['distance'] = list_distance_1
        df_result_2['trans point'] = list_trans_point
        df_result_2['distance'] = list_distance_2

        # Create directories for output
        
        if not os.path.exists('../data_nus'):
            os.makedirs('../data_nus')

        if not os.path.exists('../data_us'):
            os.makedirs('../data_us')

        if not os.path.exists('../data_nus/may_fold_white/{}/{}/temp'.format(bl,c_rate)):
            os.makedirs('../data_nus/may_fold_white/{}/{}/temp'.format(bl,c_rate))

        if not os.path.exists('../data_us/may_fold_white/{}/{}/temp'.format(bl,c_rate)):
            os.makedirs('../data_us/may_fold_white/{}/{}/temp'.format(bl,c_rate))

        # Filter time-series
        df_traj_1.to_csv('../data_nus/may_fold_white/{}/{}/temp/df_traj.csv'.format(bl,c_rate))
        df_traj_2.to_csv('../data_us/may_fold_white/{}/{}/temp/df_traj.csv'.format(bl,c_rate))

        if not os.path.exists('../data_nus/may_fold_white/{}/{}/dev'.format(bl,c_rate)):
            os.makedirs('../data_nus/may_fold_white/{}/{}/dev'.format(bl,c_rate))

        if not os.path.exists('../data_us/may_fold_white/{}/{}/dev'.format(bl,c_rate)):
            os.makedirs('../data_us/may_fold_white/{}/{}/dev'.format(bl,c_rate))

        #----------------------
        # Compute DEV for each tsid 
        #----------------------
        
        # Compute DEV for irregularly-sampled time series
        print('\nBegin DEV computation\n')

        for i in range(numSims):
            df_traj_x = df_traj_1[df_traj_1['tsid'] == i+1]['interpolated_x']
            df_traj_b = df_traj_1[df_traj_1['tsid'] == i+1]['interpolated_b']
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
            # Export DEV as individual files for dynamical eigenvalue
            filepath_dev = '../data_nus/may_fold_white/{}/{}/dev/may_fold_dev_{}.csv'.format(bl,c_rate,i+1)
            df_dev.to_csv(filepath_dev,index=False)

        # Compute DEV for regularly-sampled time series
        for i in range(numSims):
            df_traj_x = df_traj_2[df_traj_2['tsid'] == i+1]['x']
            df_traj_b = df_traj_2[df_traj_2['tsid'] == i+1]['b']
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
            # Export DEV as individual files for dynamical eigenvalue
            filepath_dev = '../data_us/may_fold_white/{}/{}/dev/may_fold_dev_{}.csv'.format(bl,c_rate,i+1)
            df_dev.to_csv(filepath_dev,index=False)

            print('DEV for realisation '+str(i+1)+' complete')

        #----------------------
        # Compute lag-1 autocorrelation for each tsid 
        #----------------------

        # set up a list to store output dataframes from lag-1 autocorrelation
        appended_degf_1 = []
        appended_degf_2 = []  
    
        print('\nBegin lag-1 autocorrelation computation\n')
    # Compute lag-1 autocorrelation for irregularly-sampled time series
        for i in range(numSims):
            df_traj_temp = df_traj_1[df_traj_1['tsid'] == i+1][['interpolated_x','interpolated_b']]
            df_traj_x = df_traj_temp['interpolated_x']
            ews_dic = ewstools.TimeSeries(data = df_traj_x)
            ews_dic.compute_auto(lag = auto_lag, rolling_window = rw_degf)

            df_degf_temp = ews_dic.ews
            df_degf_temp['tsid'] = i+1
            df_degf_temp['b'] = df_traj_temp['interpolated_b']
                
            # Add DataFrames to list
            appended_degf_1.append(df_degf_temp)

        # Compute lag-1 autocorrelation for regularly-sampled time series 
        for i in range(numSims):
            df_traj_temp = df_traj_2[df_traj_2['tsid'] == i+1][['x','b']]
            df_traj_x = df_traj_temp['x']
            ews_dic = ewstools.TimeSeries(data = df_traj_x)
            ews_dic.compute_auto(lag = auto_lag, rolling_window = rw_degf)

            df_degf_temp = ews_dic.ews
            df_degf_temp['tsid'] = i+1
            df_degf_temp['b'] = df_traj_temp['b']
                
            # Add DataFrames to list
            appended_degf_2.append(df_degf_temp)
                
            print('lag-1 autocorrelation for realisation '+str(i+1)+' complete')

        # Concatenate EWS DataFrames
        df_traj_1.reset_index(inplace=True)
        df_traj_2.reset_index(inplace=True)
        df_degf_1 = pd.concat(appended_degf_1).reset_index()
        df_degf_2 = pd.concat(appended_degf_2).reset_index()

        # Create directories for output
        if not os.path.exists('../data_nus/may_fold_white/{}/{}/sims'.format(bl,c_rate)):
            os.makedirs('../data_nus/may_fold_white/{}/{}/sims'.format(bl,c_rate))

        if not os.path.exists('../data_nus/may_fold_white/{}/{}/ac'.format(bl,c_rate)):
            os.makedirs('../data_nus/may_fold_white/{}/{}/ac'.format(bl,c_rate))

        if not os.path.exists('../data_us/may_fold_white/{}/{}/sims'.format(bl,c_rate)):
            os.makedirs('../data_us/may_fold_white/{}/{}/sims'.format(bl,c_rate))

        if not os.path.exists('../data_us/may_fold_white/{}/{}/ac'.format(bl,c_rate)):
            os.makedirs('../data_us/may_fold_white/{}/{}/ac'.format(bl,c_rate))

        # Export time series as individual files for training ML
        # Export lag-1 autocorrelation as individual files for degenerate fingerprinting
        for i in np.arange(numSims)+1:
            df_sims = df_traj_1[df_traj_1['tsid'] == i][['Time','x','b']]
            df_ac = df_degf_1[df_degf_1['tsid'] == i][['ac1','b']]
            df_ac = df_ac[df_ac['ac1'].notna()]
            filepath_sims='../data_nus/may_fold_white/{}/{}/sims/may_fold_sims_{}.csv'.format(bl,c_rate,i)
            filepath_ac='../data_nus/may_fold_white/{}/{}/ac/may_fold_ac_{}.csv'.format(bl,c_rate,i)
            df_sims.to_csv(filepath_sims,index=False)
            df_ac.to_csv(filepath_ac,index=False)

        for i in np.arange(numSims)+1:
            df_sims = df_traj_2[df_traj_2['tsid'] == i][['Time','x','b']]
            df_ac = df_degf_2[df_degf_2['tsid'] == i][['ac1','b']]
            df_ac = df_ac[df_ac['ac1'].notna()]
            filepath_sims='../data_us/may_fold_white/{}/{}/sims/may_fold_sims_{}.csv'.format(bl,c_rate,i)
            filepath_ac='../data_us/may_fold_white/{}/{}/ac/may_fold_ac_{}.csv'.format(bl,c_rate,i)
            df_sims.to_csv(filepath_sims,index=False)
            df_ac.to_csv(filepath_ac,index=False)

        df_result_1.to_csv('../data_nus/may_fold_white/{}/{}/may_fold_result.csv'.format(bl,c_rate))
        df_result_2.to_csv('../data_us/may_fold_white/{}/{}/may_fold_result.csv'.format(bl,c_rate))

        print('bl = {}, c_rate = {} has finished'.format(bl,c_rate))
