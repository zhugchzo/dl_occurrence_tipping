#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 16 2024

@author: Chengzuo Zhuge

Simulate Amazon forest model 
Simulations going through transcritical bifurcation
Compute lag-1 autocorrelation by BB method
Compute DEV

"""

# import python libraries
import numpy as np
import pandas as pd
import os
import math
import random
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

#--------------------------------
# Global parameters
#–-----------------------------


# Simulation parameters
dt = 0.01
t0 = 0
tburn = 500 # burn-in period
numSims = 10
sigma = 0.01 # noise intensity

# EWS parameters
rw_bb = 0.25

#----------------------------------
# Simulate model
#----------------------------------

# Model

def de_fun(v,p,g):
    if v >= 0.1:
        return v*p*(1-v) - g*v
    else:
        return 0.1*p*(1-v) - g*v

def recov_fun(v,p,g):
    if v >= 0.1:
        rrate = p - g - 2*p*v
        return rrate
    else:
        rrate = -0.1*p - g
        return rrate


# Model parameters
g = 0.004

pbif = 0.004 # bifurcation point (computed in Mathematica)
ph = -0.1  # bifurcation parameter high

for i in np.linspace(0.9,0.1,11):
    
    pl = round(i,2) # bifurcation parameter low

    for c_rate in [-1e-5,-2e-5,-3e-5,-4e-5,-5e-5]:

        sim_len = int((ph-pl)*dt/c_rate)

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

            v0 = 1 # intial condition (equilibrium value computed in Mathematica)
            equi0 = 1

            ac = random.uniform(-1,1)
            
            tmax = int(np.random.uniform(250,500)) # randomly selected sequence length
            n_random = np.random.uniform(5,500) # length of the randomly selected sequence to the bifurcation
            series_len = tmax + int(n_random)

            t = np.arange(t0,sim_len,dt)
            v = np.zeros(len(t))
            equi = np.zeros(len(t))
            n = np.zeros(len(t))
            rrate = np.zeros(len(t))
            p = pd.Series(np.linspace(pl,ph,len(t)),index=t)
            
            # Create brownian increments (s.d. sqrt(dt))
            dW_burn = np.random.normal(loc=0, scale=sigma*np.sqrt(dt), size = int(tburn/dt))
            dW = np.random.normal(loc=0, scale=sigma*np.sqrt(dt), size = len(t))

            # Run burn-in period on x0
            for i in range(int(tburn/dt)):
                v0 = v0 + de_fun(v0,p[0],g)*dt + dW_burn[i]
                equi0 = equi0 + de_fun(equi0,p[0],g)*dt

            rrate0 = recov_fun(equi0,p[0],g)

            if math.isinf(v0):
                print('sim inf appears')
                v0 = 1 # intial condition (equilibrium value computed in Mathematica)
                equi0 = 1
                continue  

            # Initial condition post burn-in period
            v[0]=v0
            equi[0]=equi0
            rrate[0]=rrate0
            
            rrate_record = []

            # Run simulation
            for i in range(len(t)-1):
                v[i+1] = v[i] + de_fun(v[i],p.iloc[i],g)*dt + n[i]
                equi[i+1] = equi[i] + de_fun(equi[i],p.iloc[i],g)*dt
                n[i+1] = ac*n[i] + dW[i]
                rrate[i+1] = recov_fun(equi[i+1],p.iloc[i+1],g)

                if math.isinf(v[i+1]):
                    print('sim inf appears')
                    break             
                # Determine the tipping point by the recovery rate changing from negative to positive
                if rrate[i] < 0 and rrate[i+1] > 0:
                    rrate_record.append(i+1)
                    break

            if len(rrate_record) != 0:

                trans_time = rrate_record[0] # the time of tipping point

                ts = np.arange(0,trans_time)

                d1 = np.linspace(1,len(ts)-2,len(ts)-2)
                np.random.shuffle(d1)
                d1 = list(np.sort(d1[0:series_len-2]))
                d1.insert(0,0)
                d1.append(len(ts)-1)

                d2 = np.linspace(0,trans_time,series_len)
                d2 = list(map(int,d2))

            else:
                print('Cant find rrate = 0')
                continue

            # Store series data in a temporary DataFrame
            data = {'tsid': (j+1)*np.ones(len(t)),'Time': t,'x': v, 'p': p.values}
            df_temp = pd.DataFrame(data)
            trans_point = df_temp.loc[trans_time-1,'p']
        
            df_temp_1 = df_temp.iloc[d1].copy() # irregularly-sampled time series
            df_temp_1['Time'] = np.arange(0,series_len)
            df_temp_1.set_index('Time', inplace=True)
            df_cut_1 = df_temp_1.iloc[0:tmax].copy()

            # Get the minimum and maximum values of the original 'p' column
            p_min = df_cut_1['p'].min()
            p_max = df_cut_1['p'].max()

            # Create a new uniformly distributed 'p' column
            new_p_values = np.linspace(p_min, p_max, tmax)

            # Interpolate the 'State variable'
            df_cut_1_reverse = df_cut_1.sort_values(by='p', ascending=True)
            interpolated_values = np.interp(new_p_values, df_cut_1_reverse['p'], df_cut_1_reverse['x'])

            df_cut_1['interpolated_x'] = interpolated_values[::-1]
            df_cut_1['interpolated_p'] = new_p_values[::-1]

            df_temp_2 = df_temp.iloc[d2].copy() # regularly-sampled time series
            df_temp_2['Time'] = np.arange(0,series_len)
            df_temp_2.set_index('Time', inplace=True)
            df_cut_2 = df_temp_2.iloc[0:tmax].copy()

            nearest_point_1 = df_cut_1.loc[tmax-1,'p']
            nearest_point_2 = df_cut_2.loc[tmax-1,'p']
            distance_1 = abs(trans_point - nearest_point_1)
            distance_2 = abs(trans_point - nearest_point_2)

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

        if not os.path.exists('../data_nus/amazon_branch_red/{}/{}/temp'.format(pl,c_rate)):
            os.makedirs('../data_nus/amazon_branch_red/{}/{}/temp'.format(pl,c_rate))

        if not os.path.exists('../data_us/amazon_branch_red/{}/{}/temp'.format(pl,c_rate)):
            os.makedirs('../data_us/amazon_branch_red/{}/{}/temp'.format(pl,c_rate))

        # Filter time-series
        df_traj_1.to_csv('../data_nus/amazon_branch_red/{}/{}/temp/df_traj.csv'.format(pl,c_rate))
        df_traj_2.to_csv('../data_us/amazon_branch_red/{}/{}/temp/df_traj.csv'.format(pl,c_rate))

        if not os.path.exists('../data_nus/amazon_branch_red/{}/{}/dev'.format(pl,c_rate)):
            os.makedirs('../data_nus/amazon_branch_red/{}/{}/dev'.format(pl,c_rate))

        if not os.path.exists('../data_us/amazon_branch_red/{}/{}/dev'.format(pl,c_rate)):
            os.makedirs('../data_us/amazon_branch_red/{}/{}/dev'.format(pl,c_rate))

        #----------------------
        # Compute DEV for each tsid 
        #----------------------
        
        # Compute DEV for irregularly-sampled time series
        print('\nBegin DEV computation\n')

        for i in range(numSims):
            df_traj_x = df_traj_1[df_traj_1['tsid'] == i+1]['interpolated_x']
            df_traj_p = df_traj_1[df_traj_1['tsid'] == i+1]['interpolated_p']
            rdf_x = pandas2ri.py2rpy(df_traj_x)
            rdf_p = pandas2ri.py2rpy(df_traj_p)
            globalenv['x_time_series'] = rdf_x
            globalenv['p_time_series'] = rdf_p
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
                p <- p_time_series[j]
                
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
                matrix_result[index,2] <- p
                matrix_result[index,3] <- mean(matrix_eigen[,1],na.rm=TRUE)
            }

            result <- matrix_result
            '''
            dev_p = r(rscript)[:,1]
            dev_x = r(rscript)[:,2]

            df_dev = pd.DataFrame(data=None,columns=['DEV','p'])
            df_dev['DEV'] = dev_x
            df_dev['p'] = dev_p
            # Export DEV as individual files for dynamical eigenvalue
            filepath_dev = '../data_nus/amazon_branch_red/{}/{}/dev/amazon_branch_dev_{}.csv'.format(pl,c_rate,i+1)
            df_dev.to_csv(filepath_dev,index=False)

        # Compute DEV for regularly-sampled time series
        for i in range(numSims):
            df_traj_x = df_traj_2[df_traj_2['tsid'] == i+1]['x']
            df_traj_p = df_traj_2[df_traj_2['tsid'] == i+1]['p']
            rdf_x = pandas2ri.py2rpy(df_traj_x)
            rdf_p = pandas2ri.py2rpy(df_traj_p)
            globalenv['x_time_series'] = rdf_x
            globalenv['p_time_series'] = rdf_p
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
                p <- p_time_series[j]
                
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
                matrix_result[index,2] <- p
                matrix_result[index,3] <- mean(matrix_eigen[,1],na.rm=TRUE)
            }

            result <- matrix_result
            '''
            dev_p = r(rscript)[:,1]
            dev_x = r(rscript)[:,2]

            df_dev = pd.DataFrame(data=None,columns=['DEV','p'])
            df_dev['DEV'] = dev_x
            df_dev['p'] = dev_p
            # Export DEV as individual files for dynamical eigenvalue
            filepath_dev = '../data_us/amazon_branch_red/{}/{}/dev/amazon_branch_dev_{}.csv'.format(pl,c_rate,i+1)
            df_dev.to_csv(filepath_dev,index=False)

            print('DEV for realisation '+str(i+1)+' complete')

        # Create directories for output
        if not os.path.exists('../data_nus/amazon_branch_red/{}/{}/ac'.format(pl,c_rate)):
            os.makedirs('../data_nus/amazon_branch_red/{}/{}/ac'.format(pl,c_rate))

        if not os.path.exists('../data_us/amazon_branch_red/{}/{}/ac'.format(pl,c_rate)):
            os.makedirs('../data_us/amazon_branch_red/{}/{}/ac'.format(pl,c_rate))

        print('\nBegin lag-1 autocorrelation red computation\n')
        # Compute lag-1 autocorrelation red for irregularly-sampled time series
        for i in range(numSims):
            df_traj_temp = df_traj_1[df_traj_1['tsid'] == i+1][['interpolated_x','interpolated_p']]
            x = df_traj_temp['interpolated_x']
            p = df_traj_temp['interpolated_p']
            ac_r,p_ac = ac_red(x,p,rw_bb)
            df_ac = pd.DataFrame(data=None,columns=['AC red','p'])
            df_ac['AC red'] = ac_r
            df_ac['p'] = p_ac
            # Export lag-1 autocorrelation red as individual files for BB method
            filepath_ac = '../data_nus/amazon_branch_red/{}/{}/ac/amazon_branch_ac_{}.csv'.format(pl,c_rate,i+1)
            df_ac.to_csv(filepath_ac,index=False)

        for i in range(numSims):
            df_traj_temp = df_traj_2[df_traj_2['tsid'] == i+1][['x','p']]
            x = df_traj_temp['x']
            p = df_traj_temp['p']
            ac_r,p_ac = ac_red(x,p,rw_bb)
            df_ac = pd.DataFrame(data=None,columns=['AC red','p'])
            df_ac['AC red'] = ac_r
            df_ac['p'] = p_ac
            # Export lag-1 autocorrelation red as individual files for BB method
            filepath_ac = '../data_us/amazon_branch_red/{}/{}/ac/amazon_branch_ac_{}.csv'.format(pl,c_rate,i+1)
            df_ac.to_csv(filepath_ac,index=False)

            print('lag-1 autocorrelation red for realisation '+str(i+1)+' complete')

        df_traj_1.reset_index(inplace=True)
        df_traj_2.reset_index(inplace=True)

        # Create directories for output
        if not os.path.exists('../data_nus/amazon_branch_red/{}/{}/sims'.format(pl,c_rate)):
            os.makedirs('../data_nus/amazon_branch_red/{}/{}/sims'.format(pl,c_rate))

        if not os.path.exists('../data_us/amazon_branch_red/{}/{}/sims'.format(pl,c_rate)):
            os.makedirs('../data_us/amazon_branch_red/{}/{}/sims'.format(pl,c_rate))

         # Export time series as individual files for training ML
        for i in np.arange(numSims)+1:
            df_sims = df_traj_1[df_traj_1['tsid'] == i][['Time','x','p']]
            filepath='../data_nus/amazon_branch_red/{}/{}/sims/amazon_branch_sims_{}.csv'.format(pl,c_rate,i)
            df_sims.to_csv(filepath,index=False)

        for i in np.arange(numSims)+1:
            df_sims = df_traj_2[df_traj_2['tsid'] == i][['Time','x','p']]
            filepath='../data_us/amazon_branch_red/{}/{}/sims/amazon_branch_sims_{}.csv'.format(pl,c_rate,i)
            df_sims.to_csv(filepath,index=False)

        df_result_1.to_csv('../data_nus/amazon_branch_red/{}/{}/amazon_branch_result.csv'.format(pl,c_rate))
        df_result_2.to_csv('../data_us/amazon_branch_red/{}/{}/amazon_branch_result.csv'.format(pl,c_rate))

        print('pl = {}, c_rate = {} has finished'.format(pl,c_rate))
