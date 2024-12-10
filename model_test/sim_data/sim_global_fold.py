#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 16 2024

@author: Chengzuo Zhuge

Simulate Global temperature model 
Simulations going through Fold bifurcation
Compute lag-1 autocorrelation by BB method
Compute DEV

"""

# import python libraries
import numpy as np
import pandas as pd
import os
import ewstools
import random
import math

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
dt = 1
t0 = 0
tburn = 500 # burn-in period
numSims = 10
sigma_T = 0.01 # noise intensity


# EWS parameters
rw_bb = 0.25

#----------------------------------
# Simulate model
#----------------------------------

# Model

def de_fun(tem,u,i_0,b_2,a_2,sigma,c,e_SA):
    a = e_SA*sigma/c
    b_u = u*i_0*b_2/(4*c)
    d_u = u*i_0*(1-a_2)/(4*c)
    return -a*tem**4 + b_u*tem + d_u

def recov_fun(tem,u,i_0,b_2,sigma,c,e_SA):
    a = e_SA*sigma/c
    b_u = u*i_0*b_2/(4*c)
    rrate = -4*a*tem**3 + b_u
    return rrate

# Model parameters
i_0 = 71944000
b_2 = 0.009
a_2 = 2.8
sigma = 0.003
c = 1e8
e_SA = 0.69

ucrit = 0.9621 # bifurcation point (computed in Mathematica)
uh = 0.9  # bifurcation parameter high

for i in np.linspace(1.4,1.2,11):

    ul = round(i,2) # bifurcation parameter low

    for c_rate in [-5e-7,-6e-7,-7e-7,-8e-7,-9e-7]:

        sim_len = int((uh-ul)*dt/c_rate)

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

            tem0 = 368.586 # intial condition (equilibrium value computed in Mathematica)
            equi0 = 368.586

            ac = random.uniform(-1,1)    

            tmax = int(np.random.uniform(250,500)) # randomly selected sequence length
            n_random = np.random.uniform(5,500) # length of the randomly selected sequence to the bifurcation
            series_len = tmax + int(n_random)

            t = np.arange(t0,sim_len,dt)

            tem = np.zeros(len(t))
            equi = np.zeros(len(t))
            n = np.zeros(len(t))
            rrate = np.zeros(len(t))

            u = pd.Series(np.linspace(ul,uh,len(t)),index=t)

            # Create brownian increments (s.d. sqrt(dt))
            dW_burn = np.random.normal(loc=0, scale=sigma_T*np.sqrt(dt), size = int(tburn/dt))
            dW = np.random.normal(loc=0, scale=sigma_T*np.sqrt(dt), size = len(t))

            # Run burn-in period on x0
            for i in range(int(tburn/dt)):
                tem0 = tem0 + de_fun(tem0,u[0],i_0,b_2,a_2,sigma,c,e_SA)*dt + dW_burn[i]
                equi0 = equi0 + de_fun(equi0,u[0],i_0,b_2,a_2,sigma,c,e_SA)*dt

            rrate0 = recov_fun(equi0,u[0],i_0,b_2,sigma,c,e_SA)

            if  math.isinf(tem0):
                print('sim inf appears')
                tem0 = 368.586 # intial condition (equilibrium value computed in Mathematica)
                equi0 = 368.586
                continue   

            # Initial condition post burn-in period
            tem[0]=tem0
            equi[0]=equi0
            n[0]=dW[0]
            rrate[0]=rrate0

            rrate_record = []
            
            # Run simulation
            for i in range(len(t)-1):
                tem[i+1] = tem[i] + de_fun(tem[i],u.iloc[i],i_0,b_2,a_2,sigma,c,e_SA)*dt + n[i]
                equi[i+1] = equi[i] + de_fun(equi[i],u.iloc[i],i_0,b_2,a_2,sigma,c,e_SA)*dt
                n[i+1] = ac*n[i] + dW[i]
                rrate[i+1] = recov_fun(equi[i+1],u.iloc[i+1],i_0,b_2,sigma,c,e_SA)

                if math.isinf(tem[i+1]):
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
            data = {'tsid': (j+1)*np.ones(len(t)),'Time': t,'x': tem, 'u': u.values}
            df_temp = pd.DataFrame(data)
            trans_point = df_temp.loc[trans_time-1,'u']

            df_temp_1 = df_temp.iloc[d1].copy() # irregularly-sampled time series
            df_temp_1['Time'] = np.arange(0,series_len)
            df_temp_1.set_index('Time', inplace=True)
            df_cut_1 = df_temp_1.iloc[0:tmax].copy()

            # Get the minimum and maximum values of the original 'u' column
            u_min = df_cut_1['u'].min()
            u_max = df_cut_1['u'].max()

            # Create a new uniformly distributed 'u' column
            new_u_values = np.linspace(u_min, u_max, tmax)

            # Interpolate the 'State variable'
            df_cut_1_reverse = df_cut_1.sort_values(by='u', ascending=True)
            interpolated_values = np.interp(new_u_values, df_cut_1_reverse['u'], df_cut_1_reverse['x'])

            df_cut_1['interpolated_x'] = interpolated_values[::-1]
            df_cut_1['interpolated_u'] = new_u_values[::-1]

            df_temp_2 = df_temp.iloc[d2].copy() # regularly-sampled time series
            df_temp_2['Time'] = np.arange(0,series_len)
            df_temp_2.set_index('Time', inplace=True)
            df_cut_2 = df_temp_2.iloc[0:tmax].copy()

            nearest_point_1 = df_cut_1.loc[tmax-1,'u']
            nearest_point_2 = df_cut_2.loc[tmax-1,'u']
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

        if not os.path.exists('../data_nus/global_red/{}/{}/temp'.format(ul,c_rate)):
            os.makedirs('../data_nus/global_red/{}/{}/temp'.format(ul,c_rate))

        if not os.path.exists('../data_us/global_red/{}/{}/temp'.format(ul,c_rate)):
            os.makedirs('../data_us/global_red/{}/{}/temp'.format(ul,c_rate))

        # Filter time-series
        df_traj_1.to_csv('../data_nus/global_red/{}/{}/temp/df_traj.csv'.format(ul,c_rate))
        df_traj_2.to_csv('../data_us/global_red/{}/{}/temp/df_traj.csv'.format(ul,c_rate))

        if not os.path.exists('../data_nus/global_red/{}/{}/dev'.format(ul,c_rate)):
            os.makedirs('../data_nus/global_red/{}/{}/dev'.format(ul,c_rate))

        if not os.path.exists('../data_us/global_red/{}/{}/dev'.format(ul,c_rate)):
            os.makedirs('../data_us/global_red/{}/{}/dev'.format(ul,c_rate))

        #----------------------
        # Compute DEV for each tsid 
        #----------------------
        
        # Compute DEV for irregularly-sampled time series
        print('\nBegin DEV computation\n')

        for i in range(numSims):
            df_traj_x = df_traj_1[df_traj_1['tsid'] == i+1]['interpolated_x']
            df_traj_u = df_traj_1[df_traj_1['tsid'] == i+1]['interpolated_u']
            rdf_x = pandas2ri.py2rpy(df_traj_x)
            rdf_u = pandas2ri.py2rpy(df_traj_u)
            globalenv['x_time_series'] = rdf_x
            globalenv['u_time_series'] = rdf_u
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
                u <- u_time_series[j]
                
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
                matrix_result[index,2] <- u
                matrix_result[index,3] <- mean(matrix_eigen[,1],na.rm=TRUE)
            }

            result <- matrix_result
            '''
            dev_u = r(rscript)[:,1]
            dev_x = r(rscript)[:,2]

            df_dev = pd.DataFrame(data=None,columns=['DEV','u'])
            df_dev['DEV'] = dev_x
            df_dev['u'] = dev_u
            # Export DEV as individual files for dynamical eigenvalue
            filepath_dev = '../data_nus/global_red/{}/{}/dev/global_dev_{}.csv'.format(ul,c_rate,i+1)
            df_dev.to_csv(filepath_dev,index=False)

        # Compute DEV for regularly-sampled time series
        for i in range(numSims):
            df_traj_x = df_traj_2[df_traj_2['tsid'] == i+1]['x']
            df_traj_u = df_traj_2[df_traj_2['tsid'] == i+1]['u']
            rdf_x = pandas2ri.py2rpy(df_traj_x)
            rdf_u = pandas2ri.py2rpy(df_traj_u)
            globalenv['x_time_series'] = rdf_x
            globalenv['u_time_series'] = rdf_u
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
                u <- u_time_series[j]
                
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
                matrix_result[index,2] <- u
                matrix_result[index,3] <- mean(matrix_eigen[,1],na.rm=TRUE)
            }

            result <- matrix_result
            '''
            dev_u = r(rscript)[:,1]
            dev_x = r(rscript)[:,2]

            df_dev = pd.DataFrame(data=None,columns=['DEV','u'])
            df_dev['DEV'] = dev_x
            df_dev['u'] = dev_u
            # Export DEV as individual files for dynamical eigenvalue
            filepath_dev = '../data_us/global_red/{}/{}/dev/global_dev_{}.csv'.format(ul,c_rate,i+1)
            df_dev.to_csv(filepath_dev,index=False)

            print('DEV for realisation '+str(i+1)+' complete')

        # Create directories for output
        if not os.path.exists('../data_nus/global_red/{}/{}/ac'.format(ul,c_rate)):
            os.makedirs('../data_nus/global_red/{}/{}/ac'.format(ul,c_rate))

        if not os.path.exists('../data_us/global_red/{}/{}/ac'.format(ul,c_rate)):
            os.makedirs('../data_us/global_red/{}/{}/ac'.format(ul,c_rate))

        print('\nBegin lag-1 autocorrelation red computation\n')
        # Compute lag-1 autocorrelation red for irregularly-sampled time series
        for i in range(numSims):
            df_traj_temp = df_traj_1[df_traj_1['tsid'] == i+1][['interpolated_x','interpolated_u']]
            x = df_traj_temp['interpolated_x']
            u = df_traj_temp['interpolated_u']
            ac_r,u_ac = ac_red(x,u,rw_bb)
            df_ac = pd.DataFrame(data=None,columns=['AC red','u'])
            df_ac['AC red'] = ac_r
            df_ac['u'] = u_ac
            # Export lag-1 autocorrelation red as individual files for BB method
            filepath_ac = '../data_nus/global_red/{}/{}/ac/global_ac_{}.csv'.format(ul,c_rate,i+1)
            df_ac.to_csv(filepath_ac,index=False)

        # Compute lag-1 autocorrelation red for regularly-sampled time series
            df_traj_temp = df_traj_2[df_traj_2['tsid'] == i+1][['x','u']]
            x = df_traj_temp['x']
            u = df_traj_temp['u']
            ac_r,u_ac = ac_red(x,u,rw_bb)
            df_ac = pd.DataFrame(data=None,columns=['AC red','u'])
            df_ac['AC red'] = ac_r
            df_ac['u'] = u_ac
            # Export lag-1 autocorrelation red as individual files for BB method
            filepath_ac = '../data_us/global_red/{}/{}/ac/global_ac_{}.csv'.format(ul,c_rate,i+1)
            df_ac.to_csv(filepath_ac,index=False)

            print('lag-1 autocorrelation red for realisation '+str(i+1)+' complete')

        df_traj_1.reset_index(inplace=True)
        df_traj_2.reset_index(inplace=True)

        # Create directories for output
        if not os.path.exists('../data_nus/global_red/{}/{}/sims'.format(ul,c_rate)):
            os.makedirs('../data_nus/global_red/{}/{}/sims'.format(ul,c_rate))

        if not os.path.exists('../data_us/global_red/{}/{}/sims'.format(ul,c_rate)):
            os.makedirs('../data_us/global_red/{}/{}/sims'.format(ul,c_rate))

        # Export time series as individual files for training ML
        for i in np.arange(numSims)+1:
            df_sims = df_traj_1[df_traj_1['tsid'] == i][['Time','x','u']]
            filepath='../data_nus/global_red/{}/{}/sims/global_sims_{}.csv'.format(ul,c_rate,i)
            df_sims.to_csv(filepath,index=False)

        for i in np.arange(numSims)+1:
            df_sims = df_traj_2[df_traj_2['tsid'] == i][['Time','x','u']]
            filepath='../data_us/global_red/{}/{}/sims/global_sims_{}.csv'.format(ul,c_rate,i)
            df_sims.to_csv(filepath,index=False)

        df_result_1.to_csv('../data_nus/global_red/{}/{}/global_result.csv'.format(ul,c_rate))
        df_result_2.to_csv('../data_us/global_red/{}/{}/global_result.csv'.format(ul,c_rate))

        print('ul = {}, c_rate = {} has finished'.format(ul,c_rate))

