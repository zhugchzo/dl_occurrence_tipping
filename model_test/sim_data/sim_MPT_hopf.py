#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 16 2024

@author: Chengzuo Zhuge

Simulate MPT model 
Simulations going through Hopf bifurcation
Compute residual time series
Compute lag-1 autocorrelation by BB method
Compute DEV

"""

# import python libraries
import numpy as np
import pandas as pd
import os
import ewstools
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
numSims = 50
seed = 0 # random number generation seed
sigma_x = 0.01 # noise intensity
sigma_y = 0.01
sigma_z = 0.01

# EWS parameters
dt2 = 1 # spacing between time-series for EWS computation
rw = 0.25 # rolling window, also used for computing lag-1 autocorrelation
span = 0.2 # bandwidth
lags = [1] # autocorrelation lag times
ews = ['var','ac']


#----------------------------------
# Simulate model
#----------------------------------

# Model

def de_fun_x(x,y):
    return -x-y

def de_fun_y(y,z,u,p,s):
    return -p*z+u*y+s*z**2-y*z**2

def de_fun_z(x,z,q):
    return -q*(x+z)

def recov_fun(y,z,u,p,q,s):

    j11 = -1
    j12 = -1
    j13 = 0
    j21 = 0
    j22 = u-z**2
    j23 = -p+2*z*(s-y)
    j31 = -q
    j32 = 0
    j33 = -q
    
    jac = np.array([[j11,j12,j13],[j21,j22,j23],[j31,j32,j33]])

    try:
        evals = np.linalg.eigvals(jac)
    except:
        return 1997.0209 # if error; This particular number has no other significance, it's just the author's birthday 

    re_evals = [lam.real for lam in evals]
    dom_eval_re = max(re_evals)

    rrate = dom_eval_re

    return rrate

# Model parameters
p = 1
q = 1.2
s = 0.8

#ul = 0
ubif = 0.35 # bifurcation point (computed in Mathematica)
uh = 1.2*ubif # bifurcation parameter high
c_rate = 8.4e-6 # change in the bifurcation parameter in dt

for i in np.linspace(0,0.3,11):
    
    ul = round(i,1) # bifurcation parameter low
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

        x0 = 0 # intial condition (equilibrium value computed in Mathematica)
        y0 = 0
        z0 = 0
        equix0 = 0
        equiy0 = 0
        equiz0 = 0

        ac = random.uniform(-1,1)
        
        tmax = int(np.random.uniform(250,500)) # randomly selected sequence length
        n_random = np.random.uniform(5,500) # length of the randomly selected sequence to the bifurcation
        series_len = tmax + int(n_random)

        t = np.arange(t0,sim_len,dt)
        x = np.zeros(len(t))
        y = np.zeros(len(t))
        z = np.zeros(len(t))
        equix = np.zeros(len(t))
        equiy = np.zeros(len(t))
        equiz = np.zeros(len(t))
        nx = np.zeros(len(t))
        ny = np.zeros(len(t))
        nz = np.zeros(len(t))
        rrate = np.zeros(len(t))
        u = pd.Series(np.linspace(ul,uh,len(t)),index=t)
        
        # Create brownian increments (s.d. sqrt(dt))
        dW_x_burn = np.random.normal(loc=0, scale=sigma_x*np.sqrt(dt), size = int(tburn/dt))
        dW_x = np.random.normal(loc=0, scale=sigma_x*np.sqrt(dt), size = len(t))
        
        dW_y_burn = np.random.normal(loc=0, scale=sigma_y*np.sqrt(dt), size = int(tburn/dt))
        dW_y = np.random.normal(loc=0, scale=sigma_y*np.sqrt(dt), size = len(t))

        dW_z_burn = np.random.normal(loc=0, scale=sigma_z*np.sqrt(dt), size = int(tburn/dt))
        dW_z = np.random.normal(loc=0, scale=sigma_z*np.sqrt(dt), size = len(t))
        
        # Run burn-in period on x0
        for i in range(int(tburn/dt)):
            x0 = x0 + de_fun_x(x0,y0)*dt + dW_x_burn[i]
            y0 = y0 + de_fun_y(y0,z0,u[0],p,s)*dt + dW_y_burn[i]
            z0 = z0 + de_fun_z(x0,z0,q)*dt + dW_z_burn[i]
            equix0 = equix0 + de_fun_x(equix0,equiy0)*dt
            equiy0 = equiy0 + de_fun_y(equiy0,equiz0,u[0],p,s)*dt
            equiz0 = equiz0 + de_fun_z(equix0,equiz0,q)*dt

        rrate0 = recov_fun(equiy0,equiz0,u[0],p,q,s)
        
        if rrate0 == 1997.0209:
            print('rrate Nan appears')
            x0 = 0 # intial condition (equilibrium value computed in Mathematica)
            y0 = 0
            z0 = 0
            equix0 = 0
            equiy0 = 0
            equiz0 = 0
            continue

        if  math.isnan(x0) or math.isnan(y0) or math.isnan(z0):
            print('sim Nan appears')
            x0 = 0 # intial condition (equilibrium value computed in Mathematica)
            y0 = 0
            z0 = 0
            equix0 = 0
            equiy0 = 0
            equiz0 = 0
            continue          
        
        # Initial condition post burn-in period
        x[0]=x0
        y[0]=y0
        z[0]=z0
        equix[0]=equix0
        equiy[0]=equiy0
        equiz[0]=equiz0
        nx[0]=dW_x[0]
        ny[0]=dW_y[0]
        nz[0]=dW_z[0]
        rrate[0]=rrate0

        rrate_record = []

        # Run simulation
        for i in range(len(t)-1):
            x[i+1] = x[i] + de_fun_x(x[i],y[i])*dt + nx[i]
            y[i+1] = y[i] + de_fun_y(y[i],z[i],u.iloc[i],p,s)*dt + ny[i]
            z[i+1] = z[i] + de_fun_z(x[i],z[i],q)*dt + nz[i]
            equix[i+1] = equix[i] + de_fun_x(equix[i],equiy[i])*dt
            equiy[i+1] = equiy[i] + de_fun_y(equiy[i],equiz[i],u.iloc[i],p,s)*dt
            equiz[i+1] = equiz[i] + de_fun_z(equix[i],equiz[i],q)*dt
            nx[i+1] = ac*nx[i] + dW_x[i]
            ny[i+1] = ac*ny[i] + dW_y[i]
            nz[i+1] = ac*nz[i] + dW_z[i]
            rrate[i+1] = recov_fun(equiy[i+1],equiz[i+1],u.iloc[i+1],p,q,s)

            if rrate[i+1] == 1997.0209:
                print('rrate Nan appears')
                break

            if math.isnan(x[i+1]) or math.isnan(y[i+1]) or math.isnan(z[i+1]):
                print('sim Nan appears')
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
        data = {'tsid': (j+1)*np.ones(len(t)),'Time': t,'x': x,'y': y,'z': z,'u': u.values}
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
        interpolated_x_values = np.interp(new_u_values, df_cut_1['u'], df_cut_1['x'])
        interpolated_y_values = np.interp(new_u_values, df_cut_1['u'], df_cut_1['y'])
        interpolated_z_values = np.interp(new_u_values, df_cut_1['u'], df_cut_1['z'])

        df_cut_1['interpolated_x'] = interpolated_x_values
        df_cut_1['interpolated_y'] = interpolated_y_values
        df_cut_1['interpolated_z'] = interpolated_z_values
        df_cut_1['interpolated_u'] = new_u_values

        df_temp_2 = df_temp.iloc[d2].copy() # regularly-sampled time series
        df_temp_2['Time'] = np.arange(0,series_len)
        df_temp_2.set_index('Time', inplace=True)
        df_cut_2 = df_temp_2.iloc[0:tmax].copy()

        nearest_point_1 = df_cut_1.loc[tmax-1,'u']
        nearest_point_2 = df_cut_2.loc[tmax-1,'u']
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
    df_traj1 = pd.concat(list_traj1_append) # irregularly-sampled time series
    df_traj2 = pd.concat(list_traj2_append) # regularly-sampled time series
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

    if not os.path.exists('../data_nus/MPT_hopf_red/{}/temp'.format(ul)):
        os.makedirs('../data_nus/MPT_hopf_red/{}/temp'.format(ul))

    if not os.path.exists('../data_us/MPT_hopf_red/{}/temp'.format(ul)):
        os.makedirs('../data_us/MPT_hopf_red/{}/temp'.format(ul))

    # Filter time-series
    df_traj1.to_csv('../data_nus/MPT_hopf_red/{}/temp/df_traj.csv'.format(ul))
    df_traj2.to_csv('../data_us/MPT_hopf_red/{}/temp/df_traj.csv'.format(ul))

    if not os.path.exists('../data_nus/MPT_hopf_red/{}/dev'.format(ul)):
        os.makedirs('../data_nus/MPT_hopf_red/{}/dev'.format(ul))

    if not os.path.exists('../data_us/MPT_hopf_red/{}/dev'.format(ul)):
        os.makedirs('../data_us/MPT_hopf_red/{}/dev'.format(ul))

    #----------------------
    # Compute DEV for each tsid 
    #----------------------
    
    # Compute DEV for irregularly-sampled time series
    print('\nBegin DEV computation\n')

    for i in range(numSims):
        df_traj_x = df_traj1[df_traj1['tsid'] == i+1]['interpolated_x']
        df_traj_u = df_traj1[df_traj1['tsid'] == i+1]['interpolated_u']
        rdf_x = pandas2ri.py2rpy(df_traj_x)
        rdf_u = pandas2ri.py2rpy(df_traj_u)
        globalenv['x_time_series'] = rdf_x
        globalenv['u_time_series'] = rdf_u
        rscript = '''
        E <- 8
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
        filepath_dev = '../data_nus/MPT_hopf_red/{}/dev/MPT_hopf_500_dev_{}.csv'.format(ul,i+1)
        df_dev.to_csv(filepath_dev,index=False)

    # Compute DEV for regularly-sampled time series
    for i in range(numSims):
        df_traj_x = df_traj2[df_traj2['tsid'] == i+1]['x']
        df_traj_u = df_traj2[df_traj2['tsid'] == i+1]['u']
        rdf_x = pandas2ri.py2rpy(df_traj_x)
        rdf_u = pandas2ri.py2rpy(df_traj_u)
        globalenv['x_time_series'] = rdf_x
        globalenv['u_time_series'] = rdf_u
        rscript = '''
        E <- 8
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
        filepath_dev = '../data_us/MPT_hopf_red/{}/dev/MPT_hopf_500_dev_{}.csv'.format(ul,i+1)
        df_dev.to_csv(filepath_dev,index=False)

        print('DEV for realisation '+str(i+1)+' complete')

    #----------------------
    # Compute residual time series for each tsid 
    #----------------------

    # set up a list to store output dataframes from residual time series
    appended_ews_1 = []
    appended_ews_2 = []

    print('\nBegin residual time series computation\n')
   # Compute residual time series for irregularly-sampled time series
    for i in range(numSims):
        # loop through variable
        for var in ['x','y','z']:
            df_traj_temp = df_traj1[df_traj1['tsid'] == i+1]
            ews_dic = ewstools.core.ews_compute(df_traj_temp[var], 
                            roll_window = rw,
                            smooth='Lowess',
                            span = span,
                            lag_times = lags, 
                            ews = ews)
            
            # The DataFrame of EWS
            df_ews_temp = ews_dic['EWS metrics']
            
            # Include a column in the DataFrames for realisation number and variable
            df_ews_temp['tsid'] = i+1
            df_ews_temp['Variable'] = var
            df_ews_temp['u'] = df_traj_temp['u']
                
            # Add DataFrames to list
            appended_ews_1.append(df_ews_temp)
            
    # Compute residual time series for regularly-sampled time series
    # loop through variable
        for var in ['x','y','z']:
            df_traj_temp = df_traj2[df_traj2['tsid'] == i+1]
            ews_dic = ewstools.core.ews_compute(df_traj_temp[var], 
                            roll_window = rw,
                            smooth='Lowess',
                            span = span,
                            lag_times = lags, 
                            ews = ews)
            
            # The DataFrame of EWS
            df_ews_temp = ews_dic['EWS metrics']
            
            # Include a column in the DataFrames for realisation number and variable
            df_ews_temp['tsid'] = i+1
            df_ews_temp['Variable'] = var
            df_ews_temp['u'] = df_traj_temp['u']
                
            # Add DataFrames to list
            appended_ews_2.append(df_ews_temp)
            
        # Print status every realisation
        print('residual time series for realisation '+str(i+1)+' complete')

    # Concatenate EWS DataFrames
    df_ews_1 = pd.concat(appended_ews_1).reset_index()
    df_ews_2 = pd.concat(appended_ews_2).reset_index()

    # Create directories for output
    if not os.path.exists('../data_nus/MPT_hopf_red/{}/ac'.format(ul)):
        os.makedirs('../data_nus/MPT_hopf_red/{}/ac'.format(ul))

    if not os.path.exists('../data_us/MPT_hopf_red/{}/ac'.format(ul)):
        os.makedirs('../data_us/MPT_hopf_red/{}/ac'.format(ul))

    print('\nBegin lag-1 autocorrelation red computation\n')
    # Compute lag-1 autocorrelation red for irregularly-sampled time series
    for i in range(numSims):
        df_traj_temp = df_traj1[df_traj1['tsid'] == i+1][['interpolated_x','interpolated_u']]
        x = df_traj_temp['interpolated_x']
        u = df_traj_temp['interpolated_u']
        ac_r,u_ac = ac_red(x,u,rw)
        df_ac = pd.DataFrame(data=None,columns=['AC red','u'])
        df_ac['AC red'] = ac_r
        df_ac['u'] = u_ac
        # Export lag-1 autocorrelation red as individual files for BB method
        filepath_ac = '../data_nus/MPT_hopf_red/{}/ac/MPT_hopf_500_ac_{}.csv'.format(ul,i+1)
        df_ac.to_csv(filepath_ac,index=False)

        print('AC red for realisation '+str(i+1)+' complete')

    for i in range(numSims):
        df_traj_temp = df_traj2[df_traj2['tsid'] == i+1][['x','u']]
        x = df_traj_temp['x']
        u = df_traj_temp['u']
        ac_r,u_ac = ac_red(x,u,rw)
        df_ac = pd.DataFrame(data=None,columns=['AC red','u'])
        df_ac['AC red'] = ac_r
        df_ac['u'] = u_ac
        # Export lag-1 autocorrelation red as individual files for BB method
        filepath_ac = '../data_us/MPT_hopf_red/{}/ac/MPT_hopf_500_ac_{}.csv'.format(ul,i+1)
        df_ac.to_csv(filepath_ac,index=False)

        print('lag-1 autocorrelation red for realisation '+str(i+1)+' complete')

    # Create directories for output
    if not os.path.exists('../data_nus/MPT_hopf_red/{}/ews'.format(ul)):
        os.makedirs('../data_nus/MPT_hopf_red/{}/ews'.format(ul))

    if not os.path.exists('../data_nus/MPT_hopf_red/{}/resids'.format(ul)):
        os.makedirs('../data_nus/MPT_hopf_red/{}/resids'.format(ul))

    if not os.path.exists('../data_us/MPT_hopf_red/{}/ews'.format(ul)):
        os.makedirs('../data_us/MPT_hopf_red/{}/ews'.format(ul))

    if not os.path.exists('../data_us/MPT_hopf_red/{}/resids'.format(ul)):
        os.makedirs('../data_us/MPT_hopf_red/{}/resids'.format(ul))

    # Export EWS data
    df_ews_1.to_csv('../data_nus/MPT_hopf_red/{}/ews/df_ews_forced.csv'.format(ul))
    df_ews_2.to_csv('../data_us/MPT_hopf_red/{}/ews/df_ews_forced.csv'.format(ul))

    # Export residuals as individual files for training ML
    for i in np.arange(numSims)+1:
        df_resids = df_ews_1[(df_ews_1['tsid'] == i) & (df_ews_1['Variable'] == 'x')][['Time','Residuals','u']]
        filepath='../data_nus/MPT_hopf_red/{}/resids/MPT_hopf_500_resids_{}.csv'.format(ul,i)
        df_resids.to_csv(filepath,index=False)

    for i in np.arange(numSims)+1:
        df_resids = df_ews_2[(df_ews_2['tsid'] == i) & (df_ews_2['Variable'] == 'x')][['Time','Residuals','u']]
        filepath='../data_us/MPT_hopf_red/{}/resids/MPT_hopf_500_resids_{}.csv'.format(ul,i)
        df_resids.to_csv(filepath,index=False)

    df_result_1.to_csv('../data_nus/MPT_hopf_red/{}/MPT_hopf_500_result.csv'.format(ul))
    df_result_2.to_csv('../data_us/MPT_hopf_red/{}/MPT_hopf_500_result.csv'.format(ul))

    print('ul = {} has finished'.format(ul))