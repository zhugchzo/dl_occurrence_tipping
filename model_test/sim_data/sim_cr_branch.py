#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 16 2024

@author: Chengzuo Zhuge

Simulate Consumer-resource model 
Simulations going through transcritical bifurcation
Compute residual time series
Compute lag-1 autocorrelation by degenerate fingerprinting
Compute DEV

"""

# import python libraries
import numpy as np
import pandas as pd
import os
import ewstools
import math
from sklearn.decomposition import PCA
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
dt = 0.1
t0 = 0
tburn = 500 # burn-in period
numSims = 20
seed = 0 # random number generation seed
sigma_x = 0.01 # noise intensity
sigma_y = 0.01


# EWS parameters
dt2 = 1 # spacing between time-series for EWS computation
rw = 0.25 # rolling window
span = 0.2 # bandwidth
lags = [1] # autocorrelation lag times
ews = ['var','ac']

# degenerate fingerprinting parameters
rw_degf = 0.1 # rolling window

#----------------------------------
# Simulate model
#----------------------------------

# Model

def de_fun_x(x,y,g,k,a,h):
    return g*x*(1-x/k) - (a*x*y)/(1+a*h*x)

def de_fun_y(x,y,e,a,h,m):
    return e*a*x*y/(1+a*h*x) - m*y

def recov_fun(x,y,g,k,e,a,h,m):

    j11 = g - 2*g*x/k - a*y/(1+a*h*x)**2
    j12 = -a*x/(1+a*h*x)
    j21 = e*a*y/(1+a*h*x)**2
    j22 = e*a*x/(1+a*h*x) - m

    jac = np.array([[j11,j12],[j21,j22]])

    try:
        evals = np.linalg.eigvals(jac)
    except:
        return 1997.0209 # if error; This particular number has no other significance, it's just the author's birthday

    re_evals = [lam.real for lam in evals]
    dom_eval_re = max(re_evals)

    rrate = dom_eval_re

    return rrate   
    
# Model parameters
sf = 4 # scale factor
g = 1*sf
k = 1.7
h = 0.6/sf
e = 0.5
m = 0.5*sf

abif = 5.88 # bifurcation point (computed in Mathematica)
ah = 1.2*abif # control parameter final value
c_rate = 8.112e-5 # bifurcation parameter change rate

for i in np.linspace(0*sf,1.25*sf,11):

    al = round(i,2) # bifurcation parameter low
    sim_len = int((ah-al)*dt/c_rate)

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

        x0 = 1.7 # intial condition (equilibrium value computed in Mathematica)
        y0 = 0.1
        equix0 = 1.7
        equiy0 = 0.1
        
        tmax = int(np.random.uniform(250,500)) # randomly selected sequence length
        n = np.random.uniform(100,500) # length of the randomly selected sequence to the bifurcation
        series_len = tmax + int(n)

        t = np.arange(t0,sim_len,dt)
        x = np.zeros(len(t))
        y = np.zeros(len(t))
        equix = np.zeros(len(t))
        equiy = np.zeros(len(t))
        rrate = np.zeros(len(t))
        a = pd.Series(np.linspace(al,ah,len(t)),index=t)
        
        # Create brownian increments (s.d. sqrt(dt))
        dW_x_burn = np.random.normal(loc=0, scale=sigma_x*np.sqrt(dt), size = int(tburn/dt))
        dW_x = np.random.normal(loc=0, scale=sigma_x*np.sqrt(dt), size = len(t))
        
        dW_y_burn = np.random.normal(loc=0, scale=sigma_y*np.sqrt(dt), size = int(tburn/dt))
        dW_y = np.random.normal(loc=0, scale=sigma_y*np.sqrt(dt), size = len(t))
        
        # Run burn-in period on x0
        for i in range(int(tburn/dt)):
            x0 = x0 + de_fun_x(x0,y0,g,k,a[0],h)*dt + dW_x_burn[i]
            y0 = y0 + de_fun_y(x0,y0,e,a[0],h,m)*dt + dW_y_burn[i]
            equix0 = equix0 + de_fun_x(equix0,equiy0,g,k,a[0],h)*dt
            equiy0 = equiy0 + de_fun_y(equix0,equiy0,e,a[0],h,m)*dt

        rrate0 = recov_fun(equix0,equiy0,g,k,e,a[0],h,m)

        if rrate0 == 1997.0209 or math.isnan(x0) or math.isnan(y0):
            print('Nan appears')
            x0 = 1.7 # intial condition (equilibrium value computed in Mathematica)
            y0 = 0.1
            equix0 = 1.7
            equiy0 = 0.1
            continue
            
        # Initial condition post burn-in period
        x[0]=x0
        y[0]=y0
        equix[0]=equix0
        equiy[0]=equiy0
        rrate[0]=rrate0

        rrate_record = []

        # Run simulation
        for i in range(len(t)-1):
            x[i+1] = x[i] + de_fun_x(x[i],y[i],g,k,a.iloc[i],h)*dt + dW_x[i]
            y[i+1] = y[i] + de_fun_y(x[i],y[i],e,a.iloc[i],h,m)*dt + dW_y[i]
            equix[i+1] = equix[i] + de_fun_x(equix[i],equiy[i],g,k,a.iloc[i],h)*dt
            equiy[i+1] = equiy[i] + de_fun_y(equix[i],equiy[i],e,a.iloc[i],h,m)*dt
            rrate[i+1] = recov_fun(equix[i+1],equiy[i+1],g,k,e,a.iloc[i+1],h,m)

            if rrate[i+1] == 1997.0209:
                print('rrate Nan appears')
                break

            if math.isnan(x[i+1]) or math.isnan(y[i+1]):
                print('sim Nan appears')
                break              
            # Determine the tipping point by the recovery rate changing from negative to positive
            if rrate[i] < 0 and rrate[i+1] > 0:
                rrate_record.append(i+1)
                break

            # make sure that state variable remains >= 0 
            if x[i+1] < 0:
                x[i+1] = 0
            if y[i+1] < 0:
                y[i+1] = 0
    

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
        data = {'tsid': (j+1)*np.ones(len(t)),'Time': t,'x': x,'y': y,'a': a.values}
        df_temp = pd.DataFrame(data)
        trans_point = df_temp.loc[trans_time-1,'a']

        df_temp_1 = df_temp.iloc[d1].copy() # irregularly-sampled time series
        df_temp_1['Time'] = np.arange(0,series_len)
        df_temp_1.set_index('Time', inplace=True)
        df_cut_1 = df_temp_1.iloc[0:tmax].copy()

        df_temp_2 = df_temp.iloc[d2].copy() # regularly-sampled time series
        df_temp_2['Time'] = np.arange(0,series_len)
        df_temp_2.set_index('Time', inplace=True)
        df_cut_2 = df_temp_2.iloc[0:tmax].copy()

        nearest_point_1 = df_cut_1.loc[tmax-1,'a']
        nearest_point_2 = df_cut_2.loc[tmax-1,'a']
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

    if not os.path.exists('../data_uniform'):
        os.makedirs('../data_uniform')

    if not os.path.exists('../data_nus/cr_branch_white/{}/temp'.format(al)):
        os.makedirs('../data_nus/cr_branch_white/{}/temp'.format(al))

    if not os.path.exists('../data_uniform/cr_branch_white/{}/temp'.format(al)):
        os.makedirs('../data_uniform/cr_branch_white/{}/temp'.format(al))

    # Filter time-series
    df_traj1.to_csv('../data_nus/cr_branch_white/{}/temp/df_traj.csv'.format(al))
    df_traj2.to_csv('../data_uniform/cr_branch_white/{}/temp/df_traj.csv'.format(al))

    if not os.path.exists('../data_nus/cr_branch_white/{}/dev'.format(al)):
        os.makedirs('../data_nus/cr_branch_white/{}/dev'.format(al))

    if not os.path.exists('../data_uniform/cr_branch_white/{}/dev'.format(al)):
        os.makedirs('../data_uniform/cr_branch_white/{}/dev'.format(al))

    #----------------------
    # Compute DEV for each tsid 
    #----------------------
    
    # Compute DEV for irregularly-sampled time series
    print('\nBegin DEV computation\n')

    for i in range(numSims):
        df_traj_x = df_traj1[df_traj1['tsid'] == i+1]['x']
        df_traj_a = df_traj1[df_traj1['tsid'] == i+1]['a']
        rdf_x = pandas2ri.py2rpy(df_traj_x)
        rdf_a = pandas2ri.py2rpy(df_traj_a)
        globalenv['x_time_series'] = rdf_x
        globalenv['a_time_series'] = rdf_a
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
            a <- a_time_series[j]
            
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
            matrix_result[index,2] <- a
            matrix_result[index,3] <- mean(matrix_eigen[,1],na.rm=TRUE)
        }

        result <- matrix_result
        '''
        dev_a = r(rscript)[:,1]
        dev_x = r(rscript)[:,2]
        
        df_dev = pd.DataFrame(data=None,columns=['DEV','a'])
        df_dev['DEV'] = dev_x
        df_dev['a'] = dev_a
        # Export DEV as individual files for dynamical eigenvalue
        filepath_dev = '../data_nus/cr_branch_white/{}/dev/cr_branch_500_dev_{}.csv'.format(al,i+1)
        df_dev.to_csv(filepath_dev,index=False)

    # Compute DEV for regularly-sampled time series
    for i in range(numSims):
        df_traj_x = df_traj2[df_traj2['tsid'] == i+1]['x']
        df_traj_a = df_traj2[df_traj2['tsid'] == i+1]['a']
        rdf_x = pandas2ri.py2rpy(df_traj_x)
        rdf_a = pandas2ri.py2rpy(df_traj_a)
        globalenv['x_time_series'] = rdf_x
        globalenv['a_time_series'] = rdf_a
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
            a <- a_time_series[j]
            
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
            matrix_result[index,2] <- a
            matrix_result[index,3] <- mean(matrix_eigen[,1],na.rm=TRUE)
        }

        result <- matrix_result
        '''
        dev_a = r(rscript)[:,1]
        dev_x = r(rscript)[:,2]

        df_dev = pd.DataFrame(data=None,columns=['DEV','a'])
        df_dev['DEV'] = dev_x
        df_dev['a'] = dev_a
        # Export DEV as individual files for dynamical eigenvalue
        filepath_dev = '../data_uniform/cr_branch_white/{}/dev/cr_branch_500_dev_{}.csv'.format(al,i+1)
        df_dev.to_csv(filepath_dev,index=False)

        print('DEV for realisation '+str(i+1)+' complete')

    #----------------------
    # Compute residual time series and lag-1 autocorrelation for each tsid 
    #----------------------

    # set up a list to store output dataframes from residual time series and lag-1 autocorrelation
    appended_ews_1 = []
    appended_pca_1 = []
    appended_degf_1 = []
    appended_ews_2 = []
    appended_pca_2 = []
    appended_degf_2 = []

    # Compute PCA for degenerate fingerprinting
    print('\nBegin PCA computation\n')

    # Compute PCA for irregularly-sampled time series
    for i in range(numSims):
        df_traj_temp = df_traj1[df_traj1['tsid'] == i+1]
        traj = np.array(df_traj_temp[['x','y']])
        mean_traj = np.mean(traj,axis=1,keepdims=True)
        traj = traj - mean_traj

        pcamodel = PCA(1)
        pcamodel.fit(traj)

        pca = pcamodel.fit_transform(traj).reshape(-1)
        appended_pca_1.append(pd.Series(pca))

    # Compute PCA for regularly-sampled time series
        df_traj_temp = df_traj2[df_traj2['tsid'] == i+1]
        traj = np.array(df_traj_temp[['x','y']])
        mean_traj = np.mean(traj,axis=1,keepdims=True)
        traj = traj - mean_traj

        pcamodel = PCA(1)
        pcamodel.fit(traj)

        pca = pcamodel.fit_transform(traj).reshape(-1)
        appended_pca_2.append(pd.Series(pca))

        print('PCA for realisation '+str(i+1)+' complete')  
    
    df_pca_1 = pd.concat(appended_pca_1).reset_index()
    df_traj1['pca'] = df_pca_1.iloc[:,1]
    df_pca_2 = pd.concat(appended_pca_2).reset_index()
    df_traj2['pca'] = df_pca_2.iloc[:,1]

    print('\nBegin residual time series and lag-1 autocorrelation computation\n')
    # Compute residual time series and lag-1 autocorrelation for irregularly-sampled time series
    for i in range(numSims):
        # loop through variable
        for var in ['x','y']:
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
            df_ews_temp['a'] = df_traj_temp['a']
                
            # Add DataFrames to list
            appended_ews_1.append(df_ews_temp)

        # Compute lag-1 autocorrelation for PCA
        for var in ['pca']:
            df_traj_temp = df_traj1[df_traj1['tsid'] == i+1]
            ews_dic = ewstools.core.ews_compute(df_traj_temp[var], 
                            roll_window = rw_degf,
                            smooth='Lowess',
                            span = span,
                            lag_times = lags, 
                            ews = ews)
            
            # The DataFrame of EWS
            df_degf_temp = ews_dic['EWS metrics']
            
            # Include a column in the DataFrames for realisation number and variable
            df_degf_temp['tsid'] = i+1
            df_degf_temp['Variable'] = var
            df_degf_temp['a'] = df_traj_temp['a']
                
            # Add DataFrames to list
            appended_degf_1.append(df_degf_temp)

    # Compute residual time series and lag-1 autocorrelation for regularly-sampled time series
    for i in range(numSims):
        # loop through variable
        for var in ['x','y']:
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
            df_ews_temp['a'] = df_traj_temp['a']
                
            # Add DataFrames to list
            appended_ews_2.append(df_ews_temp)

        for var in ['pca']:
            df_traj_temp = df_traj2[df_traj2['tsid'] == i+1]
            ews_dic = ewstools.core.ews_compute(df_traj_temp[var], 
                            roll_window = rw_degf,
                            smooth='Lowess',
                            span = span,
                            lag_times = lags, 
                            ews = ews)
            
            # The DataFrame of EWS
            df_degf_temp = ews_dic['EWS metrics']
            
            # Include a column in the DataFrames for realisation number and variable
            df_degf_temp['tsid'] = i+1
            df_degf_temp['Variable'] = var
            df_degf_temp['a'] = df_traj_temp['a']
                
            # Add DataFrames to list
            appended_degf_2.append(df_degf_temp)
            
        print('residual time series and lag-1 autocorrelation for realisation '+str(i+1)+' complete')

    # Concatenate EWS DataFrames
    df_ews_1 = pd.concat(appended_ews_1).reset_index()
    df_degf_1 = pd.concat(appended_degf_1).reset_index()    
    df_ews_2 = pd.concat(appended_ews_2).reset_index()
    df_degf_2 = pd.concat(appended_degf_2).reset_index()

    # Create directories for output
    if not os.path.exists('../data_nus/cr_branch_white/{}/ews'.format(al)):
        os.makedirs('../data_nus/cr_branch_white/{}/ews'.format(al))

    if not os.path.exists('../data_nus/cr_branch_white/{}/resids'.format(al)):
        os.makedirs('../data_nus/cr_branch_white/{}/resids'.format(al))

    if not os.path.exists('../data_nus/cr_branch_white/{}/ac'.format(al)):
        os.makedirs('../data_nus/cr_branch_white/{}/ac'.format(al))

    if not os.path.exists('../data_uniform/cr_branch_white/{}/ews'.format(al)):
        os.makedirs('../data_uniform/cr_branch_white/{}/ews'.format(al))

    if not os.path.exists('../data_uniform/cr_branch_white/{}/resids'.format(al)):
        os.makedirs('../data_uniform/cr_branch_white/{}/resids'.format(al))

    if not os.path.exists('../data_uniform/cr_branch_white/{}/ac'.format(al)):
        os.makedirs('../data_uniform/cr_branch_white/{}/ac'.format(al))

    # Export EWS data
    df_ews_1.to_csv('../data_nus/cr_branch_white/{}/ews/df_ews_forced.csv'.format(al))
    df_ews_2.to_csv('../data_uniform/cr_branch_white/{}/ews/df_ews_forced.csv'.format(al))

    # Export residual time series as individual files for training ML
    # Export lag-1 autocorrelation as individual files for degenerate fingerprinting
    for i in np.arange(numSims)+1:
        df_resids = df_ews_1[(df_ews_1['tsid'] == i) & (df_ews_1['Variable'] == 'x')][['Time','Residuals','a']]
        df_ac = df_degf_1[(df_degf_1['tsid'] == i) & (df_degf_1['Variable'] == 'pca')][['Lag-1 AC','a']]
        df_ac = df_ac[df_ac['Lag-1 AC'].notna()]
        filepath_resids='../data_nus/cr_branch_white/{}/resids/cr_branch_500_resids_{}.csv'.format(al,i)
        filepath_ac='../data_nus/cr_branch_white/{}/ac/cr_branch_500_ac_{}.csv'.format(al,i)
        df_resids.to_csv(filepath_resids,index=False)
        df_ac.to_csv(filepath_ac,index=False)

    for i in np.arange(numSims)+1:
        df_resids = df_ews_2[(df_ews_2['tsid'] == i) & (df_ews_2['Variable'] == 'x')][['Time','Residuals','a']]
        df_ac = df_degf_2[(df_degf_2['tsid'] == i) & (df_degf_2['Variable'] == 'pca')][['Lag-1 AC','a']]
        df_ac = df_ac[df_ac['Lag-1 AC'].notna()]
        filepath_resids='../data_uniform/cr_branch_white/{}/resids/cr_branch_500_resids_{}.csv'.format(al,i)
        filepath_ac='../data_uniform/cr_branch_white/{}/ac/cr_branch_500_ac_{}.csv'.format(al,i)
        df_resids.to_csv(filepath_resids,index=False)
        df_ac.to_csv(filepath_ac,index=False)

    df_result_1.to_csv('../data_nus/cr_branch_white/{}/cr_branch_500_result.csv'.format(al))
    df_result_2.to_csv('../data_uniform/cr_branch_white/{}/cr_branch_500_result.csv'.format(al))

    print('ml = {} has finished'.format(al))