#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 16 2024

@author: Chengzuo Zhuge

Simulate Food chain model 
Simulations going through Hopf bifurcation
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
dt = 0.01
t0 = 0
tburn = 500 # burn-in period
numSims = 10
sigma_w = 0.01 # noise intensity
sigma_c = 0.01
sigma_p = 0.01

# EWS parameters
auto_lag = 1 # autocorrelation lag times
ews = ['ac1']
rw_degf = 0.1

#----------------------------------
# Simulate model
#----------------------------------

# Model

def de_fun_w(w,c,k,x_c,y_c,w_0):
    return w*(1-w/k)-x_c*y_c*c*w/(w+w_0)

def de_fun_c(w,c,p,x_c,y_c,w_0,x_p,y_p,c_0):
    return x_c*c*(y_c*w/(w+w_0)-1)-x_p*y_p*p*c/(c+c_0)

def de_fun_p(p,c,x_p,y_p,c_0):
    return x_p*p*(y_p*c/(c+c_0)-1)

def recov_fun(w,c,p,k,x_c,y_c,w_0,x_p,y_p,c_0):

    j11 = 1-2*w/k-x_c*y_c*c*w_0/(w+w_0)**2
    j12 = -x_c*y_c*w/(w+w_0)
    j13 = 0
    j21 = x_c*y_c*c*w_0/(w+w_0)**2
    j22 = x_c*(y_c*w/(w+w_0)-1)-x_p*y_p*p*c_0/(c+c_0)**2
    j23 = -x_p*y_p*c/(c+c_0)
    j31 = 0
    j32 = x_p*y_p*p*c_0/(c+c_0)**2
    j33 = x_p*(y_p*c/(c+c_0)-1)
    
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
x_c = 0.4
y_c = 2.009
x_p = 0.08
y_p = 2.876
w_0 = 0.16129
c_0 = 0.5

kbif = 0.5 # bifurcation point (computed in Mathematica)
kh = 1.2*kbif # bifurcation parameter high

for i in np.linspace(0.2,0.4,11):
    
    kl = round(i,2) # bifurcation parameter low

    for c_rate in [1e-5,2e-5,3e-5,4e-5,5e-5]:

        sim_len = int((kh-kl)*dt/c_rate)

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

            w0 = 0.15 # intial condition (equilibrium value computed in Mathematica)
            c0 = 0.08
            p0 = 0.01
            equiw0 = 0.15
            equic0 = 0.08
            equip0 = 0.01
            
            tmax = int(np.random.us(250,500)) # randomly selected sequence length
            n = np.random.us(5,500) # length of the randomly selected sequence to the bifurcation
            series_len = tmax + int(n)

            t = np.arange(t0,sim_len,dt)
            w = np.zeros(len(t))
            c = np.zeros(len(t))
            p = np.zeros(len(t))
            equiw = np.zeros(len(t))
            equic = np.zeros(len(t))
            equip = np.zeros(len(t))
            rrate = np.zeros(len(t))
            k = pd.Series(np.linspace(kl,kh,len(t)),index=t)
            
            # Create brownian increments (s.d. sqrt(dt))
            dW_w_burn = np.random.normal(loc=0, scale=sigma_w*np.sqrt(dt), size = int(tburn/dt))
            dW_w = np.random.normal(loc=0, scale=sigma_w*np.sqrt(dt), size = len(t))
            
            dW_c_burn = np.random.normal(loc=0, scale=sigma_c*np.sqrt(dt), size = int(tburn/dt))
            dW_c = np.random.normal(loc=0, scale=sigma_c*np.sqrt(dt), size = len(t))

            dW_p_burn = np.random.normal(loc=0, scale=sigma_p*np.sqrt(dt), size = int(tburn/dt))
            dW_p = np.random.normal(loc=0, scale=sigma_p*np.sqrt(dt), size = len(t))
            
            # Run burn-in period on x0
            for i in range(int(tburn/dt)):
                w0 = w0 + de_fun_w(w0,c0,k[0],x_c,y_c,w_0)*dt + dW_w_burn[i]
                c0 = c0 + de_fun_c(w0,c0,p0,x_c,y_c,w_0,x_p,y_p,c_0)*dt + dW_c_burn[i]
                p0 = p0 + de_fun_p(p0,c0,x_p,y_p,c_0)*dt + dW_p_burn[i]
                equiw0 = equiw0 + de_fun_w(equiw0,equic0,k[0],x_c,y_c,w_0)*dt
                equic0 = equic0 + de_fun_c(equiw0,equic0,equip0,x_c,y_c,w_0,x_p,y_p,c_0)*dt
                equip0 = equip0 + de_fun_p(equip0,equic0,x_p,y_p,c_0)*dt

            rrate0 = recov_fun(equiw0,equic0,equip0,k[0],x_c,y_c,w_0,x_p,y_p,c_0)
            
            if rrate0 == 1997.0209 or math.isnan(w0) or math.isnan(c0) or math.isnan(p0):
                print('Nan appears')
                w0 = 0.15 # intial condition (equilibrium value computed in Mathematica)
                c0 = 0.08
                p0 = 0.01
                equiw0 = 0.15
                equic0 = 0.08
                equip0 = 0.01
                continue
            
            # Initial condition post burn-in period
            w[0]=w0
            c[0]=c0
            p[0]=p0
            equiw[0]=equiw0
            equic[0]=equic0
            equip[0]=equip0
            rrate[0]=rrate0

            rrate_record = []

            # Run simulation
            for i in range(len(t)-1):
                w[i+1] = w[i] + de_fun_w(w[i],c[i],k.iloc[i],x_c,y_c,w_0)*dt + dW_w[i]
                c[i+1] = c[i] + de_fun_c(w[i],c[i],p[i],x_c,y_c,w_0,x_p,y_p,c_0)*dt + dW_c[i]
                p[i+1] = p[i] + de_fun_p(p[i],c[i],x_p,y_p,c_0)*dt + dW_p[i]
                equiw[i+1] = equiw[i] + de_fun_w(equiw[i],equic[i],k.iloc[i],x_c,y_c,w_0)*dt
                equic[i+1] = equic[i] + de_fun_c(equiw[i],equic[i],equip[i],x_c,y_c,w_0,x_p,y_p,c_0)*dt
                equip[i+1] = equip[i] + de_fun_p(equip[i],equic[i],x_p,y_p,c_0)*dt
                rrate[i+1] = recov_fun(equiw[i+1],equic[i+1],equip[i+1],k.iloc[i+1],x_c,y_c,w_0,x_p,y_p,c_0)

                if rrate[i+1] == 1997.0209:
                    print('rrate Nan appears')
                    break

                if math.isnan(w[i+1]) or math.isnan(c[i+1]) or math.isnan(p[i+1]):
                    print('sim Nan appears')
                    break               
                # Determine the tipping point by the recovery rate changing from negative to positive
                if rrate[i] < 0 and rrate[i+1] > 0:
                    rrate_record.append(i+1)
                    break
                        
                # make sure that state variable remains >= 0 
                if w[i+1] < 0:
                    w[i+1] = 0
                if c[i+1] < 0:
                    c[i+1] = 0
                if p[i+1] < 0:
                    p[i+1] = 0

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
            data = {'tsid': (j+1)*np.ones(len(t)),'Time': t,'x': w,'c': c,'p': p,'k': k.values}
            df_temp = pd.DataFrame(data)
            trans_point = df_temp.loc[trans_time-1,'k']

            df_temp_1 = df_temp.iloc[d1].copy() # irregularly-sampled time series
            df_temp_1['Time'] = np.arange(0,series_len)
            df_temp_1.set_index('Time', inplace=True)
            df_cut_1 = df_temp_1.iloc[0:tmax].copy()

            # Get the minimum and maximum values of the original 'k' column
            k_min = df_cut_1['k'].min()
            k_max = df_cut_1['k'].max()

            # Create a new uniformly distributed 'k' column
            new_k_values = np.linspace(k_min, k_max, tmax)

            # Interpolate the 'State variable'
            interpolated_x_values = np.interp(new_k_values, df_cut_1['k'], df_cut_1['x'])
            interpolated_c_values = np.interp(new_k_values, df_cut_1['k'], df_cut_1['c'])
            interpolated_p_values = np.interp(new_k_values, df_cut_1['k'], df_cut_1['p'])

            df_cut_1['interpolated_x'] = interpolated_x_values
            df_cut_1['interpolated_c'] = interpolated_c_values
            df_cut_1['interpolated_p'] = interpolated_p_values
            df_cut_1['interpolated_k'] = new_k_values

            df_temp_2 = df_temp.iloc[d2].copy() # regularly-sampled time series
            df_temp_2['Time'] = np.arange(0,series_len)
            df_temp_2.set_index('Time', inplace=True)
            df_cut_2 = df_temp_2.iloc[0:tmax].copy()

            nearest_point_1 = df_cut_1.loc[tmax-1,'k']
            nearest_point_2 = df_cut_2.loc[tmax-1,'k']
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

        if not os.path.exists('../data_nus/food_hopf_white/{}/{}/temp'.format(kl,c_rate)):
            os.makedirs('../data_nus/food_hopf_white/{}/{}/temp'.format(kl,c_rate))

        if not os.path.exists('../data_us/food_hopf_white/{}/{}/temp'.format(kl,c_rate)):
            os.makedirs('../data_us/food_hopf_white/{}/{}/temp'.format(kl,c_rate))

        # Filter time-series
        df_traj_1.to_csv('../data_nus/food_hopf_white/{}/{}/temp/df_traj.csv'.format(kl,c_rate))
        df_traj_2.to_csv('../data_us/food_hopf_white/{}/{}/temp/df_traj.csv'.format(kl,c_rate))

        if not os.path.exists('../data_nus/food_hopf_white/{}/{}/dev'.format(kl,c_rate)):
            os.makedirs('../data_nus/food_hopf_white/{}/{}/dev'.format(kl,c_rate))

        if not os.path.exists('../data_us/food_hopf_white/{}/{}/dev'.format(kl,c_rate)):
            os.makedirs('../data_us/food_hopf_white/{}/{}/dev'.format(kl,c_rate))

        #----------------------
        # Compute DEV for each tsid 
        #----------------------
        
        # Compute DEV for irregularly-sampled time series
        print('\nBegin DEV computation\n')

        for i in range(numSims):
            df_traj_x = df_traj_1[df_traj_1['tsid'] == i+1]['interpolated_x']
            df_traj_k = df_traj_1[df_traj_1['tsid'] == i+1]['interpolated_k']
            rdf_x = pandas2ri.py2rpy(df_traj_x)
            rdf_k = pandas2ri.py2rpy(df_traj_k)
            globalenv['x_time_series'] = rdf_x
            globalenv['k_time_series'] = rdf_k
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
                k <- k_time_series[j]
                
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
                matrix_result[index,2] <- k
                matrix_result[index,3] <- mean(matrix_eigen[,1],na.rm=TRUE)
            }

            result <- matrix_result
            '''
            dev_k = r(rscript)[:,1]
            dev_x = r(rscript)[:,2]

            df_dev = pd.DataFrame(data=None,columns=['DEV','k'])
            df_dev['DEV'] = dev_x
            df_dev['k'] = dev_k
            # Export DEV as individual files for dynamical eigenvalue
            filepath_dev = '../data_nus/food_hopf_white/{}/{}/dev/food_hopf_dev_{}.csv'.format(kl,c_rate,i+1)
            df_dev.to_csv(filepath_dev,index=False)

        # Compute DEV for regularly-sampled time series
        for i in range(numSims):
            df_traj_x = df_traj_2[df_traj_2['tsid'] == i+1]['x']
            df_traj_k = df_traj_2[df_traj_2['tsid'] == i+1]['k']
            rdf_x = pandas2ri.py2rpy(df_traj_x)
            rdf_k = pandas2ri.py2rpy(df_traj_k)
            globalenv['x_time_series'] = rdf_x
            globalenv['k_time_series'] = rdf_k
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
                k <- k_time_series[j]
                
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
                matrix_result[index,2] <- k
                matrix_result[index,3] <- mean(matrix_eigen[,1],na.rm=TRUE)
            }

            result <- matrix_result
            '''
            dev_k = r(rscript)[:,1]
            dev_x = r(rscript)[:,2]

            df_dev = pd.DataFrame(data=None,columns=['DEV','k'])
            df_dev['DEV'] = dev_x
            df_dev['k'] = dev_k
            # Export DEV as individual files for dynamical eigenvalue
            filepath_dev = '../data_us/food_hopf_white/{}/{}/dev/food_hopf_dev_{}.csv'.format(kl,c_rate,i+1)
            df_dev.to_csv(filepath_dev,index=False)

            print('DEV for realisation '+str(i+1)+' complete')

        #----------------------
        # Compute lag-1 autocorrelation for each tsid 
        #----------------------

        # set up a list to store output dataframes from PCA time series and lag-1 autocorrelation
        appended_pca_1 = []
        appended_degf_1 = []
        appended_pca_2 = []
        appended_degf_2 = []  
        
        # Compute PCA for degenerate fingerprinting
        print('\nBegin PCA computation\n')

        # Compute PCA for irregularly-sampled time series
        for i in range(numSims):
            df_traj_temp = df_traj_1[df_traj_1['tsid'] == i+1]
            traj = np.array(df_traj_temp[['interpolated_x','interpolated_c','interpolated_p']])
            mean_traj = np.mean(traj,axis=1,keepdims=True)
            traj = traj - mean_traj

            pcamodel = PCA(1)
            pcamodel.fit(traj)

            pca = pcamodel.fit_transform(traj).reshape(-1)
            appended_pca_1.append(pd.Series(pca))

        # Compute PCA for regularly-sampled time series
            df_traj_temp = df_traj_2[df_traj_2['tsid'] == i+1]
            traj = np.array(df_traj_temp[['x','c','p']])
            mean_traj = np.mean(traj,axis=1,keepdims=True)
            traj = traj - mean_traj

            pcamodel = PCA(1)
            pcamodel.fit(traj)

            pca = pcamodel.fit_transform(traj).reshape(-1)
            appended_pca_2.append(pd.Series(pca))

            print('PCA for realisation '+str(i+1)+' complete')

        df_pca_1 = pd.concat(appended_pca_1).reset_index()
        df_traj_1['pca'] = df_pca_1.iloc[:,1]  
        df_pca_2 = pd.concat(appended_pca_2).reset_index()
        df_traj_2['pca'] = df_pca_2.iloc[:,1]

        print('\nBegin lag-1 autocorrelation computation\n')
        # Compute lag-1 autocorrelation for irregularly-sampled time series
        for i in range(numSims):
            df_traj_temp = df_traj_1[df_traj_1['tsid'] == i+1][['pca','interpolated_k']]
            df_traj_x = df_traj_temp['pca']
            ews_dic = ewstools.TimeSeries(data = df_traj_x)
            ews_dic.compute_auto(lag = auto_lag, rolling_window = rw_degf)

            df_degf_temp = ews_dic.ews
            df_degf_temp['tsid'] = i+1
            df_degf_temp['k'] = df_traj_temp['interpolated_k']
                
            # Add DataFrames to list
            appended_degf_1.append(df_degf_temp)

        # Compute lag-1 autocorrelation for regularly-sampled time series
        for i in range(numSims):
            df_traj_temp = df_traj_2[df_traj_2['tsid'] == i+1][['pca','k']]
            df_traj_x = df_traj_temp['pca']
            ews_dic = ewstools.TimeSeries(data = df_traj_x)
            ews_dic.compute_auto(lag = auto_lag, rolling_window = rw_degf)

            df_degf_temp = ews_dic.ews
            df_degf_temp['tsid'] = i+1
            df_degf_temp['k'] = df_traj_temp['k']
                
            # Add DataFrames to list
            appended_degf_2.append(df_degf_temp)
                
            print('lag-1 autocorrelation for realisation '+str(i+1)+' complete')

        # Concatenate EWS DataFrames
        df_traj_1.reset_index(inplace=True)
        df_traj_2.reset_index(inplace=True)
        df_degf_1 = pd.concat(appended_degf_1).reset_index()
        df_degf_2 = pd.concat(appended_degf_2).reset_index()


        # Create directories for output
        if not os.path.exists('../data_nus/food_hopf_white/{}/{}/sims'.format(kl,c_rate)):
            os.makedirs('../data_nus/food_hopf_white/{}/{}/sims'.format(kl,c_rate))

        if not os.path.exists('../data_nus/food_hopf_white/{}/{}/ac'.format(kl,c_rate)):
            os.makedirs('../data_nus/food_hopf_white/{}/{}/ac'.format(kl,c_rate))

        if not os.path.exists('../data_us/food_hopf_white/{}/{}/sims'.format(kl,c_rate)):
            os.makedirs('../data_us/food_hopf_white/{}/{}/sims'.format(kl,c_rate))

        if not os.path.exists('../data_us/food_hopf_white/{}/{}/ac'.format(kl,c_rate)):
            os.makedirs('../data_us/food_hopf_white/{}/{}/ac'.format(kl,c_rate))

        # Export time series as individual files for training ML
        # Export lag-1 autocorrelation as individual files for degenerate fingerprinting
        for i in np.arange(numSims)+1:
            df_sims = df_traj_1[df_traj_1['tsid'] == i][['Time','x','k']]
            df_ac = df_degf_1[df_degf_1['tsid'] == i][['ac1','k']]
            df_ac = df_ac[df_ac['ac1'].notna()]
            filepath_sims='../data_nus/food_hopf_white/{}/{}/sims/food_hopf_sims_{}.csv'.format(kl,c_rate,i)
            filepath_ac='../data_nus/food_hopf_white/{}/{}/ac/food_hopf_ac_{}.csv'.format(kl,c_rate,i)
            df_sims.to_csv(filepath_sims,index=False)
            df_ac.to_csv(filepath_ac,index=False)

        for i in np.arange(numSims)+1:
            df_sims = df_traj_2[df_traj_2['tsid'] == i][['Time','x','k']]
            df_ac = df_degf_2[df_degf_2['tsid'] == i][['ac1','k']]
            df_ac = df_ac[df_ac['ac1'].notna()]
            filepath_sims='../data_us/food_hopf_white/{}/{}/sims/food_hopf_sims_{}.csv'.format(kl,c_rate,i)
            filepath_ac='../data_us/food_hopf_white/{}/{}/ac/food_hopf_ac_{}.csv'.format(kl,c_rate,i)
            df_sims.to_csv(filepath_sims,index=False)
            df_ac.to_csv(filepath_ac,index=False)

        df_result_1.to_csv('../data_nus/food_hopf_white/{}/{}/food_hopf_result.csv'.format(kl,c_rate))
        df_result_2.to_csv('../data_us/food_hopf_white/{}/{}/food_hopf_result.csv'.format(kl,c_rate))

        print('kl = {}, c_rate = {} has finished'.format(kl,c_rate))