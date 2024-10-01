#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

dt = 0.01

# Positive cubic coeff - subcritical bif
# Negative cubic coeff - supercritical bif
sup_cubic_coeff = -1
sub_cubic_coeff = 1

def de_fun_sup_pf(x, mu, dict_coeffs):
    
    # Get sum of higher-order terms
    sum_hot = 0
    for order in dict_coeffs.keys():
        sum_hot += dict_coeffs[order] * x**order
    
    sup_delta_f = mu*x + sup_cubic_coeff * x**3 + sum_hot
    
    return sup_delta_f

def de_fun_sub_pf(x, mu, dict_coeffs):
    
    # Get sum of higher-order terms
    sum_hot = 0
    for order in dict_coeffs.keys():
        sum_hot += dict_coeffs[order] * x**order
    
    sub_delta_f = mu*x + sub_cubic_coeff * x**3 + sum_hot
    
    return sub_delta_f

def recov_fun_sup_pf(x, mu, dict_coeffs):

    # Get sum of higher-order terms
    recov_sum_hot = 0
    for order in dict_coeffs.keys():
        recov_sum_hot += dict_coeffs[order] * order * x**(order-1)
    
    sup_rrate = mu + sup_cubic_coeff * 3 * x**2 + recov_sum_hot

    return sup_rrate

# def recov_fun_sub_pf(x, mu, dict_coeffs):

#     # Get sum of higher-order terms
#     recov_sum_hot = 0
#     for order in dict_coeffs.keys():
#         recov_sum_hot += dict_coeffs[order] * order * x**(order-1)
    
#     sub_rrate = mu + sub_cubic_coeff * 3 * x**2 + recov_sum_hot

#     return sub_rrate


def simulate_pf(bl=-1, bh=0.2, tmax=500, series_len=500, tburn=100,
                sigma=0.01, max_order=10):
    '''
    Simulate a trajectory of the normal form for the pitchfork
    bifurcation with the bifurcation parameter going from bl to bh
    
    Return deviation from analytical equilibrium (residuals)
    
    Parameters:
        bl: starting value of bifurcation parameter
        bh: end value of bifurcation parameter
        tmax: number of time points
        tburn: number of time points in burn-in period
        sigma: noise amplitude
        max_order: highest-order term in normal form expansion
        dev_thresh: threshold deviation that defines start of transition
        
    Output:
        pd.DataFrame
            Contains time and state variable.
            State is Nan after model transitions.
            Returns all Nan if model diverges during burn-in period.         
    '''

    # Initial condition (equilibrium)
    sup_x0 = 0
    sup_equix0 = 0
    sub_x0 = 0
    # sub_equix0 = 0

    # Time values 
    t = np.arange(0,tmax,dt)
    # Linearly increasing bifurcation parameter
    b = pd.Series(np.linspace(bl,bh,len(t)),index=t) 
    
    # Set random coefficients for higher-order terms
    # Note PF goes up to order 3 so include orders 4 and above.
    dict_coeffs = {order: np.random.normal(0,1) for \
                   order in np.arange(4,max_order+1)}
    
    # Create brownian increments
    dW_burn = np.random.normal(loc=0, scale=sigma, size = int(tburn))
    dW = np.random.normal(loc=0, scale=sigma, size = len(t))
        
    # Run burn-in period on x0
    for i in range(int(tburn)):
        sup_x0 = sup_x0 + de_fun_sup_pf(sup_x0,bl,dict_coeffs) + dW_burn[i]
        sup_equix0 = sup_equix0 + de_fun_sup_pf(sup_equix0,bl,dict_coeffs)

        sub_x0 = sub_x0 + de_fun_sub_pf(sub_x0,bl,dict_coeffs) + dW_burn[i]
        # sub_equix0 = sub_equix0 + de_fun_sub_pf(sub_equix0,bl,dict_coeffs)

        # If blows up
        if abs(sup_x0) > 1e6 or abs(sup_equix0) > 1e6 or abs(sub_x0) > 1e6:
            print('Model diverged during burn in period')

            return 1,1,1,1,1
        
    sup_rrate0 = recov_fun_sup_pf(sup_equix0,bl,dict_coeffs)
    # sub_rrate0 = recov_fun_sub_pf(sub_equix0,bl,dict_coeffs)
        
    # Run simulation
    sup_x = np.zeros(len(t))
    sup_equix = np.zeros(len(t))
    sup_rrate = np.zeros(len(t))

    sub_x = np.zeros(len(t))
    # sub_equix = np.zeros(len(t))
    # sub_rrate = np.zeros(len(t))

    sup_x[0] = sup_x0
    sup_equix[0] = sup_equix0
    sup_rrate[0] = sup_rrate0

    sub_x[0] = sub_x0
    # sub_equix[0] = sub_equix0
    # sub_rrate[0] = sub_rrate0

    sup_rrate_record = []
    # sub_rrate_record = []

    # sup_break = 0
    # sub_break = 0

    for i in range(len(t)-1):
        sup_x[i+1] = sup_x[i] + de_fun_sup_pf(sup_x[i],b.iloc[i],dict_coeffs) + dW[i]
        sup_equix[i+1] = sup_equix[i] + de_fun_sup_pf(sup_equix[i],b.iloc[i],dict_coeffs)
        sup_rrate[i+1] = recov_fun_sup_pf(sup_equix[i+1],b.iloc[i+1],dict_coeffs)

        sub_x[i+1] = sub_x[i] + de_fun_sub_pf(sub_x[i],b.iloc[i],dict_coeffs) + dW[i]
        # sub_equix[i+1] = sub_equix[i] + de_fun_sub_pf(sub_equix[i],b.iloc[i],dict_coeffs)
        # sub_rrate[i+1] = recov_fun_sub_pf(sub_equix[i+1],b.iloc[i+1],dict_coeffs)
        
        # Determine the tipping point by the recovery rate changing from negative to positive
        if sup_rrate[i] < 0 and sup_rrate[i+1] > 0:
            sup_rrate_record.append(i+1)
            
            #sup_break = 1
            break
        
        # if sub_rrate[i] < 0 and sub_rrate[i+1] > 0:
        #     sub_rrate_record.append(i+1)

        #     sub_break = 1

        # if sup_break and sub_break:
        #     break

    if len(sup_rrate_record) != 0:

        sup_trans_time = sup_rrate_record[0] # the time of tipping point
        # sub_trans_time = sub_rrate_record[0]

        sup_ts = np.arange(0,sup_trans_time)
        # sub_ts = np.arange(0,sub_trans_time)

        c1 = np.linspace(1,len(sup_ts)-2,len(sup_ts)-2)
        np.random.shuffle(c1)
        c1 = list(np.sort(c1[0:series_len-2]))
        c1.insert(0,0)
        c1.append(len(sup_ts)-1)

        # c2 = np.linspace(1,len(sub_ts)-2,len(sub_ts)-2)
        # np.random.shuffle(c2)
        # c2 = list(np.sort(c2[0:series_len-2]))
        # c2.insert(0,0)
        # c2.append(len(sub_ts)-1)

    else:
        print('Cant find rrate = 0')
        
        return 1,1,1,1,1
            
    # Store series data in a temporary DataFrame
    sup_data = {'Time': t,'x': sup_x,'b': b.values}
    sub_data = {'Time': t,'x': sub_x,'b': b.values}

    sup_df_temp = pd.DataFrame(sup_data)
    sub_df_temp = pd.DataFrame(sub_data)

    sup_trans_point = sup_df_temp.loc[sup_trans_time-1,'b']
    sub_trans_point = sub_df_temp.loc[sup_trans_time-1,'b']

    sup_df_temp_1 = sup_df_temp.iloc[c1].copy() # irregularly-sampled time series
    sup_df_temp_1['Time'] = np.arange(0,series_len)
    sup_df_temp_1.set_index('Time', inplace=True)
    sup_df_cut_1 = sup_df_temp_1.iloc[0:tmax].copy()

    sub_df_temp_1 = sub_df_temp.iloc[c1].copy() # irregularly-sampled time series
    sub_df_temp_1['Time'] = np.arange(0,series_len)
    sub_df_temp_1.set_index('Time', inplace=True)
    sub_df_cut_1 = sub_df_temp_1.iloc[0:tmax].copy()
        
    
    return 0,sup_df_cut_1, sub_df_cut_1, sup_trans_point, sub_trans_point




