#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on May 4 2024

@author: Chengzuo Zhuge

Modified by: Thomas M. Bury


"""

import numpy as np
import pandas as pd
import csv
import random
import math

    
def sim_model(model, relative_rate=1, sim_len=500, series_len=500, sigma=0.1, null_sim=False, null_location=0):
    '''
    Function to run a stochastic simulation of model up to bifurcation point
    Input:
        model (class) : contains details of model to simulate
        relative_rate : relative rate of change of the bifurcation parameter
        series_len : number of points in time series
        sigma (float) : amplitude factor of GWN - total amplitude also
            depends on parameter values
        null_sim (bool) : Null simulation (bifurcation parameter fixed) or
            transient simulation (bifurcation parameter increments to bifurcation point)
        null_location (float) : Value in [0,1] to determine location along bifurcation branch
            where null is simulated. Value is proportion of distance to the 
            bifurcation from initial point (0 is initial point, 1 is bifurcation point)
    Output:
        DataFrame of trajectories indexed by time
    '''
    
    # Simulation parameters
    dt = 0.01
    t0 = 0
    tburn = 100 # burn-in period
    ac = random.uniform(0.3,1) # lag-1 autoregressive coefficient of red noise
#   seed = 0 # random number generation 
    
    # Bifurcation point of model
    bcrit = model.bif_value # bifurcation point
    
    # Initial value of bifurcation parameter
    bl = model.pars[model.bif_param]
    # Final value of bifurcation parameter
    bh = 1.2*bcrit - 0.2*bl # bcrit + 0.2*(bcrit-bl)
    
    s0 = model.equi_init    # initial condition
    equi0 = model.equi_init

    # Parameter labels
    parlabels_a = ['a'+str(i) for i in np.arange(1,11)]
    parlabels_b = ['b'+str(i) for i in np.arange(1,11)]
    
    # Model equations

    def de_fun(s,pars):
        '''
        Input:
        s is state vector
        pars is dictionary of parameter values
        
        Output:
        array [dxdt, dydt]
        
        '''
        
        # Obtain model parameters from dictionary
        pars_a = np.array([pars[k] for k in parlabels_a])
        pars_b = np.array([pars[k] for k in parlabels_b])
        
        # Polynomial forms up to third order
        x = s[0]
        y = s[1]
        polys = np.array([1,x,y,x**2,x*y,y**2,x**3,x**2*y,x*y**2,y**3])
        
        dxdt = np.dot(pars_a, polys)
        dydt = np.dot(pars_b, polys)
                      
        return np.array([dxdt, dydt])

    def recov_fun(equi,pars):

        pars_a = np.array([pars[k] for k in parlabels_a])
        pars_b = np.array([pars[k] for k in parlabels_b])

        [a2,a3,a4,a5,a6,a7,a8,a9,a10] = pars_a[1:]
        [b2,b3,b4,b5,b6,b7,b8,b9,b10] = pars_b[1:]

        pars_j11 = np.array([a2,a4,a5,a7,a8,a9])
        pars_j12 = np.array([a3,a6,a5,a10,a9,a8])
        pars_j21 = np.array([b2,b4,b5,b7,b8,b9])
        pars_j22 = np.array([b3,b6,b5,b10,b9,b8])

        x = equi[0]
        y = equi[1]
        polys_j1 = np.array([1,2*x,y,3*x**2,2*x*y,y**2])
        polys_j2 = np.array([1,2*y,x,3*y**2,2*x*y,x**2])

        # df1/dx
        j11 = np.dot(pars_j11,polys_j1)
        # df1/dy
        j12 = np.dot(pars_j12,polys_j2)
        # df2/dx
        j21 = np.dot(pars_j21,polys_j1)
        # df2/dy
        j22 = np.dot(pars_j22,polys_j2)

        # Assign component to Jacobian
        jac = np.array([[j11,j12],[j21,j22]])
        
        # Compute eigenvalues
        try:
            evals = np.linalg.eigvals(jac)
        except:
            return 1997.0209 # if error; This particular number has no other significance, it's just the author's birthday

        # Compute the real part of the dominant eigenvalue (smallest magnitude)
        re_evals = [lam.real for lam in evals]
        dom_eval_re = max(re_evals)
        # Recovery rate is amplitude of this
        rrate = dom_eval_re

        return rrate      

    # Initialise arrays to store single time-series data
    t = np.arange(t0, sim_len*relative_rate, dt)# relative_rate : relative rate of change of the bifurcation parameter
                                               # series_len : number of points in time series
    sw = np.zeros([len(t),2])
    sr = np.zeros([len(t),2])
    equi = np.zeros([len(t),2])
    n = np.zeros([len(t),2])
    r = np.zeros(len(t))
    
    # Import recovery rate
    with open('output_model/rrate.csv') as csvfile:
        rrate_raw = list(csv.reader(csvfile))          

    # Set up bifurcation parameter b, that increases linearly in time from bl to bh
    b = pd.Series(np.linspace(bl,bh,len(t)),index=t)
    
    ## Implement Euler Maryuyama for stocahstic simulation
    
    # Create brownian increments (s.d. sqrt(dt))
    dW_burn = np.random.normal(loc=0, scale=sigma*np.sqrt(dt), size = [int(tburn/dt),2])
    dW = np.random.normal(loc=0, scale=sigma*np.sqrt(dt), size = [len(t),2])
    
    # Run burn-in period on s0
    for i in range(int(tburn/dt)):
        s0 = s0 + de_fun(s0, model.pars)*dt + dW_burn[i]
        
    # Initial condition post burn-in period
    sw[0] = s0
    sr[0] = s0
    equi[0] = equi0
    n[0] = dW[0]
    r[0] = float(rrate_raw[0][0])
    
    rrate_record = []

    # Run simulation
    for i in range(len(t)-1):
        # Update bifurcation parameter
        pars = dict(model.pars)
        parr = dict(model.pars)
        pars[model.bif_param] = b.iloc[i]
        parr[model.bif_param] = b.iloc[i+1]
        
        sw[i+1] = sw[i] + de_fun(sw[i],pars)*dt + dW[i] # Equilibrium point in white noise
        sr[i+1] = sr[i] + de_fun(sr[i],pars)*dt + n[i]  # Equilibrium point in red noise
        equi[i+1] = equi[i] + de_fun(equi[i],pars)*dt # Equilibrium point
        n[i+1] = ac*n[i] + dW[i] # red noise
        r[i+1] = recov_fun(equi[i+1],parr) # rrate

        if r[i+1] == 1997.0209: # rrate Nan appears
            return 1,1,True,False,False
        
        if math.isnan(sw[i+1][0]) or math.isnan(sr[i+1][0]) or math.isnan(sw[i+1][1]) or math.isnan(sr[i+1][1]):# sim Nan appears
            return 1,1,False,False,True
        
        # Determine the tipping point by the recovery rate changing from negative to positive
        if r[i] < 0 and r[i+1] > 0: 
            rrate_record.append(i+1)
            break
    
    if len(rrate_record) != 0:

        trans_time = rrate_record[0]

        if trans_time < series_len:
            return 1,1,False,True,False # Cant find rrate = 0

        ts = np.arange(0,trans_time)

        c = np.linspace(1,len(ts)-2,len(ts)-2)
        np.random.shuffle(c)
        c = list(np.sort(c[0:series_len-2]))
        c.insert(0,0)
        c.append(len(ts)-1)

        # Store series data in a DataFrame
        data = {'Time': t,'xw': sw[:,0],'yw': sw[:,1],'xr': sr[:,0],'yr': sr[:,1],'b': b.values,'rrate': r}
        df_traj = pd.DataFrame(data)

        # Filter dataframe according to spacing
        df_traj_filt = df_traj.iloc[c].copy()
        
        # Replace time column with integers for compatibility
        # with trans_detect
        df_traj_filt['Time'] = np.arange(0,series_len)
        df_traj_filt.set_index('Time', inplace=True)

        return df_traj_filt,bcrit,False,False,False # No mistakes

    else:

        return 1,1,False,True,False # Cant find rrate = 0
   
    
    
    

