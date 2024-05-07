#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on: March 22, 2024

@author: Chengzuo Zhuge

Simulate sleep-wake model 
Simulations going through fold/fold hysteresis loop

"""

# import python libraries
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import os

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
seed = 0 # random number generation seed
sigma_v = 0.01 # noise intensity
sigma_m = 0.01

#----------------------------------
# Simulate model
#----------------------------------

# def exp(-(v-theta)/sigma) function

def s(V,theta,sigma):
    return math.exp(-(V-theta)/sigma)

# Model (see A.J.K.Phillips and P.A.Robinson 2007)

def de_fun_v(Vv,Vm,t_v,vm,Q_max,theta,sigma,D):
    return (-Vv + vm*Q_max/(1+s(Vm,theta,sigma)) + D)/t_v

def de_fun_m(Vv,Vm,t_m,mv,Q_max,theta,sigma,maQa):
    return (-Vm + maQa + mv*Q_max/(1+s(Vv,theta,sigma)))/t_m

def recov_fun(Vv,Vm,t_v,t_m,vm,mv,Q_max,theta,sigma):

    j11 = -1/t_v
    j12 = -vm*Q_max*s(Vm,theta,sigma)/((t_v*sigma)*(1+s(Vm,theta,sigma))**2)
    j21 = -mv*Q_max*s(Vv,theta,sigma)/((t_m*sigma)*(1+s(Vv,theta,sigma))**2)
    j22 = -1/t_m

    jac = np.array([[j11,j12],[j21,j22]])

    evals = np.linalg.eigvals(jac)

    re_evals = [lam.real for lam in evals]
    dom_eval_re = max(re_evals)

    rrate = dom_eval_re

    return rrate

# Model parameters
t_v = 10
t_m = 10
vm = -1.9
mv = -1.9
maQa = 1
Q_max = 100
theta = 10
sigma = 3

Dl = 0.1
Dh = 1.9
D_rate = 0.5/3600 # rate: delta D/s

sim_len = int((Dh-Dl)/D_rate)

# 1---------------------------------------------------------------------------------

v10 = -8.5
m10 = 1
equiv10 = -8.5
equim10 = 1

t = np.arange(t0,sim_len,dt)
v1 = np.zeros(len(t))
m1 = np.zeros(len(t))
equiv1 = np.zeros(len(t))
equim1 = np.zeros(len(t))
rrate1 = np.zeros(len(t))
d1 = pd.Series(np.linspace(Dl,Dh,len(t)),index=t)

# Create brownian increments (s.d. sqrt(dt))
dW_v_burn = np.random.normal(loc=0, scale=sigma_v*np.sqrt(dt), size = int(tburn/dt))
dW_v = np.random.normal(loc=0, scale=sigma_v*np.sqrt(dt), size = len(t))

dW_m_burn = np.random.normal(loc=0, scale=sigma_m*np.sqrt(dt), size = int(tburn/dt))
dW_m = np.random.normal(loc=0, scale=sigma_m*np.sqrt(dt), size = len(t))

# Run burn-in period on x0
for i in range(int(tburn/dt)):
    v10 = v10 + de_fun_v(v10,m10,t_v,vm,Q_max,theta,sigma,d1[0])*dt + dW_v_burn[i]
    m10 = m10 + de_fun_m(v10,m10,t_m,mv,Q_max,theta,sigma,maQa)*dt + dW_m_burn[i]
    equiv10 = equiv10 + de_fun_v(equiv10,equim10,t_v,vm,Q_max,theta,sigma,d1[0])*dt
    equim10 = equim10 + de_fun_m(equiv10,equim10,t_m,mv,Q_max,theta,sigma,maQa)*dt

rrate10 = recov_fun(v10,m10,t_v,t_m,vm,mv,Q_max,theta,sigma)

# Initial condition post burn-in period
v1[0]=v10
m1[0]=m10
equiv1[0]=equiv10
equim1[0]=equim10
rrate1[0]=rrate10

rrate_record1 = []

# Run simulation
for i in range(len(t)-1):
    v1[i+1] = v1[i] + de_fun_v(v1[i],m1[i],t_v,vm,Q_max,theta,sigma,d1.iloc[i])*dt + dW_v[i]
    m1[i+1] = m1[i] + de_fun_m(v1[i],m1[i],t_m,mv,Q_max,theta,sigma,maQa)*dt + dW_m[i]
    equiv1[i+1] = equiv1[i] + de_fun_v(equiv1[i],equim1[i],t_v,vm,Q_max,theta,sigma,d1.iloc[i])*dt
    equim1[i+1] = equim1[i] + de_fun_m(equiv1[i],equim1[i],t_m,mv,Q_max,theta,sigma,maQa)*dt
    rrate1[i+1] = recov_fun(equiv1[i+1],equim1[i+1],t_v,t_m,vm,mv,Q_max,theta,sigma)

    if rrate1[i] < 0 and rrate1[i+1] > 0:
        rrate_record1.append(i+1)

trans_time1 = rrate_record1[0]
trans_value1 = d1.iloc[trans_time1-1]

# ts = np.arange(0,trans_time1)

# s1 = np.linspace(1,len(ts)-2,len(ts)-2)
# np.random.shuffle(s1)
# s1 = list(np.sort(s1[0:series_len-2]))
# s1.insert(0,0)
# s1.append(len(ts)-1)

# 2---------------------------------------------------------------------------------

v20 = 1
m20 = -8
equiv20 = 1
equim20 = -8

t = np.arange(t0,sim_len,dt)
v2 = np.zeros(len(t))
m2 = np.zeros(len(t))
equiv2 = np.zeros(len(t))
equim2 = np.zeros(len(t))
rrate2 = np.zeros(len(t))
d2 = pd.Series(np.linspace(Dh,Dl,len(t)),index=t)

# Create brownian increments (s.d. sqrt(dt))
dW_v_burn = np.random.normal(loc=0, scale=sigma_v*np.sqrt(dt), size = int(tburn/dt))
dW_v = np.random.normal(loc=0, scale=sigma_v*np.sqrt(dt), size = len(t))

dW_m_burn = np.random.normal(loc=0, scale=sigma_m*np.sqrt(dt), size = int(tburn/dt))
dW_m = np.random.normal(loc=0, scale=sigma_m*np.sqrt(dt), size = len(t))

# Run burn-in period on x0
for i in range(int(tburn/dt)):
    v20 = v20 + de_fun_v(v20,m20,t_v,vm,Q_max,theta,sigma,d2[0])*dt + dW_v_burn[i]
    m20 = m20 + de_fun_m(v20,m20,t_m,mv,Q_max,theta,sigma,maQa)*dt + dW_m_burn[i]
    equiv20 = equiv20 + de_fun_v(equiv20,equim20,t_v,vm,Q_max,theta,sigma,d2[0])*dt
    equim20 = equim20 + de_fun_m(equiv20,equim20,t_m,mv,Q_max,theta,sigma,maQa)*dt

rrate20 = recov_fun(v20,m20,t_v,t_m,vm,mv,Q_max,theta,sigma)

# Initial condition post burn-in period
v2[0]=v20
m2[0]=m20
equiv2[0]=equiv20
equim2[0]=equim20
rrate2[0]=rrate20

rrate_record2 = []

# Run simulation
for i in range(len(t)-1):
    v2[i+1] = v2[i] + de_fun_v(v2[i],m2[i],t_v,vm,Q_max,theta,sigma,d2.iloc[i])*dt + dW_v[i]
    m2[i+1] = m2[i] + de_fun_m(v2[i],m2[i],t_m,mv,Q_max,theta,sigma,maQa)*dt + dW_m[i]
    equiv2[i+1] = equiv2[i] + de_fun_v(equiv2[i],equim2[i],t_v,vm,Q_max,theta,sigma,d2.iloc[i])*dt
    equim2[i+1] = equim2[i] + de_fun_m(equiv2[i],equim2[i],t_m,mv,Q_max,theta,sigma,maQa)*dt
    rrate2[i+1] = recov_fun(equiv2[i+1],equim2[i+1],t_v,t_m,vm,mv,Q_max,theta,sigma)     

    if rrate2[i] < 0 and rrate2[i+1] > 0:
        rrate_record2.append(i+1)

trans_time2 = rrate_record2[0]
trans_value2 = d2.iloc[trans_time2-1]

# ts = np.arange(0,trans_time2)

# s2 = np.linspace(1,len(ts)-2,len(ts)-2)
# np.random.shuffle(s2)
# s2 = list(np.sort(s2[0:series_len-2]))
# s2.insert(0,0)
# s2.append(len(ts)-1)

data1 = {'Time': t,'v': v1,'m': m1,'D': d1.values}
data2 = {'Time': t,'v': v2,'m': m2,'D': d2.values}
trans = {'trans_time1':trans_time1,'trans_value1':trans_value1,'trans_time2':trans_time2,'trans_value2':trans_value2}

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)
trans = pd.DataFrame(trans,index=[0])

df1.to_csv('../data_hysteresis/sleep-wake_white/original series/sleep-wake_forward.csv')
df2.to_csv('../data_hysteresis/sleep-wake_white/original series/sleep-wake_reverse.csv')
trans.to_csv('../data_hysteresis/sleep-wake_white/original series/sleep-wake_trans.csv')

l1 = np.linspace(Dl,Dh,len(t))
l2 = np.linspace(Dh,Dl,len(t))
plt.scatter(l1,v1,s=1,c='blue')
plt.scatter(l2,v2,s=1,c='red')
plt.axvline(trans_value1,color='blue', linestyle='--')
plt.axvline(trans_value2,color='red', linestyle='--')
plt.savefig('sleep-wake.png',dpi=600)
plt.show()



