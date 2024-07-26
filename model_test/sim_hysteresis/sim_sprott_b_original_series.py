#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on: March 22, 2024

@author: Chengzuo Zhuge

Simulate Sprott B model 
Simulations going through hopf/hopf hysteresis bursting

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
sigma_x = 0.01 # noise intensity
sigma_y = 0.01
sigma_z = 0.01

#----------------------------------
# Simulate model
#----------------------------------

# Model (see gellner et al. 2016)

def de_fun_x(x,y,a):
    return a*(y-x)

def de_fun_y(x,z,k,beta):
    return x*z + beta*math.cos(k)

def de_fun_z(x,y,b):
    return b - x*y

def recov_fun(x,y,z,a):

    j11 = -a
    j12 = a
    j13 = 0
    j21 = z
    j22 = 0
    j23 = x
    j31 = -y
    j32 = -x
    j33 = 0

    jac = np.array([[j11,j12,j13],[j21,j22,j23],[j31,j32,j33]])

    evals = np.linalg.eigvals(jac)

    re_evals = [lam.real for lam in evals]
    dom_eval_re = max(re_evals)

    rrate = dom_eval_re

    return rrate

# Model parameters
a = 8
b = 2.89
beta = 5

kl = math.pi
kh = 2*math.pi
k_rate = math.pi/1000 # rate: delta k in a unit time

sim_len = int((kh-kl)/k_rate)
tmax = 500
series_len = 800

# 1---------------------------------------------------------------------------------

x10 = 1.7
y10 = 1.7
z10 = 2.94
equix10 = 1.7
equiy10 = 1.7
equiz10 = 2.94

t = np.arange(t0,sim_len,dt)
x1 = np.zeros(len(t))
y1 = np.zeros(len(t))
z1 = np.zeros(len(t))
equix1 = np.zeros(len(t))
equiy1 = np.zeros(len(t))
equiz1 = np.zeros(len(t))
rrate1 = np.zeros(len(t))
k1 = pd.Series(np.linspace(kl,kh,len(t)),index=t)

# Create brownian increments (s.d. sqrt(dt))
dW_x_burn = np.random.normal(loc=0, scale=sigma_x*np.sqrt(dt), size = int(tburn/dt))
dW_x = np.random.normal(loc=0, scale=sigma_x*np.sqrt(dt), size = len(t))

dW_y_burn = np.random.normal(loc=0, scale=sigma_y*np.sqrt(dt), size = int(tburn/dt))
dW_y = np.random.normal(loc=0, scale=sigma_y*np.sqrt(dt), size = len(t))

dW_z_burn = np.random.normal(loc=0, scale=sigma_z*np.sqrt(dt), size = int(tburn/dt))
dW_z = np.random.normal(loc=0, scale=sigma_z*np.sqrt(dt), size = len(t))

# Run burn-in period on x0
for i in range(int(tburn/dt)):
    x10 = x10 + de_fun_x(x10,y10,a)*dt + dW_x_burn[i]
    y10 = y10 + de_fun_y(x10,z10,k1[0],beta)*dt + dW_y_burn[i]
    z10 = z10 + de_fun_z(x10,y10,b)*dt + dW_z_burn[i]
    equix10 = equix10 + de_fun_x(equix10,equiy10,a)*dt
    equiy10 = equiy10 + de_fun_y(equix10,equiz10,k1[0],beta)*dt
    equiz10 = equiz10 + de_fun_z(equix10,equiy10,b)*dt

rrate10 = recov_fun(equix10,equiy10,equiz10,a)

# Initial condition post burn-in period
x1[0]=x10
y1[0]=y10
z1[0]=z10
equix1[0]=equix10
equiy1[0]=equiy10
equiz1[0]=equiz10
rrate1[0]=rrate10

rrate_record1 = []

# Run simulation
for i in range(len(t)-1):
    x1[i+1] = x1[i] + de_fun_x(x1[i],y1[i],a)*dt + dW_x[i]
    y1[i+1] = y1[i] + de_fun_y(x1[i],z1[i],k1.iloc[i],beta)*dt + dW_y[i]
    z1[i+1] = z1[i] + de_fun_z(x1[i],y1[i],b)*dt + dW_z[i]
    equix1[i+1] = equix1[i] + de_fun_x(equix1[i],equiy1[i],a)*dt
    equiy1[i+1] = equiy1[i] + de_fun_y(equix1[i],equiz1[i],k1.iloc[i],beta)*dt
    equiz1[i+1] = equiz1[i] + de_fun_z(equix1[i],equiy1[i],b)*dt
    rrate1[i+1] = recov_fun(equix1[i+1],equiy1[i+1],equiz1[i+1],a)

    if rrate1[i] < 0 and rrate1[i+1] > 0:
        rrate_record1.append(i+1)

trans_time1 = rrate_record1[0]
trans_value1 = k1.iloc[trans_time1-1]

ts = np.arange(0,trans_time1)

# s1 = np.linspace(1,len(ts)-2,len(ts)-2)
# np.random.shuffle(s1)
# s1 = list(np.sort(s1[0:series_len-2]))
# s1.insert(0,0)
# s1.append(len(ts)-1)

# 2---------------------------------------------------------------------------------

x20 = -1.7
y20 = -1.7
z20 = 2.94
equix20 = -1.7
equiy20 = -1.7
equiz20 = 2.94

t = np.arange(t0,sim_len,dt)
x2 = np.zeros(len(t))
y2 = np.zeros(len(t))
z2 = np.zeros(len(t))
equix2 = np.zeros(len(t))
equiy2 = np.zeros(len(t))
equiz2 = np.zeros(len(t))
rrate2 = np.zeros(len(t))
k2 = pd.Series(np.linspace(kh,kl,len(t)),index=t)

# Create brownian increments (s.d. sqrt(dt))
dW_x_burn = np.random.normal(loc=0, scale=sigma_x*np.sqrt(dt), size = int(tburn/dt))
dW_x = np.random.normal(loc=0, scale=sigma_x*np.sqrt(dt), size = len(t))

dW_y_burn = np.random.normal(loc=0, scale=sigma_y*np.sqrt(dt), size = int(tburn/dt))
dW_y = np.random.normal(loc=0, scale=sigma_y*np.sqrt(dt), size = len(t))

dW_z_burn = np.random.normal(loc=0, scale=sigma_z*np.sqrt(dt), size = int(tburn/dt))
dW_z = np.random.normal(loc=0, scale=sigma_z*np.sqrt(dt), size = len(t))

# Run burn-in period on x0
for i in range(int(tburn/dt)):
    x20 = x20 + de_fun_x(x20,y20,a)*dt + dW_x_burn[i]
    y20 = y20 + de_fun_y(x20,z20,k2[0],beta)*dt + dW_y_burn[i]
    z20 = z20 + de_fun_z(x20,y20,b)*dt + dW_z_burn[i]
    equix20 = equix20 + de_fun_x(equix20,equiy20,a)*dt
    equiy20 = equiy20 + de_fun_y(equix20,equiz20,k2[0],beta)*dt
    equiz20 = equiz20 + de_fun_z(equix20,equiy20,b)*dt

rrate20 = recov_fun(equix20,equiy20,equiz20,a)

# Initial condition post burn-in period
x2[0]=x20
y2[0]=y20
z2[0]=z20
equix2[0]=equix20
equiy2[0]=equiy20
equiz2[0]=equiz20
rrate2[0]=rrate20

rrate_record2 = []

# Run simulation
for i in range(len(t)-1):
    x2[i+1] = x2[i] + de_fun_x(x2[i],y2[i],a)*dt + dW_x[i]
    y2[i+1] = y2[i] + de_fun_y(x2[i],z2[i],k2.iloc[i],beta)*dt + dW_y[i]
    z2[i+1] = z2[i] + de_fun_z(x2[i],y2[i],b)*dt + dW_z[i]
    equix2[i+1] = equix2[i] + de_fun_x(equix2[i],equiy2[i],a)*dt
    equiy2[i+1] = equiy2[i] + de_fun_y(equix2[i],equiz2[i],k2.iloc[i],beta)*dt
    equiz2[i+1] = equiz2[i] + de_fun_z(equix2[i],equiy2[i],b)*dt
    rrate2[i+1] = recov_fun(equix2[i+1],equiy2[i+1],equiz2[i+1],a)

    if rrate2[i] < 0 and rrate2[i+1] > 0:
        rrate_record2.append(i+1)

trans_time2 = rrate_record2[0]
trans_value2 = k2.iloc[trans_time2-1]

ts = np.arange(0,trans_time2)

data1 = {'Time': t,'x': x1,'y': y1,'z': z1,'k': k1.values}
data2 = {'Time': t,'x': x2,'y': y2,'z': z2,'k': k2.values}
trans = {'trans_time1':trans_time1,'trans_value1':trans_value1,'trans_time2':trans_time2,'trans_value2':trans_value2}

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)
trans = pd.DataFrame(trans,index=[0])

df1.to_csv('../data_hysteresis/sprott_b_white/original series/sprott_b_forward.csv')
df2.to_csv('../data_hysteresis/sprott_b_white/original series/sprott_b_reverse.csv')
trans.to_csv('../data_hysteresis/sprott_b_white/original series/sprott_b_trans.csv')

l1 = np.linspace(kl,kh,len(t))
l2 = np.linspace(kh,kl,len(t))

plt.scatter(l1,z1,s=1,c='blue')
plt.scatter(l2,z2,s=1,c='red')
plt.axvline(trans_value1,color='blue', linestyle='--')
plt.axvline(trans_value2,color='red', linestyle='--')
#plt.savefig('sleep-wake.png',dpi=600)
plt.show()