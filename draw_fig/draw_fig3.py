#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on: March 22, 2024

@author: Zhuge Chengzuo

Draw the predicted results of the sleep-wake model, fold/fold hysteresis loop
Draw the predicted results of the Sprot B model, hopf/hopf hysteresis bursting

"""

# import python libraries
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.interpolate import interp1d
import os

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)

# Get the directory containing the current file
current_dir = os.path.dirname(current_file_path)

# Change the working directory to the directory of the current file
os.chdir(current_dir)

plt.figure(figsize=(10,4))

subplt = plt.subplot(1,2,1)

df_swf = pd.read_csv('../model_test/data_hysteresis/sleep-wake_white/original series/sleep-wake_forward.csv')
df_swr = pd.read_csv('../model_test/data_hysteresis/sleep-wake_white/original series/sleep-wake_reverse.csv')
preds_sw = pd.read_csv('../model_results/sleep-wake.csv')

sw_forward_state = df_swf['v']
sw_forward_parameter = df_swf['D']
sw_reverse_state = df_swr['v']
sw_reverse_parameter = df_swr['D']

sw_forward_state = sw_forward_state[::100]
sw_forward_parameter = sw_forward_parameter[::100]
sw_reverse_state = sw_reverse_state[::100]
sw_reverse_parameter = sw_reverse_parameter[::100]

sw_forward_state = sw_forward_state.values
sw_forward_parameter = sw_forward_parameter.values
sw_reverse_state = sw_reverse_state.values
sw_reverse_parameter = sw_reverse_parameter.values

sw_forward_parameter_cover = sw_forward_parameter[sw_forward_parameter < 1]
sw_forward_parameter_bottom = sw_forward_parameter[sw_forward_parameter >= 1]
sw_reverse_parameter_cover = sw_reverse_parameter[sw_reverse_parameter > 1]
sw_reverse_parameter_bottom = sw_reverse_parameter[sw_reverse_parameter <= 1]

sw_forward_state_cover = sw_forward_state[:len(sw_forward_parameter_cover)]
sw_forward_state_bottom = sw_forward_state[len(sw_forward_parameter_cover):]
sw_reverse_state_cover = sw_reverse_state[:len(sw_reverse_parameter_cover)]
sw_reverse_state_bottom = sw_reverse_state[len(sw_reverse_parameter_cover):]

swf_pred = preds_sw['preds_f'].item()
swr_pred = preds_sw['preds_r'].item()

swf_s = preds_sw['f_start'].item()
swf_o = preds_sw['f_over'].item()
swr_s = preds_sw['r_start'].item()
swr_o = preds_sw['r_over'].item()

subplt.scatter(sw_forward_parameter_bottom,sw_forward_state_bottom,c='crimson',s=0.5)
subplt.scatter(sw_reverse_parameter_bottom,sw_reverse_state_bottom,c='royalblue',s=0.5)
subplt.scatter(sw_forward_parameter_cover,sw_forward_state_cover,c='crimson',s=0.5)
subplt.scatter(sw_reverse_parameter_cover,sw_reverse_state_cover,c='royalblue',s=0.5)

subplt.axvline(swf_pred,color='crimson',linestyle='--')
subplt.axvline(swr_pred,color='royalblue',linestyle='--')

for i in np.linspace(swf_s,swf_o,500):
    plt.axvline(i,color='silver',alpha=0.02)
for i in np.linspace(swr_s,swr_o,500):
    plt.axvline(i,color='silver',alpha=0.02)

plt.xticks([swf_s,swf_o,swr_s,swr_o])
plt.xlabel(r'$D$',fontsize=14)
plt.ylabel(r'$V_v$',fontsize=14)

ax = plt.gca()

ax.annotate('', xy=(swf_o, -7.8), xytext=(swf_s, -7.8), arrowprops=dict(color='black', arrowstyle='->'))
ax.annotate('', xy=(swr_o, -7.8), xytext=(swr_s, -7.8), arrowprops=dict(color='black', arrowstyle='->'))

ax.annotate('', xy=(0.5, -7.15), xytext=(0.3, -7.5), arrowprops=dict(color='crimson', arrowstyle='->'))
ax.annotate('', xy=(1.6, 1.55), xytext=(1.8,1.85), arrowprops=dict(color='royalblue', arrowstyle='->'))
ax.annotate('', xy=(1.25, -2.5), xytext=(1.225, -3.5), arrowprops=dict(color='crimson', arrowstyle='->'))
ax.annotate('', xy=(0.775, -3), xytext=(0.8,-2), arrowprops=dict(color='royalblue', arrowstyle='->'))

subplt.set_title('Sleep-Wake Fold/Fold Hysteresis Loop (2D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0, 1.05,'a',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})

subplt = plt.subplot(1,2,2)

df_sbf = pd.read_csv('../model_test/data_hysteresis/sprott_b_white/original series/sprott_b_forward.csv')
df_sbr = pd.read_csv('../model_test/data_hysteresis/sprott_b_white/original series/sprott_b_reverse.csv')
preds_sb = pd.read_csv('../model_results/sprott_b.csv')

sb_forward_state = df_sbf['z']
sb_forward_parameter = df_sbf['k']
sb_reverse_state = df_sbr['z']
sb_reverse_parameter = df_sbr['k']

sb_forward_state = sb_forward_state[::10]
sb_forward_parameter = sb_forward_parameter[::10]
sb_reverse_state = sb_reverse_state[::10]
sb_reverse_parameter = sb_reverse_parameter[::10]

sb_forward_state = sb_forward_state.values
sb_forward_parameter = sb_forward_parameter.values
sb_reverse_state = sb_reverse_state.values
sb_reverse_parameter = sb_reverse_parameter.values

sb_forward_parameter_cover = sb_forward_parameter[sb_forward_parameter < 4.7]
sb_forward_parameter_bottom = sb_forward_parameter[sb_forward_parameter >= 4.7]
sb_reverse_parameter_cover = sb_reverse_parameter[sb_reverse_parameter > 4.7]
sb_reverse_parameter_bottom = sb_reverse_parameter[sb_reverse_parameter <= 4.7]

sb_forward_state_cover = sb_forward_state[:len(sb_forward_parameter_cover)]
sb_forward_state_bottom = sb_forward_state[len(sb_forward_parameter_cover):]
sb_reverse_state_cover = sb_reverse_state[:len(sb_reverse_parameter_cover)]
sb_reverse_state_bottom = sb_reverse_state[len(sb_reverse_parameter_cover):]

sbf_pred = preds_sb['preds_f'].item()
sbr_pred = preds_sb['preds_r'].item()

sbf_s = preds_sb['f_start'].item()
sbf_o = preds_sb['f_over'].item()
sbr_s = preds_sb['r_start'].item()
sbr_o = preds_sb['r_over'].item()

subplt.scatter(sb_forward_parameter_bottom,sb_forward_state_bottom,c='crimson',s=0.5)
subplt.scatter(sb_reverse_parameter_bottom,sb_reverse_state_bottom,c='royalblue',s=0.5)
subplt.scatter(sb_forward_parameter_cover,sb_forward_state_cover,c='crimson',s=0.5)
subplt.scatter(sb_reverse_parameter_cover,sb_reverse_state_cover,c='royalblue',s=0.5)

subplt.axvline(sbf_pred,color='crimson',linestyle='--')
subplt.axvline(sbr_pred,color='royalblue',linestyle='--')

for i in np.linspace(sbf_s,sbf_o,500):
    subplt.axvline(i,color='silver',alpha=0.02)
for i in np.linspace(sbr_s,sbr_o,500):
    subplt.axvline(i,color='silver',alpha=0.02)

plt.xticks([np.pi, 1.3*np.pi, 1.7*np.pi, 2*np.pi],['$\pi$', '$1.3\pi$', '$1.7\pi$', '$2\pi$'])
plt.xlabel(r'$k$',fontsize=14)
plt.ylabel(r'$z$',fontsize=14)

ax = plt.gca()

ax.annotate('', xy=(sbf_o, -7.1), xytext=(sbf_s, -7.1), arrowprops=dict(facecolor='black', arrowstyle='->'))
ax.annotate('', xy=(sbr_o, -7.1), xytext=(sbr_s, -7.1), arrowprops=dict(facecolor='black', arrowstyle='->'))

ax.annotate('', xy=(3.768, -2), xytext=(3.454, -2.5), arrowprops=dict(color='crimson', arrowstyle='->'))
ax.annotate('', xy=(5.652, -2), xytext=(5.966,-2.5), arrowprops=dict(color='royalblue', arrowstyle='->'))

subplt.set_title('Sprott B Hopf/Hopf Hysteresis Bursting (3D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0, 1.05,'b',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})

plt.tight_layout()
plt.savefig('../figures/FIG3.png',dpi=600)


