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
import matplotlib.pyplot as plt
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
preds_sw = pd.read_csv('../results/sleep-wake.csv')

initial_point_swf = []
initial_point_swr = []

ground_truth_swf = 1.15282788241984
ground_truth_swr = 0.883226316248411

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

ss_swf = list(preds_sw['ssf_list'].values)
ss_swr = list(preds_sw['ssr_list'].values)
preds_dl_swf = list(preds_sw['preds_f_list'].values)
preds_dl_swr = list(preds_sw['preds_r_list'].values)

for ssf,ssr in zip(ss_swf,ss_swr):
    df_swf = pd.read_csv('../model_test/data_hysteresis/sleep-wake_white/sleep-wake_forward_{}.csv'.format(ssf))
    df_swr = pd.read_csv('../model_test/data_hysteresis/sleep-wake_white/sleep-wake_reverse_{}.csv'.format(ssr))

    initial_xf = df_swf['v'].iloc[0]
    initial_bf = df_swf['D'].iloc[0]    

    initial_xr = df_swr['v'].iloc[0]
    initial_br = df_swr['D'].iloc[0]

    initial_point_swf.append([initial_bf,initial_xf])
    initial_point_swr.append([initial_br,initial_xr])

subplt.scatter(sw_forward_parameter_bottom,sw_forward_state_bottom,c='crimson',s=0.5)
subplt.scatter(sw_reverse_parameter_bottom,sw_reverse_state_bottom,c='royalblue',s=0.5)
subplt.scatter(sw_forward_parameter_cover,sw_forward_state_cover,c='crimson',s=0.5)
subplt.scatter(sw_reverse_parameter_cover,sw_reverse_state_cover,c='royalblue',s=0.5)

subplt.axvline(ground_truth_swf,color='crimson',linestyle='--')
subplt.axvline(ground_truth_swr,color='royalblue',linestyle='--')

for i in range(len(ss_swf)):

    subplt.scatter(initial_point_swf[i][0],initial_point_swf[i][1], color='darkorange', s=24, marker='|', zorder=3)
    subplt.scatter(preds_dl_swf[i],initial_point_swf[i][1], color='darkorange', s=12, marker='o', zorder=3)
    subplt.plot([initial_point_swf[i][0],preds_dl_swf[i]],[initial_point_swf[i][1],initial_point_swf[i][1]], color='darkorange', linewidth=1, alpha=0.5)

    subplt.scatter(initial_point_swr[i][0],initial_point_swr[i][1], color='darkorange', s=24, marker='|', zorder=3)
    subplt.scatter(preds_dl_swr[i],initial_point_swr[i][1], color='darkorange', s=12, marker='o', zorder=3)
    subplt.plot([initial_point_swr[i][0],preds_dl_swr[i]],[initial_point_swr[i][1],initial_point_swr[i][1]], color='darkorange', linewidth=1, alpha=0.5)

plt.xticks([0.1, ground_truth_swr, ground_truth_swf, 1.9],['0.1', '0.883', '1.153', '1.9'],fontsize=10)
plt.xlabel(r'$D$',fontsize=14)
plt.ylabel(r'$V_v$',fontsize=14)

ax = plt.gca()

ax.annotate('', xy=(0.5, -6.15), xytext=(0.3, -6.5), arrowprops=dict(color='crimson', arrowstyle='->'))
ax.annotate('', xy=(1.6, 0.55), xytext=(1.8,0.85), arrowprops=dict(color='royalblue', arrowstyle='->'))
ax.annotate('', xy=(1.25, -2.5), xytext=(1.225, -3.5), arrowprops=dict(color='crimson', arrowstyle='->'))
ax.annotate('', xy=(0.775, -3), xytext=(0.8,-2), arrowprops=dict(color='royalblue', arrowstyle='->'))

subplt.set_title('Sleep-Wake Fold/Fold Hysteresis Loop (2D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0, 1.05,'a',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})

subplt = plt.subplot(1,2,2)

df_sbf = pd.read_csv('../model_test/data_hysteresis/sprott_b_white/original series/sprott_b_forward.csv')
df_sbr = pd.read_csv('../model_test/data_hysteresis/sprott_b_white/original series/sprott_b_reverse.csv')
preds_sb = pd.read_csv('../results/sprott_b.csv')

initial_point_sbf = []
initial_point_sbr = []

ground_truth_sbf = 4.58982161367064
ground_truth_sbr = 4.83495634709873

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

ss_sbf = list(preds_sb['ssf_list'].values)
ss_sbr = list(preds_sb['ssr_list'].values)
preds_dl_sbf = list(preds_sb['preds_f_list'].values)
preds_dl_sbr = list(preds_sb['preds_r_list'].values)

for ssf,ssr in zip(ss_sbf,ss_sbr):
    df_sbf = pd.read_csv('../model_test/data_hysteresis/sprott_b_white/sprott_b_forward_{}.csv'.format(ssf))
    df_sbr = pd.read_csv('../model_test/data_hysteresis/sprott_b_white/sprott_b_reverse_{}.csv'.format(ssr))

    initial_xf = df_sbf['z'].iloc[0]
    initial_bf = df_sbf['k'].iloc[0]    

    initial_xr = df_sbr['z'].iloc[0]
    initial_br = df_sbr['k'].iloc[0]

    initial_point_sbf.append([initial_bf,initial_xf])
    initial_point_sbr.append([initial_br,initial_xr])

subplt.scatter(sb_forward_parameter_bottom,sb_forward_state_bottom,c='crimson',s=0.5)
subplt.scatter(sb_reverse_parameter_bottom,sb_reverse_state_bottom,c='royalblue',s=0.5)
subplt.scatter(sb_forward_parameter_cover,sb_forward_state_cover,c='crimson',s=0.5)
subplt.scatter(sb_reverse_parameter_cover,sb_reverse_state_cover,c='royalblue',s=0.5)

subplt.axvline(ground_truth_sbf,color='crimson',linestyle='--')
subplt.axvline(ground_truth_sbr,color='royalblue',linestyle='--')

for i in range(len(ss_sbf)):

    subplt.scatter(initial_point_sbf[i][0],initial_point_sbf[i][1], color='darkorange', s=24, marker='|', zorder=3)
    subplt.scatter(preds_dl_sbf[i],initial_point_sbf[i][1], color='darkorange', s=12, marker='o', zorder=3)
    subplt.plot([initial_point_sbf[i][0],preds_dl_sbf[i]],[initial_point_sbf[i][1],initial_point_sbf[i][1]], color='darkorange', linewidth=1, alpha=0.5)

    subplt.scatter(initial_point_sbr[i][0],initial_point_sbr[i][1], color='darkorange', s=24, marker='|', zorder=3)
    subplt.scatter(preds_dl_sbr[i],initial_point_sbr[i][1], color='darkorange', s=12, marker='o', zorder=3)
    subplt.plot([initial_point_sbr[i][0],preds_dl_sbr[i]],[initial_point_sbr[i][1],initial_point_sbr[i][1]], color='darkorange', linewidth=1, alpha=0.5)

y_min, y_max = plt.ylim()

plt.xticks([np.pi, 1.461*np.pi, 1.539*np.pi, 2*np.pi],['$\pi$', '', '', '$2\pi$'],fontsize=10)
plt.text(1.41*3.14, y_min-0.8, '$1.461\pi$', fontsize=10, ha='center', va='center', color='black')
plt.text(1.59*3.14, y_min-0.8, '$1.539\pi$', fontsize=10, ha='center', va='center', color='black')
plt.xlabel(r'$k$',fontsize=14)
plt.ylabel(r'$z$',fontsize=14)

ax = plt.gca()

ax.annotate('', xy=(3.768, -2), xytext=(3.454, -2.5), arrowprops=dict(color='crimson', arrowstyle='->'))
ax.annotate('', xy=(5.652, -2), xytext=(5.966,-2.5), arrowprops=dict(color='royalblue', arrowstyle='->'))

subplt.set_title('Sprott B Hopf/Hopf Hysteresis Bursting (3D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0, 1.05,'b',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})

plt.subplots_adjust(top=0.9, bottom=0.12, left=0.065, right=0.99, hspace=0.32, wspace=0.15)
plt.savefig('../figures/FIG5.pdf',format='pdf',dpi=600)


