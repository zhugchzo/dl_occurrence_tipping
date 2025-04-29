import pandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.lines as mlines

plt.figure(figsize=(12,9))

font = {'family':'Times New Roman','weight':'normal','size': 18}
times_font = fm.FontProperties(family='Times New Roman', style='normal')

subplt = plt.subplot(3,2,1)

df = pandas.read_csv('../results/data_noise/may_fold_sigma.csv')
df_sigma = df['sigma']
df_dl = df['error_dl']
df_dl_min = df['min_dl']
df_dl_max = df['max_dl']

sigma_list = list(df_sigma.values)

error_list_dl = list(df_dl.values)
error_list_dl_min = list(df_dl_min.values)
error_list_dl_max = list(df_dl_max.values)

line_dl, = subplt.plot(sigma_list, error_list_dl, color='crimson', linewidth=1.5)
subplt.fill_between(sigma_list, error_list_dl_min, error_list_dl_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_dl = subplt.scatter(sigma_list, error_list_dl, color='crimson', s=30, marker='o')

plt.xticks(sigma_list, ['0.01', '0.025', '0.05', '0.075', '0.1', '0.125', '0.15'], fontproperties=times_font)
plt.ylim(-0.007,0.107)
plt.yticks([0, 0.07, 0.14], ['0', '0.07', '0.14'], fontproperties=times_font)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)

plt.ylabel('Mean relative error',font)

subplt.set_title('May Harvesting Fold Model (1D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'a',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})

subplt = plt.subplot(3,2,2)
subplt.axis('off') 

legend_dl = mlines.Line2D([], [], color='crimson',  marker='o',markersize=10, label='DL Algorithm', linestyle='-', markeredgewidth=1.5)

handles = [legend_dl]
labels = [h.get_label() for h in handles]

subplt.legend(
    handles=handles,
    labels=labels,
    loc='center',
    prop={'size': 30},
    frameon=False
)

subplt = plt.subplot(3,2,3)

df = pandas.read_csv('../results/data_noise/cr_branch_sigma.csv')
df_sigma = df['sigma']
df_dl = df['error_dl']
df_dl_min = df['min_dl']
df_dl_max = df['max_dl']

sigma_list = list(df_sigma.values)

error_list_dl = list(df_dl.values)
error_list_dl_min = list(df_dl_min.values)
error_list_dl_max = list(df_dl_max.values)

line_dl, = subplt.plot(sigma_list, error_list_dl, color='crimson', linewidth=1.5)
subplt.fill_between(sigma_list, error_list_dl_min, error_list_dl_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_dl = subplt.scatter(sigma_list, error_list_dl, color='crimson', s=30, marker='o')

plt.xticks(sigma_list,fontproperties=times_font)
plt.ylim(-0.006,0.106)
plt.yticks([0, 0.06, 0.12], ['0', '0.06', '0.12'], fontproperties=times_font)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)

plt.ylabel('Mean relative error',font)

subplt.set_title('Consumer Resource Transcritical Model (2D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'b',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})

subplt = plt.subplot(3,2,4)

df = pandas.read_csv('../results/data_noise/global_fold_sigma.csv')
df_sigma = df['sigma']
df_dl = df['error_dl']
df_dl_min = df['min_dl']
df_dl_max = df['max_dl']

sigma_list = list(df_sigma.values)

error_list_dl = list(df_dl.values)
error_list_dl_min = list(df_dl_min.values)
error_list_dl_max = list(df_dl_max.values)

line_dl, = subplt.plot(sigma_list, error_list_dl, color='crimson', linewidth=1.5)
subplt.fill_between(sigma_list, error_list_dl_min, error_list_dl_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_dl = subplt.scatter(sigma_list, error_list_dl, color='crimson', s=30, marker='o')

plt.xticks(sigma_list,fontproperties=times_font)
plt.ylim(-0.006,0.106)
plt.yticks([0, 0.06, 0.12], ['0', '0.06', '0.12'], fontproperties=times_font)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)

subplt.set_title('Global Energy Balance Fold Model (1D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'c',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})

subplt = plt.subplot(3,2,5)

df = pandas.read_csv('../results/data_noise/MPT_hopf_sigma.csv')
df_sigma = df['sigma']
df_dl = df['error_dl']
df_dl_min = df['min_dl']
df_dl_max = df['max_dl']

sigma_list = list(df_sigma.values)

error_list_dl = list(df_dl.values)
error_list_dl_min = list(df_dl_min.values)
error_list_dl_max = list(df_dl_max.values)

line_dl, = subplt.plot(sigma_list, error_list_dl, color='crimson', linewidth=1.5)
subplt.fill_between(sigma_list, error_list_dl_min, error_list_dl_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_dl = subplt.scatter(sigma_list, error_list_dl, color='crimson', s=30, marker='o')

plt.xticks(sigma_list,fontproperties=times_font)
plt.ylim(-0.007,0.107)
plt.yticks([0, 0.07, 0.14], ['0', '0.07', '0.14'], fontproperties=times_font)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)

plt.xlabel('Amplitude of noise',font,labelpad=7)
plt.ylabel('Mean relative error',font)

subplt.set_title('Middle Pleistocene Transition Hopf Model (3D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'d',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})

subplt = plt.subplot(3,2,6)

df = pandas.read_csv('../results/data_noise/amazon_branch_sigma.csv')
df_sigma = df['sigma']
df_dl = df['error_dl']
df_dl_min = df['min_dl']
df_dl_max = df['max_dl']

sigma_list = list(df_sigma.values)

error_list_dl = list(df_dl.values)
error_list_dl_min = list(df_dl_min.values)
error_list_dl_max = list(df_dl_max.values)

line_dl, = subplt.plot(sigma_list, error_list_dl, color='crimson', linewidth=1.5)
subplt.fill_between(sigma_list, error_list_dl_min, error_list_dl_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_dl = subplt.scatter(sigma_list, error_list_dl, color='crimson', s=30, marker='o')

plt.xticks(sigma_list,fontproperties=times_font)
plt.ylim(-0.006,0.106)
plt.yticks([0, 0.06, 0.12], ['0', '0.06', '0.12'], fontproperties=times_font)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)

plt.xlabel('Amplitude of noise',font,labelpad=7)

subplt.set_title('Amazon Rainforest Dieback Transcritical Model (1D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'e',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})

plt.subplots_adjust(top=0.96, bottom=0.075, left=0.07, right=0.99, hspace=0.32, wspace=0.15)
plt.savefig('../figures/SFIG12.pdf',format='pdf',dpi=600)