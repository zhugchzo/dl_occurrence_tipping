import pandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.lines as mlines
import os

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)

# Get the directory containing the current file
current_dir = os.path.dirname(current_file_path)

# Change the working directory to the directory of the current file
os.chdir(current_dir)

numtest = 50

plt.figure(figsize=(12,9))

font = {'family':'Times New Roman','weight':'normal','size': 18}
times_font = fm.FontProperties(family='Times New Roman', style='normal')

subplt = plt.subplot(3,2,1)

df = pandas.read_csv('../results/may_fold/may_fold_bl_nus.csv')

df_bl = df['bl']
df_dl = df['error_dl']
df_dl_min = df['min_dl']
df_dl_max = df['max_dl']

df_lstm = df['error_lstm']
df_lstm_min = df['min_lstm']
df_lstm_max = df['max_lstm']

bl_list = list(df_bl.values)

error_list_dl = list(df_dl.values)
error_list_dl_min = list(df_dl_min.values)
error_list_dl_max = list(df_dl_max.values)

error_list_lstm = list(df_lstm.values)
error_list_lstm_min = list(df_lstm_min.values)
error_list_lstm_max = list(df_lstm_max.values)

line_dl, = subplt.plot(bl_list, error_list_dl, color='crimson', linewidth=1.5, label='DL Algorithm')
subplt.fill_between(bl_list, error_list_dl_min, error_list_dl_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_dl = subplt.scatter(bl_list, error_list_dl, color='crimson', s=30, marker='o')

line_lstm, = subplt.plot(bl_list, error_list_lstm, color='blueviolet', linewidth=1.5, label='LSTM (ablation study)')
subplt.fill_between(bl_list, error_list_lstm_min, error_list_lstm_max, color='blueviolet', alpha=0.25, edgecolor='none')
scatter_lstm = subplt.scatter(bl_list, error_list_lstm, color='blueviolet', s=30, marker='D')

legend_dl = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='DL Algorithm', linestyle='-', markeredgewidth=1.5)
legend_lstm = mlines.Line2D([], [], color='blueviolet',  marker='D',markersize=5, label='LSTM (ablation study)', linestyle='-', markeredgewidth=1.5)

plt.xticks(bl_list,fontproperties=times_font)
plt.ylim(-0.03,0.63)
plt.yticks([0,0.3,0.6],fontproperties=times_font)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)
#act = plt.hlines(0,bl_list[0],bl_list[-1],linestyles='dashed',colors='black',label='Ground Truth',linewidth=2)

plt.ylabel('Mean relative error',font)
handles = [legend_dl, legend_lstm]
labels = [h.get_label() for h in handles]

subplt.set_title('May Harvesting Fold Model (1D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'a',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=2,prop={'size':10})

subplt = plt.subplot(3,2,2)

df = pandas.read_csv('../results/food_hopf/food_hopf_kl_nus.csv')

df_bl = df['kl']
df_dl = df['error_dl']
df_dl_min = df['min_dl']
df_dl_max = df['max_dl']

df_lstm = df['error_lstm']
df_lstm_min = df['min_lstm']
df_lstm_max = df['max_lstm']

bl_list = list(df_bl.values)

error_list_dl = list(df_dl.values)
error_list_dl_min = list(df_dl_min.values)
error_list_dl_max = list(df_dl_max.values)

error_list_lstm = list(df_lstm.values)
error_list_lstm_min = list(df_lstm_min.values)
error_list_lstm_max = list(df_lstm_max.values)

line_dl, = subplt.plot(bl_list, error_list_dl, color='crimson', linewidth=1.5, label='DL Algorithm')
subplt.fill_between(bl_list, error_list_dl_min, error_list_dl_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_dl = subplt.scatter(bl_list, error_list_dl, color='crimson', s=30, marker='o')

line_lstm, = subplt.plot(bl_list, error_list_lstm, color='blueviolet', linewidth=1.5, label='LSTM (ablation study)')
subplt.fill_between(bl_list, error_list_lstm_min, error_list_lstm_max, color='blueviolet', alpha=0.25, edgecolor='none')
scatter_lstm = subplt.scatter(bl_list, error_list_lstm, color='blueviolet', s=30, marker='D')

legend_dl = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='DL Algorithm', linestyle='-', markeredgewidth=1.5)
legend_lstm = mlines.Line2D([], [], color='blueviolet',  marker='D',markersize=5, label='LSTM (ablation study)', linestyle='-', markeredgewidth=1.5)

plt.xticks(bl_list,fontproperties=times_font)
plt.ylim(-0.03,0.63)
plt.yticks([0,0.3,0.6],fontproperties=times_font)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)
#act = plt.hlines(0,bl_list[0],bl_list[-1],linestyles='dashed',colors='black',label='Ground Truth',linewidth=2)

handles = [legend_dl, legend_lstm]
labels = [h.get_label() for h in handles]

subplt.set_title('Chaotic Food Chain Hopf Model (3D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'b',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=1,prop={'size':10})

subplt = plt.subplot(3,2,3)

df = pandas.read_csv('../results/cr_branch/cr_branch_al_nus.csv')

df_bl = df['al']
df_dl = df['error_dl']
df_dl_min = df['min_dl']
df_dl_max = df['max_dl']

df_lstm = df['error_lstm']
df_lstm_min = df['min_lstm']
df_lstm_max = df['max_lstm']

bl_list = list(df_bl.values)

error_list_dl = list(df_dl.values)
error_list_dl_min = list(df_dl_min.values)
error_list_dl_max = list(df_dl_max.values)

error_list_lstm = list(df_lstm.values)
error_list_lstm_min = list(df_lstm_min.values)
error_list_lstm_max = list(df_lstm_max.values)

line_dl, = subplt.plot(bl_list, error_list_dl, color='crimson', linewidth=1.5, label='DL Algorithm')
subplt.fill_between(bl_list, error_list_dl_min, error_list_dl_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_dl = subplt.scatter(bl_list, error_list_dl, color='crimson', s=30, marker='o')

line_lstm, = subplt.plot(bl_list, error_list_lstm, color='blueviolet', linewidth=1.5, label='LSTM (ablation study)')
subplt.fill_between(bl_list, error_list_lstm_min, error_list_lstm_max, color='blueviolet', alpha=0.25, edgecolor='none')
scatter_lstm = subplt.scatter(bl_list, error_list_lstm, color='blueviolet', s=30, marker='D')

legend_dl = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='DL Algorithm', linestyle='-', markeredgewidth=1.5)
legend_lstm = mlines.Line2D([], [], color='blueviolet',  marker='D',markersize=5, label='LSTM (ablation study)', linestyle='-', markeredgewidth=1.5)

plt.xticks(bl_list,fontproperties=times_font)
plt.ylim(-0.12,2.52)
plt.yticks([0,1.2,2.4],fontproperties=times_font)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)
#act = plt.hlines(0,bl_list[0],bl_list[-1],linestyles='dashed',colors='black',label='Ground Truth',linewidth=2)

plt.ylabel('Mean relative error',font)
handles = [legend_dl, legend_lstm]
labels = [h.get_label() for h in handles]

subplt.set_title('Consumer Resource Transcritical Model (2D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'c',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=2,prop={'size':10})

subplt = plt.subplot(3,2,4)

df = pandas.read_csv('../results/global_fold/global_fold_ul_nus.csv')

df_bl = df['ul']
df_dl = df['error_dl']
df_dl_min = df['min_dl']
df_dl_max = df['max_dl']

df_lstm = df['error_lstm']
df_lstm_min = df['min_lstm']
df_lstm_max = df['max_lstm']

bl_list = list(df_bl.values)

error_list_dl = list(df_dl.values)
error_list_dl_min = list(df_dl_min.values)
error_list_dl_max = list(df_dl_max.values)

error_list_lstm = list(df_lstm.values)
error_list_lstm_min = list(df_lstm_min.values)
error_list_lstm_max = list(df_lstm_max.values)

line_dl, = subplt.plot(bl_list, error_list_dl, color='crimson', linewidth=1.5, label='DL Algorithm')
subplt.fill_between(bl_list, error_list_dl_min, error_list_dl_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_dl = subplt.scatter(bl_list, error_list_dl, color='crimson', s=30, marker='o')

line_lstm, = subplt.plot(bl_list, error_list_lstm, color='blueviolet', linewidth=1.5, label='LSTM (ablation study)')
subplt.fill_between(bl_list, error_list_lstm_min, error_list_lstm_max, color='blueviolet', alpha=0.25, edgecolor='none')
scatter_lstm = subplt.scatter(bl_list, error_list_lstm, color='blueviolet', s=30, marker='D')

legend_dl = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='DL Algorithm', linestyle='-', markeredgewidth=1.5)
legend_lstm = mlines.Line2D([], [], color='blueviolet',  marker='D',markersize=5, label='LSTM (ablation study)', linestyle='-', markeredgewidth=1.5)

plt.xticks(bl_list,fontproperties=times_font)
plt.ylim(-0.06,1.26)
plt.yticks([0,0.6,1.2],fontproperties=times_font)
plt.gca().invert_xaxis()
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)
#act = plt.hlines(0,bl_list[0],bl_list[-1],linestyles='dashed',colors='black',label='Ground Truth',linewidth=2)

handles = [legend_dl, legend_lstm]
labels = [h.get_label() for h in handles]

subplt.set_title('Global Energy Balance Fold Model (1D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'d',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=1,prop={'size':10})

subplt = plt.subplot(3,2,5)

df = pandas.read_csv('../results/MPT_hopf/MPT_hopf_ul_nus.csv')

df_bl = df['ul']
df_dl = df['error_dl']
df_dl_min = df['min_dl']
df_dl_max = df['max_dl']

df_lstm = df['error_lstm']
df_lstm_min = df['min_lstm']
df_lstm_max = df['max_lstm']

bl_list = list(df_bl.values)

error_list_dl = list(df_dl.values)
error_list_dl_min = list(df_dl_min.values)
error_list_dl_max = list(df_dl_max.values)

error_list_lstm = list(df_lstm.values)
error_list_lstm_min = list(df_lstm_min.values)
error_list_lstm_max = list(df_lstm_max.values)

line_dl, = subplt.plot(bl_list, error_list_dl, color='crimson', linewidth=1.5, label='DL Algorithm')
subplt.fill_between(bl_list, error_list_dl_min, error_list_dl_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_dl = subplt.scatter(bl_list, error_list_dl, color='crimson', s=30, marker='o')

line_lstm, = subplt.plot(bl_list, error_list_lstm, color='blueviolet', linewidth=1.5, label='LSTM (ablation study)')
subplt.fill_between(bl_list, error_list_lstm_min, error_list_lstm_max, color='blueviolet', alpha=0.25, edgecolor='none')
scatter_lstm = subplt.scatter(bl_list, error_list_lstm, color='blueviolet', s=30, marker='D')

legend_dl = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='DL Algorithm', linestyle='-', markeredgewidth=1.5)
legend_lstm = mlines.Line2D([], [], color='blueviolet',  marker='D',markersize=5, label='LSTM (ablation study)', linestyle='-', markeredgewidth=1.5)

plt.xticks(bl_list,fontproperties=times_font)
plt.ylim(-0.03,0.63)
plt.yticks([0,0.3,0.6],fontproperties=times_font)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)

plt.xlabel('Initial parameter',font)
plt.ylabel('Mean relative error',font)
handles = [legend_dl, legend_lstm]
labels = [h.get_label() for h in handles]

subplt.set_title('Middle Pleistocene Transition Hopf Model (3D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'e',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=2,prop={'size':10})

subplt = plt.subplot(3,2,6)

df = pandas.read_csv('../results/amazon_branch/amazon_branch_pl_nus.csv')

df_bl = df['pl']
df_dl = df['error_dl']
df_dl_min = df['min_dl']
df_dl_max = df['max_dl']

df_lstm = df['error_lstm']
df_lstm_min = df['min_lstm']
df_lstm_max = df['max_lstm']

bl_list = list(df_bl.values)

error_list_dl = list(df_dl.values)
error_list_dl_min = list(df_dl_min.values)
error_list_dl_max = list(df_dl_max.values)

error_list_lstm = list(df_lstm.values)
error_list_lstm_min = list(df_lstm_min.values)
error_list_lstm_max = list(df_lstm_max.values)

line_dl, = subplt.plot(bl_list, error_list_dl, color='crimson', linewidth=1.5, label='DL Algorithm')
subplt.fill_between(bl_list, error_list_dl_min, error_list_dl_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_dl = subplt.scatter(bl_list, error_list_dl, color='crimson', s=30, marker='o')

line_lstm, = subplt.plot(bl_list, error_list_lstm, color='blueviolet', linewidth=1.5, label='LSTM (ablation study)')
subplt.fill_between(bl_list, error_list_lstm_min, error_list_lstm_max, color='blueviolet', alpha=0.25, edgecolor='none')
scatter_lstm = subplt.scatter(bl_list, error_list_lstm, color='blueviolet', s=30, marker='D')

legend_dl = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='DL Algorithm', linestyle='-', markeredgewidth=1.5)
legend_lstm = mlines.Line2D([], [], color='blueviolet',  marker='D',markersize=5, label='LSTM (ablation study)', linestyle='-', markeredgewidth=1.5)

plt.xticks(bl_list,fontproperties=times_font)
plt.ylim(-0.04,0.84)
plt.yticks([0,0.4,0.8],fontproperties=times_font)
plt.gca().invert_xaxis()
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)

plt.xlabel('Initial parameter',font)
handles = [legend_dl, legend_lstm]
labels = [h.get_label() for h in handles]

subplt.set_title('Amazon Rainforest Dieback Transcritical Model (1D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'f',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=1,prop={'size':10})

plt.tight_layout()
plt.savefig('../figures/SFIG2.pdf',format='pdf',dpi=600)