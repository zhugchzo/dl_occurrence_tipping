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

plt.figure(figsize=(12,9))

font = {'family':'Times New Roman','weight':'normal','size': 18}
font_x = fm.FontProperties(family='Times New Roman', style='normal', size=12)
font_y = fm.FontProperties(family='Times New Roman', style='normal', size=15)

subplt = plt.subplot(3,2,1)

df = pandas.read_csv('../results/may_fold/may_fold_bh_nus.csv')

df_bh = df['bh']

df_dl = df['error_dl']
df_dl_min = df['min_dl']
df_dl_max = df['max_dl']

df_ac = df['error_ac']
df_ac_min = df['min_ac']
df_ac_max = df['max_ac']

df_dev = df['error_dev']
df_dev_min = df['min_dev']
df_dev_max = df['max_dev']

df_lstm = df['error_lstm']
df_lstm_min = df['min_lstm']
df_lstm_max = df['max_lstm']

bh_list = list(df_bh.values)

error_list_dl = list(df_dl.values)
error_list_dl_min = list(df_dl_min.values)
error_list_dl_max = list(df_dl_max.values)

error_list_ac = list(df_ac.values)
error_list_ac_min = list(df_ac_min.values)
error_list_ac_max = list(df_ac_max.values)

error_list_dev = list(df_dev.values)
error_list_dev_min = list(df_dev_min.values)
error_list_dev_max = list(df_dev_max.values)

error_list_lstm = list(df_lstm.values)
error_list_lstm_min = list(df_lstm_min.values)
error_list_lstm_max = list(df_lstm_max.values)

line_ac, = subplt.plot(bh_list, error_list_ac, color='royalblue', linewidth=0.5, label='Degenerate Fingerprinting')
subplt.fill_between(bh_list, error_list_ac_min, error_list_ac_max, color='royalblue', alpha=0.25, edgecolor='none')
scatter_ac = subplt.scatter(bh_list, error_list_ac, color='royalblue', s=5, marker='^')

line_dev, = subplt.plot(bh_list, error_list_dev, color='forestgreen', linewidth=0.5, label='DEV')
subplt.fill_between(bh_list, error_list_dev_min, error_list_dev_max, color='forestgreen', alpha=0.25, edgecolor='none')
scatter_dev = subplt.scatter(bh_list, error_list_dev, color='forestgreen', s=5, marker='s')

line_dl, = subplt.plot(bh_list, error_list_dl, color='crimson', linewidth=0.5, label='DL Algorithm')
subplt.fill_between(bh_list, error_list_dl_min, error_list_dl_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_dl = subplt.scatter(bh_list, error_list_dl, color='crimson', s=5, marker='o',zorder=5)

line_lstm, = subplt.plot(bh_list, error_list_lstm, color='blueviolet', linewidth=0.5, label='LSTM (ablation study)')
subplt.fill_between(bh_list, error_list_lstm_min, error_list_lstm_max, color='blueviolet', alpha=0.25, edgecolor='none')
scatter_lstm = subplt.scatter(bh_list, error_list_lstm, color='blueviolet', s=5, marker='D')

# 创建自定义图例
legend_ac = mlines.Line2D([], [], color='royalblue', marker='^', markersize=5, label='Degenerate Fingerprinting', linestyle='-', markeredgewidth=1.5)
legend_dev = mlines.Line2D([], [], color='forestgreen', marker='s', markersize=5, label='DEV', linestyle='-', markeredgewidth=1.5)
legend_dl = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='DL Algorithm', linestyle='-', markeredgewidth=1.5)
legend_lstm = mlines.Line2D([], [], color='blueviolet',  marker='D',markersize=5, label='LSTM (ablation study)', linestyle='-', markeredgewidth=1.5)

xtick = np.round(np.linspace(max(bh_list),min(bh_list),8),2)
plt.xticks(xtick,fontproperties=font_x)
plt.ylim(-0.2,4.2)
plt.yticks([0,2,4],fontproperties=font_y)
plt.gca().invert_xaxis()
ax = plt.gca()
#act = plt.hlines(0,bh_list[0],bh_list[-1],linestyles='dashed',colors='black',label='Ground Truth',linewidth=0.5)

plt.ylabel('Mean relative error',font,labelpad=13)
handles = [legend_dl, legend_ac, legend_dev, legend_lstm]
labels = [h.get_label() for h in handles]

subplt.set_title('May Harvesting Fold Model (1D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'a',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=2,prop={'size':10})

subplt = plt.subplot(3,2,2)

df = pandas.read_csv('../results/food_hopf/food_hopf_kh_nus.csv')

df_kh = df['kh']

df_dl = df['error_dl']
df_dl_min = df['min_dl']
df_dl_max = df['max_dl']

df_ac = df['error_ac']
df_ac_min = df['min_ac']
df_ac_max = df['max_ac']

df_dev = df['error_dev']
df_dev_min = df['min_dev']
df_dev_max = df['max_dev']

df_lstm = df['error_lstm']
df_lstm_min = df['min_lstm']
df_lstm_max = df['max_lstm']

kh_list = list(df_kh.values)

error_list_dl = list(df_dl.values)
error_list_dl_min = list(df_dl_min.values)
error_list_dl_max = list(df_dl_max.values)

error_list_ac = list(df_ac.values)
error_list_ac_min = list(df_ac_min.values)
error_list_ac_max = list(df_ac_max.values)

error_list_dev = list(df_dev.values)
error_list_dev_min = list(df_dev_min.values)
error_list_dev_max = list(df_dev_max.values)

error_list_lstm = list(df_lstm.values)
error_list_lstm_min = list(df_lstm_min.values)
error_list_lstm_max = list(df_lstm_max.values)

line_ac, = subplt.plot(kh_list, error_list_ac, color='royalblue', linewidth=0.5, label='Degenerate Fingerprinting')
subplt.fill_between(kh_list, error_list_ac_min, error_list_ac_max, color='royalblue', alpha=0.25, edgecolor='none')
scatter_ac = subplt.scatter(kh_list, error_list_ac, color='royalblue', s=5, marker='^')

line_dev, = subplt.plot(kh_list, error_list_dev, color='forestgreen', linewidth=0.5, label='DEV')
subplt.fill_between(kh_list, error_list_dev_min, error_list_dev_max, color='forestgreen', alpha=0.25, edgecolor='none')
scatter_dev = subplt.scatter(kh_list, error_list_dev, color='forestgreen', s=5, marker='s')

line_dl, = subplt.plot(kh_list, error_list_dl, color='crimson', linewidth=0.5, label='DL Algorithm')
subplt.fill_between(kh_list, error_list_dl_min, error_list_dl_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_dl = subplt.scatter(kh_list, error_list_dl, color='crimson', s=5, marker='o',zorder=5)

line_lstm, = subplt.plot(kh_list, error_list_lstm, color='blueviolet', linewidth=0.5, label='LSTM (ablation study)')
subplt.fill_between(kh_list, error_list_lstm_min, error_list_lstm_max, color='blueviolet', alpha=0.25, edgecolor='none')
scatter_lstm = subplt.scatter(kh_list, error_list_lstm, color='blueviolet', s=5, marker='D')

# 创建自定义图例
legend_ac = mlines.Line2D([], [], color='royalblue', marker='^', markersize=5, label='Degenerate Fingerprinting', linestyle='-', markeredgewidth=1.5)
legend_dev = mlines.Line2D([], [], color='forestgreen', marker='s', markersize=5, label='DEV', linestyle='-', markeredgewidth=1.5)
legend_dl = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='DL Algorithm', linestyle='-', markeredgewidth=1.5)
legend_lstm = mlines.Line2D([], [], color='blueviolet',  marker='D',markersize=5, label='LSTM (ablation study)', linestyle='-', markeredgewidth=1.5)

xtick = np.round(np.linspace(max(kh_list),min(kh_list),8),2)
plt.xticks(xtick,fontproperties=font_x)
plt.ylim(-0.2,4.2)
plt.yticks([0,2,4],fontproperties=font_y)
plt.gca().invert_xaxis()
ax = plt.gca()

#act = plt.hlines(0,bh_list[0],bh_list[-1],linestyles='dashed',colors='black',label='Ground Truth',linewidth=0.5)

handles = [legend_dl, legend_ac, legend_dev, legend_lstm]
labels = [h.get_label() for h in handles]

subplt.set_title('Chaotic Food Chain Hopf Model (3D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'b',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=2,prop={'size':10})

subplt = plt.subplot(3,2,3)

df = pandas.read_csv('../results/cr_branch/cr_branch_ah_nus.csv')

df_ah = df['ah']

df_dl = df['error_dl']
df_dl_min = df['min_dl']
df_dl_max = df['max_dl']

df_ac = df['error_ac']
df_ac_min = df['min_ac']
df_ac_max = df['max_ac']

df_dev = df['error_dev']
df_dev_min = df['min_dev']
df_dev_max = df['max_dev']

df_lstm = df['error_lstm']
df_lstm_min = df['min_lstm']
df_lstm_max = df['max_lstm']

ah_list = list(df_ah.values)

error_list_dl = list(df_dl.values)
error_list_dl_min = list(df_dl_min.values)
error_list_dl_max = list(df_dl_max.values)

error_list_ac = list(df_ac.values)
error_list_ac_min = list(df_ac_min.values)
error_list_ac_max = list(df_ac_max.values)

error_list_dev = list(df_dev.values)
error_list_dev_min = list(df_dev_min.values)
error_list_dev_max = list(df_dev_max.values)

error_list_lstm = list(df_lstm.values)
error_list_lstm_min = list(df_lstm_min.values)
error_list_lstm_max = list(df_lstm_max.values)

line_ac, = subplt.plot(ah_list, error_list_ac, color='royalblue', linewidth=0.5, label='Degenerate Fingerprinting')
subplt.fill_between(ah_list, error_list_ac_min, error_list_ac_max, color='royalblue', alpha=0.25, edgecolor='none')
scatter_ac = subplt.scatter(ah_list, error_list_ac, color='royalblue', s=5, marker='^')

line_dev, = subplt.plot(ah_list, error_list_dev, color='forestgreen', linewidth=0.5, label='DEV')
subplt.fill_between(ah_list, error_list_dev_min, error_list_dev_max, color='forestgreen', alpha=0.25, edgecolor='none')
scatter_dev = subplt.scatter(ah_list, error_list_dev, color='forestgreen', s=5, marker='s')

line_dl, = subplt.plot(ah_list, error_list_dl, color='crimson', linewidth=0.5, label='DL Algorithm')
subplt.fill_between(ah_list, error_list_dl_min, error_list_dl_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_dl = subplt.scatter(ah_list, error_list_dl, color='crimson', s=5, marker='o',zorder=5)

line_lstm, = subplt.plot(ah_list, error_list_lstm, color='blueviolet', linewidth=0.5, label='LSTM (ablation study)')
subplt.fill_between(ah_list, error_list_lstm_min, error_list_lstm_max, color='blueviolet', alpha=0.25, edgecolor='none')
scatter_lstm = subplt.scatter(ah_list, error_list_lstm, color='blueviolet', s=5, marker='D')

# 创建自定义图例
legend_ac = mlines.Line2D([], [], color='royalblue', marker='^', markersize=5, label='Degenerate Fingerprinting', linestyle='-', markeredgewidth=1.5)
legend_dev = mlines.Line2D([], [], color='forestgreen', marker='s', markersize=5, label='DEV', linestyle='-', markeredgewidth=1.5)
legend_dl = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='DL Algorithm', linestyle='-', markeredgewidth=1.5)
legend_lstm = mlines.Line2D([], [], color='blueviolet',  marker='D',markersize=5, label='LSTM (ablation study)', linestyle='-', markeredgewidth=1.5)

xtick = np.round(np.linspace(max(ah_list),min(ah_list),8),1)
plt.xticks(xtick,fontproperties=font_x)
plt.ylim(-0.3,6.3)
plt.yticks([0,3,6],fontproperties=font_y)
plt.gca().invert_xaxis()
ax = plt.gca()

#act = plt.hlines(0,bh_list[0],bh_list[-1],linestyles='dashed',colors='black',label='Ground Truth',linewidth=0.5)

plt.ylabel('Mean relative error',font,labelpad=13)
handles = [legend_dl, legend_ac, legend_dev, legend_lstm]
labels = [h.get_label() for h in handles]

subplt.set_title('Consumer Resource Transcritical Model (2D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'c',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=2,prop={'size':10})

subplt = plt.subplot(3,2,4)

df = pandas.read_csv('../results/global_fold/global_fold_uh_nus.csv')

df_uh = df['uh']

df_dl = df['error_dl']
df_dl_min = df['min_dl']
df_dl_max = df['max_dl']

df_ac = df['error_ac']
df_ac_min = df['min_ac']
df_ac_max = df['max_ac']

df_dev = df['error_dev']
df_dev_min = df['min_dev']
df_dev_max = df['max_dev']

df_lstm = df['error_lstm']
df_lstm_min = df['min_lstm']
df_lstm_max = df['max_lstm']

uh_list = list(df_uh.values)

error_list_dl = list(df_dl.values)
error_list_dl_min = list(df_dl_min.values)
error_list_dl_max = list(df_dl_max.values)

error_list_ac = list(df_ac.values)
error_list_ac_min = list(df_ac_min.values)
error_list_ac_max = list(df_ac_max.values)

error_list_dev = list(df_dev.values)
error_list_dev_min = list(df_dev_min.values)
error_list_dev_max = list(df_dev_max.values)

error_list_lstm = list(df_lstm.values)
error_list_lstm_min = list(df_lstm_min.values)
error_list_lstm_max = list(df_lstm_max.values)

line_ac, = subplt.plot(uh_list, error_list_ac, color='royalblue', linewidth=0.5, label='BB Method')
subplt.fill_between(uh_list, error_list_ac_min, error_list_ac_max, color='royalblue', alpha=0.25, edgecolor='none')
scatter_ac = subplt.scatter(uh_list, error_list_ac, color='royalblue', s=5, marker='v')

line_dev, = subplt.plot(uh_list, error_list_dev, color='forestgreen', linewidth=0.5, label='DEV')
subplt.fill_between(uh_list, error_list_dev_min, error_list_dev_max, color='forestgreen', alpha=0.25, edgecolor='none')
scatter_dev = subplt.scatter(uh_list, error_list_dev, color='forestgreen', s=5, marker='s')

line_dl, = subplt.plot(uh_list, error_list_dl, color='crimson', linewidth=0.5, label='DL Algorithm')
subplt.fill_between(uh_list, error_list_dl_min, error_list_dl_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_dl = subplt.scatter(uh_list, error_list_dl, color='crimson', s=5, marker='o',zorder=5)

line_lstm, = subplt.plot(uh_list, error_list_lstm, color='blueviolet', linewidth=0.5, label='LSTM (ablation study)')
subplt.fill_between(uh_list, error_list_lstm_min, error_list_lstm_max, color='blueviolet', alpha=0.25, edgecolor='none')
scatter_lstm = subplt.scatter(uh_list, error_list_lstm, color='blueviolet', s=5, marker='D')

# 创建自定义图例
legend_ac = mlines.Line2D([], [], color='royalblue', marker='v', markersize=5, label='BB Method', linestyle='-', markeredgewidth=1.5)
legend_dev = mlines.Line2D([], [], color='forestgreen', marker='s', markersize=5, label='DEV', linestyle='-', markeredgewidth=1.5)
legend_dl = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='DL Algorithm', linestyle='-', markeredgewidth=1.5)
legend_lstm = mlines.Line2D([], [], color='blueviolet',  marker='D',markersize=5, label='LSTM (ablation study)', linestyle='-', markeredgewidth=1.5)

xtick = np.round(np.linspace(max(uh_list),min(uh_list),8),2)
plt.xticks(xtick,fontproperties=font_x)
plt.ylim(-0.3,6.3)
plt.yticks([0,3,6],fontproperties=font_y)
plt.gca().invert_xaxis()
ax = plt.gca()

#act = plt.hlines(0,bh_list[0],bh_list[-1],linestyles='dashed',colors='black',label='Ground Truth',linewidth=0.5)

handles = [legend_dl, legend_ac, legend_dev, legend_lstm]
labels = [h.get_label() for h in handles]

subplt.set_title('Global Energy Balance Fold Model (1D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'d',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=2,prop={'size':10})

subplt = plt.subplot(3,2,5)

df = pandas.read_csv('../results/MPT_hopf/MPT_hopf_uh_nus.csv')

df_uh = df['uh']

df_dl = df['error_dl']
df_dl_min = df['min_dl']
df_dl_max = df['max_dl']

df_ac = df['error_ac']
df_ac_min = df['min_ac']
df_ac_max = df['max_ac']

df_dev = df['error_dev']
df_dev_min = df['min_dev']
df_dev_max = df['max_dev']

df_lstm = df['error_lstm']
df_lstm_min = df['min_lstm']
df_lstm_max = df['max_lstm']

uh_list = list(df_uh.values)

error_list_dl = list(df_dl.values)
error_list_dl_min = list(df_dl_min.values)
error_list_dl_max = list(df_dl_max.values)

error_list_ac = list(df_ac.values)
error_list_ac_min = list(df_ac_min.values)
error_list_ac_max = list(df_ac_max.values)

error_list_dev = list(df_dev.values)
error_list_dev_min = list(df_dev_min.values)
error_list_dev_max = list(df_dev_max.values)

error_list_lstm = list(df_lstm.values)
error_list_lstm_min = list(df_lstm_min.values)
error_list_lstm_max = list(df_lstm_max.values)

line_ac, = subplt.plot(uh_list, error_list_ac, color='royalblue', linewidth=0.5, label='BB Method')
subplt.fill_between(uh_list, error_list_ac_min, error_list_ac_max, color='royalblue', alpha=0.25, edgecolor='none')
scatter_ac = subplt.scatter(uh_list, error_list_ac, color='royalblue', s=5, marker='v')

line_dev, = subplt.plot(uh_list, error_list_dev, color='forestgreen', linewidth=0.5, label='DEV')
subplt.fill_between(uh_list, error_list_dev_min, error_list_dev_max, color='forestgreen', alpha=0.25, edgecolor='none')
scatter_dev = subplt.scatter(uh_list, error_list_dev, color='forestgreen', s=5, marker='s')

line_dl, = subplt.plot(uh_list, error_list_dl, color='crimson', linewidth=0.5, label='DL Algorithm')
subplt.fill_between(uh_list, error_list_dl_min, error_list_dl_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_dl = subplt.scatter(uh_list, error_list_dl, color='crimson', s=5, marker='o',zorder=5)

line_lstm, = subplt.plot(uh_list, error_list_lstm, color='blueviolet', linewidth=0.5, label='LSTM (ablation study)')
subplt.fill_between(uh_list, error_list_lstm_min, error_list_lstm_max, color='blueviolet', alpha=0.25, edgecolor='none')
scatter_lstm = subplt.scatter(uh_list, error_list_lstm, color='blueviolet', s=5, marker='D')

# 创建自定义图例
legend_ac = mlines.Line2D([], [], color='royalblue', marker='v', markersize=5, label='BB Method', linestyle='-', markeredgewidth=1.5)
legend_dev = mlines.Line2D([], [], color='forestgreen', marker='s', markersize=5, label='DEV', linestyle='-', markeredgewidth=1.5)
legend_dl = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='DL Algorithm', linestyle='-', markeredgewidth=1.5)
legend_lstm = mlines.Line2D([], [], color='blueviolet',  marker='D',markersize=5, label='LSTM (ablation study)', linestyle='-', markeredgewidth=1.5)

xtick = np.round(np.linspace(max(uh_list),min(uh_list),8),2)
plt.xticks(xtick,fontproperties=font_x)
plt.ylim(-0.2,4.2)
plt.yticks([0,2,4],fontproperties=font_y)
plt.gca().invert_xaxis()
ax = plt.gca()

#act = plt.hlines(0,bh_list[0],bh_list[-1],linestyles='dashed',colors='black',label='Ground Truth',linewidth=0.5)

plt.xlabel('Distance to tipping point',font,labelpad=7)
plt.ylabel('Mean relative error',font,labelpad=13)
handles = [legend_dl, legend_ac, legend_dev, legend_lstm]
labels = [h.get_label() for h in handles]

subplt.set_title('Middle Pleistocene Transition Hopf Model (3D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'e',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=2,prop={'size':10})

subplt = plt.subplot(3,2,6)

df = pandas.read_csv('../results/amazon_branch/amazon_branch_ph_nus.csv')

df_ph = df['ph']

df_dl = df['error_dl']
df_dl_min = df['min_dl']
df_dl_max = df['max_dl']

df_ac = df['error_ac']
df_ac_min = df['min_ac']
df_ac_max = df['max_ac']

df_dev = df['error_dev']
df_dev_min = df['min_dev']
df_dev_max = df['max_dev']

df_lstm = df['error_lstm']
df_lstm_min = df['min_lstm']
df_lstm_max = df['max_lstm']

ph_list = list(df_ph.values)

error_list_dl = list(df_dl.values)
error_list_dl_min = list(df_dl_min.values)
error_list_dl_max = list(df_dl_max.values)

error_list_ac = list(df_ac.values)
error_list_ac_min = list(df_ac_min.values)
error_list_ac_max = list(df_ac_max.values)

error_list_dev = list(df_dev.values)
error_list_dev_min = list(df_dev_min.values)
error_list_dev_max = list(df_dev_max.values)

error_list_lstm = list(df_lstm.values)
error_list_lstm_min = list(df_lstm_min.values)
error_list_lstm_max = list(df_lstm_max.values)

line_ac, = subplt.plot(ph_list, error_list_ac, color='royalblue', linewidth=0.5, label='BB Method')
subplt.fill_between(ph_list, error_list_ac_min, error_list_ac_max, color='royalblue', alpha=0.25, edgecolor='none')
scatter_ac = subplt.scatter(ph_list, error_list_ac, color='royalblue', s=5, marker='v')

line_dev, = subplt.plot(ph_list, error_list_dev, color='forestgreen', linewidth=0.5, label='DEV')
subplt.fill_between(ph_list, error_list_dev_min, error_list_dev_max, color='forestgreen', alpha=0.25, edgecolor='none')
scatter_dev = subplt.scatter(ph_list, error_list_dev, color='forestgreen', s=5, marker='s')

line_dl, = subplt.plot(ph_list, error_list_dl, color='crimson', linewidth=0.5, label='DL Algorithm')
subplt.fill_between(ph_list, error_list_dl_min, error_list_dl_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_dl = subplt.scatter(ph_list, error_list_dl, color='crimson', s=5, marker='o',zorder=5)

line_lstm, = subplt.plot(ph_list, error_list_lstm, color='blueviolet', linewidth=0.5, label='LSTM (ablation study)')
subplt.fill_between(ph_list, error_list_lstm_min, error_list_lstm_max, color='blueviolet', alpha=0.25, edgecolor='none')
scatter_lstm = subplt.scatter(ph_list, error_list_lstm, color='blueviolet', s=5, marker='D')

# 创建自定义图例
legend_ac = mlines.Line2D([], [], color='royalblue', marker='v', markersize=5, label='BB Method', linestyle='-', markeredgewidth=1.5)
legend_dev = mlines.Line2D([], [], color='forestgreen', marker='s', markersize=5, label='DEV', linestyle='-', markeredgewidth=1.5)
legend_dl = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='DL Algorithm', linestyle='-', markeredgewidth=1.5)
legend_lstm = mlines.Line2D([], [], color='blueviolet',  marker='D',markersize=5, label='LSTM (ablation study)', linestyle='-', markeredgewidth=1.5)

xtick = np.round(np.linspace(max(ph_list),min(ph_list),8),2)
plt.xticks(xtick,fontproperties=font_x)
plt.ylim(-0.2,4.2)
plt.yticks([0,2,4],fontproperties=font_y)
plt.gca().invert_xaxis()
ax = plt.gca()

#act = plt.hlines(0,bh_list[0],bh_list[-1],linestyles='dashed',colors='black',label='Ground Truth',linewidth=0.5)

plt.xlabel('Distance to tipping point',font,labelpad=7)
handles = [legend_dl, legend_ac, legend_dev, legend_lstm]
labels = [h.get_label() for h in handles]

subplt.set_title('Amazon Rainforest Dieback Transcritical Model (1D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'f',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=2,prop={'size':10})

plt.subplots_adjust(top=0.96, bottom=0.075, left=0.055, right=0.99, hspace=0.32, wspace=0.08)
plt.savefig('../figures/SFIG4.pdf',format='pdf',dpi=600)

