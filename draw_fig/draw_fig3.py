import pandas
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
times_font = fm.FontProperties(family='Times New Roman', style='normal')

subplt = plt.subplot(3,2,1)

df = pandas.read_csv('../results/may_fold/may_fold_bl_us.csv')

df_bl = df['bl']

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

bl_list = list(df_bl.values)

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

line_ac, = subplt.plot(bl_list, error_list_ac, color='royalblue', linewidth=1.5, label='Degenerate Fingerprinting')
subplt.fill_between(bl_list, error_list_ac_min, error_list_ac_max, color='royalblue', alpha=0.25, edgecolor='none')
scatter_ac = subplt.scatter(bl_list, error_list_ac, color='royalblue', s=30, marker='^')

line_dev, = subplt.plot(bl_list, error_list_dev, color='forestgreen', linewidth=1.5, label='DEV')
subplt.fill_between(bl_list, error_list_dev_min, error_list_dev_max, color='forestgreen', alpha=0.25, edgecolor='none')
scatter_dev = subplt.scatter(bl_list, error_list_dev, color='forestgreen', s=30, marker='s')

line_dl, = subplt.plot(bl_list, error_list_dl, color='crimson', linewidth=1.5, label='DL Algorithm')
subplt.fill_between(bl_list, error_list_dl_min, error_list_dl_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_dl = subplt.scatter(bl_list, error_list_dl, color='crimson', s=30, marker='o',zorder=5)

line_lstm, = subplt.plot(bl_list, error_list_lstm, color='blueviolet', linewidth=1.5, label='LSTM (ablation study)')
subplt.fill_between(bl_list, error_list_lstm_min, error_list_lstm_max, color='blueviolet', alpha=0.25, edgecolor='none')
scatter_lstm = subplt.scatter(bl_list, error_list_lstm, color='blueviolet', s=30, marker='D')

# 创建自定义图例
legend_ac = mlines.Line2D([], [], color='royalblue', marker='^', markersize=5, label='Degenerate Fingerprinting', linestyle='-', markeredgewidth=1.5)
legend_dev = mlines.Line2D([], [], color='forestgreen', marker='s', markersize=5, label='DEV', linestyle='-', markeredgewidth=1.5)
legend_dl = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='DL Algorithm', linestyle='-', markeredgewidth=1.5)
legend_lstm = mlines.Line2D([], [], color='blueviolet',  marker='D',markersize=5, label='LSTM (ablation study)', linestyle='-', markeredgewidth=1.5)

plt.xticks(bl_list,fontproperties=times_font)
plt.ylim(-0.2,4.2)
plt.yticks([0,2,4],fontproperties=times_font)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)

plt.ylabel('Mean relative error',font,labelpad=13)
handles = [legend_dl, legend_ac, legend_dev, legend_lstm]
labels = [h.get_label() for h in handles]

subplt.set_title('May Harvesting Fold Model (1D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'a',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=1,prop={'size':10})

subplt = plt.subplot(3,2,2)

df = pandas.read_csv('../results/food_hopf/food_hopf_kl_us.csv')

df_kl = df['kl']

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

kl_list = list(df_kl.values)

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

line_ac, = subplt.plot(kl_list, error_list_ac, color='royalblue', linewidth=1.5, label='Degenerate Fingerprinting')
subplt.fill_between(kl_list, error_list_ac_min, error_list_ac_max, color='royalblue', alpha=0.25, edgecolor='none')
scatter_ac = subplt.scatter(kl_list, error_list_ac, color='royalblue', s=30, marker='^')

line_dev, = subplt.plot(kl_list, error_list_dev, color='forestgreen', linewidth=1.5, label='DEV')
subplt.fill_between(kl_list, error_list_dev_min, error_list_dev_max, color='forestgreen', alpha=0.25, edgecolor='none')
scatter_dev = subplt.scatter(kl_list, error_list_dev, color='forestgreen', s=30, marker='s')

line_dl, = subplt.plot(kl_list, error_list_dl, color='crimson', linewidth=1.5, label='DL Algorithm')
subplt.fill_between(kl_list, error_list_dl_min, error_list_dl_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_dl = subplt.scatter(kl_list, error_list_dl, color='crimson', s=30, marker='o',zorder=5)

line_lstm, = subplt.plot(kl_list, error_list_lstm, color='blueviolet', linewidth=1.5, label='LSTM (ablation study)')
subplt.fill_between(kl_list, error_list_lstm_min, error_list_lstm_max, color='blueviolet', alpha=0.25, edgecolor='none')
scatter_lstm = subplt.scatter(kl_list, error_list_lstm, color='blueviolet', s=30, marker='D')

# 创建自定义图例
legend_ac = mlines.Line2D([], [], color='royalblue', marker='^', markersize=5, label='Degenerate Fingerprinting', linestyle='-', markeredgewidth=1.5)
legend_dev = mlines.Line2D([], [], color='forestgreen', marker='s', markersize=5, label='DEV', linestyle='-', markeredgewidth=1.5)
legend_dl = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='DL Algorithm', linestyle='-', markeredgewidth=1.5)
legend_lstm = mlines.Line2D([], [], color='blueviolet',  marker='D',markersize=5, label='LSTM (ablation study)', linestyle='-', markeredgewidth=1.5)

plt.xticks(kl_list,fontproperties=times_font)
plt.ylim(-0.2,4.2)
plt.yticks([0,2,4],fontproperties=times_font)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)
#act = plt.hlines(0,bl_list[0],bl_list[-1],linestyles='dashed',colors='black',label='Ground Truth',linewidth=1.5)

handles = [legend_dl, legend_ac, legend_dev, legend_lstm]
labels = [h.get_label() for h in handles]

subplt.set_title('Chaotic Food Chain Hopf Model (3D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'b',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=1,prop={'size':10})

subplt = plt.subplot(3,2,3)

df = pandas.read_csv('../results/cr_branch/cr_branch_al_us.csv')

df_al = df['al']

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

al_list = list(df_al.values)

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

line_ac, = subplt.plot(al_list, error_list_ac, color='royalblue', linewidth=1.5, label='Degenerate Fingerprinting')
subplt.fill_between(al_list, error_list_ac_min, error_list_ac_max, color='royalblue', alpha=0.25, edgecolor='none')
scatter_ac = subplt.scatter(al_list, error_list_ac, color='royalblue', s=30, marker='^')

line_dev, = subplt.plot(al_list, error_list_dev, color='forestgreen', linewidth=1.5, label='DEV')
subplt.fill_between(al_list, error_list_dev_min, error_list_dev_max, color='forestgreen', alpha=0.25, edgecolor='none')
scatter_dev = subplt.scatter(al_list, error_list_dev, color='forestgreen', s=30, marker='s')

line_dl, = subplt.plot(al_list, error_list_dl, color='crimson', linewidth=1.5, label='DL Algorithm')
subplt.fill_between(al_list, error_list_dl_min, error_list_dl_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_dl = subplt.scatter(al_list, error_list_dl, color='crimson', s=30, marker='o',zorder=5)

line_lstm, = subplt.plot(al_list, error_list_lstm, color='blueviolet', linewidth=1.5, label='LSTM (ablation study)')
subplt.fill_between(al_list, error_list_lstm_min, error_list_lstm_max, color='blueviolet', alpha=0.25, edgecolor='none')
scatter_lstm = subplt.scatter(al_list, error_list_lstm, color='blueviolet', s=30, marker='D')

# 创建自定义图例
legend_ac = mlines.Line2D([], [], color='royalblue', marker='^', markersize=5, label='Degenerate Fingerprinting', linestyle='-', markeredgewidth=1.5)
legend_dev = mlines.Line2D([], [], color='forestgreen', marker='s', markersize=5, label='DEV', linestyle='-', markeredgewidth=1.5)
legend_dl = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='DL Algorithm', linestyle='-', markeredgewidth=1.5)
legend_lstm = mlines.Line2D([], [], color='blueviolet',  marker='D',markersize=5, label='LSTM (ablation study)', linestyle='-', markeredgewidth=1.5)

plt.xticks(al_list,fontproperties=times_font)
plt.ylim(-0.25,5.25)
plt.yticks([0,2.5,5],fontproperties=times_font)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)
#act = plt.hlines(0,bl_list[0],bl_list[-1],linestyles='dashed',colors='black',label='Ground Truth',linewidth=1.5)

plt.ylabel('Mean relative error',font)
handles = [legend_dl, legend_ac, legend_dev, legend_lstm]
labels = [h.get_label() for h in handles]

subplt.set_title('Consumer Resource Transcritical Model (2D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'c',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=1,prop={'size':10})

subplt = plt.subplot(3,2,4)

df = pandas.read_csv('../results/global_fold/global_fold_ul_us.csv')

df_ul = df['ul']

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

ul_list = list(df_ul.values)

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

line_ac, = subplt.plot(ul_list, error_list_ac, color='royalblue', linewidth=1.5, label='BB Method')
subplt.fill_between(ul_list, error_list_ac_min, error_list_ac_max, color='royalblue', alpha=0.25, edgecolor='none')
scatter_ac = subplt.scatter(ul_list, error_list_ac, color='royalblue', s=30, marker='v')

line_dev, = subplt.plot(ul_list, error_list_dev, color='forestgreen', linewidth=1.5, label='DEV')
subplt.fill_between(ul_list, error_list_dev_min, error_list_dev_max, color='forestgreen', alpha=0.25, edgecolor='none')
scatter_dev = subplt.scatter(ul_list, error_list_dev, color='forestgreen', s=30, marker='s')

line_dl, = subplt.plot(ul_list, error_list_dl, color='crimson', linewidth=1.5, label='DL Algorithm')
subplt.fill_between(ul_list, error_list_dl_min, error_list_dl_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_dl = subplt.scatter(ul_list, error_list_dl, color='crimson', s=30, marker='o',zorder=5)

line_lstm, = subplt.plot(ul_list, error_list_lstm, color='blueviolet', linewidth=1.5, label='LSTM (ablation study)')
subplt.fill_between(ul_list, error_list_lstm_min, error_list_lstm_max, color='blueviolet', alpha=0.25, edgecolor='none')
scatter_lstm = subplt.scatter(ul_list, error_list_lstm, color='blueviolet', s=30, marker='D')

# 创建自定义图例
legend_ac = mlines.Line2D([], [], color='royalblue', marker='v', markersize=5, label='BB Method', linestyle='-', markeredgewidth=1.5)
legend_dev = mlines.Line2D([], [], color='forestgreen', marker='s', markersize=5, label='DEV', linestyle='-', markeredgewidth=1.5)
legend_dl = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='DL Algorithm', linestyle='-', markeredgewidth=1.5)
legend_lstm = mlines.Line2D([], [], color='blueviolet',  marker='D',markersize=5, label='LSTM (ablation study)', linestyle='-', markeredgewidth=1.5)

plt.xticks(ul_list,fontproperties=times_font)
plt.ylim(-0.2,4.2)
plt.yticks([0,2,4],fontproperties=times_font)
plt.gca().invert_xaxis()
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)
#act = plt.hlines(0,bl_list[0],bl_list[-1],linestyles='dashed',colors='black',label='Ground Truth',linewidth=1.5)

handles = [legend_dl, legend_ac, legend_dev, legend_lstm]
labels = [h.get_label() for h in handles]

subplt.set_title('Global Energy Balance Fold Model (1D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'d',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=1,prop={'size':10})

subplt = plt.subplot(3,2,5)

df = pandas.read_csv('../results/MPT_hopf/MPT_hopf_ul_us.csv')

df_ul = df['ul']

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

ul_list = list(df_ul.values)

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

line_ac, = subplt.plot(ul_list, error_list_ac, color='royalblue', linewidth=1.5, label='BB Method')
subplt.fill_between(ul_list, error_list_ac_min, error_list_ac_max, color='royalblue', alpha=0.25, edgecolor='none')
scatter_ac = subplt.scatter(ul_list, error_list_ac, color='royalblue', s=30, marker='v')

line_dev, = subplt.plot(ul_list, error_list_dev, color='forestgreen', linewidth=1.5, label='DEV')
subplt.fill_between(ul_list, error_list_dev_min, error_list_dev_max, color='forestgreen', alpha=0.25, edgecolor='none')
scatter_dev = subplt.scatter(ul_list, error_list_dev, color='forestgreen', s=30, marker='s')

line_dl, = subplt.plot(ul_list, error_list_dl, color='crimson', linewidth=1.5, label='DL Algorithm')
subplt.fill_between(ul_list, error_list_dl_min, error_list_dl_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_dl = subplt.scatter(ul_list, error_list_dl, color='crimson', s=30, marker='o',zorder=5)

line_lstm, = subplt.plot(ul_list, error_list_lstm, color='blueviolet', linewidth=1.5, label='LSTM (ablation study)')
subplt.fill_between(ul_list, error_list_lstm_min, error_list_lstm_max, color='blueviolet', alpha=0.25, edgecolor='none')
scatter_lstm = subplt.scatter(ul_list, error_list_lstm, color='blueviolet', s=30, marker='D')

# 创建自定义图例
legend_ac = mlines.Line2D([], [], color='royalblue', marker='v', markersize=5, label='BB Method', linestyle='-', markeredgewidth=1.5)
legend_dev = mlines.Line2D([], [], color='forestgreen', marker='s', markersize=5, label='DEV', linestyle='-', markeredgewidth=1.5)
legend_dl = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='DL Algorithm', linestyle='-', markeredgewidth=1.5)
legend_lstm = mlines.Line2D([], [], color='blueviolet',  marker='D',markersize=5, label='LSTM (ablation study)', linestyle='-', markeredgewidth=1.5)

plt.xticks(ul_list,fontproperties=times_font)
plt.ylim(-0.2,4.2)
plt.yticks([0,2,4],fontproperties=times_font)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)
#act = plt.hlines(0,bl_list[0],bl_list[-1],linestyles='dashed',colors='black',label='Ground Truth',linewidth=1.5)

plt.xlabel('Initial value of bifurcation parameter',font,labelpad=7)
plt.ylabel('Mean relative error',font,labelpad=13)
handles = [legend_dl, legend_ac, legend_dev, legend_lstm]
labels = [h.get_label() for h in handles]

subplt.set_title('Middle Pleistocene Transition Hopf Model (3D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'e',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=1,prop={'size':10})

subplt = plt.subplot(3,2,6)

df = pandas.read_csv('../results/amazon_branch/amazon_branch_pl_us.csv')

df_pl = df['pl']

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

pl_list = list(df_pl.values)

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

line_ac, = subplt.plot(pl_list, error_list_ac, color='royalblue', linewidth=1.5, label='BB Method')
subplt.fill_between(pl_list, error_list_ac_min, error_list_ac_max, color='royalblue', alpha=0.25, edgecolor='none')
scatter_ac = subplt.scatter(pl_list, error_list_ac, color='royalblue', s=30, marker='v')

line_dev, = subplt.plot(pl_list, error_list_dev, color='forestgreen', linewidth=1.5, label='DEV')
subplt.fill_between(pl_list, error_list_dev_min, error_list_dev_max, color='forestgreen', alpha=0.25, edgecolor='none')
scatter_dev = subplt.scatter(pl_list, error_list_dev, color='forestgreen', s=30, marker='s')

line_dl, = subplt.plot(pl_list, error_list_dl, color='crimson', linewidth=1.5, label='DL Algorithm')
subplt.fill_between(pl_list, error_list_dl_min, error_list_dl_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_dl = subplt.scatter(pl_list, error_list_dl, color='crimson', s=30, marker='o',zorder=5)

line_lstm, = subplt.plot(pl_list, error_list_lstm, color='blueviolet', linewidth=1.5, label='LSTM (ablation study)')
subplt.fill_between(pl_list, error_list_lstm_min, error_list_lstm_max, color='blueviolet', alpha=0.25, edgecolor='none')
scatter_lstm = subplt.scatter(pl_list, error_list_lstm, color='blueviolet', s=30, marker='D')

# 创建自定义图例
legend_ac = mlines.Line2D([], [], color='royalblue', marker='v', markersize=5, label='BB Method', linestyle='-', markeredgewidth=1.5)
legend_dev = mlines.Line2D([], [], color='forestgreen', marker='s', markersize=5, label='DEV', linestyle='-', markeredgewidth=1.5)
legend_dl = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='DL Algorithm', linestyle='-', markeredgewidth=1.5)
legend_lstm = mlines.Line2D([], [], color='blueviolet',  marker='D',markersize=5, label='LSTM (ablation study)', linestyle='-', markeredgewidth=1.5)

plt.xticks(pl_list,fontproperties=times_font)
plt.ylim(-0.2,4.2)
plt.yticks([0,2,4],fontproperties=times_font)
plt.gca().invert_xaxis()
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)
#act = plt.hlines(0,bl_list[0],bl_list[-1],linestyles='dashed',colors='black',label='Ground Truth',linewidth=1.5)

plt.xlabel('Initial value of bifurcation parameter',font,labelpad=7)
handles = [legend_dl, legend_ac, legend_dev, legend_lstm]
labels = [h.get_label() for h in handles]

subplt.set_title('Amazon Rainforest Dieback Transcritical Model (1D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'f',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=1,prop={'size':10})

plt.subplots_adjust(top=0.96, bottom=0.075, left=0.055, right=0.99, hspace=0.32, wspace=0.08)
plt.savefig('../figures/FIG3.pdf',format='pdf',dpi=600)

