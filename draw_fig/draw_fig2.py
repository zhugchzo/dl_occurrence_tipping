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

df = pandas.read_csv('../results/may_fold_us.csv')

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

df_dl_null = df['error_dl_null']
df_dl_null_min = df['min_dl_null']
df_dl_null_max = df['max_dl_null']

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

error_list_dl_null = list(df_dl_null.values)
error_list_dl_null_min = list(df_dl_null_min.values)
error_list_dl_null_max = list(df_dl_null_max.values)

line_ac, = subplt.plot(bl_list, error_list_ac, color='royalblue', linewidth=1.5, label='Degenerate Fingerprinting')
subplt.fill_between(bl_list, error_list_ac_min, error_list_ac_max, color='royalblue', alpha=0.25, edgecolor='none')
scatter_ac = subplt.scatter(bl_list, error_list_ac, color='royalblue', s=30, marker='^')

line_dev, = subplt.plot(bl_list, error_list_dev, color='forestgreen', linewidth=1.5, label='DEV')
subplt.fill_between(bl_list, error_list_dev_min, error_list_dev_max, color='forestgreen', alpha=0.25, edgecolor='none')
scatter_dev = subplt.scatter(bl_list, error_list_dev, color='forestgreen', s=30, marker='s')

line_dl, = subplt.plot(bl_list, error_list_dl, color='crimson', linewidth=1.5, label='DL Algorithm')
subplt.fill_between(bl_list, error_list_dl_min, error_list_dl_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_dl = subplt.scatter(bl_list, error_list_dl, color='crimson', s=30, marker='o')

line_dl_null, = subplt.plot(bl_list, error_list_dl_null, color='blueviolet', linewidth=1.5, label='Null Model')
subplt.fill_between(bl_list, error_list_dl_null_min, error_list_dl_null_max, color='blueviolet', alpha=0.25, edgecolor='none')
scatter_dl_null = subplt.scatter(bl_list, error_list_dl_null, color='blueviolet', s=30, marker='D')

# 创建自定义图例
legend_ac = mlines.Line2D([], [], color='royalblue', marker='^', markersize=5, label='Degenerate Fingerprinting', linestyle='-', markeredgewidth=1.5)
legend_dev = mlines.Line2D([], [], color='forestgreen', marker='s', markersize=5, label='DEV', linestyle='-', markeredgewidth=1.5)
legend_dl = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='DL Algorithm', linestyle='-', markeredgewidth=1.5)
legend_dl_null = mlines.Line2D([], [], color='blueviolet',  marker='D',markersize=5, label='Null Model', linestyle='-', markeredgewidth=1.5)

plt.xticks(bl_list,fontproperties=times_font)
plt.ylim(-0.3,6.3)
plt.yticks([0,3,6],fontproperties=times_font)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)
#act = plt.hlines(0,bl_list[0],bl_list[-1],linestyles='dashed',colors='black',label='Ground Truth',linewidth=1.5)

plt.ylabel('Mean relative error',font)
handles = [legend_dl, legend_ac, legend_dev, legend_dl_null]
labels = [h.get_label() for h in handles]

subplt.set_title('May Harvesting Fold Model (1D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'a',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=2,prop={'size':10})

subplt = plt.subplot(3,2,2)

df = pandas.read_csv('../results/food_hopf_us.csv')

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

df_dl_null = df['error_dl_null']
df_dl_null_min = df['min_dl_null']
df_dl_null_max = df['max_dl_null']

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

error_list_dl_null = list(df_dl_null.values)
error_list_dl_null_min = list(df_dl_null_min.values)
error_list_dl_null_max = list(df_dl_null_max.values)

line_ac, = subplt.plot(kl_list, error_list_ac, color='royalblue', linewidth=1.5, label='Degenerate Fingerprinting')
subplt.fill_between(kl_list, error_list_ac_min, error_list_ac_max, color='royalblue', alpha=0.25, edgecolor='none')
scatter_ac = subplt.scatter(kl_list, error_list_ac, color='royalblue', s=30, marker='^')

line_dev, = subplt.plot(kl_list, error_list_dev, color='forestgreen', linewidth=1.5, label='DEV')
subplt.fill_between(kl_list, error_list_dev_min, error_list_dev_max, color='forestgreen', alpha=0.25, edgecolor='none')
scatter_dev = subplt.scatter(kl_list, error_list_dev, color='forestgreen', s=30, marker='s')

line_dl, = subplt.plot(kl_list, error_list_dl, color='crimson', linewidth=1.5, label='DL Algorithm')
subplt.fill_between(kl_list, error_list_dl_min, error_list_dl_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_dl = subplt.scatter(kl_list, error_list_dl, color='crimson', s=30, marker='o')

line_dl_null, = subplt.plot(kl_list, error_list_dl_null, color='blueviolet', linewidth=1.5, label='Null Model')
subplt.fill_between(kl_list, error_list_dl_null_min, error_list_dl_null_max, color='blueviolet', alpha=0.25, edgecolor='none')
scatter_dl_null = subplt.scatter(kl_list, error_list_dl_null, color='blueviolet', s=30, marker='D')

# 创建自定义图例
legend_ac = mlines.Line2D([], [], color='royalblue', marker='^', markersize=5, label='Degenerate Fingerprinting', linestyle='-', markeredgewidth=1.5)
legend_dev = mlines.Line2D([], [], color='forestgreen', marker='s', markersize=5, label='DEV', linestyle='-', markeredgewidth=1.5)
legend_dl = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='DL Algorithm', linestyle='-', markeredgewidth=1.5)
legend_dl_null = mlines.Line2D([], [], color='blueviolet',  marker='D',markersize=5, label='Null Model', linestyle='-', markeredgewidth=1.5)

plt.xticks(kl_list,fontproperties=times_font)
plt.ylim(-0.3,6.3)
plt.yticks([0,3,6],fontproperties=times_font)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)
#act = plt.hlines(0,bl_list[0],bl_list[-1],linestyles='dashed',colors='black',label='Ground Truth',linewidth=1.5)

handles = [legend_dl, legend_ac, legend_dev, legend_dl_null]
labels = [h.get_label() for h in handles]

subplt.set_title('Chaotic Food Chain Hopf Model (3D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'b',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=2,prop={'size':10})

subplt = plt.subplot(3,2,3)

df = pandas.read_csv('../results/cr_branch_us.csv')

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

df_dl_null = df['error_dl_null']
df_dl_null_min = df['min_dl_null']
df_dl_null_max = df['max_dl_null']

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

error_list_dl_null = list(df_dl_null.values)
error_list_dl_null_min = list(df_dl_null_min.values)
error_list_dl_null_max = list(df_dl_null_max.values)

line_ac, = subplt.plot(al_list, error_list_ac, color='royalblue', linewidth=1.5, label='Degenerate Fingerprinting')
subplt.fill_between(al_list, error_list_ac_min, error_list_ac_max, color='royalblue', alpha=0.25, edgecolor='none')
scatter_ac = subplt.scatter(al_list, error_list_ac, color='royalblue', s=30, marker='^')

line_dev, = subplt.plot(al_list, error_list_dev, color='forestgreen', linewidth=1.5, label='DEV')
subplt.fill_between(al_list, error_list_dev_min, error_list_dev_max, color='forestgreen', alpha=0.25, edgecolor='none')
scatter_dev = subplt.scatter(al_list, error_list_dev, color='forestgreen', s=30, marker='s')

line_dl, = subplt.plot(al_list, error_list_dl, color='crimson', linewidth=1.5, label='DL Algorithm')
subplt.fill_between(al_list, error_list_dl_min, error_list_dl_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_dl = subplt.scatter(al_list, error_list_dl, color='crimson', s=30, marker='o')

line_dl_null, = subplt.plot(al_list, error_list_dl_null, color='blueviolet', linewidth=1.5, label='Null Model')
subplt.fill_between(al_list, error_list_dl_null_min, error_list_dl_null_max, color='blueviolet', alpha=0.25, edgecolor='none')
scatter_dl_null = subplt.scatter(al_list, error_list_dl_null, color='blueviolet', s=30, marker='D')

# 创建自定义图例
legend_ac = mlines.Line2D([], [], color='royalblue', marker='^', markersize=5, label='Degenerate Fingerprinting', linestyle='-', markeredgewidth=1.5)
legend_dev = mlines.Line2D([], [], color='forestgreen', marker='s', markersize=5, label='DEV', linestyle='-', markeredgewidth=1.5)
legend_dl = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='DL Algorithm', linestyle='-', markeredgewidth=1.5)
legend_dl_null = mlines.Line2D([], [], color='blueviolet',  marker='D',markersize=5, label='Null Model', linestyle='-', markeredgewidth=1.5)

plt.xticks(al_list,fontproperties=times_font)
plt.ylim(-1,21)
plt.yticks([0,10,20],fontproperties=times_font)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)
#act = plt.hlines(0,bl_list[0],bl_list[-1],linestyles='dashed',colors='black',label='Ground Truth',linewidth=1.5)

plt.ylabel('Mean relative error',font)
handles = [legend_dl, legend_ac, legend_dev, legend_dl_null]
labels = [h.get_label() for h in handles]

subplt.set_title('Consumer Resource Transcritical Model (2D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'c',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=1,prop={'size':10})

subplt = plt.subplot(3,2,4)

df = pandas.read_csv('../results/global_fold_us.csv')

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

df_dl_null = df['error_dl_null']
df_dl_null_min = df['min_dl_null']
df_dl_null_max = df['max_dl_null']

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

error_list_dl_null = list(df_dl_null.values)
error_list_dl_null_min = list(df_dl_null_min.values)
error_list_dl_null_max = list(df_dl_null_max.values)

line_ac, = subplt.plot(ul_list, error_list_ac, color='royalblue', linewidth=1.5, label='BB Method')
subplt.fill_between(ul_list, error_list_ac_min, error_list_ac_max, color='royalblue', alpha=0.25, edgecolor='none')
scatter_ac = subplt.scatter(ul_list, error_list_ac, color='royalblue', s=30, marker='v')

line_dev, = subplt.plot(ul_list, error_list_dev, color='forestgreen', linewidth=1.5, label='DEV')
subplt.fill_between(ul_list, error_list_dev_min, error_list_dev_max, color='forestgreen', alpha=0.25, edgecolor='none')
scatter_dev = subplt.scatter(ul_list, error_list_dev, color='forestgreen', s=30, marker='s')

line_dl, = subplt.plot(ul_list, error_list_dl, color='crimson', linewidth=1.5, label='DL Algorithm')
subplt.fill_between(ul_list, error_list_dl_min, error_list_dl_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_dl = subplt.scatter(ul_list, error_list_dl, color='crimson', s=30, marker='o')

line_dl_null, = subplt.plot(ul_list, error_list_dl_null, color='blueviolet', linewidth=1.5, label='Null Model')
subplt.fill_between(ul_list, error_list_dl_null_min, error_list_dl_null_max, color='blueviolet', alpha=0.25, edgecolor='none')
scatter_dl_null = subplt.scatter(ul_list, error_list_dl_null, color='blueviolet', s=30, marker='D')

# 创建自定义图例
legend_ac = mlines.Line2D([], [], color='royalblue', marker='v', markersize=5, label='BB Method', linestyle='-', markeredgewidth=1.5)
legend_dev = mlines.Line2D([], [], color='forestgreen', marker='s', markersize=5, label='DEV', linestyle='-', markeredgewidth=1.5)
legend_dl = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='DL Algorithm', linestyle='-', markeredgewidth=1.5)
legend_dl_null = mlines.Line2D([], [], color='blueviolet',  marker='D',markersize=5, label='Null Model', linestyle='-', markeredgewidth=1.5)

plt.xticks(ul_list,fontproperties=times_font)
plt.ylim(-0.4,8.4)
plt.yticks([0,4,8],fontproperties=times_font)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)
#act = plt.hlines(0,bl_list[0],bl_list[-1],linestyles='dashed',colors='black',label='Ground Truth',linewidth=1.5)

handles = [legend_dl, legend_ac, legend_dev, legend_dl_null]
labels = [h.get_label() for h in handles]

subplt.set_title('Global Energy Balance Fold Model (1D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'d',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=2,prop={'size':10})

subplt = plt.subplot(3,2,5)

df = pandas.read_csv('../results/MPT_hopf_us.csv')

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

df_dl_null = df['error_dl_null']
df_dl_null_min = df['min_dl_null']
df_dl_null_max = df['max_dl_null']

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

error_list_dl_null = list(df_dl_null.values)
error_list_dl_null_min = list(df_dl_null_min.values)
error_list_dl_null_max = list(df_dl_null_max.values)

line_ac, = subplt.plot(ul_list, error_list_ac, color='royalblue', linewidth=1.5, label='BB Method')
subplt.fill_between(ul_list, error_list_ac_min, error_list_ac_max, color='royalblue', alpha=0.25, edgecolor='none')
scatter_ac = subplt.scatter(ul_list, error_list_ac, color='royalblue', s=30, marker='v')

line_dev, = subplt.plot(ul_list, error_list_dev, color='forestgreen', linewidth=1.5, label='DEV')
subplt.fill_between(ul_list, error_list_dev_min, error_list_dev_max, color='forestgreen', alpha=0.25, edgecolor='none')
scatter_dev = subplt.scatter(ul_list, error_list_dev, color='forestgreen', s=30, marker='s')

line_dl, = subplt.plot(ul_list, error_list_dl, color='crimson', linewidth=1.5, label='DL Algorithm')
subplt.fill_between(ul_list, error_list_dl_min, error_list_dl_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_dl = subplt.scatter(ul_list, error_list_dl, color='crimson', s=30, marker='o')

line_dl_null, = subplt.plot(ul_list, error_list_dl_null, color='blueviolet', linewidth=1.5, label='Null Model')
subplt.fill_between(ul_list, error_list_dl_null_min, error_list_dl_null_max, color='blueviolet', alpha=0.25, edgecolor='none')
scatter_dl_null = subplt.scatter(ul_list, error_list_dl_null, color='blueviolet', s=30, marker='D')

# 创建自定义图例
legend_ac = mlines.Line2D([], [], color='royalblue', marker='v', markersize=5, label='BB Method', linestyle='-', markeredgewidth=1.5)
legend_dev = mlines.Line2D([], [], color='forestgreen', marker='s', markersize=5, label='DEV', linestyle='-', markeredgewidth=1.5)
legend_dl = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='DL Algorithm', linestyle='-', markeredgewidth=1.5)
legend_dl_null = mlines.Line2D([], [], color='blueviolet',  marker='D',markersize=5, label='Null Model', linestyle='-', markeredgewidth=1.5)

plt.xticks(ul_list,fontproperties=times_font)
plt.ylim(-0.4,8.4)
plt.yticks([0,4,8],fontproperties=times_font)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)
#act = plt.hlines(0,bl_list[0],bl_list[-1],linestyles='dashed',colors='black',label='Ground Truth',linewidth=1.5)

plt.xlabel('Initial parameter',font)
plt.ylabel('Mean relative error',font)
handles = [legend_dl, legend_ac, legend_dev, legend_dl_null]
labels = [h.get_label() for h in handles]

subplt.set_title('Middle Pleistocene Transition Hopf Model (3D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'e',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=2,prop={'size':10})

subplt = plt.subplot(3,2,6)

df = pandas.read_csv('../results/amazon_branch_us.csv')

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

df_dl_null = df['error_dl_null']
df_dl_null_min = df['min_dl_null']
df_dl_null_max = df['max_dl_null']

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

error_list_dl_null = list(df_dl_null.values)
error_list_dl_null_min = list(df_dl_null_min.values)
error_list_dl_null_max = list(df_dl_null_max.values)

line_ac, = subplt.plot(pl_list, error_list_ac, color='royalblue', linewidth=1.5, label='BB Method')
subplt.fill_between(pl_list, error_list_ac_min, error_list_ac_max, color='royalblue', alpha=0.25, edgecolor='none')
scatter_ac = subplt.scatter(pl_list, error_list_ac, color='royalblue', s=30, marker='v')

line_dev, = subplt.plot(pl_list, error_list_dev, color='forestgreen', linewidth=1.5, label='DEV')
subplt.fill_between(pl_list, error_list_dev_min, error_list_dev_max, color='forestgreen', alpha=0.25, edgecolor='none')
scatter_dev = subplt.scatter(pl_list, error_list_dev, color='forestgreen', s=30, marker='s')

line_dl, = subplt.plot(pl_list, error_list_dl, color='crimson', linewidth=1.5, label='DL Algorithm')
subplt.fill_between(pl_list, error_list_dl_min, error_list_dl_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_dl = subplt.scatter(pl_list, error_list_dl, color='crimson', s=30, marker='o')

line_dl_null, = subplt.plot(pl_list, error_list_dl_null, color='blueviolet', linewidth=1.5, label='Null Model')
subplt.fill_between(pl_list, error_list_dl_null_min, error_list_dl_null_max, color='blueviolet', alpha=0.25, edgecolor='none')
scatter_dl_null = subplt.scatter(pl_list, error_list_dl_null, color='blueviolet', s=30, marker='D')

# 创建自定义图例
legend_ac = mlines.Line2D([], [], color='royalblue', marker='v', markersize=5, label='BB Method', linestyle='-', markeredgewidth=1.5)
legend_dev = mlines.Line2D([], [], color='forestgreen', marker='s', markersize=5, label='DEV', linestyle='-', markeredgewidth=1.5)
legend_dl = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='DL Algorithm', linestyle='-', markeredgewidth=1.5)
legend_dl_null = mlines.Line2D([], [], color='blueviolet',  marker='D',markersize=5, label='Null Model', linestyle='-', markeredgewidth=1.5)

plt.xticks(pl_list,fontproperties=times_font)
plt.ylim(-0.3,6.3)
plt.yticks([0,3,6],fontproperties=times_font)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)
#act = plt.hlines(0,bl_list[0],bl_list[-1],linestyles='dashed',colors='black',label='Ground Truth',linewidth=1.5)

plt.xlabel('Initial parameter',font)
handles = [legend_dl, legend_ac, legend_dev, legend_dl_null]
labels = [h.get_label() for h in handles]

subplt.set_title('Amazon Rainforest Dieback Transcritical Model (1D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'f',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=2,prop={'size':10})

plt.tight_layout()
plt.savefig('../figures/FIG2.png',dpi=600)

