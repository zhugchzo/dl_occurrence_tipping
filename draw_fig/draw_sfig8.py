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

df_fold_dl = pandas.read_csv('../results/single_DL_model/test_DL_fold.csv')
df_hopf_dl = pandas.read_csv('../results/single_DL_model/test_DL_hopf.csv')
df_branch_dl = pandas.read_csv('../results/single_DL_model/test_DL_branch.csv')

subplt = plt.subplot(3,2,1)

df_bl = df_fold_dl['bl']

df_fold = df_fold_dl['error_fold_white']
df_fold_min = df_fold_dl['min_fold_white']
df_fold_max = df_fold_dl['max_fold_white']

df_hopf = df_hopf_dl['error_fold_white']
df_hopf_min = df_hopf_dl['min_fold_white']
df_hopf_max = df_hopf_dl['max_fold_white']

df_branch = df_branch_dl['error_fold_white']
df_branch_min = df_branch_dl['min_fold_white']
df_branch_max = df_branch_dl['max_fold_white']

bl_list = list(df_bl.values)

error_list_fold = list(df_fold.values)
error_list_fold_min = list(df_fold_min.values)
error_list_fold_max = list(df_fold_max.values)

error_list_hopf = list(df_hopf.values)
error_list_hopf_min = list(df_hopf_min.values)
error_list_hopf_max = list(df_hopf_max.values)

error_list_branch = list(df_branch.values)
error_list_branch_min = list(df_branch_min.values)
error_list_branch_max = list(df_branch_max.values)

line_fold, = subplt.plot(bl_list, error_list_fold, color='crimson', linewidth=1.5, label='Fold DL Model')
subplt.fill_between(bl_list, error_list_fold_min, error_list_fold_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_fold = subplt.scatter(bl_list, error_list_fold, color='crimson', s=30, marker='o')

line_hopf, = subplt.plot(bl_list, error_list_hopf, color='royalblue', linewidth=1.5, label='Hopf DL Model')
subplt.fill_between(bl_list, error_list_hopf_min, error_list_hopf_max, color='royalblue', alpha=0.25, edgecolor='none')
scatter_hopf = subplt.scatter(bl_list, error_list_hopf, color='royalblue', s=30, marker='^')

line_branch, = subplt.plot(bl_list, error_list_branch, color='forestgreen', linewidth=1.5, label='Transcritical DL Model')
subplt.fill_between(bl_list, error_list_branch_min, error_list_branch_max, color='forestgreen', alpha=0.25, edgecolor='none')
scatter_branch = subplt.scatter(bl_list, error_list_branch, color='forestgreen', s=30, marker='s')

legend_fold = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='Fold DL Model', linestyle='-', markeredgewidth=1.5)
legend_hopf = mlines.Line2D([], [], color='royalblue',  marker='^',markersize=5, label='Hopf DL Model', linestyle='-', markeredgewidth=1.5)
legend_branch = mlines.Line2D([], [], color='forestgreen',  marker='s',markersize=5, label='Transcritical DL Model', linestyle='-', markeredgewidth=1.5)

plt.xticks(bl_list,fontproperties=times_font)
plt.ylim(-0.03,0.63)
plt.yticks([0,0.3,0.6],fontproperties=times_font)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)

plt.ylabel('Mean relative error',font)
handles = [legend_fold,legend_hopf,legend_branch]
labels = [h.get_label() for h in handles]

subplt.set_title('May Harvesting Fold Model (1D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'a',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=1,prop={'size':10})

subplt = plt.subplot(3,2,2)

df_bl = df_fold_dl['kl']

df_fold = df_fold_dl['error_hopf_white']
df_fold_min = df_fold_dl['min_hopf_white']
df_fold_max = df_fold_dl['max_hopf_white']

df_hopf = df_hopf_dl['error_hopf_white']
df_hopf_min = df_hopf_dl['min_hopf_white']
df_hopf_max = df_hopf_dl['max_hopf_white']

df_branch = df_branch_dl['error_hopf_white']
df_branch_min = df_branch_dl['min_hopf_white']
df_branch_max = df_branch_dl['max_hopf_white']

bl_list = list(df_bl.values)

error_list_fold = list(df_fold.values)
error_list_fold_min = list(df_fold_min.values)
error_list_fold_max = list(df_fold_max.values)

error_list_hopf = list(df_hopf.values)
error_list_hopf_min = list(df_hopf_min.values)
error_list_hopf_max = list(df_hopf_max.values)

error_list_branch = list(df_branch.values)
error_list_branch_min = list(df_branch_min.values)
error_list_branch_max = list(df_branch_max.values)

line_fold, = subplt.plot(bl_list, error_list_fold, color='crimson', linewidth=1.5, label='Fold DL Model')
subplt.fill_between(bl_list, error_list_fold_min, error_list_fold_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_fold = subplt.scatter(bl_list, error_list_fold, color='crimson', s=30, marker='o')

line_hopf, = subplt.plot(bl_list, error_list_hopf, color='royalblue', linewidth=1.5, label='Hopf DL Model')
subplt.fill_between(bl_list, error_list_hopf_min, error_list_hopf_max, color='royalblue', alpha=0.25, edgecolor='none')
scatter_hopf = subplt.scatter(bl_list, error_list_hopf, color='royalblue', s=30, marker='^')

line_branch, = subplt.plot(bl_list, error_list_branch, color='forestgreen', linewidth=1.5, label='Transcritical DL Model')
subplt.fill_between(bl_list, error_list_branch_min, error_list_branch_max, color='forestgreen', alpha=0.25, edgecolor='none')
scatter_branch = subplt.scatter(bl_list, error_list_branch, color='forestgreen', s=30, marker='s')

legend_fold = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='Fold DL Model', linestyle='-', markeredgewidth=1.5)
legend_hopf = mlines.Line2D([], [], color='royalblue',  marker='^',markersize=5, label='Hopf DL Model', linestyle='-', markeredgewidth=1.5)
legend_branch = mlines.Line2D([], [], color='forestgreen',  marker='s',markersize=5, label='Transcritical DL Model', linestyle='-', markeredgewidth=1.5)

plt.xticks(bl_list,fontproperties=times_font)
plt.ylim(-0.03,0.63)
plt.yticks([0,0.3,0.6],fontproperties=times_font)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)

handles = [legend_fold,legend_hopf,legend_branch]
labels = [h.get_label() for h in handles]

subplt.set_title('Chaotic Food Chain Hopf Model (3D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'b',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=1,prop={'size':10})

subplt = plt.subplot(3,2,3)

df_bl = df_fold_dl['al']

df_fold = df_fold_dl['error_branch_white']
df_fold_min = df_fold_dl['min_branch_white']
df_fold_max = df_fold_dl['max_branch_white']

df_hopf = df_hopf_dl['error_branch_white']
df_hopf_min = df_hopf_dl['min_branch_white']
df_hopf_max = df_hopf_dl['max_branch_white']

df_branch = df_branch_dl['error_branch_white']
df_branch_min = df_branch_dl['min_branch_white']
df_branch_max = df_branch_dl['max_branch_white']

bl_list = list(df_bl.values)

error_list_fold = list(df_fold.values)
error_list_fold_min = list(df_fold_min.values)
error_list_fold_max = list(df_fold_max.values)

error_list_hopf = list(df_hopf.values)
error_list_hopf_min = list(df_hopf_min.values)
error_list_hopf_max = list(df_hopf_max.values)

error_list_branch = list(df_branch.values)
error_list_branch_min = list(df_branch_min.values)
error_list_branch_max = list(df_branch_max.values)

line_fold, = subplt.plot(bl_list, error_list_fold, color='crimson', linewidth=1.5, label='Fold DL Model')
subplt.fill_between(bl_list, error_list_fold_min, error_list_fold_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_fold = subplt.scatter(bl_list, error_list_fold, color='crimson', s=30, marker='o')

line_hopf, = subplt.plot(bl_list, error_list_hopf, color='royalblue', linewidth=1.5, label='Hopf DL Model')
subplt.fill_between(bl_list, error_list_hopf_min, error_list_hopf_max, color='royalblue', alpha=0.25, edgecolor='none')
scatter_hopf = subplt.scatter(bl_list, error_list_hopf, color='royalblue', s=30, marker='^')

line_branch, = subplt.plot(bl_list, error_list_branch, color='forestgreen', linewidth=1.5, label='Transcritical DL Model')
subplt.fill_between(bl_list, error_list_branch_min, error_list_branch_max, color='forestgreen', alpha=0.25, edgecolor='none')
scatter_branch = subplt.scatter(bl_list, error_list_branch, color='forestgreen', s=30, marker='s')

legend_fold = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='Fold DL Model', linestyle='-', markeredgewidth=1.5)
legend_hopf = mlines.Line2D([], [], color='royalblue',  marker='^',markersize=5, label='Hopf DL Model', linestyle='-', markeredgewidth=1.5)
legend_branch = mlines.Line2D([], [], color='forestgreen',  marker='s',markersize=5, label='Transcritical DL Model', linestyle='-', markeredgewidth=1.5)

plt.xticks(bl_list,fontproperties=times_font)
plt.ylim(-0.08,1.68)
plt.yticks([0,0.8,1.6],fontproperties=times_font)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)

plt.ylabel('Mean relative error',font)
handles = [legend_fold,legend_hopf,legend_branch]
labels = [h.get_label() for h in handles]

subplt.set_title('Consumer Resource Transcritical Model (2D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'c',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=1,prop={'size':10})

subplt = plt.subplot(3,2,4)

df_bl = df_fold_dl['ul1']

df_fold = df_fold_dl['error_fold_red']
df_fold_min = df_fold_dl['min_fold_red']
df_fold_max = df_fold_dl['max_fold_red']

df_hopf = df_hopf_dl['error_fold_red']
df_hopf_min = df_hopf_dl['min_fold_red']
df_hopf_max = df_hopf_dl['max_fold_red']

df_branch = df_branch_dl['error_fold_red']
df_branch_min = df_branch_dl['min_fold_red']
df_branch_max = df_branch_dl['max_fold_red']

bl_list = list(df_bl.values)

error_list_fold = list(df_fold.values)
error_list_fold_min = list(df_fold_min.values)
error_list_fold_max = list(df_fold_max.values)

error_list_hopf = list(df_hopf.values)
error_list_hopf_min = list(df_hopf_min.values)
error_list_hopf_max = list(df_hopf_max.values)

error_list_branch = list(df_branch.values)
error_list_branch_min = list(df_branch_min.values)
error_list_branch_max = list(df_branch_max.values)

line_fold, = subplt.plot(bl_list, error_list_fold, color='crimson', linewidth=1.5, label='Fold DL Model')
subplt.fill_between(bl_list, error_list_fold_min, error_list_fold_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_fold = subplt.scatter(bl_list, error_list_fold, color='crimson', s=30, marker='o')

line_hopf, = subplt.plot(bl_list, error_list_hopf, color='royalblue', linewidth=1.5, label='Hopf DL Model')
subplt.fill_between(bl_list, error_list_hopf_min, error_list_hopf_max, color='royalblue', alpha=0.25, edgecolor='none')
scatter_hopf = subplt.scatter(bl_list, error_list_hopf, color='royalblue', s=30, marker='^')

line_branch, = subplt.plot(bl_list, error_list_branch, color='forestgreen', linewidth=1.5, label='Transcritical DL Model')
subplt.fill_between(bl_list, error_list_branch_min, error_list_branch_max, color='forestgreen', alpha=0.25, edgecolor='none')
scatter_branch = subplt.scatter(bl_list, error_list_branch, color='forestgreen', s=30, marker='s')

legend_fold = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='Fold DL Model', linestyle='-', markeredgewidth=1.5)
legend_hopf = mlines.Line2D([], [], color='royalblue',  marker='^',markersize=5, label='Hopf DL Model', linestyle='-', markeredgewidth=1.5)
legend_branch = mlines.Line2D([], [], color='forestgreen',  marker='s',markersize=5, label='Transcritical DL Model', linestyle='-', markeredgewidth=1.5)

plt.xticks(bl_list,fontproperties=times_font)
plt.ylim(-0.03,0.63)
plt.yticks([0,0.3,0.6],fontproperties=times_font)
plt.gca().invert_xaxis()
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)

handles = [legend_fold,legend_hopf,legend_branch]
labels = [h.get_label() for h in handles]

subplt.set_title('Global Energy Balance Fold Model (1D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'d',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=1,prop={'size':10})

subplt = plt.subplot(3,2,5)

df_bl = df_fold_dl['ul2']

df_fold = df_fold_dl['error_hopf_red']
df_fold_min = df_fold_dl['min_hopf_red']
df_fold_max = df_fold_dl['max_hopf_red']

df_hopf = df_hopf_dl['error_hopf_red']
df_hopf_min = df_hopf_dl['min_hopf_red']
df_hopf_max = df_hopf_dl['max_hopf_red']

df_branch = df_branch_dl['error_hopf_red']
df_branch_min = df_branch_dl['min_hopf_red']
df_branch_max = df_branch_dl['max_hopf_red']

bl_list = list(df_bl.values)

error_list_fold = list(df_fold.values)
error_list_fold_min = list(df_fold_min.values)
error_list_fold_max = list(df_fold_max.values)

error_list_hopf = list(df_hopf.values)
error_list_hopf_min = list(df_hopf_min.values)
error_list_hopf_max = list(df_hopf_max.values)

error_list_branch = list(df_branch.values)
error_list_branch_min = list(df_branch_min.values)
error_list_branch_max = list(df_branch_max.values)

line_fold, = subplt.plot(bl_list, error_list_fold, color='crimson', linewidth=1.5, label='Fold DL Model')
subplt.fill_between(bl_list, error_list_fold_min, error_list_fold_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_fold = subplt.scatter(bl_list, error_list_fold, color='crimson', s=30, marker='o')

line_hopf, = subplt.plot(bl_list, error_list_hopf, color='royalblue', linewidth=1.5, label='Hopf DL Model')
subplt.fill_between(bl_list, error_list_hopf_min, error_list_hopf_max, color='royalblue', alpha=0.25, edgecolor='none')
scatter_hopf = subplt.scatter(bl_list, error_list_hopf, color='royalblue', s=30, marker='^')

line_branch, = subplt.plot(bl_list, error_list_branch, color='forestgreen', linewidth=1.5, label='Transcritical DL Model')
subplt.fill_between(bl_list, error_list_branch_min, error_list_branch_max, color='forestgreen', alpha=0.25, edgecolor='none')
scatter_branch = subplt.scatter(bl_list, error_list_branch, color='forestgreen', s=30, marker='s')

legend_fold = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='Fold DL Model', linestyle='-', markeredgewidth=1.5)
legend_hopf = mlines.Line2D([], [], color='royalblue',  marker='^',markersize=5, label='Hopf DL Model', linestyle='-', markeredgewidth=1.5)
legend_branch = mlines.Line2D([], [], color='forestgreen',  marker='s',markersize=5, label='Transcritical DL Model', linestyle='-', markeredgewidth=1.5)

plt.xticks(bl_list,fontproperties=times_font)
plt.ylim(-0.03,0.63)
plt.yticks([0,0.3,0.6],fontproperties=times_font)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)
#act = plt.hlines(0,bl_list[0],bl_list[-1],linestyles='dashed',colors='black',label='Ground Truth',linewidth=2)

plt.xlabel('Initial parameter',font)
plt.ylabel('Mean relative error',font)
handles = [legend_fold,legend_hopf,legend_branch]
labels = [h.get_label() for h in handles]

subplt.set_title('Middle Pleistocene Transition Hopf Model (3D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'e',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=1,prop={'size':10})

subplt = plt.subplot(3,2,6)

df_bl = df_fold_dl['pl']

df_fold = df_fold_dl['error_branch_red']
df_fold_min = df_fold_dl['min_branch_red']
df_fold_max = df_fold_dl['max_branch_red']

df_hopf = df_hopf_dl['error_branch_red']
df_hopf_min = df_hopf_dl['min_branch_red']
df_hopf_max = df_hopf_dl['max_branch_red']

df_branch = df_branch_dl['error_branch_red']
df_branch_min = df_branch_dl['min_branch_red']
df_branch_max = df_branch_dl['max_branch_red']

bl_list = list(df_bl.values)

error_list_fold = list(df_fold.values)
error_list_fold_min = list(df_fold_min.values)
error_list_fold_max = list(df_fold_max.values)

error_list_hopf = list(df_hopf.values)
error_list_hopf_min = list(df_hopf_min.values)
error_list_hopf_max = list(df_hopf_max.values)

error_list_branch = list(df_branch.values)
error_list_branch_min = list(df_branch_min.values)
error_list_branch_max = list(df_branch_max.values)

line_fold, = subplt.plot(bl_list, error_list_fold, color='crimson', linewidth=1.5, label='Fold DL Model')
subplt.fill_between(bl_list, error_list_fold_min, error_list_fold_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_fold = subplt.scatter(bl_list, error_list_fold, color='crimson', s=30, marker='o')

line_hopf, = subplt.plot(bl_list, error_list_hopf, color='royalblue', linewidth=1.5, label='Hopf DL Model')
subplt.fill_between(bl_list, error_list_hopf_min, error_list_hopf_max, color='royalblue', alpha=0.25, edgecolor='none')
scatter_hopf = subplt.scatter(bl_list, error_list_hopf, color='royalblue', s=30, marker='^')

line_branch, = subplt.plot(bl_list, error_list_branch, color='forestgreen', linewidth=1.5, label='Transcritical DL Model')
subplt.fill_between(bl_list, error_list_branch_min, error_list_branch_max, color='forestgreen', alpha=0.25, edgecolor='none')
scatter_branch = subplt.scatter(bl_list, error_list_branch, color='forestgreen', s=30, marker='s')

legend_fold = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='Fold DL Model', linestyle='-', markeredgewidth=1.5)
legend_hopf = mlines.Line2D([], [], color='royalblue',  marker='^',markersize=5, label='Hopf DL Model', linestyle='-', markeredgewidth=1.5)
legend_branch = mlines.Line2D([], [], color='forestgreen',  marker='s',markersize=5, label='Transcritical DL Model', linestyle='-', markeredgewidth=1.5)

plt.xticks(bl_list,fontproperties=times_font)
plt.ylim(-0.04,0.84)
plt.yticks([0,0.4,0.8],fontproperties=times_font)
plt.gca().invert_xaxis()
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)

plt.xlabel('Initial parameter',font)
handles = [legend_fold,legend_hopf,legend_branch]
labels = [h.get_label() for h in handles]

subplt.set_title('Amazon Rainforest Dieback Transcritical Model (1D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'f',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=1,prop={'size':10})

plt.tight_layout()
plt.savefig('../figures/SFIG8.pdf',format='pdf',dpi=600)