import pandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.lines as mlines

fig, ax = plt.subplots(figsize=(12,6))

font = {'family':'Times New Roman','weight':'normal','size': 25}
times_font = fm.FontProperties(family='Times New Roman', style='normal')

df_fold = pandas.read_csv('../results/may_fold/may_fold_bl_nus.csv')
df_foldnt = pandas.read_csv('../results/fold_network_bl_nus.csv')

df_fold_bl = df_fold['bl']
df_foldnt_bl = df_foldnt['bl']

df_dl_fold = df_fold['error_dl']
df_dl_fold_min = df_fold['min_dl']
df_dl_fold_max = df_fold['max_dl']

df_dl_foldnt = df_foldnt['error_dl']
df_dl_foldnt_min = df_foldnt['min_dl']
df_dl_foldnt_max = df_foldnt['max_dl']

fold_bl_list = list(df_fold_bl.values)
foldnt_bl_list = list(df_foldnt_bl.values)

error_list_dl_fold = list(df_dl_fold.values)
error_list_dl_fold_min = list(df_dl_fold_min.values)
error_list_dl_fold_max = list(df_dl_fold_max.values)

error_list_dl_foldnt = list(df_dl_foldnt.values)
error_list_dl_foldnt_min = list(df_dl_foldnt_min.values)
error_list_dl_foldnt_max = list(df_dl_foldnt_max.values)

ax.plot(fold_bl_list, error_list_dl_fold, color='crimson', linewidth=2)
ax.fill_between(fold_bl_list, error_list_dl_fold_min, error_list_dl_fold_max, color='crimson', alpha=0.25, edgecolor='none')
ax.scatter(fold_bl_list, error_list_dl_fold, color='crimson', s=100, marker='o')

ax.plot(fold_bl_list, error_list_dl_foldnt, color='royalblue', linewidth=2)
ax.fill_between(fold_bl_list, error_list_dl_foldnt_min, error_list_dl_foldnt_max, color='royalblue', alpha=0.25, edgecolor='none')
ax.scatter(fold_bl_list, error_list_dl_foldnt, color='royalblue', s=100, marker='^')

legend_dl_fold = mlines.Line2D([], [], color='crimson',  marker='o',markersize=8, label='Low-dimensional Fold Model', linestyle='-', markeredgewidth=2)
legend_dl_foldnt = mlines.Line2D([], [], color='royalblue',  marker='^',markersize=8, label=r'High-dimensional Fold Networked Model ($d<2m$)', linestyle='-', markeredgewidth=2)

ax.set_xticks(fold_bl_list)
ax.set_xticklabels(fold_bl_list, fontproperties=times_font)
ax.set_ylim(-0.07,1.47)
ax.set_yticks([0,0.7,1.4])
ax.set_yticklabels([0,0.7,1.4],fontproperties=times_font)
ax.tick_params(axis='x', colors='crimson', labelsize=20)
ax.tick_params(axis='y', labelsize=20)

ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(fold_bl_list)
ax2.xaxis.set_ticks_position('bottom')
ax2.xaxis.set_label_position('bottom')
ax2.spines['bottom'].set_position(('outward', 30))
ax2.spines['top'].set_visible(False)
ax2.set_xticklabels(['1.0','1.1','1.2','1.3','1.4','1.5','1.6','1.7','1.8','1.9','2.0'], fontproperties=times_font)
ax2.tick_params(axis='x', colors='royalblue', labelsize=20)

ax.set_xlabel('Initial value of bifurcation parameter',font,labelpad=40)
ax.set_ylabel('Mean relative error',font,labelpad=7)
handles = [legend_dl_fold, legend_dl_foldnt]
labels = [h.get_label() for h in handles]

plt.legend(handles=handles,labels=labels,loc=2,prop={'size':20})

plt.subplots_adjust(top=0.98, bottom=0.21, left=0.08, right=0.99)
plt.savefig('../figures/SFIG17.pdf',format='pdf',dpi=600)