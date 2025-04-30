import pandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.lines as mlines

plt.figure(figsize=(12,9))

font = {'family':'Times New Roman','weight':'normal','size': 18}
times_font = fm.FontProperties(family='Times New Roman', style='normal')

subplt = plt.subplot(3,2,1)

df = pandas.read_csv('../results/informer/may_fold_bl_nus.csv')
df_bl_nus = df['bl']
df_dl = df['error_dl']
df_dl_min = df['min_dl']
df_dl_max = df['max_dl']

dr_list = list(df_bl_nus.values)

error_list_dl = list(df_dl.values)
error_list_dl_min = list(df_dl_min.values)
error_list_dl_max = list(df_dl_max.values)

line_dl, = subplt.plot(dr_list, error_list_dl, color='crimson', linewidth=1.5)
subplt.fill_between(dr_list, error_list_dl_min, error_list_dl_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_dl = subplt.scatter(dr_list, error_list_dl, color='crimson', s=30, marker='o')

legend_dl = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='Informer', linestyle='-', markeredgewidth=1.5)

plt.xticks(dr_list,fontproperties=times_font)
plt.ylim(-0.02,0.42)
plt.yticks([0,0.2,0.4],fontproperties=times_font)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)

plt.ylabel('Mean relative error',font)
handles = [legend_dl]
labels = [h.get_label() for h in handles]

subplt.set_title('May Harvesting Fold Model (1D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'a',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=2,prop={'size':15},frameon=False)

subplt = plt.subplot(3,2,2)

df = pandas.read_csv('../results/informer/food_hopf_bl_nus.csv')
df_bl_nus = df['kl']
df_dl = df['error_dl']
df_dl_min = df['min_dl']
df_dl_max = df['max_dl']

dr_list = list(df_bl_nus.values)

error_list_dl = list(df_dl.values)
error_list_dl_min = list(df_dl_min.values)
error_list_dl_max = list(df_dl_max.values)

line_dl, = subplt.plot(dr_list, error_list_dl, color='crimson', linewidth=1.5)
subplt.fill_between(dr_list, error_list_dl_min, error_list_dl_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_dl = subplt.scatter(dr_list, error_list_dl, color='crimson', s=30, marker='o')

legend_dl = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='Informer', linestyle='-', markeredgewidth=1.5)

plt.xticks(dr_list,fontproperties=times_font)
plt.ylim(-0.02,0.42)
plt.yticks([0,0.2,0.4],fontproperties=times_font)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)

handles = [legend_dl]
labels = [h.get_label() for h in handles]

subplt.set_title('Chaotic Food Chain Hopf Model (3D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'b',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=1,prop={'size':15},frameon=False)

subplt = plt.subplot(3,2,3)

df = pandas.read_csv('../results/informer/cr_branch_bl_nus.csv')
df_bl_nus = df['al']
df_dl = df['error_dl']
df_dl_min = df['min_dl']
df_dl_max = df['max_dl']

dr_list = list(df_bl_nus.values)

error_list_dl = list(df_dl.values)
error_list_dl_min = list(df_dl_min.values)
error_list_dl_max = list(df_dl_max.values)

line_dl, = subplt.plot(dr_list, error_list_dl, color='crimson', linewidth=1.5)
subplt.fill_between(dr_list, error_list_dl_min, error_list_dl_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_dl = subplt.scatter(dr_list, error_list_dl, color='crimson', s=30, marker='o')

legend_dl = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='Informer', linestyle='-', markeredgewidth=1.5)

plt.xticks(dr_list,fontproperties=times_font)
plt.ylim(-0.02,0.42)
plt.yticks([0,0.2,0.4],fontproperties=times_font)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)

plt.ylabel('Mean relative error',font)
handles = [legend_dl]
labels = [h.get_label() for h in handles]

subplt.set_title('Consumer Resource Transcritical Model (2D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'c',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=2,prop={'size':15},frameon=False)

subplt = plt.subplot(3,2,4)

df = pandas.read_csv('../results/informer/global_fold_bl_nus.csv')
df_bl_nus = df['ul']
df_dl = df['error_dl']
df_dl_min = df['min_dl']
df_dl_max = df['max_dl']

dr_list = list(df_bl_nus.values)

error_list_dl = list(df_dl.values)
error_list_dl_min = list(df_dl_min.values)
error_list_dl_max = list(df_dl_max.values)

line_dl, = subplt.plot(dr_list, error_list_dl, color='crimson', linewidth=1.5)
subplt.fill_between(dr_list, error_list_dl_min, error_list_dl_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_dl = subplt.scatter(dr_list, error_list_dl, color='crimson', s=30, marker='o')

legend_dl = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='Informer', linestyle='-', markeredgewidth=1.5)

plt.xticks(dr_list,fontproperties=times_font)
plt.ylim(-0.03,0.63)
plt.yticks([0,0.3,0.6],fontproperties=times_font)
plt.gca().invert_xaxis()
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)

handles = [legend_dl]
labels = [h.get_label() for h in handles]

subplt.set_title('Global Energy Balance Fold Model (1D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'d',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=2,prop={'size':15},frameon=False)

subplt = plt.subplot(3,2,5)

df = pandas.read_csv('../results/informer/MPT_hopf_bl_nus.csv')
df_bl_nus = df['ul']
df_dl = df['error_dl']
df_dl_min = df['min_dl']
df_dl_max = df['max_dl']

dr_list = list(df_bl_nus.values)

error_list_dl = list(df_dl.values)
error_list_dl_min = list(df_dl_min.values)
error_list_dl_max = list(df_dl_max.values)

line_dl, = subplt.plot(dr_list, error_list_dl, color='crimson', linewidth=1.5)
subplt.fill_between(dr_list, error_list_dl_min, error_list_dl_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_dl = subplt.scatter(dr_list, error_list_dl, color='crimson', s=30, marker='o')

legend_dl = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='Informer', linestyle='-', markeredgewidth=1.5)

plt.xticks(dr_list,fontproperties=times_font)
plt.ylim(-0.02,0.42)
plt.yticks([0,0.2,0.4],fontproperties=times_font)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)

plt.xlabel('Initial value of bifurcation parameter',font,labelpad=7)
plt.ylabel('Mean relative error',font)
handles = [legend_dl]
labels = [h.get_label() for h in handles]

subplt.set_title('Middle Pleistocene Transition Hopf Model (3D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'e',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=2,prop={'size':15},frameon=False)

subplt = plt.subplot(3,2,6)

df = pandas.read_csv('../results/informer/amazon_branch_bl_nus.csv')
df_bl_nus = df['pl']
df_dl = df['error_dl']
df_dl_min = df['min_dl']
df_dl_max = df['max_dl']

dr_list = list(df_bl_nus.values)

error_list_dl = list(df_dl.values)
error_list_dl_min = list(df_dl_min.values)
error_list_dl_max = list(df_dl_max.values)

line_dl, = subplt.plot(dr_list, error_list_dl, color='crimson', linewidth=1.5)
subplt.fill_between(dr_list, error_list_dl_min, error_list_dl_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_dl = subplt.scatter(dr_list, error_list_dl, color='crimson', s=30, marker='o')

legend_dl = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='Informer', linestyle='-', markeredgewidth=1.5)

plt.xticks(dr_list,fontproperties=times_font)
plt.ylim(-0.03,0.63)
plt.yticks([0,0.3,0.6],fontproperties=times_font)
plt.gca().invert_xaxis()
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)

plt.xlabel('Initial value of bifurcation parameter',font,labelpad=7)
handles = [legend_dl]
labels = [h.get_label() for h in handles]

subplt.set_title('Amazon Rainforest Dieback Transcritical Model (1D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'f',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=1,prop={'size':15},frameon=False)

plt.subplots_adjust(top=0.96, bottom=0.075, left=0.055, right=0.99, hspace=0.32, wspace=0.08)
plt.savefig('../figures/SFIG18.pdf',format='pdf',dpi=600)