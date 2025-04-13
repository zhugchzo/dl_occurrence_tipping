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
times_font = fm.FontProperties(family='Times New Roman', style='normal')

subplt = plt.subplot(3,2,1)

df = pandas.read_csv('../results/may_fold/may_fold_bl_nus.csv')
df_bl = df['bl']
df_dl = df['error_dl']
df_dl_min = df['min_dl']
df_dl_max = df['max_dl']

df_u = pandas.read_csv('../results/may_fold/may_fold_bl_us.csv')
df_dl_u = df_u['error_dl']
df_dl_u_min = df_u['min_dl']
df_dl_u_max = df_u['max_dl']

bl_list = list(df_bl.values)

error_list_dl = list(df_dl.values)
error_list_dl_min = list(df_dl_min.values)
error_list_dl_max = list(df_dl_max.values)

error_list_dl_u = list(df_dl_u.values)
error_list_dl_u_min = list(df_dl_u_min.values)
error_list_dl_u_max = list(df_dl_u_max.values)

line_dl, = subplt.plot(bl_list, error_list_dl, color='crimson', linewidth=1.5, label='Irregularly-Sampled Data')
subplt.fill_between(bl_list, error_list_dl_min, error_list_dl_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_dl = subplt.scatter(bl_list, error_list_dl, color='crimson', s=30, marker='o')

line_dl_u, = subplt.plot(bl_list, error_list_dl_u, color='darkorange', linewidth=1.5, label='Regularly-Sampled Data')
subplt.fill_between(bl_list, error_list_dl_u_min, error_list_dl_u_max, color='darkorange', alpha=0.25, edgecolor='none')
scatter_dl_u = subplt.scatter(bl_list, error_list_dl_u, color='darkorange', s=30, marker='x')

legend_dl = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='Irregularly-Sampled Data', linestyle='-', markeredgewidth=1.5)
legend_dl_u = mlines.Line2D([], [], color='darkorange',  marker='x',markersize=5, label='Regularly-Sampled Data', linestyle='-', markeredgewidth=1.5)

plt.xticks(bl_list,fontproperties=times_font)
plt.ylim(-0.02,0.42)
plt.yticks([0,0.2,0.4],fontproperties=times_font)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)
#act = plt.hlines(0,bl_list[0],bl_list[-1],linestyles='dashed',colors='black',label='Ground Truth',linewidth=2)

plt.ylabel('Mean relative error',font)
handles = [legend_dl, legend_dl_u]
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

df_u = pandas.read_csv('../results/food_hopf/food_hopf_kl_us.csv')
df_dl_u = df_u['error_dl']
df_dl_u_min = df_u['min_dl']
df_dl_u_max = df_u['max_dl']

bl_list = list(df_bl.values)

error_list_dl = list(df_dl.values)
error_list_dl_min = list(df_dl_min.values)
error_list_dl_max = list(df_dl_max.values)

error_list_dl_u = list(df_dl_u.values)
error_list_dl_u_min = list(df_dl_u_min.values)
error_list_dl_u_max = list(df_dl_u_max.values)

line_dl, = subplt.plot(bl_list, error_list_dl, color='crimson', linewidth=1.5, label='Irregularly-Sampled Data')
subplt.fill_between(bl_list, error_list_dl_min, error_list_dl_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_dl = subplt.scatter(bl_list, error_list_dl, color='crimson', s=30, marker='o')

line_dl_u, = subplt.plot(bl_list, error_list_dl_u, color='darkorange', linewidth=1.5, label='Regularly-Sampled Data')
subplt.fill_between(bl_list, error_list_dl_u_min, error_list_dl_u_max, color='darkorange', alpha=0.25, edgecolor='none')
scatter_dl_u = subplt.scatter(bl_list, error_list_dl_u, color='darkorange', s=30, marker='x')

legend_dl = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='Irregularly-Sampled Data', linestyle='-', markeredgewidth=1.5)
legend_dl_u = mlines.Line2D([], [], color='darkorange',  marker='x',markersize=5, label='Regularly-Sampled Data', linestyle='-', markeredgewidth=1.5)

plt.xticks(bl_list,fontproperties=times_font)
plt.ylim(-0.02,0.42)
plt.yticks([0,0.2,0.4],fontproperties=times_font)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)
#act = plt.hlines(0,bl_list[0],bl_list[-1],linestyles='dashed',colors='black',label='Ground Truth',linewidth=2)

handles = [legend_dl, legend_dl_u]
labels = [h.get_label() for h in handles]

subplt.set_title('Chaotic Food Chain Hopf Model (3D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'b',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=2,prop={'size':10})

subplt = plt.subplot(3,2,3)

df = pandas.read_csv('../results/cr_branch/cr_branch_al_nus.csv')
df_bl = df['al']
df_dl = df['error_dl']
df_dl_min = df['min_dl']
df_dl_max = df['max_dl']

df_u = pandas.read_csv('../results/cr_branch/cr_branch_al_us.csv')
df_dl_u = df_u['error_dl']
df_dl_u_min = df_u['min_dl']
df_dl_u_max = df_u['max_dl']

bl_list = list(df_bl.values)

error_list_dl = list(df_dl.values)
error_list_dl_min = list(df_dl_min.values)
error_list_dl_max = list(df_dl_max.values)

error_list_dl_u = list(df_dl_u.values)
error_list_dl_u_min = list(df_dl_u_min.values)
error_list_dl_u_max = list(df_dl_u_max.values)

line_dl, = subplt.plot(bl_list, error_list_dl, color='crimson', linewidth=1.5, label='Irregularly-Sampled Data')
subplt.fill_between(bl_list, error_list_dl_min, error_list_dl_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_dl = subplt.scatter(bl_list, error_list_dl, color='crimson', s=30, marker='o')

line_dl_u, = subplt.plot(bl_list, error_list_dl_u, color='darkorange', linewidth=1.5, label='Regularly-Sampled Data')
subplt.fill_between(bl_list, error_list_dl_u_min, error_list_dl_u_max, color='darkorange', alpha=0.25, edgecolor='none')
scatter_dl_u = subplt.scatter(bl_list, error_list_dl_u, color='darkorange', s=30, marker='x')

legend_dl = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='Irregularly-Sampled Data', linestyle='-', markeredgewidth=1.5)
legend_dl_u = mlines.Line2D([], [], color='darkorange',  marker='x',markersize=5, label='Regularly-Sampled Data', linestyle='-', markeredgewidth=1.5)

plt.xticks(bl_list,fontproperties=times_font)
plt.ylim(-0.02,0.42)
plt.yticks([0,0.2,0.4],fontproperties=times_font)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)
#act = plt.hlines(0,bl_list[0],bl_list[-1],linestyles='dashed',colors='black',label='Ground Truth',linewidth=2)

plt.ylabel('Mean relative error',font)
handles = [legend_dl, legend_dl_u]
labels = [h.get_label() for h in handles]

subplt.set_title('Consumer Resource Transcritical Model (2D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'c',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=1,prop={'size':10})

subplt = plt.subplot(3,2,4)

df = pandas.read_csv('../results/global_fold/global_fold_ul_nus.csv')
df_bl = df['ul']
df_dl = df['error_dl']
df_dl_min = df['min_dl']
df_dl_max = df['max_dl']

df_u = pandas.read_csv('../results/global_fold/global_fold_ul_us.csv')
df_dl_u = df_u['error_dl']
df_dl_u_min = df_u['min_dl']
df_dl_u_max = df_u['max_dl']

bl_list = list(df_bl.values)

error_list_dl = list(df_dl.values)
error_list_dl_min = list(df_dl_min.values)
error_list_dl_max = list(df_dl_max.values)

error_list_dl_u = list(df_dl_u.values)
error_list_dl_u_min = list(df_dl_u_min.values)
error_list_dl_u_max = list(df_dl_u_max.values)

line_dl, = subplt.plot(bl_list, error_list_dl, color='crimson', linewidth=1.5, label='Irregularly-Sampled Data')
subplt.fill_between(bl_list, error_list_dl_min, error_list_dl_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_dl = subplt.scatter(bl_list, error_list_dl, color='crimson', s=30, marker='o')

line_dl_u, = subplt.plot(bl_list, error_list_dl_u, color='darkorange', linewidth=1.5, label='Regularly-Sampled Data')
subplt.fill_between(bl_list, error_list_dl_u_min, error_list_dl_u_max, color='darkorange', alpha=0.25, edgecolor='none')
scatter_dl_u = subplt.scatter(bl_list, error_list_dl_u, color='darkorange', s=30, marker='x')

legend_dl = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='Irregularly-Sampled Data', linestyle='-', markeredgewidth=1.5)
legend_dl_u = mlines.Line2D([], [], color='darkorange',  marker='x',markersize=5, label='Regularly-Sampled Data', linestyle='-', markeredgewidth=1.5)

plt.xticks(bl_list,fontproperties=times_font)
plt.ylim(-0.02,0.42)
plt.yticks([0,0.2,0.4],fontproperties=times_font)
plt.gca().invert_xaxis()
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)
#act = plt.hlines(0,bl_list[0],bl_list[-1],linestyles='dashed',colors='black',label='Ground Truth',linewidth=2)

handles = [legend_dl, legend_dl_u]
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

df_u = pandas.read_csv('../results/MPT_hopf/MPT_hopf_ul_us.csv')
df_dl_u = df_u['error_dl']
df_dl_u_min = df_u['min_dl']
df_dl_u_max = df_u['max_dl']

bl_list = list(df_bl.values)

error_list_dl = list(df_dl.values)
error_list_dl_min = list(df_dl_min.values)
error_list_dl_max = list(df_dl_max.values)

error_list_dl_u = list(df_dl_u.values)
error_list_dl_u_min = list(df_dl_u_min.values)
error_list_dl_u_max = list(df_dl_u_max.values)

line_dl, = subplt.plot(bl_list, error_list_dl, color='crimson', linewidth=1.5, label='Irregularly-Sampled Data')
subplt.fill_between(bl_list, error_list_dl_min, error_list_dl_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_dl = subplt.scatter(bl_list, error_list_dl, color='crimson', s=30, marker='o')

line_dl_u, = subplt.plot(bl_list, error_list_dl_u, color='darkorange', linewidth=1.5, label='Regularly-Sampled Data')
subplt.fill_between(bl_list, error_list_dl_u_min, error_list_dl_u_max, color='darkorange', alpha=0.25, edgecolor='none')
scatter_dl_u = subplt.scatter(bl_list, error_list_dl_u, color='darkorange', s=30, marker='x')

legend_dl = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='Irregularly-Sampled Data', linestyle='-', markeredgewidth=1.5)
legend_dl_u = mlines.Line2D([], [], color='darkorange',  marker='x',markersize=5, label='Regularly-Sampled Data', linestyle='-', markeredgewidth=1.5)

plt.xticks(bl_list,fontproperties=times_font)
plt.ylim(-0.02,0.42)
plt.yticks([0,0.2,0.4],fontproperties=times_font)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)
#act = plt.hlines(0,bl_list[0],bl_list[-1],linestyles='dashed',colors='black',label='Ground Truth',linewidth=2)

plt.xlabel('Initial value of bifurcation parameter',font,labelpad=7)
plt.ylabel('Mean relative error',font)
handles = [legend_dl, legend_dl_u]
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

df_u = pandas.read_csv('../results/amazon_branch/amazon_branch_pl_us.csv')
df_dl_u = df_u['error_dl']
df_dl_u_min = df_u['min_dl']
df_dl_u_max = df_u['max_dl']

bl_list = list(df_bl.values)

error_list_dl = list(df_dl.values)
error_list_dl_min = list(df_dl_min.values)
error_list_dl_max = list(df_dl_max.values)

error_list_dl_u = list(df_dl_u.values)
error_list_dl_u_min = list(df_dl_u_min.values)
error_list_dl_u_max = list(df_dl_u_max.values)

line_dl, = subplt.plot(bl_list, error_list_dl, color='crimson', linewidth=1.5, label='Irregularly-Sampled Data')
subplt.fill_between(bl_list, error_list_dl_min, error_list_dl_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_dl = subplt.scatter(bl_list, error_list_dl, color='crimson', s=30, marker='o')

line_dl_u, = subplt.plot(bl_list, error_list_dl_u, color='darkorange', linewidth=1.5, label='Regularly-Sampled Data')
subplt.fill_between(bl_list, error_list_dl_u_min, error_list_dl_u_max, color='darkorange', alpha=0.25, edgecolor='none')
scatter_dl_u = subplt.scatter(bl_list, error_list_dl_u, color='darkorange', s=30, marker='x')

legend_dl = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='Irregularly-Sampled Data', linestyle='-', markeredgewidth=1.5)
legend_dl_u = mlines.Line2D([], [], color='darkorange',  marker='x',markersize=5, label='Regularly-Sampled Data', linestyle='-', markeredgewidth=1.5)

plt.xticks(bl_list,fontproperties=times_font)
plt.ylim(-0.02,0.42)
plt.yticks([0,0.2,0.4],fontproperties=times_font)
plt.gca().invert_xaxis()
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)

plt.xlabel('Initial value of bifurcation parameter',font,labelpad=7)
handles = [legend_dl, legend_dl_u]
labels = [h.get_label() for h in handles]

subplt.set_title('Amazon Rainforest Dieback Transcritical Model (1D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'f',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=2,prop={'size':10})

plt.subplots_adjust(top=0.96, bottom=0.075, left=0.055, right=0.99, hspace=0.32, wspace=0.08)
plt.savefig('../figures/SFIG10.pdf',format='pdf',dpi=600)