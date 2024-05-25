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

df = pandas.read_csv('../results/may_fold_nus.csv')

df_bl = df['bl']

df_dl = df['error_dl']
df_dl_min = df['min_dl']
df_dl_max = df['max_dl']

df_dl_combined = df['error_dl_combined']
df_dl_combined_min = df['min_dl_combined']
df_dl_combined_max = df['max_dl_combined']

df_dl_null = df['error_dl_null']
df_dl_null_min = df['min_dl_null']
df_dl_null_max = df['max_dl_null']

bl_list = list(df_bl.values)

error_list_dl = list(df_dl.values)
error_list_dl_min = list(df_dl_min.values)
error_list_dl_max = list(df_dl_max.values)

error_list_dl_combined = list(df_dl_combined.values)
error_list_dl_combined_min = list(df_dl_combined_min.values)
error_list_dl_combined_max = list(df_dl_combined_max.values)

error_list_dl_null = list(df_dl_null.values)
error_list_dl_null_min = list(df_dl_null_min.values)
error_list_dl_null_max = list(df_dl_null_max.values)

line_dl, = subplt.plot(bl_list, error_list_dl, color='crimson', linewidth=1.5, label='DL Algorithm')
subplt.fill_between(bl_list, error_list_dl_min, error_list_dl_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_dl = subplt.scatter(bl_list, error_list_dl, color='crimson', s=30, marker='o')

line_dl_combined, = subplt.plot(bl_list, error_list_dl_combined, color='dimgray', linewidth=1.5, label='Combined Model')
subplt.fill_between(bl_list, error_list_dl_combined_min, error_list_dl_combined_max, color='dimgray', alpha=0.25, edgecolor='none')
scatter_dl_combined = subplt.scatter(bl_list, error_list_dl_combined, color='dimgray', s=30, marker='x')

line_dl_null, = subplt.plot(bl_list, error_list_dl_null, color='blueviolet', linewidth=1.5, label='Null Model')
subplt.fill_between(bl_list, error_list_dl_null_min, error_list_dl_null_max, color='blueviolet', alpha=0.25, edgecolor='none')
scatter_dl_null = subplt.scatter(bl_list, error_list_dl_null, color='blueviolet', s=30, marker='D')

legend_dl = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='DL Algorithm', linestyle='-', markeredgewidth=1.5)
legend_dl_combined = mlines.Line2D([], [], color='dimgray',  marker='x',markersize=5, label='Combined Model', linestyle='-', markeredgewidth=1.5)
legend_dl_null = mlines.Line2D([], [], color='blueviolet',  marker='D',markersize=5, label='Null Model', linestyle='-', markeredgewidth=1.5)

plt.xticks(bl_list,fontproperties=times_font)
plt.ylim(-0.25,5.25)
plt.yticks([0,2,5],fontproperties=times_font)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)
#act = plt.hlines(0,bl_list[0],bl_list[-1],linestyles='dashed',colors='black',label='Ground Truth',linewidth=2)

plt.ylabel('Mean relative error',font)
handles = [legend_dl, legend_dl_null, legend_dl_combined]
labels = [h.get_label() for h in handles]

subplt.set_title('May Harvesting Fold Model (1D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'a',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=2,prop={'size':10})

subplt = plt.subplot(3,2,2)

df = pandas.read_csv('../results/food_hopf_nus.csv')

df_kl = df['kl']

df_dl = df['error_dl']
df_dl_min = df['min_dl']
df_dl_max = df['max_dl']

df_dl_combined = df['error_dl_combined']
df_dl_combined_min = df['min_dl_combined']
df_dl_combined_max = df['max_dl_combined']

df_dl_null = df['error_dl_null']
df_dl_null_min = df['min_dl_null']
df_dl_null_max = df['max_dl_null']

kl_list = list(df_kl.values)

error_list_dl = list(df_dl.values)
error_list_dl_min = list(df_dl_min.values)
error_list_dl_max = list(df_dl_max.values)

error_list_dl_combined = list(df_dl_combined.values)
error_list_dl_combined_min = list(df_dl_combined_min.values)
error_list_dl_combined_max = list(df_dl_combined_max.values)

error_list_dl_null = list(df_dl_null.values)
error_list_dl_null_min = list(df_dl_null_min.values)
error_list_dl_null_max = list(df_dl_null_max.values)

line_dl, = subplt.plot(kl_list, error_list_dl, color='crimson', linewidth=1.5, label='DL Algorithm')
subplt.fill_between(kl_list, error_list_dl_min, error_list_dl_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_dl = subplt.scatter(kl_list, error_list_dl, color='crimson', s=30, marker='o')

line_dl_combined, = subplt.plot(kl_list, error_list_dl_combined, color='dimgray', linewidth=1.5, label='Combined Model')
subplt.fill_between(kl_list, error_list_dl_combined_min, error_list_dl_combined_max, color='dimgray', alpha=0.25, edgecolor='none')
scatter_dl_combined = subplt.scatter(kl_list, error_list_dl_combined, color='dimgray', s=30, marker='x')

line_dl_null, = subplt.plot(kl_list, error_list_dl_null, color='blueviolet', linewidth=1.5, label='Null Model')
subplt.fill_between(kl_list, error_list_dl_null_min, error_list_dl_null_max, color='blueviolet', alpha=0.25, edgecolor='none')
scatter_dl_null = subplt.scatter(kl_list, error_list_dl_null, color='blueviolet', s=30, marker='D')

legend_dl = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='DL Algorithm', linestyle='-', markeredgewidth=1.5)
legend_dl_combined = mlines.Line2D([], [], color='dimgray',  marker='x',markersize=5, label='Combined Model', linestyle='-', markeredgewidth=1.5)
legend_dl_null = mlines.Line2D([], [], color='blueviolet',  marker='D',markersize=5, label='Null Model', linestyle='-', markeredgewidth=1.5)

plt.xticks(kl_list,fontproperties=times_font)
plt.ylim(-0.25,5.25)
plt.yticks([0,2,5],fontproperties=times_font)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)
#act = plt.hlines(0,kl_list[0],kl_list[-1],linestyles='dashed',colors='black',label='Ground Truth',linewidth=2)

handles = [legend_dl, legend_dl_null, legend_dl_combined]
labels = [h.get_label() for h in handles]

subplt.set_title('Food Chain Hopf Model (3D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'b',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=2,prop={'size':10})

subplt = plt.subplot(3,2,3)

df = pandas.read_csv('../results/cr_branch_nus.csv')

df_al = df['al']

df_dl = df['error_dl']
df_dl_min = df['min_dl']
df_dl_max = df['max_dl']

df_dl_combined = df['error_dl_combined']
df_dl_combined_min = df['min_dl_combined']
df_dl_combined_max = df['max_dl_combined']

df_dl_null = df['error_dl_null']
df_dl_null_min = df['min_dl_null']
df_dl_null_max = df['max_dl_null']

al_list = list(df_al.values)

error_list_dl = list(df_dl.values)
error_list_dl_min = list(df_dl_min.values)
error_list_dl_max = list(df_dl_max.values)

error_list_dl_combined = list(df_dl_combined.values)
error_list_dl_combined_min = list(df_dl_combined_min.values)
error_list_dl_combined_max = list(df_dl_combined_max.values)

error_list_dl_null = list(df_dl_null.values)
error_list_dl_null_min = list(df_dl_null_min.values)
error_list_dl_null_max = list(df_dl_null_max.values)

line_dl, = subplt.plot(al_list, error_list_dl, color='crimson', linewidth=1.5, label='DL Algorithm')
subplt.fill_between(al_list, error_list_dl_min, error_list_dl_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_dl = subplt.scatter(al_list, error_list_dl, color='crimson', s=30, marker='o')

line_dl_combined, = subplt.plot(al_list, error_list_dl_combined, color='dimgray', linewidth=1.5, label='Combined Model')
subplt.fill_between(al_list, error_list_dl_combined_min, error_list_dl_combined_max, color='dimgray', alpha=0.25, edgecolor='none')
scatter_dl_combined = subplt.scatter(al_list, error_list_dl_combined, color='dimgray', s=30, marker='x')

line_dl_null, = subplt.plot(al_list, error_list_dl_null, color='blueviolet', linewidth=1.5, label='Null Model')
subplt.fill_between(al_list, error_list_dl_null_min, error_list_dl_null_max, color='blueviolet', alpha=0.25, edgecolor='none')
scatter_dl_null = subplt.scatter(al_list, error_list_dl_null, color='blueviolet', s=30, marker='D')

legend_dl = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='DL Algorithm', linestyle='-', markeredgewidth=1.5)
legend_dl_combined = mlines.Line2D([], [], color='dimgray',  marker='x',markersize=5, label='Combined Model', linestyle='-', markeredgewidth=1.5)
legend_dl_null = mlines.Line2D([], [], color='blueviolet',  marker='D',markersize=5, label='Null Model', linestyle='-', markeredgewidth=1.5)

plt.xticks(al_list,fontproperties=times_font)
plt.ylim(-0.25,5.25)
plt.yticks([0,2,5],fontproperties=times_font)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)
#act = plt.hlines(0,al_list[0],al_list[-1],linestyles='dashed',colors='black',label='Ground Truth',linewidth=2)

plt.ylabel('Mean relative error',font)
handles = [legend_dl, legend_dl_null, legend_dl_combined]
labels = [h.get_label() for h in handles]

subplt.set_title('Consumer Resource Transcritical Model (2D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'c',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=2,prop={'size':10})

subplt = plt.subplot(3,2,4)

df = pandas.read_csv('../results/global_fold_nus.csv')

df_ul = df['ul']

df_dl = df['error_dl']
df_dl_min = df['min_dl']
df_dl_max = df['max_dl']

df_dl_combined = df['error_dl_combined']
df_dl_combined_min = df['min_dl_combined']
df_dl_combined_max = df['max_dl_combined']

df_dl_null = df['error_dl_null']
df_dl_null_min = df['min_dl_null']
df_dl_null_max = df['max_dl_null']

ul_list = list(df_ul.values)

error_list_dl = list(df_dl.values)
error_list_dl_min = list(df_dl_min.values)
error_list_dl_max = list(df_dl_max.values)

error_list_dl_combined = list(df_dl_combined.values)
error_list_dl_combined_min = list(df_dl_combined_min.values)
error_list_dl_combined_max = list(df_dl_combined_max.values)

error_list_dl_null = list(df_dl_null.values)
error_list_dl_null_min = list(df_dl_null_min.values)
error_list_dl_null_max = list(df_dl_null_max.values)

line_dl, = subplt.plot(ul_list, error_list_dl, color='crimson', linewidth=1.5, label='DL Algorithm')
subplt.fill_between(ul_list, error_list_dl_min, error_list_dl_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_dl = subplt.scatter(ul_list, error_list_dl, color='crimson', s=30, marker='o')

line_dl_combined, = subplt.plot(ul_list, error_list_dl_combined, color='dimgray', linewidth=1.5, label='Combined Model')
subplt.fill_between(ul_list, error_list_dl_combined_min, error_list_dl_combined_max, color='dimgray', alpha=0.25, edgecolor='none')
scatter_dl_combined = subplt.scatter(ul_list, error_list_dl_combined, color='dimgray', s=30, marker='x')

line_dl_null, = subplt.plot(ul_list, error_list_dl_null, color='blueviolet', linewidth=1.5, label='Null Model')
subplt.fill_between(ul_list, error_list_dl_null_min, error_list_dl_null_max, color='blueviolet', alpha=0.25, edgecolor='none')
scatter_dl_null = subplt.scatter(ul_list, error_list_dl_null, color='blueviolet', s=30, marker='D')

legend_dl = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='DL Algorithm', linestyle='-', markeredgewidth=1.5)
legend_dl_combined = mlines.Line2D([], [], color='dimgray',  marker='x',markersize=5, label='Combined Model', linestyle='-', markeredgewidth=1.5)
legend_dl_null = mlines.Line2D([], [], color='blueviolet',  marker='D',markersize=5, label='Null Model', linestyle='-', markeredgewidth=1.5)

plt.xticks(ul_list,fontproperties=times_font)
plt.ylim(-0.25,5.25)
plt.yticks([0,2,5],fontproperties=times_font)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)
#act = plt.hlines(0,ul_list[0],ul_list[-1],linestyles='dashed',colors='black',label='Ground Truth',linewidth=2)

handles = [legend_dl, legend_dl_null, legend_dl_combined]
labels = [h.get_label() for h in handles]

subplt.set_title('Global Energy Balance Fold Model (1D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'d',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=2,prop={'size':10})

subplt = plt.subplot(3,2,5)

df = pandas.read_csv('../results/MPT_hopf_nus.csv')

df_ul = df['ul']

df_dl = df['error_dl']
df_dl_min = df['min_dl']
df_dl_max = df['max_dl']

df_dl_combined = df['error_dl_combined']
df_dl_combined_min = df['min_dl_combined']
df_dl_combined_max = df['max_dl_combined']

df_dl_null = df['error_dl_null']
df_dl_null_min = df['min_dl_null']
df_dl_null_max = df['max_dl_null']

ul_list = list(df_ul.values)

error_list_dl = list(df_dl.values)
error_list_dl_min = list(df_dl_min.values)
error_list_dl_max = list(df_dl_max.values)

error_list_dl_combined = list(df_dl_combined.values)
error_list_dl_combined_min = list(df_dl_combined_min.values)
error_list_dl_combined_max = list(df_dl_combined_max.values)

error_list_dl_null = list(df_dl_null.values)
error_list_dl_null_min = list(df_dl_null_min.values)
error_list_dl_null_max = list(df_dl_null_max.values)

line_dl, = subplt.plot(ul_list, error_list_dl, color='crimson', linewidth=1.5, label='DL Algorithm')
subplt.fill_between(ul_list, error_list_dl_min, error_list_dl_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_dl = subplt.scatter(ul_list, error_list_dl, color='crimson', s=30, marker='o')

line_dl_combined, = subplt.plot(ul_list, error_list_dl_combined, color='dimgray', linewidth=1.5, label='Combined Model')
subplt.fill_between(ul_list, error_list_dl_combined_min, error_list_dl_combined_max, color='dimgray', alpha=0.25, edgecolor='none')
scatter_dl_combined = subplt.scatter(ul_list, error_list_dl_combined, color='dimgray', s=30, marker='x')

line_dl_null, = subplt.plot(ul_list, error_list_dl_null, color='blueviolet', linewidth=1.5, label='Null Model')
subplt.fill_between(ul_list, error_list_dl_null_min, error_list_dl_null_max, color='blueviolet', alpha=0.25, edgecolor='none')
scatter_dl_null = subplt.scatter(ul_list, error_list_dl_null, color='blueviolet', s=30, marker='D')

legend_dl = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='DL Algorithm', linestyle='-', markeredgewidth=1.5)
legend_dl_combined = mlines.Line2D([], [], color='dimgray',  marker='x',markersize=5, label='Combined Model', linestyle='-', markeredgewidth=1.5)
legend_dl_null = mlines.Line2D([], [], color='blueviolet',  marker='D',markersize=5, label='Null Model', linestyle='-', markeredgewidth=1.5)

plt.xticks(ul_list,fontproperties=times_font)
plt.ylim(-0.25,5.25)
plt.yticks([0,2,5],fontproperties=times_font)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)
#act = plt.hlines(0,pl_list[0],pl_list[-1],linestyles='dashed',colors='black',label='Ground Truth',linewidth=2)

plt.xlabel('Initial parameter',font)
plt.ylabel('Mean relative error',font)
handles = [legend_dl, legend_dl_null, legend_dl_combined]
labels = [h.get_label() for h in handles]

subplt.set_title('Middle Pleistocene Transition Hopf Model (3D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'e',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=2,prop={'size':10})

subplt = plt.subplot(3,2,6)

df = pandas.read_csv('../results/amazon_branch_nus.csv')

df_pl = df['pl']

df_dl = df['error_dl']
df_dl_min = df['min_dl']
df_dl_max = df['max_dl']

df_dl_combined = df['error_dl_combined']
df_dl_combined_min = df['min_dl_combined']
df_dl_combined_max = df['max_dl_combined']

df_dl_null = df['error_dl_null']
df_dl_null_min = df['min_dl_null']
df_dl_null_max = df['max_dl_null']

pl_list = list(df_pl.values)

error_list_dl = list(df_dl.values)
error_list_dl_min = list(df_dl_min.values)
error_list_dl_max = list(df_dl_max.values)

error_list_dl_combined = list(df_dl_combined.values)
error_list_dl_combined_min = list(df_dl_combined_min.values)
error_list_dl_combined_max = list(df_dl_combined_max.values)

error_list_dl_null = list(df_dl_null.values)
error_list_dl_null_min = list(df_dl_null_min.values)
error_list_dl_null_max = list(df_dl_null_max.values)

line_dl, = subplt.plot(pl_list, error_list_dl, color='crimson', linewidth=1.5, label='DL Algorithm')
subplt.fill_between(pl_list, error_list_dl_min, error_list_dl_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_dl = subplt.scatter(pl_list, error_list_dl, color='crimson', s=30, marker='o')

line_dl_combined, = subplt.plot(pl_list, error_list_dl_combined, color='dimgray', linewidth=1.5, label='Combined Model')
subplt.fill_between(pl_list, error_list_dl_combined_min, error_list_dl_combined_max, color='dimgray', alpha=0.25, edgecolor='none')
scatter_dl_combined = subplt.scatter(pl_list, error_list_dl_combined, color='dimgray', s=30, marker='x')

line_dl_null, = subplt.plot(pl_list, error_list_dl_null, color='blueviolet', linewidth=1.5, label='Null Model')
subplt.fill_between(pl_list, error_list_dl_null_min, error_list_dl_null_max, color='blueviolet', alpha=0.25, edgecolor='none')
scatter_dl_null = subplt.scatter(pl_list, error_list_dl_null, color='blueviolet', s=30, marker='D')

legend_dl = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='DL Algorithm', linestyle='-', markeredgewidth=1.5)
legend_dl_combined = mlines.Line2D([], [], color='dimgray',  marker='x',markersize=5, label='Combined Model', linestyle='-', markeredgewidth=1.5)
legend_dl_null = mlines.Line2D([], [], color='blueviolet',  marker='D',markersize=5, label='Null Model', linestyle='-', markeredgewidth=1.5)

plt.xticks(pl_list,fontproperties=times_font)
plt.ylim(-0.25,5.25)
plt.yticks([0,2,5],fontproperties=times_font)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)
#act = plt.hlines(0,pl_list[0],pl_list[-1],linestyles='dashed',colors='black',label='Ground Truth',linewidth=2)

plt.xlabel('Initial parameter',font)
handles = [legend_dl, legend_dl_null, legend_dl_combined]
labels = [h.get_label() for h in handles]

subplt.set_title('Amazon Rainforest Dieback Transcritical Model (1D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'f',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=2,prop={'size':10})

plt.tight_layout()
plt.savefig('../figures/FIG3.png',dpi=600)