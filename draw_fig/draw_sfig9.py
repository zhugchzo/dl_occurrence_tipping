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

plt.figure(figsize=(12,6))

font = {'family':'Times New Roman','weight':'normal','size': 25}
times_font = fm.FontProperties(family='Times New Roman', style='normal')

df = pandas.read_csv('../results/pitchfork_nus.csv')

df_rl = df['rl']

df_dl_sup = df['error_dl_sup']
df_dl_sup_min = df['min_dl_sup']
df_dl_sup_max = df['max_dl_sup']

df_dl_sub = df['error_dl_sub']
df_dl_sub_min = df['min_dl_sub']
df_dl_sub_max = df['max_dl_sub']

rl_list = list(df_rl.values)

error_list_dl_sup = list(df_dl_sup.values)
error_list_dl_sup_min = list(df_dl_sup_min.values)
error_list_dl_sup_max = list(df_dl_sup_max.values)

error_list_dl_sub = list(df_dl_sub.values)
error_list_dl_sub_min = list(df_dl_sub_min.values)
error_list_dl_sub_max = list(df_dl_sub_max.values)

line_dl_sup, = plt.plot(rl_list, error_list_dl_sup, color='blueviolet', linewidth=2, label='Supercritical DL Model')
plt.fill_between(rl_list, error_list_dl_sup_min, error_list_dl_sup_max, color='blueviolet', alpha=0.25, edgecolor='none')
scatter_dl_sup = plt.scatter(rl_list, error_list_dl_sup, color='blueviolet', s=100, marker='o')

line_dl_sub, = plt.plot(rl_list, error_list_dl_sub, color='darkorange', linewidth=2, label='Subcritical DL Model')
plt.fill_between(rl_list, error_list_dl_sub_min, error_list_dl_sub_max, color='darkorange', alpha=0.25, edgecolor='none')
scatter_dl_sub = plt.scatter(rl_list, error_list_dl_sub, color='darkorange', s=100, marker='x')

legend_dl_sup = mlines.Line2D([], [], color='blueviolet',  marker='o',markersize=8, label='Supercritical DL Model', linestyle='-', markeredgewidth=2)
legend_dl_sub = mlines.Line2D([], [], color='darkorange',  marker='x',markersize=8, label='Subcritical DL Model', linestyle='-', markeredgewidth=2)

plt.xticks(rl_list,fontproperties=times_font)
plt.ylim(-0.03,0.63)
plt.yticks([0,0.3,0.6],fontproperties=times_font)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=20)

plt.xlabel('Initial parameter',font)
plt.ylabel('Mean relative error',font)
handles = [legend_dl_sup, legend_dl_sub]
labels = [h.get_label() for h in handles]

plt.title('Supercritical pitchfork Model (1D)',fontdict={'family':'Times New Roman','size':30,'weight':'bold'},y=1.02)
plt.legend(handles=handles,labels=labels,loc=2,prop={'size':20})

plt.tight_layout()
plt.savefig('../figures/SFIG9.pdf',format='pdf',dpi=600)