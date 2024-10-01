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

df = pandas.read_csv('../results/may_fold/may_fold_crate_us.csv')

df_bl = df['bl']

df_1 = df['error_dl_1e-05']
df_1_min = df['min_dl_1e-05']
df_1_max = df['max_dl_1e-05']

df_2 = df['error_dl_2e-05']
df_2_min = df['min_dl_2e-05']
df_2_max = df['max_dl_2e-05']

df_3 = df['error_dl_3e-05']
df_3_min = df['min_dl_3e-05']
df_3_max = df['max_dl_3e-05']

df_4 = df['error_dl_4e-05']
df_4_min = df['min_dl_4e-05']
df_4_max = df['max_dl_4e-05']

df_5 = df['error_dl_5e-05']
df_5_min = df['min_dl_5e-05']
df_5_max = df['max_dl_5e-05']

bl_list = list(df_bl.values)

error_list_1 = list(df_1.values)
error_list_1_min = list(df_1_min.values)
error_list_1_max = list(df_1_max.values)

error_list_2 = list(df_2.values)
error_list_2_min = list(df_2_min.values)
error_list_2_max = list(df_2_max.values)

error_list_3 = list(df_3.values)
error_list_3_min = list(df_3_min.values)
error_list_3_max = list(df_3_max.values)

error_list_4 = list(df_4.values)
error_list_4_min = list(df_4_min.values)
error_list_4_max = list(df_4_max.values)

error_list_5 = list(df_5.values)
error_list_5_min = list(df_5_min.values)
error_list_5_max = list(df_5_max.values)

line_1, = subplt.plot(bl_list, error_list_1, color='crimson', linewidth=1.5, label='1e-05')
subplt.fill_between(bl_list, error_list_1_min, error_list_1_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_1 = subplt.scatter(bl_list, error_list_1, color='crimson', s=30, marker='o')

line_2, = subplt.plot(bl_list, error_list_2, color='royalblue', linewidth=1.5, label='2e-05')
subplt.fill_between(bl_list, error_list_2_min, error_list_2_max, color='royalblue', alpha=0.25, edgecolor='none')
scatter_2 = subplt.scatter(bl_list, error_list_2, color='royalblue', s=30, marker='^')

line_3, = subplt.plot(bl_list, error_list_3, color='forestgreen', linewidth=1.5, label='3e-05')
subplt.fill_between(bl_list, error_list_3_min, error_list_3_max, color='forestgreen', alpha=0.25, edgecolor='none')
scatter_3 = subplt.scatter(bl_list, error_list_3, color='forestgreen', s=30, marker='s')

line_4 = subplt.plot(bl_list, error_list_4, color='blueviolet', linewidth=1.5, label='4e-05')
subplt.fill_between(bl_list, error_list_4_min, error_list_4_max, color='blueviolet', alpha=0.25, edgecolor='none')
scatter_4 = subplt.scatter(bl_list, error_list_4, color='blueviolet', s=30, marker='D')

line_5 = subplt.plot(bl_list, error_list_5, color='darkorange', linewidth=1.5, label='5e-05')
subplt.fill_between(bl_list, error_list_5_min, error_list_5_max, color='darkorange', alpha=0.25, edgecolor='none')
scatter_5 = subplt.scatter(bl_list, error_list_5, color='darkorange', s=30, marker='*')

# 创建自定义图例
legend_1 = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='1e-05', linestyle='-', markeredgewidth=1.5)
legend_2 = mlines.Line2D([], [], color='royalblue', marker='^', markersize=5, label='2e-05', linestyle='-', markeredgewidth=1.5)
legend_3 = mlines.Line2D([], [], color='forestgreen', marker='s', markersize=5, label='3e-05', linestyle='-', markeredgewidth=1.5)
legend_4 = mlines.Line2D([], [], color='blueviolet',  marker='D',markersize=5, label='4e-05', linestyle='-', markeredgewidth=1.5)
legend_5 = mlines.Line2D([], [], color='darkorange',  marker='*',markersize=5, label='5e-05', linestyle='-', markeredgewidth=1.5)

plt.xticks(bl_list,fontproperties=times_font)
plt.ylim(-0.03,0.63)
plt.yticks([0,0.3,0.6],fontproperties=times_font)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)

plt.ylabel('Mean relative error',font)
handles = [legend_1, legend_2, legend_3, legend_4, legend_5]
labels = [h.get_label() for h in handles]

subplt.set_title('May Harvesting Fold Model (1D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'a',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=1,prop={'size':10},ncol=2)

subplt = plt.subplot(3,2,2)

df = pandas.read_csv('../results/food_hopf/food_hopf_crate_us.csv')

df_kl = df['kl']

df_1 = df['error_dl_1e-05']
df_1_min = df['min_dl_1e-05']
df_1_max = df['max_dl_1e-05']

df_2 = df['error_dl_2e-05']
df_2_min = df['min_dl_2e-05']
df_2_max = df['max_dl_2e-05']

df_3 = df['error_dl_3e-05']
df_3_min = df['min_dl_3e-05']
df_3_max = df['max_dl_3e-05']

df_4 = df['error_dl_4e-05']
df_4_min = df['min_dl_4e-05']
df_4_max = df['max_dl_4e-05']

df_5 = df['error_dl_5e-05']
df_5_min = df['min_dl_5e-05']
df_5_max = df['max_dl_5e-05']

kl_list = list(df_kl.values)

error_list_1 = list(df_1.values)
error_list_1_min = list(df_1_min.values)
error_list_1_max = list(df_1_max.values)

error_list_2 = list(df_2.values)
error_list_2_min = list(df_2_min.values)
error_list_2_max = list(df_2_max.values)

error_list_3 = list(df_3.values)
error_list_3_min = list(df_3_min.values)
error_list_3_max = list(df_3_max.values)

error_list_4 = list(df_4.values)
error_list_4_min = list(df_4_min.values)
error_list_4_max = list(df_4_max.values)

error_list_5 = list(df_5.values)
error_list_5_min = list(df_5_min.values)
error_list_5_max = list(df_5_max.values)

line_1, = subplt.plot(kl_list, error_list_1, color='crimson', linewidth=1.5, label='1e-05')
subplt.fill_between(kl_list, error_list_1_min, error_list_1_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_1 = subplt.scatter(kl_list, error_list_1, color='crimson', s=30, marker='o')

line_2, = subplt.plot(kl_list, error_list_2, color='royalblue', linewidth=1.5, label='2e-05')
subplt.fill_between(kl_list, error_list_2_min, error_list_2_max, color='royalblue', alpha=0.25, edgecolor='none')
scatter_2 = subplt.scatter(kl_list, error_list_2, color='royalblue', s=30, marker='^')

line_3, = subplt.plot(kl_list, error_list_3, color='forestgreen', linewidth=1.5, label='3e-05')
subplt.fill_between(kl_list, error_list_3_min, error_list_3_max, color='forestgreen', alpha=0.25, edgecolor='none')
scatter_3 = subplt.scatter(kl_list, error_list_3, color='forestgreen', s=30, marker='s')

line_4 = subplt.plot(kl_list, error_list_4, color='blueviolet', linewidth=1.5, label='4e-05')
subplt.fill_between(kl_list, error_list_4_min, error_list_4_max, color='blueviolet', alpha=0.25, edgecolor='none')
scatter_4 = subplt.scatter(kl_list, error_list_4, color='blueviolet', s=30, marker='D')

line_5 = subplt.plot(kl_list, error_list_5, color='darkorange', linewidth=1.5, label='5e-05')
subplt.fill_between(kl_list, error_list_5_min, error_list_5_max, color='darkorange', alpha=0.25, edgecolor='none')
scatter_5 = subplt.scatter(kl_list, error_list_5, color='darkorange', s=30, marker='*')

# 创建自定义图例
legend_1 = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='1e-05', linestyle='-', markeredgewidth=1.5)
legend_2 = mlines.Line2D([], [], color='royalblue', marker='^', markersize=5, label='2e-05', linestyle='-', markeredgewidth=1.5)
legend_3 = mlines.Line2D([], [], color='forestgreen', marker='s', markersize=5, label='3e-05', linestyle='-', markeredgewidth=1.5)
legend_4 = mlines.Line2D([], [], color='blueviolet',  marker='D',markersize=5, label='4e-05', linestyle='-', markeredgewidth=1.5)
legend_5 = mlines.Line2D([], [], color='darkorange',  marker='*',markersize=5, label='5e-05', linestyle='-', markeredgewidth=1.5)

plt.xticks(kl_list,fontproperties=times_font)
plt.ylim(-0.03,0.63)
plt.yticks([0,0.3,0.6],fontproperties=times_font)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)

handles = [legend_1, legend_2, legend_3, legend_4, legend_5]
labels = [h.get_label() for h in handles]

subplt.set_title('Chaotic Food Chain Hopf Model (3D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'b',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=1,prop={'size':10},ncol=2)

subplt = plt.subplot(3,2,3)

df = pandas.read_csv('../results/cr_branch/cr_branch_crate_us.csv')

df_al = df['al']

df_1 = df['error_dl_0.0001']
df_1_min = df['min_dl_0.0001']
df_1_max = df['max_dl_0.0001']

df_2 = df['error_dl_0.0002']
df_2_min = df['min_dl_0.0002']
df_2_max = df['max_dl_0.0002']

df_3 = df['error_dl_0.0003']
df_3_min = df['min_dl_0.0003']
df_3_max = df['max_dl_0.0003']

df_4 = df['error_dl_0.0004']
df_4_min = df['min_dl_0.0004']
df_4_max = df['max_dl_0.0004']

df_5 = df['error_dl_0.0005']
df_5_min = df['min_dl_0.0005']
df_5_max = df['max_dl_0.0005']

al_list = list(df_al.values)

error_list_1 = list(df_1.values)
error_list_1_min = list(df_1_min.values)
error_list_1_max = list(df_1_max.values)

error_list_2 = list(df_2.values)
error_list_2_min = list(df_2_min.values)
error_list_2_max = list(df_2_max.values)

error_list_3 = list(df_3.values)
error_list_3_min = list(df_3_min.values)
error_list_3_max = list(df_3_max.values)

error_list_4 = list(df_4.values)
error_list_4_min = list(df_4_min.values)
error_list_4_max = list(df_4_max.values)

error_list_5 = list(df_5.values)
error_list_5_min = list(df_5_min.values)
error_list_5_max = list(df_5_max.values)

line_1, = subplt.plot(al_list, error_list_1, color='crimson', linewidth=1.5, label='0.0001')
subplt.fill_between(al_list, error_list_1_min, error_list_1_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_1 = subplt.scatter(al_list, error_list_1, color='crimson', s=30, marker='o')

line_2, = subplt.plot(al_list, error_list_2, color='royalblue', linewidth=1.5, label='0.0002')
subplt.fill_between(al_list, error_list_2_min, error_list_2_max, color='royalblue', alpha=0.25, edgecolor='none')
scatter_2 = subplt.scatter(al_list, error_list_2, color='royalblue', s=30, marker='^')

line_3, = subplt.plot(al_list, error_list_3, color='forestgreen', linewidth=1.5, label='0.0003')
subplt.fill_between(al_list, error_list_3_min, error_list_3_max, color='forestgreen', alpha=0.25, edgecolor='none')
scatter_3 = subplt.scatter(al_list, error_list_3, color='forestgreen', s=30, marker='s')

line_4 = subplt.plot(al_list, error_list_4, color='blueviolet', linewidth=1.5, label='0.0004')
subplt.fill_between(al_list, error_list_4_min, error_list_4_max, color='blueviolet', alpha=0.25, edgecolor='none')
scatter_4 = subplt.scatter(al_list, error_list_4, color='blueviolet', s=30, marker='D')

line_5 = subplt.plot(al_list, error_list_5, color='darkorange', linewidth=1.5, label='0.0005')
subplt.fill_between(al_list, error_list_5_min, error_list_5_max, color='darkorange', alpha=0.25, edgecolor='none')
scatter_5 = subplt.scatter(al_list, error_list_5, color='darkorange', s=30, marker='*')

# 创建自定义图例
legend_1 = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='0.0001', linestyle='-', markeredgewidth=1.5)
legend_2 = mlines.Line2D([], [], color='royalblue', marker='^', markersize=5, label='0.0002', linestyle='-', markeredgewidth=1.5)
legend_3 = mlines.Line2D([], [], color='forestgreen', marker='s', markersize=5, label='0.0003', linestyle='-', markeredgewidth=1.5)
legend_4 = mlines.Line2D([], [], color='blueviolet',  marker='D',markersize=5, label='0.0004', linestyle='-', markeredgewidth=1.5)
legend_5 = mlines.Line2D([], [], color='darkorange',  marker='*',markersize=5, label='0.0005', linestyle='-', markeredgewidth=1.5)

plt.xticks(al_list,fontproperties=times_font)
plt.ylim(-0.03,0.63)
plt.yticks([0,0.3,0.6],fontproperties=times_font)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)

plt.ylabel('Mean relative error',font)
handles = [legend_1, legend_2, legend_3, legend_4, legend_5]
labels = [h.get_label() for h in handles]

subplt.set_title('Consumer Resource Transcritical Model (2D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'c',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=2,prop={'size':10},ncol=2)

subplt = plt.subplot(3,2,4)

df = pandas.read_csv('../results/global_fold/global_fold_crate_us.csv')

df_ul = df['ul']

df_1 = df['error_dl_-5e-07']
df_1_min = df['min_dl_-5e-07']
df_1_max = df['max_dl_-5e-07']

df_2 = df['error_dl_-6e-07']
df_2_min = df['min_dl_-6e-07']
df_2_max = df['max_dl_-6e-07']

df_3 = df['error_dl_-7e-07']
df_3_min = df['min_dl_-7e-07']
df_3_max = df['max_dl_-7e-07']

df_4 = df['error_dl_-8e-07']
df_4_min = df['min_dl_-8e-07']
df_4_max = df['max_dl_-8e-07']

df_5 = df['error_dl_-9e-07']
df_5_min = df['min_dl_-9e-07']
df_5_max = df['max_dl_-9e-07']

ul_list = list(df_ul.values)

error_list_1 = list(df_1.values)
error_list_1_min = list(df_1_min.values)
error_list_1_max = list(df_1_max.values)

error_list_2 = list(df_2.values)
error_list_2_min = list(df_2_min.values)
error_list_2_max = list(df_2_max.values)

error_list_3 = list(df_3.values)
error_list_3_min = list(df_3_min.values)
error_list_3_max = list(df_3_max.values)

error_list_4 = list(df_4.values)
error_list_4_min = list(df_4_min.values)
error_list_4_max = list(df_4_max.values)

error_list_5 = list(df_5.values)
error_list_5_min = list(df_5_min.values)
error_list_5_max = list(df_5_max.values)

line_1, = subplt.plot(ul_list, error_list_1, color='crimson', linewidth=1.5, label='-5e-07')
subplt.fill_between(ul_list, error_list_1_min, error_list_1_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_1 = subplt.scatter(ul_list, error_list_1, color='crimson', s=30, marker='o')

line_2, = subplt.plot(ul_list, error_list_2, color='royalblue', linewidth=1.5, label='-6e-07')
subplt.fill_between(ul_list, error_list_2_min, error_list_2_max, color='royalblue', alpha=0.25, edgecolor='none')
scatter_2 = subplt.scatter(ul_list, error_list_2, color='royalblue', s=30, marker='^')

line_3, = subplt.plot(ul_list, error_list_3, color='forestgreen', linewidth=1.5, label='-7e-07')
subplt.fill_between(ul_list, error_list_3_min, error_list_3_max, color='forestgreen', alpha=0.25, edgecolor='none')
scatter_3 = subplt.scatter(ul_list, error_list_3, color='forestgreen', s=30, marker='s')

line_4 = subplt.plot(ul_list, error_list_4, color='blueviolet', linewidth=1.5, label='-8e-07')
subplt.fill_between(ul_list, error_list_4_min, error_list_4_max, color='blueviolet', alpha=0.25, edgecolor='none')
scatter_4 = subplt.scatter(ul_list, error_list_4, color='blueviolet', s=30, marker='D')

line_5 = subplt.plot(ul_list, error_list_5, color='darkorange', linewidth=1.5, label='-9e-07')
subplt.fill_between(ul_list, error_list_5_min, error_list_5_max, color='darkorange', alpha=0.25, edgecolor='none')
scatter_5 = subplt.scatter(ul_list, error_list_5, color='darkorange', s=30, marker='*')

# 创建自定义图例
legend_1 = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='-5e-07', linestyle='-', markeredgewidth=1.5)
legend_2 = mlines.Line2D([], [], color='royalblue', marker='^', markersize=5, label='-6e-07', linestyle='-', markeredgewidth=1.5)
legend_3 = mlines.Line2D([], [], color='forestgreen', marker='s', markersize=5, label='-7e-07', linestyle='-', markeredgewidth=1.5)
legend_4 = mlines.Line2D([], [], color='blueviolet',  marker='D',markersize=5, label='-8e-07', linestyle='-', markeredgewidth=1.5)
legend_5 = mlines.Line2D([], [], color='darkorange',  marker='*',markersize=5, label='-9e-07', linestyle='-', markeredgewidth=1.5)

plt.xticks(ul_list,fontproperties=times_font)
plt.ylim(-0.03,0.63)
plt.yticks([0,0.3,0.6],fontproperties=times_font)
plt.gca().invert_xaxis()
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)

handles = [legend_1, legend_2, legend_3, legend_4, legend_5]
labels = [h.get_label() for h in handles]

subplt.set_title('Global Energy Balance Fold Model (1D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'d',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=1,prop={'size':10},ncol=2)

subplt = plt.subplot(3,2,5)

df = pandas.read_csv('../results/MPT_hopf/MPT_hopf_crate_us.csv')

df_ul = df['ul']

df_1 = df['error_dl_1e-05']
df_1_min = df['min_dl_1e-05']
df_1_max = df['max_dl_1e-05']

df_2 = df['error_dl_2e-05']
df_2_min = df['min_dl_2e-05']
df_2_max = df['max_dl_2e-05']

df_3 = df['error_dl_3e-05']
df_3_min = df['min_dl_3e-05']
df_3_max = df['max_dl_3e-05']

df_4 = df['error_dl_4e-05']
df_4_min = df['min_dl_4e-05']
df_4_max = df['max_dl_4e-05']

df_5 = df['error_dl_5e-05']
df_5_min = df['min_dl_5e-05']
df_5_max = df['max_dl_5e-05']

ul_list = list(df_ul.values)

error_list_1 = list(df_1.values)
error_list_1_min = list(df_1_min.values)
error_list_1_max = list(df_1_max.values)

error_list_2 = list(df_2.values)
error_list_2_min = list(df_2_min.values)
error_list_2_max = list(df_2_max.values)

error_list_3 = list(df_3.values)
error_list_3_min = list(df_3_min.values)
error_list_3_max = list(df_3_max.values)

error_list_4 = list(df_4.values)
error_list_4_min = list(df_4_min.values)
error_list_4_max = list(df_4_max.values)

error_list_5 = list(df_5.values)
error_list_5_min = list(df_5_min.values)
error_list_5_max = list(df_5_max.values)

line_1, = subplt.plot(ul_list, error_list_1, color='crimson', linewidth=1.5, label='1e-05')
subplt.fill_between(ul_list, error_list_1_min, error_list_1_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_1 = subplt.scatter(ul_list, error_list_1, color='crimson', s=30, marker='o')

line_2, = subplt.plot(ul_list, error_list_2, color='royalblue', linewidth=1.5, label='2e-05')
subplt.fill_between(ul_list, error_list_2_min, error_list_2_max, color='royalblue', alpha=0.25, edgecolor='none')
scatter_2 = subplt.scatter(ul_list, error_list_2, color='royalblue', s=30, marker='^')

line_3, = subplt.plot(ul_list, error_list_3, color='forestgreen', linewidth=1.5, label='3e-05')
subplt.fill_between(ul_list, error_list_3_min, error_list_3_max, color='forestgreen', alpha=0.25, edgecolor='none')
scatter_3 = subplt.scatter(ul_list, error_list_3, color='forestgreen', s=30, marker='s')

line_4 = subplt.plot(ul_list, error_list_4, color='blueviolet', linewidth=1.5, label='4e-05')
subplt.fill_between(ul_list, error_list_4_min, error_list_4_max, color='blueviolet', alpha=0.25, edgecolor='none')
scatter_4 = subplt.scatter(ul_list, error_list_4, color='blueviolet', s=30, marker='D')

line_5 = subplt.plot(ul_list, error_list_5, color='darkorange', linewidth=1.5, label='5e-05')
subplt.fill_between(ul_list, error_list_5_min, error_list_5_max, color='darkorange', alpha=0.25, edgecolor='none')
scatter_5 = subplt.scatter(ul_list, error_list_5, color='darkorange', s=30, marker='*')

# 创建自定义图例
legend_1 = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='1e-05', linestyle='-', markeredgewidth=1.5)
legend_2 = mlines.Line2D([], [], color='royalblue', marker='^', markersize=5, label='2e-05', linestyle='-', markeredgewidth=1.5)
legend_3 = mlines.Line2D([], [], color='forestgreen', marker='s', markersize=5, label='3e-05', linestyle='-', markeredgewidth=1.5)
legend_4 = mlines.Line2D([], [], color='blueviolet',  marker='D',markersize=5, label='4e-05', linestyle='-', markeredgewidth=1.5)
legend_5 = mlines.Line2D([], [], color='darkorange',  marker='*',markersize=5, label='5e-05', linestyle='-', markeredgewidth=1.5)

plt.xticks(ul_list,fontproperties=times_font)
plt.ylim(-0.03,0.63)
plt.yticks([0,0.3,0.6],fontproperties=times_font)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)

plt.xlabel('Initial parameter',font)
plt.ylabel('Mean relative error',font)
handles = [legend_1, legend_2, legend_3, legend_4, legend_5]
labels = [h.get_label() for h in handles]

subplt.set_title('Middle Pleistocene Transition Hopf Model (3D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'e',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=1,prop={'size':10},ncol=2)

subplt = plt.subplot(3,2,6)

df = pandas.read_csv('../results/amazon_branch/amazon_branch_crate_us.csv')

df_pl = df['pl']

df_1 = df['error_dl_-1e-05']
df_1_min = df['min_dl_-1e-05']
df_1_max = df['max_dl_-1e-05']

df_2 = df['error_dl_-2e-05']
df_2_min = df['min_dl_-2e-05']
df_2_max = df['max_dl_-2e-05']

df_3 = df['error_dl_-3e-05']
df_3_min = df['min_dl_-3e-05']
df_3_max = df['max_dl_-3e-05']

df_4 = df['error_dl_-4e-05']
df_4_min = df['min_dl_-4e-05']
df_4_max = df['max_dl_-4e-05']

df_5 = df['error_dl_-5e-05']
df_5_min = df['min_dl_-5e-05']
df_5_max = df['max_dl_-5e-05']

pl_list = list(df_pl.values)

error_list_1 = list(df_1.values)
error_list_1_min = list(df_1_min.values)
error_list_1_max = list(df_1_max.values)

error_list_2 = list(df_2.values)
error_list_2_min = list(df_2_min.values)
error_list_2_max = list(df_2_max.values)

error_list_3 = list(df_3.values)
error_list_3_min = list(df_3_min.values)
error_list_3_max = list(df_3_max.values)

error_list_4 = list(df_4.values)
error_list_4_min = list(df_4_min.values)
error_list_4_max = list(df_4_max.values)

error_list_5 = list(df_5.values)
error_list_5_min = list(df_5_min.values)
error_list_5_max = list(df_5_max.values)

line_1, = subplt.plot(pl_list, error_list_1, color='crimson', linewidth=1.5, label='-1e-05')
subplt.fill_between(pl_list, error_list_1_min, error_list_1_max, color='crimson', alpha=0.25, edgecolor='none')
scatter_1 = subplt.scatter(pl_list, error_list_1, color='crimson', s=30, marker='o')

line_2, = subplt.plot(pl_list, error_list_2, color='royalblue', linewidth=1.5, label='-2e-05')
subplt.fill_between(pl_list, error_list_2_min, error_list_2_max, color='royalblue', alpha=0.25, edgecolor='none')
scatter_2 = subplt.scatter(pl_list, error_list_2, color='royalblue', s=30, marker='^')

line_3, = subplt.plot(pl_list, error_list_3, color='forestgreen', linewidth=1.5, label='-3e-05')
subplt.fill_between(pl_list, error_list_3_min, error_list_3_max, color='forestgreen', alpha=0.25, edgecolor='none')
scatter_3 = subplt.scatter(pl_list, error_list_3, color='forestgreen', s=30, marker='s')

line_4 = subplt.plot(pl_list, error_list_4, color='blueviolet', linewidth=1.5, label='-4e-05')
subplt.fill_between(pl_list, error_list_4_min, error_list_4_max, color='blueviolet', alpha=0.25, edgecolor='none')
scatter_4 = subplt.scatter(pl_list, error_list_4, color='blueviolet', s=30, marker='D')

line_5 = subplt.plot(pl_list, error_list_5, color='darkorange', linewidth=1.5, label='-5e-05')
subplt.fill_between(pl_list, error_list_5_min, error_list_5_max, color='darkorange', alpha=0.25, edgecolor='none')
scatter_5 = subplt.scatter(pl_list, error_list_5, color='darkorange', s=30, marker='*')

# 创建自定义图例
legend_1 = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='-1e-05', linestyle='-', markeredgewidth=1.5)
legend_2 = mlines.Line2D([], [], color='royalblue', marker='^', markersize=5, label='-2e-05', linestyle='-', markeredgewidth=1.5)
legend_3 = mlines.Line2D([], [], color='forestgreen', marker='s', markersize=5, label='-3e-05', linestyle='-', markeredgewidth=1.5)
legend_4 = mlines.Line2D([], [], color='blueviolet',  marker='D',markersize=5, label='-4e-05', linestyle='-', markeredgewidth=1.5)
legend_5 = mlines.Line2D([], [], color='darkorange',  marker='*',markersize=5, label='-5e-05', linestyle='-', markeredgewidth=1.5)

plt.xticks(pl_list,fontproperties=times_font)
plt.ylim(-0.03,0.63)
plt.yticks([0,0.3,0.6],fontproperties=times_font)
plt.gca().invert_xaxis()
ax = plt.gca()
ax.tick_params(axis='both', labelsize=15)

plt.xlabel('Initial parameter',font)
handles = [legend_1, legend_2, legend_3, legend_4, legend_5]
labels = [h.get_label() for h in handles]

subplt.set_title('Amazon Rainforest Dieback Transcritical Model (1D)',fontdict={'family':'Times New Roman','size':14,'weight':'bold'})
left_title = ax.text(0.02, 1.05,'f',ha='left', transform=ax.transAxes,fontdict={'family':'Times New Roman','size':18,'weight':'bold'})
subplt.legend(handles=handles,labels=labels,loc=1,prop={'size':10},ncol=2)

plt.tight_layout()
plt.savefig('../figures/SFIG5.pdf',format='pdf',dpi=600)

