import pandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import FuncFormatter
import os

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)

# Get the directory containing the current file
current_dir = os.path.dirname(current_file_path)

# Change the working directory to the directory of the current file
os.chdir(current_dir)

def custom_formatter(x, pos):
    if x.is_integer():
        return f"{int(x)}"
    else:
        return f"{x:g}"

font_title = {'family':'Microsoft YaHei','weight':'bold','size': 15}
font_axis = {'family':'Times New Roman','weight':'normal','size': 14}

df_20 = pandas.read_csv('../results/thermoacoustic_20_hopf.csv')
df_40 = pandas.read_csv('../results/thermoacoustic_40_hopf.csv')
df_60 = pandas.read_csv('../results/thermoacoustic_60_hopf.csv')

sample_start_20 = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6]
sample_start_40 = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6]
sample_start_60 = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8]

ground_truth_20 = 1.72
ground_truth_40 = 1.76
ground_truth_60 = 1.87

initial_point_20 = []
initial_point_40 = []
initial_point_60 = []

end_point_20 = []
end_point_40 = []
end_point_60 = []

index_ac_20 =[]
index_ac_40 =[]
index_ac_60 =[]

index_dev_20 =[]
index_dev_40 =[]
index_dev_60 =[]

for ss in sample_start_20:
    df_sims = pandas.read_csv('../empirical_test/data_nus/thermoacoustic_20_hopf/{}/thermoacoustic_20_hopf_interpolate.csv'.format(ss))
    initial_x = df_sims['x'].iloc[0]
    initial_b = df_sims['b'].iloc[0]    
    end_b = df_sims['b'].iloc[-1]

    initial_point_20.append([initial_b,initial_x])
    end_point_20.append(end_b)

for ss in sample_start_40:
    df_sims = pandas.read_csv('../empirical_test/data_nus/thermoacoustic_40_hopf/{}/thermoacoustic_40_hopf_interpolate.csv'.format(ss))
    initial_x = df_sims['x'].iloc[0]
    initial_b = df_sims['b'].iloc[0]    
    end_b = df_sims['b'].iloc[-1]

    initial_point_40.append([initial_b,initial_x])
    end_point_40.append(end_b)

for ss in sample_start_60:
    df_sims = pandas.read_csv('../empirical_test/data_nus/thermoacoustic_60_hopf/{}/thermoacoustic_60_hopf_interpolate.csv'.format(ss))
    initial_x = df_sims['x'].iloc[0]
    initial_b = df_sims['b'].iloc[0]    
    end_b = df_sims['b'].iloc[-1]

    initial_point_60.append([initial_b,initial_x])
    end_point_60.append(end_b)

# 20mv
ss_20 = df_20['ss_list'].values
preds_dl_20 = df_20['preds_dl_list'].values
preds_ac_20 = df_20['preds_ac_list'].values
preds_dev_20 = df_20['preds_dev_list'].values

preds_dl_20  = list(preds_dl_20)
preds_ac_20  = list(preds_ac_20)
preds_dev_20  = list(preds_dev_20)

for i in range(len(preds_ac_20)):
    if preds_ac_20[i] > end_point_20[i] and preds_ac_20[i] < 2.5:
        index_ac_20.append(i)

for i in range(len(preds_dev_20)):
    if preds_dev_20[i] > end_point_20[i] and preds_dev_20[i] < 2.5:
        index_dev_20.append(i)
        
# 40mv
ss_40 = df_40['ss_list'].values
preds_dl_40 = df_40['preds_dl_list'].values
preds_ac_40 = df_40['preds_ac_list'].values
preds_dev_40 = df_40['preds_dev_list'].values

preds_dl_40  = list(preds_dl_40)
preds_ac_40  = list(preds_ac_40)
preds_dev_40  = list(preds_dev_40)

for i in range(len(preds_ac_40)):
    if preds_ac_40[i] > end_point_40[i] and preds_ac_40[i] < 2.5:
        index_ac_40.append(i)

for i in range(len(preds_dev_40)):
    if preds_dev_40[i] > end_point_40[i] and preds_dev_40[i] < 2.5:
        index_dev_40.append(i)

# 60mv
ss_60 = df_60['ss_list'].values
preds_dl_60 = df_60['preds_dl_list'].values
preds_ac_60 = df_60['preds_ac_list'].values
preds_dev_60 = df_60['preds_dev_list'].values

preds_dl_60  = list(preds_dl_60)
preds_ac_60  = list(preds_ac_60)
preds_dev_60  = list(preds_dev_60)

for i in range(len(preds_ac_60)):
    if preds_ac_60[i] > end_point_60[i] and preds_ac_60[i] < 2.5:
        index_ac_60.append(i)

for i in range(len(preds_dev_60)):
    if preds_dev_60[i] > end_point_60[i] and preds_dev_60[i] < 2.5:
        index_dev_60.append(i)

title = ['A1','A2','A3','B1','B2','B3','C1','C2','C3']

fig, axs = plt.subplots(3, 3, figsize=(12, 9))

for p in range(3):

    ss = ss_20
    preds_dl = preds_dl_20
    preds_ac = preds_ac_20
    preds_dev = preds_dev_20
    initial_point = initial_point_20
    ground_truth = ground_truth_20

    index_ac = index_ac_20    
    index_dev = index_dev_20

    txt_1 = open('../empirical_test/empirical_data/thermoacoustic_20mv.txt')
    datalines_1 = txt_1.readlines()

    dataset_1 = []

    for data in datalines_1:
        data = data.strip().split('\t')
        dataset_1.append(data)

    for data in dataset_1:
        data[0]=(float(data[0])-3.65630914691268160E+9)*0.02
        data[1]=float(data[1])*1000/0.2175

    dataset_1 = np.array(dataset_1)
    dataset_1 = dataset_1[::60]

    x_1 = dataset_1[:,0]
    y_1 = dataset_1[:,1]

    subplt = axs[0,p]

    subplt.scatter(x_1,y_1,c='black',s=0.5)
    subplt.axvline(ground_truth,color='slategray',linestyle='--',label='Ground Truth')

    ymin, ymax = subplt.get_ylim()
    y_ticks = np.linspace(ymax,ymin,len(ss)+2)
    y_ticks = y_ticks[1:len(ss)+1]

    if p == 0:
        for i in range(len(ss)):
            if i % 2 == 0:
                subplt.scatter(initial_point[i][0],initial_point[i][1], color='crimson', s=40, marker='|',zorder=3)
                subplt.scatter(preds_dl[i],y_ticks[int(i/2)], color='crimson', s=20, marker='o',zorder=3)
                subplt.plot([initial_point[i][0],preds_dl[i]],[initial_point[i][1],y_ticks[int(i/2)]], color='crimson', linewidth=1.5, alpha=0.5, label='DL Algorithm')
            else:
                subplt.scatter(initial_point[i][0],initial_point[i][1], color='crimson', s=40, marker='|',zorder=3)
                subplt.scatter(preds_dl[i],y_ticks[int(-(i+1)/2)], color='crimson', s=20, marker='o',zorder=3)
                subplt.plot([initial_point[i][0],preds_dl[i]],[initial_point[i][1],y_ticks[int(-(i+1)/2)]], color='crimson', linewidth=1.5, alpha=0.5, label='DL Algorithm')

    if p == 1:

        ss_ac = [ss[i] for i in index_ac]
        initial_point_ac = [initial_point[i] for i in index_ac]
        preds_ac_clean = [preds_ac[i] for i in index_ac]
        y_ticks_ac = np.linspace(ymax,ymin,len(ss_ac)+2)

        for i in range(len(ss_ac)):
            if i % 2 == 0:
                subplt.scatter(initial_point_ac[i][0],initial_point_ac[i][1], color='royalblue', s=40, marker='|', zorder=3)
                subplt.scatter(preds_ac_clean[i],y_ticks_ac[int(i/2)], color='royalblue', s=20, marker='^', zorder=3)
                subplt.plot([initial_point_ac[i][0],preds_ac_clean[i]],[initial_point_ac[i][1],y_ticks_ac[int(i/2)]], color='royalblue', linewidth=1.5, alpha=0.5, label='Degenerate Fingerprinting')
            else:
                subplt.scatter(initial_point_ac[i][0],initial_point_ac[i][1], color='royalblue', s=40, marker='|', zorder=3)
                subplt.scatter(preds_ac_clean[i],y_ticks_ac[int(-(i+1)/2)], color='royalblue', s=20, marker='^', zorder=3)
                subplt.plot([initial_point_ac[i][0],preds_ac_clean[i]],[initial_point_ac[i][1],y_ticks_ac[int(-(i+1)/2)]], color='royalblue', linewidth=1.5, alpha=0.5, label='Degenerate Fingerprinting')

    if p == 2:

        ss_dev = [ss[i] for i in index_dev]
        initial_point_dev = [initial_point[i] for i in index_dev]
        preds_dev_clean = [preds_dev[i] for i in index_dev]
        y_ticks_dev = np.linspace(ymax,ymin,len(ss_dev)+2)

        for i in range(len(ss_dev)):
            if i % 2 == 0:
                subplt.scatter(initial_point_dev[i][0],initial_point_dev[i][1], color='forestgreen', s=40, marker='|', zorder=3)
                subplt.scatter(preds_dev_clean[i],y_ticks_dev[int(i/2)], color='forestgreen', s=20, marker='s', zorder=3)
                subplt.plot([initial_point_dev[i][0],preds_dev_clean[i]],[initial_point_dev[i][1],y_ticks_dev[int(i/2)]], color='forestgreen', linewidth=1.5, alpha=0.5, label='DEV')
            else:
                subplt.scatter(initial_point_dev[i][0],initial_point_dev[i][1], color='forestgreen', s=40, marker='|', zorder=3)
                subplt.scatter(preds_dev_clean[i],y_ticks_dev[int(-(i+1)/2)], color='forestgreen', s=20, marker='s', zorder=3)
                subplt.plot([initial_point_dev[i][0],preds_dev_clean[i]],[initial_point_dev[i][1],y_ticks_dev[int(-(i+1)/2)]], color='forestgreen', linewidth=1.5, alpha=0.5, label='DEV')

    subplt.set_title(title[p],loc='left',fontdict=font_title)
    subplt.set_xticks([0,1,ground_truth])
    subplt.xaxis.set_major_formatter(FuncFormatter(custom_formatter))
    subplt.tick_params(axis='both', labelsize=10)
    subplt.set_xlabel('Voltage (20mV/s)',font_axis)
    subplt.set_ylabel('Acoustic pressure (Pa)',font_axis)

for p in range(3):

    ss = ss_40
    preds_dl = preds_dl_40
    preds_ac = preds_ac_40
    preds_dev = preds_dev_40    
    initial_point = initial_point_40
    ground_truth = ground_truth_40

    index_ac = index_ac_40    
    index_dev = index_dev_40

    txt_2 = open('../empirical_test/empirical_data/thermoacoustic_40mv.txt')
    datalines_2 = txt_2.readlines()

    dataset_2 = []

    for data in datalines_2:
        data = data.strip().split('\t')
        dataset_2.append(data)

    for data in dataset_2:
        data[0]=(float(data[0])-3.65629963087196300E+9)*0.04
        data[1]=float(data[1])*1000/0.2175

    dataset_2 = np.array(dataset_2)
    dataset_2 = dataset_2[::30]

    x_2 = dataset_2[:,0]
    y_2 = dataset_2[:,1]

    subplt = axs[1,p]

    subplt.scatter(x_2,y_2,c='black',s=0.5)
    subplt.axvline(ground_truth,color='slategray',linestyle='--',label='Ground Truth')

    ymin, ymax = subplt.get_ylim()
    y_ticks = np.linspace(ymax,ymin,len(ss)+2)

    if p == 0:
        for i in range(len(ss)):
            if i % 2 == 0:
                subplt.scatter(initial_point[i][0],initial_point[i][1], color='crimson', s=40, marker='|',zorder=3)
                subplt.scatter(preds_dl[i],y_ticks[int(i/2)], color='crimson', s=20, marker='o',zorder=3)
                subplt.plot([initial_point[i][0],preds_dl[i]],[initial_point[i][1],y_ticks[int(i/2)]], color='crimson', linewidth=1.5, alpha=0.5, label='DL Algorithm')
            else:
                subplt.scatter(initial_point[i][0],initial_point[i][1], color='crimson', s=40, marker='|',zorder=3)
                subplt.scatter(preds_dl[i],y_ticks[int(-(i+1)/2)], color='crimson', s=20, marker='o',zorder=3)
                subplt.plot([initial_point[i][0],preds_dl[i]],[initial_point[i][1],y_ticks[int(-(i+1)/2)]], color='crimson', linewidth=1.5, alpha=0.5, label='DL Algorithm')

    if p == 1:

        ss_ac = [ss[i] for i in index_ac]
        initial_point_ac = [initial_point[i] for i in index_ac]
        preds_ac_clean = [preds_ac[i] for i in index_ac]
        y_ticks_ac = np.linspace(ymax,ymin,len(ss_ac)+2)

        for i in range(len(ss_ac)):
            if i % 2 == 0:
                subplt.scatter(initial_point_ac[i][0],initial_point_ac[i][1], color='royalblue', s=40, marker='|', zorder=3)
                subplt.scatter(preds_ac_clean[i],y_ticks_ac[int(i/2)], color='royalblue', s=20, marker='^', zorder=3)
                subplt.plot([initial_point_ac[i][0],preds_ac_clean[i]],[initial_point_ac[i][1],y_ticks_ac[int(i/2)]], color='royalblue', linewidth=1.5, alpha=0.5, label='Degenerate Fingerprinting')
            else:
                subplt.scatter(initial_point_ac[i][0],initial_point_ac[i][1], color='royalblue', s=40, marker='|', zorder=3)
                subplt.scatter(preds_ac_clean[i],y_ticks_ac[int(-(i+1)/2)], color='royalblue', s=20, marker='^', zorder=3)
                subplt.plot([initial_point_ac[i][0],preds_ac_clean[i]],[initial_point_ac[i][1],y_ticks_ac[int(-(i+1)/2)]], color='royalblue', linewidth=1.5, alpha=0.5, label='Degenerate Fingerprinting')

    if p == 2:

        ss_dev = [ss[i] for i in index_dev]
        initial_point_dev = [initial_point[i] for i in index_dev]
        preds_dev_clean = [preds_dev[i] for i in index_dev]
        y_ticks_dev = np.linspace(ymax,ymin,len(ss_dev)+2)

        for i in range(len(ss_dev)):
            if i % 2 == 0:
                subplt.scatter(initial_point_dev[i][0],initial_point_dev[i][1], color='forestgreen', s=40, marker='|', zorder=3)
                subplt.scatter(preds_dev_clean[i],y_ticks_dev[int(i/2)], color='forestgreen', s=20, marker='s', zorder=3)
                subplt.plot([initial_point_dev[i][0],preds_dev_clean[i]],[initial_point_dev[i][1],y_ticks_dev[int(i/2)]], color='forestgreen', linewidth=1.5, alpha=0.5, label='DEV')
            else:
                subplt.scatter(initial_point_dev[i][0],initial_point_dev[i][1], color='forestgreen', s=40, marker='|', zorder=3)
                subplt.scatter(preds_dev_clean[i],y_ticks_dev[int(-(i+1)/2)], color='forestgreen', s=20, marker='s', zorder=3)
                subplt.plot([initial_point_dev[i][0],preds_dev_clean[i]],[initial_point_dev[i][1],y_ticks_dev[int(-(i+1)/2)]], color='forestgreen', linewidth=1.5, alpha=0.5, label='DEV')

    subplt.set_title(title[3+p],loc='left',fontdict=font_title)
    subplt.set_xticks([0,1,ground_truth])
    subplt.xaxis.set_major_formatter(FuncFormatter(custom_formatter))
    subplt.tick_params(axis='both', labelsize=10)
    subplt.set_xlabel('Voltage (40mV/s)',font_axis)
    subplt.set_ylabel('Acoustic pressure (Pa)',font_axis)

for p in range(3):

    ss = ss_60
    preds_dl = preds_dl_60
    preds_ac = preds_ac_60
    preds_dev = preds_dev_60
    initial_point = initial_point_60
    ground_truth = ground_truth_60

    index_ac = index_ac_60  
    index_dev = index_dev_60

    txt_3 = open('../empirical_test/empirical_data/thermoacoustic_60mv.txt')
    datalines_3 = txt_3.readlines()

    dataset_3 = []

    for data in datalines_3:
        data = data.strip().split('\t')
        dataset_3.append(data)

    for data in dataset_3:
        data[0]=(float(data[0])-3.65630991104181150E+9)*0.06
        data[1]=float(data[1])*1000/0.2175

    dataset_3 = np.array(dataset_3)
    dataset_3 = dataset_3[::20]

    x_3 = dataset_3[:,0]
    y_3 = dataset_3[:,1]

    subplt = axs[2,p]

    subplt.scatter(x_3,y_3,c='black',s=0.5)
    subplt.axvline(ground_truth,color='slategray',linestyle='--',label='Ground Truth')

    ymin, ymax = subplt.get_ylim()
    y_ticks = np.linspace(ymax,ymin,len(ss)+2)

    if p == 0:
        for i in range(len(ss)):
            if i % 2 == 0:
                subplt.scatter(initial_point[i][0],initial_point[i][1], color='crimson', s=40, marker='|',zorder=3)
                subplt.scatter(preds_dl[i],y_ticks[int(i/2)], color='crimson', s=20, marker='o',zorder=3)
                subplt.plot([initial_point[i][0],preds_dl[i]],[initial_point[i][1],y_ticks[int(i/2)]], color='crimson', linewidth=1.5, alpha=0.5, label='DL Algorithm')
            else:
                subplt.scatter(initial_point[i][0],initial_point[i][1], color='crimson', s=40, marker='|',zorder=3)
                subplt.scatter(preds_dl[i],y_ticks[int(-(i+1)/2)], color='crimson', s=20, marker='o',zorder=3)
                subplt.plot([initial_point[i][0],preds_dl[i]],[initial_point[i][1],y_ticks[int(-(i+1)/2)]], color='crimson', linewidth=1.5, alpha=0.5, label='DL Algorithm')

    if p == 1:

        ss_ac = [ss[i] for i in index_ac]
        initial_point_ac = [initial_point[i] for i in index_ac]
        preds_ac_clean = [preds_ac[i] for i in index_ac]
        y_ticks_ac = np.linspace(ymax,ymin,len(ss_ac)+2)

        for i in range(len(ss_ac)):
            if i % 2 == 0:
                subplt.scatter(initial_point_ac[i][0],initial_point_ac[i][1], color='royalblue', s=40, marker='|', zorder=3)
                subplt.scatter(preds_ac_clean[i],y_ticks_ac[int(i/2)], color='royalblue', s=20, marker='^', zorder=3)
                subplt.plot([initial_point_ac[i][0],preds_ac_clean[i]],[initial_point_ac[i][1],y_ticks_ac[int(i/2)]], color='royalblue', linewidth=1.5, alpha=0.5, label='Degenerate Fingerprinting')
            else:
                subplt.scatter(initial_point_ac[i][0],initial_point_ac[i][1], color='royalblue', s=40, marker='|', zorder=3)
                subplt.scatter(preds_ac_clean[i],y_ticks_ac[int(-(i+1)/2)], color='royalblue', s=20, marker='^', zorder=3)
                subplt.plot([initial_point_ac[i][0],preds_ac_clean[i]],[initial_point_ac[i][1],y_ticks_ac[int(-(i+1)/2)]], color='royalblue', linewidth=1.5, alpha=0.5, label='Degenerate Fingerprinting')

    if p == 2:

        ss_dev = [ss[i] for i in index_dev]
        initial_point_dev = [initial_point[i] for i in index_dev]
        preds_dev_clean = [preds_dev[i] for i in index_dev]
        y_ticks_dev = np.linspace(ymax,ymin,len(ss_dev)+2)

        for i in range(len(ss_dev)):
            if i % 2 == 0:
                subplt.scatter(initial_point_dev[i][0],initial_point_dev[i][1], color='forestgreen', s=40, marker='|', zorder=3)
                subplt.scatter(preds_dev_clean[i],y_ticks_dev[int(i/2)], color='forestgreen', s=20, marker='s', zorder=3)
                subplt.plot([initial_point_dev[i][0],preds_dev_clean[i]],[initial_point_dev[i][1],y_ticks_dev[int(i/2)]], color='forestgreen', linewidth=1.5, alpha=0.5, label='DEV')
            else:
                subplt.scatter(initial_point_dev[i][0],initial_point_dev[i][1], color='forestgreen', s=40, marker='|', zorder=3)
                subplt.scatter(preds_dev_clean[i],y_ticks_dev[int(-(i+1)/2)], color='forestgreen', s=20, marker='s', zorder=3)
                subplt.plot([initial_point_dev[i][0],preds_dev_clean[i]],[initial_point_dev[i][1],y_ticks_dev[int(-(i+1)/2)]], color='forestgreen', linewidth=1.5, alpha=0.5, label='DEV')

    subplt.set_title(title[6+p],loc='left',fontdict=font_title)
    subplt.set_xticks([0,1,ground_truth])
    subplt.xaxis.set_major_formatter(FuncFormatter(custom_formatter))
    subplt.tick_params(axis='both', labelsize=10)
    subplt.set_xlabel('Voltage (60mV/s)',font_axis)
    subplt.set_ylabel('Acoustic pressure (Pa)',font_axis)

plt.subplots_adjust(top=0.92, bottom=0.06, left=0.07, right=0.99, hspace=0.32, wspace=0.3)

legend_ac = mlines.Line2D([], [], color='royalblue', marker='^', markersize=5, label='Degenerate Fingerprinting', linestyle='-')
legend_dev = mlines.Line2D([], [], color='forestgreen', marker='s', markersize=5, label='DEV', linestyle='-')
legend_dl = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='DL Algorithm', linestyle='-')
legend_gt = mlines.Line2D([], [], color='slategray', label='Ground Truth', linestyle='--')

handles = [legend_dl, legend_ac, legend_dev, legend_gt]
labels = [h.get_label() for h in handles]

fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,1), ncol=4, frameon=False, fontsize=15)

plt.savefig('../figures/SFIG7.pdf',format='pdf',dpi=600)
