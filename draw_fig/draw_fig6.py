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

df_fold = pandas.read_csv('../results/microcosm_fold.csv')
df_hopf = pandas.read_csv('../results/thermoacoustic_40_hopf.csv')

sample_start_fold = [0, 500, 1000, 5000, 5500, 6000, 6500]
sample_start_hopf = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6]

ground_truth_fold = 1091
ground_truth_hopf = 1.76

initial_point_fold = []
initial_point_hopf = []

end_point_hopf = []

index_ac_hopf =[]
index_dev_hopf =[]

for ss in sample_start_fold:
    df_sims = pandas.read_csv('../empirical_test/data_nus/microcosm_fold/{}/microcosm_fold_interpolate.csv'.format(ss))
    initial_x = df_sims['x'].iloc[0]
    initial_b = df_sims['b'].iloc[0]    

    initial_point_fold.append([initial_b,initial_x])

for ss in sample_start_hopf:
    df_sims = pandas.read_csv('../empirical_test/data_nus/thermoacoustic_40_hopf/{}/thermoacoustic_40_hopf_interpolate.csv'.format(ss))
    initial_x = df_sims['x'].iloc[0]
    initial_b = df_sims['b'].iloc[0]    
    end_b = df_sims['b'].iloc[-1]

    initial_point_hopf.append([initial_b,initial_x])
    end_point_hopf.append(end_b)

# fold
ss_fold = df_fold['ss_list'].values
preds_dl_fold = df_fold['preds_dl_list'].values
preds_ac_fold = df_fold['preds_ac_list'].values
preds_dev_fold = df_fold['preds_dev_list'].values

preds_dl_fold  = list(preds_dl_fold)
preds_ac_fold  = list(preds_ac_fold)
preds_dev_fold  = list(preds_dev_fold)

# hopf 60mv
ss_hopf = df_hopf['ss_list'].values
preds_dl_hopf = df_hopf['preds_dl_list'].values
preds_ac_hopf = df_hopf['preds_ac_list'].values
preds_dev_hopf = df_hopf['preds_dev_list'].values

preds_dl_hopf  = list(preds_dl_hopf)
preds_ac_hopf  = list(preds_ac_hopf)
preds_dev_hopf  = list(preds_dev_hopf)

for i in range(len(preds_ac_hopf)):
    if preds_ac_hopf[i] > end_point_hopf[i] and preds_ac_hopf[i] < 2.5:
        index_ac_hopf.append(i)

for i in range(len(preds_dev_hopf)):
    if preds_dev_hopf[i] > end_point_hopf[i] and preds_dev_hopf[i] < 2.5:
        index_dev_hopf.append(i)

title = ['A1','A2','A3','B1','B2','B3']

fig, axs = plt.subplots(2, 3, figsize=(12, 8))

for p in range(3):

    ss = ss_fold
    preds_dl = preds_dl_fold
    preds_ac = preds_ac_fold
    preds_dev = preds_dev_fold
    initial_point = initial_point_fold
    ground_truth = ground_truth_fold

    txt = open('../empirical_test/empirical_data/microcosm.txt')
    datalines = txt.readlines()
    dataset = []

    for data in datalines:
        data = data.strip().split('\t')
        dataset.append(data)

    clean_NA = []
    for i in range(1,len(dataset)-1):
        if dataset[i][1] != 'NA' and dataset[i+1][1] == 'NA':
            clean_NA.append(i)
        if dataset[i-1][1] == 'NA' and dataset[i][1] != 'NA':
            clean_NA.append(i)

    for i in range(0,len(clean_NA),2):
        indexmin = clean_NA[i]
        indexmax = clean_NA[i+1]
        NA_approximate = np.linspace(float(dataset[indexmin][1]),float(dataset[indexmax][1]),indexmax-indexmin+1)
        for j in range(indexmin,indexmax+1):
            dataset[j][1] = NA_approximate[j-indexmin]

    for data in dataset:
        data[0]=477+float(data[0])*23
        data[1]=float(data[1])

    dataset = np.array(dataset)

    x = dataset[:,0]
    y = dataset[:,1]

    subplt = axs[0,p]

    subplt.scatter(x,y,c='black',s=0.5)
    subplt.axvline(ground_truth,color='slategray',linestyle='--',label='Ground Truth')

    if p == 0:
        for i in range(len(ss)):
            subplt.scatter(initial_point[i][0],initial_point[i][1], color='crimson', s=40, marker='|', zorder=3)
            subplt.scatter(preds_dl[i],initial_point[i][1], color='crimson', s=20, marker='o', zorder=3)
            subplt.plot([initial_point[i][0],preds_dl[i]],[initial_point[i][1],initial_point[i][1]], color='crimson', linewidth=1.5, alpha=0.5, label='DL Algorithm')

    if p == 1:
        for i in range(len(ss)):
            subplt.scatter(initial_point[i][0],initial_point[i][1], color='royalblue', s=40, marker='|', zorder=3)
            subplt.scatter(preds_ac[i],initial_point[i][1], color='royalblue', s=20, marker='^', zorder=3)
            subplt.plot([initial_point[i][0],preds_ac[i]],[initial_point[i][1],initial_point[i][1]], color='royalblue', linewidth=1.5, alpha=0.5, label='Degenerate Fingerprinting')

    if p == 2:
        for i in range(len(ss)):
            subplt.scatter(initial_point[i][0],initial_point[i][1], color='forestgreen', s=40, marker='|', zorder=3)
            subplt.scatter(preds_dev[i],initial_point[i][1], color='forestgreen', s=20, marker='s', zorder=3)
            subplt.plot([initial_point[i][0],preds_dev[i]],[initial_point[i][1],initial_point[i][1]], color='forestgreen', linewidth=1.5, alpha=0.5, label='DEV')

    subplt.set_title(title[p],loc='left',fontdict=font_title)
    subplt.set_xticks([477,800,ground_truth])
    subplt.tick_params(axis='both', labelsize=10)
    subplt.set_xlabel('Light irradiance ($\\mu$mol photons m$^{-2}$ s$^{-1}$)',font_axis,labelpad=2)
    subplt.set_ylabel('Light attenuation coefficient (m$^{-1}$)',font_axis,labelpad=12)

for p in range(3):

    ss = ss_hopf
    preds_dl = preds_dl_hopf
    preds_ac = preds_ac_hopf
    preds_dev = preds_dev_hopf
    initial_point = initial_point_hopf
    ground_truth = ground_truth_hopf

    index_ac = index_ac_hopf  
    index_dev = index_dev_hopf

    txt = open('../empirical_test/empirical_data/thermoacoustic_40mv.txt')
    datalines = txt.readlines()

    dataset = []

    for data in datalines:
        data = data.strip().split('\t')
        dataset.append(data)

    for data in dataset:
        data[0]=(float(data[0])-3.65629963087196300E+9)*0.04
        data[1]=float(data[1])*1000/0.2175

    dataset = np.array(dataset)
    dataset = dataset[::30]

    x = dataset[:,0]
    y = dataset[:,1]

    subplt = axs[1,p]

    subplt.scatter(x,y,c='black',s=0.5)
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
    subplt.set_xlabel('Voltage (V)',font_axis,labelpad=2)
    subplt.set_ylabel('Acoustic pressure (Pa)',font_axis,labelpad=6)

plt.subplots_adjust(top=0.9, bottom=0.07, left=0.07, right=0.99, hspace=0.4, wspace=0.3)

legend_ac = mlines.Line2D([], [], color='royalblue', marker='^', markersize=5, label='Degenerate Fingerprinting', linestyle='-')
legend_dev = mlines.Line2D([], [], color='forestgreen', marker='s', markersize=5, label='DEV', linestyle='-')
legend_dl = mlines.Line2D([], [], color='crimson',  marker='o',markersize=5, label='DL Algorithm', linestyle='-')
legend_gt = mlines.Line2D([], [], color='slategray', label='Ground Truth', linestyle='--')

handles = [legend_dl, legend_ac, legend_dev, legend_gt]
labels = [h.get_label() for h in handles]

fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,1), ncol=4, frameon=False, fontsize=15)

plt.savefig('../figures/FIG6.pdf',format='pdf',dpi=600)
