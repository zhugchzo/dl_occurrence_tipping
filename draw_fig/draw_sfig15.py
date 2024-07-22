import pandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)

# Get the directory containing the current file
current_dir = os.path.dirname(current_file_path)

# Change the working directory to the directory of the current file
os.chdir(current_dir)

seq_len = 500
font_0 = {'family':'Times New Roman','weight':'bold','size': 12}
font_1 = {'family':'Times New Roman','weight':'normal','size': 10}
font = {'family':'Times New Roman','weight':'normal','size': 12}

df_dl = pandas.read_csv('../results/empirical_nus_results_dl.csv')
df_ac = pandas.read_csv('../results/empirical_nus_results_ac.csv')
df_dev = pandas.read_csv('../results/empirical_nus_results_dev.csv')
df_null = pandas.read_csv('../results/empirical_nus_results_null.csv')

# dl
preds_dl_1 = df_dl['preds_20'].values
preds_dl_2 = df_dl['preds_21'].values
preds_dl_3 = df_dl['preds_22'].values

preds_dl_1 = list(preds_dl_1)
preds_dl_2 = list(preds_dl_2)
preds_dl_3 = list(preds_dl_3)

# ac
preds_ac_1 = df_ac['preds_20'].values
preds_ac_2 = df_ac['preds_21'].values
preds_ac_3 = df_ac['preds_22'].values

preds_ac_1 = list(preds_ac_1)
preds_ac_2 = list(preds_ac_2)
preds_ac_3 = list(preds_ac_3)

# dev
preds_dev_1 = df_dev['preds_20'].values
preds_dev_2 = df_dev['preds_21'].values
preds_dev_3 = df_dev['preds_22'].values

preds_dev_1 = list(preds_dev_1)
preds_dev_2 = list(preds_dev_2)
preds_dev_3 = list(preds_dev_3)

# null model
preds_null_1 = df_null['preds_20'].values
preds_null_2 = df_null['preds_21'].values
preds_null_3 = df_null['preds_22'].values

preds_null_1 = list(preds_null_1)
preds_null_2 = list(preds_null_2)
preds_null_3 = list(preds_null_3)


title = ['A1','B1','C1','A2','B2','C2','A3','B3','C3','A4','B4','C4']

fig, axs = plt.subplots(4, 3, figsize=(12, 9))

thermoacoustic_par_range_list = ['0-1','0.05-1.05','0.1-1.1','0.15-1.15']

for p in range(len(thermoacoustic_par_range_list)):

    par_range = thermoacoustic_par_range_list[p]

    preds_dl = preds_dl_1[p]
    preds_ac = preds_ac_1[p]
    preds_dev = preds_dev_1[p]
    preds_null = preds_null_1[p]

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
    dataset_1 = dataset_1[::200]

    x_1 = dataset_1[:,0]
    y_1 = dataset_1[:,1]

    subplt = axs[p,0]

    if p == 0:

        subplt.scatter(x_1,y_1,c='black',s=0.5)
        subplt.axvline(preds_dl,color='crimson',linestyle='--',label='DL Algorithm')
        subplt.axvline(preds_ac,color='royalblue',linestyle='-.',label='Degenerate Fingerprinting',alpha=0.9)
        subplt.axvline(preds_dev,color='forestgreen',linestyle='-.',label='DEV',alpha=0.9)
        subplt.axvline(preds_null,color='blueviolet',linestyle=':',label='Null Model',alpha=0.9)

    else:

        subplt.scatter(x_1,y_1,c='black',s=0.5)
        subplt.axvline(preds_ac,color='royalblue',linestyle='-.',label='Degenerate Fingerprinting',alpha=0.9)
        subplt.axvline(preds_dev,color='forestgreen',linestyle='-.',label='DEV',alpha=0.9)
        subplt.axvline(preds_null,color='blueviolet',linestyle=':',label='Null Model',alpha=0.9)     
        subplt.axvline(preds_dl,color='crimson',linestyle='--',label='DL Algorithm')

    par_range_min = float(par_range.split('-')[0])
    par_range_max = float(par_range.split('-')[1])
    for i in np.linspace(par_range_min,par_range_max,500):
        subplt.axvline(i,color='silver',alpha=0.02)

    subplt.set_title(title[1+3*p-1],loc='left')
    subplt.set_xticks([par_range_min,par_range_max,2.4])
    subplt.tick_params(axis='both', labelsize=10)
    if p == 3:
        subplt.set_xlabel('Voltage (20mV/s)',font_0)
    subplt.set_ylabel('Acoustic pressure (Pa)',font)

for p in range(len(thermoacoustic_par_range_list)):

    par_range = thermoacoustic_par_range_list[p]

    preds_dl = preds_dl_2[p]
    preds_ac = preds_ac_2[p]
    preds_dev = preds_dev_2[p]
    preds_null = preds_null_2[p]

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
    dataset_2 = dataset_2[::100]

    x_2 = dataset_2[:,0]
    y_2 = dataset_2[:,1]

    subplt = axs[p,1]

    subplt.scatter(x_2,y_2,c='black',s=0.5)
    subplt.axvline(preds_dl,color='crimson',linestyle='--',label='DL Algorithm')
    subplt.axvline(preds_ac,color='royalblue',linestyle='-.',label='Degenerate Fingerprinting',alpha=0.9)
    subplt.axvline(preds_dev,color='forestgreen',linestyle='-.',label='DEV',alpha=0.9)
    subplt.axvline(preds_null,color='blueviolet',linestyle=':',label='Null Model',alpha=0.9)

    par_range_min = float(par_range.split('-')[0])
    par_range_max = float(par_range.split('-')[1])
    for i in np.linspace(par_range_min,par_range_max,500):
        subplt.axvline(i,color='silver',alpha=0.02)

    subplt.set_title(title[2+3*p-1],loc='left')
    subplt.set_xticks([par_range_min,par_range_max,2.4])
    subplt.tick_params(axis='both', labelsize=10)
    if p == 3:
        subplt.set_xlabel('Voltage (40mV/s)',font_0)
    subplt.set_ylabel('Acoustic pressure (Pa)',font)

for p in range(len(thermoacoustic_par_range_list)):

    par_range = thermoacoustic_par_range_list[p]

    preds_dl = preds_dl_3[p]
    preds_ac = preds_ac_3[p]
    preds_dev = preds_dev_3[p]
    preds_null = preds_null_3[p]

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
    dataset_3 = dataset_3[::65]

    x_3 = dataset_3[:,0]
    y_3 = dataset_3[:,1]

    subplt = axs[p,2]

    subplt.scatter(x_3,y_3,c='black',s=0.5)
    subplt.axvline(preds_dl,color='crimson',linestyle='--',label='DL Algorithm')
    subplt.axvline(preds_ac,color='royalblue',linestyle='-.',label='Degenerate Fingerprinting',alpha=0.9)
    subplt.axvline(preds_dev,color='forestgreen',linestyle='-.',label='DEV',alpha=0.9)
    subplt.axvline(preds_null,color='blueviolet',linestyle=':',label='Null Model',alpha=0.9)

    par_range_min = float(par_range.split('-')[0])
    par_range_max = float(par_range.split('-')[1])
    for i in np.linspace(par_range_min,par_range_max,500):
        subplt.axvline(i,color='silver',alpha=0.02)

    subplt.set_title(title[3+3*p-1],loc='left')
    subplt.set_xticks([par_range_min,par_range_max,2.4])
    subplt.tick_params(axis='both', labelsize=10)
    if p == 3:
        subplt.set_xlabel('Voltage (60mV/s)',font_0)
    subplt.set_ylabel('Acoustic pressure (Pa)',font)

plt.subplots_adjust(top=0.93, bottom=0.055, left=0.07, right=0.99, hspace=0.4, wspace=0.5)

handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,1), ncol=4, frameon=False, fontsize=12)

plt.savefig('../figures/SFIG15.png',dpi=600)
