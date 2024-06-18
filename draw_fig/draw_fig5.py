import pandas
import numpy as np
import matplotlib.pyplot as plt
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

seq_len = 500
font_0 = {'family':'Times New Roman','weight':'bold','size': 12}
font_1 = {'family':'Times New Roman','weight':'normal','size': 10}
font = {'family':'Times New Roman','weight':'normal','size': 12}

df_dl = pandas.read_csv('../results/empirical_nus_results_dl.csv')
df_ac = pandas.read_csv('../results/empirical_nus_results_ac.csv')
df_dev = pandas.read_csv('../results/empirical_nus_results_dev.csv')
df_null = pandas.read_csv('../results/empirical_nus_results_null.csv')
df_combined = pandas.read_csv('../results/empirical_nus_results_combined.csv')

# dl
preds_dl_1 = df_dl['preds_1'].values
preds_dl_2 = df_dl['preds_20'].values
preds_dl_3 = df_dl['preds_3'].values
preds_dl_4 = df_dl['preds_4'].values

preds_dl_1 = list(preds_dl_1)
preds_dl_2 = list(preds_dl_2)
preds_dl_3 = list(preds_dl_3)
preds_dl_4 = list(preds_dl_4)

# ac
preds_ac_1 = df_ac['preds_1'].values
preds_ac_2 = df_ac['preds_20'].values
preds_ac_3 = df_ac['preds_3'].values
preds_ac_4 = df_ac['preds_4'].values

preds_ac_1 = list(preds_ac_1)
preds_ac_2 = list(preds_ac_2)
preds_ac_3 = list(preds_ac_3)
preds_ac_4 = list(preds_ac_4)

# dev
preds_dev_1 = df_dev['preds_1'].values
preds_dev_2 = df_dev['preds_20'].values
preds_dev_3 = df_dev['preds_3'].values
preds_dev_4 = df_dev['preds_4'].values

preds_dev_1 = list(preds_dev_1)
preds_dev_2 = list(preds_dev_2)
preds_dev_3 = list(preds_dev_3)
preds_dev_4 = list(preds_dev_4)

# null model
preds_null_1 = df_null['preds_1'].values
preds_null_2 = df_null['preds_20'].values
preds_null_3 = df_null['preds_3'].values
preds_null_4 = df_null['preds_4'].values

preds_null_1 = list(preds_null_1)
preds_null_2 = list(preds_null_2)
preds_null_3 = list(preds_null_3)
preds_null_4 = list(preds_null_4)

# combined model
preds_combined_1 = df_combined['preds_1'].values
preds_combined_2 = df_combined['preds_20'].values
preds_combined_3 = df_combined['preds_3'].values
preds_combined_4 = df_combined['preds_4'].values

preds_combined_1 = list(preds_combined_1)
preds_combined_2 = list(preds_combined_2)
preds_combined_3 = list(preds_combined_3)
preds_combined_4 = list(preds_combined_4)

title = ['A1','B1','C1','D1','A2','B2','C2','D2','A3','B3','C3','D3','A4','B4','C4','D4']

fig, axs = plt.subplots(4, 4, figsize=(12, 9))

microcosm_par_range_list = ['0-15','1-16','2-17','3-17.5']

for p in range(len(microcosm_par_range_list)):

    par_range = microcosm_par_range_list[p]

    preds_dl = preds_dl_1[p]
    preds_ac = preds_ac_1[p]
    preds_dev = preds_dev_1[p]
    preds_null = preds_null_1[p]
    preds_combined = preds_combined_1[p]

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
        data[0]=(477+float(data[0])*23)/1000
        data[1]=float(data[1])

    dataset = np.array(dataset)

    x = dataset[:,0]
    y = dataset[:,1]

    subplt = axs[p,0]

    subplt.scatter(x*1000,y,c='black',s=0.5)

    subplt.axvline(preds_dl,color='crimson',linestyle='--',label='DL Algorithm')
    subplt.axvline(preds_ac,color='royalblue',linestyle='-.',alpha=0.9,label='Degenerate Fingerprinting')
    if p == 0:
        rmline = subplt.axvline(477,color='aqua',linestyle='-.',alpha=0.9,label='BB Method')
    subplt.axvline(preds_dev,color='forestgreen',linestyle='-.',alpha=0.9,label='DEV')
    subplt.axvline(preds_null,color='blueviolet',linestyle=':',alpha=0.9,label='Null Model')
    subplt.axvline(preds_combined,color='darkorange',linestyle=':',alpha=0.9,label='Combined Model')

    if p == 3:
        par_range_min = 477+float(par_range.split('-')[0])*23
        par_range_max = 477+float(par_range.split('-')[1])*23

    par_range_min = 477+float(par_range.split('-')[0])*23
    par_range_max = 477+float(par_range.split('-')[1])*23
    for i in np.linspace(par_range_min,par_range_max,500):
        subplt.axvline(i,color='silver',alpha=0.02)
    subplt.set_title(title[1+4*p-1],loc='left')
    subplt.set_xticks([par_range_min,par_range_max,1135])
    subplt.xaxis.set_major_formatter(FuncFormatter(custom_formatter))
    subplt.tick_params(axis='both', labelsize=10)
    if p == 3:
        subplt.set_xlabel('Light irradiance ($\\mu$mol photons m$^{-2}$ s$^{-1}$)',font_0)
    subplt.set_ylabel('Light attenuation coefficient (m$^{-1}$)',font_1)

thermoacoustic_par_range_list = ['0-1','0.05-1.05','0.1-1.1','0.15-1.15']

for p in range(len(thermoacoustic_par_range_list)):

    par_range = thermoacoustic_par_range_list[p]

    preds_dl = preds_dl_2[p]
    preds_ac = preds_ac_2[p]
    preds_dev = preds_dev_2[p]
    preds_null = preds_null_2[p]
    preds_combined = preds_combined_2[p]

    txt = open('../empirical_test/empirical_data/thermoacoustic_20mv.txt')
    datalines = txt.readlines()
    dataset = []
    for data in datalines:
        data = data.strip().split()
        dataset.append(data)

    for data in dataset:
        data[0] = (float(data[0])-3.65630914691268160E+9)*0.02
        data[1] = float(data[1])*1000/0.2175

    dataset = np.array(dataset[::200])

    x = dataset[:,0]
    y = dataset[:,1]

    subplt = axs[p,1]

    subplt.scatter(x,y,c='black',label='40mv/s',s=0.5)

    subplt.axvline(preds_dl,color='crimson',linestyle='--',label='DL Algorithm')
    subplt.axvline(preds_ac,color='royalblue',linestyle='-.',alpha=0.9,label='Degenerate Fingerprinting')
    subplt.axvline(preds_dev,color='forestgreen',linestyle='-.',alpha=0.9,label='DEV')
    subplt.axvline(preds_null,color='blueviolet',linestyle=':',alpha=0.9,label='Null Model')
    subplt.axvline(preds_combined,color='darkorange',linestyle=':',alpha=0.9,label='Combined Model')

    par_range_min = float(par_range.split('-')[0])
    par_range_max = float(par_range.split('-')[1])
    for i in np.linspace(par_range_min,par_range_max,500):
        subplt.axvline(i,color='silver',alpha=0.02)
    subplt.set_title(title[2+4*p-1],loc='left')
    subplt.set_xticks([par_range_min,par_range_max,2.4])
    subplt.xaxis.set_major_formatter(FuncFormatter(custom_formatter))
    subplt.tick_params(axis='both', labelsize=10)
    if p == 3:
        subplt.set_xlabel('Voltage (V)',font_0)
    subplt.set_ylabel('Acoustic pressure (Pa)',font)

Mo_par_range_list = ['160-142','159-141','158-140','157-139']

for p in range(len(Mo_par_range_list)):

    par_range = Mo_par_range_list[p]

    preds_dl = preds_dl_3[p]
    preds_ac = preds_ac_3[p]
    preds_dev = preds_dev_3[p]
    preds_null = preds_null_3[p]
    preds_combined = preds_combined_3[p]

    txt = open('../empirical_test/empirical_data/hypoxia_64PE_Mo_fold.txt')
    datalines = txt.readlines()
    dataset_orgin = []
    for data in datalines:
        data = data.strip().split('\t')
        dataset_orgin.append(data)

    dataset = []
    for data in dataset_orgin:
        data[0]=-float(data[0])
        data[4]=float(data[4])
        dataset.append([data[0],data[4]])

    dataset = np.array(dataset)

    x = dataset[:,0]
    y = dataset[:,1]

    subplt = axs[p,2]

    subplt.scatter(x,y,c='black',s=0.5)

    subplt.axvline(preds_dl,color='crimson',linestyle='--',label='DL Algorithm')
    subplt.axvline(preds_ac,color='aqua',linestyle='-.',alpha=0.9,label='BB Method')
    subplt.axvline(preds_dev,color='forestgreen',linestyle='-.',alpha=0.9,label='DEV')
    subplt.axvline(preds_null,color='blueviolet',linestyle=':',alpha=0.9,label='Null Model')
    subplt.axvline(preds_combined,color='darkorange',linestyle=':',alpha=0.9,label='Combined Model')

    par_range_min = -float(par_range.split('-')[0])
    par_range_max = -float(par_range.split('-')[1])
    for i in np.linspace(par_range_min,par_range_max,500):
        subplt.axvline(i,color='silver',alpha=0.02)
    subplt.set_title(title[3+4*p-1],loc='left')
    subplt.set_xticks([par_range_min,par_range_max,-125])
    subplt.tick_params(axis='both', labelsize=10)
    if p == 3:
        subplt.set_xlabel('Age (kyr BP)',font_0)
    subplt.set_ylabel('Molybdenum (mg/kg)',font)

U_par_range_list = ['300-268','298-266','296-264','294-262']

for p in range(len(U_par_range_list)):

    par_range = U_par_range_list[p]

    preds_dl = preds_dl_4[p]
    preds_ac = preds_ac_4[p]
    preds_dev = preds_dev_4[p]
    preds_null = preds_null_4[p]
    preds_combined = preds_combined_4[p]

    txt = open('../empirical_test/empirical_data/hypoxia_64PE_U_branch.txt')
    datalines = txt.readlines()
    dataset_orgin = []
    for data in datalines:
        data = data.strip().split('\t')
        dataset_orgin.append(data)

    dataset = []
    for data in dataset_orgin:
        data[0]=-float(data[0])
        data[6]=float(data[6])
        dataset.append([data[0],data[6]])

    dataset = np.array(dataset)

    x = dataset[:,0]
    y = dataset[:,1]

    subplt = axs[p,3]

    subplt.scatter(x,y,c='black',s=0.5)

    subplt.axvline(preds_dl,color='crimson',linestyle='--',label='DL Algorithm')
    subplt.axvline(preds_ac,color='aqua',linestyle='-.',alpha=0.9,label='BB Method')
    subplt.axvline(preds_dev,color='forestgreen',linestyle='-.',alpha=0.9,label='DEV')
    subplt.axvline(preds_null,color='blueviolet',linestyle=':',alpha=0.9,label='Null Model')
    subplt.axvline(preds_combined,color='darkorange',linestyle=':',alpha=0.9,label='Combined Model')

    par_range_min = -float(par_range.split('-')[0])
    par_range_max = -float(par_range.split('-')[1])
    for i in np.linspace(par_range_min,par_range_max,500):
        subplt.axvline(i,color='silver',alpha=0.02)
    subplt.set_title(title[4+4*p-1],loc='left')
    subplt.set_xticks([par_range_min,par_range_max,-240])
    subplt.tick_params(axis='both', labelsize=10)
    if p == 3:
        subplt.set_xlabel('Age (kyr BP)',font_0)
    subplt.set_ylabel('Uranium (mg/kg)',font)

plt.subplots_adjust(top=0.9, bottom=0.055, left=0.05, right=0.99, hspace=0.4, wspace=0.4)

handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,1), ncol=3, frameon=False, fontsize=12)

rmline.remove()

plt.savefig('../figures/FIG5.png',dpi=600)

