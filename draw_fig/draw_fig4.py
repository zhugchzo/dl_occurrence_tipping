import pandas
import numpy as np
import matplotlib.pyplot as plt
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

microcosm_par_range_list = ['0-14','1-15','2-16','3-17']

df = pandas.read_csv('../results/empirical_results.csv')

preds_1 = df['preds_1'].values
preds_20 = df['preds_20'].values
preds_21 = df['preds_21'].values
preds_22 = df['preds_22'].values
preds_3 = df['preds_3'].values
preds_4 = df['preds_4'].values

preds_1 = list(preds_1)
preds_20 = list(preds_20)
preds_21 = list(preds_21)
preds_22 = list(preds_22)
preds_3 = list(preds_3)
preds_4 = list(preds_4)

title = ['A1','B1','C1','D1','A2','B2','C2','D2','A3','B3','C3','D3','A4','B4','C4','D4']

plt.figure(figsize=(12,9))

for p in range(len(microcosm_par_range_list)):

    par_range = microcosm_par_range_list[p]

    preds = preds_1[p]

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
        data[0]=(500+float(data[0])*23)/1000
        data[1]=float(data[1])

    dataset = np.array(dataset)

    x = dataset[:,0]
    y = dataset[:,1]

    subplt = plt.subplot(4,4,1+4*p)

    subplt.scatter(x*1000,y,c='black',s=0.5)
    subplt.axvline(preds,color='crimson',linestyle='--')
    par_range_min = 500+float(par_range.split('-')[0])*23
    par_range_max = 500+float(par_range.split('-')[1])*23
    for i in np.linspace(par_range_min,par_range_max,500):
        subplt.axvline(i,color='silver',alpha=0.02)
    subplt.set_title(title[1+4*p-1],loc='left')
    plt.xticks([par_range_min,par_range_max,1150])
    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=10)
    if p == 3:
        plt.xlabel('Light irradiance ($\\mu$mol photons m$^{-2}$ s$^{-1}$)',font_0)
    plt.ylabel('Light attenuation coefficient (m$^{-1}$)',font_1)

thermoacoustic_par_range_list = ['0-1','0.05-1.05','0.1-1.1','0.15-1.15']

for p in range(len(thermoacoustic_par_range_list)):

    par_range = thermoacoustic_par_range_list[p]

    preds_0 = preds_20[p]
    preds_1 = preds_21[p]
    preds_2 = preds_22[p]

    txt_0 = open('../empirical_test/empirical_data/thermoacoustic_20mv.txt')
    datalines_0 = txt_0.readlines()
    txt_1 = open('../empirical_test/empirical_data/thermoacoustic_40mv.txt')
    datalines_1 = txt_1.readlines()
    txt_2 = open('../empirical_test/empirical_data/thermoacoustic_60mv.txt')
    datalines_2 = txt_2.readlines()

    dataset_0 = []
    dataset_1 = []
    dataset_2 = []

    for data in datalines_0:
        data = data.strip().split('\t')
        dataset_0.append(data)

    for data in dataset_0:
        data[0]=(float(data[0])-3.65630914691268160E+9)*0.02
        data[1]=float(data[1])*1000/0.2175

    dataset_0 = np.array(dataset_0)

    for data in datalines_1:
        data = data.strip().split('\t')
        dataset_1.append(data)

    for data in dataset_1:
        data[0]=(float(data[0])-3.65629963087196300E+9)*0.04
        data[1]=float(data[1])*1000/0.2175

    dataset_1 = np.array(dataset_1)

    for data in datalines_2:
        data = data.strip().split('\t')
        dataset_2.append(data)

    for data in dataset_2:
        data[0]=(float(data[0])-3.65630991104181150E+9)*0.06
        data[1]=float(data[1])*1000/0.2175

    dataset_2 = np.array(dataset_2)

    x_0 = dataset_0[:,0]
    y_0 = dataset_0[:,1]
    x_1 = dataset_1[:,0]
    y_1 = dataset_1[:,1]
    x_2 = dataset_2[:,0]
    y_2 = dataset_2[:,1]

    subplt = plt.subplot(4,4,2+4*p)

    subplt.scatter(x_0,y_0,c='black',label='20mv/s',s=0.5)
    subplt.scatter(x_1,y_1,c='sandybrown',label='40mv/s',s=0.5)
    subplt.scatter(x_2,y_2,c='wheat',label='60mv/s',s=0.5)
    subplt.axvline(preds_0,color='royalblue',linestyle='--')
    subplt.axvline(preds_1,color='crimson',linestyle='--')
    subplt.axvline(preds_2,color='forestgreen',linestyle='--')
    par_range_min = float(par_range.split('-')[0])
    par_range_max = float(par_range.split('-')[1])
    for i in np.linspace(par_range_min,par_range_max,500):
        subplt.axvline(i,color='silver',alpha=0.02)
    subplt.set_title(title[2+4*p-1],loc='left')
    plt.xticks([par_range_min,par_range_max,2.4])
    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=10)
    if p == 3:
        plt.xlabel('Voltage (V)',font_0)
    plt.ylabel('Acoustic pressure (Pa)',font)

Mo_par_range_list = ['160-142','159-141','158-140','157-139']

for p in range(len(Mo_par_range_list)):

    par_range = Mo_par_range_list[p]

    preds = preds_3[p]

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

    subplt = plt.subplot(4,4,3+4*p)

    subplt.scatter(x,y,c='black',s=0.5)
    subplt.axvline(preds,color='crimson',linestyle='--')
    par_range_min = -float(par_range.split('-')[0])
    par_range_max = -float(par_range.split('-')[1])
    for i in np.linspace(par_range_min,par_range_max,500):
        subplt.axvline(i,color='silver',alpha=0.02)
    subplt.set_title(title[3+4*p-1],loc='left')
    plt.xticks([par_range_min,par_range_max,-125])
    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=10)
    if p == 3:
        plt.xlabel('Age (kyr BP)',font_0)
    plt.ylabel('Molybdenum (mg/kg)',font)

U_par_range_list = ['300-268','298-266','296-264','294-262']

for p in range(len(U_par_range_list)):

    par_range = U_par_range_list[p]

    preds = preds_4[p]

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

    subplt = plt.subplot(4,4,4+4*p)

    subplt.scatter(x,y,c='black',s=0.5)
    subplt.axvline(preds,color='crimson',linestyle='--')
    par_range_min = -float(par_range.split('-')[0])
    par_range_max = -float(par_range.split('-')[1])
    for i in np.linspace(par_range_min,par_range_max,500):
        subplt.axvline(i,color='silver',alpha=0.02)
    subplt.set_title(title[4+4*p-1],loc='left')
    plt.xticks([par_range_min,par_range_max,-240])
    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=10)
    if p == 3:
        plt.xlabel('Age (kyr BP)',font_0)
    plt.ylabel('Uranium (mg/kg)',font)

plt.tight_layout()
plt.savefig('../figures/FIG4.png',dpi=600)

