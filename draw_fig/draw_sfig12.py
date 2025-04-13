import pandas
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
import matplotlib.lines as mlines
from matplotlib import font_manager

fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(14, 10))
axes = axes.flatten()

font_x = font_manager.FontProperties(family='Times New Roman', size=24, weight='normal')
font_y = font_manager.FontProperties(family='Times New Roman', size=24, weight='normal')
times_font = fm.FontProperties(family='Times New Roman', style='normal', size=14)

numtest = 10

# # 每组两张图，一共6组
# group_titles = [f'Group {i+1}' for i in range(6)]

legend_resids = mlines.Line2D([], [], color='crimson', label='Residuals', linestyle='-', alpha=0.5)
legend_b = mlines.Line2D([], [], color='royalblue', label='Parameter', linestyle='-', alpha=0.5)

handles_resids = [legend_resids]
labels_resids = [h.get_label() for h in handles_resids]
handles_b = [legend_b]
labels_b = [h.get_label() for h in handles_b]

# may fold

ax1 = axes[0]
ax2 = axes[1]

for c_rate in [1e-5,2e-5,3e-5,4e-5,5e-5]:

    for n in range(1,numtest+1):

        df_saliency = pandas.read_csv('../results/saliency/may_fold/saliency_feature_{}_{}.csv'.format(c_rate,n))

        col = ['saliency_feature_x','saliency_feature_b']
        
        saliency = df_saliency[col].to_numpy()

        saliency_resids = saliency[:,0]
        saliency_b = saliency[:,1]

        k = len(saliency_resids)

        t = np.linspace(0,1,k)

        ax1.plot(t,saliency_resids,c='crimson',linestyle='-',alpha=0.2)
        ax2.plot(t,saliency_b,c='royalblue',linestyle='-',alpha=0.2)

ax1.set_xticks([0, 1])
ax2.set_xticks([0, 1])
ax1.set_xticklabels(['0', '1'], fontproperties=times_font)
ax2.set_xticklabels(['0', '1'], fontproperties=times_font)

ax1.set_yticks([0, 0.1])
ax2.set_yticks([0, 0.1])
ax1.set_yticklabels(['0', '0.1'], fontproperties=times_font)
ax2.set_yticklabels(['0', '0.1'], fontproperties=times_font)
ax1.set_ylim(-0.005,0.105)
ax2.set_ylim(-0.005,0.105)

# ax1.set_ylabel('Saliency Score',font)

ax1.legend(handles=handles_resids,labels=labels_resids,prop={'size':15},frameon=False)
ax2.legend(handles=handles_b,labels=labels_b,prop={'size':15},frameon=False)

# food hopf

ax3 = axes[2]
ax4 = axes[3]

for c_rate in [1e-5,2e-5,3e-5,4e-5,5e-5]:

    for n in range(1,numtest+1):

        df_saliency = pandas.read_csv('../results/saliency/food_hopf/saliency_feature_{}_{}.csv'.format(c_rate,n))

        col = ['saliency_feature_x','saliency_feature_b']
        
        saliency = df_saliency[col].to_numpy()

        saliency_resids = saliency[:,0]
        saliency_b = saliency[:,1]

        k = len(saliency_resids)

        t = np.linspace(0,1,k)

        ax3.plot(t,saliency_resids,c='crimson',linestyle='-',alpha=0.2)
        ax4.plot(t,saliency_b,c='royalblue',linestyle='-',alpha=0.2)

ax3.set_xticks([0, 1])
ax4.set_xticks([0, 1])
ax3.set_xticklabels(['0', '1'], fontproperties=times_font)
ax4.set_xticklabels(['0', '1'], fontproperties=times_font)

ax3.set_yticks([0, 0.1])
ax4.set_yticks([0, 0.1])
ax3.set_yticklabels(['0', '0.1'], fontproperties=times_font)
ax4.set_yticklabels(['0', '0.1'], fontproperties=times_font)
ax3.set_ylim(-0.005,0.105)
ax4.set_ylim(-0.005,0.105)

ax3.legend(handles=handles_resids,labels=labels_resids,prop={'size':15},frameon=False)
ax4.legend(handles=handles_b,labels=labels_b,prop={'size':15},frameon=False)

# cr branch

ax5 = axes[4]
ax6 = axes[5]

for c_rate in [1e-4,2e-4,3e-4,4e-4,5e-4]:

    for n in range(1,numtest+1):

        df_saliency = pandas.read_csv('../results/saliency/cr_branch/saliency_feature_{}_{}.csv'.format(c_rate,n))

        col = ['saliency_feature_x','saliency_feature_b']
        
        saliency = df_saliency[col].to_numpy()

        saliency_resids = saliency[:,0]
        saliency_b = saliency[:,1]

        k = len(saliency_resids)

        t = np.linspace(0,1,k)

        ax5.plot(t,saliency_resids,c='crimson',linestyle='-',alpha=0.2)
        ax6.plot(t,saliency_b,c='royalblue',linestyle='-',alpha=0.2)

ax5.set_xticks([0, 1])
ax6.set_xticks([0, 1])
ax5.set_xticklabels(['0', '1'], fontproperties=times_font)
ax6.set_xticklabels(['0', '1'], fontproperties=times_font)

ax5.set_yticks([0, 0.14])
ax6.set_yticks([0, 0.14])
ax5.set_yticklabels(['0', '0.14'], fontproperties=times_font)
ax6.set_yticklabels(['0', '0.14'], fontproperties=times_font)
ax5.set_ylim(-0.007,0.147)
ax6.set_ylim(-0.007,0.147)

# ax5.set_ylabel('Saliency Score',font,labelpad=-3)

# global fold

ax7 = axes[6]
ax8 = axes[7]

for c_rate in [-5e-7,-6e-7,-7e-7,-8e-7,-9e-7]:

    for n in range(1,numtest+1):

        df_saliency = pandas.read_csv('../results/saliency/global_fold/saliency_feature_{}_{}.csv'.format(c_rate,n))

        col = ['saliency_feature_x','saliency_feature_b']
        
        saliency = df_saliency[col].to_numpy()

        saliency_resids = saliency[:,0]
        saliency_b = saliency[:,1]

        k = len(saliency_resids)

        t = np.linspace(0,1,k)

        ax7.plot(t,saliency_resids,c='crimson',linestyle='-',alpha=0.2)
        ax8.plot(t,saliency_b,c='royalblue',linestyle='-',alpha=0.2)

ax7.set_xticks([0, 1])
ax8.set_xticks([0, 1])
ax7.set_xticklabels(['0', '1'], fontproperties=times_font)
ax8.set_xticklabels(['0', '1'], fontproperties=times_font)

ax7.set_yticks([0, 0.12])
ax8.set_yticks([0, 0.12])
ax7.set_yticklabels(['0', '0.12'], fontproperties=times_font)
ax8.set_yticklabels(['0', '0.12'], fontproperties=times_font)
ax7.set_ylim(-0.006,0.126)
ax8.set_ylim(-0.006,0.126)

# MPT hopf

ax9 = axes[8]
ax10 = axes[9]

for c_rate in [1e-5,2e-5,3e-5,4e-5,5e-5]:

    for n in range(1,numtest+1):

        df_saliency = pandas.read_csv('../results/saliency/MPT_hopf/saliency_feature_{}_{}.csv'.format(c_rate,n))

        col = ['saliency_feature_x','saliency_feature_b']
        
        saliency = df_saliency[col].to_numpy()

        saliency_resids = saliency[:,0]
        saliency_b = saliency[:,1]

        k = len(saliency_resids)

        t = np.linspace(0,1,k)

        ax9.plot(t,saliency_resids,c='crimson',linestyle='-',alpha=0.2)
        ax10.plot(t,saliency_b,c='royalblue',linestyle='-',alpha=0.2)

ax9.set_xticks([0, 1])
ax10.set_xticks([0, 1])
ax9.set_xticklabels(['0', '1'], fontproperties=times_font)
ax10.set_xticklabels(['0', '1'], fontproperties=times_font)

ax9.set_yticks([0, 0.12])
ax10.set_yticks([0, 0.12])
ax9.set_yticklabels(['0', '0.12'], fontproperties=times_font)
ax10.set_yticklabels(['0', '0.12'], fontproperties=times_font)
ax9.set_ylim(-0.006,0.126)
ax10.set_ylim(-0.006,0.126)

# ax9.set_ylabel('Saliency Score',font,labelpad=-2)

# amazon branch

ax11 = axes[10]
ax12 = axes[11]

for c_rate in [-1e-5,-2e-5,-3e-5,-4e-5,-5e-5]:

    for n in range(1,numtest+1):

        df_saliency = pandas.read_csv('../results/saliency/amazon_branch/saliency_feature_{}_{}.csv'.format(c_rate,n))

        col = ['saliency_feature_x','saliency_feature_b']
        
        saliency = df_saliency[col].to_numpy()

        saliency_resids = saliency[:,0]
        saliency_b = saliency[:,1]

        k = len(saliency_resids)

        t = np.linspace(0,1,k)

        ax11.plot(t,saliency_resids,c='crimson',linestyle='-',alpha=0.2)
        ax12.plot(t,saliency_b,c='royalblue',linestyle='-',alpha=0.2)

ax11.set_xticks([0, 1])
ax12.set_xticks([0, 1])
ax11.set_xticklabels(['0', '1'], fontproperties=times_font)
ax12.set_xticklabels(['0', '1'], fontproperties=times_font)

ax11.set_yticks([0, 0.1])
ax12.set_yticks([0, 0.1])
ax11.set_yticklabels(['0', '0.1'], fontproperties=times_font)
ax12.set_yticklabels(['0', '0.1'], fontproperties=times_font)
ax11.set_ylim(-0.005,0.105)
ax12.set_ylim(-0.005,0.105)

fig.text(0.3, 0.97, 'May Harvesting Fold Model (1D)', ha='center', va='bottom', fontsize=15, fontweight='bold')
fig.text(0.77, 0.97, 'Chaotic Food Chain Hopf Model (3D)', ha='center', va='bottom', fontsize=15, fontweight='bold')
fig.text(0.28, 0.65, 'Consumer Resource Transcritical Model (2D)', ha='center', va='bottom', fontsize=15, fontweight='bold')
fig.text(0.77, 0.65, 'Global Energy Balance Fold Model (1D)', ha='center', va='bottom', fontsize=15, fontweight='bold')
fig.text(0.28, 0.33, 'Middle Pleistocene Transition Hopf Model (3D)', ha='center', va='bottom', fontsize=15, fontweight='bold')
fig.text(0.77, 0.33, 'Amazon Rainforest Dieback Transcritical Model (1D)', ha='center', va='bottom', fontsize=15, fontweight='bold')


fig.supxlabel('Normalized timepoints',x=0.535, y=0.005, fontproperties=font_x)
fig.supylabel('Saliency score',x=0.005, y=0.5, fontproperties=font_y)

plt.subplots_adjust(top=0.96, bottom=0.07, left=0.055, right=0.99, hspace=0.3, wspace=0.2)
plt.savefig('../figures/SFIG12.pdf',format='pdf',dpi=600)





