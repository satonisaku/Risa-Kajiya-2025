import numpy as np
import scikit_posthocs
from scipy import signal
from scipy import fftpack
from matplotlib import pyplot as plt
import joblib
import os

fig = plt.figure(figsize=(18, 12))
ax1 = fig.add_subplot(2, 3, 1)
ax2 = fig.add_subplot(2, 3, 2)
ax3 = fig.add_subplot(2, 3, 3)
ax4 = fig.add_subplot(2, 3, 4)
ax5 = fig.add_subplot(2, 3, 5)
ax6 = fig.add_subplot(2, 3, 6)


# vCA1
#rats = ['hoegaarden181115', 'innis190601', 'karmeliet190901', 'maredsous200224', 'nostrum200304','oberon200325']

# vCA1
rats = ['duvel190505','hoegaarden181115', 'innis190601','karmeliet190901','leffe200124','maredsous200224', 'nostrum200304','oberon200325']


C=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr, _) = joblib.load(data_dir + rat_name + '_vCA1ex_BLAex_N_N_SWR_corr.joblib')
    (Corr) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_coactivity_all_nrem_swr.joblib')
    C=np.append(C,Corr)

C1=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (Corr) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_coactivity_all_nrem_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_vCA1ex_NN_RR_NR_RN.joblib')
    vCA1_NN = np.where(NN == 1)[0]
    C1=np.append(C1,Corr[vCA1_NN])

C2=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (Corr) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_coactivity_all_nrem_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_vCA1ex_NN_RR_NR_RN.joblib')
    vCA1_RR = np.where(RR == 4)[0]
    C2=np.append(C2,Corr[vCA1_RR])


nr=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (Corr) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_coactivity_all_nrem_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_vCA1ex_NN_RR_NR_RN.joblib')
    vCA1_NR = np.where(NR == 2)[0]
    nr=np.append(nr,Corr[vCA1_NR])

rn=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (Corr) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_coactivity_all_nrem_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_vCA1ex_NN_RR_NR_RN.joblib')
    vCA1_RN = np.where(RN == 2)[0]
    rn=np.append(rn,Corr[vCA1_RN])

C3=[]
C3=np.append(nr,rn)

F_mean=np.nanmean(C1)
L_mean=np.nanmean(C2)
F_std=np.nanstd(C1)
L_std=np.nanstd(C2)
F_sem = F_std / np.sqrt(len(C1))
L_sem = L_std / np.sqrt(len(C2))
C3_mean=np.nanmean(C3)
C3_std=np.nanstd(C3)
C3_sem = C3_std / np.sqrt(len(C3))

means=[F_mean,L_mean,C3_mean]
sem=[F_sem,L_sem,C3_sem]


x=np.arange(3)
labels=('N-N','R-R','N-R')
ax1.bar(x,means,yerr=sem,zorder=10,color=['cornflowerblue','violet','green'])
ax1.set_xticks(x)
ax1.set_xticklabels(labels,fontsize=15)
ax1.set_ylabel('Co-activity Zscore',fontsize=15)
props = {'arrowstyle': '-','linewidth':1}
#ax1.annotate('', xy=(0.2,L_mean*1.02), xytext=(0.8,L_mean*1.02), arrowprops=props)
#ax1.text(0.45,L_mean*1.02 , "**",fontsize=15)
ax1.set_title('vCA1ex_vCA1ex', fontsize=20)

from scipy import stats
result = stats.mannwhitneyu(C1,C2,alternative='two-sided')
print(result.pvalue)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((means,sem), os.path.expanduser(data_dir +'Y_fig_3_e_vCA1ex_2.joblib'), compress=3)

result = stats.kruskal(C1,C2,C3)
print(result.pvalue)
import scikit_posthocs as sp
#perform Nemenyi post-hoc test
sp.posthoc_dunn([C1,C2,C3])
sp.posthoc_dscf([C1,C2,C3])

weights1= np.ones_like(C1) / len(C1)
weights2= np.ones_like(C2) / len(C2)
weights3= np.ones_like(C3) / len(C3)
n1, bins1, patches1 = ax5.hist(C1,bins = np.linspace(min(C1),20), weights=weights1,alpha=0, label='Frequency',color='cornflowerblue')
n2, bins2, patches2 = ax5.hist(C2,bins = np.linspace(min(C2),20), weights=weights2,alpha=0, label='Frequency',color='violet')
n3, bins3, patches3= ax5.hist(C3,bins = np.linspace(min(C3),20), weights=weights3,alpha=0, label='Frequency',color='green')
# 第2軸用値の算出
y2 = np.add.accumulate(n1) / n1.sum()
x2 = np.convolve(bins1, np.ones(2) / 2, mode="same")[1:]
y3 = np.add.accumulate(n2) / n2.sum()
x3 = np.convolve(bins2, np.ones(2) / 2, mode="same")[1:]
y4 = np.add.accumulate(n3) / n3.sum()
x4 = np.convolve(bins3, np.ones(2) / 2, mode="same")[1:]
# 第2軸のプロット
lines = ax4.plot(x2, y2, ls='--', color='cornflowerblue',label='Cumulative ratio')
lines = ax4.plot(x3, y3, ls='--',color='violet',label='Cumulative ratio')
lines = ax4.plot(x4, y4, ls='--',color='green',label='Cumulative ratio')
ax4.grid(visible=False)
ax4.set_xlabel("Co-activity Zscore",fontsize=15)
#ax14.set_ylabel('relative frequency')
ax4.set_ylabel('cumulative ratio',fontsize=15)
ax4.set_title('vCA1ex_vCA1ex', fontsize=20)
#ax5.text(0.95, 0.95, "p=0.01", va='top', ha='right', transform=ax5.transAxes, fontsize=10)

# ks test
from scipy.stats import ks_2samp
result=ks_2samp(C1,C2)
print(result)
result=ks_2samp(C1,C3)
print(result)
result=ks_2samp(C2,C3)
print(result)



# PL5
rats = ['hoegaarden181115','innis190601', 'jever190814','karmeliet190901','leffe200124','maredsous200224','nostrum200304', 'oberon200325']


C1=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (Corr) = joblib.load(data_dir + rat_name + '_Y_PL5ex_coactivity_all_nrem_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_PL5ex_NN_RR_NR_RN.joblib')
    vCA1_NN = np.where(NN == 1)[0]
    C1=np.append(C1,Corr[vCA1_NN])

C2=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (Corr) = joblib.load(data_dir + rat_name + '_Y_PL5ex_coactivity_all_nrem_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_PL5ex_NN_RR_NR_RN.joblib')
    vCA1_RR = np.where(RR == 4)[0]
    C2=np.append(C2,Corr[vCA1_RR])


nr=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (Corr) = joblib.load(data_dir + rat_name + '_Y_PL5ex_coactivity_all_nrem_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_PL5ex_NN_RR_NR_RN.joblib')
    vCA1_NR = np.where(NR == 2)[0]
    nr=np.append(nr,Corr[vCA1_NR])

rn=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (Corr) = joblib.load(data_dir + rat_name + '_Y_PL5ex_coactivity_all_nrem_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_PL5ex_NN_RR_NR_RN.joblib')
    vCA1_RN = np.where(RN == 2)[0]
    rn=np.append(rn,Corr[vCA1_RN])

C3=[]
C3=np.append(nr,rn)


F_mean=np.nanmean(C1)
L_mean=np.nanmean(C2)
F_std=np.nanstd(C1)
L_std=np.nanstd(C2)
F_sem = F_std / np.sqrt(len(C1))
L_sem = L_std / np.sqrt(len(C2))
C3_mean=np.nanmean(C3)
C3_std=np.nanstd(C3)
C3_sem = C3_std / np.sqrt(len(C3))


means=[F_mean,L_mean,C3_mean]
sem=[F_sem,L_sem,C3_sem]


x=np.arange(3)
labels=('N-N','R-R','N-R')
ax2.bar(x,means,yerr=sem,zorder=10,color=['cornflowerblue','violet','green'])
ax2.set_xticks(x)
ax2.set_xticklabels(labels,fontsize=15)
ax2.set_ylabel('Co-activity Zscore',fontsize=15)
props = {'arrowstyle': '-','linewidth':1}
#ax1.annotate('', xy=(0.2,L_mean*1.02), xytext=(0.8,L_mean*1.02), arrowprops=props)
#ax1.text(0.45,L_mean*1.02 , "**",fontsize=15)
ax2.annotate('', xy=(0.2,F_mean*1.01), xytext=(0.8,F_mean*1.01), arrowprops=props)
ax2.text(0.4,F_mean*1.01 , "***",fontsize=15)
ax2.annotate('', xy=(0.2,F_mean*1.06), xytext=(1.8,F_mean*1.06), arrowprops=props)
ax2.text(0.9,F_mean*1.06 , "***",fontsize=15)
ax2.annotate('', xy=(1.2,F_mean*1.01), xytext=(1.8,F_mean*1.01), arrowprops=props)
ax2.text(1.4,F_mean*1.01 , "***",fontsize=15)
ax2.set_title('PL5ex_PL5ex', fontsize=20)

from scipy import stats
result = stats.mannwhitneyu(C1,C2,alternative='two-sided')
print(result.pvalue)

result = stats.kruskal(C1,C2,C3)
print(result.pvalue)
import scikit_posthocs as sp
#perform Nemenyi post-hoc test
#sp.posthoc_dunn([C1,C2,C3])
sp.posthoc_dscf([C1,C2,C3])

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((means,sem), os.path.expanduser(data_dir +'Y_fig_3_e_PL5ex_2.joblib'), compress=3)

weights1= np.ones_like(C1) / len(C1)
weights2= np.ones_like(C2) / len(C2)
weights3= np.ones_like(C3) / len(C3)
n1, bins1, patches1 = ax5.hist(C1,bins = np.linspace(min(C1),20), weights=weights1,alpha=0, label='Frequency',color='cornflowerblue')
n2, bins2, patches2 = ax5.hist(C2,bins = np.linspace(min(C2),20), weights=weights2,alpha=0, label='Frequency',color='violet')
n3, bins3, patches3= ax5.hist(C3,bins = np.linspace(min(C3),20), weights=weights3,alpha=0, label='Frequency',color='green')
# 第2軸用値の算出
y2 = np.add.accumulate(n1) / n1.sum()
x2 = np.convolve(bins1, np.ones(2) / 2, mode="same")[1:]
y3 = np.add.accumulate(n2) / n2.sum()
x3 = np.convolve(bins2, np.ones(2) / 2, mode="same")[1:]
y4 = np.add.accumulate(n3) / n3.sum()
x4 = np.convolve(bins3, np.ones(2) / 2, mode="same")[1:]
# 第2軸のプロット
lines = ax5.plot(x2, y2, ls='--', color='cornflowerblue',label='Cumulative ratio')
lines = ax5.plot(x3, y3, ls='--',color='violet',label='Cumulative ratio')
lines = ax5.plot(x4, y4, ls='--',color='green',label='Cumulative ratio')
ax5.grid(visible=False)
ax5.set_xlabel("Co-activity Zscore",fontsize=15)
#ax14.set_ylabel('relative frequency')
ax5.set_ylabel('cumulative ratio',fontsize=15)
ax5.set_title('PL5ex_PL5ex', fontsize=20)
#ax5.text(0.95, 0.95, "p=0.01", va='top', ha='right', transform=ax5.transAxes, fontsize=10)

# ks test
from scipy.stats import ks_2samp
result=ks_2samp(C1,C2)
print(result)
result=ks_2samp(C1,C3)
print(result)
result=ks_2samp(C2,C3)
print(result)

#n, bins, patches = ax5.hist(C1, color= 'cornflowerblue')
#n, bins, patches = ax5.hist(C2, color= 'violet')
#n, bins, patches = ax5.hist(C3, color= 'green')


#BLA
#rats = ['guiness181002','hoegaarden181115', 'innis190601',  'maredsous200224', 'nostrum200304', 'oberon200325']

# BLA
rats = [ 'duvel190505', 'estrella180808', 'guiness181002', 'hoegaarden181115','innis190601', 'leffe200124', 'maredsous200224', 'nostrum200304', 'oberon200325']


C1=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (Corr) = joblib.load(data_dir + rat_name + '_Y_BLAex_coactivity_all_nrem_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_BLAex_NN_RR_NR_RN.joblib')
    vCA1_NN = np.where(NN == 1)[0]
    C1=np.append(C1,Corr[vCA1_NN])

C2=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (Corr) = joblib.load(data_dir + rat_name + '_Y_BLAex_coactivity_all_nrem_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_BLAex_NN_RR_NR_RN.joblib')
    vCA1_RR = np.where(RR == 4)[0]
    C2=np.append(C2,Corr[vCA1_RR])


nr=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (Corr) = joblib.load(data_dir + rat_name + '_Y_BLAex_coactivity_all_nrem_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_BLAex_NN_RR_NR_RN.joblib')
    vCA1_NR = np.where(NR == 2)[0]
    nr=np.append(nr,Corr[vCA1_NR])

rn=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (Corr) = joblib.load(data_dir + rat_name + '_Y_BLAex_coactivity_all_nrem_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_BLAex_NN_RR_NR_RN.joblib')
    vCA1_RN = np.where(RN == 2)[0]
    rn=np.append(rn,Corr[vCA1_RN])

C3=[]
C3=np.append(nr,rn)


F_mean=np.nanmean(C1)
L_mean=np.nanmean(C2)
F_std=np.nanstd(C1)
L_std=np.nanstd(C2)
F_sem = F_std / np.sqrt(len(C1))
L_sem = L_std / np.sqrt(len(C2))
C3_mean=np.nanmean(C3)
C3_std=np.nanstd(C3)
C3_sem = C3_std / np.sqrt(len(C3))


means=[F_mean,L_mean,C3_mean]
sem=[F_sem,L_sem,C3_sem]


x=np.arange(3)
labels=('N-N','R-R','N-R')
ax3.bar(x,means,yerr=sem,zorder=10,color=['cornflowerblue','violet','green'])
ax3.set_xticks(x)
ax3.set_xticklabels(labels,fontsize=15)
ax3.set_ylabel('Co-activity Zscore',fontsize=15)
props = {'arrowstyle': '-','linewidth':1}
#ax1.annotate('', xy=(0.2,L_mean*1.02), xytext=(0.8,L_mean*1.02), arrowprops=props)
#ax1.text(0.45,L_mean*1.02 , "**",fontsize=15)
ax3.annotate('', xy=(0.2,L_mean*1.01), xytext=(0.8,L_mean*1.01), arrowprops=props)
ax3.text(0.4,L_mean*1.01 , "*",fontsize=15)
ax3.annotate('', xy=(0.2,L_mean*1.06), xytext=(1.8,L_mean*1.06), arrowprops=props)
ax3.text(0.9,L_mean*1.06 , "*",fontsize=15)
ax3.annotate('', xy=(1.2,L_mean*1.01), xytext=(1.8,L_mean*1.01), arrowprops=props)
ax3.text(1.4,L_mean*1.01 , "***",fontsize=15)
ax3.set_title('BLAex_BLAex', fontsize=20)

from scipy import stats
result = stats.mannwhitneyu(C1,C2,alternative='two-sided')
print(result.pvalue)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((means,sem), os.path.expanduser(data_dir +'Y_fig_3_e_BLAex_2.joblib'), compress=3)

result = stats.kruskal(C1,C2,C3)
print(result.pvalue)
import scikit_posthocs as sp
#perform Nemenyi post-hoc test
sp.posthoc_dunn([C1,C2,C3])
sp.posthoc_dscf([C1,C2,C3])

weights1= np.ones_like(C1) / len(C1)
weights2= np.ones_like(C2) / len(C2)
weights3= np.ones_like(C3) / len(C3)
n1, bins1, patches1 = ax5.hist(C1,bins = np.linspace(min(C1),20), weights=weights1,alpha=0, label='Frequency',color='cornflowerblue')
n2, bins2, patches2 = ax5.hist(C2,bins = np.linspace(min(C2),20), weights=weights2,alpha=0, label='Frequency',color='violet')
n3, bins3, patches3= ax5.hist(C3,bins = np.linspace(min(C3),20), weights=weights3,alpha=0, label='Frequency',color='green')
# 第2軸用値の算出
y2 = np.add.accumulate(n1) / n1.sum()
x2 = np.convolve(bins1, np.ones(2) / 2, mode="same")[1:]
y3 = np.add.accumulate(n2) / n2.sum()
x3 = np.convolve(bins2, np.ones(2) / 2, mode="same")[1:]
y4 = np.add.accumulate(n3) / n3.sum()
x4 = np.convolve(bins3, np.ones(2) / 2, mode="same")[1:]
# 第2軸のプロット
lines = ax6.plot(x2, y2, ls='--', color='cornflowerblue',label='Cumulative ratio')
lines = ax6.plot(x3, y3, ls='--',color='violet',label='Cumulative ratio')
lines = ax6.plot(x4, y4, ls='--',color='green',label='Cumulative ratio')
ax6.grid(visible=False)
ax6.set_xlabel("Co-activity Zscore",fontsize=15)
#ax14.set_ylabel('relative frequency')
ax6.set_ylabel('cumulative ratio',fontsize=15)
ax6.set_title('BLAex_BLAex', fontsize=20)
#ax5.text(0.95, 0.95, "p=0.01", va='top', ha='right', transform=ax5.transAxes, fontsize=10)

# ks test
from scipy.stats import ks_2samp
result=ks_2samp(C1,C2)
print(result)
result=ks_2samp(C1,C3)
print(result)
result=ks_2samp(C2,C3)
print(result)


#plt.subplots_adjust(top=0.85,bottom=0.2,hspace=0.35,wspace=0.3,left=0.05,right=0.95)
fig.suptitle('co-activity Zscore in SWR during nrem', fontsize=20)

plt.savefig('test13.pdf')


np.savetxt(r'C:\Users\saton\Dropbox\python\fig4_a_NN.csv', C1, delimiter=",")
#print(np.loadtxt(r'C:\Users\saton\Dropbox\python\fig4_a_NN.csv', delimiter=','))
np.savetxt(r'C:\Users\saton\Dropbox\python\fig4_a_RR.csv', C2, delimiter=",")
np.savetxt(r'C:\Users\saton\Dropbox\python\fig4_a_NR.csv', C3, delimiter=",")
