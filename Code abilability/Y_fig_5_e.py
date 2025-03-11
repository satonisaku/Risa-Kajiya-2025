import numpy as np
import scikit_posthocs
from scipy import signal
from scipy import fftpack
from matplotlib import pyplot as plt
import joblib
import os

def cohens_d(x1, x2):
    n1 = len(x1)
    n2 = len(x2)
    x1_mean = x1.mean()
    x2_mean = x2.mean()
    s1 = x1.std()
    s2 = x2.std()
    s = np.sqrt((n1*np.square(s1)+n2*np.square(s2))/(n1+n2))
    d = np.abs(x1_mean-x2_mean)/s
    return d

plt.rcParams["font.size"] = 20
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

# vCA1 PL5
rats = ['hoegaarden181115','innis190601', 'karmeliet190901', 'leffe200124', 'maredsous200224', 'nostrum200304','oberon200325']

#rats = ['duvel190505']
#rats = ['hoegaarden181115']
A1=[]
A2=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(a_hc0, a2_hc0) = joblib.load(data_dir + rat_name + '_PL5_vCA1_slope_fr_wake.joblib')
    (a_hc0, a2_hc0) = joblib.load(data_dir + rat_name + '_Y_vCA1_PL5_slope_fr_swr_4_N.joblib')
    A1=np.append(A1,a_hc0)
    A2 = np.append(A2, a2_hc0)

A1_mean=np.nanmean(A1)
A2_mean=np.nanmean(A2)

gain_std_nid=np.nanstd(A1)
gain_std_rid=np.nanstd(A2)
sem_nid = gain_std_nid / np.sqrt(len(A1))
sem_rid = gain_std_rid / np.sqrt(len(A2))
mean=[A1_mean,A2_mean]
sem=[sem_nid,sem_rid]

ax1.bar([0], [A1_mean],color='cornflowerblue',label=['nrem'],yerr=sem[0])
ax1.bar([1], [A2_mean],color='violet',label=['rem'],yerr=sem[1])
labels=('PL5_N','PL5_R')
ax1.set_xticks([0,1])
ax1.set_xticklabels(labels, fontsize=30)
ax1.set_yticks(np.arange(0, 0.015,0.005))
#plt.legend(fontsize=10,loc='upper right')
ax1.set_ylabel('vCA1 N/vCA1 N+R vs.PL5 unit fr (r)', fontsize=20)
props = {'arrowstyle': '-','linewidth':1}
ax1.annotate('', xy=(0.2,A1_mean*1.01), xytext=(0.8,A1_mean*1.01), arrowprops=props)
ax1.text(0.43,A1_mean*1.01 , "*", fontsize=30)
ax1.set_title('PL5ex', fontsize=30)

from scipy import stats
result = stats.mannwhitneyu(A1,A2,alternative='two-sided')
print(result.pvalue)
#fig.suptitle('SWR-BLAex', fontsize=20)

r=stats.wilcoxon(A1,np.zeros(A1.shape[0]),alternative='two-sided')
print(r)
r=stats.wilcoxon(A2,np.zeros(A2.shape[0]),alternative='two-sided')
print(r)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((mean,sem), os.path.expanduser(data_dir +'Y_fig_5_e_PL5ex.joblib'), compress=3)

print(cohens_d(A1, A2))

# vCA1 BLA
rats = ['duvel190505','hoegaarden181115', 'innis190601',  'leffe200124', 'maredsous200224', 'nostrum200304','oberon200325']

#rats = ['duvel190505']
A1=[]
A2=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (a_hc0, a2_hc0) = joblib.load(data_dir + rat_name + '_Y_vCA1_BLA_slope_fr_swr_4_N.joblib')
    A1=np.append(A1,a_hc0)
    A2 = np.append(A2, a2_hc0)

A1_mean=np.nanmean(A1)
A2_mean=np.nanmean(A2)


gain_std_nid=np.nanstd(A1)
gain_std_rid=np.nanstd(A2)
sem_nid = gain_std_nid / np.sqrt(len(A1))
sem_rid = gain_std_rid / np.sqrt(len(A2))
mean=[A1_mean,A2_mean]
sem=[sem_nid,sem_rid]

ax2.bar([0], [A1_mean],color='cornflowerblue',label=['nrem'],yerr=sem[0])
ax2.bar([1], [A2_mean],color='violet',label=['rem'],yerr=sem[1])
labels=('BLA_N','BLA_R')
ax2.set_xticks([0,1])
ax2.set_xticklabels(labels, fontsize=30)
#ax2.set_yticks(np.arange(0, 0.5,0.1))
#plt.legend(fontsize=10,loc='upper right')
ax2.set_ylabel('vCA1 N/vCA1 N+R vs.BLA unit fr (r)', fontsize=20)
props = {'arrowstyle': '-','linewidth':1}
ax2.annotate('', xy=(0.2,A1_mean*1.01), xytext=(0.8,A1_mean*1.01), arrowprops=props)
ax2.text(0.4,A1_mean*1.01 , "***", fontsize=30)
ax2.set_title('BLAex', fontsize=30)

from scipy import stats
result = stats.mannwhitneyu(A1,A2,alternative='two-sided')
print(result.pvalue)
#fig.suptitle('SWR-BLAex', fontsize=20)

r=stats.wilcoxon(A1,np.zeros(A1.shape[0]),alternative='two-sided')
print(r)
r=stats.wilcoxon(A2,np.zeros(A2.shape[0]),alternative='two-sided')
print(r)

#plt.subplots_adjust(hspace=0.4,wspace=0.3,left=0.05,right=0.95)
#fig.suptitle('co participation each cell', fontsize=20)

plt.subplots_adjust(hspace=0.4,wspace=0.4)
#plt.savefig('test9.pdf')

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((mean,sem), os.path.expanduser(data_dir +'Y_fig_5_e_BLAex.joblib'), compress=3)



print(cohens_d(A1, A2))
import numpy as np
import scikit_posthocs
from scipy import signal
from scipy import fftpack
from matplotlib import pyplot as plt
import joblib
import os

plt.rcParams["font.size"] = 20
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

#HFO
# BLA vCA1
rats = ['duvel190505','hoegaarden181115', 'innis190601',  'leffe200124', 'maredsous200224', 'nostrum200304','oberon200325']

#rats = ['duvel190505']
A1=[]
A2=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (a_hc0, a2_hc0) = joblib.load(data_dir + rat_name + '_Y_BLA_vCA1_slope_fr_hfo_4_N.joblib')
    A1=np.append(A1,a_hc0)
    A2 = np.append(A2, a2_hc0)

A1_mean=np.nanmean(A1)
A2_mean=np.nanmean(A2)

gain_std_nid=np.nanstd(A1)
gain_std_rid=np.nanstd(A2)
sem_nid = gain_std_nid / np.sqrt(len(A1))
sem_rid = gain_std_rid / np.sqrt(len(A2))
mean=[A1_mean,A2_mean]
sem=[sem_nid,sem_rid]

ax1.bar([0], [A1_mean],color='cornflowerblue',label=['nrem'],yerr=sem[0])
ax1.bar([1], [A2_mean],color='violet',label=['rem'],yerr=sem[1])
labels=('vCA1_N','vCA1_R')
ax1.set_xticks([0,1])
ax1.set_xticklabels(labels, fontsize=30)
#ax1.set_yticks(np.arange(0, 0.5,0.1))
#plt.legend(fontsize=10,loc='upper right')
ax1.set_ylabel('BLA N/BLA N+R vs.vCA1 unit fr (r)', fontsize=20)
props = {'arrowstyle': '-','linewidth':1}
ax1.annotate('', xy=(0.2,A1_mean+0.01), xytext=(0.9,A1_mean+0.01), arrowprops=props)
ax1.text(0.43,A1_mean+0.01 , "**", fontsize=30)
ax1.set_title('vCA1ex', fontsize=30)

from scipy import stats
result = stats.mannwhitneyu(A1,A2,alternative='two-sided')
print(result.pvalue)
#fig.suptitle('SWR-BLAex', fontsize=20)

r=stats.wilcoxon(A1,np.zeros(A1.shape[0]),alternative='two-sided')
print(r)
r=stats.wilcoxon(A2,np.zeros(A2.shape[0]),alternative='two-sided')
print(r)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((mean,sem), os.path.expanduser(data_dir +'Y_fig_5_g_vCA1ex.joblib'), compress=3)

print(cohens_d(A1, A2))
# PL5_BLA
rats = ['hoegaarden181115', 'innis190601', 'leffe200124', 'maredsous200224', 'nostrum200304', 'oberon200325']


A1=[]
A2=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (a_hc0, a2_hc0) = joblib.load(data_dir + rat_name + '_Y_BLA_PL5_slope_fr_hfo_4_N.joblib')
    A1=np.append(A1,a_hc0)
    A2 = np.append(A2, a2_hc0)


A1_mean=np.nanmean(A1)
A2_mean=np.nanmean(A2)

gain_std_nid=np.nanstd(A1)
gain_std_rid=np.nanstd(A2)
sem_nid = gain_std_nid / np.sqrt(len(A1))
sem_rid = gain_std_rid / np.sqrt(len(A2))
mean=[A1_mean,A2_mean]
sem=[sem_nid,sem_rid]

ax2.bar([0], [A1_mean],color='cornflowerblue',label=['nrem'],yerr=sem[0])
ax2.bar([1], [A2_mean],color='violet',label=['rem'],yerr=sem[1])
labels=('PL5_N','PL5_R')
ax2.set_xticks([0,1])
ax2.set_xticklabels(labels, fontsize=30)
#ax2.set_yticks(np.arange(0, 0.5,0.1))
#plt.legend(fontsize=10,loc='upper right')
ax2.set_ylabel('BLA N/BLA N+R vs.PL5 unit fr (r)', fontsize=20)
props = {'arrowstyle': '-','linewidth':1}
ax2.annotate('', xy=(0.2,A1_mean*1.01), xytext=(0.9,A1_mean*1.01), arrowprops=props)
ax2.text(0.43,A1_mean*1.01 , "**", fontsize=30)
ax2.set_title('PL5ex', fontsize=30)

from scipy import stats
result = stats.mannwhitneyu(A1,A2,alternative='two-sided')
print(result.pvalue)

r=stats.wilcoxon(A1,np.zeros(A1.shape[0]),alternative='two-sided')
print(r)
r=stats.wilcoxon(A2,np.zeros(A2.shape[0]),alternative='two-sided')
print(r)

#fig.suptitle('SWR-BLAex', fontsize=20)
plt.subplots_adjust(hspace=0.4,wspace=0.4)
#plt.savefig('test9.pdf')

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((mean,sem), os.path.expanduser(data_dir +'Y_fig_5_g_PL5ex.joblib'), compress=3)

print(cohens_d(A1, A2))
import numpy as np
import scikit_posthocs
from scipy import signal
from scipy import fftpack
from matplotlib import pyplot as plt
import joblib
import os

plt.rcParams["font.size"] = 20
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)


plt.subplots_adjust(hspace=0.4,wspace=0.4)
#spindle

# PL5 vCA1
rats = ['hoegaarden181115', 'innis190601', 'karmeliet190901', 'leffe200124', 'maredsous200224', 'nostrum200304', 'oberon200325']


A1=[]
A2=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (a_hc0, a2_hc0) = joblib.load(data_dir + rat_name + '_Y_PL5_vCA1_slope_fr_spindle_4_N.joblib')
    A1=np.append(A1,a_hc0)
    A2 = np.append(A2, a2_hc0)

A1_mean=np.nanmean(A1)
A2_mean=np.nanmean(A2)

gain_std_nid=np.nanstd(A1)
gain_std_rid=np.nanstd(A2)
sem_nid = gain_std_nid / np.sqrt(len(A1))
sem_rid = gain_std_rid / np.sqrt(len(A2))
mean=[A1_mean,A2_mean]
sem=[sem_nid,sem_rid]

ax1.bar([0], [A1_mean],color='cornflowerblue',label=['nrem'],yerr=sem[0])
ax1.bar([1], [A2_mean],color='violet',label=['rem'],yerr=sem[1])
labels=('vCA1_N','vCA1_R')
ax1.set_xticks([0,1])
ax1.set_xticklabels(labels, fontsize=30)
#ax1.set_yticks(np.arange(0, 0.5,0.1))
#plt.legend(fontsize=10,loc='upper right')
ax1.set_ylabel('PL5 N/PL5 N+R vs.vCA1 unit fr (r)', fontsize=20)
props = {'arrowstyle': '-','linewidth':1}
ax1.annotate('', xy=(0.2,A1_mean+0.01), xytext=(0.9,A1_mean+0.01), arrowprops=props)
ax1.text(0.5,A1_mean+0.01 , "*", fontsize=30)
ax1.set_title('vCA1ex', fontsize=30)

from scipy import stats
result = stats.mannwhitneyu(A1,A2,alternative='two-sided')
print(result.pvalue)
#fig.suptitle('SWR-BLAex', fontsize=20)

r=stats.wilcoxon(A1,np.zeros(A1.shape[0]),alternative='two-sided')
print(r)
r=stats.wilcoxon(A2,np.zeros(A2.shape[0]),alternative='two-sided')
print(r)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((mean,sem), os.path.expanduser(data_dir +'Y_fig_5_f_vCA1ex.joblib'), compress=3)

print(cohens_d(A1, A2))

# PL5 BLA
rats = ['hoegaarden181115', 'innis190601', 'leffe200124', 'maredsous200224', 'nostrum200304', 'oberon200325']

A1=[]
A2=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (a_hc0, a2_hc0) = joblib.load(data_dir + rat_name + '_Y_PL5_BLA_slope_fr_spindle_4_N.joblib')
    A1=np.append(A1,a_hc0)
    A2 = np.append(A2, a2_hc0)


A1_mean=np.nanmean(A1)
A2_mean=np.nanmean(A2)

gain_std_nid=np.nanstd(A1)
gain_std_rid=np.nanstd(A2)
sem_nid = gain_std_nid / np.sqrt(len(A1))
sem_rid = gain_std_rid / np.sqrt(len(A2))
mean=[A1_mean,A2_mean]
sem=[sem_nid,sem_rid]


ax2.bar([0], [A1_mean],color='cornflowerblue',label=['nrem'],yerr=sem[0])
ax2.bar([1], [A2_mean],color='violet',label=['rem'],yerr=sem[1])
labels=('BLA_N','BLA_R')
ax2.set_xticks([0,1])
ax2.set_xticklabels(labels, fontsize=30)
#ax2.set_yticks(np.arange(-1.5, 0.5,0.5))
#plt.legend(fontsize=10,loc='upper right')
ax2.set_ylabel('PL5 N/PL5 N+R vs.BLA unit fr (r)', fontsize=20)
props = {'arrowstyle': '-','linewidth':1}
ax2.annotate('', xy=(0.2,A1_mean*1.05), xytext=(0.9,A1_mean*1.05), arrowprops=props)
ax2.text(0.4,A1_mean*1.05 , "***", fontsize=30)
ax2.set_title('BLAex', fontsize=30)

from scipy import stats
result = stats.mannwhitneyu(A1,A2,alternative='two-sided')
print(result.pvalue)
#fig.suptitle('SWR-BLAex', fontsize=20)

r=stats.wilcoxon(A1,np.zeros(A1.shape[0]),alternative='two-sided')
print(r)
r=stats.wilcoxon(A2,np.zeros(A2.shape[0]),alternative='two-sided')
print(r)

#plt.savefig('test9.pdf')

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((mean,sem), os.path.expanduser(data_dir +'Y_fig_5_f_BLAex.joblib'), compress=3)

print(cohens_d(A1, A2))

import numpy as np
import scikit_posthocs
from scipy import signal
from scipy import fftpack
from matplotlib import pyplot as plt
import joblib
import os

plt.rcParams["font.size"] = 20
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

# crip

# PL5 vCA1
rats = ['hoegaarden181115', 'innis190601', 'karmeliet190901', 'leffe200124', 'maredsous200224', 'nostrum200304', 'oberon200325']

A1=[]
A2=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (a_hc0, a2_hc0) = joblib.load(data_dir + rat_name + '_Y_PL5_vCA1_slope_fr_crip_4_N.joblib')
    A1=np.append(A1,a_hc0)
    A2 = np.append(A2, a2_hc0)

A1_mean=np.nanmean(A1)
A2_mean=np.nanmean(A2)
gain_std_nid=np.nanstd(A1)
gain_std_rid=np.nanstd(A2)
sem_nid = gain_std_nid / np.sqrt(len(A1))
sem_rid = gain_std_rid / np.sqrt(len(A2))
sem=[sem_nid,sem_rid]
mean=[A1_mean,A2_mean]

ax1.bar([0], [A1_mean],color='cornflowerblue',label=['nrem'],yerr=sem[0])
ax1.bar([1], [A2_mean],color='violet',label=['rem'],yerr=sem[1])
labels=('vCA1_N','vCA1_R')
ax1.set_xticks([0,1])
ax1.set_xticklabels(labels, fontsize=30)
#ax1.set_yticks(np.arange(0, 0.5,0.1))
#plt.legend(fontsize=10,loc='upper right')
ax1.set_ylabel('PL5 N/PL5 N+R vs.vCA1 unit fr (r)', fontsize=20)
props = {'arrowstyle': '-','linewidth':1}
#ax1.annotate('', xy=(0.2,A1_mean+0.01), xytext=(0.8,A1_mean+0.01), arrowprops=props)
#ax1.text(0.45,A1_mean+0.01 , "**", fontsize=20)
ax1.set_title('vCA1ex', fontsize=30)

from scipy import stats
result = stats.mannwhitneyu(A1,A2,alternative='two-sided')
print(result.pvalue)
#fig.suptitle('SWR-BLAex', fontsize=20)

r=stats.wilcoxon(A1,np.zeros(A1.shape[0]),alternative='two-sided')
print(r)
r=stats.wilcoxon(A2,np.zeros(A2.shape[0]),alternative='two-sided')
print(r)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((mean,sem), os.path.expanduser(data_dir +'Y_fig_5_h_vCA1ex.joblib'), compress=3)

print(cohens_d(A1, A2))

# PL5 BLA
rats = ['hoegaarden181115', 'innis190601', 'leffe200124', 'maredsous200224', 'nostrum200304', 'oberon200325']

A1=[]
A2=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (a_hc0, a2_hc0) = joblib.load(data_dir + rat_name + '_Y_PL5_BLA_slope_fr_crip_4_N.joblib')
    A1=np.append(A1,a_hc0)
    A2 = np.append(A2, a2_hc0)

A1_mean=np.nanmean(A1)
A2_mean=np.nanmean(A2)

gain_std_nid=np.nanstd(A1)
gain_std_rid=np.nanstd(A2)
sem_nid = gain_std_nid / np.sqrt(len(A1))
sem_rid = gain_std_rid / np.sqrt(len(A2))
sem=[sem_nid,sem_rid]
mean=[A1_mean,A2_mean]

ax2.bar([0], [A1_mean],color='cornflowerblue',label=['nrem'],yerr=sem[0])
ax2.bar([1], [A2_mean],color='violet',label=['rem'],yerr=sem[1])
labels=('BLA_N','BLA_R')
ax2.set_xticks([0,1])
ax2.set_xticklabels(labels, fontsize=30)
#ax2.set_yticks(np.arange(0, 0.5,0.1))
#plt.legend(fontsize=10,loc='upper right')
ax2.set_ylabel('PL5 N/PL5 N+R vs.BLA unit fr (r)', fontsize=20)
props = {'arrowstyle': '-','linewidth':1}
ax2.annotate('', xy=(0.2,A1_mean+0.01), xytext=(0.9,A1_mean+0.01), arrowprops=props)
ax2.text(0.4,A1_mean+0.01 , "***", fontsize=30)
ax2.set_title('BLAex', fontsize=30)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((mean,sem), os.path.expanduser(data_dir +'Y_fig_5_h_BLAex.joblib'), compress=3)


plt.subplots_adjust(hspace=0.4,wspace=0.4)

from scipy import stats
result = stats.mannwhitneyu(A1,A2,alternative='two-sided')
print(result.pvalue)
#fig.suptitle('SWR-BLAex', fontsize=20)

r=stats.wilcoxon(A1,np.zeros(A1.shape[0]),alternative='two-sided')
print(r)
r=stats.wilcoxon(A2,np.zeros(A2.shape[0]),alternative='two-sided')
print(r)
#plt.savefig('test9.pdf')


print(cohens_d(A1, A2))
import numpy as np
import scikit_posthocs
from scipy import signal
from scipy import fftpack
from matplotlib import pyplot as plt
import joblib
import os



#R
plt.rcParams["font.size"] = 20
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

# vCA1 PL5
rats = ['hoegaarden181115','innis190601', 'karmeliet190901', 'leffe200124', 'maredsous200224', 'nostrum200304','oberon200325']

#rats = ['duvel190505']
#rats = ['hoegaarden181115']
A1=[]
A2=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(a_hc0, a2_hc0) = joblib.load(data_dir + rat_name + '_PL5_vCA1_slope_fr_wake.joblib')
    (a_hc0, a2_hc0) = joblib.load(data_dir + rat_name + '_Y_vCA1_PL5_slope_fr_swr_4_R.joblib')
    A1=np.append(A1,a_hc0)
    A2 = np.append(A2, a2_hc0)

A1_mean=np.nanmean(A1)
A2_mean=np.nanmean(A2)

gain_std_nid=np.nanstd(A1)
gain_std_rid=np.nanstd(A2)
sem_nid = gain_std_nid / np.sqrt(len(A1))
sem_rid = gain_std_rid / np.sqrt(len(A2))
mean=[A1_mean,A2_mean]
sem=[sem_nid,sem_rid]

ax1.bar([0], [A1_mean],color='cornflowerblue',label=['nrem'],yerr=sem[0])
ax1.bar([1], [A2_mean],color='violet',label=['rem'],yerr=sem[1])
labels=('PL5_N','PL5_R')
ax1.set_xticks([0,1])
ax1.set_xticklabels(labels, fontsize=30)
ax1.set_yticks(np.arange(0, 0.015,0.005))
#plt.legend(fontsize=10,loc='upper right')
ax1.set_ylabel('vCA1 N/vCA1 N+R vs.PL5 unit fr (r)', fontsize=20)
props = {'arrowstyle': '-','linewidth':1}
ax1.annotate('', xy=(0.2,A1_mean*1.01), xytext=(0.8,A1_mean*1.01), arrowprops=props)
ax1.text(0.43,A1_mean*1.01 , "*", fontsize=30)
ax1.set_title('PL5ex', fontsize=30)

from scipy import stats
result = stats.mannwhitneyu(A1,A2,alternative='two-sided')
print(result.pvalue)
#fig.suptitle('SWR-BLAex', fontsize=20)

print(cohens_d(A1, A2))

# vCA1 BLA
rats = ['duvel190505','hoegaarden181115', 'innis190601',  'leffe200124', 'maredsous200224', 'nostrum200304','oberon200325']

#rats = ['duvel190505']
A1=[]
A2=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (a_hc0, a2_hc0) = joblib.load(data_dir + rat_name + '_Y_vCA1_BLA_slope_fr_swr_4_R.joblib')
    A1=np.append(A1,a_hc0)
    A2 = np.append(A2, a2_hc0)

A1_mean=np.nanmean(A1)
A2_mean=np.nanmean(A2)


gain_std_nid=np.nanstd(A1)
gain_std_rid=np.nanstd(A2)
sem_nid = gain_std_nid / np.sqrt(len(A1))
sem_rid = gain_std_rid / np.sqrt(len(A2))
mean=[A1_mean,A2_mean]
sem=[sem_nid,sem_rid]

ax2.bar([0], [A1_mean],color='cornflowerblue',label=['nrem'],yerr=sem[0])
ax2.bar([1], [A2_mean],color='violet',label=['rem'],yerr=sem[1])
labels=('BLA_N','BLA_R')
ax2.set_xticks([0,1])
ax2.set_xticklabels(labels, fontsize=30)
#ax2.set_yticks(np.arange(0, 0.5,0.1))
#plt.legend(fontsize=10,loc='upper right')
ax2.set_ylabel('vCA1 N/vCA1 N+R vs.BLA unit fr (r)', fontsize=20)
props = {'arrowstyle': '-','linewidth':1}
ax2.annotate('', xy=(0.2,A1_mean*1.01), xytext=(0.8,A1_mean*1.01), arrowprops=props)
ax2.text(0.4,A1_mean*1.01 , "***", fontsize=30)
ax2.set_title('BLAex', fontsize=30)

from scipy import stats
result = stats.mannwhitneyu(A1,A2,alternative='two-sided')
print(result.pvalue)
#fig.suptitle('SWR-BLAex', fontsize=20)

#plt.subplots_adjust(hspace=0.4,wspace=0.3,left=0.05,right=0.95)
#fig.suptitle('co participation each cell', fontsize=20)

plt.subplots_adjust(hspace=0.4,wspace=0.4)
#plt.savefig('test9.pdf')


plt.rcParams["font.size"] = 20
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

#HFO
# BLA vCA1
rats = ['duvel190505','hoegaarden181115', 'innis190601',  'leffe200124', 'maredsous200224', 'nostrum200304','oberon200325']

#rats = ['duvel190505']
A1=[]
A2=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (a_hc0, a2_hc0) = joblib.load(data_dir + rat_name + '_Y_BLA_vCA1_slope_fr_hfo_4_R.joblib')
    A1=np.append(A1,a_hc0)
    A2 = np.append(A2, a2_hc0)

A1_mean=np.nanmean(A1)
A2_mean=np.nanmean(A2)

gain_std_nid=np.nanstd(A1)
gain_std_rid=np.nanstd(A2)
sem_nid = gain_std_nid / np.sqrt(len(A1))
sem_rid = gain_std_rid / np.sqrt(len(A2))
mean=[A1_mean,A2_mean]
sem=[sem_nid,sem_rid]

ax1.bar([0], [A1_mean],color='cornflowerblue',label=['nrem'],yerr=sem[0])
ax1.bar([1], [A2_mean],color='violet',label=['rem'],yerr=sem[1])
labels=('vCA1_N','vCA1_R')
ax1.set_xticks([0,1])
ax1.set_xticklabels(labels, fontsize=30)
#ax1.set_yticks(np.arange(0, 0.5,0.1))
#plt.legend(fontsize=10,loc='upper right')
ax1.set_ylabel('BLA N/BLA N+R vs.vCA1 unit fr (r)', fontsize=20)
props = {'arrowstyle': '-','linewidth':1}
ax1.annotate('', xy=(0.2,A1_mean+0.01), xytext=(0.9,A1_mean+0.01), arrowprops=props)
ax1.text(0.43,A1_mean+0.01 , "**", fontsize=30)
ax1.set_title('vCA1ex', fontsize=30)

from scipy import stats
result = stats.mannwhitneyu(A1,A2,alternative='two-sided')
print(result.pvalue)
#fig.suptitle('SWR-BLAex', fontsize=20)

# PL5_BLA
rats = ['hoegaarden181115', 'innis190601', 'leffe200124', 'maredsous200224', 'nostrum200304', 'oberon200325']


A1=[]
A2=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (a_hc0, a2_hc0) = joblib.load(data_dir + rat_name + '_Y_BLA_PL5_slope_fr_hfo_4_R.joblib')
    A1=np.append(A1,a_hc0)
    A2 = np.append(A2, a2_hc0)


A1_mean=np.nanmean(A1)
A2_mean=np.nanmean(A2)

gain_std_nid=np.nanstd(A1)
gain_std_rid=np.nanstd(A2)
sem_nid = gain_std_nid / np.sqrt(len(A1))
sem_rid = gain_std_rid / np.sqrt(len(A2))
mean=[A1_mean,A2_mean]
sem=[sem_nid,sem_rid]

ax2.bar([0], [A1_mean],color='cornflowerblue',label=['nrem'],yerr=sem[0])
ax2.bar([1], [A2_mean],color='violet',label=['rem'],yerr=sem[1])
labels=('PL5_N','PL5_R')
ax2.set_xticks([0,1])
ax2.set_xticklabels(labels, fontsize=30)
#ax2.set_yticks(np.arange(0, 0.5,0.1))
#plt.legend(fontsize=10,loc='upper right')
ax2.set_ylabel('BLA N/BLA N+R vs.PL5 unit fr (r)', fontsize=20)
props = {'arrowstyle': '-','linewidth':1}
ax2.annotate('', xy=(0.2,A1_mean*1.01), xytext=(0.9,A1_mean*1.01), arrowprops=props)
ax2.text(0.43,A1_mean*1.01 , "**", fontsize=30)
ax2.set_title('PL5ex', fontsize=30)

from scipy import stats
result = stats.mannwhitneyu(A1,A2,alternative='two-sided')
print(result.pvalue)
#fig.suptitle('SWR-BLAex', fontsize=20)
plt.subplots_adjust(hspace=0.4,wspace=0.4)
#plt.savefig('test9.pdf')


import numpy as np
import scikit_posthocs
from scipy import signal
from scipy import fftpack
from matplotlib import pyplot as plt
import joblib
import os

plt.rcParams["font.size"] = 20
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)


plt.subplots_adjust(hspace=0.4,wspace=0.4)
#spindle

# PL5 vCA1
rats = ['hoegaarden181115', 'innis190601', 'karmeliet190901', 'leffe200124', 'maredsous200224', 'nostrum200304', 'oberon200325']


A1=[]
A2=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (a_hc0, a2_hc0) = joblib.load(data_dir + rat_name + '_Y_PL5_vCA1_slope_fr_spindle_4_R.joblib')
    A1=np.append(A1,a_hc0)
    A2 = np.append(A2, a2_hc0)

A1_mean=np.nanmean(A1)
A2_mean=np.nanmean(A2)

gain_std_nid=np.nanstd(A1)
gain_std_rid=np.nanstd(A2)
sem_nid = gain_std_nid / np.sqrt(len(A1))
sem_rid = gain_std_rid / np.sqrt(len(A2))
mean=[A1_mean,A2_mean]
sem=[sem_nid,sem_rid]

ax1.bar([0], [A1_mean],color='cornflowerblue',label=['nrem'],yerr=sem[0])
ax1.bar([1], [A2_mean],color='violet',label=['rem'],yerr=sem[1])
labels=('vCA1_N','vCA1_R')
ax1.set_xticks([0,1])
ax1.set_xticklabels(labels, fontsize=30)
#ax1.set_yticks(np.arange(0, 0.5,0.1))
#plt.legend(fontsize=10,loc='upper right')
ax1.set_ylabel('PL5 N/PL5 N+R vs.vCA1 unit fr (r)', fontsize=20)
props = {'arrowstyle': '-','linewidth':1}
ax1.annotate('', xy=(0.2,A1_mean+0.01), xytext=(0.9,A1_mean+0.01), arrowprops=props)
ax1.text(0.5,A1_mean+0.01 , "*", fontsize=30)
ax1.set_title('vCA1ex', fontsize=30)

from scipy import stats
result = stats.mannwhitneyu(A1,A2,alternative='two-sided')
print(result.pvalue)
#fig.suptitle('SWR-BLAex', fontsize=20)

# PL5 BLA
rats = ['hoegaarden181115', 'innis190601', 'leffe200124', 'maredsous200224', 'nostrum200304', 'oberon200325']

A1=[]
A2=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (a_hc0, a2_hc0) = joblib.load(data_dir + rat_name + '_Y_PL5_BLA_slope_fr_spindle_4_R.joblib')
    A1=np.append(A1,a_hc0)
    A2 = np.append(A2, a2_hc0)


A1_mean=np.nanmean(A1)
A2_mean=np.nanmean(A2)

gain_std_nid=np.nanstd(A1)
gain_std_rid=np.nanstd(A2)
sem_nid = gain_std_nid / np.sqrt(len(A1))
sem_rid = gain_std_rid / np.sqrt(len(A2))
mean=[A1_mean,A2_mean]
sem=[sem_nid,sem_rid]


ax2.bar([0], [A1_mean],color='cornflowerblue',label=['nrem'],yerr=sem[0])
ax2.bar([1], [A2_mean],color='violet',label=['rem'],yerr=sem[1])
labels=('BLA_N','BLA_R')
ax2.set_xticks([0,1])
ax2.set_xticklabels(labels, fontsize=30)
#ax2.set_yticks(np.arange(-1.5, 0.5,0.5))
#plt.legend(fontsize=10,loc='upper right')
ax2.set_ylabel('PL5 N/PL5 N+R vs.BLA unit fr (r)', fontsize=20)
props = {'arrowstyle': '-','linewidth':1}
ax2.annotate('', xy=(0.2,A1_mean*1.05), xytext=(0.9,A1_mean*1.05), arrowprops=props)
ax2.text(0.4,A1_mean*1.05 , "***", fontsize=30)
ax2.set_title('BLAex', fontsize=30)

from scipy import stats
result = stats.mannwhitneyu(A1,A2,alternative='two-sided')
print(result.pvalue)
#fig.suptitle('SWR-BLAex', fontsize=20)

#plt.savefig('test9.pdf')


import numpy as np
import scikit_posthocs
from scipy import signal
from scipy import fftpack
from matplotlib import pyplot as plt
import joblib
import os

plt.rcParams["font.size"] = 20
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

# crip

# PL5 vCA1
rats = ['hoegaarden181115', 'innis190601', 'karmeliet190901', 'leffe200124', 'maredsous200224', 'nostrum200304', 'oberon200325']

A1=[]
A2=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (a_hc0, a2_hc0) = joblib.load(data_dir + rat_name + '_Y_PL5_vCA1_slope_fr_crip_4_R.joblib')
    A1=np.append(A1,a_hc0)
    A2 = np.append(A2, a2_hc0)

A1_mean=np.nanmean(A1)
A2_mean=np.nanmean(A2)
gain_std_nid=np.nanstd(A1)
gain_std_rid=np.nanstd(A2)
sem_nid = gain_std_nid / np.sqrt(len(A1))
sem_rid = gain_std_rid / np.sqrt(len(A2))
sem=[sem_nid,sem_rid]
mean=[A1_mean,A2_mean]

ax1.bar([0], [A1_mean],color='cornflowerblue',label=['nrem'],yerr=sem[0])
ax1.bar([1], [A2_mean],color='violet',label=['rem'],yerr=sem[1])
labels=('vCA1_N','vCA1_R')
ax1.set_xticks([0,1])
ax1.set_xticklabels(labels, fontsize=30)
#ax1.set_yticks(np.arange(0, 0.5,0.1))
#plt.legend(fontsize=10,loc='upper right')
ax1.set_ylabel('PL5 N/PL5 N+R vs.vCA1 unit fr (r)', fontsize=20)
props = {'arrowstyle': '-','linewidth':1}
#ax1.annotate('', xy=(0.2,A1_mean+0.01), xytext=(0.8,A1_mean+0.01), arrowprops=props)
#ax1.text(0.45,A1_mean+0.01 , "**", fontsize=20)
ax1.set_title('vCA1ex', fontsize=30)

from scipy import stats
result = stats.mannwhitneyu(A1,A2,alternative='two-sided')
print(result.pvalue)
#fig.suptitle('SWR-BLAex', fontsize=20)


# PL5 BLA
rats = ['hoegaarden181115', 'innis190601', 'leffe200124', 'maredsous200224', 'nostrum200304', 'oberon200325']

A1=[]
A2=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (a_hc0, a2_hc0) = joblib.load(data_dir + rat_name + '_Y_PL5_BLA_slope_fr_crip_4_R.joblib')
    A1=np.append(A1,a_hc0)
    A2 = np.append(A2, a2_hc0)

A1_mean=np.nanmean(A1)
A2_mean=np.nanmean(A2)

gain_std_nid=np.nanstd(A1)
gain_std_rid=np.nanstd(A2)
sem_nid = gain_std_nid / np.sqrt(len(A1))
sem_rid = gain_std_rid / np.sqrt(len(A2))
sem=[sem_nid,sem_rid]
mean=[A1_mean,A2_mean]

ax2.bar([0], [A1_mean],color='cornflowerblue',label=['nrem'],yerr=sem[0])
ax2.bar([1], [A2_mean],color='violet',label=['rem'],yerr=sem[1])
labels=('BLA_N','BLA_R')
ax2.set_xticks([0,1])
ax2.set_xticklabels(labels, fontsize=30)
#ax2.set_yticks(np.arange(0, 0.5,0.1))
#plt.legend(fontsize=10,loc='upper right')
ax2.set_ylabel('PL5 N/PL5 N+R vs.BLA unit fr (r)', fontsize=20)
props = {'arrowstyle': '-','linewidth':1}
ax2.annotate('', xy=(0.2,A1_mean+0.01), xytext=(0.9,A1_mean+0.01), arrowprops=props)
ax2.text(0.4,A1_mean+0.01 , "***", fontsize=30)
ax2.set_title('BLAex', fontsize=30)



plt.subplots_adjust(hspace=0.4,wspace=0.4)

from scipy import stats
result = stats.mannwhitneyu(A1,A2,alternative='two-sided')
print(result.pvalue)
#fig.suptitle('SWR-BLAex', fontsize=20)

#plt.savefig('test9.pdf')

import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import os
from scipy import stats
import random

pdf=PdfPages("test7.pdf")
fignums=plt.get_fignums()
for fignum in fignums:
    plt.figure(fignum)
    pdf.savefig()
pdf.close()