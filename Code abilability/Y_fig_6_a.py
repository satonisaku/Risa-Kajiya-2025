import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import os
from scipy import stats
import seaborn as sns
sns.set_context('poster')

fig = plt.figure(figsize=(18, 5))
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)

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

#vCA1
rats=['duvel190505','hoegaarden181115','innis190601','karmeliet190901','leffe200124','maredsous200224','nostrum200304','oberon200325']


data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
(all_time_N_id,all_time_R_id)=joblib.load(data_dir + 'Y_vCA1ex_n_r_id_2_50_2.joblib')
NNid = np.where(all_time_N_id == 1)
RRid = np.where(all_time_R_id == 2)


N=[]
R=[]
W=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data\\'
    (N_fr,R_fr,W_fr)=joblib.load(data_dir + rat_name + '_Y_vCA1ex_n_r_w.joblib')
    N_cell = np.mean(N_fr, axis=1)
    R_cell=np.mean(R_fr, axis=1)
    W_cell = np.mean(W_fr, axis=1)
    N=np.append(N,N_cell)
    R=np.append(R,R_cell)
    W=np.append(W,W_cell)

# N_id
N_mean_nid=np.nanmean(N[NNid])
R_mean_nid=np.nanmean(R[NNid])
W_mean_nid=np.nanmean(W[NNid])
N_std_nid=np.nanstd(N[NNid])
R_std_nid=np.nanstd(R[NNid])
W_std_nid=np.nanstd(W[NNid])
N_sem_nid = N_std_nid / np.sqrt(len(N))
R_sem_nid = R_std_nid / np.sqrt(len(R))
W_sem_nid = W_std_nid / np.sqrt(len(W))
sem_nid=[N_sem_nid,R_sem_nid,W_sem_nid]
# R_id
N_mean_rid=np.nanmean(N[RRid])
R_mean_rid=np.nanmean(R[RRid])
W_mean_rid=np.nanmean(W[RRid])
N_std_rid=np.nanstd(N[RRid])
R_std_rid=np.nanstd(R[RRid])
W_std_rid=np.nanstd(W[RRid])
N_sem_rid = N_std_rid / np.sqrt(len(N[RRid]))
R_sem_rid = R_std_rid / np.sqrt(len(R[RRid]))
W_sem_rid = W_std_rid / np.sqrt(len(W[RRid]))
sem_rid=[N_sem_rid,R_sem_rid,W_sem_rid]

ax1.bar([1,3,5], [N_mean_nid,R_mean_nid,W_mean_nid],color='cornflowerblue',yerr=sem_nid,label=['nrem'])
ax1.bar([2,4,6], [N_mean_rid,R_mean_rid,W_mean_rid],color='violet',yerr=sem_rid,label=['rem'])
labels=('NREM','REM','Wake')
ax1.set_xticks([1.5,3.5,5.5])
ax1.set_xticklabels(labels, fontsize=15)
ax1.legend(fontsize=15,loc='upper left')
props = {'connectionstyle': 'bar', 'arrowstyle': '-','linewidth':1}
#plt.annotate('', xy=(1,1.73), xytext=(2,1.73), arrowprops=props)
ax1.annotate('', xy=(3,2.1), xytext=(4,2.1), arrowprops=props)
ax1.annotate('', xy=(5,3.9), xytext=(6,3.9), arrowprops=props)
#plt.text(1.38, 1.83, "***")
ax1.text(3.4, 2.3, "**")
ax1.text(5.4, 4.0, "**")
#ax1.set_ylabel('Fr(Hz)')
#ax1.set_title('vCA1ex', fontsize=15)

r1=stats.mannwhitneyu(N[NNid],N[RRid],alternative='two-sided')
r2=stats.mannwhitneyu(R[NNid],R[RRid],alternative='two-sided')
r3=stats.mannwhitneyu(W[NNid],W[RRid],alternative='two-sided')
print(r1)
print(r2)
print(r3)

mean=[N_mean_nid,N_mean_rid,R_mean_nid,R_mean_rid,W_mean_nid,W_mean_rid]
sem=[N_sem_nid,N_sem_rid,R_sem_nid,R_sem_rid,W_sem_nid,W_sem_rid]
data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
#joblib.dump((mean,sem), os.path.expanduser(data_dir +'fig_5_a_vCA1ex.joblib'), compress=3)
joblib.dump((mean,sem), os.path.expanduser(data_dir +'Y_fig_6_a_vCA1ex.joblib'), compress=3)

print(cohens_d(N[NNid], N[RRid]))
print(cohens_d(R[NNid], R[RRid]))
print(cohens_d(W[NNid], W[RRid]))

#PL5
rats=['hoegaarden181115','innis190601','jever190814','karmeliet190901','leffe200124','maredsous200224','nostrum200304','oberon200325']

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
(all_time_N_id,all_time_R_id)=joblib.load(data_dir + 'Y_PL5ex_n_r_id_2_50_2.joblib')
NNid = np.where(all_time_N_id == 1)
RRid = np.where(all_time_R_id == 2)

N=[]
R=[]
W=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data\\'
    (N_fr,R_fr,W_fr)=joblib.load(data_dir + rat_name + '_Y_PL5ex_n_r_w.joblib')

    N_cell = np.mean(N_fr, axis=1)
    R_cell=np.mean(R_fr, axis=1)
    W_cell = np.mean(W_fr, axis=1)
    N=np.append(N,N_cell)
    R=np.append(R,R_cell)
    W=np.append(W,W_cell)


# N_id
N_mean_nid=np.nanmean(N[NNid])
R_mean_nid=np.nanmean(R[NNid])
W_mean_nid=np.nanmean(W[NNid])
N_std_nid=np.nanstd(N[NNid])
R_std_nid=np.nanstd(R[NNid])
W_std_nid=np.nanstd(W[NNid])
N_sem_nid = N_std_nid / np.sqrt(len(N))
R_sem_nid = R_std_nid / np.sqrt(len(R))
W_sem_nid = W_std_nid / np.sqrt(len(W))
sem_nid=[N_sem_nid,R_sem_nid,W_sem_nid]
# R_id
N_mean_rid=np.nanmean(N[RRid])
R_mean_rid=np.nanmean(R[RRid])
W_mean_rid=np.nanmean(W[RRid])
N_std_rid=np.nanstd(N[RRid])
R_std_rid=np.nanstd(R[RRid])
W_std_rid=np.nanstd(W[RRid])
N_sem_rid = N_std_rid / np.sqrt(len(N[RRid]))
R_sem_rid = R_std_rid / np.sqrt(len(R[RRid]))
W_sem_rid = W_std_rid / np.sqrt(len(W[RRid]))
sem_rid=[N_sem_rid,R_sem_rid,W_sem_rid]


ax2.bar([1,3,5], [N_mean_nid,R_mean_nid,W_mean_nid],color='cornflowerblue',yerr=sem_nid,label=['nrem'])
ax2.bar([2,4,6], [N_mean_rid,R_mean_rid,W_mean_rid],color='violet',yerr=sem_rid,label=['rem'])
ax2.set_xticks([1.5,3.5,5.5])
ax2.set_xticklabels(labels, fontsize=15)
props = {'connectionstyle': 'bar', 'arrowstyle': '-','linewidth':1}
ax2.annotate('', xy=(1,2.2), xytext=(2,2.2), arrowprops=props)
ax2.annotate('', xy=(3,2.6), xytext=(4,2.6), arrowprops=props)
ax2.annotate('', xy=(5,2.85), xytext=(6,2.85), arrowprops=props)
ax2.text(1.38, 2.35, "***")
ax2.text(3.38, 2.75, "***")
ax2.text(5.38, 3, "***")
#ax2.set_ylabel('Fr(Hz)')
#ax2.set_title('PL5ex', fontsize=15)

r1=stats.mannwhitneyu(N[NNid],N[RRid],alternative='two-sided')
r2=stats.mannwhitneyu(R[NNid],R[RRid],alternative='two-sided')
r3=stats.mannwhitneyu(W[NNid],W[RRid],alternative='two-sided')
print(r1)
print(r2)
print(r3)

mean=[N_mean_nid,N_mean_rid,R_mean_nid,R_mean_rid,W_mean_nid,W_mean_rid]
sem=[N_sem_nid,N_sem_rid,R_sem_nid,R_sem_rid,W_sem_nid,W_sem_rid]
data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
#joblib.dump((mean,sem), os.path.expanduser(data_dir +'fig_5_a_PL5ex.joblib'), compress=3)
joblib.dump((mean,sem), os.path.expanduser(data_dir +'Y_fig_6_a_PL5ex.joblib'), compress=3)

print(cohens_d(N[NNid], N[RRid]))
print(cohens_d(R[NNid], R[RRid]))
print(cohens_d(W[NNid], W[RRid]))

#BLA
rats=['achel180320','booyah180430','duvel190505','estrella180808','guiness181002','hoegaarden181115','innis190601','jever190814','leffe200124','maredsous200224','nostrum200304','oberon200325']

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
(all_time_N_id,all_time_R_id)=joblib.load(data_dir + 'Y_BLAex_n_r_id_2_50_2.joblib')
NNid = np.where(all_time_N_id == 1)
RRid = np.where(all_time_R_id == 2)

N=[]
R=[]
W=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data\\'
    (N_fr,R_fr,W_fr)=joblib.load(data_dir + rat_name + '_Y_BLAex_n_r_w.joblib')

    N_cell = np.mean(N_fr, axis=1)
    R_cell=np.mean(R_fr, axis=1)
    W_cell = np.mean(W_fr, axis=1)
    N=np.append(N,N_cell)
    R=np.append(R,R_cell)
    W=np.append(W,W_cell)

# N_id
N_mean_nid=np.nanmean(N[NNid])
R_mean_nid=np.nanmean(R[NNid])
W_mean_nid=np.nanmean(W[NNid])
N_std_nid=np.nanstd(N[NNid])
R_std_nid=np.nanstd(R[NNid])
W_std_nid=np.nanstd(W[NNid])
N_sem_nid = N_std_nid / np.sqrt(len(N))
R_sem_nid = R_std_nid / np.sqrt(len(R))
W_sem_nid = W_std_nid / np.sqrt(len(W))
sem_nid=[N_sem_nid,R_sem_nid,W_sem_nid]
# R_id
N_mean_rid=np.nanmean(N[RRid])
R_mean_rid=np.nanmean(R[RRid])
W_mean_rid=np.nanmean(W[RRid])
N_std_rid=np.nanstd(N[RRid])
R_std_rid=np.nanstd(R[RRid])
W_std_rid=np.nanstd(W[RRid])
N_sem_rid = N_std_rid / np.sqrt(len(N[RRid]))
R_sem_rid = R_std_rid / np.sqrt(len(R[RRid]))
W_sem_rid = W_std_rid / np.sqrt(len(W[RRid]))
sem_rid=[N_sem_rid,R_sem_rid,W_sem_rid]


ax3.bar([1,3,5], [N_mean_nid,R_mean_nid,W_mean_nid],color='cornflowerblue',yerr=sem_nid)
ax3.bar([2,4,6], [N_mean_rid,R_mean_rid,W_mean_rid],color='violet',yerr=sem_rid)
ax3.set_xticks([1.5,3.5,5.5])
ax3.set_xticklabels(labels, fontsize=15)
props = {'connectionstyle': 'bar', 'arrowstyle': '-','linewidth':1}
ax3.annotate('', xy=(1,1.73), xytext=(2,1.73), arrowprops=props)
#ax3.annotate('', xy=(3,1.73), xytext=(4,1.73), arrowprops=props)
ax3.annotate('', xy=(5,2.1), xytext=(6,2.1), arrowprops=props)
ax3.text(1.4, 1.83, "***")
#ax3.text(3.4, 1.83, "**")
ax3.text(5.38, 2.20, "***")
#ax3.set_ylabel('Fr(Hz)')
#ax3.set_title('BLAex', fontsize=15)

r1=stats.mannwhitneyu(N[NNid],N[RRid],alternative='two-sided')
r2=stats.mannwhitneyu(R[NNid],R[RRid],alternative='two-sided')
r3=stats.mannwhitneyu(W[NNid],W[RRid],alternative='two-sided')
print(r1)
print(r2)
print(r3)

mean=[N_mean_nid,N_mean_rid,R_mean_nid,R_mean_rid,W_mean_nid,W_mean_rid]
sem=[N_sem_nid,N_sem_rid,R_sem_nid,R_sem_rid,W_sem_nid,W_sem_rid]
data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
#joblib.dump((mean,sem), os.path.expanduser(data_dir +'fig_5_a_BLAex.joblib'), compress=3)
joblib.dump((mean,sem), os.path.expanduser(data_dir +'Y_fig_6_a_BLAex.joblib'), compress=3)

ax1.set_title('vCA1ex', fontsize=25)
ax2.set_title('PL5ex', fontsize=25)
ax3.set_title('BLAex', fontsize=25)

print(cohens_d(N[NNid], N[RRid]))
print(cohens_d(R[NNid], R[RRid]))
print(cohens_d(W[NNid], W[RRid]))

#fig.suptitle('Last-First during Extended sleep', fontsize=20)
plt.subplots_adjust(wspace=0.3)

plt.savefig('test8.pdf')

plt.savefig('test8.pdf')