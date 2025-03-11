import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from scipy import stats
import seaborn as sns
sns.set_context('poster')

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


fig = plt.figure(figsize=(24, 8))
ax1 = fig.add_subplot(2, 6, 1)
ax2 = fig.add_subplot(2, 6, 2)
ax3 = fig.add_subplot(2, 6, 3)
ax4 = fig.add_subplot(2, 6, 4)
ax5 = fig.add_subplot(2, 6, 5)
ax6 = fig.add_subplot(2, 6, 6)
ax7 = fig.add_subplot(2, 6, 7)
ax8 = fig.add_subplot(2, 6, 8)
ax9 = fig.add_subplot(2, 6, 9)
ax10 = fig.add_subplot(2, 6, 10)
ax11 = fig.add_subplot(2, 6, 11)
ax12 = fig.add_subplot(2, 6, 12)

#vCA1
rats=['duvel190505','hoegaarden181115','innis190601','karmeliet190901','leffe200124','maredsous200224','nostrum200304','oberon200325']

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
(all_time_N_id,all_time_R_id)=joblib.load(data_dir + 'Y_vCA1ex_n_r_id_2_50_2.joblib')
NNid = np.where(all_time_N_id == 1)
RRid = np.where(all_time_R_id == 2)

N_in=[]
N_out=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (N_in_cell,N_out_cell,*_) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_swr.joblib')
    N_in = np.append(N_in, N_in_cell)
    N_out = np.append(N_out, N_out_cell)

gain_N=N_in/N_out

gain_nid_N=np.nanmean(gain_N[NNid])
gain_rid_N=np.nanmean(gain_N[RRid])
gain_std_nid=np.nanstd(gain_N[NNid])
gain_std_rid=np.nanstd(gain_N[RRid])
gain_sem_nid = gain_std_nid / np.sqrt(len(gain_N[NNid]))
gain_sem_rid = gain_std_rid / np.sqrt(len(gain_N[RRid]))
gain=[gain_nid_N,gain_rid_N]
sem_nid_gain_N=[gain_sem_nid,gain_sem_rid]

labels=('N','R')
ax1.bar([0], [gain_nid_N],color='cornflowerblue',label=['nrem'],yerr=sem_nid_gain_N[0])
ax1.bar([1], [gain_rid_N],color='violet',label=['rem'],yerr=sem_nid_gain_N[1])
ax1.set_xticks([0,1])
ax1.set_xticklabels(labels, fontsize=15)
props = {'arrowstyle': '-','linewidth':1}
#ax1.annotate('', xy=(0.2,gain_rid_N+0.02), xytext=(0.8,gain_rid_N+0.02), arrowprops=props)
#ax1.text(0.37,gain_rid_N+0.05 , "**")
#ax7.set_ylabel('ratio', fontsize=15)
#ax7.set_title('gain', fontsize=15)


result = stats.mannwhitneyu(gain_N[NNid],gain_N[RRid],alternative='two-sided')
print(result)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((gain,sem_nid_gain_N), os.path.expanduser(data_dir +'Y_fig_3_a_vCA1ex_2.joblib'), compress=3)

print(cohens_d(gain_N[NNid], gain_N[RRid]))

#PL5
rats=['hoegaarden181115','innis190601','jever190814','karmeliet190901','leffe200124','maredsous200224','nostrum200304','oberon200325']

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
(all_time_N_id,all_time_R_id)=joblib.load(data_dir + 'Y_PL5ex_n_r_id_2_50_2.joblib')
NNid = np.where(all_time_N_id == 1)
RRid = np.where(all_time_R_id == 2)

N_in=[]
N_out=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (N_in_cell,N_out_cell,*_) = joblib.load(data_dir + rat_name + '_Y_PL5ex_swr.joblib')
    N_in = np.append(N_in, N_in_cell)
    N_out = np.append(N_out, N_out_cell)

gain_N=N_in/N_out

gain_nid_N=np.nanmean(gain_N[NNid])
gain_rid_N=np.nanmean(gain_N[RRid])
gain_std_nid=np.nanstd(gain_N[NNid])
gain_std_rid=np.nanstd(gain_N[RRid])
gain_sem_nid = gain_std_nid / np.sqrt(len(gain_N[NNid]))
gain_sem_rid = gain_std_rid / np.sqrt(len(gain_N[RRid]))
gain=[gain_nid_N,gain_rid_N]
sem_nid_gain_N=[gain_sem_nid,gain_sem_rid]

ax2.bar([0], [gain_nid_N],color='cornflowerblue',label=['nrem'],yerr=sem_nid_gain_N[0])
ax2.bar([1], [gain_rid_N],color='violet',label=['rem'],yerr=sem_nid_gain_N[1])
ax2.set_xticks([0,1])
ax2.set_xticklabels(labels, fontsize=15)
#ax8.set_title('Within/between_wake_freeze', fontsize=15)
props = {'arrowstyle': '-','linewidth':1}
#ax2.annotate('', xy=(0.2,gain_rid_N+0.03), xytext=(0.8,gain_rid_N+0.03), arrowprops=props)
#ax2.text(0.35,gain_rid_N+0.03 , "***")
#ax8.set_title('gain', fontsize=15)

result = stats.mannwhitneyu(gain_N[NNid],gain_N[RRid],alternative='two-sided')
print(result.pvalue)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((gain,sem_nid_gain_N), os.path.expanduser(data_dir +'Y_fig_3_a_PL5ex.joblib'), compress=3)

print(cohens_d(gain_N[NNid], gain_N[RRid]))

#BLA
rats = ['achel180320',  'duvel190505', 'estrella180808', 'guiness181002', 'hoegaarden181115', 'innis190601','jever190814', 'leffe200124', 'maredsous200224', 'nostrum200304', 'oberon200325']


data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
(all_time_N_id,all_time_R_id)=joblib.load(data_dir + 'Y_BLAex_n_r_id_2_50_2_outof_b.joblib')
NNid = np.where(all_time_N_id == 1)
RRid = np.where(all_time_R_id == 2)

N_in=[]
N_out=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (N_in_cell,N_out_cell,*_) = joblib.load(data_dir + rat_name + '_Y_BLAex_swr.joblib')
    N_in = np.append(N_in, N_in_cell)
    N_out = np.append(N_out, N_out_cell)

gain_N=N_in/N_out

gain_nid_N=np.nanmean(gain_N[NNid])
gain_rid_N=np.nanmean(gain_N[RRid])
gain_std_nid=np.nanstd(gain_N[NNid])
gain_std_rid=np.nanstd(gain_N[RRid])
gain_sem_nid = gain_std_nid / np.sqrt(len(gain_N[NNid]))
gain_sem_rid = gain_std_rid / np.sqrt(len(gain_N[RRid]))
gain=[gain_nid_N,gain_rid_N]
sem_nid_gain_N=[gain_sem_nid,gain_sem_rid]


ax3.bar([0], [gain_nid_N],color='cornflowerblue',label=['nrem'],yerr=sem_nid_gain_N[0])
ax3.bar([1], [gain_rid_N],color='violet',label=['rem'],yerr=sem_nid_gain_N[1])
ax3.set_xticks([0,1])
ax3.set_xticklabels(labels, fontsize=15)
#ax9.set_title('Within/between_wake_freeze', fontsize=15)
props = {'arrowstyle': '-','linewidth':1}
#ax3.annotate('', xy=(0.2,gain_rid_N+0.05), xytext=(0.8,gain_rid_N+0.05), arrowprops=props)
#ax3.text(0.35,gain_rid_N+0.05 , "***")
#ax9.set_title('gain', fontsize=15)

result = stats.mannwhitneyu(gain_N[NNid],gain_N[RRid],alternative='two-sided')
print(result.pvalue)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((gain,sem_nid_gain_N), os.path.expanduser(data_dir +'Y_fig_3_a_BLAex.joblib'), compress=3)

print(cohens_d(gain_N[NNid], gain_N[RRid]))

#vCA1
rats=['duvel190505','hoegaarden181115','innis190601','karmeliet190901','leffe200124','maredsous200224','nostrum200304','oberon200325']

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
(all_time_N_id,all_time_R_id)=joblib.load(data_dir + 'Y_vCA1ex_n_r_id_2_50_2.joblib')
NNid = np.where(all_time_N_id == 1)
RRid = np.where(all_time_R_id == 2)

N_in=[]
N_out=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (N_in_cell,N_out_cell,*_) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_spindle.joblib')
    N_in = np.append(N_in, N_in_cell)
    N_out = np.append(N_out, N_out_cell)

gain_N=N_in/N_out

gain_nid_N=np.nanmean(gain_N[NNid])
gain_rid_N=np.nanmean(gain_N[RRid])
gain_std_nid=np.nanstd(gain_N[NNid])
gain_std_rid=np.nanstd(gain_N[RRid])
gain_sem_nid = gain_std_nid / np.sqrt(len(gain_N[NNid]))
gain_sem_rid = gain_std_rid / np.sqrt(len(gain_N[RRid]))
gain=[gain_nid_N,gain_rid_N]
sem_nid_gain_N=[gain_sem_nid,gain_sem_rid]

labels=('N','R')
ax4.bar([0], [gain_nid_N],color='cornflowerblue',label=['nrem'],yerr=sem_nid_gain_N[0])
ax4.bar([1], [gain_rid_N],color='violet',label=['rem'],yerr=sem_nid_gain_N[1])
ax4.set_xticks([0,1])
ax4.set_xticklabels(labels, fontsize=15)
props = {'arrowstyle': '-','linewidth':1}
#ax4.annotate('', xy=(0.2,gain_rid_N+0.05), xytext=(0.8,gain_rid_N+0.05), arrowprops=props)
#ax4.text(0.35,gain_rid_N+0.05 , "***")
#ax7.set_ylabel('ratio', fontsize=15)
#ax7.set_title('gain', fontsize=15)

result = stats.mannwhitneyu(gain_N[NNid],gain_N[RRid],alternative='two-sided')
print(result.pvalue)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((gain,sem_nid_gain_N), os.path.expanduser(data_dir +'Y_fig_3_b_vCA1ex_2.joblib'), compress=3)

print(cohens_d(gain_N[NNid], gain_N[RRid]))

#PL5
rats=['hoegaarden181115','innis190601','jever190814','karmeliet190901','leffe200124','maredsous200224','nostrum200304','oberon200325']

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
(all_time_N_id,all_time_R_id)=joblib.load(data_dir + 'Y_PL5ex_n_r_id_2_50_2.joblib')
NNid = np.where(all_time_N_id == 1)
RRid = np.where(all_time_R_id == 2)


N_in=[]
N_out=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (N_in_cell,N_out_cell,*_) = joblib.load(data_dir + rat_name + '_Y_PL5ex_spindle.joblib')
    N_in = np.append(N_in, N_in_cell)
    N_out = np.append(N_out, N_out_cell)

gain_N=N_in/N_out

gain_nid_N=np.nanmean(gain_N[NNid])
gain_rid_N=np.nanmean(gain_N[RRid])
gain_std_nid=np.nanstd(gain_N[NNid])
gain_std_rid=np.nanstd(gain_N[RRid])
gain_sem_nid = gain_std_nid / np.sqrt(len(gain_N[NNid]))
gain_sem_rid = gain_std_rid / np.sqrt(len(gain_N[RRid]))
gain=[gain_nid_N,gain_rid_N]
sem_nid_gain_N=[gain_sem_nid,gain_sem_rid]

ax5.bar([0], [gain_nid_N],color='cornflowerblue',label=['nrem'],yerr=sem_nid_gain_N[0])
ax5.bar([1], [gain_rid_N],color='violet',label=['rem'],yerr=sem_nid_gain_N[1])
ax5.set_xticks([0,1])
ax5.set_xticklabels(labels, fontsize=15)
#ax8.set_title('Within/between_wake_freeze', fontsize=15)
props = {'arrowstyle': '-','linewidth':1}
#ax5.annotate('', xy=(0.2,gain_rid_N+0.03), xytext=(0.8,gain_rid_N+0.03), arrowprops=props)
#ax5.text(0.35,gain_rid_N+0.03 , "***")
#ax8.set_title('gain', fontsize=15)

result = stats.mannwhitneyu(gain_N[NNid],gain_N[RRid],alternative='two-sided')
print(result.pvalue)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((gain,sem_nid_gain_N), os.path.expanduser(data_dir +'Y_fig_3_b_PL5ex.joblib'), compress=3)

print(cohens_d(gain_N[NNid], gain_N[RRid]))

#BLA
#rats=['duvel190505','estrella180808','guiness181002','hoegaarden181115','innis190601','jever190814','leffe200124','maredsous200224','nostrum200304','oberon200325']

# BLA
rats = ['achel180320',  'duvel190505', 'estrella180808', 'guiness181002', 'hoegaarden181115', 'innis190601','jever190814', 'leffe200124', 'maredsous200224', 'nostrum200304', 'oberon200325']


data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
(all_time_N_id,all_time_R_id)=joblib.load(data_dir + 'Y_BLAex_n_r_id_2_50_2_outof_b.joblib')
NNid = np.where(all_time_N_id == 1)
RRid = np.where(all_time_R_id == 2)

N_in=[]
N_out=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (N_in_cell,N_out_cell,*_) = joblib.load(data_dir + rat_name + '_Y_BLAex_spindle.joblib')
    N_in = np.append(N_in, N_in_cell)
    N_out = np.append(N_out, N_out_cell)

gain_N=N_in/N_out

gain_nid_N=np.nanmean(gain_N[NNid])
gain_rid_N=np.nanmean(gain_N[RRid])
gain_std_nid=np.nanstd(gain_N[NNid])
gain_std_rid=np.nanstd(gain_N[RRid])
gain_sem_nid = gain_std_nid / np.sqrt(len(gain_N[NNid]))
gain_sem_rid = gain_std_rid / np.sqrt(len(gain_N[RRid]))
gain=[gain_nid_N,gain_rid_N]
sem_nid_gain_N=[gain_sem_nid,gain_sem_rid]


ax6.bar([0], [gain_nid_N],color='cornflowerblue',label=['nrem'],yerr=sem_nid_gain_N[0])
ax6.bar([1], [gain_rid_N],color='violet',label=['rem'],yerr=sem_nid_gain_N[1])
ax6.set_xticks([0,1])
ax6.set_xticklabels(labels, fontsize=15)
#ax9.set_title('Within/between_wake_freeze', fontsize=15)
props = {'arrowstyle': '-','linewidth':1}
#ax6.annotate('', xy=(0.2,gain_rid_N+0.05), xytext=(0.8,gain_rid_N+0.05), arrowprops=props)
#ax6.text(0.35,gain_rid_N+0.05 , "***")
#ax9.set_title('gain', fontsize=15)

result = stats.mannwhitneyu(gain_N[NNid],gain_N[RRid],alternative='two-sided')
print(result.pvalue)

#fig.suptitle('   SWR', fontsize=20)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((gain,sem_nid_gain_N), os.path.expanduser(data_dir +'Y_fig_3_b_BLAex.joblib'), compress=3)

print(cohens_d(gain_N[NNid], gain_N[RRid]))

#vCA1
rats=['duvel190505','hoegaarden181115','innis190601','karmeliet190901','leffe200124','maredsous200224','nostrum200304','oberon200325']

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
(all_time_N_id,all_time_R_id)=joblib.load(data_dir + 'Y_vCA1ex_n_r_id_2_50_2.joblib')
NNid = np.where(all_time_N_id == 1)
RRid = np.where(all_time_R_id == 2)

N_in=[]
N_out=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (N_in_cell,N_out_cell,*_) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_hfo.joblib')
    N_in = np.append(N_in, N_in_cell)
    N_out = np.append(N_out, N_out_cell)

gain_N=N_in/N_out

gain_nid_N=np.nanmean(gain_N[NNid])
gain_rid_N=np.nanmean(gain_N[RRid])
gain_std_nid=np.nanstd(gain_N[NNid])
gain_std_rid=np.nanstd(gain_N[RRid])
gain_sem_nid = gain_std_nid / np.sqrt(len(gain_N[NNid]))
gain_sem_rid = gain_std_rid / np.sqrt(len(gain_N[RRid]))
gain=[gain_nid_N,gain_rid_N]
sem_nid_gain_N=[gain_sem_nid,gain_sem_rid]

labels=('N','R')
ax7.bar([0], [gain_nid_N],color='cornflowerblue',label=['nrem'],yerr=sem_nid_gain_N[0])
ax7.bar([1], [gain_rid_N],color='violet',label=['rem'],yerr=sem_nid_gain_N[1])
ax7.set_xticks([0,1])
ax7.set_xticklabels(labels, fontsize=15)
props = {'arrowstyle': '-','linewidth':1}
#ax7.annotate('', xy=(0.2,gain_rid_N+0.05), xytext=(0.8,gain_rid_N+0.05), arrowprops=props)
#ax7.text(0.45,gain_rid_N+0.05 , "**")
#ax7.set_ylabel('ratio', fontsize=15)
#ax7.set_title('gain', fontsize=15)


result = stats.mannwhitneyu(gain_N[NNid],gain_N[RRid],alternative='two-sided')
print(result.pvalue)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((gain,sem_nid_gain_N), os.path.expanduser(data_dir +'Y_fig_3_c_vCA1ex_2.joblib'), compress=3)

print(cohens_d(gain_N[NNid], gain_N[RRid]))

#PL5
rats=['hoegaarden181115','innis190601','jever190814','karmeliet190901','leffe200124','maredsous200224','nostrum200304','oberon200325']

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
(all_time_N_id,all_time_R_id)=joblib.load(data_dir + 'Y_PL5ex_n_r_id_2_50_2.joblib')
NNid = np.where(all_time_N_id == 1)
RRid = np.where(all_time_R_id == 2)

N_in=[]
N_out=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (N_in_cell,N_out_cell,*_) = joblib.load(data_dir + rat_name + '_Y_PL5ex_hfo.joblib')
    N_in = np.append(N_in, N_in_cell)
    N_out = np.append(N_out, N_out_cell)

gain_N=N_in/N_out

gain_nid_N=np.nanmean(gain_N[NNid])
gain_rid_N=np.nanmean(gain_N[RRid])
gain_std_nid=np.nanstd(gain_N[NNid])
gain_std_rid=np.nanstd(gain_N[RRid])
gain_sem_nid = gain_std_nid / np.sqrt(len(gain_N[NNid]))
gain_sem_rid = gain_std_rid / np.sqrt(len(gain_N[RRid]))
sem_nid_gain_N=[gain_sem_nid,gain_sem_rid]
gain=[gain_nid_N,gain_rid_N]

ax8.bar([0], [gain_nid_N],color='cornflowerblue',label=['nrem'],yerr=sem_nid_gain_N[0])
ax8.bar([1], [gain_rid_N],color='violet',label=['rem'],yerr=sem_nid_gain_N[1])
ax8.set_xticks([0,1])
ax8.set_xticklabels(labels, fontsize=15)
#ax8.set_title('Within/between_wake_freeze', fontsize=15)
props = {'arrowstyle': '-','linewidth':1}
#ax8.annotate('', xy=(0.2,gain_rid_N+0.03), xytext=(0.8,gain_rid_N+0.03), arrowprops=props)
#ax8.text(0.35,gain_rid_N+0.03 , "***")
#ax8.set_title('gain', fontsize=15)

result = stats.mannwhitneyu(gain_N[NNid],gain_N[RRid],alternative='two-sided')
print(result.pvalue)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((gain,sem_nid_gain_N), os.path.expanduser(data_dir +'Y_fig_3_c_PL5ex.joblib'), compress=3)

print(cohens_d(gain_N[NNid], gain_N[RRid]))

#BLA
rats = ['achel180320',  'duvel190505', 'estrella180808', 'guiness181002', 'hoegaarden181115', 'innis190601','jever190814', 'leffe200124', 'maredsous200224', 'nostrum200304', 'oberon200325']


data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
(all_time_N_id,all_time_R_id)=joblib.load(data_dir + 'Y_BLAex_n_r_id_2_50_2_outof_b.joblib')
NNid = np.where(all_time_N_id == 1)
RRid = np.where(all_time_R_id == 2)


N_in=[]
N_out=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (N_in_cell,N_out_cell,*_) = joblib.load(data_dir + rat_name + '_Y_BLAex_hfo.joblib')
    N_in = np.append(N_in, N_in_cell)
    N_out = np.append(N_out, N_out_cell)

gain_N=N_in/N_out

gain_nid_N=np.nanmean(gain_N[NNid])
gain_rid_N=np.nanmean(gain_N[RRid])
gain_std_nid=np.nanstd(gain_N[NNid])
gain_std_rid=np.nanstd(gain_N[RRid])
gain_sem_nid = gain_std_nid / np.sqrt(len(gain_N[NNid]))
gain_sem_rid = gain_std_rid / np.sqrt(len(gain_N[RRid]))
sem_nid_gain_N=[gain_sem_nid,gain_sem_rid]
gain=[gain_nid_N,gain_rid_N]

ax9.bar([0], [gain_nid_N],color='cornflowerblue',label=['nrem'],yerr=sem_nid_gain_N[0])
ax9.bar([1], [gain_rid_N],color='violet',label=['rem'],yerr=sem_nid_gain_N[1])
ax9.set_xticks([0,1])
ax9.set_xticklabels(labels, fontsize=15)
#ax9.set_title('Within/between_wake_freeze', fontsize=15)
props = {'arrowstyle': '-','linewidth':1}
#ax9.annotate('', xy=(0.2,gain_rid_N+0.03), xytext=(0.8,gain_rid_N+0.03), arrowprops=props)
#ax9.text(0.35,gain_rid_N+0.03 , "***")
#ax9.set_title('gain', fontsize=15)

result = stats.mannwhitneyu(gain_N[NNid],gain_N[RRid],alternative='two-sided')
print(result.pvalue)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((gain,sem_nid_gain_N), os.path.expanduser(data_dir +'Y_fig_3_c_BLAex.joblib'), compress=3)

print(cohens_d(gain_N[NNid], gain_N[RRid]))

#vCA1
rats=['duvel190505','hoegaarden181115','innis190601','karmeliet190901','leffe200124','maredsous200224','nostrum200304','oberon200325']

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
(all_time_N_id,all_time_R_id)=joblib.load(data_dir + 'Y_vCA1ex_n_r_id_2_50_2.joblib')
NNid = np.where(all_time_N_id == 1)
RRid = np.where(all_time_R_id == 2)

N_in=[]
N_out=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (N_in_cell,N_out_cell,*_) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_crip.joblib')
    N_in = np.append(N_in, N_in_cell)
    N_out = np.append(N_out, N_out_cell)

gain_N=N_in/N_out

gain_nid_N=np.nanmean(gain_N[NNid])
gain_rid_N=np.nanmean(gain_N[RRid])
gain_std_nid=np.nanstd(gain_N[NNid])
gain_std_rid=np.nanstd(gain_N[RRid])
gain_sem_nid = gain_std_nid / np.sqrt(len(gain_N[NNid]))
gain_sem_rid = gain_std_rid / np.sqrt(len(gain_N[RRid]))
sem_nid_gain_N=[gain_sem_nid,gain_sem_rid]
gain=[gain_nid_N,gain_rid_N]

labels=('N','R')
ax10.bar([0], [gain_nid_N],color='cornflowerblue',label=['nrem'],yerr=sem_nid_gain_N[0])
ax10.bar([1], [gain_rid_N],color='violet',label=['rem'],yerr=sem_nid_gain_N[1])
ax10.set_xticks([0,1])
ax10.set_xticklabels(labels, fontsize=15)
props = {'arrowstyle': '-','linewidth':1}
#ax10.annotate('', xy=(0.2,gain_rid_N+0.05), xytext=(0.8,gain_rid_N+0.05), arrowprops=props)
#ax10.text(0.45,gain_rid_N+0.05 , "**")
#ax7.set_ylabel('ratio', fontsize=15)
#ax7.set_title('gain', fontsize=15)


result = stats.mannwhitneyu(gain_N[NNid],gain_N[RRid],alternative='two-sided')
print(result.pvalue)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((gain,sem_nid_gain_N), os.path.expanduser(data_dir +'Y_fig_3_d_vCA1ex_2.joblib'), compress=3)

print(cohens_d(gain_N[NNid], gain_N[RRid]))

#PL5
rats=['hoegaarden181115','innis190601','jever190814','karmeliet190901','leffe200124','maredsous200224','nostrum200304','oberon200325']

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
(all_time_N_id,all_time_R_id)=joblib.load(data_dir + 'Y_PL5ex_n_r_id_2_50_2.joblib')
NNid = np.where(all_time_N_id == 1)
RRid = np.where(all_time_R_id == 2)

N_in=[]
N_out=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (N_in_cell,N_out_cell,*_) = joblib.load(data_dir + rat_name + '_Y_PL5ex_crip.joblib')
    N_in = np.append(N_in, N_in_cell)
    N_out = np.append(N_out, N_out_cell)

gain_N=N_in/N_out

gain_nid_N=np.nanmean(gain_N[NNid])
gain_rid_N=np.nanmean(gain_N[RRid])
gain_std_nid=np.nanstd(gain_N[NNid])
gain_std_rid=np.nanstd(gain_N[RRid])
gain_sem_nid = gain_std_nid / np.sqrt(len(gain_N[NNid]))
gain_sem_rid = gain_std_rid / np.sqrt(len(gain_N[RRid]))
sem_nid_gain_N=[gain_sem_nid,gain_sem_rid]
gain=[gain_nid_N,gain_rid_N]

ax11.bar([0], [gain_nid_N],color='cornflowerblue',label=['nrem'],yerr=sem_nid_gain_N[0])
ax11.bar([1], [gain_rid_N],color='violet',label=['rem'],yerr=sem_nid_gain_N[1])
ax11.set_xticks([0,1])
ax11.set_xticklabels(labels, fontsize=15)
#ax8.set_title('Within/between_wake_freeze', fontsize=15)
props = {'arrowstyle': '-','linewidth':1}
#ax2.annotate('', xy=(0.2,gain_rid_N+0.03), xytext=(0.8,gain_rid_N+0.03), arrowprops=props)
#ax2.text(0.4,gain_rid_N+0.03 , "***")
#ax8.set_title('gain', fontsize=15)

result = stats.mannwhitneyu(gain_N[NNid],gain_N[RRid],alternative='two-sided')
print(result.pvalue)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((gain,sem_nid_gain_N), os.path.expanduser(data_dir +'Y_fig_3_d_PL5ex.joblib'), compress=3)

print(cohens_d(gain_N[NNid], gain_N[RRid]))

#BLA
rats = ['achel180320',  'duvel190505', 'estrella180808', 'guiness181002', 'hoegaarden181115', 'innis190601','jever190814', 'leffe200124', 'maredsous200224', 'nostrum200304', 'oberon200325']


data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
(all_time_N_id,all_time_R_id)=joblib.load(data_dir + 'Y_BLAex_n_r_id_2_50_2_outof_b.joblib')
NNid = np.where(all_time_N_id == 1)
RRid = np.where(all_time_R_id == 2)

N_in=[]
N_out=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (N_in_cell,N_out_cell,*_) = joblib.load(data_dir + rat_name + '_Y_BLAex_crip.joblib')
    N_in = np.append(N_in, N_in_cell)
    N_out = np.append(N_out, N_out_cell)

gain_N=N_in/N_out

gain_nid_N=np.nanmean(gain_N[NNid])
gain_rid_N=np.nanmean(gain_N[RRid])
gain_std_nid=np.nanstd(gain_N[NNid])
gain_std_rid=np.nanstd(gain_N[RRid])
gain_sem_nid = gain_std_nid / np.sqrt(len(gain_N[NNid]))
gain_sem_rid = gain_std_rid / np.sqrt(len(gain_N[RRid]))
sem_nid_gain_N=[gain_sem_nid,gain_sem_rid]
gain=[gain_nid_N,gain_rid_N]

ax12.bar([0], [gain_nid_N],color='cornflowerblue',label=['nrem'],yerr=sem_nid_gain_N[0])
ax12.bar([1], [gain_rid_N],color='violet',label=['rem'],yerr=sem_nid_gain_N[1])
ax12.set_xticks([0,1])
ax12.set_xticklabels(labels, fontsize=15)
#ax9.set_title('Within/between_wake_freeze', fontsize=15)
props = {'arrowstyle': '-','linewidth':1}
#ax12.annotate('', xy=(0.2,gain_rid_N+0.05), xytext=(0.8,gain_rid_N+0.05), arrowprops=props)
#ax12.text(0.4,gain_rid_N+0.05 , "***")
#ax9.set_title('gain', fontsize=15)

result = stats.mannwhitneyu(gain_N[NNid],gain_N[RRid],alternative='two-sided')
print(result.pvalue)

print(cohens_d(gain_N[NNid], gain_N[RRid]))

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((gain,sem_nid_gain_N), os.path.expanduser(data_dir +'Y_fig_3_d_BLAex.joblib'), compress=3)

plt.subplots_adjust(hspace=0.3)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'IPAPGothic'

#plt.gcf().text(0.20,0.92,"vCA1ex", fontsize=15)
#plt.gcf().text(0.48,0.92,"PL5ex", fontsize=15)
#plt.gcf().text(0.75,0.92,"BLAex", fontsize=15)
plt.subplots_adjust(wspace=0.4,hspace=0.45)

ax1.set_title('vCA1ex', fontsize=25)
ax2.set_title('PL5ex', fontsize=25)
ax3.set_title('BLAex', fontsize=25)
ax4.set_title('vCA1ex', fontsize=25)
ax5.set_title('PL5ex', fontsize=25)
ax6.set_title('BLAex', fontsize=25)
ax7.set_title('vCA1ex', fontsize=25)
ax8.set_title('PL5ex', fontsize=25)
ax9.set_title('BLAex', fontsize=25)
ax10.set_title('vCA1ex', fontsize=25)
ax11.set_title('PL5ex', fontsize=25)
ax12.set_title('BLAex', fontsize=25)



from matplotlib.backends.backend_pdf import PdfPages
pdf=PdfPages("test8.pdf")
fignums=plt.get_fignums()
for fignum in fignums:
    plt.figure(fignum)
    pdf.savefig()
pdf.close()

plt.savefig('test8.pdf')
#fig.savefig("test8.pdf")

hfont = {'fontname':'DejaVu Sans'}

plt.xlabel('xlabel', **hfont)

import matplotlib.font_manager as fm
fm.findSystemFonts()

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

print([f.name for f in fm.fontManager.ttflist])

import matplotlib.pyplot as plt
# デフォルトの設定を確認
print(plt.rcParams["font.family"])

import matplotlib
matplotlib.get_cachedir()