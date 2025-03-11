import joblib
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

from matplotlib.backends.backend_pdf import PdfPages
import os
from scipy import stats
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, portrait
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import webbrowser

fig = plt.figure(figsize=(24, 8))
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)


#vCA1
rats=['duvel190505','hoegaarden181115','innis190601','karmeliet190901','leffe200124','maredsous200224','nostrum200304','oberon200325']

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
(all_time_N_id,all_time_R_id)=joblib.load(data_dir + 'Y_vCA1ex_n_r_id_2_50_2.joblib')
NNid = np.where(all_time_N_id == 1)
RRid = np.where(all_time_R_id == 2)


#shock1回分と
FZ_mean=[]
Before=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data2\\'
    (Fr) = joblib.load(data_dir + rat_name + '_vCA1ex_shock_session1.joblib')
    (Fr_mean,Fr_sd) = joblib.load(data_dir + rat_name + '_vCA1ex_before_first_cue.joblib')
    FR_Z = []
    for n in range(12):
        fz = []
        for i in range(Fr[0].shape[0]):
            z=(Fr[n][i])
            fz=np.append(fz,z)
        Fz = fz.reshape(-1, 59)
        FR_Z.append(Fz)
    FR_Z_mean=np.nanmean(FR_Z, axis=0)
    FZ_mean=np.append(FZ_mean,FR_Z_mean)
    #初めのCueの前2秒のmean　Fr
    Before=np.append(Before,Fr_mean)

FR_cue=FZ_mean.reshape(-1,59)
FR_cue[np.isinf(FR_cue)] = np.nan
N_in=np.nanmax(FR_cue[:,20:40],axis=1)
#N_out=np.nanmean(FR_cue[:,0:20],axis=1)
N_out=Before

gain=N_in/(N_out)
gain[np.isinf(gain)] = np.nan

gain_nid_N=np.nanmean(gain[NNid])
gain_rid_N=np.nanmean(gain[RRid])
gain_std_nid=np.nanstd(gain[NNid])
gain_std_rid=np.nanstd(gain[RRid])
gain_sem_nid = gain_std_nid / np.sqrt(len(gain[NNid]))
gain_sem_rid = gain_std_rid / np.sqrt(len(gain[RRid]))
sem_nid_gain_N=[gain_sem_nid,gain_sem_rid]

result3 = stats.mannwhitneyu(gain[NNid],gain[RRid],alternative='two-sided')
print(result3)

ax1.bar([0], [gain_nid_N],color='cornflowerblue',label=['nrem'],yerr=sem_nid_gain_N[0])
ax1.bar([1], [gain_rid_N],color='violet',label=['rem'],yerr=sem_nid_gain_N[1])
labels=('N','R')
ax1.set_xticks([0,1])
ax1.set_xticklabels(labels, fontsize=15)
#ax4.set_title('shock gain', fontsize=15)
props = {'arrowstyle': '-','linewidth':1}
ax1.annotate('', xy=(0.2,gain_rid_N*1.1), xytext=(0.8,gain_rid_N*1.1), arrowprops=props)
ax1.text(0.5,gain_rid_N*1.1 , "*")
ax1.set_ylabel('Ratio',fontsize=15)
#ax4.set_title('vCA1ex_gain', fontsize=15)

gain=[gain_nid_N,gain_rid_N]
data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
#joblib.dump((gain,sem_nid_gain_N), os.path.expanduser(data_dir +'fig_5_e_vCA1ex.joblib'), compress=3)
joblib.dump((gain,sem_nid_gain_N), os.path.expanduser(data_dir +'Y_fig_6_e_vCA1ex.joblib'), compress=3)

#PL5
rats=['hoegaarden181115','innis190601','jever190814','karmeliet190901','leffe200124','maredsous200224','nostrum200304','oberon200325']

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
(all_time_N_id,all_time_R_id)=joblib.load(data_dir + 'Y_PL5ex_n_r_id_2_50_2.joblib')
NNid = np.where(all_time_N_id == 1)
RRid = np.where(all_time_R_id == 2)

#shock1回分と
FZ_mean=[]
Before=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data2\\'
    (Fr) = joblib.load(data_dir + rat_name + '_PL5ex_shock_session1.joblib')
    (Fr_mean,Fr_sd) = joblib.load(data_dir + rat_name + '_PL5ex_before_first_cue.joblib')
    FR_Z = []
    for n in range(12):
        fz = []
        for i in range(Fr[0].shape[0]):
            z=(Fr[n][i])
            fz=np.append(fz,z)
        Fz = fz.reshape(-1, 59)
        FR_Z.append(Fz)
    FR_Z_mean=np.nanmean(FR_Z, axis=0)
    FZ_mean=np.append(FZ_mean,FR_Z_mean)
    Before=np.append(Before,Fr_mean)

FR_cue=FZ_mean.reshape(-1,59)
FR_cue[np.isinf(FR_cue)] = np.nan

N_in=np.nanmax(FR_cue[:,20:40],axis=1)
#N_out=np.nanmean(FR_cue[:,0:20],axis=1)
N_out=Before


gain=N_in/(N_out)
gain[np.isinf(gain)] = np.nan



gain_nid_N=np.nanmean(gain[NNid])
gain_rid_N=np.nanmean(gain[RRid])
gain_std_nid=np.nanstd(gain[NNid])
gain_std_rid=np.nanstd(gain[RRid])
gain_sem_nid = gain_std_nid / np.sqrt(len(gain[NNid]))
gain_sem_rid = gain_std_rid / np.sqrt(len(gain[RRid]))
sem_nid_gain_N=[gain_sem_nid,gain_sem_rid]


result3 = stats.mannwhitneyu(gain[NNid],gain[RRid],alternative='two-sided')
print(result3)

ax2.bar([0], [gain_nid_N],color='cornflowerblue',label=['nrem'],yerr=sem_nid_gain_N[0])
ax2.bar([1], [gain_rid_N],color='violet',label=['rem'],yerr=sem_nid_gain_N[1])
labels=('N','R')
ax2.set_xticks([0,1])
ax2.set_xticklabels(labels, fontsize=15)
#ax5.set_title('shock gain', fontsize=15)
props = {'arrowstyle': '-','linewidth':1}
#ax5.annotate('', xy=(0.2,gain_rid_N*1.1), xytext=(0.8,gain_rid_N*1.1), arrowprops=props)
#ax5.text(0.5,gain_rid_N*1.1 , "*")
#ax5.set_ylabel('ratio')
#ax5.set_title('vCA1ex_gain', fontsize=15)

gain=[gain_nid_N,gain_rid_N]
data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
#joblib.dump((gain,sem_nid_gain_N), os.path.expanduser(data_dir +'fig_5_e_PL5ex.joblib'), compress=3)
joblib.dump((gain,sem_nid_gain_N), os.path.expanduser(data_dir +'Y_fig_6_e_PL5ex.joblib'), compress=3)

#BLA
rats=['achel180320','booyah180430','duvel190505','estrella180808','guiness181002','hoegaarden181115','innis190601','jever190814','leffe200124','maredsous200224','nostrum200304','oberon200325']

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
(all_time_N_id,all_time_R_id)=joblib.load(data_dir + 'Y_BLAex_n_r_id_2_50_2.joblib')
NNid = np.where(all_time_N_id == 1)
RRid = np.where(all_time_R_id == 2)


#shock1回分と
FZ_mean=[]
Before=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data2\\'
    (Fr) = joblib.load(data_dir + rat_name + '_BLAex_shock_session1.joblib')
    (Fr_mean,Fr_sd) = joblib.load(data_dir + rat_name + '_BLAex_before_first_cue.joblib')
    FR_Z = []
    for n in range(12):
        fz = []
        for i in range(Fr[0].shape[0]):
            z=(Fr[n][i])
            fz=np.append(fz,z)
        Fz = fz.reshape(-1, 59)
        FR_Z.append(Fz)
    FR_Z_mean=np.nanmean(FR_Z, axis=0)
    FZ_mean=np.append(FZ_mean,FR_Z_mean)
    Before=np.append(Before,Fr_mean)

FR_cue=FZ_mean.reshape(-1,59)
FR_cue[np.isinf(FR_cue)] = np.nan
N_in=np.nanmax(FR_cue[:,20:40],axis=1)
#N_out=np.nanmean(FR_cue[:,0:20],axis=1)
N_out=Before
gain=N_in/(N_out)
gain[np.isinf(gain)] = np.nan

gain_nid_N=np.nanmean(gain[NNid])
gain_rid_N=np.nanmean(gain[RRid])
gain_std_nid=np.nanstd(gain[NNid])
gain_std_rid=np.nanstd(gain[RRid])
gain_sem_nid = gain_std_nid / np.sqrt(len(gain[NNid]))
gain_sem_rid = gain_std_rid / np.sqrt(len(gain[RRid]))
sem_nid_gain_N=[gain_sem_nid,gain_sem_rid]

result3 = stats.mannwhitneyu(gain[NNid],gain[RRid],alternative='two-sided')
print(result3)

ax3.bar([0], [gain_nid_N],color='cornflowerblue',label=['nrem'],yerr=sem_nid_gain_N[0])
ax3.bar([1], [gain_rid_N],color='violet',label=['rem'],yerr=sem_nid_gain_N[1])
labels=('N','R')
ax3.set_xticks([0,1])
ax3.set_xticklabels(labels, fontsize=15)
#x6.set_title('shock gain', fontsize=15)
props = {'arrowstyle': '-','linewidth':1}
ax3.annotate('', xy=(0.2,gain_rid_N*1.1), xytext=(0.8,gain_rid_N*1.1), arrowprops=props)
ax3.text(0.5,gain_rid_N*1.1 , "*")
#ax6.set_ylabel('ratio')
#ax4.set_title('vCA1ex_gain', fontsize=15)

plt.subplots_adjust(hspace=0.4,wspace=0.35)
#fig.suptitle('active cell rate in SWR during nrem sleep', fontsize=20)

gain=[gain_nid_N,gain_rid_N]
data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
#joblib.dump((gain,sem_nid_gain_N), os.path.expanduser(data_dir +'fig_5_e_BLAex.joblib'), compress=3)
joblib.dump((gain,sem_nid_gain_N), os.path.expanduser(data_dir +'Y_fig_6_e_BLAex.joblib'), compress=3)

ax1.set_title('vCA1ex', fontsize=25)
ax2.set_title('PL5ex', fontsize=25)
ax3.set_title('BLAex', fontsize=25)

plt.savefig('test9.pdf')