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


fig = plt.figure(figsize=(12, 3))
ax1 = fig.add_subplot(2, 3, 1)
ax2 = fig.add_subplot(2, 3, 2)
ax3 = fig.add_subplot(2, 3, 3)
ax4 = fig.add_subplot(2, 3, 4)
ax5 = fig.add_subplot(2, 3, 5)
ax6 = fig.add_subplot(2, 3, 6)


#vCA1
rats=['duvel190505','hoegaarden181115','innis190601','karmeliet190901','leffe200124','maredsous200224','nostrum200304','oberon200325']

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
(all_time_N_id,all_time_R_id)=joblib.load(data_dir + 'Y_vCA1ex_n_r_id_2_50_2.joblib')
NNid = np.where(all_time_N_id == 1)
RRid = np.where(all_time_R_id == 2)

FR_Z=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data2\\'
    (Fr) = joblib.load(data_dir + rat_name + '_vCA1ex_shock_12_session1.joblib')
    (Fr_mean,Fr_sd) = joblib.load(data_dir + rat_name + '_vCA1ex_before_first_cue.joblib')
    #data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data\\'
    #(Fr_mean, Fr_sd) = joblib.load(data_dir + rat_name + '_vCA1ex_nrem_Z.joblib')
    fz=[]
    for i in range(Fr.shape[0]):
        z=(Fr[i]-Fr_mean[i])/Fr_sd[i]
        fz=np.append(fz, z)
    fz = fz.reshape(Fr.shape[0], -1)
    FR_Z=np.append(FR_Z,fz)
FR_cue=FR_Z.reshape(-1,59)
FR_cue[np.isinf(FR_cue)] = np.nan


FR_cue_nid=FR_cue[NNid]
FR_cue_nid_mean=np.nanmean(FR_cue_nid,axis=1)
FR_cue_rid=FR_cue[RRid]
FR_cue_rid_mean=np.nanmean(FR_cue_rid,axis=1)
flat_F= []
original_list = FR_cue_nid
for l in original_list:
    for item in l:
        flat_F.append(item)
flat_L= []
original_list = FR_cue_rid
for l in original_list:
    for item in l:
        flat_L.append(item)

FR_cue_nid_mean=np.nanmean(FR_cue_nid,axis=0)
FR_cue_rid_mean=np.nanmean(FR_cue_rid,axis=0)
FR_cue_nid_std=np.nanstd(FR_cue_nid,axis=0)
FR_cue_rid_std=np.nanstd(FR_cue_rid,axis=0)
FR_cue_nid_sem=FR_cue_nid_std/np.sqrt(len(FR_cue_nid))
FR_cue_rid_sem=FR_cue_rid_std/np.sqrt(len(FR_cue_rid))

x = np.arange(-20,39)
ax1.plot(x,FR_cue_nid_mean ,color='cornflowerblue')
ax1.plot(x,FR_cue_rid_mean ,color='violet')
yerr=FR_cue_nid_sem
ax1.axvspan(0, 20, facecolor='yellow', alpha=0.1)
ax1.fill_between(x,FR_cue_nid_mean+yerr,FR_cue_nid_mean-yerr,color='cornflowerblue', alpha=0.15)
yerr=FR_cue_rid_sem
ax1.fill_between(x,FR_cue_rid_mean+yerr,FR_cue_rid_mean-yerr,color='violet', alpha=0.15)
labels=("-1","0","1","2","3")
ax1.set_xticks([-10,0,10,20,30])
ax1.set_xticklabels(labels, fontsize=10)
ax1.set_xlabel('Time(s)',fontsize=10)
ax1.set_ylabel('Z score',fontsize=15)
#ax1.set_title('vCA1ex_shock_all', fontsize=15)


data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
#joblib.dump((FR_cue_nid_mean,FR_cue_nid_sem), os.path.expanduser(data_dir +'fig_5_d_vCA1ex_nrem_active.joblib'), compress=3)
#joblib.dump((FR_cue_rid_mean,FR_cue_rid_sem), os.path.expanduser(data_dir +'fig_5_d_vCA1ex_rem_active.joblib'), compress=3)

joblib.dump((FR_cue_nid_mean,FR_cue_nid_sem), os.path.expanduser(data_dir +'Y_fig_6_d_vCA1ex_nrem_active.joblib'), compress=3)
joblib.dump((FR_cue_rid_mean,FR_cue_rid_sem), os.path.expanduser(data_dir +'Y_fig_6_d_vCA1ex_rem_active.joblib'), compress=3)


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

ax4.bar([0], [gain_nid_N],color='cornflowerblue',label=['nrem'],yerr=sem_nid_gain_N[0])
ax4.bar([1], [gain_rid_N],color='violet',label=['rem'],yerr=sem_nid_gain_N[1])
labels=('N','R')
ax4.set_xticks([0,1])
ax4.set_xticklabels(labels, fontsize=15)
#ax4.set_title('shock gain', fontsize=15)
props = {'arrowstyle': '-','linewidth':1}
ax4.annotate('', xy=(0.2,gain_rid_N*1.1), xytext=(0.8,gain_rid_N*1.1), arrowprops=props)
ax4.text(0.5,gain_rid_N*1.1 , "*")
ax4.set_ylabel('ratio',fontsize=15)
#ax4.set_title('vCA1ex_gain', fontsize=15)

#PL5
rats=['hoegaarden181115','innis190601','jever190814','karmeliet190901','leffe200124','maredsous200224','nostrum200304','oberon200325']

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
(all_time_N_id,all_time_R_id)=joblib.load(data_dir + 'Y_PL5ex_n_r_id_2_50_2.joblib')
NNid = np.where(all_time_N_id == 1)
RRid = np.where(all_time_R_id == 2)


FR_Z=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data2\\'
    (Fr) = joblib.load(data_dir + rat_name + '_PL5ex_shock_12_session1.joblib')
    (Fr_mean,Fr_sd) = joblib.load(data_dir + rat_name + '_PL5ex_before_first_cue.joblib')
    #data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data\\'
    #(Fr_mean, Fr_sd) = joblib.load(data_dir + rat_name + '_vCA1ex_nrem_Z.joblib')
    fz=[]
    for i in range(Fr.shape[0]):
        z=(Fr[i]-Fr_mean[i])/Fr_sd[i]
        fz=np.append(fz, z)
    fz = fz.reshape(Fr.shape[0], -1)
    FR_Z=np.append(FR_Z,fz)
FR_cue=FR_Z.reshape(-1,59)
FR_cue[np.isinf(FR_cue)] = np.nan


FR_cue_nid=FR_cue[NNid]
FR_cue_nid_mean=np.nanmean(FR_cue_nid,axis=1)
FR_cue_rid=FR_cue[RRid]
FR_cue_rid_mean=np.nanmean(FR_cue_rid,axis=1)
flat_F= []
original_list = FR_cue_nid
for l in original_list:
    for item in l:
        flat_F.append(item)
flat_L= []
original_list = FR_cue_rid
for l in original_list:
    for item in l:
        flat_L.append(item)

FR_cue_nid_mean=np.nanmean(FR_cue_nid,axis=0)
FR_cue_rid_mean=np.nanmean(FR_cue_rid,axis=0)
FR_cue_nid_std=np.nanstd(FR_cue_nid,axis=0)
FR_cue_rid_std=np.nanstd(FR_cue_rid,axis=0)
FR_cue_nid_sem=FR_cue_nid_std/np.sqrt(len(FR_cue_nid))
FR_cue_rid_sem=FR_cue_rid_std/np.sqrt(len(FR_cue_rid))

x = np.arange(-20,39)
ax2.plot(x,FR_cue_nid_mean ,color='cornflowerblue')
ax2.plot(x,FR_cue_rid_mean ,color='violet')
yerr=FR_cue_nid_sem
ax2.axvspan(0, 20, facecolor='yellow', alpha=0.1)
ax2.fill_between(x,FR_cue_nid_mean+yerr,FR_cue_nid_mean-yerr,color='cornflowerblue', alpha=0.15)
yerr=FR_cue_rid_sem
ax2.fill_between(x,FR_cue_rid_mean+yerr,FR_cue_rid_mean-yerr,color='violet', alpha=0.15)
labels=("-1","0","1","2","3")
ax2.set_xticks([-10,0,10,20,30])
ax2.set_xticklabels(labels, fontsize=10)
ax2.set_xlabel('Time(s)',fontsize=10)
#ax2.set_ylabel('Z score',fontsize=10)
#ax2.set_title('PL5ex_shock_all', fontsize=15)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
#joblib.dump((FR_cue_nid_mean,FR_cue_nid_sem), os.path.expanduser(data_dir +'fig_5_d_PL5ex_nrem_active.joblib'), compress=3)
#joblib.dump((FR_cue_rid_mean,FR_cue_rid_sem), os.path.expanduser(data_dir +'fig_5_d_PL5ex_rem_active.joblib'), compress=3)

joblib.dump((FR_cue_nid_mean,FR_cue_nid_sem), os.path.expanduser(data_dir +'Y_fig_6_d_PL5ex_nrem_active.joblib'), compress=3)
joblib.dump((FR_cue_rid_mean,FR_cue_rid_sem), os.path.expanduser(data_dir +'Y_fig_6_d_PL5ex_rem_active.joblib'), compress=3)

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

ax5.bar([0], [gain_nid_N],color='cornflowerblue',label=['nrem'],yerr=sem_nid_gain_N[0])
ax5.bar([1], [gain_rid_N],color='violet',label=['rem'],yerr=sem_nid_gain_N[1])
labels=('N','R')
ax5.set_xticks([0,1])
ax5.set_xticklabels(labels, fontsize=15)
#ax5.set_title('shock gain', fontsize=15)
props = {'arrowstyle': '-','linewidth':1}
#ax5.annotate('', xy=(0.2,gain_rid_N*1.1), xytext=(0.8,gain_rid_N*1.1), arrowprops=props)
#ax5.text(0.5,gain_rid_N*1.1 , "*")
#ax5.set_ylabel('ratio')
#ax5.set_title('vCA1ex_gain', fontsize=15)


#BLA
rats=['achel180320','booyah180430','duvel190505','estrella180808','guiness181002','hoegaarden181115','innis190601','jever190814','leffe200124','maredsous200224','nostrum200304','oberon200325']

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
(all_time_N_id,all_time_R_id)=joblib.load(data_dir + 'Y_BLAex_n_r_id_2_50_2.joblib')
NNid = np.where(all_time_N_id == 1)
RRid = np.where(all_time_R_id == 2)

FR_Z=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data2\\'
    (Fr) = joblib.load(data_dir + rat_name + '_BLAex_shock_12_session1.joblib')
    (Fr_mean,Fr_sd) = joblib.load(data_dir + rat_name + '_BLAex_before_first_cue.joblib')
    #data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data\\'
    #(Fr_mean, Fr_sd) = joblib.load(data_dir + rat_name + '_vCA1ex_nrem_Z.joblib')
    fz=[]
    for i in range(Fr.shape[0]):
        z=(Fr[i]-Fr_mean[i])/Fr_sd[i]
        fz=np.append(fz, z)
    fz = fz.reshape(Fr.shape[0], -1)
    FR_Z=np.append(FR_Z,fz)
FR_cue=FR_Z.reshape(-1,59)
FR_cue[np.isinf(FR_cue)] = np.nan


FR_cue_nid=FR_cue[NNid]
FR_cue_nid_mean=np.nanmean(FR_cue_nid,axis=1)
FR_cue_rid=FR_cue[RRid]
FR_cue_rid_mean=np.nanmean(FR_cue_rid,axis=1)
flat_F= []
original_list = FR_cue_nid
for l in original_list:
    for item in l:
        flat_F.append(item)
flat_L= []
original_list = FR_cue_rid
for l in original_list:
    for item in l:
        flat_L.append(item)

FR_cue_nid_mean=np.nanmean(FR_cue_nid,axis=0)
FR_cue_rid_mean=np.nanmean(FR_cue_rid,axis=0)
FR_cue_nid_std=np.nanstd(FR_cue_nid,axis=0)
FR_cue_rid_std=np.nanstd(FR_cue_rid,axis=0)
FR_cue_nid_sem=FR_cue_nid_std/np.sqrt(len(FR_cue_nid))
FR_cue_rid_sem=FR_cue_rid_std/np.sqrt(len(FR_cue_rid))

x = np.arange(-20,39)
ax3.plot(x,FR_cue_nid_mean ,color='cornflowerblue')
ax3.plot(x,FR_cue_rid_mean ,color='violet')
yerr=FR_cue_nid_sem
ax3.axvspan(0, 20, facecolor='yellow', alpha=0.1)
ax3.fill_between(x,FR_cue_nid_mean+yerr,FR_cue_nid_mean-yerr,color='cornflowerblue', alpha=0.15)
yerr=FR_cue_rid_sem
ax3.fill_between(x,FR_cue_rid_mean+yerr,FR_cue_rid_mean-yerr,color='violet', alpha=0.15)
labels=("-1","0","1","2","3")
ax3.set_xticks([-10,0,10,20,30])
ax3.set_xticklabels(labels, fontsize=10)
ax3.set_xlabel('Time(s)',fontsize=10)
#ax3.set_ylabel('Z score',fontsize=10)
#ax3.set_title('BLAex_shock_all', fontsize=15)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
#joblib.dump((FR_cue_nid_mean,FR_cue_nid_sem), os.path.expanduser(data_dir +'fig_5_d_BLAex_nrem_active.joblib'), compress=3)
#joblib.dump((FR_cue_rid_mean,FR_cue_rid_sem), os.path.expanduser(data_dir +'fig_5_d_BLAex_rem_active.joblib'), compress=3)

joblib.dump((FR_cue_nid_mean,FR_cue_nid_sem), os.path.expanduser(data_dir +'Y_fig_6_d_BLAex_nrem_active.joblib'), compress=3)
joblib.dump((FR_cue_rid_mean,FR_cue_rid_sem), os.path.expanduser(data_dir +'Y_fig_6_d_BLAex_rem_active.joblib'), compress=3)


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

ax6.bar([0], [gain_nid_N],color='cornflowerblue',label=['nrem'],yerr=sem_nid_gain_N[0])
ax6.bar([1], [gain_rid_N],color='violet',label=['rem'],yerr=sem_nid_gain_N[1])
labels=('N','R')
ax6.set_xticks([0,1])
ax6.set_xticklabels(labels, fontsize=15)
#x6.set_title('shock gain', fontsize=15)
props = {'arrowstyle': '-','linewidth':1}
ax6.annotate('', xy=(0.2,gain_rid_N*1.1), xytext=(0.8,gain_rid_N*1.1), arrowprops=props)
ax6.text(0.5,gain_rid_N*1.1 , "*")
#ax6.set_ylabel('ratio')
#ax4.set_title('vCA1ex_gain', fontsize=15)

plt.subplots_adjust(hspace=0.4,wspace=0.35)
#fig.suptitle('active cell rate in SWR during nrem sleep', fontsize=20)

ax1.set_title('vCA1ex', fontsize=25)
ax2.set_title('PL5ex', fontsize=25)
ax3.set_title('BLAex', fontsize=25)

plt.savefig('test9.pdf')
