import seaborn as sns
sns.set_context('poster')

import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from scipy import stats
from scipy.stats import spearmanr

#vCA1
rats=['duvel190505','hoegaarden181115','innis190601','karmeliet190901','leffe200124','maredsous200224','nostrum200304','oberon200325']

fig = plt.figure(figsize=(18, 8))
ax1 = fig.add_subplot(2, 3, 1)
ax2 = fig.add_subplot(2, 3, 2)
ax3 = fig.add_subplot(2, 3, 3)
ax4 = fig.add_subplot(2, 3, 4)
ax5 = fig.add_subplot(2, 3, 5)
ax6 = fig.add_subplot(2, 3, 6)


#Zscore
FFZ=[]
XX=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data\\'
    (F,X,Fr2) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_allnrem.joblib')
    (Fr_mean, Fr_sd) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_nrem_Z.joblib')
    XX.append(X)
    fz = []
    for i in range(Fr_mean.shape[0]):
        z = (F[i] - Fr_mean[i]) / Fr_sd[i]
        fz = np.append(fz, z)
    fz=fz.reshape(Fr_mean.shape[0],-1)
    FFZ.append(fz)

FFFZ=[]
for n in range(len(FFZ)):
    for m in range(FFZ[n].shape[0]):
        FFFZ.append(FFZ[n][m].tolist())

flat_F= []
original_list = FFFZ
for l in original_list:
    for item in l:
        flat_F.append(item)



data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
(all_time_N_id,all_time_R_id)=joblib.load(data_dir + 'Y_vCA1ex_n_r_id_2_50_2.joblib')
NNid = np.where(all_time_N_id == 1)
RRid = np.where(all_time_R_id == 2)

F_nid=[]
for n in NNid[0]:
    F_nid=np.append(F_nid,FFFZ[n])
F_rid=[]
for n in RRid[0]:
    F_rid=np.append(F_rid,FFFZ[n])

XXX=[]
for n in range(len(XX)):
    for m in range(FFZ[n].shape[0]):
        XXX.append(XX[n])
X_nid=[]
for n in NNid[0]:
    X_nid=np.append(X_nid,XXX[n])
X_rid=[]
for n in RRid[0]:
    X_rid=np.append(X_rid,XXX[n])

x=np.concatenate([X_nid],0)
y=np.concatenate([F_nid],0)
c=np.column_stack([X_nid,F_nid])
c2=c[c[:,0].argsort(),:]

ax1.scatter(x, y, color='cornflowerblue', s=1)
#ax1.set_xlabel('Time(sec)')
#ax1.set_ylabel('Zscore', fontsize=15)
#ax1.legend()
ax1.set_title('vCA1ex', fontsize=25)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((x,y), os.path.expanduser(data_dir +'Y_fig_2_b_vCA1ex_nrem_active_2.joblib'), compress=3)


g0,g1,g2,g3,g4,g5,g6,g7,g8,g9=np.array_split(c2,10,0)
g0mean=np.nanmean(g0,axis=0)
g1mean=np.nanmean(g1,axis=0)
g2mean=np.nanmean(g2,axis=0)
g3mean=np.nanmean(g3,axis=0)
g4mean=np.nanmean(g4,axis=0)
g5mean=np.nanmean(g5,axis=0)
g6mean=np.nanmean(g6,axis=0)
g7mean=np.nanmean(g7,axis=0)
g8mean=np.nanmean(g8,axis=0)
g9mean=np.nanmean(g9,axis=0)

g0std=np.nanstd(g0,axis=0)
g1std=np.nanstd(g1,axis=0)
g2std=np.nanstd(g2,axis=0)
g3std=np.nanstd(g3,axis=0)
g4std=np.nanstd(g4,axis=0)
g5std=np.nanstd(g5,axis=0)
g6std=np.nanstd(g6,axis=0)
g7std=np.nanstd(g7,axis=0)
g8std=np.nanstd(g8,axis=0)
g9std=np.nanstd(g9,axis=0)

g0sem = g0std[1]/np.sqrt(len(g0))
g1sem = g1std[1]/np.sqrt(len(g1))
g2sem = g2std[1]/np.sqrt(len(g2))
g3sem = g3std[1]/np.sqrt(len(g3))
g4sem = g4std[1]/np.sqrt(len(g4))
g5sem = g5std[1]/np.sqrt(len(g5))
g6sem = g6std[1]/np.sqrt(len(g6))
g7sem = g7std[1]/np.sqrt(len(g7))
g8sem = g8std[1]/np.sqrt(len(g8))
g9sem = g9std[1]/np.sqrt(len(g9))

xx=(g0mean[0],g1mean[0],g2mean[0],g3mean[0],g4mean[0],g5mean[0],g6mean[0],g7mean[0],g8mean[0],g9mean[0])
yy=(g0mean[1],g1mean[1],g2mean[1],g3mean[1],g4mean[1],g5mean[1],g6mean[1],g7mean[1],g8mean[1],g9mean[1])
y_err=(g0sem,g1sem,g2sem,g3sem,g4sem,g5sem,g6sem,g7sem,g8sem,g9sem)

a, b = np.polyfit(x, y, 1)
Y2 = a * x + b
np.corrcoef(x,y)[0][1]
#r,p=pearsonr(x, y)
r, p = spearmanr(x, y)
print('y=', a, '*x', '+', b)
print('r=', r)
print('p=', p)

ax4.set_ylim([-0.35, 0.25])
ax4.plot(x,Y2,color='cornflowerblue')
ax4.errorbar(xx,yy,yerr=y_err,fmt='o',color='cornflowerblue')
ax4.text(0.45, 0.8, "r=-0.06   ", va='top', ha='left', transform=ax4.transAxes ,fontsize=25)
#ax4.legend(loc='upper left')
#ax4.set_ylabel('Zscore', fontsize=15)
#ax4.set_xlabel('Time(sec)')

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((x,y,xx,yy,y_err,Y2), os.path.expanduser(data_dir +'Y_fig_2_b_bottom_vCA1ex_nrem_active_2_sp.joblib'), compress=3)

x=np.concatenate([X_rid],0)
y=np.concatenate([F_rid],0)
c=np.column_stack([X_rid,F_rid])
c2=c[c[:,0].argsort(),:]

ax1.scatter(x, y, color='violet', s=1)
#ax2.set_xlabel('Time(sec)')
#ax1.set_title('vCA1ex', fontsize=15)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((x,y), os.path.expanduser(data_dir +'Y_fig_2_b_vCA1ex_rem_active_2.joblib'), compress=3)

g0,g1,g2,g3,g4,g5,g6,g7,g8,g9=np.array_split(c2,10,0)
g0mean=np.nanmean(g0,axis=0)
g1mean=np.nanmean(g1,axis=0)
g2mean=np.nanmean(g2,axis=0)
g3mean=np.nanmean(g3,axis=0)
g4mean=np.nanmean(g4,axis=0)
g5mean=np.nanmean(g5,axis=0)
g6mean=np.nanmean(g6,axis=0)
g7mean=np.nanmean(g7,axis=0)
g8mean=np.nanmean(g8,axis=0)
g9mean=np.nanmean(g9,axis=0)

g0std=np.nanstd(g0,axis=0)
g1std=np.nanstd(g1,axis=0)
g2std=np.nanstd(g2,axis=0)
g3std=np.nanstd(g3,axis=0)
g4std=np.nanstd(g4,axis=0)
g5std=np.nanstd(g5,axis=0)
g6std=np.nanstd(g6,axis=0)
g7std=np.nanstd(g7,axis=0)
g8std=np.nanstd(g8,axis=0)
g9std=np.nanstd(g9,axis=0)

g0sem = g0std[1]/np.sqrt(len(g0))
g1sem = g1std[1]/np.sqrt(len(g1))
g2sem = g2std[1]/np.sqrt(len(g2))
g3sem = g3std[1]/np.sqrt(len(g3))
g4sem = g4std[1]/np.sqrt(len(g4))
g5sem = g5std[1]/np.sqrt(len(g5))
g6sem = g6std[1]/np.sqrt(len(g6))
g7sem = g7std[1]/np.sqrt(len(g7))
g8sem = g8std[1]/np.sqrt(len(g8))
g9sem = g9std[1]/np.sqrt(len(g9))

xx=(g0mean[0],g1mean[0],g2mean[0],g3mean[0],g4mean[0],g5mean[0],g6mean[0],g7mean[0],g8mean[0],g9mean[0])
yy=(g0mean[1],g1mean[1],g2mean[1],g3mean[1],g4mean[1],g5mean[1],g6mean[1],g7mean[1],g8mean[1],g9mean[1])
y_err=(g0sem,g1sem,g2sem,g3sem,g4sem,g5sem,g6sem,g7sem,g8sem,g9sem)

a, b = np.polyfit(x, y, 1)
Y2 = a * x + b
np.corrcoef(x,y)[0][1]
#r,p=pearsonr(x, y)
r, p = spearmanr(x, y)
print('y=', a, '*x', '+', b)
print('r=', r)
print('p=', p)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((x,y,xx,yy,y_err,Y2), os.path.expanduser(data_dir +'Y_fig_2_b_bottom_vCA1ex_rem_active_2_sp.joblib'), compress=3)

ax4.set_ylim([-0.35, 0.25])
ax4.plot(x,Y2,color='violet')
ax4.errorbar(xx,yy,yerr=y_err,fmt='o',color='violet')
ax4.text(0.45, 0.9, "r=-0.11***", va='top', ha='left', transform=ax4.transAxes ,fontsize=25)
#ax4.set_xlabel('Time(sec)')

#PL5
rats=['hoegaarden181115','innis190601','jever190814','karmeliet190901','leffe200124','maredsous200224','nostrum200304','oberon200325']
(all_time_N_id,all_time_R_id)=joblib.load('PL5ex_n_r_id_1.joblib')
NNid = np.where(all_time_N_id== 1)
RRid = np.where(all_time_R_id == 2)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
(all_time_N_id,all_time_R_id)=joblib.load(data_dir + 'Y_PL5ex_n_r_id_2_50_2.joblib')
NNid = np.where(all_time_N_id == 1)
RRid = np.where(all_time_R_id == 2)


#Zscore
FFZ=[]
XX=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data\\'
    (F,X,Fr2) = joblib.load(data_dir + rat_name + '_Y_PL5ex_allnrem.joblib')
    (Fr_mean, Fr_sd) = joblib.load(data_dir + rat_name + '_Y_PL5ex_nrem_Z.joblib')
    XX.append(X)
    fz = []
    for i in range(Fr_mean.shape[0]):
        z = (F[i] - Fr_mean[i]) / Fr_sd[i]
        fz = np.append(fz, z)
    fz=fz.reshape(Fr_mean.shape[0],-1)
    FFZ.append(fz)

FFFZ=[]
for n in range(len(FFZ)):
    for m in range(FFZ[n].shape[0]):
        FFFZ.append(FFZ[n][m].tolist())

flat_F= []
original_list = FFFZ
for l in original_list:
    for item in l:
        flat_F.append(item)


F_nid=[]
for n in NNid[0]:
    F_nid=np.append(F_nid,FFFZ[n])
F_rid=[]
for n in RRid[0]:
    F_rid=np.append(F_rid,FFFZ[n])

XXX=[]
for n in range(len(XX)):
    for m in range(FFZ[n].shape[0]):
        XXX.append(XX[n])
X_nid=[]
for n in NNid[0]:
    X_nid=np.append(X_nid,XXX[n])
X_rid=[]
for n in RRid[0]:
    X_rid=np.append(X_rid,XXX[n])

x=np.concatenate([X_nid],0)
y=np.concatenate([F_nid],0)
c=np.column_stack([X_nid,F_nid])
c2=c[c[:,0].argsort(),:]

ax2.scatter(x, y, color='cornflowerblue', s=1)
#ax3.set_xlabel('Time(sec)')
#ax2.set_title('PL5ex N', fontsize=15)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((x,y), os.path.expanduser(data_dir +'Y_fig_2_b_PL5ex_nrem_active_2.joblib'), compress=3)

g0,g1,g2,g3,g4,g5,g6,g7,g8,g9=np.array_split(c2,10,0)
g0mean=np.nanmean(g0,axis=0)
g1mean=np.nanmean(g1,axis=0)
g2mean=np.nanmean(g2,axis=0)
g3mean=np.nanmean(g3,axis=0)
g4mean=np.nanmean(g4,axis=0)
g5mean=np.nanmean(g5,axis=0)
g6mean=np.nanmean(g6,axis=0)
g7mean=np.nanmean(g7,axis=0)
g8mean=np.nanmean(g8,axis=0)
g9mean=np.nanmean(g9,axis=0)

g0std=np.nanstd(g0,axis=0)
g1std=np.nanstd(g1,axis=0)
g2std=np.nanstd(g2,axis=0)
g3std=np.nanstd(g3,axis=0)
g4std=np.nanstd(g4,axis=0)
g5std=np.nanstd(g5,axis=0)
g6std=np.nanstd(g6,axis=0)
g7std=np.nanstd(g7,axis=0)
g8std=np.nanstd(g8,axis=0)
g9std=np.nanstd(g9,axis=0)

g0sem = g0std[1]/np.sqrt(len(g0))
g1sem = g1std[1]/np.sqrt(len(g1))
g2sem = g2std[1]/np.sqrt(len(g2))
g3sem = g3std[1]/np.sqrt(len(g3))
g4sem = g4std[1]/np.sqrt(len(g4))
g5sem = g5std[1]/np.sqrt(len(g5))
g6sem = g6std[1]/np.sqrt(len(g6))
g7sem = g7std[1]/np.sqrt(len(g7))
g8sem = g8std[1]/np.sqrt(len(g8))
g9sem = g9std[1]/np.sqrt(len(g9))

xx=(g0mean[0],g1mean[0],g2mean[0],g3mean[0],g4mean[0],g5mean[0],g6mean[0],g7mean[0],g8mean[0],g9mean[0])
yy=(g0mean[1],g1mean[1],g2mean[1],g3mean[1],g4mean[1],g5mean[1],g6mean[1],g7mean[1],g8mean[1],g9mean[1])
y_err=(g0sem,g1sem,g2sem,g3sem,g4sem,g5sem,g6sem,g7sem,g8sem,g9sem)

a, b = np.polyfit(x, y, 1)
Y2 = a * x + b
np.corrcoef(x,y)[0][1]
#r,p=pearsonr(x, y)
r, p = spearmanr(x, y)
print('y=', a, '*x', '+', b)
print('r=', r)
print('p=', p)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((x,y,xx,yy,y_err,Y2), os.path.expanduser(data_dir +'Y_fig_2_b_bottom_PL5ex_nrem_active_2_sp.joblib'), compress=3)

ax5.set_ylim([-0.3, 0.25])
ax5.plot(x,Y2,color='cornflowerblue')
ax5.errorbar(xx,yy,yerr=y_err,fmt='o',color='cornflowerblue')
ax5.text(0.45, 0.8, "r=-0.08***", va='top', ha='left', transform=ax5.transAxes ,fontsize=25)
#ax5.set_xlabel('Time(sec)')

x=np.concatenate([X_rid],0)
y=np.concatenate([F_rid],0)
c=np.column_stack([X_rid,F_rid])
c2=c[c[:,0].argsort(),:]

ax2.scatter(x, y, color='violet', s=1,alpha=0.5)
#ax4.set_xlabel('Time(sec)')
ax2.set_title('PL5ex', fontsize=25)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((x,y), os.path.expanduser(data_dir +'Y_fig_2_b_PL5ex_rem_active_2.joblib'), compress=3)

g0,g1,g2,g3,g4,g5,g6,g7,g8,g9=np.array_split(c2,10,0)
g0mean=np.nanmean(g0,axis=0)
g1mean=np.nanmean(g1,axis=0)
g2mean=np.nanmean(g2,axis=0)
g3mean=np.nanmean(g3,axis=0)
g4mean=np.nanmean(g4,axis=0)
g5mean=np.nanmean(g5,axis=0)
g6mean=np.nanmean(g6,axis=0)
g7mean=np.nanmean(g7,axis=0)
g8mean=np.nanmean(g8,axis=0)
g9mean=np.nanmean(g9,axis=0)

g0std=np.nanstd(g0,axis=0)
g1std=np.nanstd(g1,axis=0)
g2std=np.nanstd(g2,axis=0)
g3std=np.nanstd(g3,axis=0)
g4std=np.nanstd(g4,axis=0)
g5std=np.nanstd(g5,axis=0)
g6std=np.nanstd(g6,axis=0)
g7std=np.nanstd(g7,axis=0)
g8std=np.nanstd(g8,axis=0)
g9std=np.nanstd(g9,axis=0)

g0sem = g0std[1]/np.sqrt(len(g0))
g1sem = g1std[1]/np.sqrt(len(g1))
g2sem = g2std[1]/np.sqrt(len(g2))
g3sem = g3std[1]/np.sqrt(len(g3))
g4sem = g4std[1]/np.sqrt(len(g4))
g5sem = g5std[1]/np.sqrt(len(g5))
g6sem = g6std[1]/np.sqrt(len(g6))
g7sem = g7std[1]/np.sqrt(len(g7))
g8sem = g8std[1]/np.sqrt(len(g8))
g9sem = g9std[1]/np.sqrt(len(g9))

xx=(g0mean[0],g1mean[0],g2mean[0],g3mean[0],g4mean[0],g5mean[0],g6mean[0],g7mean[0],g8mean[0],g9mean[0])
yy=(g0mean[1],g1mean[1],g2mean[1],g3mean[1],g4mean[1],g5mean[1],g6mean[1],g7mean[1],g8mean[1],g9mean[1])
y_err=(g0sem,g1sem,g2sem,g3sem,g4sem,g5sem,g6sem,g7sem,g8sem,g9sem)

a, b = np.polyfit(x, y, 1)
Y2 = a * x + b
np.corrcoef(x,y)[0][1]
#r,p=pearsonr(x, y)
r, p = spearmanr(x, y)

print('y=', a, '*x', '+', b)
print('r=', r)
print('p=', p)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((x,y,xx,yy,y_err,Y2), os.path.expanduser(data_dir +'Y_fig_2_b_bottom_PL5ex_rem_active_2_sp.joblib'), compress=3)

ax5.set_xlabel('Time(sec)', fontsize=25)
ax5.set_ylim([-0.3, 0.25])
ax5.plot(x,Y2,color='violet')
ax5.errorbar(xx,yy,yerr=y_err,fmt='o',color='violet')
ax5.text(0.45, 0.9, "r=-0.09***", va='top', ha='left', transform=ax5.transAxes ,fontsize=25)
#ax5.set_xlabel('Time(sec)')


#BLA
#rats=['duvel190505','estrella180808','guiness181002','hoegaarden181115','innis190601','jever190814','leffe200124','maredsous200224','nostrum200304','oberon200325']

rats = ['achel180320', 'booyah180430', 'duvel190505', 'estrella180808', 'guiness181002', 'hoegaarden181115',
        'innis190601', 'jever190814', 'leffe200124', 'maredsous200224', 'nostrum200304', 'oberon200325']

#data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
#(all_time_N_id,all_time_R_id)=joblib.load(data_dir + 'Y_BLAex_n_r_id_2_50_2_outof_a_b.joblib')
#NNid = np.where(all_time_N_id == 1)
#RRid = np.where(all_time_R_id == 2)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
(all_time_N_id,all_time_R_id)=joblib.load(data_dir + 'Y_BLAex_n_r_id_2_50_2.joblib')
NNid = np.where(all_time_N_id == 1)
RRid = np.where(all_time_R_id == 2)

#Zscore
FFZ=[]
XX=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data\\'
    (F,X,Fr2) = joblib.load(data_dir + rat_name + '_Y_BLAex_allnrem.joblib')
    (Fr_mean, Fr_sd) = joblib.load(data_dir + rat_name + '_Y_BLAex_nrem_Z.joblib')
    XX.append(X)
    fz = []
    for i in range(Fr_mean.shape[0]):
        z = (F[i] - Fr_mean[i]) / Fr_sd[i]
        fz = np.append(fz, z)
    fz=fz.reshape(Fr_mean.shape[0],-1)
    FFZ.append(fz)

FFFZ=[]
for n in range(len(FFZ)):
    for m in range(FFZ[n].shape[0]):
        FFFZ.append(FFZ[n][m].tolist())

flat_F= []
original_list = FFFZ
for l in original_list:
    for item in l:
        flat_F.append(item)


F_nid=[]
for n in NNid[0]:
    F_nid=np.append(F_nid,FFFZ[n])
F_rid=[]
for n in RRid[0]:
    F_rid=np.append(F_rid,FFFZ[n])

XXX=[]
for n in range(len(XX)):
    for m in range(FFZ[n].shape[0]):
        XXX.append(XX[n])
X_nid=[]
for n in NNid[0]:
    X_nid=np.append(X_nid,XXX[n])
X_rid=[]
for n in RRid[0]:
    X_rid=np.append(X_rid,XXX[n])

x=np.concatenate([X_nid],0)
y=np.concatenate([F_nid],0)
c=np.column_stack([X_nid,F_nid])
c2=c[c[:,0].argsort(),:]

ax3.scatter(x, y, color='cornflowerblue', s=1)
#ax5.set_xlabel('Time(sec)')
#ax3.set_title('BLAex N', fontsize=15)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((x,y), os.path.expanduser(data_dir +'Y_fig_2_b_BLAex_nrem_active_2.joblib'), compress=3)

g0,g1,g2,g3,g4,g5,g6,g7,g8,g9=np.array_split(c2,10,0)
g0mean=np.nanmean(g0,axis=0)
g1mean=np.nanmean(g1,axis=0)
g2mean=np.nanmean(g2,axis=0)
g3mean=np.nanmean(g3,axis=0)
g4mean=np.nanmean(g4,axis=0)
g5mean=np.nanmean(g5,axis=0)
g6mean=np.nanmean(g6,axis=0)
g7mean=np.nanmean(g7,axis=0)
g8mean=np.nanmean(g8,axis=0)
g9mean=np.nanmean(g9,axis=0)

g0std=np.nanstd(g0,axis=0)
g1std=np.nanstd(g1,axis=0)
g2std=np.nanstd(g2,axis=0)
g3std=np.nanstd(g3,axis=0)
g4std=np.nanstd(g4,axis=0)
g5std=np.nanstd(g5,axis=0)
g6std=np.nanstd(g6,axis=0)
g7std=np.nanstd(g7,axis=0)
g8std=np.nanstd(g8,axis=0)
g9std=np.nanstd(g9,axis=0)

g0sem = g0std[1]/np.sqrt(len(g0))
g1sem = g1std[1]/np.sqrt(len(g1))
g2sem = g2std[1]/np.sqrt(len(g2))
g3sem = g3std[1]/np.sqrt(len(g3))
g4sem = g4std[1]/np.sqrt(len(g4))
g5sem = g5std[1]/np.sqrt(len(g5))
g6sem = g6std[1]/np.sqrt(len(g6))
g7sem = g7std[1]/np.sqrt(len(g7))
g8sem = g8std[1]/np.sqrt(len(g8))
g9sem = g9std[1]/np.sqrt(len(g9))

xx=(g0mean[0],g1mean[0],g2mean[0],g3mean[0],g4mean[0],g5mean[0],g6mean[0],g7mean[0],g8mean[0],g9mean[0])
yy=(g0mean[1],g1mean[1],g2mean[1],g3mean[1],g4mean[1],g5mean[1],g6mean[1],g7mean[1],g8mean[1],g9mean[1])
y_err=(g0sem,g1sem,g2sem,g3sem,g4sem,g5sem,g6sem,g7sem,g8sem,g9sem)

a, b = np.polyfit(x, y, 1)
Y2 = a * x + b
np.corrcoef(x,y)[0][1]
#r,p=pearsonr(x, y)
r, p = spearmanr(x, y)
print('y=', a, '*x', '+', b)
print('r=', r)
print('p=', p)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((x,y,xx,yy,y_err,Y2), os.path.expanduser(data_dir +'Y_fig_2_b_bottom_BLAex_nrem_active_2_sp.joblib'), compress=3)

ax6.set_ylim([-0.3, 0.25])
ax6.plot(x,Y2,color='cornflowerblue')
ax6.errorbar(xx,yy,yerr=y_err,fmt='o',color='cornflowerblue')
ax6.text(0.45, 0.8, "r=-0.09**", va='top', ha='left', transform=ax6.transAxes ,fontsize=25)
#ax3.set_xlabel('Time(sec)')

x=np.concatenate([X_rid],0)
y=np.concatenate([F_rid],0)
c=np.column_stack([X_rid,F_rid])
c2=c[c[:,0].argsort(),:]

ax3.scatter(x, y, color='violet', s=1,alpha=0.5)
#ax6.set_xlabel('Time(sec)')
ax3.set_title('BLAex', fontsize=25)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((x,y), os.path.expanduser(data_dir +'Y_fig_2_b_BLAex_rem_active_2.joblib'), compress=3)

g0,g1,g2,g3,g4,g5,g6,g7,g8,g9=np.array_split(c2,10,0)
g0mean=np.nanmean(g0,axis=0)
g1mean=np.nanmean(g1,axis=0)
g2mean=np.nanmean(g2,axis=0)
g3mean=np.nanmean(g3,axis=0)
g4mean=np.nanmean(g4,axis=0)
g5mean=np.nanmean(g5,axis=0)
g6mean=np.nanmean(g6,axis=0)
g7mean=np.nanmean(g7,axis=0)
g8mean=np.nanmean(g8,axis=0)
g9mean=np.nanmean(g9,axis=0)

g0std=np.nanstd(g0,axis=0)
g1std=np.nanstd(g1,axis=0)
g2std=np.nanstd(g2,axis=0)
g3std=np.nanstd(g3,axis=0)
g4std=np.nanstd(g4,axis=0)
g5std=np.nanstd(g5,axis=0)
g6std=np.nanstd(g6,axis=0)
g7std=np.nanstd(g7,axis=0)
g8std=np.nanstd(g8,axis=0)
g9std=np.nanstd(g9,axis=0)

g0sem = g0std[1]/np.sqrt(len(g0))
g1sem = g1std[1]/np.sqrt(len(g1))
g2sem = g2std[1]/np.sqrt(len(g2))
g3sem = g3std[1]/np.sqrt(len(g3))
g4sem = g4std[1]/np.sqrt(len(g4))
g5sem = g5std[1]/np.sqrt(len(g5))
g6sem = g6std[1]/np.sqrt(len(g6))
g7sem = g7std[1]/np.sqrt(len(g7))
g8sem = g8std[1]/np.sqrt(len(g8))
g9sem = g9std[1]/np.sqrt(len(g9))

xx=(g0mean[0],g1mean[0],g2mean[0],g3mean[0],g4mean[0],g5mean[0],g6mean[0],g7mean[0],g8mean[0],g9mean[0])
yy=(g0mean[1],g1mean[1],g2mean[1],g3mean[1],g4mean[1],g5mean[1],g6mean[1],g7mean[1],g8mean[1],g9mean[1])
y_err=(g0sem,g1sem,g2sem,g3sem,g4sem,g5sem,g6sem,g7sem,g8sem,g9sem)

a, b = np.polyfit(x, y, 1)
Y2 = a * x + b
np.corrcoef(x,y)[0][1]
#r,p=pearsonr(x, y)
r, p = spearmanr(x, y)
print('y=', a, '*x', '+', b)
print('r=', r)
print('p=', p)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((x,y,xx,yy,y_err,Y2), os.path.expanduser(data_dir +'Y_fig_2_b_bottom_BLAex_rem_active_2_sp.joblib'), compress=3)

ax6.plot(x,Y2,color='violet')
ax6.errorbar(xx,yy,yerr=y_err,fmt='o',color='violet')
ax6.text(0.45, 0.9, "r=-0.06***", va='top', ha='left', transform=ax6.transAxes ,fontsize=25)
#ax6.set_xlabel('Time(sec)')

plt.subplots_adjust(wspace=0.32)


plt.savefig('test8.pdf')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
pdf=PdfPages("test7.pdf")
fignums=plt.get_fignums()
for fignum in fignums:
    plt.figure(fignum)
    pdf.savefig()
pdf.close()