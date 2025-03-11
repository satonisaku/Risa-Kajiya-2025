import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import os
import sys


fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(1, 6, 1)
ax2 = fig.add_subplot(1, 6, 2)
ax3 = fig.add_subplot(1, 6, 3)
ax4 = fig.add_subplot(1, 6, 4)
ax5 = fig.add_subplot(1, 6, 5)
ax6 = fig.add_subplot(1, 6, 6)

# vCA1
rats = ['duvel190505','hoegaarden181115', 'innis190601','karmeliet190901','leffe200124','maredsous200224','nostrum200304', 'oberon200325']

A = []
A1=[]
A2=[]
A3=[]
A4=[]
A5=[]
A6=[]
A7=[]
A8=[]
A9=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data\\'
    (Fr,frtargetmean1,frtargetmean2, frtargetmean3,frtargetmean4,frtargetmean5,frtargetmean6,frtargetmean7,frtargetmean8,frtargetmean9) = joblib.load(data_dir + rat_name + '_vCA1inh_allcell_color0_50.joblib')
    #A= np.append(A, Fr)
    A1 = np.append(A1,frtargetmean1)
    A2 = np.append(A2, frtargetmean2)
    A3 = np.append(A3, frtargetmean3)
    A4 = np.append(A4, frtargetmean4)
    A5 = np.append(A5, frtargetmean5)
    A6 = np.append(A6, frtargetmean6)
    A7 = np.append(A7, frtargetmean7)
    A8 = np.append(A8, frtargetmean8)
    A9 = np.append(A9, frtargetmean9)


BBinh=np.stack([A1,A2,A3,A4,A5,A6,A7,A8,A9],1)

A = []
A1=[]
A2=[]
A3=[]
A4=[]
A5=[]
A6=[]
A7=[]
A8=[]
A9=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data\\'
    (Fr,frtargetmean1,frtargetmean2, frtargetmean3,frtargetmean4,frtargetmean5,frtargetmean6,frtargetmean7,frtargetmean8,frtargetmean9) = joblib.load(data_dir + rat_name + '_vCA1ex_allcell_color0_50.joblib')
    #A= np.append(A, Fr)
    A1 = np.append(A1,frtargetmean1)
    A2 = np.append(A2, frtargetmean2)
    A3 = np.append(A3, frtargetmean3)
    A4 = np.append(A4, frtargetmean4)
    A5 = np.append(A5, frtargetmean5)
    A6 = np.append(A6, frtargetmean6)
    A7 = np.append(A7, frtargetmean7)
    A8 = np.append(A8, frtargetmean8)
    A9 = np.append(A9, frtargetmean9)


BBex=np.stack([A1,A2,A3,A4,A5,A6,A7,A8,A9],1)

BB=np.concatenate([BBinh,BBex], axis=0)

Data=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (d) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_data_modulation_index_2.joblib')
    Data=np.append(Data,d)


Datain=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (din) = joblib.load(data_dir + rat_name + '_Y_vCA1inh_data_modulation_index_2.joblib')
    Datain = np.append(Datain, din)

#Zscore
def zscore(x, axis = None):
    xmean = x.mean(axis=axis)
    xstd  = np.std(x, axis=axis)
    zscore = (x-xmean)/xstd
    return zscore

minmax=[]
for i in range(BBex.shape[0]):
    zscore(BBex[i])
    minmax = np.append(minmax,zscore(BBex[i]))
Cex=minmax.reshape(BBex.shape[0],-1)


REMex=[]
for i in range (BBex.shape[0]):
    rem=Cex[i][3]+Cex[i][4]+Cex[i][5]
    REMex=np.append(REMex,rem)


Rex=np.argsort(Data)    #modulation index jyun ex


#BB[28]   Rex 23
#BB[30]   Rex 55

minmax=[]
for i in range(BBinh.shape[0]):
    zscore(BBinh[i])
    minmax = np.append(minmax,zscore(BBinh[i]))
Cin=minmax.reshape(BBinh.shape[0],-1)

REMin=[]
for i in range (BBinh.shape[0]):
    rem=Cin[i][3]+Cin[i][4]+Cin[i][5]
    REMin=np.append(REMin,rem)

Rin=np.argsort(Datain)    #modulation index jyun ex

C1=np.concatenate([Cin,Cex], axis=0)
R1=np.concatenate([Rin,Rex+Rin.shape[0]], axis=0)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((C1,R1,BB,Rin), os.path.expanduser(data_dir +'fig_1_c_vCA1.joblib'), compress=3)

X=np.arange(0,10,1)
Y=np.arange(0,BB.shape[0]+1,1)
im2=ax1.pcolorfast(X,Y, C1[R1],cmap='rainbow')
#fig.colorbar(im2,cmap='RdPu',ax=ax1)
#fig.tight_layout()
#ax1.set_title('vCA1ex', fontsize=20)
ax1.set_xticks([1.5,4.5,7.5])
labels=('NREM','REM','NREM')
ax1.set_xticklabels(labels, fontsize=15)
#ax1.set_yticks([0,20,40,60,80])
#labels=('0','20','40','60','80')
#ax1.set_yticklabels(labels, fontsize=10)
ax1.set_ylabel('#cell', fontsize=20)
title = ('vCA1', 'PL5', 'BLA')
ax1.set_title(title[0], fontsize=20)
ax1.axhline(y=Rin.shape[0],color="white")



data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
(all_time_N_id,all_time_R_id)=joblib.load(data_dir + 'Y_vCA1ex_n_r_id_2_50_2.joblib')
NNid = np.where(all_time_N_id == 1)
RRid = np.where(all_time_R_id == 2)


NN1 = []
for n in range(R1.shape[0]):
    for m in range(NNid[0].shape[0]):
        if R1[n] == NNid[0][m]+Rin.shape[0]:
            NN1 = np.append(NN1, n)
NN1 = np.array(NN1, dtype=int)


RR1 = []
for n in range(R1.shape[0]):
    for m in range(RRid[0].shape[0]):
        if R1[n] == RRid[0][m]+Rin.shape[0]:
            RR1 = np.append(RR1, n)
RR1 = np.array(RR1, dtype=int)


n1= np.delete(R1, RR1)
n2=np.delete(n1,NN1)

nn = []
for n in range(R1.shape[0]):
    for m in range(n2.shape[0]):
        if R1[n] == n2[m]:
            nn = np.append(nn, n)
nn = np.array(nn, dtype=int)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
(all_time_N_id,all_time_R_id)=joblib.load(data_dir + 'Y_vCA1inh_n_r_id_2_50_2.joblib')
NNid = np.where(all_time_N_id == 1)
RRid = np.where(all_time_R_id == 2)


NN2 = []
for n in range(R1.shape[0]):
    for m in range(NNid[0].shape[0]):
        if R1[n] == NNid[0][m]:
            NN2 = np.append(NN2, n)
NN2 = np.array(NN2, dtype=int)

RR2 = []
for n in range(R1.shape[0]):
    for m in range(RRid[0].shape[0]):
        if R1[n] == RRid[0][m]:
            RR2 = np.append(RR2, n)
RR2 = np.array(RR2, dtype=int)


n11= np.delete(R1, RR2)
n22=np.delete(n11,NN2)
n33=np.delete(n22,np.arange(Rex.shape[0]))

nn2 = []
for n in range(R1.shape[0]):
    for m in range(n33.shape[0]):
        if R1[n] == n33[m]:
            nn2 = np.append(nn2, n)
nn2 = np.array(nn2, dtype=int)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((R1,NN1,RR1,nn,NN2,RR2,nn2,Rin), os.path.expanduser(data_dir +'fig_1_c_vCA1_label.joblib'), compress=3)

for r in range(R1.shape[0]):
    for m in range(NN1.shape[0]):
        if r == NN1[m]:
            rect = plt.Rectangle((0, r), ((r + 1) - r), 1, fill=True, facecolor='cornflowerblue')
            ax2.add_patch(rect)
    for m in range(RR1.shape[0]):
        if r == RR1[m]:
            rect = plt.Rectangle((0, r), ((r + 1) - r), 1, fill=True, facecolor='hotpink')
            ax2.add_patch(rect)
    for m in range(nn.shape[0]):
        if r == nn[m]:
            rect = plt.Rectangle((0, r), ((r + 1) - r), 1, fill=True, facecolor='grey')
            ax2.add_patch(rect)
    for m in range(NN2.shape[0]):
        if r == NN2[m]:
            rect = plt.Rectangle((0, r), ((r + 1) - r), 1, fill=True, facecolor='cornflowerblue')
            ax2.add_patch(rect)
    for m in range(RR2.shape[0]):
        if r == RR2[m]:
            rect = plt.Rectangle((0, r), ((r + 1) - r), 1, fill=True, facecolor='hotpink')
            ax2.add_patch(rect)
    for m in range(nn2.shape[0]):
        if r == nn2[m]:
            rect = plt.Rectangle((0, r), ((r + 1) - r), 1, fill=True, facecolor='grey')
            ax2.add_patch(rect)
ax2.set_xlim((0, 1))
ax2.set_ylim((0, R1.shape[0]))
ax2.set_axis_off()
ax2.axhline(y=Rin.shape[0],color="white")


# PL5
rats = ['hoegaarden181115', 'innis190601', 'jever190814', 'karmeliet190901', 'leffe200124', 'maredsous200224','nostrum200304', 'oberon200325']

A = []
A1=[]
A2=[]
A3=[]
A4=[]
A5=[]
A6=[]
A7=[]
A8=[]
A9=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data\\'
    (Fr,frtargetmean1,frtargetmean2, frtargetmean3,frtargetmean4,frtargetmean5,frtargetmean6,frtargetmean7,frtargetmean8,frtargetmean9) = joblib.load(data_dir + rat_name + '_PL5inh_allcell_color0_50.joblib')
    #A= np.append(A, Fr)
    A1 = np.append(A1,frtargetmean1)
    A2 = np.append(A2, frtargetmean2)
    A3 = np.append(A3, frtargetmean3)
    A4 = np.append(A4, frtargetmean4)
    A5 = np.append(A5, frtargetmean5)
    A6 = np.append(A6, frtargetmean6)
    A7 = np.append(A7, frtargetmean7)
    A8 = np.append(A8, frtargetmean8)
    A9 = np.append(A9, frtargetmean9)

BBinh=np.stack([A1,A2,A3,A4,A5,A6,A7,A8,A9],1)

A = []
A1=[]
A2=[]
A3=[]
A4=[]
A5=[]
A6=[]
A7=[]
A8=[]
A9=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data\\'
    (Fr,frtargetmean1,frtargetmean2, frtargetmean3,frtargetmean4,frtargetmean5,frtargetmean6,frtargetmean7,frtargetmean8,frtargetmean9) = joblib.load(data_dir + rat_name + '_PL5ex_allcell_color0_50.joblib')
    #A= np.append(A, Fr)
    A1 = np.append(A1,frtargetmean1)
    A2 = np.append(A2, frtargetmean2)
    A3 = np.append(A3, frtargetmean3)
    A4 = np.append(A4, frtargetmean4)
    A5 = np.append(A5, frtargetmean5)
    A6 = np.append(A6, frtargetmean6)
    A7 = np.append(A7, frtargetmean7)
    A8 = np.append(A8, frtargetmean8)
    A9 = np.append(A9, frtargetmean9)

BBex=np.stack([A1,A2,A3,A4,A5,A6,A7,A8,A9],1)

BB=np.concatenate([BBinh,BBex], axis=0)

Data=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (d) = joblib.load(data_dir + rat_name + '_Y_PL5ex_data_modulation_index_2.joblib')
    Data=np.append(Data,d)

Datain=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (din) = joblib.load(data_dir + rat_name + '_Y_PL5inh_data_modulation_index_2.joblib')
    Datain = np.append(Datain, din)

#Zscore
def zscore(x, axis = None):
    xmean = x.mean(axis=axis)
    xstd  = np.std(x, axis=axis)
    zscore = (x-xmean)/xstd
    return zscore

minmax=[]
for i in range(BBex.shape[0]):
    zscore(BBex[i])
    minmax = np.append(minmax,zscore(BBex[i]))
Cex=minmax.reshape(BBex.shape[0],-1)


REMex=[]
for i in range (BBex.shape[0]):
    rem=Cex[i][3]+Cex[i][4]+Cex[i][5]
    REMex=np.append(REMex,rem)


Rex=np.argsort(Data)    #modulation index jyun ex


minmax=[]
for i in range(BBinh.shape[0]):
    zscore(BBinh[i])
    minmax = np.append(minmax,zscore(BBinh[i]))
Cin=minmax.reshape(BBinh.shape[0],-1)

REMin=[]
for i in range (BBinh.shape[0]):
    rem=Cin[i][3]+Cin[i][4]+Cin[i][5]
    REMin=np.append(REMin,rem)

Rin=np.argsort(Datain)    #modulation index jyun ex

C1=np.concatenate([Cin,Cex], axis=0)
R1=np.concatenate([Rin,Rex+Rin.shape[0]], axis=0)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((C1,R1,BB,Rin), os.path.expanduser(data_dir +'fig_1_c_PL5.joblib'), compress=3)

X=np.arange(0,10,1)
Y=np.arange(0,BB.shape[0]+1,1)
im2=ax3.pcolorfast(X,Y, C1[R1],cmap='rainbow')
#fig.colorbar(im2,cmap='RdPu',ax=ax1)
#fig.tight_layout()
#ax1.set_title('vCA1ex', fontsize=20)
ax3.set_xticks([1.5,4.5,7.5])
labels=('NREM','REM','NREM')
ax3.set_xticklabels(labels, fontsize=15)
#ax1.set_yticks([0,20,40,60,80])
#labels=('0','20','40','60','80')
#ax1.set_yticklabels(labels, fontsize=10)
#ax3.set_ylabel('#cell', fontsize=20)
title = ('vCA1', 'PL5', 'BLA')
ax3.set_title(title[1], fontsize=20)
ax3.axhline(y=Rin.shape[0],color="white")



data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
(all_time_N_id,all_time_R_id)=joblib.load(data_dir + 'Y_PL5ex_n_r_id_2_50_2.joblib')
NNid = np.where(all_time_N_id == 1)
RRid = np.where(all_time_R_id == 2)


NN1 = []
for n in range(R1.shape[0]):
    for m in range(NNid[0].shape[0]):
        if R1[n] == NNid[0][m]+Rin.shape[0]:
            NN1 = np.append(NN1, n)
NN1 = np.array(NN1, dtype=int)


RR1 = []
for n in range(R1.shape[0]):
    for m in range(RRid[0].shape[0]):
        if R1[n] == RRid[0][m]+Rin.shape[0]:
            RR1 = np.append(RR1, n)
RR1 = np.array(RR1, dtype=int)


n1= np.delete(R1, RR1)
n2=np.delete(n1,NN1)

nn = []
for n in range(R1.shape[0]):
    for m in range(n2.shape[0]):
        if R1[n] == n2[m]:
            nn = np.append(nn, n)
nn = np.array(nn, dtype=int)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
(all_time_N_id,all_time_R_id)=joblib.load(data_dir + 'Y_PL5inh_n_r_id_2_50_2.joblib')
NNid = np.where(all_time_N_id == 1)
RRid = np.where(all_time_R_id == 2)


NN2 = []
for n in range(R1.shape[0]):
    for m in range(NNid[0].shape[0]):
        if R1[n] == NNid[0][m]:
            NN2 = np.append(NN2, n)
NN2 = np.array(NN2, dtype=int)

RR2 = []
for n in range(R1.shape[0]):
    for m in range(RRid[0].shape[0]):
        if R1[n] == RRid[0][m]:
            RR2 = np.append(RR2, n)
RR2 = np.array(RR2, dtype=int)


n11= np.delete(R1, RR2)
n22=np.delete(n11,NN2)
n33=np.delete(n22,np.arange(Rex.shape[0]))

nn2 = []
for n in range(R1.shape[0]):
    for m in range(n33.shape[0]):
        if R1[n] == n33[m]:
            nn2 = np.append(nn2, n)
nn2 = np.array(nn2, dtype=int)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((R1,NN1,RR1,nn,NN2,RR2,nn2,Rin), os.path.expanduser(data_dir +'fig_1_c_PL5_label.joblib'), compress=3)

for r in range(R1.shape[0]):
    for m in range(NN1.shape[0]):
        if r == NN1[m]:
            rect = plt.Rectangle((0, r), ((r + 1) - r), 1, fill=True, facecolor='cornflowerblue')
            ax4.add_patch(rect)
    for m in range(RR1.shape[0]):
        if r == RR1[m]:
            rect = plt.Rectangle((0, r), ((r + 1) - r), 1, fill=True, facecolor='hotpink')
            ax4.add_patch(rect)
    for m in range(nn.shape[0]):
        if r == nn[m]:
            rect = plt.Rectangle((0, r), ((r + 1) - r), 1, fill=True, facecolor='grey')
            ax4.add_patch(rect)
    for m in range(NN2.shape[0]):
        if r == NN2[m]:
            rect = plt.Rectangle((0, r), ((r + 1) - r), 1, fill=True, facecolor='cornflowerblue')
            ax4.add_patch(rect)
    for m in range(RR2.shape[0]):
        if r == RR2[m]:
            rect = plt.Rectangle((0, r), ((r + 1) - r), 1, fill=True, facecolor='hotpink')
            ax4.add_patch(rect)
    for m in range(nn2.shape[0]):
        if r == nn2[m]:
            rect = plt.Rectangle((0, r), ((r + 1) - r), 1, fill=True, facecolor='grey')
            ax4.add_patch(rect)
ax4.set_xlim((0, 1))
ax4.set_ylim((0, R1.shape[0]))
ax4.set_axis_off()
ax4.axhline(y=Rin.shape[0],color="white")

# BLA
rats = ['achel180320', 'booyah180430', 'duvel190505', 'estrella180808', 'guiness181002', 'hoegaarden181115', 'innis190601','jever190814', 'leffe200124', 'maredsous200224', 'nostrum200304', 'oberon200325']

A = []
A1=[]
A2=[]
A3=[]
A4=[]
A5=[]
A6=[]
A7=[]
A8=[]
A9=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data\\'
    (Fr,frtargetmean1,frtargetmean2, frtargetmean3,frtargetmean4,frtargetmean5,frtargetmean6,frtargetmean7,frtargetmean8,frtargetmean9) = joblib.load(data_dir + rat_name + '_BLAinh_allcell_color0_50.joblib')
    #A= np.append(A, Fr)
    A1 = np.append(A1,frtargetmean1)
    A2 = np.append(A2, frtargetmean2)
    A3 = np.append(A3, frtargetmean3)
    A4 = np.append(A4, frtargetmean4)
    A5 = np.append(A5, frtargetmean5)
    A6 = np.append(A6, frtargetmean6)
    A7 = np.append(A7, frtargetmean7)
    A8 = np.append(A8, frtargetmean8)
    A9 = np.append(A9, frtargetmean9)

BBinh=np.stack([A1,A2,A3,A4,A5,A6,A7,A8,A9],1)

A = []
A1 = []
A2 = []
A3 = []
A4 = []
A5 = []
A6 = []
A7 = []
A8 = []
A9 = []
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data\\'
    (Fr, frtargetmean1, frtargetmean2, frtargetmean3, frtargetmean4, frtargetmean5, frtargetmean6, frtargetmean7,frtargetmean8, frtargetmean9) = joblib.load(data_dir + rat_name + '_BLAex_allcell_color0_50.joblib')
    #A = np.append(A, Fr)
    A1 = np.append(A1, frtargetmean1)
    A2 = np.append(A2, frtargetmean2)
    A3 = np.append(A3, frtargetmean3)
    A4 = np.append(A4, frtargetmean4)
    A5 = np.append(A5, frtargetmean5)
    A6 = np.append(A6, frtargetmean6)
    A7 = np.append(A7, frtargetmean7)
    A8 = np.append(A8, frtargetmean8)
    A9 = np.append(A9, frtargetmean9)

BBex=np.stack([A1,A2,A3,A4,A5,A6,A7,A8,A9],1)

BB=np.concatenate([BBinh,BBex], axis=0)

Data=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (d) = joblib.load(data_dir + rat_name + '_Y_BLAex_data_modulation_index_2.joblib')
    Data=np.append(Data,d)

Datain=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (din) = joblib.load(data_dir + rat_name + '_Y_BLAinh_data_modulation_index_2.joblib')
    Datain = np.append(Datain, din)


#Zscore
def zscore(x, axis = None):
    xmean = x.mean(axis=axis)
    xstd  = np.std(x, axis=axis)
    zscore = (x-xmean)/xstd
    return zscore

minmax=[]
for i in range(BBex.shape[0]):
    zscore(BBex[i])
    minmax = np.append(minmax,zscore(BBex[i]))
Cex=minmax.reshape(BBex.shape[0],-1)


REMex=[]
for i in range (BBex.shape[0]):
    rem=Cex[i][3]+Cex[i][4]+Cex[i][5]
    REMex=np.append(REMex,rem)


Rex=np.argsort(Data)    #modulation index jyun ex


minmax=[]
for i in range(BBinh.shape[0]):
    zscore(BBinh[i])
    minmax = np.append(minmax,zscore(BBinh[i]))
Cin=minmax.reshape(BBinh.shape[0],-1)

REMin=[]
for i in range (BBinh.shape[0]):
    rem=Cin[i][3]+Cin[i][4]+Cin[i][5]
    REMin=np.append(REMin,rem)

Rin=np.argsort(Datain)    #modulation index jyun ex

C1=np.concatenate([Cin,Cex], axis=0)
R1=np.concatenate([Rin,Rex+Rin.shape[0]], axis=0)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((C1,R1,BB,Rin), os.path.expanduser(data_dir +'fig_1_c_BLA.joblib'), compress=3)

X=np.arange(0,10,1)
Y=np.arange(0,BB.shape[0]+1,1)
im2=ax5.pcolorfast(X,Y, C1[R1],cmap='rainbow')
#fig.colorbar(im2,cmap='RdPu',ax=ax1)
#fig.tight_layout()
#ax1.set_title('vCA1ex', fontsize=20)
ax5.set_xticks([1.5,4.5,7.5])
labels=('NREM','REM','NREM')
ax5.set_xticklabels(labels, fontsize=15)
#ax1.set_yticks([0,20,40,60,80])
#labels=('0','20','40','60','80')
#ax1.set_yticklabels(labels, fontsize=10)
#ax5.set_ylabel('#cell', fontsize=20)
title = ('vCA1', 'PL5', 'BLA')
ax5.set_title(title[2], fontsize=20)
ax5.axhline(y=Rin.shape[0],color="white")



data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
(all_time_N_id,all_time_R_id)=joblib.load(data_dir + 'Y_BLAex_n_r_id_2_50_2.joblib')
NNid = np.where(all_time_N_id == 1)
RRid = np.where(all_time_R_id == 2)


NN1 = []
for n in range(R1.shape[0]):
    for m in range(NNid[0].shape[0]):
        if R1[n] == NNid[0][m]+Rin.shape[0]:
            NN1 = np.append(NN1, n)
NN1 = np.array(NN1, dtype=int)


RR1 = []
for n in range(R1.shape[0]):
    for m in range(RRid[0].shape[0]):
        if R1[n] == RRid[0][m]+Rin.shape[0]:
            RR1 = np.append(RR1, n)
RR1 = np.array(RR1, dtype=int)


n1= np.delete(R1, RR1)
n2=np.delete(n1,NN1)

nn = []
for n in range(R1.shape[0]):
    for m in range(n2.shape[0]):
        if R1[n] == n2[m]:
            nn = np.append(nn, n)
nn = np.array(nn, dtype=int)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
(all_time_N_id,all_time_R_id)=joblib.load(data_dir + 'Y_BLAinh_n_r_id_2_50_2.joblib')
NNid = np.where(all_time_N_id == 1)
RRid = np.where(all_time_R_id == 2)


NN2 = []
for n in range(R1.shape[0]):
    for m in range(NNid[0].shape[0]):
        if R1[n] == NNid[0][m]:
            NN2 = np.append(NN2, n)
NN2 = np.array(NN2, dtype=int)

RR2 = []
for n in range(R1.shape[0]):
    for m in range(RRid[0].shape[0]):
        if R1[n] == RRid[0][m]:
            RR2 = np.append(RR2, n)
RR2 = np.array(RR2, dtype=int)


n11= np.delete(R1, RR2)
n22=np.delete(n11,NN2)
n33=np.delete(n22,np.arange(Rex.shape[0]))

nn2 = []
for n in range(R1.shape[0]):
    for m in range(n33.shape[0]):
        if R1[n] == n33[m]:
            nn2 = np.append(nn2, n)
nn2 = np.array(nn2, dtype=int)


data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((R1,NN1,RR1,nn,NN2,RR2,nn2,Rin), os.path.expanduser(data_dir +'fig_1_c_BLA_label.joblib'), compress=3)


for r in range(R1.shape[0]):
    for m in range(NN1.shape[0]):
        if r == NN1[m]:
            rect = plt.Rectangle((0, r), ((r + 1) - r), 1, fill=True, facecolor='cornflowerblue')
            ax6.add_patch(rect)
    for m in range(RR1.shape[0]):
        if r == RR1[m]:
            rect = plt.Rectangle((0, r), ((r + 1) - r), 1, fill=True, facecolor='hotpink')
            ax6.add_patch(rect)
    for m in range(nn.shape[0]):
        if r == nn[m]:
            rect = plt.Rectangle((0, r), ((r + 1) - r), 1, fill=True, facecolor='grey')
            ax6.add_patch(rect)
    for m in range(NN2.shape[0]):
        if r == NN2[m]:
            rect = plt.Rectangle((0, r), ((r + 1) - r), 1, fill=True, facecolor='cornflowerblue')
            ax6.add_patch(rect)
    for m in range(RR2.shape[0]):
        if r == RR2[m]:
            rect = plt.Rectangle((0, r), ((r + 1) - r), 1, fill=True, facecolor='hotpink')
            ax6.add_patch(rect)
    for m in range(nn2.shape[0]):
        if r == nn2[m]:
            rect = plt.Rectangle((0, r), ((r + 1) - r), 1, fill=True, facecolor='grey')
            ax6.add_patch(rect)
ax6.set_xlim((0, 1))
ax6.set_ylim((0, R1.shape[0]))
ax6.set_axis_off()
ax6.axhline(y=Rin.shape[0],color="white")

plt.subplots_adjust(wspace=0.4,hspace=0.45)



clb = plt.colorbar(im2, ax=ax6)