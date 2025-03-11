import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.stats import spearmanr

#fig = plt.figure(figsize=(24, 8))
fig = plt.figure(figsize=(18, 12))
ax1 = fig.add_subplot(2, 3, 1)
ax2 = fig.add_subplot(2, 3, 2)
ax3 = fig.add_subplot(2, 3, 3)
ax4 = fig.add_subplot(2, 3, 4)
ax5 = fig.add_subplot(2, 3, 5)
ax6 = fig.add_subplot(2, 3, 6)

# vCA1
#rats = [ 'hoegaarden181115', 'innis190601', 'karmeliet190901', 'maredsous200224','nostrum200304', 'oberon200325']

# vCA1
rats = ['duvel190505','hoegaarden181115', 'innis190601','karmeliet190901','leffe200124','maredsous200224', 'nostrum200304','oberon200325']


MM=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    data =joblib.load(data_dir + rat_name + '_Y_vCA1ex_data_modulation_index_2.joblib')
    data2 =joblib.load(data_dir + rat_name + '_Y_vCA1ex_data_modulation_index_2.joblib')
    M=[]
    for n in range(data.shape[0]):
        for m in range (data2.shape[0]):
            M=np.append(M,data[n]*data2[m])
    com = M.reshape(-1, data.shape[0])
    MM = np.append(MM,com[np.triu_indices(com.shape[0], k=1)])

X=MM

Y=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr, _) = joblib.load(data_dir + rat_name + '_vCA1ex_BLAex_N_N_SWR_corr.joblib')
    (RR) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_coactivity_all_nrem_swr.joblib')
    #(c,s) = joblib.load(data_dir + rat_name + '_vCA1ex_PL5ex_shock_mod_coactivity_conditioning.joblib')
    #X=np.append(X,c)
    Y=np.append(Y,RR)

from scipy.stats import pearsonr
ax1.scatter(X,Y, color='blue',s=10)
ax1.set_xlabel('modulation index*modulation index', fontsize=15)
ax1.set_ylabel('coactivity Zscore', fontsize=15)
ax1.text(0.95, 0.95, "r=0.03", va='top', ha='right', transform=ax1.transAxes, fontsize=20)

x=X
y=Y
idx=np.isfinite(x)&np.isfinite(y)
a, b = np.polyfit(x[idx], y[idx], 1)
Y1 = a * x + b
np.corrcoef(x[idx], y[idx])[0][1]
r, p = spearmanr(x[idx], y[idx])
print('y=', a, '*x', '+', b)
print('r=', r)
print('p=', p)
ax1.plot(x,Y1,color='black')

#data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
#joblib.dump((X,Y), os.path.expanduser(data_dir +'Y_fig_3_e_bottom_vCA1ex_all_2.joblib'), compress=3)
#joblib.dump((x,Y1), os.path.expanduser(data_dir +'Y_fig_3_e_bottom_vCA1ex_2_spearman.joblib'), compress=3)

MM=[]
C1=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    data =joblib.load(data_dir + rat_name + '_Y_vCA1ex_data_modulation_index_2.joblib')
    data2 =joblib.load(data_dir + rat_name + '_Y_vCA1ex_data_modulation_index_2.joblib')
    M=[]
    for n in range(data.shape[0]):
        for m in range (data2.shape[0]):
            M=np.append(M,data[n]*data2[m])
    com = M.reshape(-1, data.shape[0])
    MM = com[np.triu_indices(com.shape[0], k=1)]
    #(Corr) = joblib.load(data_dir + rat_name + '_vCA1ex_coactivity_all_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_vCA1ex_NN_RR_NR_RN.joblib')
    vCA1_NN = np.where(NN == 1)[0]
    C1=np.append(C1,MM[vCA1_NN])

YC1=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr, _) = joblib.load(data_dir + rat_name + '_vCA1ex_BLAex_N_N_SWR_corr.joblib')
    (Y) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_coactivity_all_nrem_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_vCA1ex_NN_RR_NR_RN.joblib')
    vCA1_NN = np.where(NN == 1)[0]
    YC1=np.append(YC1,Y[vCA1_NN])

MM=[]
C2=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    data =joblib.load(data_dir + rat_name + '_Y_vCA1ex_data_modulation_index_2.joblib')
    data2 =joblib.load(data_dir + rat_name + '_Y_vCA1ex_data_modulation_index_2.joblib')
    M=[]
    for n in range(data.shape[0]):
        for m in range (data2.shape[0]):
            M=np.append(M,data[n]*data2[m])
    com = M.reshape(-1, data.shape[0])
    MM = com[np.triu_indices(com.shape[0], k=1)]
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr) = joblib.load(data_dir + rat_name + '_vCA1ex_coactivity_all_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_vCA1ex_NN_RR_NR_RN.joblib')
    vCA1_RR = np.where(RR == 4)[0]
    C2=np.append(C2,MM[vCA1_RR])

YC2=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr, _) = joblib.load(data_dir + rat_name + '_vCA1ex_BLAex_N_N_SWR_corr.joblib')
    (Y) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_coactivity_all_nrem_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_vCA1ex_NN_RR_NR_RN.joblib')
    vCA1_RR = np.where(RR == 4)[0]
    YC2=np.append(YC2,Y[vCA1_RR])


MM=[]
nr=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    data =joblib.load(data_dir + rat_name + '_Y_vCA1ex_data_modulation_index_2.joblib')
    data2 =joblib.load(data_dir + rat_name + '_Y_vCA1ex_data_modulation_index_2.joblib')
    M=[]
    for n in range(data.shape[0]):
        for m in range (data2.shape[0]):
            M=np.append(M,data[n]*data2[m])
    com = M.reshape(-1, data.shape[0])
    MM = com[np.triu_indices(com.shape[0], k=1)]
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr) = joblib.load(data_dir + rat_name + '_vCA1ex_coactivity_all_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_vCA1ex_NN_RR_NR_RN.joblib')
    vCA1_NR = np.where(NR == 2)[0]
    nr=np.append(nr,MM[vCA1_NR])

Ynr=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr, _) = joblib.load(data_dir + rat_name + '_vCA1ex_BLAex_N_N_SWR_corr.joblib')
    (Y) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_coactivity_all_nrem_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_vCA1ex_NN_RR_NR_RN.joblib')
    vCA1_NR = np.where(NR == 2)[0]
    Ynr=np.append(Ynr,Y[vCA1_NR])


rn=[]
for rat_name in rats:
    MM = []
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    data = joblib.load(data_dir + rat_name + '_Y_vCA1ex_data_modulation_index_2.joblib')
    data2 = joblib.load(data_dir + rat_name + '_Y_vCA1ex_data_modulation_index_2.joblib')
    M = []
    for n in range(data.shape[0]):
        for m in range(data2.shape[0]):
            M = np.append(M, data[n] * data2[m])
    com = M.reshape(-1, data.shape[0])
    MM = com[np.triu_indices(com.shape[0], k=1)]
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr) = joblib.load(data_dir + rat_name + '_vCA1ex_coactivity_all_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_vCA1ex_NN_RR_NR_RN.joblib')
    vCA1_RN = np.where(RN == 2)[0]
    rn=np.append(rn,MM[vCA1_RN])

Yrn=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr, _) = joblib.load(data_dir + rat_name + '_vCA1ex_BLAex_N_N_SWR_corr.joblib')
    (Y) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_coactivity_all_nrem_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_vCA1ex_NN_RR_NR_RN.joblib')
    vCA1_RN = np.where(RN == 2)[0]
    Yrn=np.append(Yrn,Y[vCA1_RN])

C3=np.append(nr,rn)
YC3=np.append(Ynr,Yrn)

MM=[]
C4=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    data =joblib.load(data_dir + rat_name + '_Y_vCA1ex_data_modulation_index_2.joblib')
    data2 =joblib.load(data_dir + rat_name + '_Y_vCA1ex_data_modulation_index_2.joblib')
    M=[]
    for n in range(data.shape[0]):
        for m in range (data2.shape[0]):
            M=np.append(M,data[n]*data2[m])
    com = M.reshape(-1, data.shape[0])
    MM = com[np.triu_indices(com.shape[0], k=1)]
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr) = joblib.load(data_dir + rat_name + '_vCA1ex_coactivity_all_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_vCA1ex_NN_RR_NR_RN.joblib')
    vCA1_nn = (np.where((NN == 0) & (RR == 0) & (NR == 0) & (RN == 0)))[0]
    C4=np.append(C4,MM[vCA1_nn])

YC4=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr, _) = joblib.load(data_dir + rat_name + '_vCA1ex_BLAex_N_N_SWR_corr.joblib')
    (Y) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_coactivity_all_nrem_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_vCA1ex_NN_RR_NR_RN.joblib')
    vCA1_nn = (np.where((NN == 0) & (RR == 0) & (NR == 0) & (RN == 0)))[0]
    YC4=np.append(YC4,Y[vCA1_nn])

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((C1,YC1), os.path.expanduser(data_dir +'Y_fig_4_a_bottom_vCA1ex_NN.joblib'), compress=3)
joblib.dump((C2,YC2), os.path.expanduser(data_dir +'Y_fig_4_a_bottom_vCA1ex_RR.joblib'), compress=3)
joblib.dump((C3,YC3), os.path.expanduser(data_dir +'Y_fig_4_a_bottom_vCA1ex_NR.joblib'), compress=3)

joblib.dump((C4,YC4), os.path.expanduser(data_dir +'Y_fig_4_a_bottom_vCA1ex_nosignificant.joblib'), compress=3)


# PL5
rats = ['hoegaarden181115', 'innis190601', 'jever190814', 'karmeliet190901', 'leffe200124', 'maredsous200224',
        'nostrum200304', 'oberon200325']

Y=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr, _) = joblib.load(data_dir + rat_name + '_vCA1ex_BLAex_N_N_SWR_corr.joblib')
    (RR) = joblib.load(data_dir + rat_name + '_Y_PL5ex_coactivity_all_nrem_swr.joblib')
    #(c,s) = joblib.load(data_dir + rat_name + '_vCA1ex_PL5ex_shock_mod_coactivity_conditioning.joblib')
    #X=np.append(X,c)
    Y=np.append(Y,RR)

MM=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    data =joblib.load(data_dir + rat_name + '_Y_PL5ex_data_modulation_index_2.joblib')
    data2 =joblib.load(data_dir + rat_name + '_Y_PL5ex_data_modulation_index_2.joblib')
    M=[]
    for n in range(data.shape[0]):
        for m in range (data2.shape[0]):
            M=np.append(M,data[n]*data2[m])
    com = M.reshape(-1, data.shape[0])
    MM = np.append(MM,com[np.triu_indices(com.shape[0], k=1)])

X=MM

from scipy.stats import pearsonr
ax2.scatter(X,Y, color='blue',s=10)
ax2.set_xlabel('modulation index*modulation index', fontsize=15)
ax2.set_ylabel('coactivity Zscore', fontsize=15)
ax2.text(0.95, 0.95, "r=0.19***", va='top', ha='right', transform=ax2.transAxes, fontsize=20)

x=X
y=Y
idx=np.isfinite(x)&np.isfinite(y)
a, b = np.polyfit(x[idx], y[idx], 1)
Y1 = a * x + b
np.corrcoef(x[idx], y[idx])[0][1]
r, p = spearmanr(x[idx], y[idx])
print('y=', a, '*x', '+', b)
print('r=', r)
print('p=', p)
ax2.plot(x,Y1,color='black')

#data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
#joblib.dump((X,Y), os.path.expanduser(data_dir +'Y_fig_3_e_bottom_PL5ex_all_2.joblib'), compress=3)
#joblib.dump((x,Y1), os.path.expanduser(data_dir +'Y_fig_3_e_bottom_PL5ex_2.joblib'), compress=3)

MM=[]
C1=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    data =joblib.load(data_dir + rat_name + '_Y_PL5ex_data_modulation_index_2.joblib')
    data2 =joblib.load(data_dir + rat_name + '_Y_PL5ex_data_modulation_index_2.joblib')
    M=[]
    for n in range(data.shape[0]):
        for m in range (data2.shape[0]):
            M=np.append(M,data[n]*data2[m])
    com = M.reshape(-1, data.shape[0])
    MM = com[np.triu_indices(com.shape[0], k=1)]
    #(Corr) = joblib.load(data_dir + rat_name + '_vCA1ex_coactivity_all_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_PL5ex_NN_RR_NR_RN.joblib')
    vCA1_NN = np.where(NN == 1)[0]
    C1=np.append(C1,MM[vCA1_NN])

YC1=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr, _) = joblib.load(data_dir + rat_name + '_vCA1ex_BLAex_N_N_SWR_corr.joblib')
    (Y) = joblib.load(data_dir + rat_name + '_Y_PL5ex_coactivity_all_nrem_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_PL5ex_NN_RR_NR_RN.joblib')
    vCA1_NN = np.where(NN == 1)[0]
    YC1=np.append(YC1,Y[vCA1_NN])

MM=[]
C2=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    data =joblib.load(data_dir + rat_name + '_Y_PL5ex_data_modulation_index_2.joblib')
    data2 =joblib.load(data_dir + rat_name + '_Y_PL5ex_data_modulation_index_2.joblib')
    M=[]
    for n in range(data.shape[0]):
        for m in range (data2.shape[0]):
            M=np.append(M,data[n]*data2[m])
    com = M.reshape(-1, data.shape[0])
    MM = com[np.triu_indices(com.shape[0], k=1)]
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr) = joblib.load(data_dir + rat_name + '_vCA1ex_coactivity_all_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_PL5ex_NN_RR_NR_RN.joblib')
    vCA1_RR = np.where(RR == 4)[0]
    C2=np.append(C2,MM[vCA1_RR])

YC2=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr, _) = joblib.load(data_dir + rat_name + '_vCA1ex_BLAex_N_N_SWR_corr.joblib')
    (Y) = joblib.load(data_dir + rat_name + '_Y_PL5ex_coactivity_all_nrem_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_PL5ex_NN_RR_NR_RN.joblib')
    vCA1_RR = np.where(RR == 4)[0]
    YC2=np.append(YC2,Y[vCA1_RR])


MM=[]
nr=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    data =joblib.load(data_dir + rat_name + '_Y_PL5ex_data_modulation_index_2.joblib')
    data2 =joblib.load(data_dir + rat_name + '_Y_PL5ex_data_modulation_index_2.joblib')
    M=[]
    for n in range(data.shape[0]):
        for m in range (data2.shape[0]):
            M=np.append(M,data[n]*data2[m])
    com = M.reshape(-1, data.shape[0])
    MM = com[np.triu_indices(com.shape[0], k=1)]
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr) = joblib.load(data_dir + rat_name + '_vCA1ex_coactivity_all_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_PL5ex_NN_RR_NR_RN.joblib')
    vCA1_NR = np.where(NR == 2)[0]
    nr=np.append(nr,MM[vCA1_NR])

Ynr=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr, _) = joblib.load(data_dir + rat_name + '_vCA1ex_BLAex_N_N_SWR_corr.joblib')
    (Y) = joblib.load(data_dir + rat_name + '_Y_PL5ex_coactivity_all_nrem_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_PL5ex_NN_RR_NR_RN.joblib')
    vCA1_NR = np.where(NR == 2)[0]
    Ynr=np.append(Ynr,Y[vCA1_NR])


rn=[]
for rat_name in rats:
    MM = []
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    data = joblib.load(data_dir + rat_name + '_Y_PL5ex_data_modulation_index_2.joblib')
    data2 = joblib.load(data_dir + rat_name + '_Y_PL5ex_data_modulation_index_2.joblib')
    M = []
    for n in range(data.shape[0]):
        for m in range(data2.shape[0]):
            M = np.append(M, data[n] * data2[m])
    com = M.reshape(-1, data.shape[0])
    MM = com[np.triu_indices(com.shape[0], k=1)]
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr) = joblib.load(data_dir + rat_name + '_vCA1ex_coactivity_all_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_PL5ex_NN_RR_NR_RN.joblib')
    vCA1_RN = np.where(RN == 2)[0]
    rn=np.append(rn,MM[vCA1_RN])

Yrn=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr, _) = joblib.load(data_dir + rat_name + '_vCA1ex_BLAex_N_N_SWR_corr.joblib')
    (Y) = joblib.load(data_dir + rat_name + '_Y_PL5ex_coactivity_all_nrem_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_PL5ex_NN_RR_NR_RN.joblib')
    vCA1_RN = np.where(RN == 2)[0]
    Yrn=np.append(Yrn,Y[vCA1_RN])

C3=np.append(nr,rn)
YC3=np.append(Ynr,Yrn)

MM=[]
C4=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    data =joblib.load(data_dir + rat_name + '_Y_PL5ex_data_modulation_index_2.joblib')
    data2 =joblib.load(data_dir + rat_name + '_Y_PL5ex_data_modulation_index_2.joblib')
    M=[]
    for n in range(data.shape[0]):
        for m in range (data2.shape[0]):
            M=np.append(M,data[n]*data2[m])
    com = M.reshape(-1, data.shape[0])
    MM = com[np.triu_indices(com.shape[0], k=1)]
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr) = joblib.load(data_dir + rat_name + '_vCA1ex_coactivity_all_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_PL5ex_NN_RR_NR_RN.joblib')
    vCA1_nn = (np.where((NN == 0) & (RR == 0) & (NR == 0) & (RN == 0)))[0]
    C4=np.append(C4,MM[vCA1_nn])

YC4=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr, _) = joblib.load(data_dir + rat_name + '_vCA1ex_BLAex_N_N_SWR_corr.joblib')
    (Y) = joblib.load(data_dir + rat_name + '_Y_PL5ex_coactivity_all_nrem_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_PL5ex_NN_RR_NR_RN.joblib')
    vCA1_nn = (np.where((NN == 0) & (RR == 0) & (NR == 0) & (RN == 0)))[0]
    YC4=np.append(YC4,Y[vCA1_nn])


data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((C1,YC1), os.path.expanduser(data_dir +'Y_fig_4_a_bottom_PL5ex_NN.joblib'), compress=3)
joblib.dump((C2,YC2), os.path.expanduser(data_dir +'Y_fig_4_a_bottom_PL5ex_RR.joblib'), compress=3)
joblib.dump((C3,YC3), os.path.expanduser(data_dir +'Y_fig_4_a_bottom_PL5ex_NR.joblib'), compress=3)

joblib.dump((C4,YC4), os.path.expanduser(data_dir +'Y_fig_4_a_bottom_PL5ex_nosignificant.joblib'), compress=3)

# BLA-BLA
rats = ['guiness181002', 'hoegaarden181115', 'innis190601', 'maredsous200224', 'nostrum200304', 'oberon200325']

# BLA
rats = [ 'duvel190505', 'estrella180808', 'guiness181002', 'hoegaarden181115','innis190601', 'leffe200124', 'maredsous200224', 'nostrum200304', 'oberon200325']


Y=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr, _) = joblib.load(data_dir + rat_name + '_vCA1ex_BLAex_N_N_SWR_corr.joblib')
    (RR) = joblib.load(data_dir + rat_name + '_Y_BLAex_coactivity_all_nrem_swr.joblib')
    #(c,s) = joblib.load(data_dir + rat_name + '_vCA1ex_PL5ex_shock_mod_coactivity_conditioning.joblib')
    #X=np.append(X,c)
    Y=np.append(Y,RR)

MM=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    data =joblib.load(data_dir + rat_name + '_Y_BLAex_data_modulation_index_2.joblib')
    data2 =joblib.load(data_dir + rat_name + '_Y_BLAex_data_modulation_index_2.joblib')
    M=[]
    for n in range(data.shape[0]):
        for m in range (data2.shape[0]):
            M=np.append(M,data[n]*data2[m])
    com = M.reshape(-1, data.shape[0])
    MM = np.append(MM,com[np.triu_indices(com.shape[0], k=1)])

X=MM

from scipy.stats import pearsonr
ax3.scatter(X,Y, color='blue',s=10)
ax3.set_xlabel('modulation index*modulation index', fontsize=15)
ax3.set_ylabel('coactivity Zscore', fontsize=15)
ax3.text(0.95, 0.95, "r=0.20***", va='top', ha='right', transform=ax3.transAxes, fontsize=20)

x=X
y=Y
idx=np.isfinite(x)&np.isfinite(y)
a, b = np.polyfit(x[idx], y[idx], 1)
Y1 = a * x + b
np.corrcoef(x[idx], y[idx])[0][1]
#r, p = pearsonr(x[idx], y[idx])
r, p = spearmanr(x[idx], y[idx])
print('y=', a, '*x', '+', b)
print('r=', r)
print('p=', p)
ax3.plot(x,Y1,color='black')

#data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
#joblib.dump((X,Y), os.path.expanduser(data_dir +'Y_fig_3_e_bottom_BLAex_all_2.joblib'), compress=3)
#joblib.dump((x,Y1), os.path.expanduser(data_dir +'Y_fig_3_e_bottom_BLAex_2.joblib'), compress=3)

MM=[]
C1=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    data =joblib.load(data_dir + rat_name + '_Y_BLAex_data_modulation_index_2.joblib')
    data2 =joblib.load(data_dir + rat_name + '_Y_BLAex_data_modulation_index_2.joblib')
    M=[]
    for n in range(data.shape[0]):
        for m in range (data2.shape[0]):
            M=np.append(M,data[n]*data2[m])
    com = M.reshape(-1, data.shape[0])
    MM = com[np.triu_indices(com.shape[0], k=1)]
    #(Corr) = joblib.load(data_dir + rat_name + '_vCA1ex_coactivity_all_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_BLAex_NN_RR_NR_RN.joblib')
    vCA1_NN = np.where(NN == 1)[0]
    C1=np.append(C1,MM[vCA1_NN])

YC1=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr, _) = joblib.load(data_dir + rat_name + '_vCA1ex_BLAex_N_N_SWR_corr.joblib')
    (Y) = joblib.load(data_dir + rat_name + '_Y_BLAex_coactivity_all_nrem_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_BLAex_NN_RR_NR_RN.joblib')
    vCA1_NN = np.where(NN == 1)[0]
    YC1=np.append(YC1,Y[vCA1_NN])

MM=[]
C2=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    data =joblib.load(data_dir + rat_name + '_Y_BLAex_data_modulation_index_2.joblib')
    data2 =joblib.load(data_dir + rat_name + '_Y_BLAex_data_modulation_index_2.joblib')
    M=[]
    for n in range(data.shape[0]):
        for m in range (data2.shape[0]):
            M=np.append(M,data[n]*data2[m])
    com = M.reshape(-1, data.shape[0])
    MM = com[np.triu_indices(com.shape[0], k=1)]
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr) = joblib.load(data_dir + rat_name + '_vCA1ex_coactivity_all_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_BLAex_NN_RR_NR_RN.joblib')
    vCA1_RR = np.where(RR == 4)[0]
    C2=np.append(C2,MM[vCA1_RR])

YC2=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr, _) = joblib.load(data_dir + rat_name + '_vCA1ex_BLAex_N_N_SWR_corr.joblib')
    (Y) = joblib.load(data_dir + rat_name + '_Y_BLAex_coactivity_all_nrem_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_BLAex_NN_RR_NR_RN.joblib')
    vCA1_RR = np.where(RR == 4)[0]
    YC2=np.append(YC2,Y[vCA1_RR])


MM=[]
nr=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    data =joblib.load(data_dir + rat_name + '_Y_BLAex_data_modulation_index_2.joblib')
    data2 =joblib.load(data_dir + rat_name + '_Y_BLAex_data_modulation_index_2.joblib')
    M=[]
    for n in range(data.shape[0]):
        for m in range (data2.shape[0]):
            M=np.append(M,data[n]*data2[m])
    com = M.reshape(-1, data.shape[0])
    MM = com[np.triu_indices(com.shape[0], k=1)]
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr) = joblib.load(data_dir + rat_name + '_vCA1ex_coactivity_all_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_BLAex_NN_RR_NR_RN.joblib')
    vCA1_NR = np.where(NR == 2)[0]
    nr=np.append(nr,MM[vCA1_NR])

Ynr=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr, _) = joblib.load(data_dir + rat_name + '_vCA1ex_BLAex_N_N_SWR_corr.joblib')
    (Y) = joblib.load(data_dir + rat_name + '_Y_BLAex_coactivity_all_nrem_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_BLAex_NN_RR_NR_RN.joblib')
    vCA1_NR = np.where(NR == 2)[0]
    Ynr=np.append(Ynr,Y[vCA1_NR])


rn=[]
for rat_name in rats:
    MM = []
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    data = joblib.load(data_dir + rat_name + '_Y_BLAex_data_modulation_index_2.joblib')
    data2 = joblib.load(data_dir + rat_name + '_Y_BLAex_data_modulation_index_2.joblib')
    M = []
    for n in range(data.shape[0]):
        for m in range(data2.shape[0]):
            M = np.append(M, data[n] * data2[m])
    com = M.reshape(-1, data.shape[0])
    MM = com[np.triu_indices(com.shape[0], k=1)]
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr) = joblib.load(data_dir + rat_name + '_vCA1ex_coactivity_all_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_BLAex_NN_RR_NR_RN.joblib')
    vCA1_RN = np.where(RN == 2)[0]
    rn=np.append(rn,MM[vCA1_RN])

Yrn=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr, _) = joblib.load(data_dir + rat_name + '_vCA1ex_BLAex_N_N_SWR_corr.joblib')
    (Y) = joblib.load(data_dir + rat_name + '_Y_BLAex_coactivity_all_nrem_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_BLAex_NN_RR_NR_RN.joblib')
    vCA1_RN = np.where(RN == 2)[0]
    Yrn=np.append(Yrn,Y[vCA1_RN])

C3=np.append(nr,rn)
YC3=np.append(Ynr,Yrn)

MM=[]
C4=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    data =joblib.load(data_dir + rat_name + '_Y_BLAex_data_modulation_index_2.joblib')
    data2 =joblib.load(data_dir + rat_name + '_Y_BLAex_data_modulation_index_2.joblib')
    M=[]
    for n in range(data.shape[0]):
        for m in range (data2.shape[0]):
            M=np.append(M,data[n]*data2[m])
    com = M.reshape(-1, data.shape[0])
    MM = com[np.triu_indices(com.shape[0], k=1)]
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr) = joblib.load(data_dir + rat_name + '_vCA1ex_coactivity_all_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_BLAex_NN_RR_NR_RN.joblib')
    vCA1_nn = (np.where((NN == 0) & (RR == 0) & (NR == 0) & (RN == 0)))[0]
    C4=np.append(C4,MM[vCA1_nn])

YC4=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr, _) = joblib.load(data_dir + rat_name + '_vCA1ex_BLAex_N_N_SWR_corr.joblib')
    (Y) = joblib.load(data_dir + rat_name + '_Y_BLAex_coactivity_all_nrem_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_BLAex_NN_RR_NR_RN.joblib')
    vCA1_nn = (np.where((NN == 0) & (RR == 0) & (NR == 0) & (RN == 0)))[0]
    YC4=np.append(YC4,Y[vCA1_nn])


data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((C1,YC1), os.path.expanduser(data_dir +'Y_fig_4_a_bottom_BLAex_NN.joblib'), compress=3)
joblib.dump((C2,YC2), os.path.expanduser(data_dir +'Y_fig_4_a_bottom_BLAex_RR.joblib'), compress=3)
joblib.dump((C3,YC3), os.path.expanduser(data_dir +'Y_fig_4_a_bottom_BLAex_NR.joblib'), compress=3)

joblib.dump((C4,YC4), os.path.expanduser(data_dir +'Y_fig_4_a_bottom_BLAex_nosignificant.joblib'), compress=3)


# vCA1 PL5
rats = ['hoegaarden181115','innis190601','karmeliet190901','leffe200124','maredsous200224','nostrum200304','oberon200325']

Y=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr, _) = joblib.load(data_dir + rat_name + '_vCA1ex_BLAex_N_N_SWR_corr.joblib')
    (RR) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_PL5ex_coactivity_all_nrem_swr.joblib')
    #(c,s) = joblib.load(data_dir + rat_name + '_vCA1ex_PL5ex_shock_mod_coactivity_conditioning.joblib')
    #X=np.append(X,c)
    Y=np.append(Y,RR)

M=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    data =joblib.load(data_dir + rat_name + '_Y_vCA1ex_data_modulation_index_2.joblib')
    data2 =joblib.load(data_dir + rat_name + '_Y_PL5ex_data_modulation_index_2.joblib')
    for n in range(data.shape[0]):
        for m in range (data2.shape[0]):
            M=np.append(M,data[n]*data2[m])

X=M

from scipy.stats import pearsonr
ax4.scatter(X,Y, color='blue',s=10)
ax4.set_xlabel('modulation index*modulation index', fontsize=15)
ax4.set_ylabel('coactivity Zscore', fontsize=15)
ax4.text(0.95, 0.95, "r=0.06***", va='top', ha='right', transform=ax4.transAxes, fontsize=20)

x=X
y=Y
idx=np.isfinite(x)&np.isfinite(y)
a, b = np.polyfit(x[idx], y[idx], 1)
Y1 = a * x + b
np.corrcoef(x[idx], y[idx])[0][1]
r, p = pearsonr(x[idx], y[idx])
r, p = spearmanr(x[idx], y[idx])
print('y=', a, '*x', '+', b)
print('r=', r)
print('p=', p)
ax4.plot(x,Y1,color='black')

#data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
#joblib.dump((X,Y), os.path.expanduser(data_dir +'Y_fig_4_a_bottom_vCA1ex_PL5ex_all_2.joblib'), compress=3)
#joblib.dump((x,Y1), os.path.expanduser(data_dir +'Y_fig_4_a_bottom_vCA1ex_PL5ex_2.joblib'), compress=3)

MM=[]
C1=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    data =joblib.load(data_dir + rat_name + '_Y_vCA1ex_data_modulation_index_2.joblib')
    data2 =joblib.load(data_dir + rat_name + '_Y_PL5ex_data_modulation_index_2.joblib')
    M=[]
    for n in range(data.shape[0]):
        for m in range (data2.shape[0]):
            M=np.append(M,data[n]*data2[m])
    #com = M.reshape(-1, data.shape[0])
    #MM = com[np.triu_indices(com.shape[0], k=1)]
    #(Corr) = joblib.load(data_dir + rat_name + '_vCA1ex_coactivity_all_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_vCA1ex_PL5ex_NN_RR_NR_RN.joblib')
    vCA1_NN = np.where(NN == 1)[0]
    C1=np.append(C1,M[vCA1_NN])

YC1=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr, _) = joblib.load(data_dir + rat_name + '_vCA1ex_BLAex_N_N_SWR_corr.joblib')
    (Y) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_PL5ex_coactivity_all_nrem_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_vCA1ex_PL5ex_NN_RR_NR_RN.joblib')
    vCA1_NN = np.where(NN == 1)[0]
    YC1=np.append(YC1,Y[vCA1_NN])

MM=[]
C2=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    data =joblib.load(data_dir + rat_name + '_Y_vCA1ex_data_modulation_index_2.joblib')
    data2 =joblib.load(data_dir + rat_name + '_Y_PL5ex_data_modulation_index_2.joblib')
    M=[]
    for n in range(data.shape[0]):
        for m in range (data2.shape[0]):
            M=np.append(M,data[n]*data2[m])
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr) = joblib.load(data_dir + rat_name + '_vCA1ex_coactivity_all_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_vCA1ex_PL5ex_NN_RR_NR_RN.joblib')
    vCA1_RR = np.where(RR == 4)[0]
    C2=np.append(C2,M[vCA1_RR])

YC2=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr, _) = joblib.load(data_dir + rat_name + '_vCA1ex_BLAex_N_N_SWR_corr.joblib')
    (Y) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_PL5ex_coactivity_all_nrem_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_vCA1ex_PL5ex_NN_RR_NR_RN.joblib')
    vCA1_RR = np.where(RR == 4)[0]
    YC2=np.append(YC2,Y[vCA1_RR])


MM=[]
C3=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    data =joblib.load(data_dir + rat_name + '_Y_vCA1ex_data_modulation_index_2.joblib')
    data2 =joblib.load(data_dir + rat_name + '_Y_PL5ex_data_modulation_index_2.joblib')
    M=[]
    for n in range(data.shape[0]):
        for m in range (data2.shape[0]):
            M=np.append(M,data[n]*data2[m])
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr) = joblib.load(data_dir + rat_name + '_vCA1ex_coactivity_all_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_vCA1ex_PL5ex_NN_RR_NR_RN.joblib')
    vCA1_NR = np.where(NR == 2)[0]
    C3=np.append(C3,M[vCA1_NR])

YC3=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr, _) = joblib.load(data_dir + rat_name + '_vCA1ex_BLAex_N_N_SWR_corr.joblib')
    (Y) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_PL5ex_coactivity_all_nrem_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_vCA1ex_PL5ex_NN_RR_NR_RN.joblib')
    vCA1_NR = np.where(NR == 2)[0]
    YC3=np.append(YC3,Y[vCA1_NR])


C4=[]
for rat_name in rats:
    MM = []
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    data = joblib.load(data_dir + rat_name + '_Y_vCA1ex_data_modulation_index_2.joblib')
    data2 = joblib.load(data_dir + rat_name + '_Y_PL5ex_data_modulation_index_2.joblib')
    M = []
    for n in range(data.shape[0]):
        for m in range(data2.shape[0]):
            M = np.append(M, data[n] * data2[m])
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr) = joblib.load(data_dir + rat_name + '_vCA1ex_coactivity_all_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_vCA1ex_PL5ex_NN_RR_NR_RN.joblib')
    vCA1_RN = np.where(RN == 2)[0]
    C4=np.append(C4,M[vCA1_RN])

YC4=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr, _) = joblib.load(data_dir + rat_name + '_vCA1ex_BLAex_N_N_SWR_corr.joblib')
    (Y) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_PL5ex_coactivity_all_nrem_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_vCA1ex_PL5ex_NN_RR_NR_RN.joblib')
    vCA1_RN = np.where(RN == 2)[0]
    YC4=np.append(YC4,Y[vCA1_RN])


MM=[]
C5=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    data =joblib.load(data_dir + rat_name + '_Y_vCA1ex_data_modulation_index_2.joblib')
    data2 =joblib.load(data_dir + rat_name + '_Y_PL5ex_data_modulation_index_2.joblib')
    M=[]
    for n in range(data.shape[0]):
        for m in range (data2.shape[0]):
            M=np.append(M,data[n]*data2[m])
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr) = joblib.load(data_dir + rat_name + '_vCA1ex_coactivity_all_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_vCA1ex_PL5ex_NN_RR_NR_RN.joblib')
    vCA1_nn = (np.where((NN == 0) & (RR == 0) & (NR == 0) & (RN == 0)))[0]
    C5=np.append(C5,M[vCA1_nn])

YC5=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr, _) = joblib.load(data_dir + rat_name + '_vCA1ex_BLAex_N_N_SWR_corr.joblib')
    (Y) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_PL5ex_coactivity_all_nrem_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_vCA1ex_PL5ex_NN_RR_NR_RN.joblib')
    vCA1_nn = (np.where((NN == 0) & (RR == 0) & (NR == 0) & (RN == 0)))[0]
    YC5=np.append(YC5,Y[vCA1_nn])


data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((C1,YC1), os.path.expanduser(data_dir +'Y_fig_5_a_bottom_vCA1ex_PL5ex_NN.joblib'), compress=3)
joblib.dump((C2,YC2), os.path.expanduser(data_dir +'Y_fig_5_a_bottom_vCA1ex_PL5ex_RR.joblib'), compress=3)
joblib.dump((C3,YC3), os.path.expanduser(data_dir +'Y_fig_5_a_bottom_vCA1ex_PL5ex_NR.joblib'), compress=3)
joblib.dump((C4,YC4), os.path.expanduser(data_dir +'Y_fig_5_a_bottom_vCA1ex_PL5ex_RN.joblib'), compress=3)

joblib.dump((C5,YC5), os.path.expanduser(data_dir +'Y_fig_5_a_bottom_vCA1ex_PL5ex_nosignificant.joblib'), compress=3)

# vCA1 BLA
rats = ['duvel190505','hoegaarden181115', 'innis190601', 'leffe200124', 'maredsous200224', 'nostrum200304', 'oberon200325']

Y=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr, _) = joblib.load(data_dir + rat_name + '_vCA1ex_BLAex_N_N_SWR_corr.joblib')
    (RR) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_BLAex_coactivity_all_nrem_swr.joblib')
    #(c,s) = joblib.load(data_dir + rat_name + '_vCA1ex_PL5ex_shock_mod_coactivity_conditioning.joblib')
    #X=np.append(X,c)
    Y=np.append(Y,RR)

M=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    data =joblib.load(data_dir + rat_name + '_Y_vCA1ex_data_modulation_index_2.joblib')
    data2 =joblib.load(data_dir + rat_name + '_Y_BLAex_data_modulation_index_2.joblib')
    for n in range(data.shape[0]):
        for m in range (data2.shape[0]):
            M=np.append(M,data[n]*data2[m])

X=M

from scipy.stats import pearsonr
ax5.scatter(X,Y, color='blue',s=10)
ax5.set_xlabel('modulation index*modulation index', fontsize=15)
ax5.set_ylabel('coactivity Zscore', fontsize=15)
ax5.text(0.95, 0.95, "r=0.13***", va='top', ha='right', transform=ax5.transAxes, fontsize=20)

x=X
y=Y
idx=np.isfinite(x)&np.isfinite(y)
a, b = np.polyfit(x[idx], y[idx], 1)
Y1 = a * x + b
np.corrcoef(x[idx], y[idx])[0][1]
#r, p = pearsonr(x[idx], y[idx])
r, p = spearmanr(x[idx], y[idx])
print('y=', a, '*x', '+', b)
print('r=', r)
print('p=', p)
ax5.plot(x,Y1,color='black')

#data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
#joblib.dump((X,Y), os.path.expanduser(data_dir +'Y_fig_4_a_bottom_vCA1ex_BLAex_all_2.joblib'), compress=3)
#joblib.dump((x,Y1), os.path.expanduser(data_dir +'Y_fig_4_a_bottom_vCA1ex_BLAex_2.joblib'), compress=3)

MM=[]
C1=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    data =joblib.load(data_dir + rat_name + '_Y_vCA1ex_data_modulation_index_2.joblib')
    data2 =joblib.load(data_dir + rat_name + '_Y_BLAex_data_modulation_index_2.joblib')
    M=[]
    for n in range(data.shape[0]):
        for m in range (data2.shape[0]):
            M=np.append(M,data[n]*data2[m])
    #(Corr) = joblib.load(data_dir + rat_name + '_vCA1ex_coactivity_all_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_vCA1ex_BLAex_NN_RR_NR_RN.joblib')
    vCA1_NN = np.where(NN == 1)[0]
    C1=np.append(C1,M[vCA1_NN])

YC1=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr, _) = joblib.load(data_dir + rat_name + '_vCA1ex_BLAex_N_N_SWR_corr.joblib')
    (Y) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_BLAex_coactivity_all_nrem_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_vCA1ex_BLAex_NN_RR_NR_RN.joblib')
    vCA1_NN = np.where(NN == 1)[0]
    YC1=np.append(YC1,Y[vCA1_NN])

MM=[]
C2=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    data =joblib.load(data_dir + rat_name + '_Y_vCA1ex_data_modulation_index_2.joblib')
    data2 =joblib.load(data_dir + rat_name + '_Y_BLAex_data_modulation_index_2.joblib')
    M=[]
    for n in range(data.shape[0]):
        for m in range (data2.shape[0]):
            M=np.append(M,data[n]*data2[m])
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr) = joblib.load(data_dir + rat_name + '_vCA1ex_coactivity_all_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_vCA1ex_BLAex_NN_RR_NR_RN.joblib')
    vCA1_RR = np.where(RR == 4)[0]
    C2=np.append(C2,M[vCA1_RR])

YC2=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr, _) = joblib.load(data_dir + rat_name + '_vCA1ex_BLAex_N_N_SWR_corr.joblib')
    (Y) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_BLAex_coactivity_all_nrem_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_vCA1ex_BLAex_NN_RR_NR_RN.joblib')
    vCA1_RR = np.where(RR == 4)[0]
    YC2=np.append(YC2,Y[vCA1_RR])


MM=[]
C3=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    data =joblib.load(data_dir + rat_name + '_Y_vCA1ex_data_modulation_index_2.joblib')
    data2 =joblib.load(data_dir + rat_name + '_Y_BLAex_data_modulation_index_2.joblib')
    M=[]
    for n in range(data.shape[0]):
        for m in range (data2.shape[0]):
            M=np.append(M,data[n]*data2[m])
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr) = joblib.load(data_dir + rat_name + '_vCA1ex_coactivity_all_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_vCA1ex_BLAex_NN_RR_NR_RN.joblib')
    vCA1_NR = np.where(NR == 2)[0]
    C3=np.append(C3,M[vCA1_NR])

YC3=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr, _) = joblib.load(data_dir + rat_name + '_vCA1ex_BLAex_N_N_SWR_corr.joblib')
    (Y) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_BLAex_coactivity_all_nrem_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_vCA1ex_BLAex_NN_RR_NR_RN.joblib')
    vCA1_NR = np.where(NR == 2)[0]
    YC3=np.append(YC3,Y[vCA1_NR])


C4=[]
for rat_name in rats:
    MM = []
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    data = joblib.load(data_dir + rat_name + '_Y_vCA1ex_data_modulation_index_2.joblib')
    data2 = joblib.load(data_dir + rat_name + '_Y_BLAex_data_modulation_index_2.joblib')
    M = []
    for n in range(data.shape[0]):
        for m in range(data2.shape[0]):
            M = np.append(M, data[n] * data2[m])
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr) = joblib.load(data_dir + rat_name + '_vCA1ex_coactivity_all_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_vCA1ex_BLAex_NN_RR_NR_RN.joblib')
    vCA1_RN = np.where(RN == 2)[0]
    C4=np.append(C4,M[vCA1_RN])

YC4=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr, _) = joblib.load(data_dir + rat_name + '_vCA1ex_BLAex_N_N_SWR_corr.joblib')
    (Y) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_BLAex_coactivity_all_nrem_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_vCA1ex_BLAex_NN_RR_NR_RN.joblib')
    vCA1_RN = np.where(RN == 2)[0]
    YC4=np.append(YC4,Y[vCA1_RN])


MM=[]
C5=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    data =joblib.load(data_dir + rat_name + '_Y_vCA1ex_data_modulation_index_2.joblib')
    data2 =joblib.load(data_dir + rat_name + '_Y_BLAex_data_modulation_index_2.joblib')
    M=[]
    for n in range(data.shape[0]):
        for m in range (data2.shape[0]):
            M=np.append(M,data[n]*data2[m])
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr) = joblib.load(data_dir + rat_name + '_vCA1ex_coactivity_all_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_vCA1ex_BLAex_NN_RR_NR_RN.joblib')
    vCA1_nn = (np.where((NN == 0) & (RR == 0) & (NR == 0) & (RN == 0)))[0]
    C5=np.append(C5,M[vCA1_nn])

YC5=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr, _) = joblib.load(data_dir + rat_name + '_vCA1ex_BLAex_N_N_SWR_corr.joblib')
    (Y) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_BLAex_coactivity_all_nrem_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_vCA1ex_BLAex_NN_RR_NR_RN.joblib')
    vCA1_nn = (np.where((NN == 0) & (RR == 0) & (NR == 0) & (RN == 0)))[0]
    YC5=np.append(YC5,Y[vCA1_nn])

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((C1,YC1), os.path.expanduser(data_dir +'Y_fig_5_a_bottom_vCA1ex_BLAex_NN.joblib'), compress=3)
joblib.dump((C2,YC2), os.path.expanduser(data_dir +'Y_fig_5_a_bottom_vCA1ex_BLAex_RR.joblib'), compress=3)
joblib.dump((C3,YC3), os.path.expanduser(data_dir +'Y_fig_5_a_bottom_vCA1ex_BLAex_NR.joblib'), compress=3)
joblib.dump((C4,YC4), os.path.expanduser(data_dir +'Y_fig_5_a_bottom_vCA1ex_BLAex_RN.joblib'), compress=3)

joblib.dump((C5,YC5), os.path.expanduser(data_dir +'Y_fig_5_a_bottom_vCA1ex_BLAex_nosignificant.joblib'), compress=3)


# PL5_BLA
rats = ['hoegaarden181115', 'innis190601', 'leffe200124', 'maredsous200224', 'nostrum200304', 'oberon200325']

Y=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr, _) = joblib.load(data_dir + rat_name + '_vCA1ex_BLAex_N_N_SWR_corr.joblib')
    (RR) = joblib.load(data_dir + rat_name + '_Y_PL5ex_BLAex_coactivity_all_nrem_swr.joblib')
    #(c,s) = joblib.load(data_dir + rat_name + '_vCA1ex_PL5ex_shock_mod_coactivity_conditioning.joblib')
    #X=np.append(X,c)
    Y=np.append(Y,RR)

M=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    data =joblib.load(data_dir + rat_name + '_Y_PL5ex_data_modulation_index_2.joblib')
    data2 =joblib.load(data_dir + rat_name + '_Y_BLAex_data_modulation_index_2.joblib')
    for n in range(data.shape[0]):
        for m in range (data2.shape[0]):
            M=np.append(M,data[n]*data2[m])

X=M

from scipy.stats import pearsonr
ax6.scatter(X,Y, color='blue',s=10)
ax6.set_xlabel('modulation index*modulation index', fontsize=15)
ax6.set_ylabel('coactivity Zscore', fontsize=15)
ax6.text(0.95, 0.95, "r=0.18***", va='top', ha='right', transform=ax6.transAxes, fontsize=20)

x=X
y=Y
idx=np.isfinite(x)&np.isfinite(y)
a, b = np.polyfit(x[idx], y[idx], 1)
Y1 = a * x + b
np.corrcoef(x[idx], y[idx])[0][1]
#r, p = pearsonr(x[idx], y[idx])
r, p = spearmanr(x[idx], y[idx])
print('y=', a, '*x', '+', b)
print('r=', r)
print('p=', p)
ax6.plot(x,Y1,color='black')


#data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
#joblib.dump((X,Y), os.path.expanduser(data_dir +'Y_fig_4_a_bottom_PL5ex_BLAex_all_2.joblib'), compress=3)
#joblib.dump((x,Y1), os.path.expanduser(data_dir +'Y_fig_4_a_bottom_PL5ex_BLAex_2.joblib'), compress=3)


MM=[]
C1=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    data =joblib.load(data_dir + rat_name + '_Y_PL5ex_data_modulation_index_2.joblib')
    data2 =joblib.load(data_dir + rat_name + '_Y_BLAex_data_modulation_index_2.joblib')
    M=[]
    for n in range(data.shape[0]):
        for m in range (data2.shape[0]):
            M=np.append(M,data[n]*data2[m])
    #(Corr) = joblib.load(data_dir + rat_name + '_vCA1ex_coactivity_all_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_PL5ex_BLAex_NN_RR_NR_RN.joblib')
    vCA1_NN = np.where(NN == 1)[0]
    C1=np.append(C1,M[vCA1_NN])

YC1=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr, _) = joblib.load(data_dir + rat_name + '_vCA1ex_BLAex_N_N_SWR_corr.joblib')
    (Y) = joblib.load(data_dir + rat_name + '_Y_PL5ex_BLAex_coactivity_all_nrem_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_PL5ex_BLAex_NN_RR_NR_RN.joblib')
    vCA1_NN = np.where(NN == 1)[0]
    YC1=np.append(YC1,Y[vCA1_NN])

MM=[]
C2=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    data =joblib.load(data_dir + rat_name + '_Y_PL5ex_data_modulation_index_2.joblib')
    data2 =joblib.load(data_dir + rat_name + '_Y_BLAex_data_modulation_index_2.joblib')
    M=[]
    for n in range(data.shape[0]):
        for m in range (data2.shape[0]):
            M=np.append(M,data[n]*data2[m])
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr) = joblib.load(data_dir + rat_name + '_vCA1ex_coactivity_all_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_PL5ex_BLAex_NN_RR_NR_RN.joblib')
    vCA1_RR = np.where(RR == 4)[0]
    C2=np.append(C2,M[vCA1_RR])

YC2=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr, _) = joblib.load(data_dir + rat_name + '_vCA1ex_BLAex_N_N_SWR_corr.joblib')
    (Y) = joblib.load(data_dir + rat_name + '_Y_PL5ex_BLAex_coactivity_all_nrem_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_PL5ex_BLAex_NN_RR_NR_RN.joblib')
    vCA1_RR = np.where(RR == 4)[0]
    YC2=np.append(YC2,Y[vCA1_RR])


MM=[]
C3=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    data =joblib.load(data_dir + rat_name + '_Y_PL5ex_data_modulation_index_2.joblib')
    data2 =joblib.load(data_dir + rat_name + '_Y_BLAex_data_modulation_index_2.joblib')
    M=[]
    for n in range(data.shape[0]):
        for m in range (data2.shape[0]):
            M=np.append(M,data[n]*data2[m])
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr) = joblib.load(data_dir + rat_name + '_vCA1ex_coactivity_all_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_PL5ex_BLAex_NN_RR_NR_RN.joblib')
    vCA1_NR = np.where(NR == 2)[0]
    C3=np.append(C3,M[vCA1_NR])

YC3=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr, _) = joblib.load(data_dir + rat_name + '_vCA1ex_BLAex_N_N_SWR_corr.joblib')
    (Y) = joblib.load(data_dir + rat_name + '_Y_PL5ex_BLAex_coactivity_all_nrem_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_PL5ex_BLAex_NN_RR_NR_RN.joblib')
    vCA1_NR = np.where(NR == 2)[0]
    YC3=np.append(YC3,Y[vCA1_NR])


C4=[]
for rat_name in rats:
    MM = []
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    data = joblib.load(data_dir + rat_name + '_Y_PL5ex_data_modulation_index_2.joblib')
    data2 = joblib.load(data_dir + rat_name + '_Y_BLAex_data_modulation_index_2.joblib')
    M = []
    for n in range(data.shape[0]):
        for m in range(data2.shape[0]):
            M = np.append(M, data[n] * data2[m])
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr) = joblib.load(data_dir + rat_name + '_vCA1ex_coactivity_all_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_PL5ex_BLAex_NN_RR_NR_RN.joblib')
    vCA1_RN = np.where(RN == 2)[0]
    C4=np.append(C4,M[vCA1_RN])

YC4=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr, _) = joblib.load(data_dir + rat_name + '_vCA1ex_BLAex_N_N_SWR_corr.joblib')
    (Y) = joblib.load(data_dir + rat_name + '_Y_PL5ex_BLAex_coactivity_all_nrem_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_PL5ex_BLAex_NN_RR_NR_RN.joblib')
    vCA1_RN = np.where(RN == 2)[0]
    YC4=np.append(YC4,Y[vCA1_RN])

MM=[]
C5=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    data =joblib.load(data_dir + rat_name + '_Y_PL5ex_data_modulation_index_2.joblib')
    data2 =joblib.load(data_dir + rat_name + '_Y_BLAex_data_modulation_index_2.joblib')
    M=[]
    for n in range(data.shape[0]):
        for m in range (data2.shape[0]):
            M=np.append(M,data[n]*data2[m])
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr) = joblib.load(data_dir + rat_name + '_vCA1ex_coactivity_all_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_PL5ex_BLAex_NN_RR_NR_RN.joblib')
    vCA1_nn = (np.where((NN == 0) & (RR == 0) & (NR == 0) & (RN == 0)))[0]
    C5=np.append(C5,M[vCA1_nn])

YC5=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr, _) = joblib.load(data_dir + rat_name + '_vCA1ex_BLAex_N_N_SWR_corr.joblib')
    (Y) = joblib.load(data_dir + rat_name + '_Y_PL5ex_BLAex_coactivity_all_nrem_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_PL5ex_BLAex_NN_RR_NR_RN.joblib')
    vCA1_nn = (np.where((NN == 0) & (RR == 0) & (NR == 0) & (RN == 0)))[0]
    YC5=np.append(YC5,Y[vCA1_nn])


data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((C1,YC1), os.path.expanduser(data_dir +'Y_fig_5_a_bottom_PL5ex_BLAex_NN.joblib'), compress=3)
joblib.dump((C2,YC2), os.path.expanduser(data_dir +'Y_fig_5_a_bottom_PL5ex_BLAex_RR.joblib'), compress=3)
joblib.dump((C3,YC3), os.path.expanduser(data_dir +'Y_fig_5_a_bottom_PL5ex_BLAex_NR.joblib'), compress=3)
joblib.dump((C4,YC4), os.path.expanduser(data_dir +'Y_fig_5_a_bottom_PL5ex_BLAex_RN.joblib'), compress=3)

joblib.dump((C5,YC5), os.path.expanduser(data_dir +'Y_fig_5_a_bottom_PL5ex_BLAex_nosignificant.joblib'), compress=3)

ax1.set_title('vCA1ex', fontsize=25)
ax2.set_title('PL5ex', fontsize=25)
ax3.set_title('BLAex', fontsize=25)
ax4.set_title('vCA1ex_PL5ex', fontsize=25)
ax5.set_title('vCA1ex_BLAex', fontsize=25)
ax6.set_title('PL5ex_BLAex', fontsize=25)

plt.subplots_adjust(hspace=0.4,wspace=0.35)