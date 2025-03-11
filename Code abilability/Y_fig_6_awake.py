import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

#fig = plt.figure(figsize=(24, 8))
fig = plt.figure(figsize=(18, 12))
ax1 = fig.add_subplot(2, 3, 1)
ax2 = fig.add_subplot(2, 3, 2)
ax3 = fig.add_subplot(2, 3, 3)
ax4 = fig.add_subplot(2, 3, 4)
ax5 = fig.add_subplot(2, 3, 5)
ax6 = fig.add_subplot(2, 3, 6)



# vCA1 PL5
rats = ['hoegaarden181115','innis190601','karmeliet190901','leffe200124','maredsous200224','nostrum200304','oberon200325']

Y=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr, _) = joblib.load(data_dir + rat_name + '_vCA1ex_BLAex_N_N_SWR_corr.joblib')
    (RR) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_PL5ex_coactivity_all_wake_swr.joblib')
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
print('y=', a, '*x', '+', b)
print('r=', r)
print('p=', p)
ax4.plot(x,Y1,color='black')

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((X,Y), os.path.expanduser(data_dir +'Y_fig_6_wake_bottom_vCA1ex_PL5ex_all_2.joblib'), compress=3)
joblib.dump((x,Y1), os.path.expanduser(data_dir +'Y_fig_6_wake_bottom__vCA1ex_PL5ex_2.joblib'), compress=3)

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
    (Y) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_PL5ex_coactivity_all_wake_swr.joblib')
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
    (Y) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_PL5ex_coactivity_all_wake_swr.joblib')
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
    (Y) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_PL5ex_coactivity_all_wake_swr.joblib')
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
    (Y) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_PL5ex_coactivity_all_wake_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_vCA1ex_PL5ex_NN_RR_NR_RN.joblib')
    vCA1_RN = np.where(RN == 2)[0]
    YC4=np.append(YC4,Y[vCA1_RN])


data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((C1,YC1), os.path.expanduser(data_dir +'Y_fig_6_wake_bottom_vCA1ex_PL5ex_NN.joblib'), compress=3)
joblib.dump((C2,YC2), os.path.expanduser(data_dir +'Y_fig_6_wake_bottom_vCA1ex_PL5ex_RR.joblib'), compress=3)
joblib.dump((C3,YC3), os.path.expanduser(data_dir +'Y_fig_6_wake_bottom_vCA1ex_PL5ex_NR.joblib'), compress=3)
joblib.dump((C4,YC4), os.path.expanduser(data_dir +'Y_fig_6_wake_bottom_vCA1ex_PL5ex_RN.joblib'), compress=3)



# vCA1 BLA
rats = ['duvel190505','hoegaarden181115', 'innis190601', 'leffe200124', 'maredsous200224', 'nostrum200304', 'oberon200325']

Y=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr, _) = joblib.load(data_dir + rat_name + '_vCA1ex_BLAex_N_N_SWR_corr.joblib')
    (RR) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_BLAex_coactivity_all_wake_swr.joblib')
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
r, p = pearsonr(x[idx], y[idx])
print('y=', a, '*x', '+', b)
print('r=', r)
print('p=', p)
ax5.plot(x,Y1,color='black')

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((X,Y), os.path.expanduser(data_dir +'Y_fig_6_wake_bottom_vCA1ex_BLAex_all_2.joblib'), compress=3)
joblib.dump((x,Y1), os.path.expanduser(data_dir +'Y_fig_6_wake_bottom_vCA1ex_BLAex_2.joblib'), compress=3)

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
    (Y) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_BLAex_coactivity_all_wake_swr.joblib')
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
    (Y) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_BLAex_coactivity_all_wake_swr.joblib')
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
    (Y) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_BLAex_coactivity_all_wake_swr.joblib')
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
    (Y) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_BLAex_coactivity_all_wake_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_vCA1ex_BLAex_NN_RR_NR_RN.joblib')
    vCA1_RN = np.where(RN == 2)[0]
    YC4=np.append(YC4,Y[vCA1_RN])


data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((C1,YC1), os.path.expanduser(data_dir +'Y_fig_6_wake_bottom_vCA1ex_BLAex_NN.joblib'), compress=3)
joblib.dump((C2,YC2), os.path.expanduser(data_dir +'Y_fig_6_wake_bottom_vCA1ex_BLAex_RR.joblib'), compress=3)
joblib.dump((C3,YC3), os.path.expanduser(data_dir +'Y_fig_6_wake_bottom_vCA1ex_BLAex_NR.joblib'), compress=3)
joblib.dump((C4,YC4), os.path.expanduser(data_dir +'Y_fig_6_wake_bottom_vCA1ex_BLAex_RN.joblib'), compress=3)



# PL5_BLA
rats = ['hoegaarden181115', 'innis190601', 'leffe200124', 'maredsous200224', 'nostrum200304', 'oberon200325']

Y=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #(Corr, _) = joblib.load(data_dir + rat_name + '_vCA1ex_BLAex_N_N_SWR_corr.joblib')
    (RR) = joblib.load(data_dir + rat_name + '_Y_PL5ex_BLAex_coactivity_all_wake_swr.joblib')
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
r, p = pearsonr(x[idx], y[idx])
print('y=', a, '*x', '+', b)
print('r=', r)
print('p=', p)
ax6.plot(x,Y1,color='black')


data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((X,Y), os.path.expanduser(data_dir +'Y_fig_6_wake_bottom_PL5ex_BLAex_all_2.joblib'), compress=3)
joblib.dump((x,Y1), os.path.expanduser(data_dir +'Y_fig_6_wake_bottom_PL5ex_BLAex_2.joblib'), compress=3)


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
    (Y) = joblib.load(data_dir + rat_name + '_Y_PL5ex_BLAex_coactivity_all_wake_swr.joblib')
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
    (Y) = joblib.load(data_dir + rat_name + '_Y_PL5ex_BLAex_coactivity_all_wake_swr.joblib')
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
    (Y) = joblib.load(data_dir + rat_name + '_Y_PL5ex_BLAex_coactivity_all_wake_swr.joblib')
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
    (Y) = joblib.load(data_dir + rat_name + '_Y_PL5ex_BLAex_coactivity_all_wake_swr.joblib')
    (NN,RR,NR,RN)= joblib.load(data_dir + rat_name + '_Y_PL5ex_BLAex_NN_RR_NR_RN.joblib')
    vCA1_RN = np.where(RN == 2)[0]
    YC4=np.append(YC4,Y[vCA1_RN])


data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((C1,YC1), os.path.expanduser(data_dir +'Y_fig_6_wake_bottom_PL5ex_BLAex_NN.joblib'), compress=3)
joblib.dump((C2,YC2), os.path.expanduser(data_dir +'Y_fig_6_wake_bottom_PL5ex_BLAex_RR.joblib'), compress=3)
joblib.dump((C3,YC3), os.path.expanduser(data_dir +'Y_fig_6_wake_bottom_PL5ex_BLAex_NR.joblib'), compress=3)
joblib.dump((C4,YC4), os.path.expanduser(data_dir +'Y_fig_6_wake_bottom_PL5ex_BLAex_RN.joblib'), compress=3)

ax1.set_title('vCA1ex', fontsize=25)
ax2.set_title('PL5ex', fontsize=25)
ax3.set_title('BLAex', fontsize=25)
ax4.set_title('vCA1ex_PL5ex', fontsize=25)
ax5.set_title('vCA1ex_BLAex', fontsize=25)
ax6.set_title('PL5ex_BLAex', fontsize=25)

