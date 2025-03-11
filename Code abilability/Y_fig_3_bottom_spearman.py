import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from scipy import stats
from scipy.stats import spearmanr
import seaborn as sns
sns.set_context('poster')

fig = plt.figure(figsize=(24, 6))
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
    s = np.sqrt((n1*np.square(s１)+n2*np.square(s2))/(n1+n2))
    d = np.abs(x1_mean-x2_mean)/s
    return d

#vCA1
rats=['duvel190505','hoegaarden181115','innis190601','karmeliet190901','leffe200124','maredsous200224','nostrum200304','oberon200325']


data=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (d) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_data_modulation_index_2.joblib')
    data=np.append(data,d)


N_in=[]
N_out=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (N_in_cell,N_out_cell,*_) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_swr.joblib')
    N_in = np.append(N_in, N_in_cell)
    N_out = np.append(N_out, N_out_cell)

gain_N=N_in/N_out

x=data
y=gain_N

idx=np.isfinite(x)&np.isfinite(y)
#idx=np.isfinite(x)&np.isfinite(y)
a, b = np.polyfit(x[idx], y[idx], 1)
Y1 = a * x + b
np.corrcoef(x[idx], y[idx])[0][1]
#r, p = pearsonr(x[idx], y[idx])
r, p = spearmanr(x[idx], y[idx])
#r, p = spearmanr(x[idx], y[idx])
print('y=', a, '*x', '+', b)
print('r=', r)
print('p=', p)
ax1.plot(x,Y1,color='black')


data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
(all_time_N_id,all_time_R_id)=joblib.load(data_dir + 'Y_vCA1ex_n_r_id_2_50_2.joblib')
NNid = np.where(all_time_N_id == 1)
RRid = np.where(all_time_R_id == 2)
non=np.where((all_time_N_id == 0) & (all_time_R_id == 0))

from scipy.stats import pearsonr
#ax1.scatter(x,y, color='blue',s=10)
ax1.scatter(x[RRid],y[RRid], color='violet',s=10)
ax1.scatter(x[NNid],y[NNid], color='cornflowerblue',s=10)
ax1.set_xlabel('Modulation Index', fontsize=20)
ax1.set_ylabel('SWR gain', fontsize=20)
ax1.text(0.95, 0.95, "r=0.23*", va='top', ha='right', transform=ax1.transAxes, fontsize=20)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((x[NNid],y[NNid]), os.path.expanduser(data_dir +'Y_fig_3_a_bottom_vCA1ex_nrem_active_2.joblib'), compress=3)
joblib.dump((x[RRid],y[RRid]), os.path.expanduser(data_dir +'Y_fig_3_a_bottom_vCA1ex_rem_active_2.joblib'), compress=3)
#joblib.dump((x,Y1), os.path.expanduser(data_dir +'Y_fig_3_a_bottom_vCA1ex_2_sp.joblib'), compress=3)

joblib.dump((x[non],y[non]), os.path.expanduser(data_dir +'Y_fig_3_a_bottom_vCA1ex_non_active_2.joblib'), compress=3)


#PL5
rats=['hoegaarden181115','innis190601','jever190814','karmeliet190901','leffe200124','maredsous200224','nostrum200304','oberon200325']

data=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (d) = joblib.load(data_dir + rat_name + '_Y_PL5ex_data_modulation_index_2.joblib')
    data=np.append(data,d)

N_in=[]
N_out=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (N_in_cell,N_out_cell,*_) = joblib.load(data_dir + rat_name + '_Y_PL5ex_swr.joblib')
    N_in = np.append(N_in, N_in_cell)
    N_out = np.append(N_out, N_out_cell)

gain_N=N_in/N_out

x=data
y=gain_N

idx=np.isfinite(x)&np.isfinite(y)
#idx=np.isfinite(x)&np.isfinite(y)
a, b = np.polyfit(x[idx], y[idx], 1)
Y1 = a * x + b
np.corrcoef(x[idx], y[idx])[0][1]
r, p = spearmanr(x[idx], y[idx])
print('y=', a, '*x', '+', b)
print('r=', r)
print('p=', p)
ax2.plot(x,Y1,color='black')


data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
(all_time_N_id,all_time_R_id)=joblib.load(data_dir + 'Y_PL5ex_n_r_id_2_50_2.joblib')
NNid = np.where(all_time_N_id == 1)
RRid = np.where(all_time_R_id == 2)
non=np.where((all_time_N_id == 0) & (all_time_R_id == 0))

from scipy.stats import pearsonr
#ax1.scatter(x,y, color='blue',s=10)
ax2.scatter(x[RRid],y[RRid], color='violet',s=10)
ax2.scatter(x[NNid],y[NNid], color='cornflowerblue',s=10)
#ax2.scatter(x,y, color='blue',s=10)
ax2.set_xlabel('Modulation Index', fontsize=20)
ax2.set_ylabel('SWR gain', fontsize=20)
ax2.text(0.95, 0.95, "r=-0.45***", va='top', ha='right', transform=ax2.transAxes, fontsize=20)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((x[NNid],y[NNid]), os.path.expanduser(data_dir +'Y_fig_3_a_bottom_PL5ex_nrem_active_2.joblib'), compress=3)
joblib.dump((x[RRid],y[RRid]), os.path.expanduser(data_dir +'Y_fig_3_a_bottom_PL5ex_rem_active_2.joblib'), compress=3)
#joblib.dump((x,Y1), os.path.expanduser(data_dir +'Y_fig_3_a_bottom_PL5ex_2_sp.joblib'), compress=3)

joblib.dump((x[non],y[non]), os.path.expanduser(data_dir +'Y_fig_3_a_bottom_PL5ex_non_active_2.joblib'), compress=3)


#BLA
rats=['achel180320', 'duvel190505','estrella180808','guiness181002','hoegaarden181115','innis190601','jever190814','leffe200124','maredsous200224','nostrum200304','oberon200325']

data=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (d) = joblib.load(data_dir + rat_name + '_Y_BLAex_data_modulation_index_2.joblib')
    data=np.append(data,d)

N_in=[]
N_out=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (N_in_cell,N_out_cell,*_) = joblib.load(data_dir + rat_name + '_Y_BLAex_swr.joblib')
    N_in = np.append(N_in, N_in_cell)
    N_out = np.append(N_out, N_out_cell)

gain_N=N_in/N_out

x=data
y=gain_N

idx=np.isfinite(x)&np.isfinite(y)
#idx=np.isfinite(x)&np.isfinite(y)
a, b = np.polyfit(x[idx], y[idx], 1)
Y1 = a * x + b
np.corrcoef(x[idx], y[idx])[0][1]
r, p = spearmanr(x[idx], y[idx])
print('y=', a, '*x', '+', b)
print('r=', r)
print('p=', p)
ax3.plot(x,Y1,color='black')

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
(all_time_N_id,all_time_R_id)=joblib.load(data_dir + 'Y_BLAex_n_r_id_2_50_2_outof_b.joblib')
NNid = np.where(all_time_N_id == 1)
RRid = np.where(all_time_R_id == 2)
non=np.where((all_time_N_id == 0) & (all_time_R_id == 0))

from scipy.stats import pearsonr
#ax1.scatter(x,y, color='blue',s=10)
ax3.scatter(x[RRid],y[RRid], color='violet',s=10)
ax3.scatter(x[NNid],y[NNid], color='cornflowerblue',s=10)
#ax3.scatter(x,y, color='blue',s=10)
ax3.set_xlabel('Modulation Index', fontsize=20)
ax3.set_ylabel('SWR gain', fontsize=20)
ax3.text(0.95, 0.95, "r=-0.26***", va='top', ha='right', transform=ax3.transAxes, fontsize=20)

plt.subplots_adjust(wspace=0.3,bottom=0.2)
ax1.set_title('vCA1ex', fontsize=25)
ax2.set_title('PL5ex', fontsize=25)
ax3.set_title('BLAex', fontsize=25)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((x[NNid],y[NNid]), os.path.expanduser(data_dir +'Y_fig_3_a_bottom_BLAex_nrem_active_2.joblib'), compress=3)
joblib.dump((x[RRid],y[RRid]), os.path.expanduser(data_dir +'Y_fig_3_a_bottom_BLAex_rem_active_2.joblib'), compress=3)
#joblib.dump((x,Y1), os.path.expanduser(data_dir +'Y_fig_3_a_bottom_BLAex_2_sp.joblib'), compress=3)


joblib.dump((x[non],y[non]), os.path.expanduser(data_dir +'Y_fig_3_a_bottom_BLAex_non_active_2.joblib'), compress=3)



sns.set_context('poster')
fig = plt.figure(figsize=(24, 6))
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)

#vCA1
rats=['duvel190505','hoegaarden181115','innis190601','karmeliet190901','leffe200124','maredsous200224','nostrum200304','oberon200325']


data=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (d) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_data_modulation_index_2.joblib')
    data=np.append(data,d)

N_in=[]
N_out=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (N_in_cell,N_out_cell,*_) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_spindle.joblib')
    N_in = np.append(N_in, N_in_cell)
    N_out = np.append(N_out, N_out_cell)

gain_N=N_in/N_out

x=data
y=gain_N

idx=np.isfinite(x)&np.isfinite(y)
#idx=np.isfinite(x)&np.isfinite(y)
a, b = np.polyfit(x[idx], y[idx], 1)
Y1 = a * x + b
np.corrcoef(x[idx], y[idx])[0][1]
r, p = spearmanr(x[idx], y[idx])
print('y=', a, '*x', '+', b)
print('r=', r)
print('p=', p)
ax1.plot(x,Y1,color='black')


data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
(all_time_N_id,all_time_R_id)=joblib.load(data_dir + 'Y_vCA1ex_n_r_id_2_50_2.joblib')
NNid = np.where(all_time_N_id == 1)
RRid = np.where(all_time_R_id == 2)
non=np.where((all_time_N_id == 0) & (all_time_R_id == 0))

from scipy.stats import pearsonr
#ax1.scatter(x,y, color='blue',s=10)
ax1.scatter(x[RRid],y[RRid], color='violet',s=10)
ax1.scatter(x[NNid],y[NNid], color='cornflowerblue',s=10)
ax1.set_xlabel('Modulation Index', fontsize=20)

from scipy.stats import pearsonr
#ax1.scatter(x,y, color='blue',s=10)
ax1.set_xlabel('Modulation Index', fontsize=20)
ax1.set_ylabel('Spindle gain', fontsize=20)
ax1.text(0.95, 0.95, "r=-0.44*", va='top', ha='right', transform=ax1.transAxes, fontsize=20)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((x[NNid],y[NNid]), os.path.expanduser(data_dir +'Y_fig_3_b_bottom_vCA1ex_nrem_active_2.joblib'), compress=3)
joblib.dump((x[RRid],y[RRid]), os.path.expanduser(data_dir +'Y_fig_3_b_bottom_vCA1ex_rem_active_2.joblib'), compress=3)
#joblib.dump((x,Y1), os.path.expanduser(data_dir +'Y_fig_3_b_bottom_vCA1ex_2.joblib'), compress=3)

joblib.dump((x[non],y[non]), os.path.expanduser(data_dir +'Y_fig_3_b_bottom_vCA1ex_non_active_2.joblib'), compress=3)


#PL5
rats=['hoegaarden181115','innis190601','jever190814','karmeliet190901','leffe200124','maredsous200224','nostrum200304','oberon200325']

data=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (d) = joblib.load(data_dir + rat_name + '_Y_PL5ex_data_modulation_index_2.joblib')
    data=np.append(data,d)

N_in=[]
N_out=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (N_in_cell,N_out_cell,*_) = joblib.load(data_dir + rat_name + '_Y_PL5ex_spindle.joblib')
    N_in = np.append(N_in, N_in_cell)
    N_out = np.append(N_out, N_out_cell)

gain_N=N_in/N_out

x=data
y=gain_N

idx=np.isfinite(x)&np.isfinite(y)
#idx=np.isfinite(x)&np.isfinite(y)
a, b = np.polyfit(x[idx], y[idx], 1)
Y1 = a * x + b
np.corrcoef(x[idx], y[idx])[0][1]
#r, p = pearsonr(x[idx], y[idx])
r, p = spearmanr(x[idx], y[idx])
print('y=', a, '*x', '+', b)
print('r=', r)
print('p=', p)
ax2.plot(x,Y1,color='black')


data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
(all_time_N_id,all_time_R_id)=joblib.load(data_dir + 'Y_PL5ex_n_r_id_2_50_2.joblib')
NNid = np.where(all_time_N_id == 1)
RRid = np.where(all_time_R_id == 2)
non=np.where((all_time_N_id == 0) & (all_time_R_id == 0))

from scipy.stats import pearsonr
#ax1.scatter(x,y, color='blue',s=10)
ax2.scatter(x[RRid],y[RRid], color='violet',s=10)
ax2.scatter(x[NNid],y[NNid], color='cornflowerblue',s=10)
#ax2.scatter(x,y, color='blue',s=10)
ax2.set_xlabel('Modulation Index', fontsize=20)
ax2.set_ylabel('Spindle gain', fontsize=20)
ax2.text(0.95, 0.95, "r=-0.35***", va='top', ha='right', transform=ax2.transAxes, fontsize=20)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((x[NNid],y[NNid]), os.path.expanduser(data_dir +'Y_fig_3_b_bottom_PL5ex_nrem_active_2.joblib'), compress=3)
joblib.dump((x[RRid],y[RRid]), os.path.expanduser(data_dir +'Y_fig_3_b_bottom_PL5ex_rem_active_2.joblib'), compress=3)
#joblib.dump((x,Y1), os.path.expanduser(data_dir +'Y_fig_3_b_bottom_PL5ex_2.joblib'), compress=3)

joblib.dump((x[non],y[non]), os.path.expanduser(data_dir +'Y_fig_3_b_bottom_PL5ex_non_active_2.joblib'), compress=3)

#BLA
rats=['achel180320', 'duvel190505','estrella180808','guiness181002','hoegaarden181115','innis190601','jever190814','leffe200124','maredsous200224','nostrum200304','oberon200325']

data=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (d) = joblib.load(data_dir + rat_name + '_Y_BLAex_data_modulation_index_2.joblib')
    data=np.append(data,d)

N_in=[]
N_out=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (N_in_cell,N_out_cell,*_) = joblib.load(data_dir + rat_name + '_Y_BLAex_spindle.joblib')
    N_in = np.append(N_in, N_in_cell)
    N_out = np.append(N_out, N_out_cell)

gain_N=N_in/N_out

x=data
y=gain_N

idx=np.isfinite(x)&np.isfinite(y)
#idx=np.isfinite(x)&np.isfinite(y)
a, b = np.polyfit(x[idx], y[idx], 1)
Y1 = a * x + b
np.corrcoef(x[idx], y[idx])[0][1]
#r, p = pearsonr(x[idx], y[idx])
r, p = spearmanr(x[idx], y[idx])
print('y=', a, '*x', '+', b)
print('r=', r)
print('p=', p)
ax3.plot(x,Y1,color='black')

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
(all_time_N_id,all_time_R_id)=joblib.load(data_dir + 'Y_BLAex_n_r_id_2_50_2_outof_b.joblib')
NNid = np.where(all_time_N_id == 1)
RRid = np.where(all_time_R_id == 2)
non=np.where((all_time_N_id == 0) & (all_time_R_id == 0))

from scipy.stats import pearsonr
#ax1.scatter(x,y, color='blue',s=10)
ax3.scatter(x[RRid],y[RRid], color='violet',s=10)
ax3.scatter(x[NNid],y[NNid], color='cornflowerblue',s=10)
#ax3.scatter(x,y, color='blue',s=10)
ax3.set_xlabel('Modulation Index', fontsize=20)
ax3.set_ylabel('Spindle gain', fontsize=20)
ax3.text(0.95, 0.95, "r=-0.39***", va='top', ha='right', transform=ax3.transAxes, fontsize=20)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((x[NNid],y[NNid]), os.path.expanduser(data_dir +'Y_fig_3_b_bottom_BLAex_nrem_active_2.joblib'), compress=3)
joblib.dump((x[RRid],y[RRid]), os.path.expanduser(data_dir +'Y_fig_3_b_bottom_BLAex_rem_active_2.joblib'), compress=3)
#joblib.dump((x,Y1), os.path.expanduser(data_dir +'Y_fig_3_b_bottom_BLAex_2.joblib'), compress=3)

joblib.dump((x[non],y[non]), os.path.expanduser(data_dir +'Y_fig_3_b_bottom_BLAex_non_active_2.joblib'), compress=3)


plt.subplots_adjust(wspace=0.3,bottom=0.2)
ax1.set_title('vCA1ex', fontsize=25)
ax2.set_title('PL5ex', fontsize=25)
ax3.set_title('BLAex', fontsize=25)


sns.set_context('poster')
fig = plt.figure(figsize=(24, 6))
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)

#vCA1
rats=['duvel190505','hoegaarden181115','innis190601','karmeliet190901','leffe200124','maredsous200224','nostrum200304','oberon200325']

data=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (d) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_data_modulation_index_2.joblib')
    data=np.append(data,d)

N_in=[]
N_out=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (N_in_cell,N_out_cell,*_) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_hfo.joblib')
    N_in = np.append(N_in, N_in_cell)
    N_out = np.append(N_out, N_out_cell)

gain_N=N_in/N_out

x=data
y=gain_N

idx=np.isfinite(x)&np.isfinite(y)
#idx=np.isfinite(x)&np.isfinite(y)
a, b = np.polyfit(x[idx], y[idx], 1)
Y1 = a * x + b
np.corrcoef(x[idx], y[idx])[0][1]
r, p = spearmanr(x[idx], y[idx])
print('y=', a, '*x', '+', b)
print('r=', r)
print('p=', p)
ax1.plot(x,Y1,color='black')

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
(all_time_N_id,all_time_R_id)=joblib.load(data_dir + 'Y_vCA1ex_n_r_id_2_50_2.joblib')
NNid = np.where(all_time_N_id == 1)
RRid = np.where(all_time_R_id == 2)
non=np.where((all_time_N_id == 0) & (all_time_R_id == 0))

from scipy.stats import pearsonr
#ax1.scatter(x,y, color='blue',s=10)
ax1.scatter(x[RRid],y[RRid], color='violet',s=10)
ax1.scatter(x[NNid],y[NNid], color='cornflowerblue',s=10)
ax1.set_xlabel('Modulation Index', fontsize=20)
#ax1.scatter(x,y, color='blue',s=10)
ax1.set_xlabel('Modulation Index', fontsize=20)
ax1.set_ylabel('HFO gain', fontsize=20)
ax1.text(0.95, 0.95, "r=0.09", va='top', ha='right', transform=ax1.transAxes, fontsize=20)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((x[NNid],y[NNid]), os.path.expanduser(data_dir +'Y_fig_3_c_bottom_vCA1ex_nrem_active_2.joblib'), compress=3)
joblib.dump((x[RRid],y[RRid]), os.path.expanduser(data_dir +'Y_fig_3_c_bottom_vCA1ex_rem_active_2.joblib'), compress=3)
#joblib.dump((x,Y1), os.path.expanduser(data_dir +'Y_fig_3_c_bottom_vCA1ex_2.joblib'), compress=3)

joblib.dump((x[non],y[non]), os.path.expanduser(data_dir +'Y_fig_3_c_bottom_vCA1ex_non_active_2.joblib'), compress=3)


#PL5
rats=['hoegaarden181115','innis190601','jever190814','karmeliet190901','leffe200124','maredsous200224','nostrum200304','oberon200325']

data=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (d) = joblib.load(data_dir + rat_name + '_Y_PL5ex_data_modulation_index_2.joblib')
    data=np.append(data,d)


N_in=[]
N_out=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (N_in_cell,N_out_cell,*_) = joblib.load(data_dir + rat_name + '_Y_PL5ex_hfo.joblib')
    N_in = np.append(N_in, N_in_cell)
    N_out = np.append(N_out, N_out_cell)

gain_N=N_in/N_out

x=data
y=gain_N

idx=np.isfinite(x)&np.isfinite(y)
#idx=np.isfinite(x)&np.isfinite(y)
a, b = np.polyfit(x[idx], y[idx], 1)
Y1 = a * x + b
np.corrcoef(x[idx], y[idx])[0][1]
r, p = spearmanr(x[idx], y[idx])
print('y=', a, '*x', '+', b)
print('r=', r)
print('p=', p)
ax2.plot(x,Y1,color='black')

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
(all_time_N_id,all_time_R_id)=joblib.load(data_dir + 'Y_PL5ex_n_r_id_2_50_2.joblib')
NNid = np.where(all_time_N_id == 1)
RRid = np.where(all_time_R_id == 2)
non=np.where((all_time_N_id == 0) & (all_time_R_id == 0))


from scipy.stats import pearsonr
#ax1.scatter(x,y, color='blue',s=10)
ax2.scatter(x[RRid],y[RRid], color='violet',s=10)
ax2.scatter(x[NNid],y[NNid], color='cornflowerblue',s=10)
#ax2.scatter(x,y, color='blue',s=10)
ax2.set_xlabel('Modulation Index', fontsize=20)
ax2.set_ylabel('HFO gain', fontsize=20)
ax2.text(0.95, 0.95, "r=-0.11*", va='top', ha='right', transform=ax2.transAxes, fontsize=20)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((x[NNid],y[NNid]), os.path.expanduser(data_dir +'Y_fig_3_c_bottom_PL5ex_nrem_active_2.joblib'), compress=3)
joblib.dump((x[RRid],y[RRid]), os.path.expanduser(data_dir +'Y_fig_3_c_bottom_PL5ex_rem_active_2.joblib'), compress=3)
#joblib.dump((x,Y1), os.path.expanduser(data_dir +'Y_fig_3_c_bottom_PL5ex_2.joblib'), compress=3)

joblib.dump((x[non],y[non]), os.path.expanduser(data_dir +'Y_fig_3_c_bottom_PL5ex_non_active_2.joblib'), compress=3)

#BLA
rats=['achel180320', 'duvel190505','estrella180808','guiness181002','hoegaarden181115','innis190601','jever190814','leffe200124','maredsous200224','nostrum200304','oberon200325']

data=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (d) = joblib.load(data_dir + rat_name + '_Y_BLAex_data_modulation_index_2.joblib')
    data=np.append(data,d)

N_in=[]
N_out=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (N_in_cell,N_out_cell,*_) = joblib.load(data_dir + rat_name + '_Y_BLAex_hfo.joblib')
    N_in = np.append(N_in, N_in_cell)
    N_out = np.append(N_out, N_out_cell)

gain_N=N_in/N_out

x=data
y=gain_N

idx=np.isfinite(x)&np.isfinite(y)
#idx=np.isfinite(x)&np.isfinite(y)
a, b = np.polyfit(x[idx], y[idx], 1)
Y1 = a * x + b
np.corrcoef(x[idx], y[idx])[0][1]
#r, p = pearsonr(x[idx], y[idx])
r, p = spearmanr(x[idx], y[idx])
print('y=', a, '*x', '+', b)
print('r=', r)
print('p=', p)
ax3.plot(x,Y1,color='black')

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
(all_time_N_id,all_time_R_id)=joblib.load(data_dir + 'Y_BLAex_n_r_id_2_50_2_outof_b.joblib')
NNid = np.where(all_time_N_id == 1)
RRid = np.where(all_time_R_id == 2)
non=np.where((all_time_N_id == 0) & (all_time_R_id == 0))

from scipy.stats import pearsonr
#ax1.scatter(x,y, color='blue',s=10)
ax3.scatter(x[RRid],y[RRid], color='violet',s=10)
ax3.scatter(x[NNid],y[NNid], color='cornflowerblue',s=10)
#ax3.scatter(x,y, color='blue',s=10)
ax3.set_xlabel('Modulation Index', fontsize=20)
ax3.set_ylabel('HFO gain', fontsize=20)
ax3.text(0.95, 0.95, "r=-0.22**", va='top', ha='right', transform=ax3.transAxes, fontsize=20)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((x[NNid],y[NNid]), os.path.expanduser(data_dir +'Y_fig_3_c_bottom_BLAex_nrem_active_2.joblib'), compress=3)
joblib.dump((x[RRid],y[RRid]), os.path.expanduser(data_dir +'Y_fig_3_c_bottom_BLAex_rem_active_2.joblib'), compress=3)
#joblib.dump((x,Y1), os.path.expanduser(data_dir +'Y_fig_3_c_bottom_BLAex_2.joblib'), compress=3)


joblib.dump((x[non],y[non]), os.path.expanduser(data_dir +'Y_fig_3_c_bottom_BLAex_non_active_2.joblib'), compress=3)


plt.subplots_adjust(wspace=0.3,bottom=0.2)
ax1.set_title('vCA1ex', fontsize=25)
ax2.set_title('PL5ex', fontsize=25)
ax3.set_title('BLAex', fontsize=25)


sns.set_context('poster')
fig = plt.figure(figsize=(24, 6))
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)

#vCA1
rats=['duvel190505','hoegaarden181115','innis190601','karmeliet190901','leffe200124','maredsous200224','nostrum200304','oberon200325']

data=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (d) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_data_modulation_index_2.joblib')
    data=np.append(data,d)

N_in=[]
N_out=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (N_in_cell,N_out_cell,*_) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_crip.joblib')
    N_in = np.append(N_in, N_in_cell)
    N_out = np.append(N_out, N_out_cell)

gain_N=N_in/N_out

x=data
y=gain_N

idx=np.isfinite(x)&np.isfinite(y)
#idx=np.isfinite(x)&np.isfinite(y)
a, b = np.polyfit(x[idx], y[idx], 1)
Y1 = a * x + b
np.corrcoef(x[idx], y[idx])[0][1]
r, p = spearmanr(x[idx], y[idx])
print('y=', a, '*x', '+', b)
print('r=', r)
print('p=', p)
ax1.plot(x,Y1,color='black')

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
(all_time_N_id,all_time_R_id)=joblib.load(data_dir + 'Y_vCA1ex_n_r_id_2_50_2.joblib')
NNid = np.where(all_time_N_id == 1)
RRid = np.where(all_time_R_id == 2)
non=np.where((all_time_N_id == 0) & (all_time_R_id == 0))

from scipy.stats import pearsonr
#ax1.scatter(x,y, color='blue',s=10)
ax1.scatter(x[RRid],y[RRid], color='violet',s=10)
ax1.scatter(x[NNid],y[NNid], color='cornflowerblue',s=10)
ax1.set_xlabel('Modulation Index', fontsize=20)
#ax1.scatter(x,y, color='blue',s=10)
ax1.set_xlabel('Modulation Index', fontsize=20)
ax1.set_ylabel('Crip gain', fontsize=20)
ax1.text(0.95, 0.95, "r=0.30**", va='top', ha='right', transform=ax1.transAxes, fontsize=20)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((x[NNid],y[NNid]), os.path.expanduser(data_dir +'Y_fig_3_d_bottom_vCA1ex_nrem_active_2.joblib'), compress=3)
joblib.dump((x[RRid],y[RRid]), os.path.expanduser(data_dir +'Y_fig_3_d_bottom_vCA1ex_rem_active_2.joblib'), compress=3)
#joblib.dump((x,Y1), os.path.expanduser(data_dir +'Y_fig_3_d_bottom_vCA1ex_2.joblib'), compress=3)

joblib.dump((x[non],y[non]), os.path.expanduser(data_dir +'Y_fig_3_d_bottom_vCA1ex_non_active_2.joblib'), compress=3)


#PL5
rats=['hoegaarden181115','innis190601','jever190814','karmeliet190901','leffe200124','maredsous200224','nostrum200304','oberon200325']

data=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (d) = joblib.load(data_dir + rat_name + '_Y_PL5ex_data_modulation_index_2.joblib')
    data=np.append(data,d)

N_in=[]
N_out=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (N_in_cell,N_out_cell,*_) = joblib.load(data_dir + rat_name + '_Y_PL5ex_crip.joblib')
    N_in = np.append(N_in, N_in_cell)
    N_out = np.append(N_out, N_out_cell)

gain_N=N_in/N_out

x=data
y=gain_N

idx=np.isfinite(x)&np.isfinite(y)
#idx=np.isfinite(x)&np.isfinite(y)
a, b = np.polyfit(x[idx], y[idx], 1)
Y1 = a * x + b
np.corrcoef(x[idx], y[idx])[0][1]
r, p = spearmanr(x[idx], y[idx])
print('y=', a, '*x', '+', b)
print('r=', r)
print('p=', p)
ax2.plot(x,Y1,color='black')

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
(all_time_N_id,all_time_R_id)=joblib.load(data_dir + 'Y_PL5ex_n_r_id_2_50_2.joblib')
NNid = np.where(all_time_N_id == 1)
RRid = np.where(all_time_R_id == 2)
non=np.where((all_time_N_id == 0) & (all_time_R_id == 0))

from scipy.stats import pearsonr
#ax1.scatter(x,y, color='blue',s=10)
ax2.scatter(x[RRid],y[RRid], color='violet',s=10)
ax2.scatter(x[NNid],y[NNid], color='cornflowerblue',s=10)
#ax2.scatter(x,y, color='blue',s=10)
ax2.set_xlabel('Modulation Index', fontsize=20)
ax2.set_ylabel('Crip gain', fontsize=20)
ax2.text(0.95, 0.95, "r=0.13**", va='top', ha='right', transform=ax2.transAxes, fontsize=20)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((x[NNid],y[NNid]), os.path.expanduser(data_dir +'Y_fig_3_d_bottom_PL5ex_nrem_active_2.joblib'), compress=3)
joblib.dump((x[RRid],y[RRid]), os.path.expanduser(data_dir +'Y_fig_3_d_bottom_PL5ex_rem_active_2.joblib'), compress=3)
#joblib.dump((x,Y1), os.path.expanduser(data_dir +'Y_fig_3_d_bottom_PL5ex_2.joblib'), compress=3)

joblib.dump((x[non],y[non]), os.path.expanduser(data_dir +'Y_fig_3_d_bottom_PL5ex_non_active_2.joblib'), compress=3)


#BLA
rats=['achel180320', 'duvel190505','estrella180808','guiness181002','hoegaarden181115','innis190601','jever190814','leffe200124','maredsous200224','nostrum200304','oberon200325']

data=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (d) = joblib.load(data_dir + rat_name + '_Y_BLAex_data_modulation_index_2.joblib')
    data=np.append(data,d)

N_in=[]
N_out=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (N_in_cell,N_out_cell,*_) = joblib.load(data_dir + rat_name + '_Y_BLAex_crip.joblib')
    N_in = np.append(N_in, N_in_cell)
    N_out = np.append(N_out, N_out_cell)

gain_N=N_in/N_out

x=data
y=gain_N

idx=np.isfinite(x)&np.isfinite(y)
#idx=np.isfinite(x)&np.isfinite(y)
a, b = np.polyfit(x[idx], y[idx], 1)
Y1 = a * x + b
np.corrcoef(x[idx], y[idx])[0][1]
#r, p = pearsonr(x[idx], y[idx])
r, p = spearmanr(x[idx], y[idx])
print('y=', a, '*x', '+', b)
print('r=', r)
print('p=', p)
ax3.plot(x,Y1,color='black')

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
(all_time_N_id,all_time_R_id)=joblib.load(data_dir + 'Y_BLAex_n_r_id_2_50_2_outof_b.joblib')
NNid = np.where(all_time_N_id == 1)
RRid = np.where(all_time_R_id == 2)
non=np.where((all_time_N_id == 0) & (all_time_R_id == 0))

from scipy.stats import pearsonr
#ax1.scatter(x,y, color='blue',s=10)
ax3.scatter(x[RRid],y[RRid], color='violet',s=10)
ax3.scatter(x[NNid],y[NNid], color='cornflowerblue',s=10)
#ax3.scatter(x,y, color='blue',s=10)
ax3.set_xlabel('Modulation Index', fontsize=20)
ax3.set_ylabel('Crip gain', fontsize=20)
ax3.text(0.95, 0.95, "r=0.13", va='top', ha='right', transform=ax3.transAxes, fontsize=20)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((x[NNid],y[NNid]), os.path.expanduser(data_dir +'Y_fig_3_d_bottom_BLAex_nrem_active_2.joblib'), compress=3)
joblib.dump((x[RRid],y[RRid]), os.path.expanduser(data_dir +'Y_fig_3_d_bottom_BLAex_rem_active_2.joblib'), compress=3)
#joblib.dump((x,Y1), os.path.expanduser(data_dir +'Y_fig_3_d_bottom_BLAex_2.joblib'), compress=3)

joblib.dump((x[non],y[non]), os.path.expanduser(data_dir +'Y_fig_3_d_bottom_BLAex_non_active_2.joblib'), compress=3)


plt.subplots_adjust(wspace=0.3,bottom=0.2)

ax1.set_title('vCA1ex', fontsize=25)
ax2.set_title('PL5ex', fontsize=25)
ax3.set_title('BLAex', fontsize=25)


import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
pdf=PdfPages("test9.pdf")
fignums=plt.get_fignums()
for fignum in fignums:
    plt.figure(fignum)
    pdf.savefig()
pdf.close()