import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from scipy import stats
from scipy.stats import pearsonr
import seaborn as sns
from scipy.stats import spearmanr
sns.set_context('poster')

fig = plt.figure(figsize=(24, 6))
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)

#vCA1
rats=['duvel190505','hoegaarden181115','innis190601','karmeliet190901','leffe200124','maredsous200224','nostrum200304','oberon200325']

#(data)=joblib.load('vCA1ex_n_r_id_1_modulation_index.joblib')
#(hc0data,hc0_N_id,hc0_R_id,hc1data,hc1_N_id,hc1_R_id,hc2data,hc2_N_id,hc2_R_id,hc3data,hc3_N_id,hc3_R_id,hc4data,hc4_N_id,hc4_R_id)=joblib.load('vCA1ex_hc_data_1.joblib')

data=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (d) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_data_modulation_index_2.joblib')
    data=np.append(data,d)


N_in=[]
N_out=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (N_in_cell,N_out_cell,*_) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_wake_swr.joblib')
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
ax1.set_ylabel('SWR gain', fontsize=20)
ax1.text(0.95, 0.95, "r=-0.38***", va='top', ha='right', transform=ax1.transAxes, fontsize=20)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
#joblib.dump((x[NNid],y[NNid]), os.path.expanduser(data_dir +'fig_5_c_bottom_vCA1ex_nrem_active.joblib'), compress=3)
#joblib.dump((x[RRid],y[RRid]), os.path.expanduser(data_dir +'fig_5_c_bottom_vCA1ex_rem_active.joblib'), compress=3)
#joblib.dump((x,Y1), os.path.expanduser(data_dir +'fig_5_c_bottom_vCA1ex.joblib'), compress=3)

joblib.dump((x[NNid],y[NNid]), os.path.expanduser(data_dir +'Y_fig_6_c_bottom_vCA1ex_nrem_active.joblib'), compress=3)
joblib.dump((x[RRid],y[RRid]), os.path.expanduser(data_dir +'Y_fig_6_c_bottom_vCA1ex_rem_active.joblib'), compress=3)
joblib.dump((x,Y1), os.path.expanduser(data_dir +'Y_fig_6_c_bottom_vCA1ex.joblib'), compress=3)

joblib.dump((x[non],y[non]), os.path.expanduser(data_dir +'Y_fig_6_c_bottom_vCA1ex_non_active.joblib'), compress=3)



#PL5
rats=['hoegaarden181115','innis190601','jever190814','karmeliet190901','leffe200124','maredsous200224','nostrum200304','oberon200325']
#(data)=joblib.load('PL5ex_n_r_id_1_modulation_index.joblib')


data=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (d) = joblib.load(data_dir + rat_name + '_Y_PL5ex_data_modulation_index_2.joblib')
    data=np.append(data,d)


N_in=[]
N_out=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (N_in_cell,N_out_cell,*_) = joblib.load(data_dir + rat_name + '_Y_PL5ex_wake_swr.joblib')
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
ax2.text(0.95, 0.95, "r=-0.12*", va='top', ha='right', transform=ax2.transAxes, fontsize=20)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
#joblib.dump((x[NNid],y[NNid]), os.path.expanduser(data_dir +'fig_5_c_bottom_PL5ex_nrem_active.joblib'), compress=3)
#joblib.dump((x[RRid],y[RRid]), os.path.expanduser(data_dir +'fig_5_c_bottom_PL5ex_rem_active.joblib'), compress=3)
#joblib.dump((x,Y1), os.path.expanduser(data_dir +'fig_5_c_bottom_PL5ex.joblib'), compress=3)

joblib.dump((x[NNid],y[NNid]), os.path.expanduser(data_dir +'Y_fig_6_c_bottom_PL5ex_nrem_active.joblib'), compress=3)
joblib.dump((x[RRid],y[RRid]), os.path.expanduser(data_dir +'Y_fig_6_c_bottom_PL5ex_rem_active.joblib'), compress=3)
joblib.dump((x,Y1), os.path.expanduser(data_dir +'Y_fig_6_c_bottom_PL5ex.joblib'), compress=3)

joblib.dump((x[non],y[non]), os.path.expanduser(data_dir +'Y_fig_6_c_bottom_PL5ex_non_active.joblib'), compress=3)


#BLA
rats=['achel180320','duvel190505','estrella180808','guiness181002','hoegaarden181115','innis190601','jever190814','leffe200124','maredsous200224','nostrum200304','oberon200325']


data=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (d) = joblib.load(data_dir + rat_name + '_Y_BLAex_data_modulation_index_2.joblib')
    data=np.append(data,d)

N_in=[]
N_out=[]
for rat_name in rats:
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (N_in_cell,N_out_cell,*_) = joblib.load(data_dir + rat_name + '_y_BLAex_wake_swr.joblib')
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
ax3.text(0.95, 0.95, "r=-0.18*", va='top', ha='right', transform=ax3.transAxes, fontsize=20)

plt.subplots_adjust(wspace=0.3,bottom=0.2)
ax1.set_title('vCA1ex', fontsize=25)
ax2.set_title('PL5ex', fontsize=25)
ax3.set_title('BLAex', fontsize=25)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
#joblib.dump((x[NNid],y[NNid]), os.path.expanduser(data_dir +'fig_5_c_bottom_BLAex_nrem_active.joblib'), compress=3)
#joblib.dump((x[RRid],y[RRid]), os.path.expanduser(data_dir +'fig_5_c_bottom_BLAex_rem_active.joblib'), compress=3)
#joblib.dump((x,Y1), os.path.expanduser(data_dir +'fig_5_c_bottom_BLAex.joblib'), compress=3)

joblib.dump((x[NNid],y[NNid]), os.path.expanduser(data_dir +'Y_fig_6_c_bottom_BLAex_nrem_active.joblib'), compress=3)
joblib.dump((x[RRid],y[RRid]), os.path.expanduser(data_dir +'Y_fig_6_c_bottom_BLAex_rem_active.joblib'), compress=3)
joblib.dump((x,Y1), os.path.expanduser(data_dir +'Y_fig_6_c_bottom_BLAex.joblib'), compress=3)

joblib.dump((x[non],y[non]), os.path.expanduser(data_dir +'Y_fig_6_c_bottom_BLAex_non_active.joblib'), compress=3)


import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
pdf=PdfPages("test9.pdf")
fignums=plt.get_fignums()
for fignum in fignums:
    plt.figure(fignum)
    pdf.savefig()
pdf.close()