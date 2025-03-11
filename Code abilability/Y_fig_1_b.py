import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import joblib
import sys
import random
import seaborn as sns
#sns.set_context('poster')
import os
import numpy as np
import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
#matplotlib.use("TkAgg")

rat_name='hoegaarden181115'


data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\dataset\\'
_, spikes, cluster_info, *_ = joblib.load(data_dir + rat_name + '\\' + rat_name + '-ok_spikes.joblib')
_, basic_info, *_ = joblib.load(data_dir + rat_name + '\\' + rat_name + '-basic_info.joblib')
_, sleep_states, *_ = joblib.load(data_dir + rat_name + '\\' + rat_name + '-sleep_states.joblib')

cell_id = np.sort(cluster_info.index.to_numpy())
cell_edge = np.hstack((cell_id - 0.5, cell_id[-1] + 0.5))
spike_time = spikes['spiketime'].values
cluster = spikes['cluster'].values

st2 = sleep_states.query('state=="nrem"')
st3 = sleep_states.query('state=="rem"')
st4 = sleep_states.query('state=="wake"')

target_id = cluster_info.query('region=="vCA1" and type=="ex"').index.to_numpy()
tg = target_id

# nrem
mid_t = (st2['end_t'] + st2['start_t']) / 2
md = mid_t.to_numpy()
TRange = list(zip(st2.start_t, st2.end_t))

n_fr = []
for n in range(md.shape[0]):
    if ((TRange[n][1] - TRange[n][0] >= 50)):
        time_edge = np.arange(*TRange[n], 1)
        time_edge = np.append(time_edge, time_edge[-1] + 1)
        spikecount, *_ = np.histogram2d(cluster, spike_time, bins=[cell_edge, time_edge])
        fr = spikecount / 1
        fr1 = fr[np.isin(cell_id, tg), :]
        n_fr.append(fr1)

N_fr = np.empty((tg.shape[0], 0), int)
for n in range(len(n_fr)):
    N_fr = np.append(N_fr, np.array(n_fr[n]), axis=1)

# rem
sleep = sleep_states.to_numpy()

idx_nrem = np.array(np.where(sleep[:, 2] == 'nrem'))
idx_rem = np.array(np.where(sleep[:, 2] == 'rem'))
idx_nrem = np.array([e for row in idx_nrem for e in row])
idx_rem = np.array([e for row in idx_rem for e in row])

idx = []
for i in range(idx_nrem.shape[0] - 1):
    tmp = np.where(idx_nrem[i] + 1 == idx_rem)
    idx = np.append(idx, tmp)
idx = np.array(idx, dtype=int)

idx2 = []
for r in idx:
    tmp2 = idx_rem[r]
    idx2 = np.append(idx2, tmp2)
idx2 = np.array(idx2, dtype=int)

TRange = list(zip(sleep_states.start_t, sleep_states.end_t))
r_fr = []
for i in idx2:
    if ((TRange[i][1] - TRange[i][0] >= 50)):
        time_edge = np.arange(*TRange[i], 1)
        time_edge = np.append(time_edge, time_edge[-1] + 1)
        spikecount, *_ = np.histogram2d(cluster, spike_time, bins=[cell_edge, time_edge])
        fr = spikecount / 1
        fr2 = fr[np.isin(cell_id, tg), :]
        r_fr.append(fr2)

R_fr = np.empty((tg.shape[0], 0), int)
for n in range(len(r_fr)):
    R_fr = np.append(R_fr, np.array(r_fr[n]), axis=1)

import random
def nr_func(g1,g2):
    g=g1+g2
    random.shuffle(g)
    new_g1=g[:len(g1)]
    new_g2=g[len(g1):]
    diff = ((sum(new_g1) / len(new_g1) - sum(new_g2) / len(new_g2))) / ((sum(new_g1) / len(new_g1) + sum(new_g2) / len(new_g2)))
    return diff

#NREM active cell example
g1=R_fr[24].tolist()
g2=N_fr[24].tolist()
data_diff = (sum(g1) / len(g1) - sum(g2) / len(g2)) / (sum(g1) / len(g1) + sum(g2) / len(g2))

g1=R_fr[1].tolist()
g2=N_fr[1].tolist()
data_diff = (sum(g1) / len(g1) - sum(g2) / len(g2)) / (sum(g1) / len(g1) + sum(g2) / len(g2))



res=[]
for i in range (1000):
    res.append(nr_func(g1,g2))

#data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
#joblib.dump((N_fr[24],R_fr[24]), os.path.expanduser(data_dir +'Y_NREM_cell.joblib'), compress=3)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((N_fr[1],R_fr[1]), os.path.expanduser(data_dir +'Y_NREM_cell_2.joblib'), compress=3)

#REM active cell example
g1=R_fr[26].tolist()
g2=N_fr[26].tolist()
data_diff2 = (sum(g1) / len(g1) - sum(g2) / len(g2)) / (sum(g1) / len(g1) + sum(g2) / len(g2))
res2=[]
for i in range (1000):
    res2.append(nr_func(g1,g2))

#data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
#joblib.dump((N_fr[26],R_fr[26]), os.path.expanduser(data_dir +'Y_REM_cell.joblib'), compress=3)

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data5\\'
joblib.dump((N_fr[0],R_fr[0]), os.path.expanduser(data_dir +'Y_REM_cell_2.joblib'), compress=3)


fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

ax1.hist(res)
ax1.axvline(data_diff,color='r')
print(np.quantile(res,[0.975,0.025]))
n_act,r_act=(np.quantile(res,[0.975,0.025]))
#ax1.set_title('REM active cell example', fontsize=20)
#ax1.set_xlabel('data index', fontsize=15)
#ax1.set_ylabel('number of times', fontsize=15)

ax2.hist(res2)
ax2.axvline(data_diff2,color='r')
print(np.quantile(res2,[0.975,0.025]))
n_act,r_act=(np.quantile(res2,[0.975,0.025]))
#ax2.set_title('NREM active cell example', fontsize=20)
#ax2.set_xlabel('data index', fontsize=15)
#ax2.set_ylabel('number of times', fontsize=15)


#data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
joblib.dump((data_diff,np.array(res),data_diff2,np.array(res2)), os.path.expanduser('Y_fig_1_c_2.joblib'), compress=3)

