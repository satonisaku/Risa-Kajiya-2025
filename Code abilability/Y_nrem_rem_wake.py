import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def main():
    plot_nrem_fr('achel180320')

def plot_nrem_fr(rat_name):
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\dataset\\'
    _, spikes, cluster_info, *_ = joblib.load(data_dir+rat_name+'\\' + rat_name + '-ok_spikes.joblib')
    _, basic_info, *_ = joblib.load(data_dir+rat_name+'\\' + rat_name+ '-basic_info.joblib')
    _, sleep_states, *_ = joblib.load(data_dir+rat_name+'\\' + rat_name + '-sleep_states.joblib')

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
            fr, *_ = np.histogram2d(cluster, spike_time, bins=[cell_edge, time_edge])
            fr1 = fr[np.isin(cell_id, tg), :]
            n_fr.append(fr1)

    N_fr = np.empty((tg.shape[0], 0), int)
    for n in range(len(n_fr)):
        N_fr = np.append(N_fr, np.array(n_fr[n]), axis=1)
    N_cell = np.mean(N_fr, axis=1)


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
    R_cell = np.mean(R_fr, axis=1)

    # wake
    mid_t = (st4['end_t'] + st4['start_t']) / 2
    wake_md = mid_t.to_numpy()
    wakeTRange = list(zip(st4.start_t, st4.end_t))

    w_fr = []
    for n in range(wake_md.shape[0]):
        if ((wakeTRange[n][1] - wakeTRange[n][0] >= 50)):
            time_edge = np.arange(*wakeTRange[n], 1)
            time_edge = np.append(time_edge, time_edge[-1] + 1)
            fr, *_ = np.histogram2d(cluster, spike_time, bins=[cell_edge, time_edge])
            fr3 = fr[np.isin(cell_id, tg), :]
            w_fr.append(fr3)

    W_fr = np.empty((tg.shape[0], 0), int)
    for n in range(len(w_fr)):
        W_fr = np.append(W_fr, np.array(w_fr[n]), axis=1)
    W_cell = np.mean(W_fr, axis=1)

    fig = plt.figure(figsize=(6, 6))

    data_dir=r'C:\Users\saton\OneDrive\デスクトップ\\data\\'
    joblib.dump((N_fr, R_fr, W_fr), os.path.expanduser(data_dir+rat_name+'_Y_vCA1ex_n_r_w.joblib'), compress=3)
    return fig

if __name__ == '__main__':
    main()