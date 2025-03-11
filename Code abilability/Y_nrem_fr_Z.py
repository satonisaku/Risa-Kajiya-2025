import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def main():
    plot_nrem_fr('duvel190505')

def plot_nrem_fr(rat_name):
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\dataset\\'
    _, spikes, cluster_info, *_ = joblib.load(data_dir+rat_name+'\\' + rat_name + '-ok_spikes.joblib')
    _, basic_info, *_ = joblib.load(data_dir+rat_name+'\\' + rat_name+ '-basic_info.joblib')
    _, sleep_states, *_ = joblib.load(data_dir+rat_name+'\\' + rat_name + '-sleep_states.joblib')

    sleep = sleep_states.to_numpy()
    st2 = sleep_states.query('state=="nrem"')
    mid_t = (st2['end_t'] + st2['start_t']) / 2
    mid_t2 = (st2['end_t'] - st2['start_t']) / 2
    md = mid_t.to_numpy()
    md2 = mid_t2.to_numpy()

    nCell_regions = cluster_info['region'].value_counts()
    celltypes = np.array(['ex', 'inh'])

    spike_time = spikes['spiketime'].values
    cluster = spikes['cluster'].values

    cell_id = np.sort(cluster_info.index.to_numpy())
    cell_edge = np.hstack((cell_id - 0.5, cell_id[-1] + 0.5))

    target_id = cluster_info.query('region=="BLA" and type=="ex"').index.to_numpy()
    tg = target_id
    TRange = list(zip(st2.start_t, st2.end_t))

    NREM=[]
    for n in range(md.shape[0]):
        if ((TRange[n][1] - TRange[n][0] >= 50)):
            NREM=np.append(NREM,n)
    NREM = np.array(NREM, dtype=int)

    Fr_mean2 = []
    for n in range(md.shape[0]):
        if ((TRange[n][1] - TRange[n][0] >= 50)):
            time_edge = (TRange[n])
            fr, *_ = np.histogram2d(cluster, spike_time, bins=[cell_edge, time_edge])
            fr=fr[np.isin(cell_id, tg), :]/(time_edge[1]-time_edge[0])
            Fr_mean2 = np.append(Fr_mean2, fr)
    Fr = Fr_mean2.reshape(NREM.shape[0], -1).T

    Fr_mean=np.nanmean(Fr,axis=1)
    Fr_sd = np.nanstd(Fr, axis=1)


    fig = plt.figure(figsize=(6, 6))

    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data\\'
    joblib.dump((Fr_mean,Fr_sd),os.path.expanduser(data_dir + rat_name + '_Y_BLAex_nrem_Z.joblib'), compress=3)
    return fig


if __name__ == '__main__':
    main()