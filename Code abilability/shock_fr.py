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
    _, pfc_slow_wave, *_ = joblib.load(data_dir+rat_name+'\\' + rat_name +'-prefrontal_slow_wave.joblib')
    #_, swr, *_ = joblib.load(data_dir+rat_name+'\\' + rat_name +'-hippocampal_swr.joblib')
    _, _, time, *_ = joblib.load(data_dir + rat_name + '\\' + rat_name + '-basic_info.joblib')
    _, cue, *_ = joblib.load(data_dir + rat_name + '\\' + rat_name + '-cues.joblib')
    _, shock, *_ = joblib.load(data_dir + rat_name + '\\' + rat_name + '-shocks.joblib')

    Data = shock.to_numpy()

    TRange = []
    for n in range(12):
        TRange = np.append(TRange, (Data[n * 32][0], Data[((n + 1) * 32 - 1)][1]))
    trange_2 = np.stack([TRange[::2] - 2,TRange[1::2] + 2,]).T
    trange = trange_2.tolist()

    spike_time = spikes['spiketime'].values
    cluster = spikes['cluster'].values

    cell_id = np.sort(cluster_info.index.to_numpy())
    cell_edge = np.hstack((cell_id - 0.5, cell_id[-1] + 0.5))

    target_id = cluster_info.query('region=="PL5" and type=="ex"').index.to_numpy()
    tg = target_id

    Fr = []
    for n in range(12):
        time_edge = np.arange(*trange[n], 0.1)
        spikecount, *_ = np.histogram2d(cluster, spike_time, bins=[cell_edge, time_edge])
        spikecount1 = spikecount[np.isin(cell_id, tg), :]/0.1
        Fr.append(spikecount1)

    Fr_mean = np.nanmean(Fr, axis=0)

    fig = plt.figure(figsize=(6, 6))

    data_dir=r'C:\Users\saton\OneDrive\デスクトップ\\data2\\'
    joblib.dump((Fr_mean), os.path.expanduser(data_dir+rat_name+'_PL5ex_shock_12_session1.joblib'), compress=3)
    joblib.dump((Fr), os.path.expanduser(data_dir + rat_name + '_PL5ex_shock_session1.joblib'), compress=3)
    return fig

if __name__ == '__main__':
    main()