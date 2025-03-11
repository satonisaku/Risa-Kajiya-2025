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
    _, pfc_slow_wave, *_ = joblib.load(data_dir+rat_name+'\\' + rat_name +'-prefrontal_slow_wave.joblib')
    _, swr, *_ = joblib.load(data_dir+rat_name+'\\' + rat_name +'-hippocampal_swr.joblib')
    _, spindle, *_ = joblib.load(data_dir + rat_name + '\\' + rat_name + '-prefrontal_spindle.joblib')
    _, crip, *_ = joblib.load(data_dir + rat_name + '\\' + rat_name + '-prefrontal_ripples.joblib')
    _, hfo, *_ = joblib.load(data_dir + rat_name + '\\' + rat_name + '-amygdalar_hfo.joblib')

    cell_id = np.sort(cluster_info.index.to_numpy())
    cell_edge = np.hstack((cell_id - 0.5, cell_id[-1] + 0.5))
    spike_time = spikes['spiketime'].values
    cluster = spikes['cluster'].values

    st2 = sleep_states.query('state=="wake"')
    Data2 = st2.to_numpy()

    mid_t = (st2['end_t'] + st2['start_t']) / 2
    md = mid_t.to_numpy()

    TRange2 = list(zip(st2.start_t, st2.end_t))

    NREM=[]
    for n in range(md.shape[0]):
        if ((TRange2[n][1] - TRange2[n][0] >= 50)):
            NREM=np.append(NREM,n)
    NREM = np.array(NREM, dtype=int)

    target_id = cluster_info.query('region=="BLA" and type=="ex"').index.to_numpy()
    tg = target_id


    events=['swr','spindle','crip','hfo']
    E=swr,spindle,crip,hfo
    DataN = {}
    for e,event in enumerate(events):
        DataN[event] = E[e][['start_t', 'end_t']].to_numpy()


    # nrem no nakano swr
    mid_t = (st2['end_t'] + st2['start_t']) / 2
    md = mid_t.to_numpy()


    II3 = []
    for e, event in enumerate(events):
        II2 = []
        for n in (NREM):
            I = []
            for i in range(E[e].shape[0]):
                if TRange2[n][0] <= DataN[event][:,0][i] and DataN[event][:,1][i] <= TRange2[n][1]:
                   I = np.append(I, i)
            II2.append(I)
        II3.append(II2)


    # in swr no time
    TT2=[]
    for e, event in enumerate(events):
        TT = []
        for n in range(len(II3[e])):
            II3[e][n] = np.array(II3[e][n], dtype=int)
            T = []
            for i in (II3[e][n]):
                time = (DataN[event][:,0][i], DataN[event][:,1][i])
                T.append(time)
            TT.append(T)
        TT2.append(TT)


    start = Data2[NREM][:, 0]
    end = Data2[NREM][:, 1]

    spike_in2 = []
    spike_out2 = []
    for e, event in enumerate(events):
        spike_in = []
        spike_out = []
        for n in range(len(TT2[e])):
            TTs = np.append(TT2[e][n], start[n])
            TTe = np.append(TTs, end[n])
            time_edge = np.sort(TTe)
            spikecount, *_ = np.histogram2d(cluster, spike_time, bins=[cell_edge, time_edge])
            spikecount1 = spikecount[np.isin(cell_id, tg), :]
            spikecount_in = spikecount1[:, 1::2]
            count_in = np.sum(spikecount_in, axis=1)
            spike_in = np.append(spike_in, count_in)
            spikecount_out = spikecount1[:, ::2]
            count_out = np.sum(spikecount_out, axis=1)
            spike_out = np.append(spike_out, count_out)
        spike_in2.append(spike_in)
        spike_out2.append(spike_out)

    swr_cell_in = spike_in2[0].reshape(-1, tg.shape[0]).T
    swr_cell_out = spike_out2[0].reshape(-1, tg.shape[0]).T


    T_in2 = []
    T_out2 = []
    for e, event in enumerate(events):
        T_in = []
        T_out = []
        for n in range(len(TT2[e])):
            TTs = np.append(TT2[e][n], start[n])
            TTe = np.append(TTs, end[n])
            time_edge = np.sort(TTe)
            t = []
            for m in range(time_edge.shape[0] - 1):
                t = np.append(t, time_edge[m + 1] - time_edge[m])
            t_in = t[1::2]
            T_in.append(t_in)
            t_out = t[::2]
            T_out.append(t_out)
        T_in2.append(T_in)
        T_out2.append(T_out)

    T_in3=[]
    T_out3=[]
    for e, event in enumerate(events):
        T_in_sum=[]
        T_out_sum = []
        for m in range(len(T_in2[0])):
            T_in_sum=np.append(T_in_sum,np.nansum(T_in2[e][m]))
            T_out_sum=np.append(T_out_sum,np.nansum(T_out2[e][m]))
        T_in3.append(T_in_sum)
        T_out3.append(T_out_sum)

    swr_in=np.nansum(swr_cell_in,axis=1)/np.nansum(T_in3[0])
    swr_out = np.nansum(swr_cell_out, axis=1) / np.nansum(T_out3[0])


    fig = plt.figure(figsize=(6, 6))
    data_dir=r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    joblib.dump((swr_in,swr_out), os.path.expanduser(data_dir+rat_name+'_Y_BLAex_wake_swr.joblib'), compress=3)
    #joblib.dump((spindle_in,spindle_out,spindle_in_F,spindle_in_L,spindle_out_F,spindle_out_L), os.path.expanduser(data_dir+rat_name+'_Y_vCA1ex_spindle.joblib'), compress=3)
    #joblib.dump((crip_in,crip_out,crip_in_F,crip_in_L,crip_out_F,crip_out_L), os.path.expanduser(data_dir+rat_name+'_Y_vCA1ex_crip.joblib'), compress=3)
    #joblib.dump((hfo_in,hfo_out,hfo_in_F,hfo_in_L,hfo_out_F,hfo_out_L), os.path.expanduser(data_dir+rat_name+'_Y_vCA1ex_hfo.joblib'), compress=3)

if __name__ == '__main__':
    main()