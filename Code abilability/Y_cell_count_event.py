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
    _, _, time, *_ = joblib.load(data_dir+rat_name+'\\' + rat_name +'-basic_info.joblib')
    _, swr, *_ = joblib.load(data_dir+rat_name+'\\' + rat_name +'-hippocampal_swr.joblib')
    _, spindle, *_ = joblib.load(data_dir + rat_name + '\\' + rat_name + '-prefrontal_spindle.joblib')
    _, crip, *_ = joblib.load(data_dir + rat_name + '\\' + rat_name + '-prefrontal_ripples.joblib')
    _, hfo, *_ = joblib.load(data_dir + rat_name + '\\' + rat_name + '-amygdalar_hfo.joblib')

    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #vCA1_N, vCA1_R =joblib.load(data_dir + rat_name + '_Y_vCA1ex_active_2_50_2.joblib')
    #vCA1_N = np.where(vCA1_N == 1)[0]
    #vCA1_R = np.where(vCA1_R == 2)[0]
    BLA_N, BLA_R = joblib.load(data_dir + rat_name + '_Y_BLAex_active_2_50_2.joblib')
    BLA_N = np.where(BLA_N == 1)[0]
    BLA_R = np.where(BLA_R == 2)[0]
    #PL5_N, PL5_R = joblib.load(data_dir + rat_name + '_Y_PL5ex_active_2_50_2.joblib')
    #PL5_N = np.where(PL5_N == 1)[0]
    #PL5_R = np.where(PL5_R == 2)[0]

    sleep = sleep_states.to_numpy()
    st4 = sleep_states.query('state=="rem" or state=="nrem"')
    Data4 = st4.to_numpy()

    st2 = sleep_states.query('state=="nrem"')
    mid_t = (st2['end_t'] + st2['start_t']) / 2
    mid_t2 = (st2['end_t'] - st2['start_t']) / 2
    md = mid_t.to_numpy()
    md2 = mid_t2.to_numpy()

    spike_time = spikes['spiketime'].values
    cluster = spikes['cluster'].values

    cell_id = np.sort(cluster_info.index.to_numpy())
    cell_edge = np.hstack((cell_id - 0.5, cell_id[-1] + 0.5))


    target_id = cluster_info.query('region=="BLA" and type=="ex"').index.to_numpy()
    tg = target_id
    TRange2 = list(zip(st2.start_t, st2.end_t))

    NREM=[]
    for n in range(md.shape[0]):
        if ((TRange2[n][1] - TRange2[n][0] >= 50)):
            NREM=np.append(NREM,n)
    NREM = np.array(NREM, dtype=int)

    vCA1_N = np.array(BLA_N, dtype=int)
    vCA1_R = np.array(BLA_R, dtype=int)
    tg_N = tg[vCA1_N]
    tg_R = tg[vCA1_R]
    array = np.append(vCA1_N, vCA1_R)

    events=['swr','spindle','crip','hfo']
    E=swr,spindle,crip,hfo
    DataN = {}
    for e,event in enumerate(events):
        DataN[event] = E[e][['start_t', 'end_t']].to_numpy()

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


    st2 = sleep_states.query('state=="nrem"')


    Data2 = st2.to_numpy()

    start = Data2[NREM][:, 0]
    end = Data2[NREM][:, 1]

    spike_in2 = []
    for e, event in enumerate(events):
        spike_in = []
        for n in range(len(TT2[e])):
            TTs = np.append(TT2[e][n], start[n])
            TTe = np.append(TTs, end[n])
            time_edge = np.sort(TTe)
            spikecount, *_ = np.histogram2d(cluster, spike_time, bins=[cell_edge, time_edge])
            spikecount1 = spikecount[np.isin(cell_id, tg_R), :]
            spikecount_in = spikecount1[:, 1::2]
            condition = spikecount_in == 0
            spikecount_in[condition] = 0
            spikecount_in[~condition] = 1
            spike_in.append(spikecount_in)
        spike_in2.append(spike_in)


    #connect
    fr1_1_swr = np.empty((tg_R.shape[0], 0), int)
    for n in range(len(spike_in2[0])):
        fr1_1_swr= np.append(fr1_1_swr, np.array(spike_in2[0][n]), axis=1)

    fr1_1_spindle = np.empty((tg_R.shape[0], 0), int)
    for n in range(len(spike_in2[0])):
        fr1_1_spindle= np.append(fr1_1_spindle, np.array(spike_in2[1][n]), axis=1)

    fr1_1_crip = np.empty((tg_R.shape[0], 0), int)
    for n in range(len(spike_in2[0])):
        fr1_1_crip= np.append(fr1_1_crip, np.array(spike_in2[2][n]), axis=1)

    fr1_1_hfo = np.empty((tg_R.shape[0], 0), int)
    for n in range(len(spike_in2[0])):
        fr1_1_hfo= np.append(fr1_1_hfo, np.array(spike_in2[3][n]), axis=1)

    spike_in2 = []
    for e, event in enumerate(events):
        spike_in = []
        spike_out = []
        for n in range(len(TT2[e])):
            TTs = np.append(TT2[e][n], start[n])
            TTe = np.append(TTs, end[n])
            time_edge = np.sort(TTe)
            spikecount, *_ = np.histogram2d(cluster, spike_time, bins=[cell_edge, time_edge])
            spikecount1 = spikecount[np.isin(cell_id, tg_N), :]
            spikecount_in = spikecount1[:, 1::2]
            condition = spikecount_in == 0
            spikecount_in[condition] = 0
            spikecount_in[~condition] = 1
            spike_in.append(spikecount_in)
        spike_in2.append(spike_in)


    # connect
    fr2_1_swr = np.empty((tg_N.shape[0], 0), int)
    for n in range(len(spike_in2[0])):
        fr2_1_swr = np.append(fr2_1_swr, np.array(spike_in2[0][n]), axis=1)

    fr2_1_spindle = np.empty((tg_N.shape[0], 0), int)
    for n in range(len(spike_in2[0])):
        fr2_1_spindle = np.append(fr2_1_spindle, np.array(spike_in2[1][n]), axis=1)

    fr2_1_crip = np.empty((tg_N.shape[0], 0), int)
    for n in range(len(spike_in2[0])):
        fr2_1_crip = np.append(fr2_1_crip, np.array(spike_in2[2][n]), axis=1)

    fr2_1_hfo = np.empty((tg_N.shape[0], 0), int)
    for n in range(len(spike_in2[0])):
        fr2_1_hfo = np.append(fr2_1_hfo, np.array(spike_in2[3][n]), axis=1)

    spike_in2 = []
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
            condition = spikecount_in == 0
            spikecount_in[condition] = 0
            spikecount_in[~condition] = 1
            spike_in.append(spikecount_in)
        spike_in2.append(spike_in)

    # connect
    fr3_1_swr = np.empty((tg.shape[0], 0), int)
    for n in range(len(spike_in2[0])):
        fr3_1_swr = np.append(fr3_1_swr, np.array(spike_in2[0][n]), axis=1)

    fr3_1_spindle = np.empty((tg.shape[0], 0), int)
    for n in range(len(spike_in2[0])):
        fr3_1_spindle = np.append(fr3_1_spindle, np.array(spike_in2[1][n]), axis=1)

    fr3_1_crip = np.empty((tg.shape[0], 0), int)
    for n in range(len(spike_in2[0])):
        fr3_1_crip = np.append(fr3_1_crip, np.array(spike_in2[2][n]), axis=1)

    fr3_1_hfo = np.empty((tg.shape[0], 0), int)
    for n in range(len(spike_in2[0])):
        fr3_1_hfo = np.append(fr3_1_hfo, np.array(spike_in2[3][n]), axis=1)



    import matplotlib.pyplot as plt
    #fig, ax = plt.subplots(1, 1)
    #ax.hist(count1_1, bins=15)
    #ax.hist(count1_2)

    fig = plt.figure(figsize=(6, 6))

    data_dir=r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    joblib.dump((fr1_1_swr,fr2_1_swr,fr3_1_swr), os.path.expanduser(data_dir+rat_name+'_Y_BLAex_cell_count_swr_2.joblib'), compress=3)
    joblib.dump((fr1_1_spindle,fr2_1_spindle,fr3_1_spindle), os.path.expanduser(data_dir+rat_name+'_Y_BLAex_cell_count_spindle_2.joblib'), compress=3)
    joblib.dump((fr1_1_crip,fr2_1_crip,fr3_1_crip), os.path.expanduser(data_dir+rat_name+'_Y_BLAex_cell_count_crip_2.joblib'), compress=3)
    joblib.dump((fr1_1_hfo,fr2_1_hfo,fr3_1_hfo), os.path.expanduser(data_dir+rat_name+'_Y_BLAex_cell_count_hfo_2.joblib'), compress=3)

    return fig

if __name__ == '__main__':
    main()