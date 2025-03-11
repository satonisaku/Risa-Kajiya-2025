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
    #vCA1_N, vCA1_R =joblib.load(data_dir + rat_name + '_vCA1ex_active.joblib')
    #BLA_N, BLA_R = joblib.load(data_dir + rat_name + '_BLAex_active.joblib')
    #PL5_N, PL5_R = joblib.load(data_dir + rat_name + '_PL5ex_active.joblib')


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

    target_id = cluster_info.query('region=="PL5" and type=="ex"').index.to_numpy()
    tg1 = target_id
    target_id = cluster_info.query('region=="BLA" and type=="ex"').index.to_numpy()
    tg2 = target_id
    TRange = list(zip(st2.start_t, st2.end_t))
    TRange2 = list(zip(st2.start_t, st2.end_t))

    NREM=[]
    for n in range(md.shape[0]):
        if ((TRange2[n][1] - TRange2[n][0] >= 50)):
            NREM=np.append(NREM,n)
    NREM = np.array(NREM, dtype=int)

    # co_firing
    def co_fire(A, B):
        A_mA = A
        B_mB = B
        # corr coeff
        return np.dot(A_mA, B_mB.T)


    events=['swr','spindle','crip','hfo']
    E=swr,spindle,crip,hfo
    DataN = {}
    for e,event in enumerate(events):
        DataN[event] = E[e][['start_t', 'end_t']].to_numpy()

    #swr = swr.to_numpy()
    #swrstart = swr[:, 0]
    #swrend = swr[:, 1]

    #hc1 = time.query('name=="homecage2"')
    #hc1_start = hc1['start_t'].to_numpy()
    #hc1_end = hc1['end_t'].to_numpy()

    #st2 = st2.to_numpy()
    #hc1_st2 = [] #hc1中のnrem sleep
    #for n in range(st2.shape[0]):
        #if (hc1_start[0] <= st2[:, 0][n] and st2[:, 1][n] <= hc1_end[0]):
            #hc1_st2 = np.append(hc1_st2, n)

    #hc1_st2 = np.array(hc1_st2, dtype=int)


    #II = []
    #for n in (hc1_st2):
        #I = []
        #for i in range(swr.shape[0]):
            #if TRange2[n][0] <= swrstart[i] and swrend[i] <= TRange2[n][1]:
                #I = np.append(I, i)
        #II.append(I)

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

    #spike_in = []
    #for n in range(len(TT)):
        #TTs = np.append(TT[n], start[hc1_st2[n]])
        #TTe = np.append(TTs, end[hc1_st2[n]])
        #time_edge = np.sort(TTe)
        #spikecount, *_ = np.histogram2d(cluster, spike_time, bins=[cell_edge, time_edge])
        #spikecount1 = spikecount[np.isin(cell_id, tg1), :]
        #spikecount_in = spikecount1[:, 1::2]
        #condition = spikecount_in == 0
        #spikecount_in[condition] = 0
        #spikecount_in[~condition] = 1
        #spike_in.append(spikecount_in)

    spike_in2 = []
    for e, event in enumerate(events):
        spike_in = []
        for n in range(len(TT2[e])):
            TTs = np.append(TT2[e][n], start[n])
            TTe = np.append(TTs, end[n])
            time_edge = np.sort(TTe)
            spikecount, *_ = np.histogram2d(cluster, spike_time, bins=[cell_edge, time_edge])
            spikecount1 = spikecount[np.isin(cell_id, tg1), :]
            spikecount_in = spikecount1[:, 1::2]
            condition = spikecount_in == 0
            spikecount_in[condition] = 0
            spikecount_in[~condition] = 1
            spike_in.append(spikecount_in)
        spike_in2.append(spike_in)


    Fr1_1=[]
    for e, event in enumerate(events):
    #connect
        fr1_1 = np.empty((tg1.shape[0], 0), int)
        for n in range(len(spike_in2[e])):
            fr1_1= np.append(fr1_1, np.array(spike_in2[e][n]), axis=1)
        Fr1_1.append(fr1_1)

    #2
    spike_in2 = []
    for e, event in enumerate(events):
        spike_in = []
        for n in range(len(TT2[e])):
            TTs = np.append(TT2[e][n], start[n])
            TTe = np.append(TTs, end[n])
            time_edge = np.sort(TTe)
            spikecount, *_ = np.histogram2d(cluster, spike_time, bins=[cell_edge, time_edge])
            spikecount1 = spikecount[np.isin(cell_id, tg2), :]
            spikecount_in = spikecount1[:, 1::2]
            condition = spikecount_in == 0
            spikecount_in[condition] = 0
            spikecount_in[~condition] = 1
            spike_in.append(spikecount_in)
        spike_in2.append(spike_in)

    Fr2_1 = []
    for e, event in enumerate(events):
        # connect
        fr2_1 = np.empty((tg2.shape[0], 0), int)
        for n in range(len(spike_in2[e])):
            fr2_1 = np.append(fr2_1, np.array(spike_in2[e][n]), axis=1)
        Fr2_1.append(fr2_1)

    RR2=[]
    for e, event in enumerate(events):
        # correlation R_R
        na = np.sum(Fr1_1[e], axis=1)
        nb = np.sum(Fr2_1[e], axis=1)
        import math
        RR =[]
        for x in range(na.shape[0]) :
            for y in range(nb.shape[0]):
                N = Fr1_1[e].shape[1]
                nab = co_fire(Fr1_1[e][x], Fr2_1[e][y])
                na_nb = na[x]*nb[y]
                E = na_nb / N
                sd = math.sqrt((na_nb * (N - na[x]) * (N - nb[y])) / (N * N * (N - 1)))
                z = (nab - E) / sd
                RR=np.append(RR,z)
        RR2.append(RR)


    fig = plt.figure(figsize=(6, 6))

    data_dir=r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    joblib.dump((RR2[0]), os.path.expanduser(data_dir+rat_name+'_Y_PL5ex_BLAex_coactivity_all_nrem_swr.joblib'), compress=3)
    joblib.dump((RR2[1]), os.path.expanduser(data_dir + rat_name + '_Y_PL5ex_BLAex_coactivity_all_nrem_spindle.joblib'), compress=3)
    joblib.dump((RR2[2]), os.path.expanduser(data_dir + rat_name + '_Y_PL5ex_BLAex_coactivity_all_nrem_crip.joblib'),compress=3)
    joblib.dump((RR2[3]), os.path.expanduser(data_dir + rat_name + '_Y_PL5ex_BLAex_coactivity_all_nrem_hfo.joblib'), compress=3)
    #joblib.dump((correlation_matrix1_de,Corr_mean), os.path.expanduser(data_dir+rat_name+'_BLAex_N_N_wake_swr_correlation.joblib'), compress=3)
    return fig

if __name__ == '__main__':
    main()