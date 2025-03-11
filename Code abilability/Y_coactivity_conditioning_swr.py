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

    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #vCA1_N, vCA1_R =joblib.load(data_dir + rat_name + '_vCA1ex_active.joblib')
    #BLA_N, BLA_R = joblib.load(data_dir + rat_name + '_BLAex_active.joblib')
    #PL5_N, PL5_R = joblib.load(data_dir + rat_name + '_PL5ex_active.joblib')


    st2 = sleep_states.query('state=="wake"')
    mid_t = (st2['end_t'] + st2['start_t']) / 2
    mid_t2 = (st2['end_t'] - st2['start_t']) / 2
    md = mid_t.to_numpy()
    md2 = mid_t2.to_numpy()

    spike_time = spikes['spiketime'].values
    cluster = spikes['cluster'].values

    cell_id = np.sort(cluster_info.index.to_numpy())
    cell_edge = np.hstack((cell_id - 0.5, cell_id[-1] + 0.5))

    target_id = cluster_info.query('region=="vCA1" and type=="ex"').index.to_numpy()
    tg1 = target_id
    target_id = cluster_info.query('region=="BLA" and type=="ex"').index.to_numpy()
    tg2 = target_id
    TRange = list(zip(st2.start_t, st2.end_t))
    TRange2 = list(zip(st2.start_t, st2.end_t))


    # conditioning
    con = time.query('name=="conditioning"')
    TRange_con = list(zip(con.start_t, con.end_t))

    # co_firing
    def co_fire(A, B):
        A_mA = A
        B_mB = B
        # corr coeff
        return np.dot(A_mA, B_mB.T)

    swr = swr.to_numpy()
    swrstart = swr[:, 0]
    swrend = swr[:, 1]

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


    II = []
    for n in range(len(TRange_con)):
        I = []
        for i in range(swr.shape[0]):
            if TRange_con[n][0] <= swrstart[i] and swrend[i] <= TRange_con[n][1]:
                I = np.append(I, i)
        II.append(I)

    TT = []
    for n in range(len(II)):
        II[n] = np.array(II[n], dtype=int)
        T = []
        for i in (II[n]):
            time = (swrstart[i], swrend[i])
            T.append(time)
        TT.append(T)


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

    spike_in = []
    for n in range(len(TT)):
        TTs = np.append(TT[n], TRange_con[n][0])
        TTe = np.append(TTs, TRange_con[n][1])
        time_edge = np.sort(TTe)
        spikecount, *_ = np.histogram2d(cluster, spike_time, bins=[cell_edge, time_edge])
        spikecount1 = spikecount[np.isin(cell_id, tg1), :]
        spikecount_in = spikecount1[:, 1::2]
        condition = spikecount_in == 0
        spikecount_in[condition] = 0
        spikecount_in[~condition] = 1
        spike_in.append(spikecount_in)

    #connect
    fr1_1 = np.empty((tg1.shape[0], 0), int)
    for n in range(len(spike_in)):
        fr1_1= np.append(fr1_1, np.array(spike_in[n]), axis=1)


    #2
    spike_in = []
    for n in range(len(TT)):
        TTs = np.append(TT[n], TRange_con[n][0])
        TTe = np.append(TTs, TRange_con[n][1])
        time_edge = np.sort(TTe)
        spikecount, *_ = np.histogram2d(cluster, spike_time, bins=[cell_edge, time_edge])
        spikecount1 = spikecount[np.isin(cell_id, tg2), :]
        spikecount_in = spikecount1[:, 1::2]
        condition = spikecount_in == 0
        spikecount_in[condition] = 0
        spikecount_in[~condition] = 1
        spike_in.append(spikecount_in)

    #connect
    fr2_1 = np.empty((tg2.shape[0], 0), int)
    for n in range(len(spike_in)):
        fr2_1= np.append(fr2_1, np.array(spike_in[n]), axis=1)


    # correlation R_R
    na = np.sum(fr1_1, axis=1)
    nb = np.sum(fr2_1, axis=1)

    import math
    RR =[]
    for x in range(na.shape[0]) :
        for y in range(nb.shape[0]):
            N = fr1_1.shape[1]
            nab = co_fire(fr1_1[x], fr2_1[y])
            na_nb = na[x]*nb[y]
            E = na_nb / N
            sd = math.sqrt((na_nb * (N - na[x]) * (N - nb[y])) / (N * N * (N - 1)))
            z = (nab - E) / sd
            RR=np.append(RR,z)



    fig = plt.figure(figsize=(6, 6))

    data_dir=r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    joblib.dump((RR), os.path.expanduser(data_dir+rat_name+'_Y_vCA1ex_BLAex_coactivity_conditioning_swr.joblib'), compress=3)
    #joblib.dump((correlation_matrix1_de,Corr_mean), os.path.expanduser(data_dir+rat_name+'_BLAex_N_N_wake_swr_correlation.joblib'), compress=3)
    return fig

if __name__ == '__main__':
    main()