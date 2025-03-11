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
    #_, spindle, *_ = joblib.load(data_dir + rat_name + '\\' + rat_name + '-prefrontal_spindle.joblib')
    #_, crip, *_ = joblib.load(data_dir + rat_name + '\\' + rat_name + '-prefrontal_ripples.joblib')
    #_, hfo, *_ = joblib.load(data_dir + rat_name + '\\' + rat_name + '-amygdalar_hfo.joblib')

    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    vCA1_N, vCA1_R =joblib.load(data_dir + rat_name + '_Y_vCA1ex_active_2_50_2.joblib')
    vCA1_N = np.where(vCA1_N == 1)[0]
    vCA1_R = np.where(vCA1_R == 2)[0]
    #BLA_N, BLA_R = joblib.load(data_dir + rat_name + '_Y_BLAex_active_2_50_2.joblib')
    #BLA_N = np.where(BLA_N == 1)[0]
    #BLA_R = np.where(BLA_R == 2)[0]
    #PL5_N, PL5_R = joblib.load(data_dir + rat_name + '_Y_PL5ex_active_2_50_2.joblib')
    #PL5_N = np.where(PL5_N == 1)[0]
    #PL5_R = np.where(PL5_R == 2)[0]

    sleep = sleep_states.to_numpy()
    st4 = sleep_states.query('state=="rem" or state=="nrem"')
    Data4 = st4.to_numpy()

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
    tg = target_id
    TRange2 = list(zip(st2.start_t, st2.end_t))

    # conditioning
    con = time.query('name=="conditioning"')
    TRange_con = list(zip(con.start_t, con.end_t))


    vCA1_N = np.array(vCA1_N, dtype=int)
    vCA1_R = np.array(vCA1_R, dtype=int)
    tg_N = tg[vCA1_N]
    tg_R = tg[vCA1_R]
    array = np.append(vCA1_N, vCA1_R)


    #swr = swr.to_numpy()
    swr = swr.to_numpy()
    swrstart = swr[:, 0]
    swrend = swr[:, 1]


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


    spike_in = []
    for n in range(len(TT)):
        TTs = np.append(TT[n], TRange_con[n][0])
        TTe = np.append(TTs,TRange_con[n][1])
        time_edge = np.sort(TTe)
        spikecount, *_ = np.histogram2d(cluster, spike_time, bins=[cell_edge, time_edge])
        spikecount1 = spikecount[np.isin(cell_id, tg_R), :]
        spikecount_in = spikecount1[:, 1::2]
        condition = spikecount_in == 0
        spikecount_in[condition] = 0
        spikecount_in[~condition] = 1
        spike_in.append(spikecount_in)

    #connect
    fr1_1 = np.empty((tg_R.shape[0], 0), int)
    for n in range(len(spike_in)):
        fr1_1= np.append(fr1_1, np.array(spike_in[n]), axis=1)


    spike_in = []
    for n in range(len(TT)):
        TTs = np.append(TT[n], TRange_con[n][0])
        TTe = np.append(TTs,TRange_con[n][1])
        time_edge = np.sort(TTe)
        spikecount, *_ = np.histogram2d(cluster, spike_time, bins=[cell_edge, time_edge])
        spikecount1 = spikecount[np.isin(cell_id, tg_N), :]
        spikecount_in = spikecount1[:, 1::2]
        condition = spikecount_in == 0
        spikecount_in[condition] = 0
        spikecount_in[~condition] = 1
        spike_in.append(spikecount_in)

    #connect
    fr2_1 = np.empty((tg_N.shape[0], 0), int)
    for n in range(len(spike_in)):
        fr2_1= np.append(fr2_1, np.array(spike_in[n]), axis=1)

    spike_in = []
    for n in range(len(TT)):
        TTs = np.append(TT[n], TRange_con[n][0])
        TTe = np.append(TTs, TRange_con[n][1])
        time_edge = np.sort(TTe)
        spikecount, *_ = np.histogram2d(cluster, spike_time, bins=[cell_edge, time_edge])
        spikecount1 = spikecount[np.isin(cell_id, tg), :]
        spikecount_in = spikecount1[:, 1::2]
        condition = spikecount_in == 0
        spikecount_in[condition] = 0
        spikecount_in[~condition] = 1
        spike_in.append(spikecount_in)

    #connect
    fr3_1 = np.empty((tg.shape[0], 0), int)
    for n in range(len(spike_in)):
        fr3_1= np.append(fr3_1, np.array(spike_in[n]), axis=1)

    #fr1_R=  np.nansum(fr1_1,axis=0)
    #fr1_N=  np.nansum(fr2_1,axis=0)
    #fr1_all = np.nansum(fr3_1, axis=0)


    import matplotlib.pyplot as plt
    #fig, ax = plt.subplots(1, 1)
    #ax.hist(count1_1, bins=15)
    #ax.hist(count1_2)

    fig = plt.figure(figsize=(6, 6))

    data_dir=r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #joblib.dump((fr1_1,fr2_1,fr3_1), os.path.expanduser(data_dir+rat_name+'_Y_vCA1ex_cell_count_swr_2.joblib'), compress=3)
    joblib.dump((fr1_1,fr2_1,fr3_1), os.path.expanduser(data_dir+rat_name+'_Y_vCA1ex_cell_count_conditioning_swr_2.joblib'), compress=3)
    #data_dir=r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #joblib.dump((fr1_1,fr2_1,fr3_1), os.path.expanduser(data_dir+rat_name+'_PL5ex_cell_count_spindle.joblib'), compress=3)
    #joblib.dump((fr1_1, fr2_1, fr3_1), os.path.expanduser(data_dir + rat_name + '_vCA1ex_cell_count_wake.joblib'),compress=3)
    #joblib.dump((predict_N_N,actual_N_N), os.path.expanduser(data_dir+rat_name+'_BLAex_N_N_co_participate.joblib'), compress=3)
    #joblib.dump((predict_R_N,actual_R_N), os.path.expanduser(data_dir+rat_name+'_BLAex_R_N_co_participate.joblib'), compress=3)
    #joblib.dump((correlation_matrix1_de,Corr_mean), os.path.expanduser(data_dir+rat_name+'_BLAex_N_N_wake_swr_correlation.joblib'), compress=3)
    #joblib.dump((predict_R_R, actual_R_R), os.path.expanduser(data_dir + rat_name + '_PL5ex_R_R_co_participate_spindle.joblib'), compress=3)
    #joblib.dump((predict_N_N, actual_N_N), os.path.expanduser(data_dir + rat_name + '_PL5ex_N_N_co_participate_spindle.joblib'),compress=3)
    #joblib.dump((predict_R_N, actual_R_N), os.path.expanduser(data_dir + rat_name + '_PL5ex_R_N_co_participate_spindle.joblib'),compress=3)
    return fig

if __name__ == '__main__':
    main()