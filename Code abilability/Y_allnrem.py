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

    sleep = sleep_states.to_numpy()
    st = sleep_states.query('state=="wake"')
    Data = st.to_numpy()
    st2 = sleep_states.query('state=="nrem"')
    Data2 = st2.to_numpy()
    st3 = sleep_states.query('state=="rem"')
    Data3 = st3.to_numpy()
    st4 = sleep_states.query('state=="rem" or state=="nrem"')
    Data4 = st4.to_numpy()

    mid_t = (st2['end_t'] + st2['start_t']) / 2
    md = mid_t.to_numpy()

    TRange2 = list(zip(st2.start_t, st2.end_t))

    NREM=[]
    for n in range(md.shape[0]):
        if ((TRange2[n][1] - TRange2[n][0] >= 50)):
            NREM=np.append(NREM,n)
    NREM = np.array(NREM, dtype=int)

    S = []
    for n in range(Data4.shape[0] - 1):
        if Data4[n + 1, 0] - Data4[n, 1] <= 60:
            S = np.append(S, n)

    D = np.delete(Data4, 2, axis=1)
    s = np.array(S, dtype=int)
    SS = (s * 2 + 1).tolist() + (s * 2 + 2).tolist()
    np.delete(D, SS)
    D2 = np.delete(D, SS).reshape(-1, 2)

    N = []
    for n in range(D2.shape[0] - 1):
        if D2[n, 1] - D2[n, 0] >= 1800:
            N = np.append(N, n)
    N = np.array(N, dtype=int)

    I = []
    X = []
    for n in N:
        for i in (NREM):
            if D2[n][0] <= md[i] <= D2[n][1]:
                I.append(i)
                x = md[i] - D2[n][0]
                X.append(x)
    I = np.array(I)


    nCell_regions = cluster_info['region'].value_counts()
    regions = nCell_regions[nCell_regions >= 5].index.to_numpy()
    celltypes = np.array(['ex', 'inh'])

    spike_time = spikes['spiketime'].values
    cluster = spikes['cluster'].values

    cell_id = np.sort(cluster_info.index.to_numpy())
    cell_edge = np.hstack((cell_id - 0.5, cell_id[-1] + 0.5))

    target_id = cluster_info.query('region=="BLA" and type=="ex"').index.to_numpy()
    tg = target_id



    Fr_mean2 = []
    for n in range(md.shape[0]):
        time_edge = (TRange2[n])
        fr, *_ = np.histogram2d(cluster, spike_time, bins=[cell_edge, time_edge])
        fr=fr/(time_edge[1]-time_edge[0])
        Fr_mean2 = np.append(Fr_mean2, fr)
    Fr2 = Fr_mean2.reshape(md.shape[0], -1).T

    FFR=[]
    for m in tg:
        for i in I:
            FFR = np.append(FFR, Fr2[m,i])
    F = FFR.reshape(-1, I.shape[0])

    fig = plt.figure(figsize=(6, 6))

    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data\\'
    joblib.dump((F,X,Fr2),os.path.expanduser(data_dir + rat_name + '_Y_BLAex_allnrem.joblib'), compress=3)
    return fig

if __name__ == '__main__':
    main()