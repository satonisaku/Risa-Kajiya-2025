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

    st = sleep_states.query('state=="wake"')
    st2 = sleep_states.query('state=="nrem"')
    st4 = sleep_states.query('state=="rem" or state=="nrem"')
    Data4 = st4.to_numpy()
    Data2 = st2.to_numpy()

    mid_t = (st2['end_t'] + st2['start_t']) / 2
    md = mid_t.to_numpy()

    events=['swr','spindle','crip','hfo']
    E=swr,spindle,crip,hfo
    DataN = {}
    for e,event in enumerate(events):
        DataN[event] = E[e][['start_t', 'end_t']].to_numpy()

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


    II = []
    XX = []
    for m in range(N.shape[0]):
        X = []
        I = []
        for i in range(md.shape[0]):
            if D2[N[m]][0] <= md[i] <= D2[N[m]][1]:
                I.append(i)
                x = md[i] - D2[N[m]][0]
                X.append(x)
        II.append(I)
        XX.append(X)

    # nrem no nakano swr
    mid_t = (st2['end_t'] + st2['start_t']) / 2
    md = mid_t.to_numpy()
    TRange2 = list(zip(st2.start_t, st2.end_t))

    NREM=[]
    for n in range(md.shape[0]):
        if ((TRange2[n][1] - TRange2[n][0] >= 50)):
            NREM=np.append(NREM,n)
    NREM = np.array(NREM, dtype=int)

    II3 = []
    N_duration2=[]
    for e, event in enumerate(events):
        II2 = []
        N_duration=[]
        for n in (NREM):
            I = []
            for i in range(E[e].shape[0]):
                if TRange2[n][0] <= DataN[event][:,0][i] and DataN[event][:,1][i] <= TRange2[n][1]:
                   I = np.append(I, i)
                   NREM_duration=(TRange2[n][1]-TRange2[n][0])
            II2.append(I)
            N_duration.append(NREM_duration)
        II3.append(II2)
        N_duration2.append(N_duration)

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

    F=[]
    for n in range(N.shape[0]):
       first=II[n][0]
       F=np.append(F,first)
    F = np.array(F, dtype=int)


    L = []
    for n in range(N.shape[0]):
        last = II[n][-1]
        L = np.append(L, last)
    L = np.array(L, dtype=int)


    start = Data2[NREM][:, 0]
    end = Data2[NREM][:, 1]


    T_in2 = []
    for e, event in enumerate(events):
        T_in = []
        for n in range(len(TT2[e])):
            T_in.append(len(TT2[e][n]))
        T_in2.append(T_in)


    frequency0=np.nansum(T_in2[0]) / np.nansum(N_duration)
    frequency1=np.nansum(T_in2[1]) / np.nansum(N_duration)
    frequency2=np.nansum(T_in2[2]) / np.nansum(N_duration)
    frequency3=np.nansum(T_in2[3]) / np.nansum(N_duration)

    fig = plt.figure(figsize=(6, 6))
    data_dir=r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    #joblib.dump((frequency0), os.path.expanduser(data_dir+rat_name+'_Y_swr_frequency.joblib'), compress=3)
    #joblib.dump((frequency1), os.path.expanduser(data_dir+rat_name+'_Y_spindle_frequency.joblib'), compress=3)
    #joblib.dump((frequency2), os.path.expanduser(data_dir+rat_name+'_Y_crip_frequency.joblib'), compress=3)
    joblib.dump((frequency3), os.path.expanduser(data_dir+rat_name+'_Y_hfo_frequency.joblib'), compress=3)

if __name__ == '__main__':
    main()