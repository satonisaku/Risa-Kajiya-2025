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
    #_, hfo, *_ = joblib.load(data_dir + rat_name + '\\' + rat_name + '-amygdalar_hfo.joblib')
    #_, ripples, *_ = joblib.load(data_dir + rat_name + '\\' + rat_name + '-prefrontal_ripples.joblib')

    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    vCA1_N, vCA1_R =joblib.load(data_dir + rat_name + '_Y_vCA1ex_active_2_50_2_hc2.joblib')
    #vCA1_N = np.where(vCA1_N == 1)[0]
    #vCA1_R = np.where(vCA1_R == 2)[0]
    #BLA_N, BLA_R = joblib.load(data_dir + rat_name + '_Y_BLAex_active_2_50_2.joblib')
    PL5_N, PL5_R = joblib.load(data_dir + rat_name + '_Y_PL5ex_active_2_50_2_hc2.joblib')
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

    target_id = cluster_info.query('region=="vCA1" and type=="ex"').index.to_numpy()
    tg1 = target_id
    target_id = cluster_info.query('region=="PL5" and type=="ex"').index.to_numpy()
    tg2 = target_id
    TRange = list(zip(st2.start_t, st2.end_t))
    TRange2 = list(zip(st2.start_t, st2.end_t))

    vCA1_N = np.array(vCA1_N, dtype=int)
    vCA1_R = np.array(vCA1_R, dtype=int)
    PL5_N = np.array(PL5_N, dtype=int)
    PL5_R = np.array(PL5_R, dtype=int)
    #BLA_N = np.array(BLA_N, dtype=int)
    #BLA_R = np.array(BLA_R, dtype=int)


    NN=[]
    for n in range(vCA1_N.shape[0]):
        for m in range(PL5_N.shape[0]):
                nn=vCA1_N[n]*PL5_N[m]
                NN=np.append(NN,nn)

    RR=[]
    for n in range(vCA1_R.shape[0]):
        for m in range(PL5_R.shape[0]):
                rr=vCA1_R[n]*PL5_R[m]
                RR=np.append(RR,rr)

    NR=[]
    for n in range(vCA1_N.shape[0]):
        for m in range(PL5_R.shape[0]):
                nr=vCA1_N[n]*PL5_R[m]
                NR=np.append(NR,nr)

    RN=[]
    for n in range(vCA1_R.shape[0]):
        for m in range(PL5_N.shape[0]):
                rn=vCA1_R[n]*PL5_N[m]
                RN=np.append(RN,rn)


    fig = plt.figure(figsize=(6, 6))

    data_dir=r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    joblib.dump((NN,RR,NR,RN), os.path.expanduser(data_dir+rat_name+'_Y_vCA1ex_PL5ex_NN_RR_NR_RN_hc2.joblib'), compress=3)
    #joblib.dump((NN), os.path.expanduser(data_dir+rat_name+'_BLAex_N_N_coactivity_Zscore_jyogai_wake.joblib'), compress=3)
    #joblib.dump((RN), os.path.expanduser(data_dir+rat_name+'_BLAex_R_N_coactivity_Zscore_jyogai_wake.joblib'), compress=3)
    #joblib.dump((NR), os.path.expanduser(data_dir+rat_name+'_BLAex_N_R_coactivity_Zscore_jyogai_wake.joblib'), compress=3)
    #joblib.dump((correlation_matrix1_de,Corr_mean), os.path.expanduser(data_dir+rat_name+'_BLAex_N_N_wake_swr_correlation.joblib'), compress=3)
    return fig