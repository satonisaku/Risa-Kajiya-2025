import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import random

def main():
    plot_nrem_fr('achel180320')

def plot_nrem_fr(rat_name):
    # 各ラットで知りたいとき
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data\\'
    (N_fr, R_fr) = joblib.load(data_dir + rat_name + '_Y_BLAex_n_r_hc4_1.joblib')

    data_diff = []
    n_act = []
    r_act = []
    for i in range(N_fr.shape[0]):
        g1 = R_fr[i].tolist()
        g2 = N_fr[i].tolist()
        if len(g1) == 0 or len(g2) == 0:
            data_diff = np.append(data_diff, 0)
        elif (sum(g1) / len(g1) + sum(g2) / len(g2)) == 0:
            data_diff = np.append(data_diff, 0)
        else:
            data_diff = np.append(data_diff,
                                  (sum(g1) / len(g1) - sum(g2) / len(g2)) / (sum(g1) / len(g1) + sum(g2) / len(g2)))

        if len(g1) == 0 or len(g2) == 0:
            n_act = np.append(n_act, 0)
            r_act = np.append(r_act, 0)
        elif (sum(g1) / len(g1) + sum(g2) / len(g2)) == 0:
            n_act = np.append(n_act, 0)
            r_act = np.append(r_act, 0)
        else:
            def nr_func(g1, g2):
                g = g1 + g2
                random.shuffle(g)
                new_g1 = g[:len(g1)]
                new_g2 = g[len(g1):]

                diff = ((sum(new_g1) / len(new_g1) - sum(new_g2) / len(new_g2))) / (
                (sum(new_g1) / len(new_g1) + sum(new_g2) / len(new_g2)))
                return diff

            res = []
            for i in range(1000):
                res.append(nr_func(g1, g2))
            r_act = np.append(r_act, (np.quantile(res, [0.975])))
            n_act = np.append(n_act, (np.quantile(res, [0.025])))

    num=[]
    n_id=[]
    for i in range(N_fr.shape[0]):
        if data_diff[i]==0:
            n_id = np.append(n_id, 0)
        elif data_diff[i]<=n_act[i]:
           num=np.append(num,data_diff[i])
           n_id = np.append(n_id, 1)
        else:
            n_id = np.append(n_id, 0)
    nremactive_count = len(num)
    print(nremactive_count)
    print(n_id)
    # N_id.append(n_id)

    num=[]
    r_id=[]
    for i in range(R_fr.shape[0]):
        if data_diff[i]==0:
            r_id = np.append(r_id, 0)
        elif data_diff[i]>=r_act[i]:
           num=np.append(num,data_diff[i])
           r_id = np.append(r_id, 2)
        else:
            r_id = np.append(r_id, 0)
    remactive_count=len(num)
    print(remactive_count)
    print(r_id)
    # R_id.append(r_id)

    N=n_id
    R=r_id


    fig = plt.figure(figsize=(6, 6))

    data_dir=r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    joblib.dump((N,R), os.path.expanduser(data_dir+rat_name+'_Y_BLAex_active_2_50_2_hc4.joblib'), compress=3)
    return fig

if __name__ == '__main__':
    main()



