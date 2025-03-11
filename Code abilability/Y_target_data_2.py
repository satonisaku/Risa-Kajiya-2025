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
    data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    (N_fr, R_fr) = joblib.load(data_dir + rat_name + '_Y_vCA1ex_n_r_1_50_2.joblib')

    data_diff = []
    n_act = []
    r_act = []
    for i in range(N_fr.shape[0]):
        g1 = R_fr[i].tolist()
        g2 = N_fr[i].tolist()
        data_diff = np.append(data_diff,
                              (sum(g1) / len(g1) - sum(g2) / len(g2)) / (sum(g1) / len(g1) + sum(g2) / len(g2)))

        def nr_func(g1, g2):
            g = g1 + g2
            random.shuffle(g)
            new_g1 = g[:len(g1)]
            new_g2 = g[len(g1):]
            diff = ((sum(new_g1) / len(new_g1) - sum(new_g2) / len(new_g2))) / (
            (sum(new_g1) / len(new_g1) + sum(new_g2) / len(new_g2)))
            return diff

        #res = []
        #for i in range(1000):
            #res.append(nr_func(g1, g2))
        #n_act = np.append(n_act, (np.quantile(res, [0.025])))
        #r_act = np.append(r_act, (np.quantile(res, [0.075])))


    fig = plt.figure(figsize=(6, 6))

    data_dir=r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'
    joblib.dump((data_diff), os.path.expanduser(data_dir+rat_name+'_Y_vCA1ex_data_modulation_index_2.joblib'), compress=3)
    return fig

if __name__ == '__main__':
    main()



