import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import os
from scipy import stats
import random

fig = plt.figure(figsize=(24, 16))
ax1 = fig.add_subplot(2, 5, 1)
ax2 = fig.add_subplot(2, 5, 2)
ax3 = fig.add_subplot(2, 5, 3)
ax4 = fig.add_subplot(2, 5, 4)
ax5 = fig.add_subplot(2, 5, 5)
ax6 = fig.add_subplot(2, 5, 6)
ax7 = fig.add_subplot(2, 5, 7)
ax8 = fig.add_subplot(2, 5, 8)
ax9 = fig.add_subplot(2, 5, 9)
ax10 = fig.add_subplot(2, 5, 10)


#vCA1
rats=['duvel190505','hoegaarden181115','innis190601','karmeliet190901','leffe200124','maredsous200224','nostrum200304','oberon200325']

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'

Data_diff=[]
for rat_name in rats:
    (data_diff) = joblib.load(data_dir+rat_name+'_Y_vCA1ex_data_modulation_index_hc0_2.joblib')
    Data_diff=np.append(Data_diff,data_diff)

hc0data=Data_diff

Data_diff=[]
for rat_name in rats:
    (data_diff) = joblib.load(data_dir+rat_name+'_Y_vCA1ex_data_modulation_index_hc1_2.joblib')
    Data_diff = np.append(Data_diff, data_diff)
hc1data=Data_diff

Data_diff=[]
for rat_name in rats:
    (data_diff) = joblib.load(data_dir+rat_name+'_Y_vCA1ex_data_modulation_index_hc2_2.joblib')
    Data_diff = np.append(Data_diff, data_diff)
hc2data=Data_diff

Data_diff=[]
for rat_name in rats:
    (data_diff) = joblib.load(data_dir+rat_name+'_Y_vCA1ex_data_modulation_index_hc3_2.joblib')
    Data_diff = np.append(Data_diff, data_diff)
hc3data=Data_diff

Data_diff=[]
for rat_name in rats:
    (data_diff) = joblib.load(data_dir+rat_name+'_Y_vCA1ex_data_modulation_index_hc4_2.joblib')
    Data_diff = np.append(Data_diff, data_diff)
hc4data=Data_diff


#joblib.dump((hc0data,hc1data,hc2data,hc3data,hc4data), os.path.expanduser('Y_vCA1ex_hc_data_2.joblib'), compress=3)


from scipy.stats import pearsonr
x=hc0data
y=hc1data
a, b = np.polyfit(x, y, 1)
Y2 = a * x + b
np.corrcoef(x, y)[0][1]
r, p1 = pearsonr(x, y)
print(r,p1)

x=hc0data
y=hc2data
a, b = np.polyfit(x, y, 1)
Y2 = a * x + b
np.corrcoef(x, y)[0][1]
r, p2 = pearsonr(x, y)
print(r,p2)

x=hc0data
y=hc3data
a, b = np.polyfit(x, y, 1)
Y2 = a * x + b
np.corrcoef(x, y)[0][1]
r, p3 = pearsonr(x, y)
print(r,p3)

x=hc0data
y=hc4data
a, b = np.polyfit(x, y, 1)
Y2 = a * x + b
np.corrcoef(x, y)[0][1]
r, p4 = pearsonr(x, y)
print(r,p4)

x=hc1data
y=hc2data
a, b = np.polyfit(x, y, 1)
Y2 = a * x + b
np.corrcoef(x, y)[0][1]
r, p5 = pearsonr(x, y)
print(r,p5)

x=hc1data
y=hc3data
a, b = np.polyfit(x, y, 1)
Y2 = a * x + b
np.corrcoef(x, y)[0][1]
r, p6 = pearsonr(x, y)
print(r,p6)

x=hc1data
y=hc4data
a, b = np.polyfit(x, y, 1)
Y2 = a * x + b
np.corrcoef(x, y)[0][1]
r, p7 = pearsonr(x, y)
print(r,p7)

x=hc2data
y=hc3data
a, b = np.polyfit(x, y, 1)
Y2 = a * x + b
np.corrcoef(x, y)[0][1]
r, p8 = pearsonr(x, y)
print(r,p8)

x=hc2data
y=hc4data
a, b = np.polyfit(x, y, 1)
Y2 = a * x + b
np.corrcoef(x, y)[0][1]
r, p9 = pearsonr(x, y)
print(r,p9)

x=hc3data
y=hc4data
a, b = np.polyfit(x, y, 1)
Y2 = a * x + b
np.corrcoef(x, y)[0][1]
r, p10 = pearsonr(x, y)
print(r,p10)

p1=(1,p1,p2,p3,p4,p1,1,p5,p6,p7,p2,p5,1,p8,p9,p3,p6,p8,1,p10,p4,p7,p9,p10,1)

#joblib.dump((p1), os.path.expanduser('Y_fig_1_f_vCA1ex.joblib'), compress=3)


# PL5
rats = ['hoegaarden181115', 'innis190601', 'jever190814', 'karmeliet190901', 'leffe200124', 'maredsous200224','nostrum200304', 'oberon200325']

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'

Data_diff=[]
for rat_name in rats:
    (data_diff) = joblib.load(data_dir+rat_name+'_Y_PL5ex_data_modulation_index_hc0_2.joblib')
    Data_diff=np.append(Data_diff,data_diff)

hc0data=Data_diff

Data_diff=[]
for rat_name in rats:
    (data_diff) = joblib.load(data_dir+rat_name+'_Y_PL5ex_data_modulation_index_hc1_2.joblib')
    Data_diff = np.append(Data_diff, data_diff)
hc1data=Data_diff

Data_diff=[]
for rat_name in rats:
    (data_diff) = joblib.load(data_dir+rat_name+'_Y_PL5ex_data_modulation_index_hc2_2.joblib')
    Data_diff = np.append(Data_diff, data_diff)
hc2data=Data_diff

Data_diff=[]
for rat_name in rats:
    (data_diff) = joblib.load(data_dir+rat_name+'_Y_PL5ex_data_modulation_index_hc3_2.joblib')
    Data_diff = np.append(Data_diff, data_diff)
hc3data=Data_diff

Data_diff=[]
for rat_name in rats:
    (data_diff) = joblib.load(data_dir+rat_name+'_Y_PL5ex_data_modulation_index_hc4_2.joblib')
    Data_diff = np.append(Data_diff, data_diff)
hc4data=Data_diff


#joblib.dump((hc0data,hc1data,hc2data,hc3data,hc4data), os.path.expanduser('Y_PL5ex_hc_data_2.joblib'), compress=3)


from scipy.stats import pearsonr
x=hc0data
y=hc1data
a, b = np.polyfit(x, y, 1)
Y2 = a * x + b
np.corrcoef(x, y)[0][1]
r, p1 = pearsonr(x, y)
print(r,p1)

x=hc0data
y=hc2data
a, b = np.polyfit(x, y, 1)
Y2 = a * x + b
np.corrcoef(x, y)[0][1]
r, p2 = pearsonr(x, y)
print(r,p2)

x=hc0data
y=hc3data
a, b = np.polyfit(x, y, 1)
Y2 = a * x + b
np.corrcoef(x, y)[0][1]
r, p3 = pearsonr(x, y)
print(r,p3)

x=hc0data
y=hc4data
a, b = np.polyfit(x, y, 1)
Y2 = a * x + b
np.corrcoef(x, y)[0][1]
r, p4 = pearsonr(x, y)
print(r,p4)

x=hc1data
y=hc2data
a, b = np.polyfit(x, y, 1)
Y2 = a * x + b
np.corrcoef(x, y)[0][1]
r, p5 = pearsonr(x, y)
print(r,p5)

x=hc1data
y=hc3data
a, b = np.polyfit(x, y, 1)
Y2 = a * x + b
np.corrcoef(x, y)[0][1]
r, p6 = pearsonr(x, y)
print(r,p6)

x=hc1data
y=hc4data
a, b = np.polyfit(x, y, 1)
Y2 = a * x + b
np.corrcoef(x, y)[0][1]
r, p7 = pearsonr(x, y)
print(r,p7)

x=hc2data
y=hc3data
a, b = np.polyfit(x, y, 1)
Y2 = a * x + b
np.corrcoef(x, y)[0][1]
r, p8 = pearsonr(x, y)
print(r,p8)

x=hc2data
y=hc4data
a, b = np.polyfit(x, y, 1)
Y2 = a * x + b
np.corrcoef(x, y)[0][1]
r, p9 = pearsonr(x, y)
print(r,p9)

x=hc3data
y=hc4data
a, b = np.polyfit(x, y, 1)
Y2 = a * x + b
np.corrcoef(x, y)[0][1]
r, p10 = pearsonr(x, y)
print(r,p10)

p1=(1,p1,p2,p3,p4,p1,1,p5,p6,p7,p2,p5,1,p8,p9,p3,p6,p8,1,p10,p4,p7,p9,p10,1)

#joblib.dump((p1), os.path.expanduser('Y_fig_1_f_PL5ex.joblib'), compress=3)



#BLA
rats = ['achel180320', 'booyah180430', 'duvel190505', 'estrella180808', 'guiness181002', 'hoegaarden181115',
        'innis190601', 'jever190814', 'leffe200124', 'maredsous200224', 'nostrum200304', 'oberon200325']

data_dir = r'C:\Users\saton\OneDrive\デスクトップ\\data3\\'

Data_diff=[]
for rat_name in rats:
    (data_diff) = joblib.load(data_dir+rat_name+'_Y_BLAex_data_modulation_index_hc0_2.joblib')
    Data_diff=np.append(Data_diff,data_diff)

hc0data=Data_diff

Data_diff=[]
for rat_name in rats:
    (data_diff) = joblib.load(data_dir+rat_name+'_Y_BLAex_data_modulation_index_hc1_2.joblib')
    Data_diff = np.append(Data_diff, data_diff)
hc1data=Data_diff

Data_diff=[]
for rat_name in rats:
    (data_diff) = joblib.load(data_dir+rat_name+'_Y_BLAex_data_modulation_index_hc2_2.joblib')
    Data_diff = np.append(Data_diff, data_diff)
hc2data=Data_diff

Data_diff=[]
for rat_name in rats:
    (data_diff) = joblib.load(data_dir+rat_name+'_Y_BLAex_data_modulation_index_hc3_2.joblib')
    Data_diff = np.append(Data_diff, data_diff)
hc3data=Data_diff

Data_diff=[]
for rat_name in rats:
    (data_diff) = joblib.load(data_dir+rat_name+'_Y_BLAex_data_modulation_index_hc4_2.joblib')
    Data_diff = np.append(Data_diff, data_diff)
hc4data=Data_diff


joblib.dump((hc0data,hc1data,hc2data,hc3data,hc4data), os.path.expanduser('Y_BLAex_hc_data_2.joblib'), compress=3)


from scipy.stats import pearsonr
x=hc0data
y=hc1data
a, b = np.polyfit(x, y, 1)
Y2 = a * x + b
np.corrcoef(x, y)[0][1]
r, p1 = pearsonr(x, y)
print(r,p1)

x=hc0data
y=hc2data
a, b = np.polyfit(x, y, 1)
Y2 = a * x + b
np.corrcoef(x, y)[0][1]
r, p2 = pearsonr(x, y)
print(r,p2)

x=hc0data
y=hc3data
a, b = np.polyfit(x, y, 1)
Y2 = a * x + b
np.corrcoef(x, y)[0][1]
r, p3 = pearsonr(x, y)
print(r,p3)

x=hc0data
y=hc4data
a, b = np.polyfit(x, y, 1)
Y2 = a * x + b
np.corrcoef(x, y)[0][1]
r, p4 = pearsonr(x, y)
print(r,p4)

x=hc1data
y=hc2data
a, b = np.polyfit(x, y, 1)
Y2 = a * x + b
np.corrcoef(x, y)[0][1]
r, p5 = pearsonr(x, y)
print(r,p5)

x=hc1data
y=hc3data
a, b = np.polyfit(x, y, 1)
Y2 = a * x + b
np.corrcoef(x, y)[0][1]
r, p6 = pearsonr(x, y)
print(r,p6)

x=hc1data
y=hc4data
a, b = np.polyfit(x, y, 1)
Y2 = a * x + b
np.corrcoef(x, y)[0][1]
r, p7 = pearsonr(x, y)
print(r,p7)

x=hc2data
y=hc3data
a, b = np.polyfit(x, y, 1)
Y2 = a * x + b
np.corrcoef(x, y)[0][1]
r, p8 = pearsonr(x, y)
print(r,p8)

x=hc2data
y=hc4data
a, b = np.polyfit(x, y, 1)
Y2 = a * x + b
np.corrcoef(x, y)[0][1]
r, p9 = pearsonr(x, y)
print(r,p9)

x=hc3data
y=hc4data
a, b = np.polyfit(x, y, 1)
Y2 = a * x + b
np.corrcoef(x, y)[0][1]
r, p10 = pearsonr(x, y)
print(r,p10)

p1=(1,p1,p2,p3,p4,p1,1,p5,p6,p7,p2,p5,1,p8,p9,p3,p6,p8,1,p10,p4,p7,p9,p10,1)

joblib.dump((p1), os.path.expanduser('Y_fig_1_f_BLAex.joblib'), compress=3)