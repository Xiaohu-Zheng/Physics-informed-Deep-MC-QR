import os
import numpy as np
import scipy.io as sio
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

path1 = '/mnt/zhengxiaohu/PIRL/Cal_Probability/comp_temper003.mat'
data1 = sio.loadmat(path1)
heat_pre_up, heat_pre_lo = data1["T_up_sum"], data1["T_lo_sum"]
path2 = '/mnt/zhengxiaohu/PIRL/Cal_Probability/comp_temper003_mean.mat'
data2 = sio.loadmat(path2)
heat_pre_mean = data2["T_up_sum"]
com = 39
my_font = fm.FontProperties(fname="/mnt/zhengxiaohu/times/times.ttf")
fig = plt.figure(figsize=(3.9,2.5))
sns.kdeplot(heat_pre_up[:, com], label='Lower boundary', cumulative=True, 
            lw='1.2', color='green', linestyle='--')
sns.kdeplot(heat_pre_mean[:, com], label='Mean probability', cumulative=True, 
            lw='1.2', color='red', linestyle='-')
sns.kdeplot(heat_pre_lo[:, com], label='Upper boundary', cumulative=True, 
            lw='1.2', color='blue', linestyle='--')
plt.xlabel('Temperature', fontsize=12, fontproperties=my_font)
plt.ylabel('Probability', fontsize=12, fontproperties=my_font)
plt.xticks(fontproperties=my_font)
plt.yticks(fontproperties=my_font)
plt.legend(prop=my_font)
save_name = '/mnt/zhengxiaohu/PIRL/Cal_Probability/cdf/cdf.png'
fig.savefig(save_name, dpi=1000, bbox_inches = 'tight', pad_inches=0.02)
save_name = '/mnt/zhengxiaohu/PIRL/Cal_Probability/cdf/cdf_com{}.pdf'.format(com+1)
fig.savefig(save_name, dpi=1000, bbox_inches = 'tight', pad_inches=0.02)
