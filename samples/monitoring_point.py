import yaml
from pathlib import Path
import numpy as np
import scipy.io as sio
from torch._C import dtype


def component_monitoring_from_mat(path):
    path = Path(path)
    data = sio.loadmat(path)
    u_pos = data['u_obs']
    zeros = np.zeros_like(u_pos)
    ones = np.ones_like(u_pos)
    u_pos = np.where(u_pos==298, zeros, ones)
    return u_pos

def component_monitoring(row_clum, size, center_pos, side_steps=20, lenth=200):
    u_pos = np.zeros((lenth, lenth))
    for i in range(len(center_pos)):
        rows = row_clum[i, 0]
        colums = row_clum[i, 1]
        com_size = size[i]
        com_cenetr = center_pos[i]
        ax1_min= int(com_cenetr[1] - com_size[1] / 2) + 1
        ax1_max= int(com_cenetr[1] + com_size[1] / 2)
        ax2_min= int(com_cenetr[0] - com_size[0] / 2)
        ax2_max= int(com_cenetr[0] + com_size[0] / 2)
        stride_ax1 = int(com_size[1] / (rows - 1))
        stride_ax2 = int(com_size[0] / (colums - 1))
        for j in range(rows):
            if j == rows - 1:
                ax1 = ax1_max
            else:
                ax1 = int(ax1_min + j * stride_ax1)

            for k in range(colums):
                if k == colums - 1:
                    ax2 = ax2_max
                else:
                    ax2 = int(ax2_min + k * stride_ax2)
                u_pos[ax1, ax2]  = 1
    if side_steps is not None:
        ax_max = len(u_pos)-1
        ax = list(range(0, len(u_pos), side_steps)) + [ax_max]
        u_pos[0, ax] = 1
        u_pos[ax, 0] = 1
        u_pos[ax_max, ax] = 1
        u_pos[ax, ax_max] = 1
    return u_pos


# path = '/mnt/zhengxiaohu_data/dataset_sat/train/train/Example0.mat'
# u_pos = component_monitoring_from_mat(path)
# save_path = '/mnt/zhengxiaohu/PIRL/samples/u_pos_A.mat'
# data = {"u_pos": u_pos}
# sio.savemat(save_path, data)

# # row_clum = np.array([[3, 3], [5, 4], [4, 4], [6, 6], [5, 5], [3, 5], [7, 4], [6, 3], [6, 4], [4, 5]])
# # size = np.array([[0.012, 0.012], [0.016, 0.03], [0.015, 0.015], [0.03, 0.03], [0.02, 0.02],
# #                  [0.03, 0.015], [0.02, 0.04], [0.015, 0.03], [0.02, 0.03], [0.03, 0.02]]) * 2000
# # center_pos = np.array([[25, 183], [175, 142], [100, 29], [160, 45], [135, 177], 
# #                        [72, 67], [42, 126], [85, 159], [129, 118], [40, 28]])

# # row_clum = np.array([[3, 3], [4, 7], [3, 6], [5, 5], [3, 6], [3, 3], [3, 3], [3, 5], 
# #                      [3, 9], [6, 6], [4, 4], [4, 4], [3, 5], [3, 3], [3, 6], [3, 6], 
# #                      [5, 5], [4, 5], [4, 4]])
# row_clum = 3 * np.ones((19, 2),dtype=int)
# row_clum[1] = np.array([[3, 4]])
# row_clum[4] = np.array([[3, 4]])
# row_clum[8] = np.array([[3, 6]])
# size = np.array([[0.0081, 0.0084], [0.0182, 0.0082], [0.0166, 0.0057], [0.0111, 0.0111], [0.0232, 0.0065], [0.005, 0.005], [0.0065, 0.0065], [0.0131, 0.0068], 
#                  [0.0327, 0.005], [0.0112, 0.0112], [0.011, 0.0084], [0.0094, 0.0083], [0.0103, 0.0062], [0.005, 0.005], [0.0158, 0.0054], [0.0163, 0.005], 
#                  [0.0085, 0.0085], [0.0085, 0.0072], [0.0069, 0.0069]]) * 2000
# center_pos = np.array([[17.5, 183.35], [60.6, 18], [21, 150.5], [118.15, 107.05], [118.05, 55.9], [10.1, 11.7], [187.3, 186.3], [66.45, 175.8], 
#                      [132.6, 184.65], [21.75, 61.3], [173.45, 100.4], [31.05, 117.05], [182.1, 13.45], [60.9, 56.1], [99.35, 143.2], [168.7, 150.8], 
#                      [176.65, 58.8], [62.9, 89.4], [127, 14.65]])

# u_pos = component_monitoring(row_clum, size, center_pos, side_steps=20, lenth=200)
# path = '/mnt/zhengxiaohu/PIRL/samples/u_pos_D.mat'
# data = {"u_pos": u_pos}
# sio.savemat(path, data)

stream = open('/mnt/zhengxiaohu_data/yml/config_big_sat_57.yml', mode='r', encoding='utf-8')
data = yaml.load(stream)
size0 = np.array(data['units'])
size = np.array(data['units']) * 170
size[3] = size0[3] *130
size[4] = size0[4] *130
size[5] = size0[5] *130
size[6] = size0[6] *130
size[7] = size0[7] *130
size[8] = size0[8] *100
size[9] = size0[9] *100
size[10] = size0[10] *100
size[11] = size0[11] *100
size[17] = size0[17] *120
size[19] = size0[19] *130
size[23] = size0[23] *130
size[24] = size0[24] *130
size[25] = size0[25] *185
size[26] = size0[26] *185
size[27] = size0[27] *190
size[28] = size0[28] *185
size[31] = size0[31] *120
size[32] = size0[32] *120
size[33] = size0[33] *120
size[34] = size0[34] *120
size[35] = size0[35] *120
size[36] = size0[36] *120
size[37] = size0[37] *180
size[38] = size0[38] *125
size[41] = size0[41] *100
size[42] = size0[42] *100
size[43] = size0[43] *100
size[49] = size0[49] *115
center_pos = np.array(data['positions'])
row_clum = 2 * np.ones((54, 2),dtype=int)
row_clum[25] = np.array([[2, 6]])
row_clum[26] = np.array([[2, 6]])
row_clum[0] = np.array([[2, 4]])
row_clum[1] = np.array([[2, 4]])
row_clum[2] = np.array([[2, 4]])
row_clum[12] = np.array([[2, 4]])
row_clum[13] = np.array([[2, 4]])
row_clum[14] = np.array([[2, 4]])
row_clum[15] = np.array([[2, 3]])
row_clum[16] = np.array([[2, 3]])
row_clum[18] = np.array([[2, 3]])
row_clum[20] = np.array([[2, 3]])
row_clum[22] = np.array([[2, 3]])
row_clum[27] = np.array([[4, 4]])
row_clum[28] = np.array([[6, 2]])
row_clum[30] = np.array([[2, 4]])
row_clum[37] = np.array([[3, 4]])
row_clum[39] = np.array([[3, 2]])
row_clum[40] = np.array([[2, 3]])
row_clum[44] = np.array([[2, 3]])
row_clum[45] = np.array([[2, 3]])
row_clum[46] = np.array([[2, 3]])
row_clum[47] = np.array([[2, 3]])
row_clum[48] = np.array([[2, 3]])
row_clum[50] = np.array([[4, 2]])
row_clum[51] = np.array([[4, 2]])
row_clum[52] = np.array([[4, 2]])
row_clum[53] = np.array([[4, 2]])
u_pos = component_monitoring(row_clum, size, center_pos, side_steps=80, lenth=400)
path = '/mnt/zhengxiaohu/PIRL/samples/u_pos_sat_big_57.mat'
data = {"u_pos": u_pos}
sio.savemat(path, data)