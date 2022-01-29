"""
Runs a model on a single node across multiple gpus.
"""
import os
import pickle
from pathlib import Path

import torch
import numpy as np
import torch.nn.functional as F
import scipy.io as sio
import matplotlib.pyplot as plt
import configargparse
from torch.utils.data import DataLoader

import sys
sys.path.append('/mnt/zhengxiaohu/PIRL')
from src.data.layout import LayoutDataset
import src.utils.np_transforms as transforms
from src.DeepRegression import Model
from src.mcqr.mcqr_regression import MC_QR_prediction_for_regression

def dataloader(hparams, dataset, shuffle=False):
    loader = DataLoader(
        dataset=dataset,
        shuffle=shuffle,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
    )
    return loader

def prepare_data(hparams):
    """Prepare dataset
    """
    size: int = hparams.input_size
    transform_layout = transforms.Compose(
        [
            transforms.Resize(size=(size, size)),
            transforms.ToTensor(),
            transforms.Normalize(
                torch.tensor([hparams.mean_layout]),
                torch.tensor([hparams.std_layout]),
            ),
        ]
    )
    transform_heat = transforms.Compose(
        [
            transforms.Resize(size=(size, size)),
            transforms.ToTensor(),
            transforms.Normalize(
                torch.tensor([hparams.mean_heat]),
                torch.tensor([hparams.std_heat]),
            ),
        ]
    )

    # here only support format "mat"
    assert hparams.data_format == "mat"
    test_dataset = LayoutDataset(
        hparams.data_root,
        list_path=hparams.test_list,
        train=False,
        transform=transform_layout,
        target_transform=transform_heat,
    )

    # assign to use in dataloaders
    test_dataset = dataloader(hparams, test_dataset, shuffle=False)
    
    return test_dataset


def probability_fun(x, threshold):
    y = torch.where(x < threshold)[0]
    p = y.size(0) / x.size(0)
    return p


def max_tempreture(root, test_list, Xs, Ys, lam):
    file_path = os.path.join(root, 'outputs', test_list)
    root_dir = os.path.join(root, 'outputs', 'mcs_pre_003')
    with open(file_path, 'r') as fp:
        N_mcs = 0
        for line in fp.readlines():
            print(N_mcs)
            
            # Data Reading
            data_path = line.strip()
            path = os.path.join(root_dir, data_path)
            data = sio.loadmat(path)
            std, heat_pre = data["std"], data["u_pre"]
            std = torch.Tensor(std)
            heat_pre = torch.Tensor(heat_pre)

            # 计算上限
            heat_pre_up = (heat_pre + lam * std).unsqueeze(0)
            heat_pre_lo = (heat_pre - lam * std).unsqueeze(0)

            T_up = torch.zeros(1, len(Xs))
            T_lo = torch.zeros(1, len(Xs))
            for j in range(len(Xs)):
                T_up[0, j] = torch.max(heat_pre_up[:, Xs[j], Ys[j]], dim=1)[0]
                T_lo[0, j] = torch.max(heat_pre_lo[:, Xs[j], Ys[j]], dim=1)[0]
            if N_mcs==0:
                T_up_sum = T_up
                T_lo_sum = T_lo
            else:
                T_up_sum = torch.cat((T_up_sum, T_up), dim=0)
                T_lo_sum = torch.cat((T_lo_sum, T_lo), dim=0)
            N_mcs += 1
    path = '/mnt/zhengxiaohu/PIRL/Cal_Probability/comp_temper003.mat'
    data = {"T_up_sum": T_up_sum.numpy(), "T_lo_sum": T_lo_sum.numpy()}
    sio.savemat(path, data)

                
def cal_com_prob(path, threshold):
    data = sio.loadmat(path)
    heat_pre_up, heat_pre_lo = data["T_up_sum"], data["T_lo_sum"]
    heat_pre_up = torch.Tensor(heat_pre_up)
    heat_pre_lo = torch.Tensor(heat_pre_lo)

    # 计算上限
    N_mcs = heat_pre_up.size(0)
    num_Cs_0_up = torch.zeros(57)
    num_Cs_0_lo = torch.zeros(57)
    for j in range(57):
        num_Cs_0_up[j] = torch.where(heat_pre_up[:, j] >= threshold[j])[0].size(0)
        num_Cs_0_lo[j] = torch.where(heat_pre_lo[:, j] >= threshold[j])[0].size(0)
    
    Pf_F_up = num_Cs_0_up / N_mcs
    Pf_F_lo = num_Cs_0_lo / N_mcs

    Pf_T_up = 1 - Pf_F_lo
    Pf_T_lo = 1- Pf_F_up
    # Pf_T_up = 1 - Pf_F_up
    # Pf_T_lo = 1 - Pf_F_lo

    Pf_F = torch.cat((Pf_F_lo.view(-1, 1), Pf_F_up.view(-1, 1)), dim=1)
    Pf_T = torch.cat((Pf_T_lo.view(-1, 1), Pf_T_up.view(-1, 1)), dim=1)
    path = '/mnt/zhengxiaohu/PIRL/Cal_Probability/comp_prob003.mat'
    data = {"Pf_F": Pf_F.numpy(), "Pf_T": Pf_T.numpy()}
    sio.savemat(path, data)

    return Pf_F, Pf_T

if __name__=="__main__":
    root = '/mnt/zhengxiaohu/PIRL/'
    test_list = 'mcs_pre_003/mcs_pre.txt'

    # loading component positions
    output1 = open('/mnt/zhengxiaohu/PIRL/component_MP/Xs', 'rb')
    Xs = pickle.load(output1)
    output2 = open('/mnt/zhengxiaohu/PIRL/component_MP/Ys', 'rb')
    Ys = pickle.load(output2)

    # conmponent threshold
    lam = 0 #0.25
    threshold = torch.ones(57)
    threshold[0] = 322.6
    threshold[1] = 322.9
    threshold[2] = 323.6
    threshold[3] = 321.05
    threshold[4] = 320.3
    threshold[5] = 318.1
    threshold[6] = 313.82
    threshold[7] = 308.65
    threshold[8] = 316.10
    threshold[9] = 318.3
    threshold[10] = 309.03
    threshold[11] = 315.5
    threshold[12] = 324.3
    threshold[13] = 325.4
    threshold[14] = 326.4
    threshold[15] = 322.3
    threshold[16] = 327.5
    threshold[17] = 330.0
    threshold[18] = 327.0
    threshold[19] = 328.0
    threshold[20] = 328.3
    threshold[21] = 325.8
    threshold[22] = 317.9
    threshold[23] = 321.0
    threshold[24] = 321.9
    threshold[25] = 326.8
    threshold[26] = 325.3
    threshold[27] = 325.5
    threshold[28] = 323.5
    threshold[29] = 328.6
    threshold[30] = 330.1
    threshold[31] = 330.8
    threshold[32] = 331.1
    threshold[33] = 330.7
    threshold[34] = 331.0
    threshold[35] = 330.3
    threshold[36] = 330.6
    threshold[37] = 321.0
    threshold[38] = 328.6
    threshold[39] = 328.4
    threshold[40] = 329.2
    threshold[41] = 329.0
    threshold[42] = 328.9
    threshold[43] = 329.6
    threshold[44] = 329.2
    threshold[45] = 329.0
    threshold[46] = 327.3
    threshold[47] = 329.0
    threshold[48] = 327.5
    threshold[49] = 330.2
    threshold[50] = 330.0
    threshold[51] = 329.9
    threshold[52] = 329.7
    threshold[53] = 327.4
    threshold[54] = 326.0
    threshold[55] = 328.7
    threshold[56] = 324.0

    # max_tempreture(root, test_list, Xs, Ys, lam)

    path = '/mnt/zhengxiaohu/PIRL/Cal_Probability/comp_temper003.mat'
    Pf_F, Pf_T = cal_com_prob(path, threshold)
    print('Pf_T:', '\n', Pf_T)