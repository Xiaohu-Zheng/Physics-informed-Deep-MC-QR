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


def main(hparams):

    if hparams.gpu == 0:
        device = torch.device("cpu")
    else:
        ngpu = "cuda:"+str(hparams.gpu-1)
        print(ngpu)
        device = torch.device(ngpu)
    model = Model(hparams).to(device)

    # print(hparams)
    # print()

    # loading component positions
    output1 = open('/mnt/zhengxiaohu/PIRL/component_MP/Xs', 'rb')
    Xs = pickle.load(output1)
    output2 = open('/mnt/zhengxiaohu/PIRL/component_MP/Ys', 'rb')
    Ys = pickle.load(output2)

    # conmponent threshold
    lam = 1
    threshold = 328 * torch.ones(57)

    # Model loading
    model_path = os.path.join(f'lightning_logs/version_' +
                              hparams.test_check_num, 'checkpoints/')
    ckpt = list(Path(model_path).glob("*.ckpt"))[0]
    # print(ckpt)

    model = model.load_from_checkpoint(str(ckpt))

    model.eval()
    model.to(device)
    mae_test = []

    # Testing Set
    test_dataset = prepare_data(hparams)

    N_mcs = 0
    num_Cs_0_up = torch.zeros(len(Xs))
    num_Cs_0_lo = torch.zeros(len(Xs))
    for i, data in enumerate(test_dataset):
        print(i)
        u_obs, heat_true = data
        heat_true = heat_true.squeeze(1)
        heat_true = heat_true * hparams.std_heat + hparams.mean_heat
        zeros = torch.zeros_like(u_obs)
        u_obs = torch.where(u_obs<0, zeros, u_obs).to(device)

        with torch.no_grad():
            heat_pre, std= MC_QR_prediction_for_regression(100, u_obs, model, tau='all')
        heat_pre = heat_pre.squeeze(0).cpu() * hparams.std_heat + hparams.mean_heat
        std = std.cpu() * hparams.std_heat
        # 计算上限
        heat_pre_up = heat_pre + lam * std
        heat_pre_lo = heat_pre - lam * std
        for j in range(len(Xs)):
            com_T_up_j = torch.max(heat_pre_up[:, Xs[j], Ys[j]], dim=1)[0]
            num_Cs_0_up_j = torch.where(com_T_up_j >= threshold[j])[0].size(0)
            num_Cs_0_up[j] += num_Cs_0_up_j

            com_T_lo_j = torch.max(heat_pre_lo[:, Xs[j], Ys[j]], dim=1)[0]
            num_Cs_0_lo_j = torch.where(com_T_lo_j >= threshold[j])[0].size(0)
            num_Cs_0_lo[j] += num_Cs_0_lo_j
        N_mcs += std.size(0)
    
    Pf_F_up = num_Cs_0_up / N_mcs
    Pf_F_lo = num_Cs_0_lo / N_mcs

    Pf_T_up = 1 - Pf_F_lo
    Pf_T_lo = 1- Pf_F_up
    print(f'Pf_F_upper={Pf_F_up}', '\n', f'Pf_F_lower={Pf_F_lo}', '\n',
          f'Pf_T_upper={Pf_T_up}', '\n', f'Pf_T_lower={Pf_T_lo}')
