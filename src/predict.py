"""
Runs a model on a single node across multiple gpus.
"""
import os
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

    heat_pre_sum = 0
    N_test = 0
    diff_T_test_pre_sum = 0
    ME = 0
    RMSE = 0
    MAE = 0
    MRE = 0
    R2 = 0
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
        T_diff = heat_true - heat_pre
        ME += torch.max(T_diff.abs().view(T_diff.size(0), -1), dim=1)[0].sum(dim=0)
        RMSE += (((T_diff ** 2 ).view(T_diff.size(0), -1).mean(dim=1)) ** 0.5).sum(dim=0)
        MAE += T_diff.abs().view(T_diff.size(0), -1).mean(dim=1).sum(dim=0)
        MRE += (T_diff.abs() / heat_true).view(T_diff.size(0), -1).mean(dim=1).sum(dim=0)

        T_mean = heat_pre.view(T_diff.size(0), -1).mean(dim=1)
        T_mean_mat = T_mean.view(T_mean.size(0),1,1).repeat(1, heat_true.size(1), heat_true.size(1))
        Rx = (T_diff ** 2).view(T_diff.size(0), -1).sum(dim=1)
        Ry = ((heat_true - T_mean_mat) ** 2).view(T_diff.size(0), -1).sum(dim=1)
        R2 += (1 - Rx / Ry ).sum(dim=0)
        
        # heat_pre_sum_inter = heat_pre.sum(dim=0)
        # diff_T_test_pre_sum += (T_diff ** 2).sum(dim=0)
        # if i ==0:
        #     T_test = heat_true
        # else:
        #     T_test = torch.cat((T_test, heat_true),dim=0)
        # heat_pre_sum += heat_pre_sum_inter
        N_test += u_obs.size(0)
    ME_mean = ME / N_test
    RMSE_mean = RMSE / N_test
    MAE_mean = MAE / N_test
    MRE_mean = MRE / N_test
    R2_mean = R2 / N_test
    # heat_pre_mean = heat_pre_sum / N_test
    # diff_T_test_mean = ((T_test - heat_pre_mean) ** 2).sum(dim=0)
    # R2_mean = (1 - diff_T_test_pre_sum / diff_T_test_mean).mean()

    print(f'ME_mean={ME_mean.item()}', '\n', f'RMSE_mean={RMSE_mean.item()}', '\n', 
          f'MAE_mean={MAE_mean.item()}', '\n', f'MRE_mean={MRE_mean.item()}' , '\n', 
          f'R2_mean={R2_mean}')
