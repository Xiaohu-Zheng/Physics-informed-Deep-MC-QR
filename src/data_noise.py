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

    # print(hparams)
    # print()

    # Testing Set
    test_dataset = prepare_data(hparams)
    zeros = torch.zeros(hparams.input_size, hparams.input_size)
    for i, data in enumerate(test_dataset):
        print(i)
        u_obs, heat_true = data
        u_obs = u_obs.squeeze(0).squeeze(0)
        heat_true = heat_true.squeeze(0).squeeze(0)
        u_obs = u_obs * hparams.std_heat + hparams.mean_heat
        heat_true = heat_true * hparams.std_heat + hparams.mean_heat
        u_obs = torch.where(u_obs==hparams.mean_heat, zeros, u_obs)
        u_obs = u_obs.numpy()
        heat_true = heat_true.numpy()

        # data_dir = '/mnt/zhengxiaohu_data/datasetD2_top005_noise/train/train/'
        # data_dir = '/mnt/zhengxiaohu_data/datasetD2_top005_noise/test/test/'
        # data_dir = '/mnt/zhengxiaohu_data/dataset_sat_57_006_noise/test/test/'
        data_dir = '/mnt/zhengxiaohu_data/dataset_sat_57_center_003_noise/train/train/'
        file_name = f'Example{i}.mat'
        path = data_dir + file_name
        data = {"u": heat_true, "u_obs": u_obs}
        sio.savemat(path, data)
