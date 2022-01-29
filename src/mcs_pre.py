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
import matplotlib.colors as colors
from matplotlib.font_manager import findfont, FontProperties
import configargparse
import matplotlib.font_manager as fm

from src.DeepRegression import Model
from src.mcqr.mcqr_regression import MC_QR_prediction_for_regression


def main(hparams):

    my_font = fm.FontProperties(fname="/mnt/zhengxiaohu/times/times.ttf")

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
    root = hparams.data_root
    test_list = hparams.test_list
    file_path = os.path.join(root, test_list)
    root_dir = os.path.join(root, 'test', 'test')

    if not os.path.exists(f'./outputs/mcs_pre_003'):
        os.mkdir(f'./outputs/mcs_pre_003')
    with open(file_path, 'r') as fp:
        i = 23000
        for line in fp.readlines():
            print(i)
            # Data Reading
            data_path = line.strip()
            path = os.path.join(root_dir, data_path)
            data = sio.loadmat(path)
            u_obs = data["u_obs"]
            u_obs = torch.Tensor(u_obs)

            u_obs = ((u_obs - hparams.mean_layout) / hparams.std_layout).unsqueeze(0).unsqueeze(0)
            zeros = torch.zeros_like(u_obs)
            u_obs = torch.where(u_obs<0, zeros, u_obs)
            u_obs = u_obs.to(device)
            
            with torch.no_grad():
                heat_pre, std= MC_QR_prediction_for_regression(500, u_obs, model, tau='all')
        
            heat_pre = heat_pre.squeeze(0).cpu().numpy() * hparams.std_heat + hparams.mean_heat
            std = std.squeeze(0).cpu().numpy()* hparams.std_heat
            
            data_dir = '/mnt/zhengxiaohu/PIRL/outputs/mcs_pre_003/'
            file_name = f'mcs_{i}.mat'
            i += 1
            
            u_obs_0 = u_obs.squeeze(0).squeeze(0).cpu()
            u_obs_298 = u_obs_0 * hparams.std_heat + hparams.mean_heat
            u_obs = torch.where(u_obs_298==298, u_obs_0, u_obs_298).numpy()
            path = data_dir + file_name
            data = {"u_pre": heat_pre, 
                    "std": std}
            sio.savemat(path, data)


