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

    Levels = 50
    noise = 'sat003'
    if not os.path.exists(f'./outputs/input-{noise}'):
        os.mkdir(f'./outputs/input-{noise}')
    if not os.path.exists(f'./outputs/truth-{noise}'):
        os.mkdir(f'./outputs/truth-{noise}')
    if not os.path.exists(f'./outputs/predict_plot-{noise}'):
        os.mkdir(f'./outputs/predict_plot-{noise}')
    if not os.path.exists(f'./outputs/error-{noise}'):
        os.mkdir(f'./outputs/error-{noise}')
    if not os.path.exists(f'./outputs/alea_u-{noise}'):
        os.mkdir(f'./outputs/alea_u-{noise}')
    with open(file_path, 'r') as fp:
        i = 0
        for line in fp.readlines():
            # Data Reading
            data_path = line.strip()
            path = os.path.join(root_dir, data_path)
            data = sio.loadmat(path)
            u_true, u_obs = data["u"], data["u_obs"]
            u_obs = torch.Tensor(u_obs)
            
            # Plot u_obs and Real Temperature Field
            fig = plt.figure(figsize=(4.945,4))
            # fig = plt.figure(figsize=(22.5,5))

            num = u_obs.size(0)
            grid_x = np.linspace(0, hparams.length, num)
            grid_y = np.linspace(0, hparams.length, num)
            X, Y = np.meshgrid(grid_x, grid_y)

            plt.subplot(111)
            # plt.title('Monitoring point temperature')
            im = plt.pcolormesh(X, Y, u_obs, cmap='jet', norm=colors.PowerNorm(gamma=10))
            cb = plt.colorbar(im)
            cb.set_ticks(np.array([0, 280, 300, 310, 320, 330, 340, 350]))
            cb.ax.tick_params(labelsize=10)
            cbarlabels = cb.ax.get_yticklabels() 
            [label.set_fontname('Times New Roman') for label in cbarlabels]
            plt.xticks(fontproperties=my_font, size=10)
            plt.yticks(fontproperties=my_font, size=10)
            plt.axis('equal')
            save_name = os.path.join(f'outputs/input-{noise}', os.path.splitext(os.path.basename(path))[0]+'_MPT.png')
            fig.savefig(save_name, dpi=1000, bbox_inches = 'tight', pad_inches=0.02)

            u_obs = ((u_obs - hparams.mean_layout) / hparams.std_layout).unsqueeze(0).unsqueeze(0)
            zeros = torch.zeros_like(u_obs)
            u_obs = torch.where(u_obs<0, zeros, u_obs)
            heat = torch.Tensor((u_true - hparams.mean_heat) / hparams.std_heat).unsqueeze(0).to(device)
            u_obs = u_obs.to(device)
            
            with torch.no_grad():
                heat_pre, std= MC_QR_prediction_for_regression(500, u_obs, model, tau='all')
                mae = F.l1_loss(heat, heat_pre) * hparams.std_heat
                print(f'MAE-{i}:', mae.item())
            mae_test.append(mae.item())
            heat_pre = heat_pre.squeeze(0).cpu().numpy() * hparams.std_heat + hparams.mean_heat
            std = std.squeeze(0).cpu().numpy()* hparams.std_heat
            
            data_dir = '/mnt/zhengxiaohu/PIRL/outputs/results/'
            file_name = f'Result{i}.mat'
            i += 1
            
            u_obs_0 = u_obs.squeeze(0).squeeze(0).cpu()
            u_obs_298 = u_obs_0 * hparams.std_heat + hparams.mean_heat
            u_obs = torch.where(u_obs_298==298, u_obs_0, u_obs_298).numpy()
            path = data_dir + file_name
            data = {"u_pre": heat_pre, 
                    "u_true": heat.squeeze(0).cpu().numpy() * hparams.std_heat + hparams.mean_heat, 
                    "u_obs": u_obs, 
                    "std": std}
            sio.savemat(path, data)

            fig = plt.figure(figsize=(4.945,4))
            plt.subplot(111)
            # plt.title('Real Temperature Field')
            im = plt.contourf(X, Y, u_true, levels=Levels, cmap='jet')
            # im = plt.pcolormesh(X, Y, u_true, cmap='jet')
            plt.xticks(fontproperties=my_font)
            plt.yticks(fontproperties=my_font)
            plt.axis('equal')
            cb = plt.colorbar(im)
            cb.ax.tick_params(labelsize=10)
            cbarlabels = cb.ax.get_yticklabels() 
            [label.set_fontname('Times New Roman') for label in cbarlabels]
            save_name = os.path.join(f'outputs/truth-{noise}', os.path.splitext(os.path.basename(path))[0]+'_T_real.png')
            fig.savefig(save_name, dpi=1000, bbox_inches = 'tight', pad_inches=0.02)

            fig = plt.figure(figsize=(4.945,4))
            plt.subplot(111)
            # plt.title('Reconstructed Temperature Field')
            im = plt.contourf(X, Y, heat_pre,levels=Levels,cmap='jet')
            plt.xticks(fontproperties=my_font)
            plt.yticks(fontproperties=my_font)
            plt.axis('equal')
            cb = plt.colorbar(im)
            cb.ax.tick_params(labelsize=10)
            cbarlabels = cb.ax.get_yticklabels() 
            [label.set_fontname('Times New Roman') for label in cbarlabels]
            save_name = os.path.join(f'outputs/predict_plot-{noise}', os.path.splitext(os.path.basename(path))[0]+'_T_pre.png')
            fig.savefig(save_name, dpi=1000, bbox_inches = 'tight', pad_inches=0.02)

            fig = plt.figure(figsize=(4.945,4))
            plt.subplot(111)
            # plt.title('Absolute Error')
            im = plt.contourf(X, Y, np.abs(heat_pre-u_true),levels=Levels,cmap='jet')
            plt.xticks(fontproperties=my_font)
            plt.yticks(fontproperties=my_font)
            plt.axis('equal')
            cb = plt.colorbar(im)
            cb.ax.tick_params(labelsize=10)
            cbarlabels = cb.ax.get_yticklabels() 
            [label.set_fontname('Times New Roman') for label in cbarlabels]
            save_name = os.path.join(f'outputs/error-{noise}', os.path.splitext(os.path.basename(path))[0]+'_error.png')
            fig.savefig(save_name, dpi=1000, bbox_inches = 'tight', pad_inches=0.02)

            fig = plt.figure(figsize=(4.945,4))
            plt.subplot(111)
            # plt.title('Aleatoric uncertainty')
            im = plt.contourf(X,Y,std,levels=200,cmap='jet')
            plt.xticks(fontproperties=my_font)
            plt.yticks(fontproperties=my_font)
            plt.axis('equal')
            cb = plt.colorbar(im)
            cb.ax.tick_params(labelsize=10)
            cbarlabels = cb.ax.get_yticklabels() 
            [label.set_fontname('Times New Roman') for label in cbarlabels]
            save_name = os.path.join(f'outputs/alea_u-{noise}', os.path.splitext(os.path.basename(path))[0]+'_alea.png')
            fig.savefig(save_name, dpi=1000, bbox_inches = 'tight', pad_inches=0.02)

    mae_test = np.array(mae_test)
    print(mae_test.mean())
    np.savetxt('outputs/mae_test.csv', mae_test, fmt='%f', delimiter=',')

