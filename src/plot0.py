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

    with open(file_path, 'r') as fp:
        i = 0
        for line in fp.readlines():
            # Data Reading
            data_path = line.strip()
            path = os.path.join(root_dir, data_path)
            data = sio.loadmat(path)
            u_true, u_obs = data["u"], data["u_obs"]
            u_obs = torch.Tensor(u_obs)
            # hs_F = data["F"]
            
            # Plot u_obs and Real Temperature Field
            fig = plt.figure(figsize=(28,5))
            # fig = plt.figure(figsize=(22.5,5))

            num = u_obs.size(0)
            grid_x = np.linspace(0, hparams.length, num)
            grid_y = np.linspace(0, hparams.length, num)
            X, Y = np.meshgrid(grid_x, grid_y)

            plt.subplot(151)
            plt.title('Real Time Power')
            im = plt.pcolormesh(X, Y, u_obs, cmap='jet')
            # im = plt.pcolormesh(X,Y,hs_F)
            plt.colorbar(im)
            plt.xticks(fontproperties=my_font)
            plt.yticks(fontproperties=my_font)
            fig.tight_layout(pad=2.0, w_pad=3.0,h_pad=2.0)

            u_obs = ((u_obs - hparams.mean_layout) / hparams.std_layout).unsqueeze(0).unsqueeze(0)
            zeros = torch.zeros_like(u_obs)
            u_obs = torch.where(u_obs<0, zeros, u_obs)
            heat = torch.Tensor((u_true - hparams.mean_heat) / hparams.std_heat).unsqueeze(0).to(device)

            # 测试数据添加噪声
            # ax1, ax2, ax3, ax4 = u_obs.size()
            # noise = torch.from_numpy(np.random.normal(0, 0.000001, (ax1, ax2, ax3, ax4)).astype(np.float32))
            # ones = torch.ones_like(u_obs)
            # u_obs_noise = torch.where(u_obs==0, zeros, ones) * noise

            # datasetD
            # pos = torch.zeros_like(u_obs)
            # for i in [50,56,62]:
            #     pos[:, :, i, [94,109,124,141]] = torch.ones_like(pos[:, :, i, [94,109,124,141]]) # C5
            # for j in [180,186,192]:
            #     pos[:, :, j, [180,186,193]] = torch.ones_like(pos[:, :, j, [180,186,193]]) # C7
            # for k in [51,62,72]:
            #     pos[:, :, k, [10,21,32]] = torch.ones_like(pos[:, :, k, [10,21,32]]) # C10
            # u_obs_noise = torch.where(pos==1, u_obs_noise, zeros)
            # u_obs = (u_obs + u_obs_noise).to(device)

            # dataset_set
            # u_obs = (u_obs + u_obs_noise).to(device)
            u_obs = u_obs.to(device)
            
            with torch.no_grad():
                #u_obs=torch.flip(u_obs,dims=[2,3])
                #heat = torch.flip(heat,dims=[2,3])
                # heat_pre = model(heat_obs_taus)
                heat_pre, std= MC_QR_prediction_for_regression(100, u_obs, model, tau='all')

                mae = F.l1_loss(heat, heat_pre) * hparams.std_heat
                print('MAE:', mae)
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

            hmax = max(np.max(heat_pre), np.max(u_true))
            hmin = min(np.min(heat_pre), np.min(u_true))

            plt.subplot(152)
            plt.title('Real Temperature Field')
            if "xs" and "ys" in data.keys():
                xs, ys = data["xs"], data["ys"]
                im = plt.pcolormesh(ys, xs, u_true,cmap='jet')
                plt.axis('equal')
            else:
                #im = plt.pcolormesh(X, Y, heat.squeeze(0).squeeze(0).cpu().numpy() * hparams.std_heat + hparams.mean_heat)
                #im = plt.pcolormesh(X, Y, u_true,cmap='jet')
                im = plt.contourf(X,Y,u_true,levels=150,cmap='jet')
                plt.xticks(fontproperties=my_font)
                plt.yticks(fontproperties=my_font)
            plt.colorbar(im)

            plt.subplot(153)
            plt.title('Reconstructed Temperature Field')
            if "xs" and "ys" in data.keys():
                xs, ys = data["xs"], data["ys"]
                im = plt.pcolormesh(ys, xs, heat_pre,cmap='jet')
                # plt.axis('equal')
            else:
                im = plt.contourf(X, Y, heat_pre,levels=150,cmap='jet')
                plt.xticks(fontproperties=my_font)
                plt.yticks(fontproperties=my_font)
            plt.colorbar(im)

            plt.subplot(154)
            plt.title('Absolute Error')
            if "xs" and "ys" in data.keys():
                xs, ys = data["xs"], data["ys"]
                #im = plt.pcolormesh(xs, ys, np.abs(heat_pre-u_true)/u_true)
                im = plt.pcolormesh(ys, xs, np.abs(heat_pre-u_true),cmap='jet')
                # plt.axis('equal')
            else:
                im = plt.contourf(X, Y, np.abs(heat_pre-u_true),levels=150,cmap='jet')
                plt.xticks(fontproperties=my_font)
                plt.yticks(fontproperties=my_font)
                #im = plt.pcolormesh(X, Y, np.abs(heat_pre-u_true)/u_true)
            plt.colorbar(im)


            plt.subplot(155)
            plt.title('Aleatoric uncertainty')
            if "xs" and "ys" in data.keys():
                xs, ys = data["xs"], data["ys"]
                im = plt.pcolormesh(ys, xs, std,cmap='jet')
                plt.axis('equal')
            else:
                #im = plt.pcolormesh(X, Y, heat.squeeze(0).squeeze(0).cpu().numpy() * hparams.std_heat + hparams.mean_heat)
                #im = plt.pcolormesh(X, Y, u_true,cmap='jet')
                im = plt.contourf(X,Y,std,levels=150,cmap='jet')
                plt.xticks(fontproperties=my_font)
                plt.yticks(fontproperties=my_font)
            plt.colorbar(im)

            save_name = os.path.join('outputs/predict_plot', os.path.splitext(os.path.basename(path))[0]+'.png')
            fig.savefig(save_name, dpi=300)
            plt.close()

    mae_test = np.array(mae_test)
    print(mae_test.mean())
    np.savetxt('outputs/mae_test.csv', mae_test, fmt='%f', delimiter=',')

