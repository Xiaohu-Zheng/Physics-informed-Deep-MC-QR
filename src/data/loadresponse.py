# -*- encoding: utf-8 -*-
"""Load Response Dataset.
"""
import os
import scipy.io as sio
import numpy as np
import torch
from torchvision.datasets import VisionDataset


class LoadResponse(VisionDataset):
    """Some Information about LoadResponse dataset"""

    def __init__(
        self,
        root,
        loader,
        list_path,
        load_name="u_obs",
        resp_name="u",
        extensions=None,
        transform=None,
        target_transform=None,
        is_valid_file=None,
    ):
        super().__init__(
            root, transform=transform, target_transform=target_transform
        )
        self.list_path = list_path
        self.loader = loader
        self.load_name = load_name
        self.resp_name = resp_name
        self.extensions = extensions
        self.sample_files = make_dataset_list(root, list_path, extensions, is_valid_file)

    def __getitem__(self, index):
        path = self.sample_files[index]
        load, resp = self.loader(path, self.load_name, self.resp_name)

        if self.transform is not None:
            load = self.transform(load)
        if self.target_transform is not None:
            resp = self.target_transform(resp)
        zeros = torch.zeros_like(load)
        load = torch.where(load<0, zeros, load)

        # Add noise
        # ax1, ax2, ax3 = load.size()
        # noise = torch.from_numpy(np.random.normal(0, 0.003, (ax1, ax2, ax3)).astype(np.float32))
        # ones = torch.ones_like(load)
        # load_noise = torch.where(load==0, zeros, ones) * noise

        # noise_pos = [4, 6, 9]
        # row_clum_all = np.array([[3, 3], [4, 7], [3, 6], [5, 5], [3, 6], [3, 3], [3, 3], [3, 5], 
        #              [3, 9], [6, 6], [4, 4], [4, 4], [3, 5], [3, 3], [3, 6], [3, 6], 
        #              [5, 5], [4, 5], [4, 4]]) # dataD

        # row_clum_all = 3 * np.ones((19, 2),dtype=int)
        # row_clum_all[1] = np.array([[3, 4]])
        # row_clum_all[4] = np.array([[3, 4]])
        # row_clum_all[8] = np.array([[3, 6]])

        # size_all = np.array([[0.0081, 0.0084], [0.0182, 0.0082], [0.0166, 0.0057], [0.0111, 0.0111],
        #                  [0.0232, 0.0065], [0.005, 0.005], [0.0065, 0.0065], [0.0131, 0.0068], 
        #                  [0.0327, 0.005], [0.0112, 0.0112], [0.011, 0.0084], [0.0094, 0.0083], 
        #                  [0.0103, 0.0062], [0.005, 0.005], [0.0158, 0.0054], [0.0163, 0.005], 
        #                  [0.0085, 0.0085], [0.0085, 0.0072], [0.0069, 0.0069]]) * 2000
        # center_pos_all = np.array([[17.5, 183.35], [60.6, 18], [21, 150.5], [118.15, 107.05], 
        #                        [118.05, 55.9], [10.1, 11.7], [187.3, 186.3], [66.45, 175.8], 
        #                        [132.6, 184.65], [21.75, 61.3], [173.45, 100.4], [31.05, 117.05], 
        #                        [182.1, 13.45], [60.9, 56.1], [99.35, 143.2], [168.7, 150.8], 
        #                        [176.65, 58.8], [62.9, 89.4], [127, 14.65]])
        # row_clum = row_clum_all[noise_pos]
        # size = size_all[noise_pos]
        # center_pos = center_pos_all[noise_pos]
        # pos =torch.from_numpy(component_monitoring(row_clum, size, center_pos, side_steps=None, lenth=200))

        #data_D2
        # pos = torch.zeros_like(load)
        # for i in [50,56,62]:
        #     pos[:, i, [94,109,124,141]] = torch.ones_like(pos[:, i, [94,109,124,141]]) # C5
        # for j in [180,186,192]:
        #     pos[:, j, [180,186,193]] = torch.ones_like(pos[:, j, [180,186,193]]) # C7
        # for k in [51,62,72]:
        #     pos[:, k, [10,21,32]] = torch.ones_like(pos[:, k, [10,21,32]]) # C10
        
        # for j in [180,186,192]:
        #     pos[:, j, [180,186,193]] = torch.ones_like(pos[:, j, [180,186,193]]) # C7
        # for i in [52,57,61]:
        #     pos[:, i, [55,60,65]] = torch.ones_like(pos[:, i, [55,60,65]]) # C14
        # for k in [8,14,21]:
        #     pos[:, k, [120,126,133]] = torch.ones_like(pos[:, k, [120,126,133]]) # C19
        # load_noise_choose = torch.where(pos==1, load_noise, zeros)
        # load = load + load_noise_choose
        # resp = resp + load_noise_choose

        #####
        # load[:, 100:200, :] = load[:, 100:200, :] + load_noise[:, 100:200, :] # noise in top domain
        # load[:, 0:99, :] = load[:, 0:99, :] + load_noise[:, 0:99, :] # noise in buttom domain
        # load[:, :, 0:99] = load[:, :, 0:99] + load_noise[:, :, 0:99] # noise in left domain
        # load[:, :, 100:199] = load[:, :, 100:199] + load_noise[:, :, 100:199] # noise in right domain
        # load[:, 101:128, 91:118] = load[:, 101:128, 91:118] + load_noise[:, 101:128, 91:118] # noise in 9 center temperature measuring points

        # 所有测点加噪声
        # load = load + load_noise
        
        return load, resp

    def __len__(self):
        return len(self.sample_files)


def make_dataset(root_dir, extensions=None, is_valid_file=None):
    """make_dataset() from torchvision.
    """
    files = []
    root_dir = os.path.expanduser(root_dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError(
            "Both extensions and is_valid_file \
                cannot be None or not None at the same time"
        )
    if extensions is not None:
        is_valid_file = lambda x: has_allowed_extension(x, extensions)

    assert os.path.isdir(root_dir), root_dir
    for root, _, fns in sorted(os.walk(root_dir, followlinks=True)):
        for fn in sorted(fns):
            path = os.path.join(root, fn)
            if is_valid_file(path):
                files.append(path)
    return files


def make_dataset_list(root_dir, list_path, extensions=None, is_valid_file=None):
    """make_dataset() from torchvision.
    """
    files = []
    root_dir = os.path.expanduser(root_dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError(
            "Both extensions and is_valid_file \
                cannot be None or not None at the same time"
        )
    if extensions is not None:
        is_valid_file = lambda x: has_allowed_extension(x, extensions)

    assert os.path.isdir(root_dir), root_dir
    with open(list_path, 'r') as rf:
        for line in rf.readlines():
            data_path = line.strip()
            path = os.path.join(root_dir, data_path)
            if is_valid_file(path):
                files.append(path)
    return files


def has_allowed_extension(filename, extensions):
    return filename.lower().endswith(extensions)


def mat_loader(path, load_name, resp_name=None):
    mats = sio.loadmat(path)
    load = mats.get(load_name)
    resp = mats.get(resp_name) if resp_name is not None else None
    return load, resp

def component_monitoring(row_clum, size, center_pos, side_steps=None, lenth=200):
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

if __name__ == "__main__":
    total_num = 50000
    with open('train'+str(total_num)+'.txt', 'w') as wf:
        for idx in range(int(total_num*0.8)):
            wf.write('Example'+str(idx)+'.mat'+'\n')
    with open('val'+str(total_num)+'.txt', 'w') as wf:
        for idx in range(int(total_num*0.8), total_num):
            wf.write('Example'+str(idx)+'.mat'+'\n')