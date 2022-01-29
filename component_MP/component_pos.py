
import torch
import pickle
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

root_path = '/mnt/zhengxiaohu/PIRL/component_MP'
data = sio.loadmat('/mnt/zhengxiaohu/PIRL/component_MP/Example_serial_number.mat')
layout = torch.from_numpy(data['F']).floor()
layout_zeros = torch.zeros_like(layout)

fig = plt.figure(figsize=(5,5))
num = layout.size(0)
grid_x = np.linspace(0, 1.6, num)
grid_y = np.linspace(0, 1.6, num)
X, Y = np.meshgrid(grid_x, grid_y)
im = plt.pcolormesh(X,Y,layout)
plt.colorbar(im)
fig.tight_layout(pad=2.0, w_pad=3.0,h_pad=2.0)
plt.savefig('/mnt/zhengxiaohu/PIRL/component_MP/layout.png')

superset = torch.cat([layout, layout_zeros])
uniset, count = superset.unique(return_counts=True)
mask = (count != 0)
result = uniset.masked_select(mask)
power = result[1:]
Xs = []
Ys = []
for i in range(len(power)):
    xs, ys = torch.where(layout==power[i])
    xs = xs.numpy().tolist()
    ys = ys.numpy().tolist()
    Xs += [xs]
    Ys += [ys]
output1 = open(root_path+'/Xs', 'wb')
pickle.dump(Xs, output1) 
output2 = open(root_path+'/Ys', 'wb')
pickle.dump(Ys, output2) 

# 读取
output1 = open('/mnt/zhengxiaohu/PIRL/component_MP/Xs', 'rb')
Xs = pickle.load(output1)
output2 = open('/mnt/zhengxiaohu/PIRL/component_MP/Ys', 'rb')
Ys = pickle.load(output2)

print(len(Xs))
print(len(Ys))
T_com = layout[Xs[0], Ys[0]]
print(T_com)
