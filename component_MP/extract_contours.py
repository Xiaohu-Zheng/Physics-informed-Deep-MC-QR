from typing import List
import numpy as np
import cv2
import matplotlib.pyplot as plt
import yaml
import scipy.io as sio

import configargparse

parser = configargparse.ArgParser(
    config_file_parser_class=configargparse.YAMLConfigFileParser,
    default_config_files=['bak.yml'],
    description="Hyper-parameters.",
)
parser.add_argument('--positions', type=yaml.safe_load, action='append')
options = parser.parse_args()

def in_contour(center, contour):
    c_x, c_y = center
    x_max, x_min = contour[:,0].max(), contour[:,0].min()
    y_max, y_min = contour[:,1].max(), contour[:,1].min()

    if c_x >= x_min and c_x <= x_max and c_y >= y_min and c_y <= y_max:
        return True
    else:
        return False

def get_contours(positions: List[List[float]], img: np.ndarray) -> List[np.ndarray]:
    '''获取组件布局图的轮廓

    params:
        options: 组件中心的位置列表，每个元素为[x, y]
        img: 通道布局矩阵,[nx, nx]
    returns:
        ordered_contours: 与组件对应的轮廓列表，List[ndarray[N, 1, 2]]
    '''
    img = img / img.max()
    img = np.where(img>0, 1, 0)
    img = img.astype(np.uint8)

    # 返回值contours是一个list，其中每一个元素是一个np.ndarray的轮廓
    contours, _ = cv2.findContours(
        image=img,
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_NONE
    )

    ordered_contours = []
    for i, center in enumerate(positions):
        for j, contour in enumerate(contours):
            # contour = np.squeeze(contour)  # reshape [n, 2]
            if in_contour(center, contour[:,0,:]):
                ordered_contours.append(contour)

    return ordered_contours

########################################################################
# example

data = sio.loadmat('Example11996.mat')
img = data['F']
plt.imsave('original.png', img)

img = img / img.max()
img = np.where(img>0, 1, 0)
img = img.astype(np.uint8)

positions = options.positions
ordered_contours = get_contours(positions, img)

# 画图验证
output = np.ones((200, 200, 3), dtype=np.uint8)*255
output[:,:,:] = np.expand_dims(img, 2)

output = cv2.drawContours(
    output,
    ordered_contours,
    0,  # -1为画出所有轮廓
    (255, 0, 0),
)
print(positions[0])
print(ordered_contours[0])

print(len(ordered_contours))
plt.imsave('output.png', output)
