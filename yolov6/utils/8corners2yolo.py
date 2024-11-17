import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
# plt.ion()
import time
import xml.etree.ElementTree as ET

name_list = ['harbor', 'helicopter', 'large-vehicle', 'plane', 'ship', 'small-vehicle']

image_path = r'OHD-SJTU-L/test/output_crop/images'
txt_path = r'OHD-SJTU-L/test/output_crop/labelxml'
output_path = r'OHD-SJTU-L/test/output_crop/labels'

def rbox2box(rboxs):
    """
    Trans rbox format to box format
    rbox 的顺序为左上 右上 右下 左下
    The order of the rbox is top left, top right, bottom right, bottom left.
    Args:
        rboxes (array/tensor): (num_gts, rbox)

    Returns:
        hbboxes (array/tensor): (num_gts, [xc yc w h angle])
    """
    assert rboxs.shape[-1] == 8
    x = rboxs[0::2] # (num, 4)
    y = rboxs[1::2]
    x_max = np.amax(x) # (num)
    x_min = np.amin(x)
    y_max = np.amax(y)
    y_min = np.amin(y)
    x_ctr, y_ctr = (x_max + x_min) / 2.0, (y_max + y_min) / 2.0 # (num)
    w = np.sqrt(np.power((x[1] - x[0]), 2) + np.power((y[1] - y[0]), 2))  # (num)
    h = np.sqrt(np.power((x[2] - x[1]), 2) + np.power((y[2] - y[1]), 2))
    vx = (x[0] + x[1]) / 2.0 - x_ctr
    vy = (y[0] + y[1]) / 2.0 - y_ctr
    v0x = np.ones_like(vx, dtype=vx.dtype) * 0.
    v0y = np.ones_like(vy, dtype=vy.dtype) * -1.
    angle_cw = 360.0 - (np.rad2deg(np.arctan2(vx * v0y - vy * v0x, v0x * vx + v0y * vy)) + 360.0) % 360.0
    x_ctr, y_ctr, w, h, angle = x_ctr.reshape(-1, 1), y_ctr.reshape(-1, 1), w.reshape(-1, 1), h.reshape(-1, 1), angle_cw.reshape(-1, 1)  # (num, 1)
    boxes = np.concatenate((x_ctr, y_ctr, w, h, angle), axis=1)
    return boxes


for t in os.listdir(txt_path):
    # file site
    fr = open(os.path.join(txt_path, t), 'r')
    imfile = os.path.join(image_path, t.split('.')[0] + '.jpg')

    if not os.path.isfile(imfile):
        continue
    print(imfile)
    im = cv2.imread(imfile)
    w = im.shape[0]
    h = im.shape[1]
    # open write txt
    with open(os.path.join(output_path, t.replace('xml','txt')), 'w', encoding='utf-8') as f:
        tree = ET.parse(os.path.join(txt_path, t))
        root = tree.getroot()
        for obj in root.findall('.//object'):
            bndbox = obj.find('bndbox')
            name = obj.find('name').text
            x0 = np.array(bndbox.find('x0').text, dtype=np.float32)
            y0 = np.array(bndbox.find('y0').text, dtype=np.float32)
            x1 = np.array(bndbox.find('x1').text, dtype=np.float32)
            y1 = np.array(bndbox.find('y1').text, dtype=np.float32)
            x2 = np.array(bndbox.find('x2').text, dtype=np.float32)
            y2 = np.array(bndbox.find('y2').text, dtype=np.float32)
            x3 = np.array(bndbox.find('x3').text, dtype=np.float32)
            y3 = np.array(bndbox.find('y3').text, dtype=np.float32)
            # x_head = np.array(bndbox.find('x_head').text, dtype=np.float32)
            # y_head = np.array(bndbox.find('y_head').text, dtype=np.float32)
            if x0>w or y0>h or x1>w or y1>h or x2>w or y2>h or x3>w or y3>h:
                continue
            if x0 < 0. or y0 < 0. or x1 < 0. or y1 < 0. or x2 < 0. or y2 < 0. or x3 < 0. or y3 < 0.:
                continue
            x,y,hh,ww,ang = rbox2box(np.array(
                [x0 / w, y0 / h, x1 / w, y1 / h, x2 / w, y2 / h,
                 x3 / w, y3 / h])).flatten()
            # print(ang)
            # north = np.array([0, 0])
            # north[1] = -1
            # v0x, v0y = north[0], north[1]
            # vx, vy = x_head - x, y_head - y
            # # v = torch.cat((labels[:, 0].unsqueeze(1), labels[:, 1].unsqueeze(1) - 1.), 1)
            # ang = 360.0 - (
            #         np.rad2deg(np.arctan2(vx * v0y - vy * v0x, v0x * vx + v0y * vy)) + 360.0) % 360 - 0.001
            # print(ang)
            #
            # x = x / weight
            # y = y / height
            # w = w / height
            # h = h / weight

            # f.writelines(str(name_list.index(data_list[10])) + ' ' + xywht + '\n')
            f.writelines(
                str(name_list.index(name)) + " " + " ".join([str(i) for i in [x, y, hh, ww, ang]]) + '\n')
            pass

        pass
    fr.close()
pass

