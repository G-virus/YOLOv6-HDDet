# -*- coding: utf-8 -*- 
# @Time : 2022/12/15 09:48 
# @Author : DDD
"""
some tools for rotation box
"""
import torch
import numpy as np
import cv2
from PIL import Image
deg2rad = np.pi / 180

def pil2cv2(image_pil):
    return cv2.cvtColor(np.asarray(image_pil), cv2.COLOR_RGB2BGR)

def cv22pil(img_bgr):
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def box2rbox(predn): # [n,7] angle 0-360
    """
    x y w h angle -> x1y1(左上) x2y2(右上) x3y3(右下) x4y4(左下)
    :param predn:  预测值 [预测框数量 , 5 (x y w h angle) ]
    :return: 旋转框的四个点 x1y1 x2y2 x3y3 x4y4
    旋转角度是垂直向上是0度,顺时针旋转360度
    """
    if isinstance(predn, np.ndarray):
        x, y, w, h, angle = predn[:, 0], predn[:, 1], predn[:, 2], predn[:, 3], predn[:, 4]
        cos = np.cos(2 * np.pi - angle * deg2rad)
        sin = np.sin(2 * np.pi - angle * deg2rad)

        x1, x2, x3, x4 = - w / 2, + w / 2, - w / 2, + w / 2
        y1, y2, y3, y4 = + h / 2, + h / 2, - h / 2, - h / 2
        x1y1 = np.stack((x1, y1), axis=0)
        x2y2 = np.stack((x2, y2), axis=0)
        x3y3 = np.stack((x3, y3), axis=0)
        x4y4 = np.stack((x4, y4), axis=0)

        x1_out = cos * x1y1[0, :] + sin * x1y1[1, :] + x
        y1_out = -sin * x1y1[0, :] + cos * x1y1[1, :] + y

        x2_out = cos * x2y2[0, :] + sin * x2y2[1, :] + x
        y2_out = -sin * x2y2[0, :] + cos * x2y2[1, :] + y

        x3_out = cos * x3y3[0, :] + sin * x3y3[1, :] + x
        y3_out = -sin * x3y3[0, :] + cos * x3y3[1, :] + y

        x4_out = cos * x4y4[0, :] + sin * x4y4[1, :] + x
        y4_out = -sin * x4y4[0, :] + cos * x4y4[1, :] + y

        return np.concatenate((x3_out.reshape(-1, 1), y3_out.reshape(-1, 1),
                               x4_out.reshape(-1, 1), y4_out.reshape(-1, 1),
                               x2_out.reshape(-1, 1), y2_out.reshape(-1, 1),
                               x1_out.reshape(-1, 1), y1_out.reshape(-1, 1)), axis=1)

    elif isinstance(predn, torch.Tensor):
        x, y, w, h, angle = predn[:, 0], predn[:, 1], predn[:, 2], predn[:, 3], predn[:, 4]
        cos = torch.cos(2 * np.pi - angle * deg2rad)
        sin = torch.sin(2 * np.pi - angle * deg2rad)
        # 每个坐标点减去中心点 就是 向量 . 向量的横纵坐标
        x1, x2, x3, x4 = - w / 2, + w / 2, - w / 2, + w / 2
        y1, y2, y3, y4 = + h / 2, + h / 2, - h / 2, - h / 2
        # x1y1.shape=[2,n]
        x1y1 = torch.stack((x1, y1), dim=0)
        x2y2 = torch.stack((x2, y2), dim=0)
        x3y3 = torch.stack((x3, y3), dim=0)
        x4y4 = torch.stack((x4, y4), dim=0)
        # 左乘旋转矩阵
        x1_out = cos * x1y1[0, :] + sin * x1y1[1, :] + x
        y1_out = -sin * x1y1[0, :] + cos * x1y1[1, :] + y

        x2_out = cos * x2y2[0, :] + sin * x2y2[1, :] + x
        y2_out = -sin * x2y2[0, :] + cos * x2y2[1, :] + y

        x3_out = cos * x3y3[0, :] + sin * x3y3[1, :] + x
        y3_out = -sin * x3y3[0, :] + cos * x3y3[1, :] + y

        x4_out = cos * x4y4[0, :] + sin * x4y4[1, :] + x
        y4_out = -sin * x4y4[0, :] + cos * x4y4[1, :] + y

        # 返回也应该是[n,8]  这么操作可以使得 k[:,0]==x1_out
        # 顺序可能改成 左上角 右上角 右下角 左下角
        return torch.cat((x3_out, y3_out, x4_out, y4_out, x2_out, y2_out, x1_out, y1_out)).reshape(8, x.shape[0]).T
    else:
        print('box2rbox error')


def box2rbox_numpy(x, y, w, h, angle): # [n,7]
    """
    x y w h angle -> x1y1(左上) x2y2(右上) x3y3(右下) x4y4(左下)
    :param predn:  预测值 [预测框数量 , 5 (x y w h angle) ]
    :return: 旋转框的四个点 x1y1 x2y2 x3y3 x4y4
    旋转角度是垂直向上是0度,顺时针旋转360度
    """

    cos = np.cos(2*np.pi - angle * deg2rad)
    sin = np.sin(2*np.pi - angle * deg2rad)
    # 每个坐标点减去中心点 就是 向量 . 向量的横纵坐标
    x1, x2, x3, x4 = - w/2, + w/2, - w/2, + w/2
    y1, y2, y3, y4 = + h/2, + h/2, - h/2, - h/2
    # 左乘旋转矩阵
    x1_out = cos * x1 + sin * y1 + x
    y1_out = -sin * x1 + cos * y1 + y

    x2_out = cos * x2 + sin * y2 + x
    y2_out = -sin * x2 + cos * y2 + y

    x3_out = cos * x3 + sin * y3 + x
    y3_out = -sin * x3 + cos * y3 + y

    x4_out = cos * x4 + sin * y4 + x
    y4_out = -sin * x4 + cos * y4 + y

    # 返回也应该是[n,8]  这么操作可以使得 k[:,0]==x1_out
    # 顺序为 左上角 右上角 右下角 左下角
    return np.array([[x3_out, y3_out],[x4_out, y4_out], [x2_out, y2_out] ,[x1_out, y1_out]])

def rbox2hbb(rboxs):
    """
    Trans rbox format to hbb format
    Args:
        rboxes (array/tensor): (num_gts, rbox) 

    Returns:
        hbboxes (array/tensor): (num_gts, [xc yc w h]) 
    """
    assert rboxs.shape[-1] == 8
    if isinstance(rboxs, torch.Tensor):
        x = rboxs[:, 0::2] # (num, 4) 
        y = rboxs[:, 1::2]
        x_max = torch.amax(x, dim=1) # (num)
        x_min = torch.amin(x, dim=1)
        y_max = torch.amax(y, dim=1)
        y_min = torch.amin(y, dim=1)
        x_ctr, y_ctr = (x_max + x_min) / 2.0, (y_max + y_min) / 2.0 # (num)
        h = y_max - y_min # (num)
        w = x_max - x_min
        x_ctr, y_ctr, w, h = x_ctr.reshape(-1, 1), y_ctr.reshape(-1, 1), w.reshape(-1, 1), h.reshape(-1, 1) # (num, 1)
        hbboxes = torch.cat((x_ctr, y_ctr, w, h), dim=1)
    else:
        x = rboxs[:, 0::2] # (num, 4) 
        y = rboxs[:, 1::2]
        x_max = np.amax(x, axis=1) # (num)
        x_min = np.amin(x, axis=1) 
        y_max = np.amax(y, axis=1)
        y_min = np.amin(y, axis=1)
        x_ctr, y_ctr = (x_max + x_min) / 2.0, (y_max + y_min) / 2.0 # (num)
        h = y_max - y_min # (num)
        w = x_max - x_min
        x_ctr, y_ctr, w, h = x_ctr.reshape(-1, 1), y_ctr.reshape(-1, 1), w.reshape(-1, 1), h.reshape(-1, 1) # (num, 1)
        hbboxes = np.concatenate((x_ctr, y_ctr, w, h), axis=1)
    return hbboxes

def rbox2box(rboxs):
    """
    Trans rbox format to box format
    rbox 的顺序为左上 右上 右下 左下
    Args:
        rboxes (array/tensor): (num_gts, rbox)

    Returns:
        hbboxes (array/tensor): (num_gts, [xc yc w h angle])
    """
    assert rboxs.shape[-1] == 8
    if isinstance(rboxs, torch.Tensor):
        x = rboxs[:, 0::2] # (num, 4)
        y = rboxs[:, 1::2]
        x_max = torch.amax(x, dim=1) # (num)
        x_min = torch.amin(x, dim=1)
        y_max = torch.amax(y, dim=1)
        y_min = torch.amin(y, dim=1)
        x_ctr, y_ctr = (x_max + x_min) / 2.0, (y_max + y_min) / 2.0 # (num)
        w = torch.sqrt(torch.pow((x[:, 1] - x[:, 0]), 2) + torch.pow((y[:, 1] - y[:, 0]), 2)) # (num)
        h = torch.sqrt(torch.pow((x[:, 2] - x[:, 1]), 2) + torch.pow((y[:, 2] - y[:, 1]), 2))
        # 向量v (此时的物体中心点与 x1y1与x2y2中点 这两个点的向量)
        vx = (x[:, 0] + x[:, 1]) / 2.0 - x_ctr
        vy = (y[:, 0] + y[:, 1]) / 2.0 - y_ctr
        v0x = torch.Tensor([0.] * vx.size()[0]).to(rboxs.device)
        v0y = torch.Tensor([-1.] * vy.size()[0]).to(rboxs.device)
        # 求v0(0,1)到v顺时针的角度
        angle_cw = 360.0 - (torch.rad2deg(torch.atan2(vx * v0y - vy * v0x, v0x * vx + v0y * vy)) + 360.0) % 360.0
        x_ctr, y_ctr, w, h, angle = x_ctr.reshape(-1, 1), y_ctr.reshape(-1, 1), w.reshape(-1, 1), h.reshape(-1, 1), angle_cw.reshape(-1, 1) # (num, 1)
        boxes = torch.cat((x_ctr, y_ctr, w, h, angle), dim=1)
    else:
        x = rboxs[:, 0::2] # (num, 4)
        y = rboxs[:, 1::2]
        x_max = np.amax(x, axis=1) # (num)
        x_min = np.amin(x, axis=1)
        y_max = np.amax(y, axis=1)
        y_min = np.amin(y, axis=1)
        x_ctr, y_ctr = (x_max + x_min) / 2.0, (y_max + y_min) / 2.0 # (num)
        w = np.sqrt(np.power((x[:, 1] - x[:, 0]), 2) + np.power((y[:, 1] - y[:, 0]), 2))  # (num)
        h = np.sqrt(np.power((x[:, 2] - x[:, 1]), 2) + np.power((y[:, 2] - y[:, 1]), 2))
        vx = (x[:, 0] + x[:, 1]) / 2.0 - x_ctr
        vy = (y[:, 0] + y[:, 1]) / 2.0 - y_ctr
        v0x = np.ones_like(vx, dtype=vx.dtype) * 0.
        v0y = np.ones_like(vy, dtype=vy.dtype) * -1.
        angle_cw = 360.0 - (np.rad2deg(np.arctan2(vx * v0y - vy * v0x, v0x * vx + v0y * vy)) + 360.0) % 360.0
        x_ctr, y_ctr, w, h, angle = x_ctr.reshape(-1, 1), y_ctr.reshape(-1, 1), w.reshape(-1, 1), h.reshape(-1, 1), angle_cw.reshape(-1, 1)  # (num, 1)
        boxes = np.concatenate((x_ctr, y_ctr, w, h, angle), axis=1)
    return boxes



def clockwise_angle(v1, v2):
    # 从v2到v1 顺时针的角度
    # 默认起点相同(0,0)
    x1,y1 = v1[:, 0],v1[:, 1]
    x2,y2 = v2[:, 0],v2[:, 1]
    dot = x1*x2+y1*y2
    det = x1*y2-y1*x2
    theta = np.arctan2(det, dot)

    for i, t in enumerate(theta):
        theta[i] = theta[i] if theta[i]>0 else 2*np.pi+theta[i]
        theta[i] = theta[i] / np.pi * 180.0
    return theta

def rbox_filter(rbox, h, w):
    """
    Filter the rbox labels which is out of the image.
    Args:
        polys (array): (num, 8)

    Return：
        keep_masks (array): (num)
    """
    x = rbox[:, 0::2] # (num, 4)
    y = rbox[:, 1::2]
    x_max = np.amax(x, axis=1) # (num)
    x_min = np.amin(x, axis=1)
    y_max = np.amax(y, axis=1)
    y_min = np.amin(y, axis=1)
    x_ctr, y_ctr = (x_max + x_min) / 2.0, (y_max + y_min) / 2.0 # (num)
    keep_masks = (x_ctr > 0) & (x_ctr < w) & (y_ctr > 0) & (y_ctr < h)
    return keep_masks


# def angle_mode_labels(angle_mode, labels, action, t2='', tbox=''):
#     if not isinstance(labels, torch.Tensor):
#         print("angle_mode_labels TYPE ERROR")
#     # action 0,1 represent encode(tangle -> x) and decoder(pangle -> 1) , x=8 16 10...
#     if action == 0:
#         if angle_mode == "MDD":
#             # 顺时针象限依次为 0 1 2 3
#             # 16维zeros -> 16维 4个ont hot
#             # labels [n]    t2 [n,16]
#             angle = (labels + 0.01) % 360
#             # print(angle)
#             # n = len(labels)
#             # 1 2 3 4 -> 0 1 2 3
#             a1 = ((angle / 90).ceil() % 5 - 1).long()
#             a2 = (((angle % 90) / 22.5).ceil() % 5 - 1).long()
#             a3 = (((angle % 22.5) / 5.625).ceil() % 5 - 1).long()
#             a4 = (((angle % 5.625) / 1.40625).ceil() % 5 - 1).long()
#             # 将四个分类转换成 one-hot 向量
#             a1_hot = torch.zeros((len(labels), 4)).to(labels.device).scatter_(1, a1.unsqueeze(1), 1)
#             a2_hot = torch.zeros((len(labels), 4)).to(labels.device).scatter_(1, a2.unsqueeze(1), 1)
#             a3_hot = torch.zeros((len(labels), 4)).to(labels.device).scatter_(1, a3.unsqueeze(1), 1)
#             a4_hot = torch.zeros((len(labels), 4)).to(labels.device).scatter_(1, a4.unsqueeze(1), 1)
#             # angle [n,16]
#             return torch.cat([a1_hot, a2_hot, a3_hot, a4_hot], dim=1).to(t2.device)
#         if angle_mode == "8":
#             return box2rbox(tbox)
#         if angle_mode == "8+1":
#             sites = box2rbox(tbox[:, 0:8])
#             head_x = torch.Tensor((sites[:, 0] + sites[:, 2]) / 2).to(t2.device)
#             head_y = torch.Tensor((sites[:, 1] + sites[:, 3]) / 2).to(t2.device)
#             return torch.cat((sites, head_x.unsqueeze(1), head_y.unsqueeze(1)), 1)
#         if angle_mode == '5+1':
#             sites = box2rbox(tbox[:, 0:5])
#             head_x = torch.Tensor((sites[:, 0] + sites[:, 2]) / 2).to(t2.device)
#             head_y = torch.Tensor((sites[:, 1] + sites[:, 3]) / 2).to(t2.device)
#             return torch.cat((head_x.unsqueeze(1), head_y.unsqueeze(1)), 1)
#
#
#     if action == 1:
#         # x维 -> 1维
#         # labels [n,x]
#         # output [n]
#         if angle_mode == "MDD":
#             angle = labels
#             a1 = torch.argmax(angle[:, 0:4], dim=1)
#             a2 = torch.argmax(angle[:, 4:8], dim=1)
#             a3 = torch.argmax(angle[:, 8:12], dim=1)
#             a4 = torch.argmax(angle[:, 12:16], dim=1)
#             output = torch.stack((a1, a2, a3, a4), dim=1)
#             output_a = output[:, 0] * 90 +  output[:, 1] * 22.5 \
#                 + output[:, 2] * 5.625 + output[:, 3] * 1.40625
#             # labels = np.concatenate((labels[:, 0:5], output_a.T.reshape((-1, 1))), axis=1)
#             return torch.round(output_a).long() % 360
#         if angle_mode == "8":
#             return rbox2box(labels)[:, -1].round().long() % 360
#         if angle_mode == '8+1':
#             return rbox2box(labels[:, 0:8])[:, -1].round().long() % 360
#         if angle_mode == '5+1':
#             # t2 is pbox
#             north = torch.zeros_like(labels).to(labels.device)
#             north[:, 1] = -1
#             v0x, v0y = north[:, 0], north[:, 1]
#             vx, vy = labels[:, 0] - t2[:, 0], labels[:, 1] - t2[:, 1]
#             # v = torch.cat((labels[:, 0].unsqueeze(1), labels[:, 1].unsqueeze(1) - 1.), 1)
#             theta = 360.0 - (torch.rad2deg(torch.atan2(vx * v0y - vy * v0x, v0x * vx + v0y * vy)) + 360.0) % 360 - 0.001
#             return torch.round(theta).long() % 360


def smooth_label(one_hot_label):
    # 假设 one_hot_label 是一个 n 行 m 列的张量，其中 m 是 one-hot 编码的长度，一个位置为 1，其余为 0
    center_index = torch.nonzero(one_hot_label)[0, 1].item()
    # 计算每个位置到中心的距离
    distances = torch.abs(torch.arange(one_hot_label.size(1)) - center_index).float()
    # 定义高斯分布的标准差，控制平滑程度
    sigma = 0.6
    # 计算高斯分布的权重，越近中心权重越大
    gaussian_weights = torch.exp(-torch.pow(distances, 2) / (2 * sigma**2))
    return gaussian_weights.view((-1, one_hot_label.shape[1]))


def angle_decode(angle_fitting_methods, angle_l, labels, status):
    if angle_fitting_methods == 'MDD' or angle_fitting_methods == 'MDD+reg':
        if status == 'detect_eval':
            # [b, ang, h, w] 此时是在Detect的eval时
            num_division = angle_l[0]
            num_deep = angle_l[1]
            angle_output = torch.zeros([labels.shape[0], 1, labels.shape[2], labels.shape[3]]).to(labels.device)
            for i in range(num_deep):
                # 在每个维度里
                deg = 360./pow(num_division, i+1)
                i_labels = labels[:, i*num_division:(i+1)*num_division, :, :]
                i_labels = torch.sigmoid(i_labels)
                angle_output += torch.argmax(i_labels, dim=1, keepdim=True) * deg
        if status == 'loss_decode':
            # [b, 13125, ang]
            num_division = angle_l[0]
            num_deep = angle_l[1]
            angle_output = torch.zeros([labels.shape[0], labels.shape[1], 1]).to(labels.device)
            for i in range(num_deep):
                # 在每个维度里
                deg = 360./pow(num_division, i+1)
                i_labels = labels[:, :, i*num_division:(i+1)*num_division]
                i_labels = torch.sigmoid(i_labels)
                angle_output += torch.argmax(i_labels, dim=2, keepdim=True) * deg
        if status == 'loss_obb_decode':
            # [n, ang]
            num_division = angle_l[0]
            num_deep = angle_l[1]
            angle_output = torch.zeros([labels.shape[0], 1]).to(labels.device)
            for i in range(num_deep):
                # 在每个维度里
                deg = 360. / pow(num_division, i + 1)
                i_labels = labels[:, i * num_division:(i + 1) * num_division]
                i_labels = torch.sigmoid(i_labels)
                angle_output += torch.argmax(i_labels, dim=1, keepdim=True) * deg
    return angle_output


# 1 -> (4,4)
def angle_encode(angle_fitting_methods, angle_l, labels):
    if angle_fitting_methods == 'MDD' or angle_fitting_methods == 'MDD+reg':
        labels = (labels + 0.0001) % 360  # 防止360和0度冲突
        num_division = angle_l[0]
        num_deep = angle_l[1]
        n = labels.shape[0]
        output_labels = torch.zeros(n, num_division * num_deep).to(labels.device)
        angle_weight = torch.zeros(n, num_division * num_deep).to(labels.device)
        for i in range(num_deep):
            # 在每个维度里
            deg = 360. / pow(num_division, i + 1)  # 角度区间大小
            cur = labels // deg  # 第几个区间
            one_hot_label = output_labels[torch.arange(0, n), i * num_division:(i + 1) * num_division]
            one_hot_label[torch.arange(0, n), cur.long().flatten()] = 1.
            one_hot_label = smooth_label(one_hot_label).to(labels.device)
            output_labels[torch.arange(0, n), i * num_division:(i + 1) * num_division] = one_hot_label

            angle_weight[torch.arange(0, n), i * num_division:(i + 1) * num_division] = 1. / pow(1.5, i)
            labels = labels - deg * cur

        # angle_weight
        if angle_fitting_methods == 'MDD+reg':
            return output_labels, angle_weight, labels
        return output_labels, angle_weight


def angle_mode2angles(angle_mode): # [n,7] angle 0-360
    """
    '360' -> 360
    """
    if angle_mode == '360':
        return 360
    if angle_mode == '4*4*4*4':
        return 16
    if angle_mode == '8':
        return 8
    if angle_mode == '8+1':
        return 10
    if angle_mode == '5+1':
        return 2

def head_iou(pangle, tangle):
    # 对比是否角度小于90度
    A, B = torch.meshgrid(pangle, tangle, indexing='xy')
    a1 = torch.abs(A - B) % 360.
    a2 = torch.min(a1, 360. - a1)
    is_true = a2 < 5.
    # 差值
    diff = torch.min(a1, a2)
    return is_true, diff

def calc_HP(epoch, ap50, hcc):
    def sigmoid50(x):
        return 1 / (1 + np.exp(-(x - 50) / 5))
    return (ap50 + sigmoid50(epoch) * hcc) / (1 + sigmoid50(epoch))

