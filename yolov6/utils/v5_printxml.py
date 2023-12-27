# -*- coding: utf-8 -*- 
# @Time : 2022/12/22 17:34 
# @Author : DDD

import cv2
import torch
import numpy as np
import os
"""
PIL 2 CV2 :
import cv2
import numpy as np
img_bgr = cv2.cvtColor(np.asarray(image_pil), cv2.COLOR_RGB2BGR)

CV2 2 PIL:
from PIL import Image
image_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

cv2.imshow("1",im)
cv2.waitKey(5000)
cv2.destroyAllWindows()

## augmentations 238行返回之前 验证标签和变换是否对应
# 验证M
im2 = cv2.warpAffine(im, M[:2], dsize=(width*2, height*2), borderValue=(114, 114, 114))
cv2.imshow("2",source_im)
cv2.waitKey(4000)
cv2.imshow("1",img)
cv2.waitKey(3000)
cv2.destroyAllWindows()
# 验证返回前的targets
targets2 = targets.copy()
targets2[:, 1:5] = xyxy2xywh(targets[:, 1:5])
for _,t in enumerate(targets2):
    im44 = im.copy()
    label3, x2, y2, w2, h2, angle2 = t
    label3 = "dumper :0.066"
    plot_one_rbox([x2, y2, w2, h2], angle2, im44, color=(namelist.index(label3.split(' ')[0]), True), label=label3, line_thickness=3)

## dataloader 819行 验证mosiac后的xyxy标签对不对
labels2 = labels.copy()
labels2[:, 1:5] = xyxy2xywh(labels2[:, 1:5])
im44 = img4.copy()

label3, x2, y2, w2, h2, angle2 = labels2[0]
label3 = "dumper :0.066"
plot_one_rbox([x2, y2, w2, h2], angle2, im44, color=(namelist.index(label3.split(' ')[0]), True), label=label3, line_thickness=3)
"""
# namelist = ['Truck2']
namelist = ['dumper', 'bulldozer', 'navvy']
deg2rad = np.pi / 180



def box2rbox_numpy(x, y, w, h, angle): # [n,7]
    """
    x y w h angle -> x1y1(左上) x2y2(右上) x3y3(左下) x4y4(右下)
    :param predn:  预测值 [预测框数量 , 7 (x y w h angle) ]
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
    return [[x3_out, y3_out],[x4_out, y4_out], [x2_out, y2_out] ,[x1_out, y1_out]]


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        # 将hex列表中所有hex格式(十六进制)的颜色转换rgb格式的颜色
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        # 颜色个数
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        # 根据输入的index 选择对应的rgb颜色
        c = self.palette[int(i) % self.n]
        # 返回选择的颜色 默认是rgb
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        # hex -> rgb
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # 初始化Colors对象 下面调用colors的时候会调用__call__函数

def plot_one_rbox(x,angle, im, color=(128, 128, 128), label=None, line_thickness=3):
    """一般会用在detect.py中在nms之后变量每一个预测框，再将每个预测框画在原图上
    使用opencv在原图im上画一个bounding box
    :params x: 预测得到的bounding box  [x1 y1 x2 y2 angle]
    :params im: 原图 要将bounding box画在这个图上  array
    :params color: bounding box线的颜色
    :params labels:  person 0.54
    :params line_thickness: bounding box的线宽
    """
    # x的格式转换成 xy(*8) 的格式
    rbox_list = box2rbox_numpy(x[0], x[1], x[2], x[3], angle)

    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness

    rbox_list = np.array(rbox_list,dtype=int)

    # 确认方向的中点 左中点 右中点
    upmid = np.round((rbox_list[0] + rbox_list[1]) / 2)
    downmid = np.round((rbox_list[2] + rbox_list[3]) / 2)
    left = np.round((rbox_list[0] + rbox_list[3]) / 2)
    right = np.round((rbox_list[1] + rbox_list[2]) / 2)
    arrow_left = np.array([left, upmid],dtype=int)
    arrow_right = np.array([upmid, right], dtype=int)
    arrow_mid = np.array([upmid, downmid], dtype=int)

    cv2.drawContours(image=im, contours=[arrow_left, arrow_right, arrow_mid], contourIdx=-1, color=color, thickness=tl)

    cv2.drawContours(image=im, contours=[rbox_list], contourIdx=-1, color=color, thickness=tl)

    # 如果label不为空还要在框框上面显示标签label + score
    if label:
        tf = max(tl - 1, 1)  # label字体的线宽 font thickness
        # cv2.getTextSize: 根据输入的label信息计算文本字符串的宽度和高度
        # 0: 文字字体类型  fontScale: 字体缩放系数  thickness: 字体笔画线宽
        # 返回retval 字体的宽高 (width, height), baseLine 相对于最底端文本的 y 坐标
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

        # 文字绘制位置为矩形框 的上面
        xmax, xmin, ymax, ymin = max(rbox_list[:,0]), min(rbox_list[:,0]), max(rbox_list[:,1]), min(rbox_list[:,1])
        x_label, y_label = int((xmax + xmin)/2), int((ymax + ymin)/2)
        # 同上面一样是个画框的步骤 标签画在矩形框内 但是线宽thickness=-1表示整个矩形都填充color颜色 w=t_size[0]
        cv2.rectangle(im, (x_label, y_label), (x_label + t_size[0] + 1, y_label + int(1.5*t_size[1])), color, -1, cv2.LINE_AA)  # filled
        # cv2.putText: 在图片上写文本 这里是在上面这个矩形框里写label + score文本
        # (c1[0], c1[1] - 2)文本左下角坐标  0: 文字样式  fontScale: 字体缩放系数
        # [225, 255, 255]: 文字颜色  thickness: tf字体笔画线宽     lineType: 线样式
        cv2.putText(im, label, (x_label, y_label + t_size[1]), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

        #临时画点 测试
        # point_list = [[464, 504], [507, 160], [816, 198], [773, 542]]
        # for point in point_list:
        #     cv2.circle(im, point, 3, (255, 0, 0), 3)

        cv2.namedWindow('test1', cv2.WINDOW_GUI_NORMAL)

        # cv2.startWindowThread()
        cv2.imshow('test1', im)
        # cv2.waitKey(1000)

        cv2.waitKey(5000)
        # cv2.destroyAllWindows()

def getvalue(txt, height, width): # 1080 1920
    if isinstance(txt, str):
        with open(txt, 'r') as f:
            for t in f.readlines():
                label = namelist[int(t.split(' ')[0])]
                x = round(float(t.split(' ')[1]) * width)
                y = round(float(t.split(' ')[2]) * height)
                w = round(float(t.split(' ')[3]) * width)
                h = round(float(t.split(' ')[4]) * height)
                angle = float(t.split(' ')[5]) * 360
    else:
        label = namelist[int(txt[0])]
        x = round(float(txt[1]) * width)
        y = round(float(txt[2]) * height)
        w = round(float(txt[3]) * width)
        h = round(float(txt[4]) * height)
        angle = float(txt[5]) * 360
    return label+' 0.99', x, y, w, h, angle

if __name__ == '__main__':
    #imgpath = input('Path : ')
    imgpath = r'/Users/ddd/Python/learn_torch/yolo/datasets/xiaoche2/images/WIN_20211211_13_59_06_Pro.jpg'
    txtpath = os.path.join(os.path.dirname(os.path.dirname(imgpath)), r'labels', imgpath.split(r'/')[-1])
    txtpath = os.path.splitext(txtpath)[0] + '.txt'

    with open(txtpath, 'r') as f:
        for t in f.readlines():
            im = cv2.imread(imgpath)
            height, width = im.shape[0:2]
            label, x, y, w, h, angle = getvalue(t.split(' '), height, width)
            # label3, x2, y2, w2, h2, angle2 = ['dumper :0.066', 392.26, 597.03, 41.667, 64.667, 0.015915 * 360.0]
            # plot_one_rbox([x2, y2, w2, h2], angle2, im, color=(namelist.index(label.split(' ')[0]), True), label=label3,
            #               line_thickness=3)
            plot_one_rbox([x, y, w, h], angle, im, color=(namelist.index(label.split(' ')[0]), True), label=label, line_thickness=3)
    pass