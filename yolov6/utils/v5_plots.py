# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Plotting utils
"""

import contextlib
import math
import os
from copy import copy
from pathlib import Path
from urllib.error import URLError

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from PIL import Image, ImageDraw, ImageFont
from utils.rotation import *
from utils import TryExcept, threaded
from utils.general import (CONFIG_DIR, FONT, LOGGER, check_font, check_requirements, clip_boxes, increment_path,
                           is_ascii, xywh2xyxy, xyxy2xywh)
from utils.metrics import fitness
from utils.segment.general import scale_image

# Settings
RANK = int(os.getenv('RANK', -1))
matplotlib.rc('font', **{'size': 11})
matplotlib.use('Agg')  # for writing to files only


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


def check_pil_font(font=FONT, size=10):
    # Return a PIL TrueType Font, downloading to CONFIG_DIR if necessary
    font = Path(font)
    font = font if font.exists() else (CONFIG_DIR / font.name)
    try:
        return ImageFont.truetype(str(font) if font.exists() else font.name, size)
    except Exception:  # download if missing
        try:
            check_font(font)
            return ImageFont.truetype(str(font), size)
        except TypeError:
            check_requirements('Pillow>=8.4.0')  # known issue https://github.com/ultralytics/yolov5/issues/5374
        except URLError:  # not online
            return ImageFont.load_default()


class Annotator:
    # YOLOv5 Annotator for train/val mosaics and jpgs and detect/hub inference annotations
    def __init__(self, im, line_width=None, font_size=None, font='Arial.ttf', pil=False, example='abc'):
        assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images.'
        non_ascii = not is_ascii(example)  # non-latin labels, i.e. asian, arabic, cyrillic
        self.pil = pil or non_ascii
        if self.pil:  # use PIL 默认用opencv
            self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
            self.draw = ImageDraw.Draw(self.im)
            self.font = check_pil_font(font='Arial.Unicode.ttf' if non_ascii else font,
                                       size=font_size or max(round(sum(self.im.size) / 2 * 0.035), 12))
        else:  # use cv2
            self.im = im
        # 框框的线宽 或者自适应生成
        self.lw = 2 #line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  # line width

    def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        # Add one xyxy box to image with label
        # label : person 0.54
        # box 是左上和右下坐标
        # 这个函数通常用在检测nms后（detect.py中）将最终的预测bounding box在原图中画出来，不过这个函数依次只能画一个框框。
        if self.pil or not is_ascii(label):
            self.draw.rectangle(box, width=self.lw, outline=color)  # box
            if label:
                w, h = self.font.getsize(label)  # text width, height
                outside = box[1] - h >= 0  # label fits outside box
                self.draw.rectangle(
                    (box[0], box[1] - h if outside else box[1], box[0] + w + 1,
                     box[1] + 1 if outside else box[1] + h + 1),
                    fill=color,
                )
                # self.draw.text((box[0], box[1]), label, fill=txt_color, font=self.font, anchor='ls')  # for PIL>8.0
                self.draw.text((box[0], box[1] - h if outside else box[1]), label, fill=txt_color, font=self.font)
        else:  # cv2 默认
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
            if label:
                tf = max(self.lw - 1, 1)  # 字体粗细 font thickness
                w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]  # text width, height
                outside = p1[1] - h >= 3
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(self.im,
                            label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                            0,
                            self.lw / 3,
                            txt_color,
                            thickness=tf,
                            lineType=cv2.LINE_AA)

    def rbox_label(self, box, label='', angle=None, color=(128, 128, 128), txt_color=(255, 255, 255)):
        # Add one xywh and angle box to image with label 真实大小
        # label : person 0.54
        # 这个函数通常用在检测nms后（detect.py中）将最终的预测bounding box在原图中画出来，不过这个函数依次只能画一个框框。
        if isinstance(box, torch.Tensor):
            box = box.cpu().numpy()
        if isinstance(box[0], torch.Tensor):
            box = [x.cpu().numpy() for x in box]
        if isinstance(self.im, Image.Image):
            im = pil2cv2(self.im)
        else:
            im = self.im
        if angle: # 传入angle说明是detect 没传入之前已经转过,说明是8个点
            rbox_list = box2rbox_numpy(box[0], box[1], box[2], box[3], angle) #只在detect用

        else:
            rbox_list = [[box[0], box[1]],[box[2], box[3]], [box[4], box[5]] ,[box[6], box[7]]]
        rbox_list = np.array(rbox_list, dtype=int)

        # 确认方向的中点 左中点 右中点
        upmid = np.round((rbox_list[0] + rbox_list[1]) / 2)
        downmid = np.round((rbox_list[2] + rbox_list[3]) / 2)
        left = np.round((rbox_list[0] + rbox_list[3]) / 2)
        right = np.round((rbox_list[1] + rbox_list[2]) / 2)
        arrow_left = np.array([left, upmid], dtype=int)
        arrow_right = np.array([upmid, right], dtype=int)
        arrow_mid = np.array([upmid, downmid], dtype=int)

        cv2.drawContours(image=im, contours=[arrow_left, arrow_right, arrow_mid], contourIdx=-1, color=color,
                         thickness=self.lw)

        cv2.drawContours(image=im, contours=[rbox_list], contourIdx=-1, color=color, thickness=self.lw)

        if label:
            tf = max(self.lw - 1, 1)  # label字体的线宽 font thickness
            # cv2.getTextSize: 根据输入的label信息计算文本字符串的宽度和高度
            # 0: 文字字体类型  fontScale: 字体缩放系数  thickness: 字体笔画线宽
            # 返回retval 字体的宽高 (width, height), baseLine 相对于最底端文本的 y 坐标
            t_size = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]

            # 文字绘制位置为矩形框 的上面
            xmax, xmin, ymax, ymin = max(rbox_list[:, 0]), min(rbox_list[:, 0]), max(rbox_list[:, 1]), min(
                rbox_list[:, 1])
            x_label, y_label = int((xmax + xmin) / 2), int((ymax + ymin) / 2)
            # 同上面一样是个画框的步骤 标签画在矩形框内 但是线宽thickness=-1表示整个矩形都填充color颜色 w=t_size[0]
            cv2.rectangle(im, (x_label, y_label), (x_label + t_size[0] + 1, y_label + int(1.5 * t_size[1])), color, -1,
                          cv2.LINE_AA)  # filled
            # cv2.putText: 在图片上写文本 这里是在上面这个矩形框里写label + score文本
            # (c1[0], c1[1] - 2)文本左下角坐标  0: 文字样式  fontScale: 字体缩放系数
            # [225, 255, 255]: 文字颜色  thickness: tf字体笔画线宽     lineType: 线样式
            cv2.putText(im, label, (x_label, y_label + t_size[1]), 0, self.lw / 3, [225, 255, 255], thickness=tf,
                        lineType=cv2.LINE_AA)
        if isinstance(self.im, Image.Image):
            self.im = cv22pil(im)



    def rectangle(self, xy, fill=None, outline=None, width=1):
        # Add rectangle to image (PIL-only)
        self.draw.rectangle(xy, fill, outline, width)

    def text(self, xy, text, txt_color=(255, 255, 255), anchor='top'):
        # Add text to image (PIL-only)
        if anchor == 'bottom':  # start y from font bottom
            w, h = self.font.getsize(text)  # text width, height
            xy[1] += 1 - h
        self.draw.text(xy, text, fill=txt_color, font=self.font)

    def fromarray(self, im):
        # Update self.im from a numpy array
        self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
        self.draw = ImageDraw.Draw(self.im)

    def result(self):
        # Return annotated image as array
        return np.asarray(self.im)


def output_to_target(output, max_det=300):
    """用在test.py中进行绘制前300个batch的预测框predictions 因为只有predictions需要修改格式 target是不需要修改格式的
        以便在plot_images中进行绘图 + 显示label

        :params output: list{tensor(8)}分别对应着当前batch的8(batch_size)张图片做完nms后的结果
                        list中每个tensor[n, 6]  n表示当前图片检测到的目标个数
                        [xywh, conf, cls, angles]
        :return np.array(pred targets): [imgid class x y w h angle conf]
                        真正的targets [imgid class x y w h angle]
                        其中num_targets为当前batch中所有检测到目标框的个数
        """
    targets = []
    if isinstance(output, torch.Tensor):
        output = output.cpu().numpy()
    for i, o in enumerate(output):
        box, conf, cls, angle = o[:max_det, :6+1].cpu().split((4, 1, 1, 1), 1)
        j = torch.full((conf.shape[0], 1), i)
        targets.append(torch.cat((j, cls, box, angle, conf), 1))
        # targets.append(torch.cat((j, cls, box, angle), 1))
    return torch.cat(targets, 0).numpy()


@threaded
def plot_images(images, targets, paths=None, fname='images.jpg', names=None):
    # Plot image grid with labels
    # targets [36,7] train in is 0-1  , val in is [384,672]
    """用在val.py中进行绘制ground truth和预测框predictions(两个图) 一起保存 或者train.py中
        将整个batch的labels都画在这个batch的images上
        Plot image grid with labels
        :params images: 当前batch的所有图片  Tensor [batch_size, 3, h, w]  且图片都是归一化后的
        :params targets:   train_end和val第一个直接来自target: Tensor[num_target, img_index(在batch中哪个图片)+class+xywh+angle]  [num_target, 6+1]
                          来自output_to_target: Tensor[num_pred, batch_id+class+xywh+angle+conf [num_pred, 7+1] 看上面
        :params paths: tuple  当前batch中所有图片的地址
                       如: '..\\datasets\\coco128\\images\\train2017\\000000000315.jpg'
        :params fname: 最终保存的文件路径 + 名字  runs\train\exp8\train_batch2.jpg
        :params names: 传入的类名 从class index可以相应的key值  但是默认是None 只显示class index不显示类名
        :params max_size: 图片的最大尺寸640  如果images有图片的大小(w/h)大于640则需要resize 如果都是小于640则不需要resize
        :params max_subplots: 最大子图个数 16
        :params mosaic: 一张大图  最多可以显示max_subplots张图片  将总多的图片(包括各自的label框框)一起贴在一起显示
                        mosaic每张图片的左上方还会显示当前图片的名字  最好以fname为名保存起来
        """
    #   #######################################################
    #   -------------train---------------val--------------pred--
    #   -xywh----------Y------------------N----------------N----
    #   -scale---------Y------------------Y----------------Y----
    #   -*360----------Y------------------N----------------N----
    #   -conf----------N------------------N----------------Y----
    #   ########################################
    #   ###############

    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if targets.size == 0:
        return
    max_size = 1920  # max image size
    max_subplots = 16  # max image subplots, i.e. 4x4
    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)
    if np.max(images[0]) <= 1:
        images *= 255  # de-normalise (optional)

    # Build Image
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
    for i, im in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        im = im.transpose(1, 2, 0)
        mosaic[y:y + h, x:x + w, :] = im

    # Resize (optional)
    scale = max_size / ns / max(h, w)
    h_ = 1
    w_ = 1
    # if scale < 1: # 画图的比例 只小不大 ###
    if np.max(targets[:, 2:]) <= 1: # if train
        targets[:, -1] *= 360.
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)
        h_ = h
        w_ = w
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))

    # Annotate
    # 这里是im的尺寸是0-255尺寸(上面x255),label是真实尺寸(和im同尺度),xy为偏移
    fs = int((h + w) * ns * 0.01)  # font size
    annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs, pil=True, example=names)
    for i in range(i + 1):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)  # borders
        if paths:
            annotator.text((x + 5, y + 5), text=Path(paths[i]).name[:40], txt_color=(220, 220, 220))  # filenames
        if len(targets) > 0:
            ti = targets[targets[:, 0] == i]  # image targets [batch_id+class+xywh+angle+(conf)]
            rbox = ti[:, 2:7]
            classes = ti[:, 1].astype('int')
            labels = True
            conf = False
            if ti.shape[1] == 8 :
                labels = False  # labels if no conf column
                conf = ti[:, 7]  #7 check for confidence presence (label vs pred)

            if rbox.shape[0]:
                rbox[:, [0, 2]] *= w_  # scale to pixels
                rbox[:, [1, 3]] *= h_
                rbox = box2rbox(rbox)

            rbox[:, [0, 2, 4, 6]] += x
            rbox[:, [1, 3, 5, 7]] += y
             #[n,8]
            for j, box in enumerate(rbox.tolist()):
                cls = classes[j]
                color = colors(cls)
                cls = names[cls] if names else cls
                if labels or conf[j] > 0.25:  # TODO 0.25 conf thresh
                    label = f'{cls}' if labels else f'{cls} {conf[j]:.1f}'
                    annotator.rbox_label(box, label, color=color)
    annotator.im.save(fname)  # save


def plot_lr_scheduler(optimizer, scheduler, epochs=300, save_dir=''):
    # Plot LR simulating training for full epochs
    optimizer, scheduler = copy(optimizer), copy(scheduler)  # do not modify originals
    y = []
    for _ in range(epochs):
        scheduler.step()
        y.append(optimizer.param_groups[0]['lr'])
    plt.plot(y, '.-', label='LR')
    plt.xlabel('epoch')
    plt.ylabel('LR')
    plt.grid()
    plt.xlim(0, epochs)
    plt.ylim(0)
    plt.savefig(Path(save_dir) / 'LR.png', dpi=200)
    plt.close()


def plot_val_txt():  # from utils.plots import *; plot_val()
    # Plot val.txt histograms
    x = np.loadtxt('val.txt', dtype=np.float32)
    box = xyxy2xywh(x[:, :4])
    cx, cy = box[:, 0], box[:, 1]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
    ax.hist2d(cx, cy, bins=600, cmax=10, cmin=0)
    ax.set_aspect('equal')
    plt.savefig('hist2d.png', dpi=300)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    ax[0].hist(cx, bins=600)
    ax[1].hist(cy, bins=600)
    plt.savefig('hist1d.png', dpi=200)


def plot_targets_txt():  # from utils.plots import *; plot_targets_txt()
    # Plot targets.txt histograms
    x = np.loadtxt('targets.txt', dtype=np.float32).T
    s = ['x targets', 'y targets', 'width targets', 'height targets']
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    ax = ax.ravel()
    for i in range(4):
        ax[i].hist(x[i], bins=100, label=f'{x[i].mean():.3g} +/- {x[i].std():.3g}')
        ax[i].legend()
        ax[i].set_title(s[i])
    plt.savefig('targets.jpg', dpi=200)


@TryExcept()  # known issue https://github.com/ultralytics/yolov5/issues/5395
def plot_labels(labels, names=(), save_dir=Path('')):
    # plot dataset labels
    LOGGER.info(f"Plotting labels to {save_dir / 'labels.jpg'}... ")
    c, b = labels[:, 0], labels[:, 1:].transpose()  # classes, boxes
    nc = int(c.max() + 1)  # number of classes
    x = pd.DataFrame(b.transpose(), columns=['x', 'y', 'width', 'height','angle'])

    # seaborn correlogram
    sn.pairplot(x, corner=True, diag_kind='auto', kind='hist', diag_kws=dict(bins=50), plot_kws=dict(pmax=0.9))
    plt.savefig(save_dir / 'labels_correlogram.jpg', dpi=200)
    plt.close()

    # matplotlib labels
    matplotlib.use('svg')  # faster
    ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()
    y = ax[0].hist(c, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    with contextlib.suppress(Exception):  # color histogram bars by class
        [y[2].patches[i].set_color([x / 255 for x in colors(i)]) for i in range(nc)]  # known issue #3195
    ax[0].set_ylabel('instances')
    if 0 < len(names) < 30:
        ax[0].set_xticks(range(len(names)))
        ax[0].set_xticklabels(list(names.values()), rotation=90, fontsize=10)
    else:
        ax[0].set_xlabel('classes')
    sn.histplot(x, x='x', y='y', ax=ax[2], bins=50, pmax=0.9)
    sn.histplot(x, x='width', y='height', ax=ax[3], bins=50, pmax=0.9)

    # rectangles
    labels[:, 1:3] = 0.5  # center
    labels[:, 1:-1] = xywh2xyxy(labels[:, 1:-1]) * 2000
    img = Image.fromarray(np.ones((2000, 2000, 3), dtype=np.uint8) * 255)
    for cls, *box in labels[:1000]:
        ImageDraw.Draw(img).rectangle(box[:-1], width=1, outline=colors(cls))  # plot
    ax[1].imshow(img)
    ax[1].axis('off')

    for a in [0, 1, 2, 3]:
        for s in ['top', 'right', 'left', 'bottom']:
            ax[a].spines[s].set_visible(False)

    plt.savefig(save_dir / 'labels.jpg', dpi=200)
    matplotlib.use('Agg')
    plt.close()

def save_one_box(xyxy, im, file=Path('im.jpg'), gain=1.02, pad=10, square=False, BGR=False, save=True):
    # Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop
    xyxy = torch.tensor(xyxy).view(-1, 4)
    b = xyxy2xywh(xyxy)  # boxes
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = xywh2xyxy(b).long()
    clip_boxes(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1 if BGR else -1)]
    if save:
        file.parent.mkdir(parents=True, exist_ok=True)  # make directory
        f = str(increment_path(file).with_suffix('.jpg'))
        # cv2.imwrite(f, crop)  # save BGR, https://github.com/ultralytics/yolov5/issues/7007 chroma subsampling issue
        Image.fromarray(crop[..., ::-1]).save(f, quality=95, subsampling=0)  # save RGB
    return crop
