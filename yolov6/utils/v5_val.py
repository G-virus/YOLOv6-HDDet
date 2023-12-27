# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 detection model on a detection dataset

Usage:
    $ python val.py --weights yolov5s.pt --data coco128.yaml --img 640

Usage - formats:
    $ python val.py --weights yolov5s.pt                 # PyTorch
                              yolov5s.torchscript        # TorchScript
                              yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                              yolov5s_openvino_model     # OpenVINO
                              yolov5s.engine             # TensorRT
                              yolov5s.mlmodel            # CoreML (macOS-only)
                              yolov5s_saved_model        # TensorFlow SavedModel
                              yolov5s.pb                 # TensorFlow GraphDef
                              yolov5s.tflite             # TensorFlow Lite
                              yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                              yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import json
import os
import sys
from pathlib import Path
from utils.rotation import box2rbox, rbox2hbb, rbox2box, angle_mode_labels, head_iou
from utils.r_iou_torch import *
import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, Profile, check_dataset, check_img_size, check_requirements, check_yaml,
                           coco80_to_coco91_class, colorstr, increment_path, non_max_suppression, print_args,
                           scale_boxes, scale_rboxes, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou, rbox_iou
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, smart_inference_mode

from mmcv.ops import box_iou_rotated


def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({
            'image_id': image_id,
            'category_id': class_map[int(p[5])],
            'bbox': [round(x, 3) for x in b],
            'score': round(p[4], 5)})


def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


def process_batch_rbox(detections, labels, iouv, nc):
    """
    # 作用1：对预测框与gt进行一一匹配
    # 作用2：对匹配上的预测框进行iou数值判断，用Ture来填充，其余没有匹配上的预测框的所以行数全部设置为False
    process_batch 的 加角度版, 利用效率低下的 r_iou_torch 计算iou
    Return correct prediction matrix
    Arguments:
        detections (array[N, 7]), x, y, w, h, angle, conf, class

        labels (array[M, 6]), class, x, y, w, h, angle
    Returns:c
        correct (array[N, 10]), for 10 IoU levels
        head_correct (nc , 2, 10) for per class  (head accuracy , head mean loss)   10 iou
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    head_correct = torch.zeros((nc, 2, iouv.shape[0])).to(detections.device)
    # xy 都设为 0 ,0
    # iou = rbox_iou(labels[:, 1:6], detections[:, :5]) source
    if torch.cuda.is_available():
        bboxes1 = labels[:, 1:6].contiguous()
        bboxes2 = detections[:, :5].contiguous()
        iou = box_iou_rotated(bboxes1, bboxes2).to(labels.device)
    else:
        iou = box_iou(rbox2hbb(box2rbox(labels[:, 1:6])), rbox2hbb((box2rbox(detections[:, :5]))))
    # iou = torch.rand_like(iou)
    # detections[:, 4] = torch.rand_like(detections[:, 4]) * 350
    head_true, head_diff = head_iou(detections[:, 4], labels[:, 5])
    correct_class = labels[:, 0:1] == detections[:, 6]
    for i in range(len(iouv)):
        # 只有同时符合以上两点条件才被赋值为True，此时返回当前矩阵的一个行列索引，x是两个元祖x1,x2
        # 点(x[0][i], x[1][i])就是 class   符合条件的预测框
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        head_x = torch.where((iou >= iouv[i]) & correct_class & head_true)
        if head_x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            head_matches = torch.cat((torch.stack(head_x, 1), iou[head_x[0], head_x[1]][:, None]), 1).cpu().numpy()
            if head_x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]  # 应该取这里 这里包括多个预测

                # 对于每个类
                head_matches = head_matches[head_matches[:, 2].argsort()[::-1]]
                for per_class in range(nc):
                    head_class_matches = head_matches[head_matches[:, 0] == per_class]
                    head_sum = matches[matches[:, 0] == per_class].shape[0]
                    head_true_sum = head_class_matches.shape[0]
                    if head_sum > 0:
                        head_correct[per_class, 0, i] = head_true_sum / (head_sum + 1e-6)
                    # loss 包括 必须>iou 还需要此分类
                    xx = torch.stack((torch.Tensor(x[0]), torch.Tensor(x[1])), 1).T
                    mask = xx[0, :] == per_class
                    xx = xx[:, mask]
                    if torch.numel(xx) > 0:
                        xx = (xx[0], xx[1])
                        head_correct[per_class, 1, i] = head_diff[xx].mean()

                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                # 这两步是确保每个gt只对应一个预测，每个预测也只对应一个gt
                # 如果出现一对多的情况，则取iou最高的那个
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                # matches:  类别  对应类别的第几个预测框  iou值
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device), head_correct


@smart_inference_mode()
def run(
        data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        max_det=300,  # maximum detections per image
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
        angle_mode='360'
):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half = model.fp16  # FP16 supported on limited backends with CUDA
        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not (pt or jit):
                batch_size = 1  # export.py models default to batch-size 1
                LOGGER.info(f'Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

        # Data
        data = check_dataset(data)  # check

    # Configure 加载数据配置信息
    model.eval()  # eval()时，框架会自动把BN和Dropout国定住，用训练好的值；不启用BatchNormalization和Dropout
    cuda = device.type != 'cpu'
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith(f'coco{os.sep}val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 11, device=device) # iou vector for mAP@0.5:0.95
    # iouv = torch.linspace(0.01, 0.5, 11,device=device)  # iou vector for mAP@0.0:0.5 just for OHD-SJTU Head Accuracy benchmark
    niou = iouv.numel()

    # Dataloader
    if not training:
        if pt and not single_cls:  # check --weights are trained on --data
            ncm = model.model.nc
            assert ncm == nc, f'{weights} ({ncm} classes) trained on different --data than what you passed ({nc} ' \
                              f'classes). Pass correct combination of --weights and --data that are trained together.'
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
        pad, rect = (0.0, False) if task == 'speed' else (0.5, pt)  # square inference for benchmarks
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task],
                                       imgsz,
                                       batch_size,
                                       stride,
                                       single_cls,
                                       pad=pad,
                                       rect=rect,
                                       workers=workers,
                                       prefix=colorstr(f'{task}: '))[0]

    seen = 0  # 初始化图片数量
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = model.names if hasattr(model, 'names') else model.module.names  # get class names 类名
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%22s' + '%11s' * 8) % (
    'Class', 'Images', 'Instances', 'P', 'R', 'mAP0', 'mAP25', 'mAP0-50', 'h_acc50')  # , 'h_acc', 'h_loss50', 'h_loss')
    tp, fp, p, r, f1, mp, mr, map50, ap50, map, map25, map0 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dt = Profile(), Profile(), Profile()  # profiling times
    loss = torch.zeros(3 + 1, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    callbacks.run('on_val_start')
    pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        callbacks.run('on_val_batch_start')
        with dt[0]:
            if cuda:
                im = im.to(device, non_blocking=True)
                targets = targets.to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            nb, _, height, width = im.shape  # batch size, channels, height, width

        # Inference 前项推断
        with dt[1]:  # preds是 打过**2框的 而且是stride尺寸的. train_out是原始0-1
            preds, train_out = model(im) if compute_loss else (model(im, augment=augment), None)

        # Loss
        if compute_loss:
            loss += compute_loss(train_out, targets, -1, angle_mode)[1]  # box, obj, cls , angle 51 是1度

        ## 一计算完损失 马上换回 xywh theta 格式
        if angle_mode == '360':
            pass
        else:
            t = []
            if isinstance(preds, (list, tuple)):
                preds = preds[0]
            for si, pred in enumerate(preds):
                # get one detect angle
                angle_site = angle_mode_labels(angle_mode, pred[:, 4 + 1 + nc:], 1, pred[:, :4])

                range360 = torch.zeros(pred.shape[0], 360).to(preds.device)
                range360[range(pred.shape[0]), angle_site] = 1
                # 扩张 n,8+16  ->  n,8+360
                t.append(torch.cat((pred[:, 0:4 + 1 + nc], range360), 1))
            preds = torch.stack(t, dim=0)
            pass

        # NMS 将真实框target的xywh(因为target是在labelimg中做了归一化的)映射到img(test)尺寸
        targets[:, 2:6 + 1] *= torch.tensor((width, height, width, height, 360), device=device)  # to pixels 此时角度是0-360
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        with dt[2]:
            preds = non_max_suppression(preds,  # 多个batch一起算
                                        conf_thres,
                                        iou_thres,
                                        labels=lb,
                                        multi_label=True,
                                        agnostic=single_cls,
                                        max_det=max_det)
        # targets 是[9,7] [imgid class x y w h angle] (384,672维度) 360
        # pred是n个bs的列表 每个列表是 [xywh, conf, cls, angles] (384,672维度) 0~360  obb的顺序是不一样的
        # Metrics 每一张图片做统计 写入预测信息到txt, 生成json字典
        for si, pred in enumerate(preds):
            # 获取第si个batch的图片的标签信息，包括x,y,w,h,cls,class,angle
            labels = targets[targets[:, 0] == si, 1:7]  # 标签记录 去掉imgid 最后一个为angle
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            head_correct = torch.zeros((nc, 2, niou), dtype=torch.float, device=device)
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct.cpu(), *torch.zeros((2, 0), device='cpu'), labels[:, 0].cpu(),
                                  torch.zeros((nc, 2, 11), device='cpu')))
                    if plots:
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue

            """
            hbb 原先的

            # Predictions
            if single_cls:
                pred[:, 5+1] = 0
            # 先处理成rbox 再xywh 再xyxy 再缩放成原图
            predn = pred.clone()
            # 从x y w h angle 转化成 x1y1 x2y2 x3y3 x4y4
            # rbox [n, 8] 传入xywh angle
            rbox = box2rbox(predn[:, [0, 1, 2, 3, 6]])
            stat_rbox = torch.cat((rbox, pred[:, 4:6]), dim=1) # [xy*8, conf, cls]
            # pred_rboxn :  xyxy conf cls
            pred_rboxn = torch.cat((xywh2xyxy(rbox2hbb(rbox)), pred[:, 4:6]), dim=1)
            # [384, 672]        (1080, 1920)    ((0.33, 0.33), (16.0, 12.0))
            scale_boxes(im[si].shape[1:], pred_rboxn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate 注意labels没有conf,所以是6维,pred是7维
            # labels [class x y w h angle]
            # 需要转化为 labelsn [class xyxy]
            if nl:
                trbox = box2rbox(labels[:, [1, 2, 3, 4, 5]]) # target boxes
                labelsn = torch.cat((labels[:, 0].reshape((labels.shape[0],1)), xywh2xyxy(rbox2hbb(trbox))),1)
                scale_boxes(im[si].shape[1:], labelsn[1:], shape, shapes[si][1])  # native-space labels

                correct = process_batch(pred_rboxn, labelsn, iouv)
                # unsupported
                # if plots:
                #     confusion_matrix.process_batch(pred_rboxn, labelsn)
            stats.append((correct.cpu(), stat_rbox[:, 8].cpu(), stat_rbox[:, 9].cpu(), labels[:, 0]))  # (correct, conf, pcls, tcls)
            """
            # Predictions 使用效率缓慢的真实r_iou后
            if single_cls:
                pred[:, 5 + 1] = 0
            # xywh先处理成rbox 再缩放成原图 再xywh
            predn = pred.clone()
            # [xywh, conf, cls, angles]
            # rbox [n, 8] 传入xywh angle
            rboxn = box2rbox(predn[:, [0, 1, 2, 3, 6]])
            # stat_rbox = torch.cat((rbox, pred[:, 4:6]), dim=1)  # [xy*8, conf, cls]
            # pred_rboxn :  xyxy conf cls
            # pred_rboxn = torch.cat((xywh2xyxy(rbox2hbb(rbox)), pred[:, 4:6]), dim=1)
            # [384, 672]        (1080, 1920)    ((0.33, 0.33), (16.0, 12.0))
            scale_rboxes(im[si].shape[1:], rboxn, shape, shapes[si][1])  # native-space pred
            boxn = rbox2box(rboxn)
            boxn = torch.cat((boxn, predn[:, 4:6]), dim=1)  # [xywh angles conf cls]

            # Evaluate 注意labels没有conf,所以是6维,pred是7维
            # labels [class x y w h angle]
            # 需要转化为 labelsn [class xyxy]
            if nl:
                trbox = box2rbox(labels[:, [1, 2, 3, 4, 5]])  # target boxes
                scale_rboxes(im[si].shape[1:], trbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0].reshape((labels.shape[0], 1)), rbox2box(trbox)), 1)

                # boxn : [xywh angles conf cls]
                # labelsn : [class x y w h angle]
                correct, head_correct = process_batch_rbox(boxn, labelsn, iouv, nc)
                # unsupported
                # if plots:
                #     confusion_matrix.process_batch(pred_rboxn, labelsn)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(),
                          labels[:, 0].cpu(), head_correct.cpu()))  # (correct, conf, pcls, tcls, pangle, tangle)

            # Save/log
            if save_txt:
                save_one_txt(boxn, save_conf, shape, file=save_dir / 'labels' / f'{path.stem}.txt')
            if save_json:
                save_one_json(boxn, jdict, path, class_map)  # append to COCO-JSON dictionary
            # 这里的pred应该处理成xyxy,所以没有意义 还没改
            callbacks.run('on_val_image_end', pred, boxn, path, names, im[si])

        # Plot images
        if plots and batch_i < 3:
            plot_images(im, targets, paths, save_dir / f'val_batch{batch_i}_labels.jpg', names)  # labels
            plot_images(im, output_to_target(preds), paths, save_dir / f'val_batch{batch_i}_pred.jpg', names)  # pred

        callbacks.run('on_val_batch_end', batch_i, im, targets, paths, shapes, preds)

    # Compute metrics 计算map
    # print([type(x) for x in zip(*stats)])
    # input()
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    # stats = [torch.cat(x, 0) for x in zip(*stats)]
    head_acc, head_acc_50, head_loss, head_loss_50 = np.zeros((4, nc))
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class, head_output = ap_per_class(*stats, nc, plot=plots, save_dir=save_dir,
                                                                   names=names)
        ap0, ap25, ap = ap[:, 0], ap[:, 5], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map0, map25, map = p.mean(), r.mean(), ap0.mean(), ap25.mean(), ap.mean()
        map50 = map ############
        # for head
        head_acc, head_acc_50 = head_output[:, 0, -1], head_output[:, 0, 0]
        head_loss, head_loss_50 = head_output[:, 1, -1], head_output[:, 1, 0]
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class

    # Print results
    pf = '%22s' + '%11i' * 2 + '%11.3g' * 6  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map0, map25, map,
                      head_acc_50.mean()))  # , head_acc.mean(), head_loss_50.mean(), head_loss.mean()))
    if nt.sum() == 0:
        LOGGER.warning(f'WARNING ⚠️ no labels found in {task} set, can not compute metrics without labels')

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap0[i], ap25[i], ap[i],
                              head_acc_50[i]))  # head_acc[i], head_loss_50[i], head_loss[i]))

    # Print speeds
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run('on_val_end', nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix)

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements('pycocotools')
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/xiaoche.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'best.pt', help='model path(s)')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=1024, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.1, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=300, help='maximum detections per image')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=0, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class') # 每个类别map
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--angle_mode', type=str, default='4*4*4*4', help='angle_mode')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(f'WARNING ⚠️ confidence threshold {opt.conf_thres} > 0.001 produces invalid results')
        if opt.save_hybrid:
            LOGGER.info('WARNING ⚠️ --save-hybrid will return high mAP from hybrid labels, not from predictions alone')
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = True  # FP16 for fastest results
        if opt.task == 'speed':  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == 'study':  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'  # filename to save to
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f'\nRunning {f} --imgsz {opt.imgsz}...')
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt='%10.4g')  # save
            os.system('zip -r study.zip study_*.txt')
            plot_val_study(x=x)  # plot


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
