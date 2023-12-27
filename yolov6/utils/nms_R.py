#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# The code is based on
# https://github.com/ultralytics/yolov5/blob/master/utils/general.py

import os
import time

import cv2
import numpy as np
import torch
import torchvision
from mmcv.ops import box_iou_rotated, nms_rotated

# Settings
torch.set_printoptions(linewidth=320, precision=5, profile="long")
np.set_printoptions(linewidth=320, formatter={"float_kind": "{:11.5g}".format})  # format short g, %precision=5
cv2.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
os.environ["NUMEXPR_MAX_THREADS"] = str(min(os.cpu_count(), 8))  # NumExpr max threads


def obb_box_iou(boxes1, boxes2):
    area1 = boxes1[:, 2] * boxes1[:, 3]
    area2 = boxes2[:, 2] * boxes2[:, 3]
    ious = []
    for i, box1 in enumerate(boxes1):
        temp_ious = []
        r1 = ((box1[0], box1[1]), (box1[2], box1[3]), box1[4])
        for j, box2 in enumerate(boxes2):
            r2 = ((box2[0], box2[1]), (box2[2], box2[3]), box2[4])
            int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
            if int_pts is not None:
                order_pts = cv2.convexHull(int_pts, returnPoints=True)

                int_area = cv2.contourArea(order_pts)

                inter = int_area * 1.0 / (area1[i] + area2[j] - int_area)
                temp_ious.append(inter)
            else:
                temp_ious.append(0.0)
        ious.append(temp_ious)
    return np.array(ious, dtype=np.float32)


def obb_box_iou_cuda(boxes1, boxes2):
    box1 = boxes1.clone()
    box2 = boxes2.clone()
    box1[:, -1] = box1[:, -1] * torch.pi / 180.0
    box2[:, -1] = box2[:, -1] * torch.pi / 180.0
    return box_iou_rotated(box1, box2, mode="iou")


def xywh2xyxy(x):
    """Convert boxes with shape [n, 4] from [x, y, w, h] to [x1, y1, x2, y2] where x1y1 is top-left, x2y2=bottom-right."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def non_max_suppression(
    prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, max_det=300
):
    """Runs Non-Maximum Suppression (NMS) on inference results.
    This code is borrowed from: https://github.com/ultralytics/yolov5/blob/47233e1698b89fc437a4fb9463c815e9171be955/utils/general.py#L775
    Args:
        prediction: (tensor), with shape [N, 5 + num_classes], N is the number of bboxes.
        conf_thres: (float) confidence threshold.
        iou_thres: (float) iou threshold.
        classes: (None or list[int]), if a list is provided, nms only keep the classes you provide.
        agnostic: (bool), when it is set to True, we do class-independent nms, otherwise, different class would do nms respectively.
        multi_label: (bool), when it is set to True, one box can have multi labels, otherwise, one box only huave one label.
        max_det:(int), max number of output bboxes.

    Returns:
         list of detections, echo item is one tensor with shape (num_boxes, 6), 6 is for [xyxy, conf, cls].
    """

    num_classes = prediction.shape[2] - 5  # number of classes
    pred_candidates = torch.logical_and(
        prediction[..., 4] > conf_thres, torch.max(prediction[..., 5:], axis=-1)[0] > conf_thres
    )  # candidates
    # Check the parameters.
    assert 0 <= conf_thres <= 1, f"conf_thresh must be in 0.0 to 1.0, however {conf_thres} is provided."
    assert 0 <= iou_thres <= 1, f"iou_thres must be in 0.0 to 1.0, however {iou_thres} is provided."

    # Function settings.
    max_wh = 4096  # maximum box width and height
    max_nms = 30000  # maximum number of boxes put into torchvision.ops.nms()
    time_limit = 10.0  # quit the function when nms cost time exceed the limit time.
    multi_label &= num_classes > 1  # multiple labels per box

    tik = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for img_idx, x in enumerate(prediction):  # image index, image inference
        x = x[pred_candidates[img_idx]]  # confidence

        # If no box remains, skip the next process.
        if not x.shape[0]:
            continue

        # confidence multiply the objectness
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix's shape is  (n,6), each row represents (xyxy, conf, cls)
        if multi_label:
            box_idx, class_idx = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[box_idx], x[box_idx, class_idx + 5, None], class_idx[:, None].float()), 1)
        else:  # Only keep the class with highest scores.
            conf, class_idx = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, class_idx.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class, only keep boxes whose category is in classes.
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        num_box = x.shape[0]  # number of boxes
        if not num_box:  # no boxes kept.
            continue
        elif num_box > max_nms:  # excess max boxes' number.
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        class_offset = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + class_offset, x[:, 4]  # boxes (offset by class), scores
        keep_box_idx = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if keep_box_idx.shape[0] > max_det:  # limit detections
            keep_box_idx = keep_box_idx[:max_det]

        output[img_idx] = x[keep_box_idx]
        if (time.time() - tik) > time_limit:
            print(f"WARNING: NMS cost time exceed the limited {time_limit}s.")
            break  # time limit exceeded

    return output


def non_max_suppression_obb(
    prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, max_det=300
):
    """Runs Non-Maximum Suppression (NMS) on inference results.
    This code is borrowed from: https://github.com/ultralytics/yolov5/blob/47233e1698b89fc437a4fb9463c815e9171be955/utils/general.py#L775
    Args:
        prediction: (tensor), with shape [N, 6 + num_classes], N is the number of bboxes.
        conf_thres: (float) confidence threshold.
        iou_thres: (float) iou threshold.
        classes: (None or list[int]), if a list is provided, nms only keep the classes you provide.
        agnostic: (bool), when it is set to True, we do class-independent nms, otherwise, different class would do nms respectively.
        multi_label: (bool), when it is set to True, one box can have multi labels, otherwise, one box only huave one label.
        max_det:(int), max number of output bboxes.

    Returns:
         list of detections, echo item is one tensor with shape (num_boxes, 6), 6 is for [xywh, conf, cls].
    """

    # NOTE [N, x, y, w, h, angle, conf, classes]
    # NOTE [N, 0, 1, 2, 3, 4,      5,      6:]
    num_classes = prediction.shape[2] - 6  # number of classes
    pred_candidates = torch.logical_and(
        prediction[..., 5] > conf_thres, torch.max(prediction[..., 6:], axis=-1)[0] > conf_thres
    )  # candidates
    # Check the parameters.
    assert 0 <= conf_thres <= 1, f"conf_thresh must be in 0.0 to 1.0, however {conf_thres} is provided."
    assert 0 <= iou_thres <= 1, f"iou_thres must be in 0.0 to 1.0, however {iou_thres} is provided."

    # Function settings.
    max_wh = 4096  # maximum box width and height
    max_nms = 30000  # maximum number of boxes put into torchvision.ops.nms()
    time_limit = 100.0  # quit the function when nms cost time exceed the limit time.
    multi_label &= num_classes > 1  # multiple labels per box

    tik = time.time()
    output = [torch.zeros((0, 7), device=prediction.device)] * prediction.shape[0]
    for img_idx, x in enumerate(prediction):  # image index, image inference
        x = x[pred_candidates[img_idx]]  # confidence

        # If no box remains, skip the next process.
        if not x.shape[0]:
            continue

        # confidence multiply the objectness
        x[:, 6:] *= x[:, 5:6]  # conf = obj_conf * cls_conf

        # (center x, center y, width, height)
        box = x[:, :4]
        angle = x[:, 4:5].clone()

        # Detections matrix's shape is  (n,6), each row represents (xyxy, conf, cls)
        if multi_label:
            box_idx, class_idx = (x[:, 6:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat(
                (box[box_idx], angle[box_idx], x[box_idx, class_idx + 6, None], class_idx[:, None].float()), 1
            )
        else:  # Only keep the class with highest scores.
            conf, class_idx = x[:, 6:].max(1, keepdim=True)
            x = torch.cat((box, angle, conf, class_idx.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class, only keep boxes whose category is in classes.
        if classes is not None:
            x = x[(x[:, 6:7] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        num_box = x.shape[0]  # number of boxes
        if not num_box:  # no boxes kept.
            continue
        elif num_box > max_nms:  # excess max boxes' number.
            x = x[x[:, 5].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        class_offset = x[:, 6:7] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4], x[:, 5]  # boxes (offset by class), scores

        boxes_xywh = boxes.clone()
        boxes_xy = (boxes_xywh[:, :2].clone() + class_offset).cpu().numpy()
        boxes_wh = boxes_xywh[:, 2:4].clone().cpu().numpy()
        boxes_angle = x[:, 4].clone().cpu().numpy()
        scores_for_cv2_nms = scores.cpu().numpy()
        boxes_for_cv2_nms = []

        for box_inds, _ in enumerate(boxes_xy):
            boxes_for_cv2_nms.append((boxes_xy[box_inds], boxes_wh[box_inds], boxes_angle[box_inds]))

        # * Rotated NMS
        keep_box_idx = cv2.dnn.NMSBoxesRotated(boxes_for_cv2_nms, scores_for_cv2_nms, conf_thres, iou_thres)
        keep_box_idx = torch.from_numpy(keep_box_idx).type(torch.LongTensor)
        keep_box_idx = keep_box_idx.squeeze(axis=-1)

        # keep_box_idx = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS

        # TODO bug, 有概率 keep_box_idx 根据iou_thres 出来为空
        try:
            if keep_box_idx.shape[0] > max_det:  # limit detections
                keep_box_idx = keep_box_idx[:max_det]
        except:
            keep_box_idx = keep_box_idx.unsqueeze(dim=-1)
            pass

        output[img_idx] = x[keep_box_idx]

        if (time.time() - tik) > time_limit:
            print(f"WARNING: NMS cost time exceed the limited {time_limit}s.")
            break  # time limit exceeded

    return output


def non_max_suppression_obb_cuda(
    prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, max_det=2000
):
    """Runs Non-Maximum Suppression (NMS) on inference results.
    This code is borrowed from: https://github.com/ultralytics/yolov5/blob/47233e1698b89fc437a4fb9463c815e9171be955/utils/general.py#L775
    Args:
        prediction: (tensor), with shape [N, 6 + num_classes], N is the number of bboxes.
        conf_thres: (float) confidence threshold.
        iou_thres: (float) iou threshold.
        classes: (None or list[int]), if a list is provided, nms only keep the classes you provide.
        agnostic: (bool), when it is set to True, we do class-independent nms, otherwise, different class would do nms respectively.
        multi_label: (bool), when it is set to True, one box can have multi labels, otherwise, one box only huave one label.
        max_det:(int), max number of output bboxes.

    Returns:
         list of detections, echo item is one tensor with shape (num_boxes, 6), 6 is for [xywh, conf, cls].
    """

    # NOTE [N, x, y, w, h, angle, conf, classes]
    # NOTE [N, 0, 1, 2, 3, 4,      5,      6:]
    num_classes = prediction.shape[2] - 6  # number of classes
    pred_candidates = torch.logical_and(
        prediction[..., 5] > conf_thres, torch.max(prediction[..., 6:], axis=-1)[0] > conf_thres
    )  # candidates
    # Check the parameters.
    assert 0 <= conf_thres <= 1, f"conf_thresh must be in 0.0 to 1.0, however {conf_thres} is provided."
    assert 0 <= iou_thres <= 1, f"iou_thres must be in 0.0 to 1.0, however {iou_thres} is provided."

    # Function settings.
    max_wh = 7680  # maximum box width and height
    max_nms = 30000  # maximum number of boxes put into torchvision.ops.nms()
    time_limit = 100.0  # quit the function when nms cost time exceed the limit time.
    multi_label &= num_classes > 1  # multiple labels per box

    tik = time.time()
    output = [torch.zeros((0, 7), device=prediction.device)] * prediction.shape[0]
    for img_idx, x in enumerate(prediction):  # image index, image inference
        x = x[pred_candidates[img_idx]]  # confidence

        # If no box remains, skip the next process.
        if not x.shape[0]:
            continue

        # confidence multiply the objectness
        x[:, 6:] *= x[:, 5:6]  # conf = obj_conf * cls_conf

        # (center x, center y, width, height)
        box = x[:, :4]
        angle = x[:, 4:5].clone()

        # Detections matrix's shape is  (n,6), each row represents (xyxy, conf, cls)
        if multi_label:
            box_idx, class_idx = (x[:, 6:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat(
                (box[box_idx], angle[box_idx], x[box_idx, class_idx + 6, None], class_idx[:, None].float()), 1
            )
        else:  # Only keep the class with highest scores.
            conf, class_idx = x[:, 6:].max(1, keepdim=True)
            x = torch.cat((box, angle, conf, class_idx.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class, only keep boxes whose category is in classes.
        if classes is not None:
            x = x[(x[:, 6:7] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        num_box = x.shape[0]  # number of boxes
        if not num_box:  # no boxes kept.
            continue
        elif num_box > max_nms:  # excess max boxes' number.
            x = x[x[:, 5].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        class_offset = x[:, 6:7] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :5].clone(), x[:, 5]  # boxes (offset by class), scores

        boxes[:, :2] += class_offset
        boxes[:, -1:] = boxes[:, -1:] * torch.pi / 180.0
        # * Rotated NMS
        _, keep_box_idx = nms_rotated(boxes, scores, iou_thres)

        # TODO bug, 有概率 keep_box_idx 根据iou_thres 出来为空
        try:
            if keep_box_idx.shape[0] > max_det:  # limit detections
                keep_box_idx = keep_box_idx[:max_det]
        except:
            keep_box_idx = keep_box_idx.unsqueeze(dim=-1)
            pass

        output[img_idx] = x[keep_box_idx]

        if (time.time() - tik) > time_limit:
            print(f"WARNING: NMS cost time exceed the limited {time_limit}s.")
            break  # time limit exceeded

    return output


def rbox2poly(obboxes):
    """
    Trans rbox format to poly format.
    Args:
        rboxes (array/tensor): (num_gts, [cx cy longSide shortSide theta]) theta∈[0, 180)
    Returns:
        polys (array/tensor): (num_gts, [x1 y1 x2 y2 x3 y3 x4 y4])
    """

    if isinstance(obboxes, torch.Tensor):
        center, longSide, shortSide, theta = obboxes[..., :2], obboxes[..., 2:3], obboxes[..., 3:4], obboxes[..., 4:5]
        Cos, Sin = torch.cos(theta * torch.pi / 180.0), torch.sin(theta * torch.pi / 180.0)

        vector1 = torch.cat((longSide / 2.0 * Cos, longSide / 2.0 * Sin), dim=-1)
        vector2 = torch.cat((shortSide / 2.0 * Sin, -shortSide / 2.0 * Cos), dim=-1)
        point1 = center - vector1 - vector2
        point2 = center - vector1 + vector2
        point3 = center + vector1 + vector2
        point4 = center + vector1 - vector2
        order = obboxes.shape[:-1]
        return torch.cat((point1, point2, point3, point4), dim=-1).reshape(*order, 8)
    else:
        center, longSide, shortSide, theta = np.split(obboxes, (2, 3, 4), axis=-1)
        Cos, Sin = np.cos(theta * np.pi / 180.0), np.sin(theta * torch.pi / 180.0)

        vector1 = np.concatenate([longSide / 2.0 * Cos, longSide / 2.0 * Sin], axis=-1)
        vector2 = np.concatenate([shortSide / 2.0 * Sin, -shortSide / 2.0 * Cos], axis=-1)

        point1 = center - vector1 - vector2
        point2 = center - vector1 + vector2
        point3 = center + vector1 + vector2
        point4 = center + vector1 - vector2
        order = obboxes.shape[:-1]
        return np.concatenate([point1, point2, point3, point4], axis=-1).reshape(*order, 8)


def rbox2poly_radius(obboxes):
    """
    Trans rbox format to poly format.
    Args:
        rboxes (array/tensor): (num_gts, [cx cy longSide shortSide theta]) theta∈[0, 180)
    Returns:
        polys (array/tensor): (num_gts, [x1 y1 x2 y2 x3 y3 x4 y4])
    """

    if isinstance(obboxes, torch.Tensor):
        center, longSide, shortSide, theta = obboxes[..., :2], obboxes[..., 2:3], obboxes[..., 3:4], obboxes[..., 4:5]
        Cos, Sin = torch.cos(theta), torch.sin(theta)

        vector1 = torch.cat((longSide / 2.0 * Cos, longSide / 2.0 * Sin), dim=-1)
        vector2 = torch.cat((shortSide / 2.0 * Sin, -shortSide / 2.0 * Cos), dim=-1)
        point1 = center - vector1 - vector2
        point2 = center - vector1 + vector2
        point3 = center + vector1 + vector2
        point4 = center + vector1 - vector2
        order = obboxes.shape[:-1]
        return torch.cat((point1, point2, point3, point4), dim=-1).reshape(*order, 8)
    else:
        center, longSide, shortSide, theta = np.split(obboxes, (2, 3, 4), axis=-1)
        Cos, Sin = np.cos(theta), np.sin(theta)

        vector1 = np.concatenate([longSide / 2.0 * Cos, longSide / 2.0 * Sin], axis=-1)
        vector2 = np.concatenate([shortSide / 2.0 * Sin, -shortSide / 2.0 * Cos], axis=-1)

        point1 = center - vector1 - vector2
        point2 = center - vector1 + vector2
        point3 = center + vector1 + vector2
        point4 = center + vector1 - vector2
        order = obboxes.shape[:-1]
        return np.concatenate([point1, point2, point3, point4], axis=-1).reshape(*order, 8)
