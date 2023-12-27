# Model validation metrics
# This code is based on
# https://github.com/ultralytics/yolov5/blob/master/utils/metrics.py

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from yolov6.utils.nms_R import obb_box_iou, obb_box_iou_cuda
from yolov6.utils.v5_rotation import head_iou

# import warnings
from . import general


def ap_per_class(tp, conf, pred_cls, target_cls, head_correct, tnc, plot=False, save_dir=".", names=(), ap_method="VOC12"):
    """Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections
    # for head
    epochs = int(head_correct.shape[0] / tnc)
    head_output = np.zeros((nc, 2, 10))
    head_detected = np.zeros_like(head_output) + 1e-8
    head_acc_l = []
    head_loss_l = []

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j], ap_method=ap_method)
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

            # Head
            head_acc_t = []
            head_loss_t = []
            for e in range(epochs):
                head_output[ci, :] = head_output[ci, :] + head_correct[ci + nc * e, :]
                # count ci class , how many > 0 values
                for c1 in range(head_correct.shape[1]):
                    for c2 in range(head_correct.shape[2]):
                        if head_correct[ci + nc * e, c1, c2] > 0:
                            head_detected[ci, c1, c2] += 1
                # plot head_acc for map50
                if head_correct[nc * e, 0, 0] > 0:
                    head_acc_t.append(head_correct[nc * e, 0, 0])
                # plot head_loss for map50
                if head_correct[nc * e, 1, 0] > 0:
                    head_loss_t.append(head_correct[nc * e, 1, 0])
            if len(head_acc_t) > 0:
                head_acc = np.array(head_acc_t)
                head_acc_1000 = np.interp(np.linspace(0, len(head_acc) - 1, 1000), np.arange(len(head_acc)), head_acc)
                head_acc_l.append(head_acc_1000)
            if len(head_loss_t) > 0:
                head_loss = np.array(head_loss_t)
                head_loss_1000 = np.interp(np.linspace(0, len(head_loss) - 1, 1000), np.arange(len(head_loss)), head_loss)
                head_loss_l.append(head_loss_1000)

    # hcc
    head_output = head_output / head_detected
    head_acc = np.array(head_acc_l)
    head_loss = np.array(head_loss_l)
    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / "PR_curve.png", names)
        plot_mc_curve(px, f1, Path(save_dir) / "F1_curve.png", names, ylabel="F1")
        plot_mc_curve(px, p, Path(save_dir) / "P_curve.png", names, ylabel="Precision")
        plot_mc_curve(px, r, Path(save_dir) / "R_curve.png", names, ylabel="Recall")
        if len(head_acc) > 1:
            plot_mc_curve(px, head_acc, Path(save_dir) / 'Head_acc_curve.png', names, ylabel='head_acc')
        if len(head_loss) > 1:
            plot_mc_curve(px, head_loss, Path(save_dir) / 'Head_loss_curve.png', names, ylabel='head_loss')
    # i = f1.mean(0).argmax()  # max F1 index
    # return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')
    return p, r, ap, f1, unique_classes.astype("int32"), head_output


def compute_ap(recall, precision, ap_method="VOC12"):
    """Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [recall[-1] + 0.01]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    # NOTE: continous: VOC12， interp： COCO
    if ap_method == "COCO" or ap_method == "coco" or ap_method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    elif ap_method == "VOC12" or ap_method == "voc12":  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve
    else:
        # NOTE VOC07
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap = ap + p / 11.0
    return ap, mpre, mrec


# Plots ----------------------------------------------------------------------------------------------------------------


def plot_pr_curve(px, py, ap, save_dir="pr_curve.png", names=()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}")  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color="grey")  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color="blue", label="all classes %.3f mAP@0.5" % ap[:, 0].mean())
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)


def plot_mc_curve(px, py, save_dir="mc_curve.png", names=(), xlabel="Confidence", ylabel="Metric"):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f"{names[i]}")  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color="grey")  # plot(confidence, metric)

    y = py.mean(0)
    ax.plot(px, y, linewidth=3, color="blue", label=f"all classes {y.max():.2f} at {px[y.argmax()]:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 7]), x1, y1, x2, y2, angle, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2, angle
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    # iou = general.box_iou(labels[:, 1:], detections[:, :4])
    # iou = obb_box_iou(labels[:, 1:].cpu().numpy(), detections[:, :5].cpu().numpy())
    # iou = torch.from_numpy(iou)
    iou = obb_box_iou_cuda(labels[:, 1:], detections[:, :5])

    correct_class = labels[:, 0:1] == detections[:, 6]
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


def process_batch_rbox(detections, labels, iouv, hccv, nc): # hccv没用
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 7]), x, y, w, h, angle, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2, angle
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    head_correct = torch.zeros((nc, 2, hccv.shape[0])).to(detections.device)
    # iou = general.box_iou(labels[:, 1:], detections[:, :4])
    # iou = obb_box_iou(labels[:, 1:].cpu().numpy(), detections[:, :5].cpu().numpy())
    # iou = torch.from_numpy(iou)
    iou = obb_box_iou_cuda(labels[:, 1:], detections[:, :5])
    head_true, head_diff = head_iou(detections[:, 4], labels[:, 5])
    correct_class = labels[:, 0:1] == detections[:, 6]
    for i in range(len(iouv)): # 10个iou尺度
        # 只有同时符合以上两点条件才被赋值为True，此时返回当前矩阵的一个行列索引，x是两个元祖x1,x2
        # 点(x[0][i], x[1][i])就是 class   符合条件的预测框
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        head_x = torch.where((iou >= 0.01) & correct_class & head_true)
        head_sum_x = torch.where((iou >= 0.01) & correct_class)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            head_matches = torch.cat((torch.stack(head_x, 1), iou[head_x[0], head_x[1]][:, None]), 1).cpu().numpy() # iou & hcc
            head_sum_matches = torch.cat((torch.stack(head_sum_x, 1), iou[head_sum_x[0], head_sum_x[1]][:, None]), 1).cpu().numpy() # only iou
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                ### hcc for every class
                # 每一列为 : 在labels中索引    在detection的索引   iou值   对应的类别    角度误差
                head_matches = head_matches[head_matches[:, 2].argsort()[::-1]]
                head_sum_matches = head_sum_matches[head_sum_matches[:, 2].argsort()[::-1]]
                # 1对1
                head_matches = head_matches[np.unique(head_matches[:, 1], return_index=True)[1]]
                head_matches = head_matches[np.unique(head_matches[:, 0], return_index=True)[1]]
                head_sum_matches = head_sum_matches[np.unique(head_sum_matches[:, 1], return_index=True)[1]]
                head_sum_matches = head_sum_matches[np.unique(head_sum_matches[:, 0], return_index=True)[1]]
                # 增加类别列
                head_matches = np.concatenate((head_matches, np.expand_dims(labels[head_matches[:, 0], 0], axis=1)),axis=1)
                head_sum_matches = np.concatenate((head_sum_matches, np.expand_dims(labels[head_sum_matches[:, 0], 0], axis=1)),
                                              axis=1)
                # add head loss
                head_sum_matches = np.concatenate(
                    (head_sum_matches, np.expand_dims(head_diff[head_sum_matches[:,0],head_sum_matches[:,1]], axis=1)),
                    axis=1)
                for per_class in range(nc):
                    # head acc
                    head_true_sum = head_matches[head_matches[:, 3] == per_class].shape[0]
                    head_sum = head_sum_matches[head_sum_matches[:, 3] == per_class].shape[0]
                    if head_sum > 0:
                        head_correct[per_class, 0, i] = head_true_sum / (head_sum + 1e-6)
                    # head loss
                    head_loss = head_sum_matches[head_sum_matches[:, 3] == per_class][:, 4]
                    if len(head_loss) > 0:
                        head_correct[per_class, 1, i] = torch.FloatTensor(np.array([head_loss.mean()]))

                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device), head_correct
class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 7]), x1, y1, x2, y2, angle, conf, class
            labels (Array[M, 6]), class, x1, y1, x2, y2, angle
        Returns:
            None, updates confusion matrix accordingly
        """
        detections = detections[detections[:, 5] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 6].int()
        # TODO FIXME
        # iou = general.box_iou(labels[:, 1:], detections[:, :4])

        # iou = obb_box_iou(labels[:, 1:].cpu().numpy(), detections[:, :5].cpu().numpy())
        # iou = torch.from_numpy(iou)
        iou = obb_box_iou_cuda(labels[:, 1:], detections[:, :5])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # background FP

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # background FN

    def matrix(self):
        return self.matrix

    def tp_fp(self):
        tp = self.matrix.diagonal()  # true positives
        fp = self.matrix.sum(1) - tp  # false positives
        # fn = self.matrix.sum(0) - tp  # false negatives (missed detections)
        return tp[:-1], fp[:-1]  # remove background class

    def plot(self, normalize=True, save_dir="", names=()):
        try:
            import seaborn as sn

            array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1e-9) if normalize else 1)  # normalize columns
            array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            nc, nn = self.nc, len(names)  # number of classes, names
            sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size
            labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
                sn.heatmap(
                    array,
                    annot=nc < 30,
                    annot_kws={"size": 8},
                    cmap="Blues",
                    fmt=".2f",
                    square=True,
                    vmin=0.0,
                    xticklabels=names + ["background FP"] if labels else "auto",
                    yticklabels=names + ["background FN"] if labels else "auto",
                ).set_facecolor((1, 1, 1))
            fig.axes[0].set_xlabel("True")
            fig.axes[0].set_ylabel("Predicted")
            fig.savefig(Path(save_dir) / "confusion_matrix.png", dpi=250)
            plt.close()
        except Exception as e:
            print(f"WARNING: ConfusionMatrix plot failure: {e}")

    def print(self):
        for i in range(self.nc + 1):
            print(" ".join(map(str, self.matrix[i])))
