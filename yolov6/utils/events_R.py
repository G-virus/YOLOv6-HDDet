#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import yaml
import logging
import shutil
import cv2
from rich.logging import RichHandler
class OBRichHandler(RichHandler):
    # Note KEYWORDS highlight use regex
    KEYWORDS = [
        "mAP",
        "Epoch",
        "iou_loss",
        "cls_loss",
        "dfl_loss",
        "ang_loss",
        "distill_loss",
    ]

class RichLogger(logging.Logger):
    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)
        self.propagate = False
        self._set_rich_logger()

    def _set_rich_logger(self):
        console_handler = OBRichHandler(
            rich_tracebacks=True, tracebacks_show_locals=True, log_time_format="[%m-%d %H:%M]"
        )
        rich_formatter = logging.Formatter("%(message)s")
        console_handler.setFormatter(rich_formatter)
        self.addHandler(console_handler)

    def set_file_path(self, file_path):
        log_dir = os.path.dirname(file_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        handler = logging.FileHandler(file_path)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.addHandler(handler)


def set_logging(name=None):
    rank = int(os.getenv("RANK", -1))
    logging.basicConfig(format="%(message)s", level=logging.INFO if (rank in (-1, 0)) else logging.WARNING)
    return logging.getLogger(name)


# LOGGER = set_logging(__name__)
LOGGER = RichLogger("YOLOv6-R")
NCOLS = min(100, shutil.get_terminal_size().columns)


def load_yaml(file_path):
    """Load data from yaml file."""
    if isinstance(file_path, str):
        with open(file_path, errors="ignore") as f:
            data_dict = yaml.safe_load(f)
    return data_dict


def save_yaml(data_dict, save_path):
    """Save data to yaml file"""
    with open(save_path, "w") as f:
        yaml.safe_dump(data_dict, f, sort_keys=False)


def write_tblog(tblogger, epoch, results, losses):
    """Display mAP and loss information to log."""
    tblogger.add_scalar("val/mAP@0.5", results[0], epoch + 1)
    tblogger.add_scalar("val/mAP@0.50:0.95", results[1], epoch + 1)
    tblogger.add_scalar("val/Head_acc", results[2], epoch + 1)
    tblogger.add_scalar("val/Head_loss", results[3], epoch + 1)

    tblogger.add_scalar("train/iou_loss", losses[0], epoch + 1)
    tblogger.add_scalar("train/dist_focalloss", losses[1], epoch + 1)
    tblogger.add_scalar("train/cls_loss", losses[2], epoch + 1)
    tblogger.add_scalar("train/angle_loss", losses[3], epoch + 1)

    tblogger.add_scalar("x/lr0", results[4], epoch + 1)
    tblogger.add_scalar("x/lr1", results[5], epoch + 1)
    tblogger.add_scalar("x/lr2", results[6], epoch + 1)


def write_tbimg(tblogger, imgs, step, type="train"):
    """Display train_batch and validation predictions to tensorboard."""
    if type == "train":
        tblogger.add_image(f"train_batch", imgs, step + 1, dataformats="HWC")
        # TODO

    elif type == "val":
        for idx, img in enumerate(imgs):
            tblogger.add_image(f"val_img_{idx + 1}", img, step + 1, dataformats="HWC")
            # TODO

    else:
        LOGGER.warning("WARNING: Unknown image type to visualize.\n")
