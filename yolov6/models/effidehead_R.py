import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolov6.assigners.anchor_generator import generate_anchors
from yolov6.layers.common import *
from yolov6.utils.general import dist2bbox, dist2Rbbox
from yolov6.utils.v5_rotation import angle_decode

class Detect(nn.Module):
    """Efficient Decoupled Head
    With hardware-aware degisn, the decoupled head is optimized with
    hybridchannels methods.
    """

    def __init__(
        self,
        num_classes=80,
        num_layers=3,
        inplace=True,
        head_layers=None,
        use_dfl=True,
        reg_max=16,
        angle_max=180,
        angle_fitting_methods="regression",
    ):  # detection layer
        super().__init__()
        assert head_layers is not None
        self.nc = num_classes  # number of classes
        # NOTE [x, y, w, h, confidence, angle]
        # NOTE [x, y, w, h, confidence, angle_class, angle_regression]
        # self.no = (
        #     num_classes + 7 if angle_fitting_methods == "MGAR" else num_classes + 6
        # )  # number of outputs per anchor TODO change this
        self.nl = num_layers  # number of detection layers
        self.grid = [torch.zeros(1)] * num_layers
        self.prior_prob = 1e-2
        self.inplace = inplace
        stride = [8, 16, 32] if num_layers == 3 else [8, 16, 32, 64]  # strides computed during build
        self.stride = torch.tensor(stride)
        self.use_dfl = use_dfl
        if angle_fitting_methods == 'MDD':
            self.angle_l = angle_max
            self.angle_max = angle_max[0] * angle_max[1]
        elif angle_fitting_methods == 'MDD+reg':
            self.angle_l = angle_max
            self.angle_max = angle_max[0] * angle_max[1]
        else:
            self.angle_max = angle_max
        self.reg_max = reg_max
        self.angle_fitting_methods = angle_fitting_methods
        self.proj_conv = nn.Conv2d(self.reg_max + 1, 1, 1, bias=False)
        #
        self.proj_angle_conv = nn.Conv2d(self.angle_max + 1, 1, 1, bias=False)
        self.grid_cell_offset = 0.5
        self.grid_cell_size = 5.0

        # Init decouple head
        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.angle_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.angle_preds = nn.ModuleList()

        # Efficient decoupled head layers
        for i in range(num_layers):
            idx = i * 7
            self.stems.append(head_layers[idx])
            self.cls_convs.append(head_layers[idx + 1])
            self.reg_convs.append(head_layers[idx + 2])
            self.angle_convs.append(head_layers[idx + 3])
            self.cls_preds.append(head_layers[idx + 4])
            self.reg_preds.append(head_layers[idx + 5])
            self.angle_preds.append(head_layers[idx + 6])

    def initialize_biases(self):

        for conv in self.cls_preds:
            b = conv.bias.view(
                -1,
            )
            b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.0)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)

        for conv in self.reg_preds:
            b = conv.bias.view(
                -1,
            )
            b.data.fill_(1.0)
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.0)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)

        for conv in self.angle_preds:
            b = conv.bias.view(
                -1,
            )
            if self.angle_fitting_methods != "regression":
                b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
            else:
                b.data.fill_(1.0)
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.0)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)
        # proj和dfl有关 先不管
        self.proj = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max + 1), requires_grad=False)
        self.proj_conv.weight = nn.Parameter(
            self.proj.view([1, self.reg_max + 1, 1, 1]).clone().detach(), requires_grad=False
        )
        if self.angle_fitting_methods == "dfl":
            # TODO, 拟合值可以改变
            self.proj_angle = (180.0 / self.angle_max) * nn.Parameter(
                torch.linspace(0, self.angle_max, self.angle_max + 1), requires_grad=False
            )
            self.proj_angle_conv.weight = nn.parameter.Parameter(
                self.proj_angle.view([1, self.angle_max + 1, 1, 1]).clone().detach(), requires_grad=False
            )

    def forward(self, x):
        # 输出分别为原始特征x、基于anchor free的分类预测和基于anchor free的回归预测
        if self.training:
            # NOTE for training
            cls_score_list = []
            reg_distri_list = []
            angle_fitting_list = []
            for i in range(self.nl):
                x[i] = self.stems[i](x[i])
                cls_x = x[i]
                reg_x = x[i]
                angle_x = x[i]
                cls_feat = self.cls_convs[i](cls_x)
                cls_output = self.cls_preds[i](cls_feat)
                reg_feat = self.reg_convs[i](reg_x)
                reg_output = self.reg_preds[i](reg_feat)
                angle_feat = self.angle_convs[i](angle_x)
                angle_output = self.angle_preds[i](angle_feat)

                if self.angle_fitting_methods == "regression":
                    angle_output = angle_output ** 2
                    # angle_output = torch.pi * torch.sigmoid(angle_output) * (1.05)
                    pass

                # NOTE [BS, C, H, W]
                cls_output = torch.sigmoid(cls_output)
                cls_score_list.append(cls_output.flatten(2).permute((0, 2, 1)))
                reg_distri_list.append(reg_output.flatten(2).permute((0, 2, 1)))
                angle_fitting_list.append(angle_output.flatten(2).permute((0, 2, 1)))

            cls_score_list = torch.cat(cls_score_list, axis=1)
            reg_distri_list = torch.cat(reg_distri_list, axis=1)
            angle_fitting_list = torch.cat(angle_fitting_list, axis=1)

            # print(torch.max(angle_fitting_list), torch.min(angle_fitting_list))

            if torch.onnx.is_in_onnx_export():
                return cls_score_list, reg_distri_list, angle_fitting_list
            else:
                return x, cls_score_list, reg_distri_list, angle_fitting_list
        else:
            # NOTE for eval
            cls_score_list = []
            reg_dist_list = []
            angle_fitting_list = []
            anchor_points, stride_tensor = generate_anchors(
                x, self.stride, self.grid_cell_size, self.grid_cell_offset, device=x[0].device, is_eval=True, mode="af"
            )

            for i in range(self.nl):
                b, _, h, w = x[i].shape
                l = h * w
                x[i] = self.stems[i](x[i])
                cls_x = x[i]
                reg_x = x[i]
                angle_x = x[i]
                cls_feat = self.cls_convs[i](cls_x)
                cls_output = self.cls_preds[i](cls_feat)
                reg_feat = self.reg_convs[i](reg_x)
                reg_output = self.reg_preds[i](reg_feat)
                angle_feat = self.angle_convs[i](angle_x)
                angle_output = self.angle_preds[i](angle_feat)

                cls_output = torch.sigmoid(cls_output)

                if self.use_dfl:
                    reg_output = reg_output.reshape([-1, 4, self.reg_max + 1, l]).permute(0, 2, 1, 3)
                    reg_output = self.proj_conv(F.softmax(reg_output, dim=1))

                # NOTE 角度需要clip下 ## 对于预测时 需要对输出进行二次处理
                if self.angle_fitting_methods == "regression":
                    angle_output = angle_output ** 2
                    # angle_output = torch.pi * torch.sigmoid(angle_output) * (1.05)
                    pass
                elif self.angle_fitting_methods == "dfl":
                    angle_output = angle_output.reshape([-1, 1, self.angle_max + 1, l]).permute(0, 2, 1, 3)
                    angle_output = self.proj_angle_conv(F.softmax(angle_output, dim=1))
                elif self.angle_fitting_methods == "csl":
                    # NOTE sigmoid / softmax
                    # [b, ang, h, w]
                    angle_output = torch.sigmoid(angle_output)
                    # angle_output = torch.softmax(angle_output)
                    angle_output = torch.argmax(angle_output, dim=1, keepdim=True)
                elif self.angle_fitting_methods == "MGAR":
                    ### 输出后的第一次处理,类似我的在val nms前的转换
                    # 直接相加得到结果 torch.Size([2, 1, 76, 104])
                    angle_output_class = torch.sigmoid(angle_output[:, : self.angle_max, :, :])
                    angle_output_class = torch.argmax(angle_output_class, dim=1, keepdim=True) * (360 / self.angle_max)
                    # regression Square
                    angle_output_regression = angle_output[:, -1:, :, :] ** 2
                    angle_output = angle_output_class + angle_output_regression
                elif self.angle_fitting_methods == "MDD":
                    angle_output = angle_decode(self.angle_fitting_methods, self.angle_l, angle_output, "detect_eval")
                elif self.angle_fitting_methods == "MDD+reg":
                    angle_output_class = angle_decode(self.angle_fitting_methods, self.angle_l, angle_output[:, : -1, :, :], "detect_eval")
                    angle_output_regression = angle_output[:, -1:, :, :] ** 2
                    angle_output = angle_output_class + angle_output_regression

                # NOTE clamp 下, 角度==180 算iou会出bug
                cls_score_list.append(cls_output.reshape([b, self.nc, l]))
                reg_dist_list.append(reg_output.reshape([b, 4, l]))
                angle_fitting_list.append(angle_output.reshape([b, 1, l]))

            # [BS, 8400(hw 拉平), n]
            cls_score_list = torch.cat(cls_score_list, axis=-1).permute(0, 2, 1)
            reg_dist_list = torch.cat(reg_dist_list, axis=-1).permute(0, 2, 1)
            angle_fitting_list = torch.cat(angle_fitting_list, axis=-1).permute(0, 2, 1)


            if self.angle_fitting_methods == "regression":
                pred_bboxes = dist2Rbbox(
                    reg_dist_list, angle_fitting_list, anchor_points, box_format="xywh"
                )

                angle_fitting_list = angle_fitting_list * 180 / torch.pi
                # angle_fitting_list = torch.clamp(angle_fitting_list, -90, 90)
                # angle_fitting_list = torch.where(angle_fitting_list <0, angle_fitting_list, angle_fitting_list+180.0)
            else:
                pred_bboxes = dist2bbox(reg_dist_list, anchor_points, box_format="xywh")

            # angle_fitting_list = torch.clamp(angle_fitting_list, 0, 179.99)



            pred_bboxes *= stride_tensor
            # NOTE [BS, 8400, 4+1+1+classes] [x, y, w, h, angle, conf, classes]
            # NOTE 暂时屏蔽angle
            # conf=====1.0
            return torch.cat(
                [
                    pred_bboxes,
                    angle_fitting_list,
                    torch.ones((b, pred_bboxes.shape[1], 1), device=pred_bboxes.device, dtype=pred_bboxes.dtype),
                    cls_score_list,
                ],
                axis=-1,
            )


def build_effidehead_layer(
    channels_list, num_anchors, num_classes, reg_max=16, num_layers=3, angle_fitting_methods="regression", angle_max=180
):

    chx = [6, 8, 10] if num_layers == 3 else [8, 9, 10, 11]

    # REVIEW num_anchors 默认为1, 应该是为了辅助训练, 这个需要考虑下 ###
    # 只有 anchor based没有使用dfl
    if angle_fitting_methods == "regression":
        assert angle_max == 1
    if angle_fitting_methods == "csl":
        assert angle_max == 360
    if angle_fitting_methods == "dfl":
        angle_max += 1
    if angle_fitting_methods == "MGAR":
        assert 180 % angle_max == 0
        angle_max += 1
        # NOTE 最后一个维度用来做regression
        # REVIEW 先不考虑
    if angle_fitting_methods == "MDD":
        assert len(angle_max) == 2
        angle_max = angle_max[0] * angle_max[1]
    if angle_fitting_methods == "MDD+reg":
        assert len(angle_max) == 2
        angle_max = angle_max[0] * angle_max[1] + 1


    head_layers = nn.Sequential(
        # stem0
        Conv(in_channels=channels_list[chx[0]], out_channels=channels_list[chx[0]], kernel_size=1, stride=1),
        # cls_conv0
        Conv(in_channels=channels_list[chx[0]], out_channels=channels_list[chx[0]], kernel_size=3, stride=1),
        # reg_conv0
        Conv(in_channels=channels_list[chx[0]], out_channels=channels_list[chx[0]], kernel_size=3, stride=1),
        # angle_conv0
        Conv(in_channels=channels_list[chx[0]], out_channels=channels_list[chx[0]], kernel_size=3, stride=1),
        # cls_pred0
        nn.Conv2d(in_channels=channels_list[chx[0]], out_channels=num_classes * num_anchors, kernel_size=1),
        # reg_pred0
        nn.Conv2d(in_channels=channels_list[chx[0]], out_channels=4 * (reg_max + num_anchors), kernel_size=1),
        # angle_pred0 ###
        nn.Conv2d(in_channels=channels_list[chx[0]], out_channels=angle_max * num_anchors, kernel_size=1),
        # stem1
        Conv(in_channels=channels_list[chx[1]], out_channels=channels_list[chx[1]], kernel_size=1, stride=1),
        # cls_conv1
        Conv(in_channels=channels_list[chx[1]], out_channels=channels_list[chx[1]], kernel_size=3, stride=1),
        # reg_conv1
        Conv(in_channels=channels_list[chx[1]], out_channels=channels_list[chx[1]], kernel_size=3, stride=1),
        # angle_conv1
        Conv(in_channels=channels_list[chx[1]], out_channels=channels_list[chx[1]], kernel_size=3, stride=1),
        # cls_pred1
        nn.Conv2d(in_channels=channels_list[chx[1]], out_channels=num_classes * num_anchors, kernel_size=1),
        # reg_pred1
        nn.Conv2d(in_channels=channels_list[chx[1]], out_channels=4 * (reg_max + num_anchors), kernel_size=1),
        # angle_pred1
        nn.Conv2d(in_channels=channels_list[chx[1]], out_channels=angle_max * num_anchors, kernel_size=1),
        # stem2
        Conv(in_channels=channels_list[chx[2]], out_channels=channels_list[chx[2]], kernel_size=1, stride=1),
        # cls_conv2
        Conv(in_channels=channels_list[chx[2]], out_channels=channels_list[chx[2]], kernel_size=3, stride=1),
        # reg_conv2
        Conv(in_channels=channels_list[chx[2]], out_channels=channels_list[chx[2]], kernel_size=3, stride=1),
        # angle_conv2
        Conv(in_channels=channels_list[chx[2]], out_channels=channels_list[chx[2]], kernel_size=3, stride=1),
        # cls_pred2
        nn.Conv2d(in_channels=channels_list[chx[2]], out_channels=num_classes * num_anchors, kernel_size=1),
        # reg_pred2
        nn.Conv2d(in_channels=channels_list[chx[2]], out_channels=4 * (reg_max + num_anchors), kernel_size=1),
        # angle_pred2
        nn.Conv2d(in_channels=channels_list[chx[2]], out_channels=angle_max * num_anchors, kernel_size=1),
    )

    if num_layers == 4:
        head_layers.add_module(
            "stem3",
            # stem3
            Conv(in_channels=channels_list[chx[3]], out_channels=channels_list[chx[3]], kernel_size=1, stride=1),
        )
        head_layers.add_module(
            "cls_conv3",
            # cls_conv3
            Conv(in_channels=channels_list[chx[3]], out_channels=channels_list[chx[3]], kernel_size=3, stride=1),
        )
        head_layers.add_module(
            "reg_conv3",
            # reg_conv3
            Conv(in_channels=channels_list[chx[3]], out_channels=channels_list[chx[3]], kernel_size=3, stride=1),
        )
        head_layers.add_module(
            "angle_conv3",
            # angle_conv3
            Conv(in_channels=channels_list[chx[3]], out_channels=channels_list[chx[3]], kernel_size=3, stride=1),
        )
        head_layers.add_module(
            "cls_pred3",
            # cls_pred3
            nn.Conv2d(in_channels=channels_list[chx[3]], out_channels=num_classes * num_anchors, kernel_size=1),
        )
        head_layers.add_module(
            "reg_pred3",
            # reg_pred3
            nn.Conv2d(in_channels=channels_list[chx[3]], out_channels=4 * (reg_max + num_anchors), kernel_size=1),
        )
        head_layers.add_module(
            "angle_pred3",
            # angle_pred3
            nn.Conv2d(in_channels=channels_list[chx[3]], out_channels=angle_max * num_anchors, kernel_size=1),
        )

    return head_layers
