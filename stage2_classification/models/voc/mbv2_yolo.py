"""
MobileNetV2 YOLO Architecture
"""

import torch
import torch.nn as nn
from torch.nn import init
import torchvision

from models.voc.mobilenetv2 import mobilenetv2
from models.voc.yolo_loss import YOLOLoss


class BasicConv(nn.Module):
    """Standard Convolution block with Batch Normalization and ReLU."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, depthwise: bool = False):
        super().__init__()
        groups = in_channels if depthwise else 1
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, 
            padding=kernel_size // 2, bias=False, groups=groups
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.bn(self.conv(x)))

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class Upsample(nn.Module):
    """Upsamples spatial dimensions by a factor of 2."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample(x)


def DepthwiseConvolution(in_filters: int, out_filters: int) -> nn.Module:
    return nn.Sequential(
        BasicConv(in_filters, in_filters, 3, depthwise=True),
        BasicConv(in_filters, in_filters, 1),
        BasicConv(in_filters, out_filters, 1),
    )


def yolo_head(filters_list: list, in_filters: int) -> nn.Module:
    return nn.Sequential(
        BasicConv(in_filters, in_filters, 3, depthwise=True),
        BasicConv(in_filters, in_filters, 1),
        BasicConv(in_filters, filters_list[0], 1),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )


class Connect(nn.Module):
    """Feature connection block with residual addition."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            BasicConv(channels, channels, 3, depthwise=True),
            BasicConv(channels, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)


class yolo(nn.Module):
    """Main MobileNetV2-YOLO architecture definition."""
    def __init__(self, config: dict):
        super().__init__()
        self.num_classes = config["yolo"]["num_classes"]
        self.num_anchors = config["yolo"]["num_anchors"]
        
        # Backbone
        model_url = 'https://raw.githubusercontent.com/d-li14/mobilenetv2.pytorch/master/pretrained/mobilenetv2-c5e733a8.pth'
        self.backbone = mobilenetv2(model_url)

        self.conv_for_S32 = BasicConv(1280, 512, 1)
        self.connect_for_S32 = Connect(512)
        self.yolo_headS32 = yolo_head([1024, self.num_anchors * (5 + self.num_classes)], 512)
        
        self.upsample = Upsample(512, 256)
        self.conv_for_S16 = DepthwiseConvolution(96, 256)
        self.connect_for_S16 = Connect(256)
        self.yolo_headS16 = yolo_head([512, self.num_anchors * (5 + self.num_classes)], 256)

        self.yolo_losses = nn.ModuleList([
            YOLOLoss(
                config["yolo"]["anchors"], config["yolo"]["mask"][i], self.num_classes, 
                [config["img_w"], config["img_h"]], config["yolo"]["iou_thres"][i]
            ) for i in range(2)
        ])

    def nms(self, preds: list) -> list:
        """Applies Non-Maximum Suppression to predicted bounding boxes."""
        nms_preds = []
        assert len(preds) == 2 
        assert len(preds[0]) == len(preds[1])
        bs = len(preds[0])
        device = preds[0][0].device

        for b in range(bs):
            pred_per_img = torch.cat((preds[0][b], preds[1][b]), 0)
            pred_boxes = torch.zeros(0, 7, requires_grad=False, device=device)
            
            if pred_per_img.size(0):
                for i in range(self.num_classes):                       
                    mask = (pred_per_img[..., 6] == i)                    
                    pred_this_cls = pred_per_img[mask]
                    
                    if pred_this_cls.size(0):
                        boxes = pred_this_cls[..., :4]
                        scores = pred_this_cls[..., 5] * pred_this_cls[..., 4]
                        index = torchvision.ops.nms(boxes, scores, 0.45)            
                        pred_boxes = torch.cat((pred_boxes, pred_this_cls[index]), 0)
                        
            nms_preds.append(pred_boxes)
        return nms_preds        

    def forward(self, x: torch.Tensor, targets: torch.Tensor = None):
        for i in range(2):
            self.yolo_losses[i].img_size = [x.size(2), x.size(3)]
            
        feature1, feature2 = self.backbone(x)
        
        S32 = self.conv_for_S32(feature2)
        S32 = self.connect_for_S32(S32)
        out0 = self.yolo_headS32(S32) 
        
        S32_Upsample = self.upsample(S32)
        S16 = self.conv_for_S16(feature1)
        S16 = S16 + S32_Upsample
        S16 = self.connect_for_S16(S16)
        out1 = self.yolo_headS16(S16)
        
        output = self.yolo_losses[0](out0, targets), self.yolo_losses[1](out1, targets)
        
        if targets is None:
            output = self.nms(output)
            
        return output