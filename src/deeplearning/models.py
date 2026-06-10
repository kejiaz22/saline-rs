# -*- coding: utf-8 -*-
"""改造的 ResNet50: 首层 8 通道, ImageNet 权重 RGB->8 通道扩展。

- conv1: 3->8 通道。前 3 通道 (B2/B3/B4) 直接继承 ImageNet RGB 权重;
  后 5 通道 (B8/B11/B12/VV/VH) 用 RGB 权重均值初始化 (保留低层纹理先验)。
- fc: 1000 -> 2 类。
"""
import torch
import torch.nn as nn
import torchvision.models as tvm
from torchvision.models import ResNet50_Weights


def build_resnet50_8channel() -> nn.Module:
    model = tvm.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    old_conv1 = model.conv1  # (64, 3, 7, 7); ImageNet 通道序 = R/G/B
    new_conv1 = nn.Conv2d(8, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        # Strict RGB-to-band alignment: ImageNet R/G/B -> B4/B3/B2 by physical
        # correspondence (波段序 idx0=B2蓝, idx1=B3绿, idx2=B4红)
        new_conv1.weight[:, 2, :, :] = old_conv1.weight[:, 0, :, :]  # R -> B4 (idx2)
        new_conv1.weight[:, 1, :, :] = old_conv1.weight[:, 1, :, :]  # G -> B3 (idx1)
        new_conv1.weight[:, 0, :, :] = old_conv1.weight[:, 2, :, :]  # B -> B2 (idx0)
        # 其余 5 通道 (B8/B11/B12/VV/VH) 用 RGB 权重均值初始化
        rgb_mean = old_conv1.weight.mean(dim=1, keepdim=True)
        for i in range(3, 8):
            new_conv1.weight[:, i:i + 1, :, :] = rgb_mean
    model.conv1 = new_conv1

    # 2 类输出
    model.fc = nn.Linear(2048, 2)
    return model
