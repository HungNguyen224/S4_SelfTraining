# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------
# Obtained from: mmseg/models/decode_heads/decode_head.py
# Modifications: Simplified for feature projection
# ---------------------------------------------------------------

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import HEADS


@HEADS.register_module()
class ProjectionHead(nn.Module):
    """Projection head for mapping decoder features to prototype space.

    Projects multi-scale decoder features to a fixed-dimensional embedding
    for contrastive learning with class prototypes.
    """

    def __init__(self,
                 in_channels,
                 out_channels=256,
                 mid_channels=None,
                 num_convs=2,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 dropout_ratio=0.0,
                 align_corners=False):
        """Initialize projection head.

        Args:
            in_channels (list[int]): Input channels from decoder levels
            out_channels (int): Output feature dimension for prototypes
            mid_channels (int, optional): Hidden dimension in MLP layers
            num_convs (int): Number of convolution layers
            conv_cfg (dict, optional): Config for conv layers
            norm_cfg (dict, optional): Config for normalization layers
            act_cfg (dict, optional): Config for activation layers
            dropout_ratio (float): Dropout ratio
            align_corners (bool): Align corners for interpolation
        """
        super().__init__()

        if not isinstance(in_channels, list):
            in_channels = [in_channels]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.align_corners = align_corners

        if mid_channels is None:
            mid_channels = out_channels
        
        # TODO: need add attention
        # Build projection layers
        self.convs = nn.ModuleList()
        for i in range(num_convs):
            in_c = in_channels[0] if i == 0 else mid_channels
            self.convs.append(
                ConvModule(
                    in_c,
                    mid_channels,
                    kernel_size=3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        # Final projection to out_channels
        self.conv_out = nn.Conv2d(mid_channels, out_channels, kernel_size=1)

        self.dropout = nn.Dropout2d(dropout_ratio)

    def forward(self, inputs):
        """Forward function.

        Args:
            inputs (list[Tensor]): List of multi-level decoder features

        Returns:
            Tensor: Projected features of shape (N, out_channels, H, W)
        """
        # TODO: consider multil scale
        # Use the highest resolution feature (last in list)
        x = inputs[-1]

        for conv in self.convs:
            x = conv(x)

        x = self.dropout(x)
        output = self.conv_out(x)

        return output

    def forward_train(self, inputs, img_metas=None):
        """Forward function for training.

        Args:
            inputs (list[Tensor]): List of multi-level decoder features
            img_metas (list[dict], optional): Image meta info

        Returns:
            Tensor: Projected features
        """
        return self.forward(inputs)
