""" Adapted from the original implementation. """
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.ops import build_conv_layer, build_norm_layer
from mmdet.utils import get_root_logger
from ..registry import BACKBONES
from .resnet import BasicBlock, Bottleneck

import collections
import dataclasses
from typing import List

import torch



class VoVNetParams:
    stem_out: int
    stage_conv_ch: List[int]  # Channel depth of
    stage_out_ch: List[int]  # The channel depth of the concatenated output
    layer_per_block: int
    block_per_stage: List[int]
    dw: bool


_STAGE_SPECS = {
    "vovnet-19-slim-dw1": [64, [64, 80, 96, 112], [56, 128, 192, 256], 1, [1, 1, 1, 1], True],
    "vovnet-19-slim-dw": [64, [64, 80, 96, 112], [112, 256, 384, 512], 3, [1, 1, 1, 1], True],
    "vovnet-19-slim": [128, [64, 80, 96, 112], [112, 256, 384, 512], 3, [1, 1, 1, 1], False],
    "vovnet-19": [128, [128, 160, 192, 224], [256, 512, 768, 1024], 3, [1, 1, 1, 1], False],
     "vovnet-39": [128, [128, 160, 192, 224], [256, 512, 768, 1024], 5, [1, 1, 2, 2], False],
    "vovnet-57": [ 128, [128, 160, 192, 224], [256, 512, 768, 1024], 5, [1, 1, 4, 3], False],
    "vovnet-99": [128, [128, 160, 192, 224], [256, 512, 768, 1024], 5, [1, 3, 9, 3], False]
}
"""
"vovnet-19-dw": VoVNetParams(64, [128, 160, 192, 224], [256, 512, 768, 1024], 3, [1, 1, 1, 1], True),
"vovnet-19-slim": VoVNetParams(
    128, [64, 80, 96, 112], [112, 256, 384, 512], 3, [1, 1, 1, 1], False
),
"vovnet-19": VoVNetParams(
    128, [128, 160, 192, 224], [256, 512, 768, 1024], 3, [1, 1, 1, 1], False
),
"vovnet-39": VoVNetParams(
    128, [128, 160, 192, 224], [256, 512, 768, 1024], 5, [1, 1, 2, 2], False
),
"vovnet-57": VoVNetParams(
    128, [128, 160, 192, 224], [256, 512, 768, 1024], 5, [1, 1, 4, 3], False
),
"vovnet-99": VoVNetParams(
    128, [128, 160, 192, 224], [256, 512, 768, 1024], 5, [1, 3, 9, 3], False
),"""

_BN_MOMENTUM = 1e-1
_BN_EPS = 1e-5


def dw_conv(
        in_channels: int, out_channels: int, stride: int = 1
) -> List[torch.nn.Module]:
    """ Depthwise separable pointwise linear convolution. """
    return [
        torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
            groups=in_channels,
            bias=False,
        ),
        torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
        torch.nn.BatchNorm2d(out_channels, eps=_BN_EPS, momentum=_BN_MOMENTUM),
        torch.nn.ReLU(inplace=True),
    ]


def conv(
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        groups: int = 1,
        kernel_size: int = 3,
        padding: int = 1,
) -> List[torch.nn.Module]:
    """ 3x3 convolution with padding."""
    return [
        torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        ),
        torch.nn.BatchNorm2d(out_channels, eps=_BN_EPS, momentum=_BN_MOMENTUM),
        torch.nn.ReLU(inplace=True),
    ]


def pointwise(in_channels: int, out_channels: int) -> List[torch.nn.Module]:
    """ Pointwise convolution."""
    return [
        torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
        torch.nn.BatchNorm2d(out_channels, eps=_BN_EPS, momentum=_BN_MOMENTUM),
        torch.nn.ReLU(inplace=True),
    ]


class ESE(torch.nn.Module):
    """This is adapted from the efficientnet Squeeze Excitation. The idea is to not
    squeeze the number of channels to keep more information."""

    def __init__(self, channel: int) -> None:
        super().__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Conv2d(channel, channel, kernel_size=1)  # (Linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.avg_pool(x)
        out = self.fc(out)
        return torch.sigmoid(out) * x


class OSA(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            stage_channels: int,
            concat_channels: int,
            layer_per_block: int,
            use_depthwise: bool = False,
    ) -> None:
        """ Implementation of an OSA layer which takes the output of its conv layers and
        concatenates them into one large tensor which is passed to the next layer. The
        goal with this concatenation is to preserve information flow through the model
        layers. This also ends up helping with small object detection.

        Args:
            in_channels: Channel depth of the input to the OSA block.
            stage_channels: Channel depth to reduce the input.
            concat_channels: Channel depth to force on the concatenated output of the
                comprising layers in a block.
            layer_per_block: The number of layers in this OSA block.
            use_depthwise: Wether to use depthwise separable pointwise linear convs.
        """
        super().__init__()
        # Keep track of the size of the final concatenation tensor.
        aggregated = in_channels
        self.isReduced = in_channels != stage_channels

        # If this OSA block is not the first in the OSA stage, we can
        # leverage the fact that subsequent OSA blocks have the same input and
        # output channel depth, concat_channels. This lets us reuse the concept of
        # a residual from ResNet models.
        self.identity = in_channels == concat_channels
        self.layers = torch.nn.ModuleList()
        self.use_depthwise = use_depthwise
        conv_op = dw_conv if use_depthwise else conv

        # If this model uses depthwise and the input channel depth needs to be reduced
        # to the stage_channels size, add a pointwise layer to adjust the depth. If the
        # model is not depthwise, let the first OSA layer do the resizing.
        if self.use_depthwise and self.isReduced:
            self.conv_reduction = torch.nn.Sequential(
                *pointwise(in_channels, stage_channels)
            )
            in_channels = stage_channels

        for _ in range(layer_per_block):
            self.layers.append(
                torch.nn.Sequential(*conv_op(in_channels, stage_channels))
            )
            in_channels = stage_channels

        # feature aggregation
        aggregated += layer_per_block * stage_channels
        self.concat = torch.nn.Sequential(*pointwise(aggregated, concat_channels))
        self.ese = ESE(concat_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.identity:
            identity_feat = x

        output = [x]
        if self.use_depthwise and self.isReduced:
            x = self.conv_reduction(x)

        # Loop through all the
        for layer in self.layers:
            x = layer(x)
            output.append(x)

        x = torch.cat(output, dim=1)
        xt = self.concat(x)
        xt = self.ese(xt)

        if self.identity:
            xt += identity_feat

        return xt


class OSA_stage(torch.nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            stage_channels: int,
            concat_channels: int,
            block_per_stage: int,
            layer_per_block: int,
            stage_num: int,
            use_depthwise: bool = False,
    ) -> None:
        """An OSA stage which is comprised of OSA blocks.
        Args:
            in_channels: Channel depth of the input to the OSA stage.
            stage_channels: Channel depth to reduce the input of the block to.
            concat_channels: Channel depth to force on the concatenated output of the
                comprising layers in a block.
            block_per_stage: Number of OSA blocks in this stage.
            layer_per_block: The number of layers per OSA block.
            stage_num: The OSA stage index.
            use_depthwise: Wether to use depthwise separable pointwise linear convs.
        """
        super().__init__()

        # Use maxpool to downsample the input to this OSA stage.
        self.add_module(
            "Pooling", torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        for idx in range(block_per_stage):
            # Add the OSA modules. If this is the first block in the stage, use the
            # proper in in channels, but the rest of the rest of the OSA layers will use
            # the concatenation channel depth outputted from the previous layer.
            self.add_module(
                f"OSA{stage_num}_{idx + 1}",
                OSA(
                    in_channels if idx == 0 else concat_channels,
                    stage_channels,
                    concat_channels,
                    layer_per_block,
                    use_depthwise=use_depthwise,
                ),
            )


@BACKBONES.register_module
class VoVNet(torch.nn.Sequential):
    def __init__(
            self, model_name: str, num_classes: int = 10, input_channels: int = 3
    ) -> None:
        """
        Args:
            model_name: Which model to create.
            num_classes: The number of classification classes.
            input_channels: The number of input channels.
        Usage:
        >>> net = VoVNet("vovnet-19-slim-dw", num_classes=1000)
        >>> with torch.no_grad():
        ...    out = net(torch.randn(1, 3, 512, 512))
        >>> print(out.shape)
        torch.Size([1, 1000])
        >>> net = VoVNet("vovnet-19-dw", num_classes=1000)
        >>> with torch.no_grad():
        ...    out = net(torch.randn(1, 3, 512, 512))
        >>> print(out.shape)
        torch.Size([1, 1000])
        >>> net = VoVNet("vovnet-19-slim", num_classes=1000)
        >>> with torch.no_grad():
        ...    out = net(torch.randn(1, 3, 512, 512))
        >>> print(out.shape)
        torch.Size([1, 1000])
        >>> net = VoVNet("vovnet-19", num_classes=1000)
        >>> with torch.no_grad():
        ...    out = net(torch.randn(1, 3, 512, 512))
        >>> print(out.shape)
        torch.Size([1, 1000])
        >>> net = VoVNet("vovnet-39", num_classes=1000)
        >>> with torch.no_grad():
        ...    out = net(torch.randn(1, 3, 512, 512))
        >>> print(out.shape)
        torch.Size([1, 1000])
        >>> net = VoVNet("vovnet-57", num_classes=1000)
        >>> with torch.no_grad():
        ...    out = net(torch.randn(1, 3, 512, 512))
        >>> print(out.shape)
        torch.Size([1, 1000])
        >>> net = VoVNet("vovnet-99", num_classes=1000)
        >>> with torch.no_grad():
        ...    out = net(torch.randn(1, 3, 512, 512))
        >>> print(out.shape)
        torch.Size([1, 1000])
           0 stem_out: int
    1 stage_conv_ch: List[int]  # Channel depth of
    2 stage_out_ch: List[int]  # The channel depth of the concatenated output
    3 layer_per_block: int
    4 block_per_stage: List[int]
     5 dw: bool
        """
        super().__init__()
        assert model_name in _STAGE_SPECS, f"{model_name} not supported."

        #stem_ch = _STAGE_SPECS[model_name].stem_out
        #config_stage_ch = _STAGE_SPECS[model_name].stage_conv_ch
        #config_concat_ch = _STAGE_SPECS[model_name].stage_out_ch
        #block_per_stage = _STAGE_SPECS[model_name].block_per_stage
        #layer_per_block = _STAGE_SPECS[model_name].layer_per_block
        #conv_type = dw_conv if _STAGE_SPECS[model_name].dw else conv
        stem_ch = _STAGE_SPECS[model_name][0]
        config_stage_ch = _STAGE_SPECS[model_name][1]
        config_concat_ch = _STAGE_SPECS[model_name][2]
        block_per_stage = _STAGE_SPECS[model_name][4]
        layer_per_block = _STAGE_SPECS[model_name][3]
        conv_type = dw_conv if _STAGE_SPECS[model_name][5] else conv

        # Construct the stem.
        stem = conv(input_channels, 64, stride=2)
        stem += conv_type(64, 64)

        # The original implementation uses a stride=2 on the conv below, but in this
        # implementation we'll just pool at every OSA stage, unlike the original
        # which doesn't pool at the first OSA stage.
        stem += conv_type(64, stem_ch)
        self.model = torch.nn.Sequential()
        self.model.add_module("stem", torch.nn.Sequential(*stem))
        self._out_feature_channels = [stem_ch]

        # Organize the outputs of each OSA stage. This is the concatentated channel
        # depth of each sub block's layer's outputs.
        in_ch_list = [stem_ch] + config_concat_ch[:-1]

        # Add the OSA modules. Typically 4 modules.
        for idx in range(len(config_stage_ch)):
            self.model.add_module(
                f"OSA_{(idx + 2)}",
                OSA_stage(
                    in_ch_list[idx],
                    config_stage_ch[idx],
                    config_concat_ch[idx],
                    block_per_stage[idx],
                    layer_per_block,
                    idx + 2,
                    _STAGE_SPECS[model_name][5],
                ),
            )

            self._out_feature_channels.append(config_concat_ch[idx])

        # Add the classification head.
        self.model.add_module(
            "classifier",
            torch.nn.Sequential(
                torch.nn.BatchNorm2d(
                    self._out_feature_channels[-1], _BN_MOMENTUM, _BN_EPS
                ),
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(self._out_feature_channels[-1], num_classes, bias=True),
            ),
        )

    #def forward(self, x: torch.Tensor) -> torch.Tensor:
        #return self.model(x)

    def forward(self, x: torch.Tensor) -> collections.OrderedDict:
        """def forward_pyramids(self, x: torch.Tensor) -> collections.OrderedDict:
        Args:
            model_name: Which model to create.
            num_classes: The number of classification classes.
            input_channels: The number of input channels.
        Usage:
        >>> net = VoVNet("vovnet-19-slim-dw", num_classes=1000)
        >>> net.delete_classification_head()
        >>> with torch.no_grad():
        ...    out = net.forward_pyramids(torch.randn(1, 3, 512, 512))
        >>> [level.shape[-1] for level in out.values()]  # Check the height/widths of levels
        [256, 128, 64, 32, 16]
        >>> [level.shape[1] for level in out.values()]  == net._out_feature_channels
        True
        """
        levels = collections.OrderedDict()
        levels0 = self.model.stem(x)
        levels[1] = self.model.OSA_2(levels0)
        levels[2] = self.model.OSA_3(levels[1])
        levels[3] = self.model.OSA_4(levels[2])
        levels[4] = self.model.OSA_5(levels[3])
        x=[levels[1],levels[2],levels[3],levels[4]]
        return x

    def delete_classification_head(self) -> None:
        """ Call this before using model as an object detection backbone. """
        del self.model.classifier

    def get_pyramid_channels(self) -> None:
        """ Return the number of channels for each pyramid level. """
        return self._out_feature_channels
    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            #if self.zero_init_residual:
             #   for m in self.modules():
              #      if isinstance(m, Bottleneck):
               #         constant_init(m.norm3, 0)
                #    elif isinstance(m, BasicBlock):
                 #       constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')
