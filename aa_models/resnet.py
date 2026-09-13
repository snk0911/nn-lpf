import torch
import torch.nn as nn
from typing import Any, Callable, Optional
from torch import Tensor

from .lpf_layers import *


__all__ = ["ResNet", "resnet18"]


_VALID_AA_TYPES = {"none", "blur", "dwt", "pasa", "dab", "asap"}

# BlurPool and PASA follow the Zhang-style residual-block integration:
# dense conv -> BN -> ReLU -> AA/downsample -> next dense conv.
_POST_ACTIVATION_AA = {"blur", "pasa"}

# WaveCNet Eq. (15) and DABPool Eq. (4) both replace a strided convolution
# by a dense convolution followed immediately by their downsampling operator.
# In this ResNet mapping that operator is therefore placed before BN/ReLU.
_POST_CONV_AA = {"dwt", "dab"}

# ASAP's official resnet_asap.py wraps every stride>1 convolution as
# ASAP_padding_one() -> dense convolution. This is the ASAPsp (small-padding)
# topology used here at the three shared stage-transition downsampling sites.
_PRE_CONV_AA = {"asap"}


def conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
) -> nn.Conv2d:
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


def _build_aa_layer(
    channels: int,
    stride: int,
    aa_type: str,
    wavelet_type: str,
    filter_size: int,
    pasa_group: int,
    dab_controller=None,
    depth_index: Optional[int] = None,
) -> nn.Module:
    """Create one method-specific anti-aliased downsampling layer."""
    layer = get_aa_layer(
        channels=channels,
        stride=stride,
        aa_type=aa_type,
        wavelet_type=wavelet_type,
        filter_size=filter_size,
        pasa_group=pasa_group,
        dab_controller=dab_controller,
        depth_index=depth_index,
    )

    return layer


class BasicBlock(nn.Module):
    """ResNet-18/34 BasicBlock with method-specific downsampling integration.

    The unfiltered baseline is the usual small-image/CIFAR-style BasicBlock:
        conv1 3x3, stride={1,2} -> BN -> ReLU -> conv2 3x3, stride=1 -> BN

    At stage transitions, the stride-2 operation is replaced according to the
    selected anti-aliasing method rather than forcing every method into one
    generic position.
    """

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        filter_size: int = 3,
        aa_type: str = "none",
        wavelet_type: str = "haar",
        pasa_group: int = 2,
        dab_controller=None,
        depth_index: Optional[int] = None,
    ) -> None:
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        if aa_type not in _VALID_AA_TYPES:
            raise ValueError(f"Unsupported aa_type: {aa_type}")

        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

        # Identity placeholders keep the forward path explicit and make the
        # method-specific order easy to inspect.
        self.aa_before_conv1 = nn.Identity()
        self.aa_after_conv1 = nn.Identity()
        self.aa_after_relu1 = nn.Identity()

        if stride == 1 or aa_type == "none":
            # Standard CIFAR/small-image ResNet-18 BasicBlock.
            self.conv1 = conv3x3(inplanes, planes, stride=stride)

        elif aa_type in _POST_ACTIVATION_AA:
            # BlurPool / PASA:
            # Conv3x3(s1) -> BN -> ReLU -> AA(s2) -> Conv3x3(s1)
            self.conv1 = conv3x3(inplanes, planes, stride=1)
            self.aa_after_relu1 = _build_aa_layer(
                channels=planes,
                stride=stride,
                aa_type=aa_type,
                wavelet_type=wavelet_type,
                filter_size=filter_size,
                pasa_group=pasa_group,
                dab_controller=dab_controller,
                depth_index=depth_index,
            )

        elif aa_type in _POST_CONV_AA:
            # WaveCNet / DABPool:
            # Conv3x3(s1) -> AA(s2) -> BN -> ReLU -> Conv3x3(s1)
            self.conv1 = conv3x3(inplanes, planes, stride=1)
            self.aa_after_conv1 = _build_aa_layer(
                channels=planes,
                stride=stride,
                aa_type=aa_type,
                wavelet_type=wavelet_type,
                filter_size=filter_size,
                pasa_group=pasa_group,
                dab_controller=dab_controller,
                depth_index=depth_index,
            )

        elif aa_type in _PRE_CONV_AA:
            # ASAP reference topology:
            # ASAP(s2) -> Conv3x3(s1) -> BN -> ReLU -> Conv3x3(s1)
            self.aa_before_conv1 = _build_aa_layer(
                channels=inplanes,
                stride=stride,
                aa_type=aa_type,
                wavelet_type=wavelet_type,
                filter_size=filter_size,
                pasa_group=pasa_group,
                dab_controller=dab_controller,
                depth_index=depth_index,
            )
            self.conv1 = conv3x3(inplanes, planes, stride=1)

        else:  # defensive; all valid methods are handled above
            raise ValueError(f"Unsupported aa_type: {aa_type}")

        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride=1)
        self.bn2 = norm_layer(planes)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.aa_before_conv1(x)
        out = self.conv1(out)
        out = self.aa_after_conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.aa_after_relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """Tiny-ImageNet ResNet-18 using the standard CIFAR/small-image adaptation.

    Stem: 3x3 convolution, stride 1, no initial max-pooling.
    Spatial reduction occurs only in the first block of layers 2, 3, and 4.
    """

    def __init__(
        self,
        block: type[BasicBlock],
        layers: list[int],
        num_classes: int = 200,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[list[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        filter_size: int = 3,
        aa_type: str = "none",
        wavelet_type: str = "haar",
        pasa_group: int = 2,
    ) -> None:
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        if aa_type not in _VALID_AA_TYPES:
            raise ValueError(
                f"Unknown aa_type {aa_type!r}. Expected one of {sorted(_VALID_AA_TYPES)}"
            )

        self.aa_type = aa_type
        self.wavelet_type = wavelet_type
        self.filter_size = filter_size
        self.pasa_group = pasa_group

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None or a 3-element "
                f"list/tuple, got {replace_stride_with_dilation}"
            )

        self.groups = groups
        self.base_width = width_per_group

        # With the CIFAR/small-image stem there is no initial pooling stage.
        # Therefore ResNet-18 has exactly three spatial downsampling stages:
        # layer2, layer3, layer4.
        num_downsample_levels = sum(
            1 for dilate in replace_stride_with_dilation if not dilate
        )
        self.dab_controller = (
            DABSigmaController(num_downsample_layers=num_downsample_levels)
            if aa_type == "dab"
            else None
        )

        # DABPool alone needs an explicit depth index in this comparison.
        # ASAP uses the official small-padding variant (ASAPsp); the alternating
        # transpose heuristic belongs to the separate ASAPstbl variant and is
        # intentionally not enabled here.
        self.dab_depth = 0

        # CIFAR/small-image stem: only the ImageNet stem is changed.
        self.conv1 = nn.Conv2d(
            3,
            self.inplanes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        # No initial MaxPool.
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: type[BasicBlock],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        out_channels = planes * block.expansion
        current_depth = (
            self.dab_depth
            if self.aa_type == "dab" and stride != 1
            else None
        )

        if stride != 1 or self.inplanes != out_channels:
            if self.aa_type == "none" or stride == 1:
                # Standard CIFAR/small-image ResNet projection shortcut.
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, out_channels, stride=stride),
                    norm_layer(out_channels),
                )

            elif self.aa_type in _POST_CONV_AA:
                # WaveCNet / DABPool:
                # Conv1x1(s1) -> AA(s2) -> BN
                shortcut_aa = _build_aa_layer(
                    channels=out_channels,
                    stride=stride,
                    aa_type=self.aa_type,
                    wavelet_type=self.wavelet_type,
                    filter_size=self.filter_size,
                    pasa_group=self.pasa_group,
                    dab_controller=self.dab_controller,
                    depth_index=current_depth,
                )
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, out_channels, stride=1),
                    shortcut_aa,
                    norm_layer(out_channels),
                )

            elif self.aa_type in (_POST_ACTIVATION_AA | _PRE_CONV_AA):
                # BlurPool / PASA / ASAP:
                # AA(s2) -> Conv1x1(s1) -> BN
                shortcut_aa = _build_aa_layer(
                    channels=self.inplanes,
                    stride=stride,
                    aa_type=self.aa_type,
                    wavelet_type=self.wavelet_type,
                    filter_size=self.filter_size,
                    pasa_group=self.pasa_group,
                    dab_controller=self.dab_controller,
                    depth_index=current_depth,
                )
                downsample = nn.Sequential(
                    shortcut_aa,
                    conv1x1(self.inplanes, out_channels, stride=1),
                    norm_layer(out_channels),
                )

            else:
                raise ValueError(f"Unsupported aa_type: {self.aa_type}")

        layers = [
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                self.filter_size,
                self.aa_type,
                self.wavelet_type,
                self.pasa_group,
                dab_controller=self.dab_controller,
                depth_index=current_depth,
            )
        ]

        self.inplanes = out_channels

        if stride != 1 and self.aa_type == "dab":
            self.dab_depth += 1

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    filter_size=self.filter_size,
                    aa_type=self.aa_type,
                    wavelet_type=self.wavelet_type,
                    pasa_group=self.pasa_group,
                    dab_controller=self.dab_controller,
                    depth_index=None,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # No initial pooling in the CIFAR/small-image adaptation.
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def resnet18(
    *,
    progress: bool = True,
    filter_size: int = 3,
    aa_type: str = "none",
    wavelet_type: str = "haar",
    pasa_group: int = 2,
    **kwargs: Any,
) -> nn.Module:
    # `progress` is retained for API compatibility with existing callers.
    del progress

    return ResNet(
        BasicBlock,
        [2, 2, 2, 2],
        filter_size=filter_size,
        aa_type=aa_type,
        wavelet_type=wavelet_type,
        pasa_group=pasa_group,
        **kwargs,
    )
