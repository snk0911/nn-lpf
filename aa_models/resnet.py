import torch
import torch.nn as nn
from typing import Any, Callable, Optional, Union, Dict
from torch import Tensor
import torch.utils.model_zoo as model_zoo
from .lpf_layers import *


__all__ = ['ResNet', 'resnet18']


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
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
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, 
                 base_width=64, dilation=1, norm_layer=None, filter_size=1, 
                 aa_type='none', wavelet_type='haar', pasa_group=2,
                 dab_controller=None, depth_index=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        # Retrieve the requested AA layer using the helper
        aa_layer = get_aa_layer(
            channels=planes, 
            stride=stride, 
            aa_type=aa_type, 
            wavelet_type=wavelet_type, 
            filter_size=filter_size, 
            pasa_group=pasa_group,
            dab_controller=dab_controller,
            depth_index=depth_index
        )

        self.conv1 = conv3x3(inplanes, planes)   # always stride 1, no AA here
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        if isinstance(aa_layer, nn.Identity):
            # Baseline: stride remains in conv2
            self.conv2 = conv3x3(
                planes,
                planes,
                stride=stride
            )

        elif aa_type in ('dab', 'dwt'):
            # DAB / WaveCNet:
            # strided Conv -> dense Conv followed by AA/downsampling
            self.conv2 = nn.Sequential(
                conv3x3(
                    planes,
                    planes,
                    stride=1
                ),
                aa_layer
            )

        else:
            # BlurPool / PASA / ASAP / avg:
            # preserve the existing integration
            self.conv2 = nn.Sequential(
                aa_layer,
                conv3x3(
                    planes,
                    planes,
                    stride=1
                )
            )

        self.bn2 = norm_layer(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

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
        filter_size: int = 1,
        aa_type: str = 'none',
        wavelet_type: str = 'haar',
        pasa_group: int = 2,
        dab_controller=None, depth_index=None
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        
        # --- AA Logic on CONV2 ---
        # In ResNet V1.5 (Bottleneck), stride is in conv2.
        aa_layer = get_aa_layer(width, stride, aa_type, wavelet_type, filter_size, pasa_group,
                        dab_controller=dab_controller, depth_index=depth_index)
        is_identity = isinstance(aa_layer, nn.Identity)

        if is_identity:
            self.conv2 = conv3x3(
                width,
                width,
                stride,
                groups,
                dilation
            )

        elif aa_type in ('dab', 'dwt'):
            self.conv2 = nn.Sequential(
                conv3x3(
                    width,
                    width,
                    stride=1,
                    groups=groups,
                    dilation=dilation
                ),
                aa_layer
            )

        else:
            self.conv2 = nn.Sequential(
                aa_layer,
                conv3x3(
                    width,
                    width,
                    stride=1,
                    groups=groups,
                    dilation=dilation
                )
            )
            
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: type[Union[BasicBlock, Bottleneck]],
        layers: list[int],
        num_classes: int = 200,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[list[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        filter_size: int = 3,
        aa_type: str = 'none',
        wavelet_type: str = 'haar',
        pasa_group: int = 2
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.aa_type = aa_type
        self.wavelet_type = wavelet_type
        self.filter_size = filter_size
        self.pasa_group = pasa_group

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None " f"or a 3-element tuple, got {replace_stride_with_dilation}")
        self.groups = groups
        self.base_width = width_per_group

        self.dab_controller = DABSigmaController(num_downsample_layers=4) if aa_type == 'dab' else None
        self.dab_depth = 0

        def get_aa_helper(c, s, d_idx=None):
            return get_aa_layer(c, s, self.aa_type, self.wavelet_type, self.filter_size, self.pasa_group,
                                dab_controller=self.dab_controller, depth_index=d_idx)
        
        # --- Head Configuration ---
        # CIFAR-Style / Tiny-ImageNet stem:
        # Keep the initial convolution dense. Spatial downsampling starts in self.pool.
        self.conv1 = nn.Conv2d(
            3,
            self.inplanes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        # ---------------------------------------------------------
        # First downsampling operation
        # ---------------------------------------------------------

        if self.aa_type == 'none':
            # Vanilla baseline:
            # MaxPool(k=3, s=2)
            self.pool = nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1
            )

        elif self.aa_type in ('dwt', 'asap'):
            # WaveCNet, Eq. (14):
            #
            # MaxPool_s=2 -> DWT_LL
            #
            # IMPORTANT:
            # Do NOT keep a stride-1 MaxPool in front of DWT.
            # DWT_2D_tiny already performs filtering + 2x downsampling.
            self.pool = get_aa_helper(
                self.inplanes,
                2
            )

        else:
            # BlurPool / PASA / DAB / avg:
            #
            # Preserve the existing integration:
            # DenseMax(s=1) -> AA/downsampling layer
            #
            # DAB additionally requires the current depth index.
            depth_index = (
                self.dab_depth
                if self.aa_type == 'dab'
                else None
            )

            aa = get_aa_helper(
                self.inplanes,
                2,
                depth_index
            )

            # Preserve the old fallback behavior for unknown aa_type values.
            if isinstance(aa, nn.Identity):
                self.pool = nn.MaxPool2d(
                    kernel_size=3,
                    stride=2,
                    padding=1
                )
            else:
                self.pool = nn.Sequential(
                    nn.MaxPool2d(
                        kernel_size=3,
                        stride=1,
                        padding=1
                    ),
                    aa
                )

                # Only DAB uses the depth counter.
                if self.aa_type == 'dab':
                    self.dab_depth += 1

        # --- Layers ---
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # --- Initialization ---
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        current_depth = (
            self.dab_depth
            if self.aa_type == 'dab'
            else None
        )

        # A projection shortcut is only needed when:
        # 1. spatial resolution changes, or
        # 2. channel count changes.
        if stride != 1 or self.inplanes != planes * block.expansion:

            out_channels = planes * block.expansion

            if self.aa_type == 'none':
                # Baseline shortcut:
                # Conv1x1 with the original stride.
                downsample = nn.Sequential(
                    conv1x1(
                        self.inplanes,
                        out_channels,
                        stride
                    ),
                    norm_layer(out_channels)
                )

            elif self.aa_type in ('dab', 'dwt'):
                # DAB / WaveCNet:
                #
                # Conv1x1(s=2)
                # becomes
                # Conv1x1(s=1) -> AA/downsampling
                #
                # Since AA runs AFTER the projection, it operates
                # on out_channels.
                shortcut_aa = get_aa_layer(
                    out_channels,
                    stride,
                    self.aa_type,
                    self.wavelet_type,
                    self.filter_size,
                    self.pasa_group,
                    dab_controller=self.dab_controller,
                    depth_index=current_depth
                )

                downsample = nn.Sequential(
                    conv1x1(
                        self.inplanes,
                        out_channels,
                        stride=1
                    ),
                    shortcut_aa,
                    norm_layer(out_channels)
                )

            else:
                # BlurPool / PASA / ASAP / avg:
                # Preserve the existing AA-before-projection integration.
                shortcut_aa = get_aa_layer(
                    self.inplanes,
                    stride,
                    self.aa_type,
                    self.wavelet_type,
                    self.filter_size,
                    self.pasa_group,
                    dab_controller=self.dab_controller,
                    depth_index=current_depth
                )

                downsample = nn.Sequential(
                    shortcut_aa,
                    conv1x1(
                        self.inplanes,
                        out_channels,
                        stride=1
                    ),
                    norm_layer(out_channels)
                )

        # One new DAB depth level per actual spatial downsampling stage.
        # Main branch and shortcut both use current_depth above;
        # only afterwards do we advance to the next level.
        if stride != 1 and self.aa_type == 'dab':
            self.dab_depth += 1

        layers = []

        layers.append(
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
                depth_index=current_depth if stride != 1 else None
            )
        )

        self.inplanes = planes * block.expansion

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
                    depth_index=None
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.pool(x)

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
    aa_type: str = 'none',
    wavelet_type: str = 'haar',
    pasa_group: int = 2,
    **kwargs: Any
) -> nn.Module:
    model = ResNet(
        BasicBlock,
        [2, 2, 2, 2],
        filter_size=filter_size,
        aa_type=aa_type,
        wavelet_type=wavelet_type,
        pasa_group=pasa_group,
        **kwargs
    )
    return model
