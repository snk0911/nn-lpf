import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, Optional, Union, Dict, List
from torch import Tensor

import torch.utils.model_zoo as model_zoo
from .lpf_layers import *

__all__ = ['PreActResNet', 'preact_resnet18', 'preact_resnet34', 'preact_resnet50', 'preact_resnet101', 'preact_resnet152']

# ---------------------------------------------------------
# helper functions (from first script)
# ---------------------------------------------------------
def get_aa_layer(
    channels: int, 
    stride: int, 
    aa_type: str, 
    wavelet_type: str,
    filter_size: int, 
    pasa_group: int,
    dab_controller=None,
    depth_index=None
) -> nn.Module:
    if stride == 1:
        return nn.Identity()

    layer_registry: Dict[str, Callable[[], nn.Module]] = {
        'blur': lambda: BlurPool(channels, filter_size=filter_size, stride=stride),
        'pasa': lambda: Downsample_PASA_group_softmax(channels, filter_size, stride, group=pasa_group),
        'dwt': lambda: DWT_2D_tiny(wavelet_type),
        'dab': lambda: DABPool(
            channels, channels, 
            kernel_size=3, stride=stride, padding=1,
            depth_index=depth_index, 
            dab_controller=dab_controller
        ),
    }

    layer_factory = layer_registry.get(aa_type)
    return layer_factory() if layer_factory else nn.Identity()

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


# ---------------------------------------------------------
# Pre-Activation Blocks with Anti-Aliasing
# ---------------------------------------------------------

class PreActBasicBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
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
        filter_size: int = 1, 
        aa_type: str = 'none', 
        wavelet_type: str = 'haar', 
        pasa_group: int = 2,
        dab_controller=None, 
        depth_index=None
    ):
        super(PreActBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.bn1 = norm_layer(inplanes)
        
        # Use AARelu if DAB is enabled, otherwise standard ReLU
        self.relu = AARelu() if aa_type == 'dab' else nn.ReLU(inplace=False)
        
        # --- Conv1 Logic (Stride location in BasicBlock) ---
        aa_layer = get_aa_layer(inplanes, stride, aa_type, wavelet_type, filter_size, pasa_group,
                                dab_controller=dab_controller, depth_index=depth_index)
        is_identity = isinstance(aa_layer, nn.Identity)

        if is_identity:
            self.conv1 = conv3x3(inplanes, planes, stride)
        else:
            # AA Method: AA Layer (stride) -> Conv3x3 (stride 1)
            self.conv1 = nn.Sequential(
                aa_layer,
                conv3x3(inplanes, planes, 1)
            )

        self.bn2 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, 1) # Conv2 is always stride 1
        
        # --- Shortcut / Downsample Logic ---
        # In PreAct, the shortcut operates on the output of the first BN/ReLU
        if stride != 1 or inplanes != self.expansion * planes:
            # The shortcut needs to perform the same spatial downsampling as the main path
            # If AA is used, we must apply AA in the shortcut as well
            aa_layer_sc = get_aa_layer(inplanes, stride, aa_type, wavelet_type, filter_size, pasa_group,
                                       dab_controller=dab_controller, depth_index=depth_index)
            is_identity_sc = isinstance(aa_layer_sc, nn.Identity)
            
            if is_identity_sc:
                self.shortcut = conv1x1(inplanes, self.expansion * planes, stride)
            else:
                self.shortcut = nn.Sequential(
                    aa_layer_sc,
                    conv1x1(inplanes, self.expansion * planes, 1)
                )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # Pre-activation: BN -> ReLU
        out = self.relu(self.bn1(x))
        
        # Shortcut connection uses the activated features
        shortcut = self.shortcut(out)
        
        out = self.conv1(out)
        out = self.conv2(self.relu(self.bn2(out)))
        
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
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
        dab_controller=None, 
        depth_index=None
    ):
        super(PreActBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        
        self.bn1 = norm_layer(inplanes)
        self.relu1 = AARelu() if aa_type == 'dab' else nn.ReLU(inplace=False)
        self.conv1 = conv1x1(inplanes, width)
        
        self.bn2 = norm_layer(width)
        self.relu2 = AARelu() if aa_type == 'dab' else nn.ReLU(inplace=False)
        
        # --- Conv2 Logic (Stride location in Bottleneck) ---
        aa_layer = get_aa_layer(width, stride, aa_type, wavelet_type, filter_size, pasa_group,
                                dab_controller=dab_controller, depth_index=depth_index)
        is_identity = isinstance(aa_layer, nn.Identity)

        if is_identity:
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
        else:
            # Unified AA: AA Layer (stride) -> 3x3 Conv (stride 1)
            self.conv2 = nn.Sequential(
                aa_layer,
                conv3x3(width, width, stride=1, groups=groups, dilation=dilation)
            )
            
        self.bn3 = norm_layer(width)
        self.relu3 = AARelu() if aa_type == 'dab' else nn.ReLU(inplace=False)
        self.conv3 = conv1x1(width, planes * self.expansion)

        # --- Shortcut Logic ---
        if stride != 1 or inplanes != planes * self.expansion:
            aa_layer_sc = get_aa_layer(inplanes, stride, aa_type, wavelet_type, filter_size, pasa_group,
                                       dab_controller=dab_controller, depth_index=depth_index)
            is_identity_sc = isinstance(aa_layer_sc, nn.Identity)
            
            if is_identity_sc:
                self.shortcut = conv1x1(inplanes, planes * self.expansion, stride)
            else:
                self.shortcut = nn.Sequential(
                    aa_layer_sc,
                    conv1x1(inplanes, planes * self.expansion, 1)
                )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # Pre-activation
        out = self.relu1(self.bn1(x))
        shortcut = self.shortcut(out)
        
        out = self.conv1(out)
        out = self.conv2(self.relu2(self.bn2(out)))
        out = self.conv3(self.relu3(self.bn3(out)))
        
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(
        self,
        block: type[Union[PreActBasicBlock, PreActBottleneck]],
        layers: List[int],
        num_classes: int = 200,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        filter_size: int = 1,
        aa_type: str = 'none',
        wavelet_type: str = 'haar',
        pasa_group: int = 2,
        pool_only: bool = True, 
    ) -> None:
        super(PreActResNet, self).__init__()
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

        self.dab_controller = DABSigmaController(num_downsample_layers=5) if aa_type == 'dab' else None
        self.dab_depth = 0

        # Helper to create AA layers with current depth tracking
        def get_aa_helper(c, s, d_idx=None):
            return get_aa_layer(c, s, self.aa_type, self.wavelet_type, self.filter_size, self.pasa_group,
                                dab_controller=self.dab_controller, depth_index=d_idx)

        # --- Head Configuration ---
        # PreAct usually doesn't have BN/ReLU before the first conv, but we maintain the
        # structural complexity (maxpool/AA) from the first script.
        
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        
        aa_head_check = get_aa_helper(self.inplanes, 2)
        is_identity = isinstance(aa_head_check, nn.Identity)

        if is_identity:
            # Baseline PreAct head (from snippet 2, simplified)
            # Note: The first script borrowed from CIFAR ResNet.
            # If no AA, we keep it simple like snippet 2 unless we want ImageNet style.
            # Given snippet 1 has maxpool, we keep it but ensure stride 1 for conv1.
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            # Anti-aliased head
            if pool_only:
                aa_1 = get_aa_helper(self.inplanes, 2, self.dab_depth); self.dab_depth += 1
                aa_2 = get_aa_helper(self.inplanes, 2, self.dab_depth); self.dab_depth += 1
                # MaxPool with stride 1, followed by AA
                self.maxpool = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1), aa_2)
            else:
                aa_1 = get_aa_helper(self.inplanes, 2, self.dab_depth); self.dab_depth += 1
                aa_2 = get_aa_helper(self.inplanes, 2, self.dab_depth); self.dab_depth += 1
                self.maxpool = nn.Sequential(aa_1, nn.MaxPool2d(kernel_size=3, stride=1, padding=1), aa_2)

        # --- Layers ---
        # PreAct blocks handle their own BN/ReLU
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

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, PreActBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, PreActBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
            
        # We track depth for DAB controller
        current_depth = self.dab_depth if self.aa_type == 'dab' else None
        
        # Note: In PreAct, the downsample (shortcut) logic is encapsulated INSIDE the block
        # because the shortcut is applied to the activated features of that block.
        # So we don't need to create a separate downsample module here like in v1.5.
        
        if stride != 1 and self.aa_type == 'dab':
            self.dab_depth += 1
            
        layers = []
        layers.append(block(self.inplanes, planes, stride, None, self.groups, 
                            self.base_width, previous_dilation, norm_layer,
                            self.filter_size, self.aa_type, self.wavelet_type, self.pasa_group,
                            dab_controller=self.dab_controller, depth_index=current_depth))
        
        self.inplanes = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, 
                                base_width=self.base_width, dilation=self.dilation, 
                                norm_layer=norm_layer, filter_size=self.filter_size,
                                aa_type=self.aa_type, wavelet_type=self.wavelet_type, 
                                pasa_group=self.pasa_group,
                                dab_controller=self.dab_controller, depth_index=None))
        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # PreAct head: Conv -> Pool (No BN/ReLU here)
        x = self.conv1(x)
        x = self.maxpool(x)

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


# ---------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------

def preact_resnet18(**kwargs: Any) -> PreActResNet:
    return PreActResNet(PreActBasicBlock, [2, 2, 2, 2], **kwargs)

def preact_resnet34(**kwargs: Any) -> PreActResNet:
    return PreActResNet(PreActBasicBlock, [3, 4, 6, 3], **kwargs)

def preact_resnet50(**kwargs: Any) -> PreActResNet:
    return PreActResNet(PreActBottleneck, [3, 4, 6, 3], **kwargs)

def preact_resnet101(**kwargs: Any) -> PreActResNet:
    return PreActResNet(PreActBottleneck, [3, 4, 23, 3], **kwargs)

def preact_resnet152(**kwargs: Any) -> PreActResNet:
    return PreActResNet(PreActBottleneck, [3, 8, 36, 3], **kwargs)