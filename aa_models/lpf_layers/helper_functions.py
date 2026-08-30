import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Callable


# ---------------------------------------------------------
# helper functions
# ---------------------------------------------------------
def get_aa_layer(
    channels: int,
    stride: int,
    aa_type: str,
    wavelet_type: str,
    filter_size: int,
    pasa_group: int,
    dab_controller=None,
    depth_index=None,
) -> nn.Module:

    from .blur import BlurPool
    from .dwt import DWT_2D_tiny
    from .asap import ASAP_padding_one
    from .pasa import Downsample_PASA_group_softmax
    from .dab import DABPool

    if stride == 1:
        return nn.Identity()

    if aa_type == "dab":
        if dab_controller is None:
            raise ValueError("DAB requires a DABSigmaController")

        if depth_index is None:
            raise ValueError("DAB requires a depth_index")

        if filter_size not in (3, 5, 7):
            raise ValueError(
                "DAB filter_size must be 3, 5, or 7."
            )

        return DABPool(
            channels,
            stride=stride,
            dab_controller=dab_controller,
            depth_index=depth_index,
            filter_size=filter_size,
            padding=filter_size // 2,
        )

    if aa_type == "avg":
        return nn.AvgPool2d(
            kernel_size=filter_size,
            stride=stride,
            padding=filter_size // 2,
        )

    if aa_type == "blur":
        return BlurPool(
            channels,
            filter_size=filter_size,
            stride=stride,
        )

    if aa_type == "dwt":
        return DWT_2D_tiny(wavelet_type)

    if aa_type == "pasa":
        return Downsample_PASA_group_softmax(
            channels,
            filter_size,
            stride,
            group=pasa_group,
        )

    if aa_type == "asap":
        return ASAP_padding_one()

    return nn.Identity()


def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        raise ValueError(f'Pad type [{pad_type}] not recognized')
    return PadLayer
