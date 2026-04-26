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
    depth_index=None
) -> nn.Module:

    from .blur import BlurPool
    from .dwt import DWT_2D_tiny
    from .pasa import Downsample_PASA_group_softmax
    from .dab import DABPool

    if stride == 1:
        return nn.Identity()

    # Map strings to lambda functions that instantiate the layers
    layer_registry: Dict[str, Callable[[], nn.Module]] = {
        'avg': lambda: nn.AvgPool2d(
            kernel_size=filter_size,
            stride=stride,
            padding=filter_size // 2
        ),
        'blur': lambda: BlurPool(channels, filter_size=filter_size, stride=stride),
        'dwt': lambda: DWT_2D_tiny(wavelet_type),
        'pasa': lambda: Downsample_PASA_group_softmax(channels, filter_size, stride, group=pasa_group),
        'dab': lambda: DABPool(
            channels, 
            stride=stride, 
            dab_controller=dab_controller, 
            depth_index=depth_index,
            filter_size=filter_size, 
            padding=filter_size//2  # Standard 'same' padding logic
        ),
    }

    # .get() returns None if key doesn't exist, triggering the fallback
    layer_factory = layer_registry.get(aa_type)

    return layer_factory() if layer_factory else nn.Identity()


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
