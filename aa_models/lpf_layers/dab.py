import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DABPool(nn.Module):
    """
    Depth-Adaptive Blur Pooling (DAB-Pool).

    A shared DABSigmaController provides the learnable standard deviation
    sigma for the current downsampling depth. Gaussian blurring and spatial
    subsampling are combined in one depthwise strided convolution.
    """

    def __init__(
        self,
        channels,
        stride=2,
        depth_index=0,
        dab_controller=None,
        filter_size=3,
        padding=None,
    ):
        super().__init__()

        if not isinstance(dab_controller, DABSigmaController):
            raise TypeError(
                "dab_controller must be an instance of DABSigmaController."
            )

        if not isinstance(channels, int) or channels <= 0:
            raise ValueError(
                "channels must be a positive integer."
            )

        if not isinstance(stride, int) or stride <= 0:
            raise ValueError(
                "stride must be a positive integer."
            )

        if (
            not isinstance(filter_size, int)
            or filter_size <= 0
            or filter_size % 2 == 0
        ):
            raise ValueError(
                "filter_size must be a positive odd integer."
            )

        if (
            not isinstance(depth_index, int)
            or not 0 <= depth_index < dab_controller.num_layers
        ):
            raise ValueError(
                f"depth_index must be an integer in "
                f"[0, {dab_controller.num_layers - 1}], "
                f"got {depth_index}."
            )
        
        # Preserve centered filtering before stride-based downsampling
        # by using symmetric padding based on the kernel radius.
        if padding is None:
            padding = filter_size // 2

        # The controller is owned and registered once by the parent network.
        # Store only a non-registered reference here so the same shared
        # controller does not appear under every DABPool in the module tree
        # and state_dict.
        object.__setattr__(self, "controller", dab_controller)

        self.depth_index = depth_index
        self.stride = stride

        self.blur = GaussianBlur2d(
            channels=channels,
            kernel_size=filter_size,
            padding=padding,
        )

    def forward(self, x):
        sigma = self.controller.get_sigma(self.depth_index)
        return self.blur(x, sigma, stride=self.stride)


class DABSigmaController(nn.Module):
    """
    Shared learnable sigma controller for DAB-Pool layers.

    The paper initializes the Gaussian standard deviation at depth D as

        sigma_D = D / 2

    and requires progressively stronger blurring for deeper downsampling
    levels. We parameterize positive increments with softplus and take their
    cumulative sum:

        delta_D > 0
        sigma_D = sum_{i=1..D} delta_i

    Therefore

        sigma_1 < sigma_2 < ... < sigma_M

    is guaranteed by construction.
    """

    def __init__(self, num_downsample_layers):
        super().__init__()

        if num_downsample_layers <= 0:
            raise ValueError("num_downsample_layers must be > 0.")

        self.num_layers = num_downsample_layers

        # We want every initial positive increment to equal 0.5.
        #
        # softplus(x) = log(1 + exp(x)) = 0.5
        # x = log(exp(0.5) - 1)
        #
        # Equal increments of 0.5 give:
        # sigma_1 = 0.5, sigma_2 = 1.0, sigma_3 = 1.5, ...
        init_val = math.log(math.expm1(0.5))

        self.deltas = nn.Parameter(
            torch.full(
                (num_downsample_layers,),
                init_val,
                dtype=torch.float32,
            )
        )

    def get_sigma(self, depth_index):
        if not 0 <= depth_index < self.num_layers:
            raise IndexError(
                f"depth_index must be in [0, {self.num_layers - 1}], "
                f"got {depth_index}."
            )

        # Each learned increment is strictly positive.
        positive_increments = F.softplus(self.deltas)

        # Cumulative positive increments guarantee monotonically increasing
        # Gaussian standard deviations with network depth.
        cumulative_sigmas = torch.cumsum(
            positive_increments,
            dim=0,
        )

        return cumulative_sigmas[depth_index]


class GaussianBlur2d(nn.Module):
    """
    Depthwise Gaussian blur with dynamically supplied sigma.

    The isotropic 2-D Gaussian is separable:

        G(x, y) = g(x) * g(y)

    Therefore only the 1-D Gaussian vector is evaluated with exp(). The full
    2-D kernel is reconstructed by an outer product. This is mathematically
    equivalent to evaluating the 2-D Gaussian directly.

    The reconstructed 2-D kernel is still applied by ONE depthwise conv2d.
    We are optimizing kernel construction here, not splitting the convolution
    itself into horizontal and vertical passes.
    """

    def __init__(self, channels, kernel_size=3, padding=0):
        super().__init__()

        if channels <= 0:
            raise ValueError("channels must be > 0.")

        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")

        if padding < 0:
            raise ValueError("padding must be >= 0.")

        self.kernel_size = kernel_size
        self.channels = channels
        self.padding = padding

        radius = kernel_size // 2

        # For kernel_size=3:
        # coords    = [-1, 0, 1]
        # coord_sq  = [ 1, 0, 1]
        coords = torch.arange(
            -radius,
            radius + 1,
            dtype=torch.float32,
        )

        # Only k squared distances are needed instead of a k x k distance grid.
        self.register_buffer(
            "coord_sq",
            coords.square(),
        )

    def _make_kernel(self, sigma):
        """
        Construct a normalized 2-D Gaussian kernel from a 1-D Gaussian vector.

        Kept as a small helper so the mathematical kernel can also be tested
        independently of conv2d.
        """
        # sigma is > 0 because DABSigmaController uses softplus increments.
        sigma_sq = sigma.square()

        # 1-D Gaussian:
        # g(x) = exp(-x^2 / (2 sigma^2))
        g = torch.exp(
            -self.coord_sq / (2.0 * sigma_sq)
        )

        # Normalize the 1-D vector. If sum(g) = 1, the outer product also has
        # sum equal to 1 because sum_ij g_i g_j = sum_i(g_i) * sum_j(g_j).
        g = g / g.sum()

        # Outer product:
        # G(x, y) = g(x) g(y)
        #
        # g[:, None] has shape [k, 1]
        # g[None, :] has shape [1, k]
        # Broadcasting produces a [k, k] matrix.
        kernel = g[:, None] * g[None, :]

        return kernel

    def forward(self, x, sigma, stride=1):
        if x.ndim != 4:
            raise ValueError(
                f"Expected x with shape [N, C, H, W], got {tuple(x.shape)}."
            )

        if x.shape[1] != self.channels:
            raise ValueError(
                f"Expected {self.channels} input channels, got {x.shape[1]}."
            )

        kernel = self._make_kernel(sigma)

        # Depthwise conv2d expects one spatial kernel per channel:
        # [out_channels, in_channels/groups, kH, kW]
        # -> [channels, 1, k, k]
        kernel = kernel.view(
            1,
            1,
            self.kernel_size,
            self.kernel_size,
        ).expand(
            self.channels,
            1,
            self.kernel_size,
            self.kernel_size,
        )

        # groups=channels applies the same spatial Gaussian independently to
        # every feature channel without mixing channels.
        #
        # With stride=2, blurring and spatial subsampling are combined in this
        # single depthwise convolution.
        return F.conv2d(
            x,
            weight=kernel,
            stride=stride,
            padding=self.padding,
            groups=self.channels,
        )