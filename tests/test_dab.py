import pytest
import torch
import torch.nn.functional as F

from aa_models.lpf_layers.dab import (
    DABPool,
    DABSigmaController,
    GaussianBlur2d,
)


@pytest.mark.parametrize(
    "num_layers, expected",
    [
        (1, [0.5]),
        (3, [0.5, 1.0, 1.5]),
        (5, [0.5, 1.0, 1.5, 2.0, 2.5]),
    ],
)
def test_sigma_initialization_follows_paper(num_layers, expected):
    """
    Hossain et al. initialize sigma_D = D / 2.
    """
    controller = DABSigmaController(num_downsample_layers=num_layers)

    sigmas = torch.stack(
        [controller.get_sigma(i) for i in range(num_layers)]
    )

    expected = torch.tensor(expected, dtype=sigmas.dtype)

    torch.testing.assert_close(
        sigmas.detach(),
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


def test_sigma_is_strictly_increasing():
    """
    Positive softplus increments plus cumulative sum must guarantee
    sigma_1 < sigma_2 < ... for arbitrary learned controller parameters.
    """
    controller = DABSigmaController(num_downsample_layers=4)

    with torch.no_grad():
        controller.deltas.copy_(
            torch.tensor([-3.0, 0.2, 2.0, -0.7])
        )

    sigmas = torch.stack(
        [controller.get_sigma(i) for i in range(controller.num_layers)]
    )

    assert torch.all(sigmas[1:] > sigmas[:-1])


@pytest.mark.parametrize("kernel_size", [3, 5, 7])
@pytest.mark.parametrize("sigma_value", [0.5, 1.0, 1.5, 2.5])
def test_separable_kernel_matches_direct_2d_gaussian(
    kernel_size,
    sigma_value,
):
    """
    The optimized separable construction must be mathematically equivalent
    to the direct 2-D Gaussian definition.
    """
    blur = GaussianBlur2d(
        channels=1,
        kernel_size=kernel_size,
        padding=kernel_size // 2,
    )

    sigma = torch.tensor(sigma_value, dtype=torch.float32)

    # New implementation: 1-D Gaussian -> outer product.
    separable_kernel = blur._make_kernel(sigma)

    # Independent reference implementation: evaluate the complete 2-D
    # Gaussian directly from x^2 + y^2.
    radius = kernel_size // 2
    coords = torch.arange(
        -radius,
        radius + 1,
        dtype=torch.float32,
    )

    yy, xx = torch.meshgrid(
        coords,
        coords,
        indexing="ij",
    )

    direct_kernel = torch.exp(
        -(xx.square() + yy.square())
        / (2.0 * sigma.square())
    )
    direct_kernel = direct_kernel / direct_kernel.sum()

    torch.testing.assert_close(
        separable_kernel,
        direct_kernel,
        rtol=1e-6,
        atol=1e-7,
    )


@pytest.mark.parametrize("kernel_size", [3, 5, 7])
def test_gaussian_kernel_is_normalized_and_symmetric(kernel_size):
    blur = GaussianBlur2d(
        channels=1,
        kernel_size=kernel_size,
        padding=kernel_size // 2,
    )

    sigma = torch.tensor(1.25)
    kernel = blur._make_kernel(sigma)

    torch.testing.assert_close(
        kernel.sum(),
        torch.tensor(1.0),
        rtol=1e-6,
        atol=1e-7,
    )

    # Symmetry in both spatial dimensions.
    torch.testing.assert_close(kernel, kernel.flip(0))
    torch.testing.assert_close(kernel, kernel.flip(1))


def test_dabpool_matches_independent_reference_convolution():
    """
    Verify the complete DABPool output against an independent direct
    2-D Gaussian reference followed by depthwise stride-2 convolution.
    """
    controller = DABSigmaController(num_downsample_layers=3)

    dab = DABPool(
        channels=2,
        stride=2,
        depth_index=1,  # initial sigma_2 = 1.0
        dab_controller=controller,
        filter_size=3,
        padding=1,
    )

    x = torch.zeros(1, 2, 7, 7, dtype=torch.float32)
    x[0, 0, 3, 3] = 1.0
    x[0, 1, 3, 3] = 2.0

    actual = dab(x)

    sigma = controller.get_sigma(1).detach()

    coords = torch.arange(-1, 2, dtype=torch.float32)
    yy, xx = torch.meshgrid(
        coords,
        coords,
        indexing="ij",
    )

    reference_kernel = torch.exp(
        -(xx.square() + yy.square())
        / (2.0 * sigma.square())
    )
    reference_kernel = reference_kernel / reference_kernel.sum()

    reference_weight = reference_kernel.view(
        1, 1, 3, 3
    ).expand(
        2, 1, 3, 3
    )

    expected = F.conv2d(
        x,
        weight=reference_weight,
        stride=2,
        padding=1,
        groups=2,
    )

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-6,
        atol=1e-7,
    )

    assert actual.shape == (1, 2, 4, 4)


def test_gradient_reaches_learnable_sigma_parameters():
    """
    A DAB layer at depth index 1 uses

        sigma_2 = delta_1 + delta_2

    after the positive softplus parameterization. Therefore the first two
    deltas must receive gradients, while the third (deeper) delta must not.
    """
    controller = DABSigmaController(num_downsample_layers=3)

    dab = DABPool(
        channels=2,
        stride=2,
        depth_index=1,
        dab_controller=controller,
        filter_size=3,
        padding=1,
    )

    torch.manual_seed(0)
    x = torch.randn(
        2,
        2,
        8,
        8,
        dtype=torch.float32,
        requires_grad=True,
    )

    y = dab(x)
    loss = y.square().mean()
    loss.backward()

    grad = controller.deltas.grad

    assert grad is not None
    assert torch.isfinite(grad).all()

    # sigma at depth 1 (the second level) depends on deltas 0 and 1.
    assert grad[0].abs() > 0
    assert grad[1].abs() > 0

    # It does not depend on the increment of the deeper third level.
    torch.testing.assert_close(
        grad[2],
        torch.tensor(0.0),
        rtol=0.0,
        atol=0.0,
    )


def test_dabpool_preserves_channels_and_downsamples_spatially():
    controller = DABSigmaController(num_downsample_layers=3)

    dab = DABPool(
        channels=16,
        stride=2,
        depth_index=0,
        dab_controller=controller,
        filter_size=3,
        padding=1,
    )

    x = torch.randn(4, 16, 64, 64)
    y = dab(x)

    assert y.shape == (4, 16, 32, 32)


def test_shared_controller_is_not_registered_inside_each_pool():
    """
    The parent ResNet owns the shared controller. Individual DABPool modules
    should only hold a plain reference to it, avoiding duplicated controller
    entries below every DABPool in state_dict/module traversal.
    """
    controller = DABSigmaController(num_downsample_layers=3)

    dab = DABPool(
        channels=8,
        stride=2,
        depth_index=0,
        dab_controller=controller,
        filter_size=3,
        padding=1,
    )

    assert "controller" not in dict(dab.named_children())
    assert all(
        not name.startswith("controller.")
        for name, _ in dab.named_parameters()
    )
