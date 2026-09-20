"""The factorised head, and the one identity everything else rests on.

``FactorisedDoseNet`` is only worth its existence if ``sum(dim=(u, v))`` of its
output *is* the quantity its depth branch predicts. If that identity ever breaks,
the IDD loss goes back to acting on a projection -- which is the situation the
architecture exists to leave -- and nothing would raise.
"""

from __future__ import annotations

import pytest
import torch

from models.network import ARCHITECTURES, DoseUNet, FactorisedDoseNet, build_network

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

# Small enough to run on a laptop, still exercising both lateral axes and enough
# depth for the dilated 1-D stack.
SHAPE = (32, 16, 8)


def _net(in_channels: int = 2, **kwargs) -> FactorisedDoseNet:
    torch.manual_seed(0)
    return build_network("factorised", in_channels=in_channels, base_features=4,
                         levels=2, depth_features=8, **kwargs)


def _input(in_channels: int = 2, batch: int = 2) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(batch, in_channels, *SHAPE)


def test_the_lateral_kernel_is_a_distribution_at_every_depth():
    net = _net()
    _, kernel = net.idd_and_kernel(_input())
    assert torch.all(kernel >= 0)
    total = kernel.sum(dim=(3, 4))
    assert torch.allclose(total, torch.ones_like(total), atol=1e-5)


def test_the_lateral_sum_of_the_output_is_exactly_the_predicted_idd():
    """The identity the IDD loss depends on.

    ``idd_distance`` is a lateral sum, and in the beamlet frame the box is
    ray-aligned, so this makes the loss act on a *parameter* of the model rather
    than on a projection of a voxel field.
    """
    net = _net()
    x = _input()
    idd, _ = net.idd_and_kernel(x)
    out = net(x)
    assert torch.allclose(out.sum(dim=(3, 4)), idd, atol=1e-5)


def test_a_zero_depth_slice_is_exactly_zero():
    """Softplus cannot do this, and past the distal fall-off it is most of the box.

    Driven through the head's own output rather than by construction, because
    what matters is that a ReLU'd IDD really does zero the slice: a Softplus
    head would leave a positive floor here and this test would fail.
    """
    net = _net()
    x = _input()
    with torch.no_grad():
        # Force the depth head strongly negative; ReLU must take it to zero.
        net.depth[-1].bias.fill_(-50.0)
    out = net(x)
    assert float(out.abs().max()) == 0.0


def test_the_output_starts_near_zero_but_alive():
    """A head that begins by predicting dose everywhere spends epochs unlearning it.

    The other half matters as much: strictly positive, so no ReLU unit is dead
    at step 0. An all-zero start would have zero gradient and never recover.
    """
    net = _net()
    out = net(_input())
    assert float(out.max()) < 1e-2
    assert float(out.min()) > 0.0


def test_every_parameter_receives_a_gradient():
    """An unused parameter makes DistributedDataParallel raise, and we train on 3-5 GPUs."""
    net = _net()
    net(_input()).sum().backward()
    starved = [name for name, p in net.named_parameters() if p.grad is None]
    assert starved == []


def test_the_trunk_is_headless_so_nothing_is_unused():
    assert _net().trunk.head is None
    with pytest.raises(RuntimeError, match="headless"):
        _net().trunk(_input())


def test_the_unet_state_dict_keys_are_unchanged():
    """`doserad_v2_24x5_best.pt` must keep loading; predictor.py loads strictly."""
    keys = set(DoseUNet(in_channels=2, base_features=8, levels=2).state_dict())
    assert any(k.startswith("encoders.0.") for k in keys)
    assert "head.weight" in keys and "head.bias" in keys
    assert not any(k.startswith("trunk.") for k in keys)


def test_extra_input_channels_reach_the_depth_branch():
    """The Bragg prior is a depth-shaped feature; pooling it away would waste it.

    The final depth conv is zero-initialised on purpose, which makes the
    branch output a constant at step 0 and would let this test pass on a network
    that never wired the input in at all. So the weights are randomised first:
    what is under test is the wiring, not the initialisation.
    """
    net = _net(in_channels=3)
    with torch.no_grad():
        torch.nn.init.normal_(net.depth[-1].weight, std=0.5)
    x = _input(3)
    baseline, _ = net.idd_and_kernel(x)
    bumped = x.clone()
    bumped[:, 2] += 1.0
    changed, _ = net.idd_and_kernel(bumped)
    assert not torch.allclose(baseline, changed)


def test_build_network_refuses_an_unknown_name():
    """The anti-proliferation guard lives in the ValueError, not in an inventory.

    Deliberately does NOT pin the contents of ``ARCHITECTURES``. The policy --
    iterate through config, add a name only against a measured deficit -- is stated in the error message, where someone adding one
    will read it. Pinning the set would only add a second place to edit.
    """
    for name in ARCHITECTURES:
        assert build_network(name, in_channels=2, base_features=4, levels=1)
    with pytest.raises(ValueError, match="unknown architecture"):
        build_network("v7_attention_gates")


def test_conditioning_starts_the_kernel_at_the_analytic_prior():
    """At init the head predicts a residual of zero, so the kernel IS the prior.

    That is the whole point of adding ``log(prior)`` before the softmax rather
    than concatenating the prior as one more feature: a feature has to be learned
    from, a log-space offset is already the answer.
    """
    net = _net(in_channels=3, condition_lateral=True)
    torch.manual_seed(3)
    x = torch.randn(2, 3, *SHAPE)
    prior = torch.rand(2, 1, *SHAPE) + 1e-6
    prior = prior / prior.sum(dim=(3, 4), keepdim=True)
    x[:, -1:] = prior

    _, kernel = net.idd_and_kernel(x)
    # Exact, because the residual head is zero-initialised when conditioning.
    assert torch.allclose(kernel, prior, atol=1e-6)


def test_conditioning_survives_a_prior_that_underflows_to_zero():
    """The analytic kernel is 0 far out; log(0) would poison the whole row.

    A softmax row is normalised across all 1024 lateral voxels, so a single -inf
    does not stay local -- it makes the entire depth slice nan.
    """
    net = _net(in_channels=3, condition_lateral=True)
    x = torch.zeros(1, 3, *SHAPE)
    x[:, -1] = 0.0  # every lateral voxel underflowed
    out = net(x)
    assert torch.isfinite(out).all()


def test_depth_conditioning_starts_the_curve_at_the_bragg_prior():
    """At init the residual is zero, so IDD is exactly ``a*Z + b``.

    The depth-axis twin of the kernel test above. `Z` is the Bragg curve, and
    what makes it worth conditioning on rather than merely feeding in is that the
    depth branch otherwise rebuilds that curve from trunk features every forward
    pass -- while the prior itself carried most of the gain in our input-channel
    ablations.
    """
    net = _net(in_channels=3, condition_depth=True, bragg_index=2)
    torch.manual_seed(4)
    x = torch.randn(2, 3, *SHAPE)
    x[:, 2] = torch.rand(2, *SHAPE)          # a unit-ish Bragg channel

    idd, _ = net.idd_and_kernel(x)
    u0 = (SHAPE[1] - net.core_u) // 2
    v0 = (SHAPE[2] - net.core_v) // 2
    core = x[:, 2:3, :, max(u0, 0):max(u0, 0) + net.core_u,
              max(v0, 0):max(v0, 0) + net.core_v].mean(dim=(3, 4))
    expected = torch.relu(net.prior_scale * core + net.prior_bias)
    assert torch.allclose(idd, expected, atol=1e-6)


def test_depth_conditioning_reads_the_channel_it_was_told_to():
    """Conditioning on the wrong channel does not raise -- it trains to a worse answer.

    `bragg` moves with `with_wepl`, unlike `lateral` which is pinned last, so the
    index is passed in. This pins that it is actually honoured: the same input
    with a different `bragg_index` must give a different curve.
    """
    torch.manual_seed(5)
    x = torch.randn(1, 4, *SHAPE)
    curves = []
    for index in (2, 3):
        net = _net(in_channels=4, condition_depth=True, bragg_index=index)
        with torch.no_grad():
            curves.append(net.idd_and_kernel(x)[0].clone())
    assert not torch.allclose(curves[0], curves[1])


def test_depth_conditioning_refuses_to_guess_the_channel():
    with pytest.raises(ValueError, match="bragg_index"):
        _net(in_channels=3, condition_depth=True)
    with pytest.raises(ValueError, match="outside"):
        _net(in_channels=3, condition_depth=True, bragg_index=7)


def test_an_unconditioned_build_cannot_load_a_conditioned_checkpoint():
    """`prior_scale`/`prior_bias` have nowhere to go, and silently dropping them
    would predict a flat-started curve from a checkpoint trained on a prior."""
    conditioned = _net(in_channels=3, condition_depth=True, bragg_index=2)
    plain = _net(in_channels=3)
    with pytest.raises(RuntimeError):
        plain.load_state_dict(conditioned.state_dict())


def test_the_alive_floor_survives_a_prior_that_is_zero_everywhere():
    """Past the Bragg peak the prior is exactly 0, and `a*Z` there is 0 too.

    Without `prior_bias` those depths would start at ReLU(0), whose gradient is
    zero -- so the distal fall-off, which is precisely what `idd_distance`
    scores, could never lift off. This pins the floor, not the zero-init of
    `depth[-1]`: that is shared with the unconditioned arm and deliberate.
    """
    net = _net(in_channels=3, condition_depth=True, bragg_index=2)
    x = torch.randn(2, 3, *SHAPE)
    x[:, 2] = 0.0                      # prior underflowed everywhere
    idd, _ = net.idd_and_kernel(x)
    assert torch.all(idd > 0), "a zero prior must not zero the whole curve"

    net(x).sum().backward()
    assert net.prior_bias.grad is not None and net.prior_bias.grad.abs() > 0
    assert torch.isfinite(net.depth[-1].weight.grad).all()


def test_the_prior_scale_learns_whenever_the_prior_carries_anything():
    """`a` is what lets the model rescale a unit-peak, dimensionless curve."""
    net = _net(in_channels=3, condition_depth=True, bragg_index=2)
    x = torch.randn(2, 3, *SHAPE)
    x[:, 2] = torch.rand(2, *SHAPE)
    net(x).sum().backward()
    assert net.prior_scale.grad.abs() > 0
