"""Dose prediction network for a single proton beamlet.

Operates on the beamlet-aligned cuboid produced by `models.geometry`: axis 0 is
depth along the ray, axes 1 and 2 are lateral. That alignment is what makes a
modest network viable -- the beamlet is centred by construction, so capacity
goes into modelling dose deposition rather than into locating the beam.

Design notes
------------
*Anisotropic pooling.* The default grid is (384, 64, 16) at (1, 1, 3) mm (24 on
the last axis in the released models). Pooling
equally on every axis would collapse the 16-voxel lateral-v axis after four
levels while barely touching depth. Downsampling is therefore per-axis, keeping
v intact until the deeper levels.

*Non-negative output.* Dose is non-negative. The voxel head uses softplus,
which strands no dead units on a target that is zero over most of its volume;
the factorised head's depth branch uses ReLU instead, so that a zero depth
slice zeroes a whole slab past the distal fall-off.

*No corridor gate.* Multiplying the output by a mask derived from ground-truth
dose would be unusable at inference. Localization comes from the beamlet frame
instead, so no such gate is needed or wanted.

Runtime carries double weight in the challenge ranking, so `base_features` and
`levels` are deliberately modest; scale them only against measured accuracy
gains.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two 3x3x3 convolutions with GroupNorm.

    GroupNorm rather than BatchNorm: beamlet batches are small when the crop is
    this large, and BatchNorm statistics get noisy exactly where dose gradients
    are steepest.
    """

    def __init__(self, in_channels: int, out_channels: int, groups: int = 8) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(min(groups, out_channels), out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(min(groups, out_channels), out_channels)
        self.skip = (
            nn.Conv3d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = F.silu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.silu(x + residual)


class DoseUNet(nn.Module):
    """Encoder-decoder over a beamlet cuboid.

    Parameters
    ----------
    in_channels:
        2 for ``[CT, energy]``; 3 with a third channel appended -- the Bragg
        prior in the released models, or WEPL.
    base_features:
        Width at the finest level, doubling each level down.
    levels:
        Number of downsampling steps.
    strides:
        Per-level stride as ``(depth, u, v)``. The default protects the
        16-voxel v axis for the first two levels.
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_features: int = 24,
        levels: int = 4,
        strides: tuple[tuple[int, int, int], ...] = (
            (2, 2, 1),
            (2, 2, 1),
            (2, 2, 2),
            (2, 2, 2),
            (2, 2, 2),
        ),
        with_head: bool = True,
    ) -> None:
        super().__init__()
        if len(strides) < levels:
            raise ValueError(f"need {levels} strides, got {len(strides)}")
        self.levels = levels
        self.strides = strides[:levels]

        widths = [base_features * 2**i for i in range(levels + 1)]

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        channels = in_channels
        for level in range(levels):
            self.encoders.append(ConvBlock(channels, widths[level]))
            self.downs.append(
                nn.Conv3d(
                    widths[level],
                    widths[level],
                    kernel_size=self.strides[level],
                    stride=self.strides[level],
                )
            )
            channels = widths[level]

        self.bottleneck = ConvBlock(widths[levels - 1], widths[levels])

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for level in reversed(range(levels)):
            self.ups.append(
                nn.ConvTranspose3d(
                    widths[level + 1],
                    widths[level],
                    kernel_size=self.strides[level],
                    stride=self.strides[level],
                )
            )
            self.decoders.append(ConvBlock(widths[level] * 2, widths[level]))

        self.out_channels = widths[0]
        # ``with_head=False`` leaves the trunk headless for FactorisedDoseNet,
        # which supplies its own. It is not an unused module: an unused
        # parameter makes DistributedDataParallel raise for want of a gradient,
        # so the head has to be absent rather than merely ignored.
        self.head = nn.Conv3d(widths[0], 1, kernel_size=1) if with_head else None
        if self.head is not None:
            # Most of the target volume is exactly zero, but softplus(0) is 0.69,
            # so a default head starts by predicting dose everywhere and spends
            # early epochs unlearning it. Biasing the head negative starts the
            # output near zero instead: softplus(-5) is about 0.0067.
            nn.init.zeros_(self.head.weight)
            nn.init.constant_(self.head.bias, -5.0)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Decoder output at full resolution, ``(B, base_features, D, U, V)``.

        Split out of :meth:`forward` so a different head can be attached without
        moving any parameter into a submodule -- the state-dict keys are the
        ones the released checkpoints were written with, and
        ``models/predictor.py`` loads strictly.
        """
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)

        x = self.bottleneck(x)

        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            # Transposed convolutions can undershoot by a voxel on odd sizes.
            if x.shape[-3:] != skip.shape[-3:]:
                x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear",
                                  align_corners=False)
            x = decoder(torch.cat([x, skip], dim=1))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.head is None:
            raise RuntimeError(
                "this DoseUNet was built headless (with_head=False); call "
                "features() or wrap it in FactorisedDoseNet"
            )
        # Dose is non-negative; Softplus keeps gradients alive in the large
        # zero region that ReLU would kill.
        return F.softplus(self.head(self.features(x)))

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class FactorisedDoseNet(nn.Module):
    """Dose as ``IDD(depth) x lateral_kernel(u, v | depth)``.

    Why factorise
    -------------
    A proton beamlet is a depth-dose curve times a lateral spread, and the
    U-Net is made to rediscover that from voxels. Writing it down buys three
    things, in descending order of how much they are worth:

    1. **``idd_distance`` becomes an output, not a projection.** In the beamlet
       frame the box is ray-aligned, so the integrated depth dose is exactly
       ``sum(dim=(u, v))`` -- and because the lateral kernel is a softmax it
       sums to one at every depth, making that sum *identically* ``IDD(d)``. A
       per-voxel loss cannot control a projection: errors along the summed axis
       cancel or accumulate. Here the loss reaches the quantity directly.
    2. **Exact zeros.** ``IDD(d) = 0`` zeroes a whole depth slice, which Softplus
       on a voxel head can never do -- and past the distal fall-off that is most
       of the box. Hence ReLU here where the voxel head needs Softplus: the
       argument for Softplus was gradient death on a mostly-zero *volume*, and a
       depth curve is not mostly zero over the range that carries dose.
    3. It is the physics we already hold, rather than capacity spent relearning it.

    The head costs a 1x1x1 convolution -- the same one the voxel head was -- plus
    a three-layer 1-D stack over depth, so at the trunk widths we train it is a
    fraction of a percent on top of the U-Net. That is a fact about the
    current settings, not a constraint: ``depth_features`` scales it freely, and
    an arm that grows it needs a baseline grown to match before an accuracy
    difference means anything.

    What it does **not** assume
    ---------------------------
    Not separability. A rank-1 ``depth (x) fixed lateral shape`` factorisation
    loses **14% in L2** on real beamlets (median over 90, ``p90 = 26%``), because
    lateral sigma grows from ~3.9 mm at entry to ~6.9 mm at the peak. The kernel
    here is **free at every depth**, so the factorisation is an identity, not an
    approximation -- it re-parameterises the output, it does not restrict it.

    Not a Gaussian. A single Gaussian leaves a **12% L1 residual** on the
    measured lateral profile (median), which is the nuclear halo the machine
    models with a *second* Gaussian. And the lateral centroid drifts with depth
    (median 6.4 mrad, 12.5 mrad on central rays) as heterogeneity deflects the
    beam, so a kernel pinned to the axis would be wrong too.

    The depth branch sees the trunk's features pooled laterally, plus the input
    channels pooled over the **central core** only -- out at +-32 mm the box is
    mostly tissue the beamlet never reaches, and averaging that in would let
    anatomy carrying no dose move the predicted range.
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_features: int = 24,
        levels: int = 4,
        depth_features: int = 64,
        core_u: int = 16,
        core_v: int = 6,
        condition_lateral: bool = False,
        condition_depth: bool = False,
        bragg_index: int | None = None,
        **trunk_kwargs,
    ) -> None:
        super().__init__()
        self.trunk = DoseUNet(
            in_channels=in_channels,
            base_features=base_features,
            levels=levels,
            with_head=False,
            **trunk_kwargs,
        )
        self.core_u, self.core_v = core_u, core_v
        # When set, the LAST input channel is the analytic double-Gaussian
        # lateral kernel and the head predicts a residual on it in log space --
        # so at initialisation the kernel IS the analytic one. Measured L1
        # residual 0.102 abdominal / 0.141 thoracic, which is a
        # good starting shape and an insufficient final one, hence residual
        # rather than replacement.
        #
        # "last channel" is a contract with `models/dataset.py`, which appends
        # it last, and with the checkpoint's `channels` list. Pinned by
        # `tests/test_factorised_network.py`.
        self.condition_lateral = condition_lateral
        width = self.trunk.out_channels

        # Lateral: logits over (u, v) per depth, normalised by softmax. One 1x1x1
        # convolution, so the lateral kernel costs essentially what the old voxel
        # head cost.
        self.lateral = nn.Conv3d(width, 1, kernel_size=1)
        if condition_lateral:
            # Zero the residual head so the kernel starts *exactly* at the
            # analytic prior rather than merely near it -- otherwise the default
            # init puts logits at std ~0.3 and the starting kernel is already
            # 3% away from the shape we went to the trouble of computing.
            #
            # Only when conditioning. Zeroing it unconditionally would give
            # the plain factorised arm a uniform starting kernel, changing the
            # arm that is currently the reference for this comparison.
            nn.init.zeros_(self.lateral.weight)
            nn.init.zeros_(self.lateral.bias)

        # Depth: mean and max of the trunk features over (u, v), plus the input
        # channels over the core. Dilated so a 3-layer stack still sees ~29 mm of
        # depth, which is the scale a Bragg peak moves on.
        depth_in = 2 * width + in_channels
        self.depth = nn.Sequential(
            nn.Conv1d(depth_in, depth_features, 5, padding=2, dilation=1),
            nn.GroupNorm(min(8, depth_features), depth_features),
            nn.SiLU(),
            nn.Conv1d(depth_features, depth_features, 5, padding=4, dilation=2),
            nn.GroupNorm(min(8, depth_features), depth_features),
            nn.SiLU(),
            nn.Conv1d(depth_features, 1, 5, padding=8, dilation=4),
        )
        # When set, the depth branch predicts a residual on the Bragg prior
        # instead of the curve from scratch: `IDD = relu(a*Z + b + residual)`,
        # the depth-axis twin of what `condition_lateral` does laterally.
        #
        # Why this and not "the network already sees the channel": it does --
        # `core` below feeds every input channel into the depth stack. But the
        # prior is the strongest single ingredient we have (it carried most of
        # the gain in our input-channel ablations) and the branch still
        # has to *rebuild* the curve from those features every forward pass.
        # Conditioning makes the curve the starting point rather than the target.
        #
        # `bragg_index` is the position in the channel list, passed in rather
        # than assumed: unlike `lateral` (pinned last), `bragg` moves depending
        # on whether `wepl` is present, and reading the wrong channel here does
        # not raise -- it conditions the depth curve on CT or energy and trains
        # perfectly happily to a worse answer.
        self.condition_depth = condition_depth
        self.bragg_index = bragg_index
        if condition_depth:
            if bragg_index is None:
                raise ValueError(
                    "condition_depth=True needs bragg_index -- the depth curve "
                    "is conditioned on the Bragg channel and nothing else "
                    "identifies which channel that is"
                )
            if not 0 <= bragg_index < in_channels:
                raise ValueError(
                    f"bragg_index={bragg_index} is outside the {in_channels} "
                    "input channels"
                )
            # a and b of `a*Z + b`, learnable: the prior is unit-peak and
            # dimensionless, so *something* has to carry the scale, and fixing it
            # would make the arm a test of our guess at that scale rather than of
            # the conditioning.
            self.prior_scale = nn.Parameter(torch.tensor(0.1))
            self.prior_bias = nn.Parameter(torch.tensor(0.01))

        # Start flat and barely positive: zero weights make the curve uniform,
        # and a small positive bias keeps every ReLU unit alive at step 0. With
        # the kernel summing to 1 over ~1024 lateral voxels this puts the initial
        # per-voxel output at ~1e-4, the same near-zero start softplus(-5) gives
        # the voxel head -- so the model does not begin by unlearning dose
        # everywhere.
        nn.init.zeros_(self.depth[-1].weight)
        # Conditioned, the alive-floor is `prior_bias` and the shape is the
        # prior, so the residual starts at exactly zero -- 0.1 here would add a
        # flat pedestal to the very curve we are conditioning on. `prior_scale`
        # 0.1 keeps the *starting magnitude* identical to the unconditioned arm,
        # so this arm differs from its baseline in the curve's SHAPE only.
        nn.init.constant_(self.depth[-1].bias, 0.0 if condition_depth else 0.1)

    def idd_and_kernel(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(idd, kernel)`` -- ``(B, 1, D)`` and ``(B, 1, D, U, V)``.

        Both are float32 regardless of autocast: the softmax runs over 1024
        lateral elements, and bf16 there costs precision in the tail of the
        kernel for no speed worth having on a 1x1x1 head.
        """
        features = self.trunk.features(x)

        logits = self.lateral(features).float()
        if self.condition_lateral:
            # Adding log(prior) before the softmax is a *multiplicative* residual
            # on the prior. The clamp keeps the far tail finite: the analytic
            # kernel underflows to 0 out at +-32 mm and log(0) would poison the
            # whole softmax row, not just that voxel.
            prior = x[:, -1:].float().clamp_min(1e-12)
            logits = logits + torch.log(prior)
        b, c, d, u, v = logits.shape
        kernel = torch.softmax(logits.reshape(b, c, d, u * v), dim=-1)
        kernel = kernel.reshape(b, c, d, u, v)

        n_u, n_v = x.shape[-2], x.shape[-1]
        u0 = max((n_u - self.core_u) // 2, 0)
        v0 = max((n_v - self.core_v) // 2, 0)
        core = x[..., u0:u0 + self.core_u, v0:v0 + self.core_v].mean(dim=(3, 4))
        pooled = torch.cat(
            [features.mean(dim=(3, 4)), features.amax(dim=(3, 4)), core.to(features.dtype)],
            dim=1,
        )
        # ReLU, not Softplus: an exactly-zero depth slice is the point (see the
        # class docstring), and softplus cannot produce one.
        raw = self.depth(pooled).float()
        if self.condition_depth:
            # The core-pooled Bragg channel IS Z(wepl(d)) as a function of depth,
            # which is the curve to start from. Pooled over the core rather than
            # the whole box for the reason the class docstring gives: out at
            # +-32 mm the box is tissue the beamlet never reaches.
            prior = core[:, self.bragg_index:self.bragg_index + 1].float()
            raw = raw + self.prior_scale * prior + self.prior_bias
        idd = F.relu(raw)
        return idd, kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idd, kernel = self.idd_and_kernel(x)
        return idd[..., None, None] * kernel

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


ARCHITECTURES = {"unet": DoseUNet, "factorised": FactorisedDoseNet}


def build_network(name: str = "unet", **kwargs) -> nn.Module:
    """Build an architecture by name.

    Two names, and the bar for a third is a *measured* deficit, not an idea. ``factorised`` earns its place by changing what the loss can
    reach rather than by adding capacity: it makes the integrated depth dose an
    output, which is the quantity ``idd_distance`` scores and the one a per-voxel
    loss cannot control.
    """
    if name not in ARCHITECTURES:
        raise ValueError(
            f"unknown architecture {name!r}; have {sorted(ARCHITECTURES)}. "
            "Iterate through config rather than adding v0..vN modules -- a "
            "family of near-identical modules is hard to reason about."
        )
    return ARCHITECTURES[name](**kwargs)
