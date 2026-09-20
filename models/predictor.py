"""Inference contract: beam metadata + CT in, per-beamlet dose on the CT grid out.

This is the seam between the two halves of the project. The submission
container owns I/O (reading the mounted `.mha` inputs, writing `JoinSeries`
stacks to the ten output slots); everything here owns geometry and the model.
Neither side needs to know the other's internals, so they can be built in
parallel.

**The symmetry guarantee.** Training and inference both build their input
through `models.geometry`, so there is exactly one implementation of "where is
this beamlet and what does the network see". Two separate implementations is
the failure this guards against: both can silently read ground-truth dose, and
a single path is what prevents it. Nothing in this
module can accept a dose array -- there is no parameter for one.

:class:`StubModel` produces plausibly shaped output so the container can be
built and exercised end to end without weights (``DOSERAD_ALLOW_STUB=1``). Its
numbers are meaningless; it exists so plumbing bugs surface without a model.

Five ways this goes wrong without raising
-----------------------------------------
1. **A ray that misses yields an all-zero volume, never a dropped entry.** A gap
   would shift ``idx_in_output`` for every beam after it, so each prediction
   would be scored against the wrong ground truth.
2. **A checkpoint carries its own architecture, channels, ``dose_scale`` and
   sampling box**, because the container ships no config. Two of those are
   *refused* on mismatch rather than trusted: wrong ``in_channels`` feeds a
   WEPL-trained network 2 channels and produces wrong dose without raising, and
   a wrong ``grid`` moves every beamlet's dose (entry depth anchors the box).
   Guarded by ``test_from_checkpoint_rejects_a_box_it_was_not_trained_with``.
3. **``body_mask`` defaults to True, which IS what ships** --
   ``submission/Dockerfile`` sets ``ARG BODY_MASK=1``.
   **AND THE MASK IS :func:`~models.geometry.body_contour`, NOT
   ``ct > AIR_HU``.** The bare threshold deletes internal air stored at the
   reconstruction floor, where their contour-masked labels have dose; an earlier
   submission shipped that variant and ``idd`` paid for it. ``body_mask=False``
   is therefore *not* the safe option -- it merely swaps a large error for a
   small one. Masking an already-masked prediction is a no-op, so raw and
   masked predictions agreeing is evidence the predictions were masked, not
   evidence that masking is irrelevant.
   Scoring code should pass
   ``body_mask=False`` on purpose -- they score the *rendered* dose and the mask
   belongs to the inference path, not to the metric.
4. **Units.** Training divides the label by ``dose_scale`` (1e-3), so the
   network emits scaled units while rendering and the ``minimum_cutoff`` clamp
   expect physical dose. :class:`TorchModel` applies the multiply; without it
   predictions come out ~1000x too large, nothing falls below cutoff, and every
   dose map counts as an implementation error on a published metric. Guarded by
   ``tests/test_predictor_checkpoint.py``.
5. **Clamping at the cutoff is not the test the evaluator runs.** Their check
   is in float64 and a torch clamp against a float32 tensor is in float32, and
   the gap between the two is what cost an early submission 5 implementation
   errors. The threshold therefore carries a margin -- see
   :func:`clamp_threshold`, guarded by ``tests/test_clamp_margin.py``.
"""

from __future__ import annotations

import collections
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import geometry_torch as GT
from .geometry import (
    AIR_HU,
    BeamletGrid,
    body_contour,
    VolumeGeometry,
    beam_frame,
    find_entry_depth_box,
)


@dataclass(frozen=True)
class BeamletRequest:
    """One dose map the submission must produce.

    Mirrors the submission metadata exactly: everything here is supplied by the
    platform. ``output_file_idx`` / ``idx_in_output`` say where the result goes;
    ``minimum_cutoff`` is the value below which the written output must be zero.
    """

    ray_source: tuple[float, float, float]
    ray_target: tuple[float, float, float]
    energy: float
    output_file_idx: int
    idx_in_output: int
    minimum_cutoff: float = 0.0

    @classmethod
    def from_metadata(cls, ray: dict, beamlet: dict) -> "BeamletRequest":
        info = beamlet["output_info"]
        return cls(
            ray_source=tuple(ray["ray_source"]),
            ray_target=tuple(ray["ray_target"]),
            energy=float(beamlet["energy"]),
            output_file_idx=int(info["output_file_idx"]),
            idx_in_output=int(info["idx_in_output"]),
            minimum_cutoff=float(info.get("minimum_cutoff", 0.0)),
        )


#: Relative margin added to a beamlet's ``minimum_cutoff`` before clamping.
#: Sized in ``clamp_threshold``; 1e-5 of a ~1e-6 cutoff is ~1e-8 of the peak.
CUTOFF_MARGIN = 1.0e-5


#: Snap the TOP of the sub-cutoff band up to the threshold instead of down to
#: zero. **NOT `c/2`.** `idd` is an L2 over the *laterally integrated* curve, so
#: sub-cutoff errors cancel rather than accumulate: across a depth slice, writing
#: zero over truth mass `T` contributes `−T` and snapping `k` voxels contributes
#: `+k·c`, so the optimum is **count-matching, `k ≈ T/c`**.
#: **That argument needs a truth that HAS dose below the cutoff.** The challenge's
#: scored labels are stored already thresholded at the served cutoff, so the band
#: holds no truth mass and snapping up only adds error: against those labels the
#: snap makes `idd` worse. ⇒ **the submitted model runs with `snap_alpha=0`**,
#: which disables the snap and zeroes the whole band (see `apply_cutoff_policy`).
#: Note the discontinuity: `0` means *off*, while a small positive α snaps almost
#: the entire band *up*. 0.15 is the measured optimum **against unthresholded
#: labels**; it is not what shipped, so `DosePredictor` defaults to 0 and
#: `submission/Dockerfile` passes `SNAP_ALPHA=0` explicitly.
SNAP_ALPHA = 0.15


def clamp_threshold(minimum_cutoff: float, margin: float = CUTOFF_MARGIN) -> float:
    """The float32 threshold to zero below, so **nothing** survives under
    ``minimum_cutoff`` when the evaluator re-reads the output in float64.

    **Clamping at the cutoff itself is not the same test the evaluator runs.** Our
    clamp is ``block < minimum_cutoff`` on a *float32* tensor and torch casts the
    Python float to the tensor's dtype, so we actually threshold at
    ``float32(c)``. For about **half of cutoffs of the size served (~1e-6)** ``float32(c) < c``, which
    opens a keep-window ``[float32(c), c)`` whose only representable float32
    value is ``float32(c)`` itself. Their `extract_beam` upcasts to float64
    before the check (`evaluate.py:939`), so a voxel sitting exactly on
    ``float32(c)`` is ``> 0 and < c`` for them and survives for us. Measured on
    that submission's own output: **exactly 5 frames, one voxel each,
    every one equal to float32(cutoff), every one with float32(c) < c** — the
    count the platform reported.

    **Why a margin rather than a float64 comparison.** 0.5 ULP of float32 —
    6e-8 relative — closes the mechanism above, and this is 170x that. The extra
    buys the sources we cannot see: their cutoff is read from ``run_manifest``
    and ours from the input JSON, and a value that differs in its 6th
    significant digit would reopen the same window with none of the evidence.
    **The margin is not free-floating** — it is bounded by
    ``test_clamp_margin.py`` at both ends, because raising it deletes real dose
    and lowering it re-opens the window.

    Cost: dose in ``[c, c*(1+1e-5))`` is zeroed. The cutoff is ~1e-3 of the
    beamlet peak, so that band is ~1e-8 of peak — below every metric's
    resolution, and below the MC noise floor by five orders of magnitude.
    """
    if minimum_cutoff <= 0:
        return 0.0
    return float(np.float32(minimum_cutoff * (1.0 + margin)))


def apply_cutoff_policy(block, minimum_cutoff: float, snap_alpha: float = 0.0):
    """The platform's ``0 or >= c`` rule, applied three ways.

    **Only two values are legal below the cutoff: zero, and the threshold.** For
    a voxel whose truth is ``t``, predicting 0 costs ``t`` and predicting the
    threshold costs ``|threshold - t|`` — so the *bottom* of the band is cheaper at
    zero and the *top* is cheaper snapped up. Clamping the whole band down, which
    is what shipped, is right whenever the truth is itself thresholded at the
    cutoff -- which the challenge's scored labels are (see ``SNAP_ALPHA``).

    **The snap target is the THRESHOLD, not ``c``.** It is the same value the
    zero-arm tests against, so a snapped voxel provably passes the check the
    evaluator runs, with the margin argument in :func:`clamp_threshold` behind it —
    rather than sitting one ULP above ``float32(c)`` and re-opening the window
    the evaluator counts as an implementation error.

    **A function and not four inline lines** because the predictor's own stub
    cannot exercise it: the stub emits a 1.4x dynamic range and a real beamlet
    spans ~1e3, so nothing ever lands in the band when it is driven end to end.
    ⇒ the arms are unit-tested here instead of being asserted vacuously.
    """
    import torch

    if minimum_cutoff <= 0:
        return block
    t = clamp_threshold(minimum_cutoff)
    if snap_alpha <= 0.0:
        return torch.where(block < t, block.new_zeros(()), block)
    return torch.where(
        block < snap_alpha * t,
        block.new_zeros(()),
        torch.where(block < t, block.new_full((), t), block),
    )


class StubModel:
    """Placeholder standing in for a trained network.

    Emits a smooth bump along the beam axis so downstream code sees non-zero,
    non-degenerate dose. It is not a physics model and its output is not
    meaningful -- it only lets the container be validated before a real network
    exists.
    """

    def __init__(self, grid: BeamletGrid | None = None) -> None:
        self.grid = grid or BeamletGrid()

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        # batch: (B, C, depth, u, v); channel 1 carries normalized energy.
        n, _, depth, n_u, n_v = batch.shape
        energy = batch[:, 1, 0, 0, 0]
        # Deeper stopping point for higher energy, purely so output varies.
        peak = np.clip(energy * depth * 0.9, 8.0, depth - 8.0)
        z = np.arange(depth)[None, :]
        profile = np.exp(-0.5 * ((z - peak[:, None]) / 18.0) ** 2)
        lateral = np.exp(
            -0.5 * ((np.arange(n_u) - (n_u - 1) / 2) / 6.0) ** 2
        )[None, :, None] * np.exp(
            -0.5 * ((np.arange(n_v) - (n_v - 1) / 2) / 2.0) ** 2
        )[None, None, :]
        return (profile[:, :, None, None] * lateral[:, None, :, :] * 1e-3).astype(
            np.float32
        )


def resolve_autocast_dtype(name: str, device: str, torch):
    """``"fp16"``/``"bf16"``/``"fp32"`` -> a torch dtype, or ``None`` for fp32.

    **Refuses rather than falling back.** ``bf16`` on pre-Ampere silicon is
    the 9.2x trap measured on a Quadro RTX 6000: ``is_bf16_supported()`` answers True
    on compute 7.5 and the arithmetic is emulated, so a silent fallback would
    report a *slower* container as a precision win and an A/B arm as a null.
    The same logic makes a CPU fallback an error: this knob exists to move the
    forward pass onto tensor cores, and a run that quietly did not is a
    measurement of nothing.
    """
    name = (name or "fp32").lower()
    if name in ("fp32", "float32", "none", ""):
        return None
    if name not in ("fp16", "float16", "bf16", "bfloat16"):
        raise ValueError(
            f"unknown inference precision {name!r}; expected fp32, fp16 or bf16"
        )
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise ValueError(
            f"precision {name!r} needs a CUDA device; got device={device!r}. "
            "Autocast on CPU would run in float32 and report a null."
        )
    if name.startswith("bf"):
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            raise ValueError(
                "bf16 is emulated below compute 8.0 (9.2x slower "
                f"than fp16 on 7.5); this device is compute "
                f"{'.'.join(map(str, torch.cuda.get_device_capability()))}. "
                "Use fp16 here."
            )
        return torch.bfloat16
    return torch.float16


class TorchModel:
    """A trained network wrapped to satisfy the model callable contract.

    Two conversions happen here and both matter.

    *Units.* Training divides the stored label by ``dose_scale`` (1e-3), so the
    network's output is in scaled units while everything downstream --
    ``render_to_volume``, and especially the ``minimum_cutoff`` clamp -- expects
    physical dose. Without the multiply, predictions are ~1000x too large, no
    voxel falls below its cutoff, and the platform counts the result as an
    implementation error. The scale is read from the checkpoint rather than
    imported so a checkpoint trained under a different one stays correct.

    *Framework.* The predictor is otherwise pure numpy; torch is imported lazily
    so the stub path keeps working in an environment without it.

    *Precision.* ``autocast`` selects the dtype the **forward pass** runs in;
    everything before and after it stays float32, and ``.float()`` below is
    what makes that true whatever the network returns. It exists because the
    forward pass is where the scored time is on the hardware the platform uses:
    measured 2026-08-24, the forward is
    **68% of device time on a Quadro RTX 6000** against 58% on an A100, and the
    device half is 75% of ``predict`` there -- so an A100 profile understates
    this lever badly.

    **A dtype is not free the way a batch size is.** Every other runtime
    change on this path was bit-identical; this one moves voxel values, so it is
    an accuracy question before it is a
    runtime one.
    """

    def __init__(
        self, net, device: str, dose_scale: float, autocast: str | None = None,
        cudnn_benchmark: bool = False, channels_last: bool = False,
        compile_mode: str | None = None,
        aoti_package: str | None = None,
        aoti_identity: dict | None = None,
    ) -> None:
        import torch

        self._torch = torch
        self.net = net
        self.device = device
        self.dose_scale = dose_scale
        self.autocast = (autocast or "fp32").lower()
        self.autocast_dtype = resolve_autocast_dtype(self.autocast, device, torch)
        self.cudnn_benchmark = bool(cudnn_benchmark)
        # Both of these change *how* the forward pass is computed, not what it
        # computes -- so both are ratio questions for a real A10G and neither may
        # be adopted on another card's evidence: through this predictor, fp16
        # took 0.59x on one development GPU and 1.44x on the A10G, while a
        # plain forward with cuDNN autotuning took 0.83x (paper §3.2). The
        # setups differ, which is why each knob is measured per card and path.
        self.channels_last = bool(channels_last) and str(device).startswith("cuda")
        if self.channels_last:
            # NDHWC. 3-D convolutions reach tensor cores through this layout and
            # not through NCDHW, which is the standing suspicion about why fp16
            # lost: the dtype was right and the layout was not.
            self.net = self.net.to(memory_format=torch.channels_last_3d)
        # **AOTI FIRST, AND IT REPLACES COMPILATION RATHER THAN JOINING IT.**
        # A loaded package is `torch.compile`'s win without its three costs --
        # no per-job compile before /health, no write-then-execute on the
        # scratch, and batch 1 is a size rather than a recompile (`models/aoti.py`).
        # A refusal is silent about speed and loud in the log: the package is
        # keyed to the checkpoint, so serving one built for other weights would
        # be wrong rather than slow.
        self.aoti = None
        self.aoti_min_batch = 1
        if aoti_package:
            from models.aoti import load_package, package_identity

            loaded = load_package(aoti_package,
                                  package_identity(aoti_identity or {}),
                                  device=str(device))
            if loaded is not None:
                self.aoti, self._eager_net, self.net = loaded, self.net, loaded
                # The exported range does not reach batch 1 -- `torch.export`
                # specialises 0 and 1 -- so the eager module stays reachable for
                # the sizes below it (`models/aoti.py`, MIN_BATCH).
                self.aoti_min_batch = int(getattr(loaded, "min_batch", 1))
                compile_mode = None

        self.compile_mode = compile_mode or None
        if self.compile_mode:
            # Compilation is minutes, and `submission/inference.py` warms the
            # path **before /health** -- so the cost lands in the platform's
            # untimed window while the benefit lands in every job.
            #
            # **AND IT MUST NEVER BE LOAD-BEARING.** Compilation depends on
            # things about the platform we cannot observe -- whether the scratch
            # mount executes, whether a C compiler is reachable -- and the final
            # phase returns no logs and allows two submissions. So dynamo is put
            # in `suppress_errors` mode: a backend that fails falls back to eager
            # for that graph instead of raising, which is the difference between
            # a slower container and a job that returns 0.
            # `warm_up` handles what this cannot catch.
            try:
                torch._dynamo.config.suppress_errors = True
            except Exception:                                     # noqa: BLE001
                # A guard that raises is worse than no guard. `warm_up` still
                # catches what this would have suppressed.
                pass
            self.net = torch.compile(self.net, mode=self.compile_mode)
        if str(device).startswith("cuda"):
            # **OFF, and measured off** (2026-08-24, on a real A10G).
            # The reasoning for turning it on was that every convolution here
            # runs at one shape, so an autotuner would pay its search once and
            # win afterwards. Both halves were wrong on the card that scores us:
            # autotune buys **exactly nothing** there in fp32 (ratio 1.000 [1.000-1.000]
            # on forward *and* predict), and the shape is **not** single --
            # a job of 500 beamlets is 62 batches of 8 and one of 4.
            # That odd last batch is the whole cost: its first appearance
            # triggers an exhaustive search, measured at **100.3 ms/beamlet
            # against 27.8** in the round that meets it. In the container every
            # job is a fresh process, so it is paid **per job, inside the timed
            # window** -- ~4% of a scored run.
            # Algorithm choice is not bit-identity -- a different algorithm
            # sums in a different order -- so this is scored, not assumed.
            # **Set per forward, not once here**, because a timing harness
            # may interleave arms: a global written at
            # build time would be whatever the last-built arm wanted, and both
            # arms would measure it.
            torch.backends.cudnn.benchmark = self.cudnn_benchmark

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(batch)).to(self.device)
            return self.forward_device(x).cpu().numpy()

    def forward_device(self, batch):
        """Same forward, tensor in and tensor out, staying on the device.

        The numpy ``__call__`` above is the model contract (`models/predictor.py`)
        and every other model honours it. This exists so the predictor can keep
        a whole beamlet on the GPU from sampling through rendering, which is
        worth more than the forward pass itself: the round trips it removes are
        two per beamlet, against ~105 ms of geometry when it runs on the CPU.
        """
        torch = self._torch
        if str(self.device).startswith("cuda"):
            torch.backends.cudnn.benchmark = self.cudnn_benchmark
        if self.channels_last:
            batch = batch.contiguous(memory_format=torch.channels_last_3d)
        with torch.inference_mode():
            if self.autocast_dtype is None:
                y = self._forward(batch)
            else:
                with torch.autocast("cuda", dtype=self.autocast_dtype):
                    y = self._forward(batch)
            if y.ndim == 5:  # (B, 1, D, U, V) -> (B, D, U, V)
                y = y[:, 0]
            # `.float()` is not decoration: under autocast ``y`` is half, and
            # everything downstream -- the cutoff clamp above all -- is float32.
            return y.float() * self.dose_scale




    def _forward(self, batch):
        """The network, with the precompiled path unable to cost a submission.

        **A raise from the fast path is a zero, not a slow job.** `warm_up`
        catches what fails at startup and `load_package` now runs the object
        before returning it (`models/aoti.py`), but neither can promise the
        1,000th call resembles the first -- a shape, a layout or a kernel the
        smoke test did not reach still lands *inside* `/invoke`, where the
        platform sees a 500 and scores nothing. Eager is exact and ~11% slower per
        job (`predict` 0.899x), so this trade is not close.
        """
        if self.aoti is None:
            return self.net(batch)
        # **BELOW THE PACKAGE'S FLOOR, EAGER IS THE FAST PATH.** A package
        # covers batch 2 upward and no export covers 1, so a singleton either
        # runs eager or is padded to 2 -- and padding costs ~2x a batch-1
        # forward to save the 0.855x compilation is worth, i.e. it is slower on
        # precisely the tiny jobs it would be for. It is also the difference
        # between a check that refuses and one that computes with a range it was
        # not built for: the container sets `AOTI_RUNTIME_CHECK_INPUTS`, and
        # this is what keeps that check from ever having to fire.
        if batch.shape[0] < self.aoti_min_batch:
            return self._eager_net(batch)
        try:
            return self.aoti(batch)
        except Exception as exc:                                  # noqa: BLE001
            print(f"[aoti] the precompiled forward RAISED inside a job "
                  f"({type(exc).__name__}: {exc}); dropping it and serving eager "
                  f"for the rest of this container. Its cause is on stderr.",
                  flush=True)
            self.disable_compilation()
            return self.net(batch)

    def disable_compilation(self) -> bool:
        """Put the eager module back, and say whether there was one to put back.

        **The fast path may not cost a submission.** `torch.compile` returns
        an ``OptimizedModule`` wrapping the original as ``_orig_mod`` -- the same
        parameter objects, not a copy -- so abandoning compilation is exact
        rather than approximate and cannot move a single output value.

        Called by `submission/inference.py` when the warmup fails while
        compiled: the first `torch.compile` container died *after* a healthy
        start, inside /invoke, because the warmup swallowed the compile error
        and left the compiled module installed for the real job to meet again.
        """
        # AOTI keeps the eager module beside it rather than inside a wrapper.
        if self.aoti is not None:
            self.net, self.aoti = self._eager_net, None
            return True
        original = getattr(self.net, "_orig_mod", None)
        if original is None:
            return False
        self.net = original
        self.compile_mode = None
        return True


def block_window(lo, hi) -> tuple[slice, slice, slice]:
    """Where a rendered block sits in the volume.

    **This reverses.** ``lo``/``hi`` come out of ``render_block`` in
    ``(x, y, z)`` while the volume is indexed ``(z, y, x)``, so a window written
    without the flip still runs, still has the block's shape, and still writes a
    plausible-looking region — of the wrong voxels. One definition, used by
    everything that indexes a block into a volume, so the two cannot drift.
    """
    return (
        slice(int(lo[2]), int(hi[2])),
        slice(int(lo[1]), int(hi[1])),
        slice(int(lo[0]), int(hi[0])),
    )


class FramePool:
    """Output frames, handed back and reused instead of allocated per beamlet.

    **What it buys, and why it is not the memcpy.** Every beamlet returns a full
    CT-grid dose map — ~85 MB — of which the box writes a median 5.5%, and
    ``np.zeros`` is a ``calloc``: the pages are not there until they are
    touched, so writing a 1.6 MB block over an 11 MB window first-touches ~2700
    pages *per beamlet*. Measured 2026-08-26 on an A10G: fresh calloc **1.47
    ms**, pre-faulted ring 0.48, a frame already hot **0.09** — against 0.09 ms
    for rewriting the same array, which is what says the cost is the faulting
    and not the copy. ⇒ **1.15 ms/beamlet, ~0.6 s of the scored number**, on a
    path where `predict` is fully serialised (event spans sum to 26.38 of a
    26.59 ms wall), so it comes straight off the wall.

    **THE CONTRACT IS THE WHOLE RISK, AND BREAKING IT IS SILENT.** A frame is
    reused only after :meth:`release` is called on it, and releasing one that is
    still owned by anybody — the writer thread, a caller holding a chunk — hands
    the next beamlet an array somebody else is reading: right shape, plausible
    dose, wrong beamlet, nothing raised. So:

    - the pool is **off unless a capacity is asked for**, and the only caller
      that asks is `submission/inference.py`, which releases a frame *after*
      ``writer.add`` has returned and never touches it again;
    - :meth:`release` accepts only frames this pool issued and is a no-op
      otherwise, so a caller that hands back something else cannot poison it;
    - `tests/test_frame_pool.py` writes a whole slot both ways and compares the
      **bytes**, which is the only check that sees a premature recycle.

    **Recycling zeroes the previous box, not the frame.** A returned frame is
    zero everywhere except where its beamlet wrote, and that window is recorded
    by :meth:`mark` at the one place that writes it (`_to_host_volume`). Zeroing
    the window is the 0.09 ms; zeroing all 85 MB would give the allocation back.
    ⇒ a window that under-states what was written leaks dose from one beamlet
    into the next, which is why the mark is taken from the same
    :func:`block_window` the write uses rather than recomputed.

    It holds no more memory than the path already peaks at: the live set is a
    batch plus the write-ahead queue either way, and the cap is sized to that.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = max(int(capacity), 0)
        # (frame, dirty window or None). A deque so the least recently used
        # frame comes back first -- every frame in the pool stays warm.
        self._free: collections.deque = collections.deque()
        # Issued frames, held BY REFERENCE and keyed by id: CPython recycles
        # ids, and a dead array's id landing on a live one is exactly the
        # aliasing this class exists not to do. Holding the reference makes the
        # id stable for as long as the entry lives -- the same rule the entry
        # and render memos use.
        self._issued: dict[int, np.ndarray] = {}
        self._dirty: dict[int, tuple] = {}
        self._lock = threading.Lock()
        self.reused = 0
        self.allocated = 0

    def take(self, shape) -> np.ndarray:
        """A zero frame of ``shape``, recycled when one is free."""
        shape = tuple(int(s) for s in shape)
        frame = dirty = None
        with self._lock:
            while self._free:
                candidate, window = self._free.popleft()
                # A different image grid: drop it rather than keep a frame no
                # caller can use. Slots do not mix images (`inference.run`), so
                # this happens once per image at most.
                if candidate.shape == shape:
                    frame, dirty = candidate, window
                    break
        if frame is None:
            frame = np.zeros(shape, dtype=np.float32)
            self.allocated += 1
        else:
            if dirty is not None:
                frame[dirty] = 0.0
            self.reused += 1
        with self._lock:
            self._issued[id(frame)] = frame
        return frame

    def mark(self, frame: np.ndarray, window: tuple) -> None:
        """Record the only region of ``frame`` that was written."""
        with self._lock:
            if id(frame) in self._issued:
                self._dirty[id(frame)] = window

    def release(self, frame) -> bool:
        """Give a frame back. False if this pool never issued it."""
        with self._lock:
            if self._issued.pop(id(frame), None) is None:
                return False
            window = self._dirty.pop(id(frame), None)
            if len(self._free) < self.capacity:
                self._free.append((frame, window))
        return True

    def clear(self) -> None:
        """Drop everything held. For a benchmark measuring a cold container."""
        with self._lock:
            self._free.clear()
            self._issued.clear()
            self._dirty.clear()

    def summary(self) -> str:
        total = self.reused + self.allocated
        share = 100.0 * self.reused / total if total else 0.0
        return (f"{self.reused}/{total} frame(s) recycled ({share:.0f}%), "
                f"{len(self._free)} pooled, cap {self.capacity}")


class DosePredictor:
    """Turns beam geometry into dose maps on the input image's own grid.

    Parameters
    ----------
    model:
        Callable mapping ``(B, C, *grid.shape)`` network input to
        ``(B, *grid.shape)`` predicted dose in the beamlet frame. Defaults to
        :class:`StubModel`.
    batch_size:
        Beamlets per forward pass. Batch size is not a lever: on an A100,
        batch 16 and 32 move ``predict`` by 0.993x and 1.013x against 8, both
        inside their own noise, so 8 costs the same at a quarter of the peak
        VRAM.
    """

    def __init__(
        self,
        model=None,
        grid: BeamletGrid | None = None,
        batch_size: int = 8,
        with_wepl: bool = False,
        with_bragg: bool = False,
        with_lateral: bool = False,
        rsp: str = "linear",
        body_mask: bool = True,
        snap_alpha: float = 0.0,        # what shipped; `SNAP_ALPHA` is the other arm
        frame_pool: int = 0,
        cuboid_memo: bool = True,
        copy_ahead: bool = False,
    ) -> None:
        self.model = model if model is not None else StubModel(grid)
        self.grid = grid or BeamletGrid()
        self.batch_size = batch_size
        # Must match what the checkpoint was trained with; from_checkpoint sets
        # these from the checkpoint rather than leaving them to the caller.
        self.with_wepl = with_wepl
        self.with_bragg = with_bragg
        self.with_lateral = with_lateral
        self.rsp = rsp
        # 0.0 disables the snap and restores the plain clamp-to-zero.
        if not 0.0 <= snap_alpha < 1.0:
            raise ValueError(f"snap_alpha must be in [0, 1); got {snap_alpha!r}")
        self.snap_alpha = float(snap_alpha)
        # ON by default, as submitted. The labels are masked by the body
        # contour, and a filled per-slice contour matches that where a bare HU
        # threshold would delete enclosed air the labels keep. Exposed only so
        # a sweep can A/B it -- there is deliberately no dilation knob, because
        # dilation does not add safety, it trades the effect away.
        self.body_mask = body_mask
        # Zero-emission counters, cumulative across every predict() call so a
        # full sweep can be read off one predictor. Counted SEPARATELY because
        # the two causes are unrelated: a frame error is degenerate geometry, a
        # missing entry is a ray we could not place on the patient. Collapsing
        # them would let a future reader blame every zero on tangency.
        self.n_zero_frame_error = 0
        self.n_zero_no_entry = 0
        # Entry depths for one image, keyed by (ray_source, ray_target) -- see
        # `predict`. ``None`` is cached too: a ray that misses the patient misses
        # it for every beamlet on it, and re-walking 1400 mm to rediscover that
        # is the most expensive way to learn nothing.
        self._entry_cache: dict = {}
        self._entry_cache_ct: np.ndarray | None = None
        # The grid those depths were walked on: the same array under a
        # different geometry is a different image.
        self._entry_cache_geom: VolumeGeometry | None = None
        # The CT as a device tensor, plus the array it was made from. Its
        # own key, NOT `_entry_cache_ct`: `predict` stamps that one *before*
        # the upload runs, so sharing it would answer "same image" for a CT
        # this has never seen and hand back the previous patient's tensor.
        self._ct_device = None
        self._ct_device_src: np.ndarray | None = None
        self._body_device = None
        self._body_device_src: np.ndarray | None = None
        # Render coordinates for the last ray, and the key they belong to.
        self._render_cache: tuple | None = None
        # The sampled CT box for the last ray -- see `_cuboid`.
        self.cuboid_memo = bool(cuboid_memo)
        self._cuboid_cache: tuple | None = None
        # Whether a chunk's copy-out may finish while the NEXT chunk's forward
        # pass runs -- see `_issue_host_volume`. **Default OFF because it was
        # measured at 0.02 s in the runtime fit**, against a 0.07 s
        # repeat spread; the slack it aims at is real and the copy-out is simply
        # not on the critical path (`submission/inference.py`, `COPY_AHEAD`).
        self.copy_ahead = bool(copy_ahead)
        self._staging_pool: dict[int, "object"] = {}
        # Pinned landing pad for the device-to-host copy, grown to the largest
        # block seen -- see `_to_host_volume`. Never handed to a caller.
        self._staging = None
        # **OFF unless a capacity is asked for**, because reuse is only safe
        # where the caller hands frames back (:class:`FramePool`) and every
        # other caller of `predict` -- the scorers, the proxies, the tests --
        # holds its results instead. `submission/inference.py` asks; nothing
        # else does.
        self.frames = FramePool(frame_pool) if frame_pool else None

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path | None,
        device: str = "cuda",
        infer_dtype: str | None = None,
        cudnn_benchmark: bool = False,
        channels_last: bool = False,
        compile_mode: str | None = None,
        **kwargs,
    ) -> "DosePredictor":
        """Load a trained network, falling back to the stub when absent.

        The checkpoint is self-describing: architecture width and depth come
        from its stored ``config``, channel count from ``in_channels``, and the
        label scaling from ``dose_scale``. Nothing here needs to agree with a
        config file at inference time, which is deliberate -- the container ships
        without one.

        ``path=None`` yields the stub, so the container remains buildable before
        any checkpoint exists. A path that is *given but missing* raises instead
        — in a submission that almost always means the weights were not baked
        into the image, and falling back silently would ship a container that
        scores as noise with nothing in the logs to explain it.
        """
        if path is None:
            return cls(model=None, **kwargs)

        if not Path(path).exists():
            raise FileNotFoundError(
                f"checkpoint {path!r} does not exist. In the container the "
                "weights are mounted at /opt/ml/model/checkpoint.pt (README step "
                "5); locally, pass the path of a downloaded .pt. Unset "
                "CHECKPOINT_PATH to use the stub deliberately."
            )

        import torch

        from .network import build_network

        if device.startswith("cuda") and not torch.cuda.is_available():
            print("[predictor] CUDA unavailable, loading on CPU", flush=True)
            device = "cpu"

        # `weights_only=True`: a checkpoint is a pickle, and this one is
        # downloaded from a release page by the README's own first command.
        # Every field written here is plain data -- tensors, dicts, ints,
        # strings -- so the safe loader reads both released files unchanged. A
        # checkpoint that needs the unsafe loader is one to inspect, not to run.
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        config = checkpoint.get("config", {}) or {}
        in_channels = int(checkpoint.get("in_channels", 2))

        # Derived from the checkpoint, never from a caller's guess: feeding a
        # WEPL-trained network 2 channels (or vice versa) does not raise, it
        # silently produces wrong dose.
        #
        # `channels` is the list the trainer wrote. Before 2026-08-15 there was
        # no such field and only one optional channel, so the count decided it
        # unambiguously -- which it no longer does, because 3 channels is now
        # either WEPL or Bragg. Old checkpoints keep the old rule; new ones say
        # what they mean.
        channels = list(checkpoint.get("channels") or [])
        if not channels:
            if in_channels not in (2, 3):
                raise NotImplementedError(
                    f"checkpoint expects {in_channels} input channels and records "
                    "no `channels` list; the legacy rule only covers 2 (CT, "
                    "energy) or 3 (+ WEPL)."
                )
            channels = ["ct", "energy"] + (["wepl"] if in_channels == 3 else [])

        if len(channels) != in_channels:
            raise ValueError(
                f"checkpoint is inconsistent: in_channels={in_channels} but "
                f"channels={channels}"
            )
        known = {"ct", "energy", "wepl", "bragg", "lateral"}
        if set(channels) - known or channels[:2] != ["ct", "energy"]:
            raise NotImplementedError(
                f"checkpoint wants channels {channels}; the inference path "
                f"builds [ct, energy] plus any of {sorted(known - {'ct', 'energy'})} "
                "in that order."
            )

        with_wepl, with_bragg = "wepl" in channels, "bragg" in channels
        with_lateral = "lateral" in channels
        if with_lateral and channels[-1] != "lateral":
            raise ValueError(
                f"channels={channels}: 'lateral' must be LAST -- "
                "FactorisedDoseNet(condition_lateral=True) reads x[:, -1] as the "
                "analytic kernel, so any other position silently conditions on "
                "the wrong channel"
            )
        for name, value in (("with_wepl", with_wepl), ("with_bragg", with_bragg),
                            ("with_lateral", with_lateral)):
            if bool(config.get(name, value)) != value:
                raise ValueError(
                    f"checkpoint is inconsistent: channels={channels} but "
                    f"config says {name}={config.get(name)}"
                )
        # Depth conditioning REUSES the Bragg channel rather than adding one, so
        # `channels` looks identical with and without it and cannot be the
        # authority here -- the flag is recorded separately and read separately.
        # Absent means a checkpoint from before the arm existed, i.e. False.
        condition_depth = bool(checkpoint.get("condition_depth",
                                              config.get("condition_depth", False)))
        if condition_depth and not with_bragg:
            raise ValueError(
                f"checkpoint says condition_depth=True but channels={channels} "
                "carry no 'bragg' -- the depth curve is conditioned on that "
                "channel, so there is nothing to condition on"
            )
        kwargs.setdefault("with_wepl", with_wepl)
        kwargs.setdefault("with_bragg", with_bragg)
        kwargs.setdefault("with_lateral", with_lateral)
        # Absent means a checkpoint from before the curve was selectable, and
        # those were all trained on the linear form.
        kwargs.setdefault("rsp", str(checkpoint.get("rsp")
                                     or config.get("rsp") or "linear"))

        # The box is a registration anchor, so a checkpoint inferred under a
        # different one predicts dose in the wrong *place* -- no crash, no
        # implementation error, just a quietly wrong map. Every checkpoint
        # records the box it was trained with, so disagreeing is an error here
        # rather than a thing to notice later.
        #
        # **Compared as boxes, not as dicts.** The old comparison was
        # `dict(stored) != {shape, entry_margin_mm, anchor}` -- it never looked
        # at a spacing, so `(768, 128, 40)` at 0.5 mm and the same shape at
        # 1.0 mm passed as the same box -- exactly the fine-spacing box family.
        # `BeamletGrid.from_dict` also lets a caller who passes no `grid` inherit
        # the checkpoint's, which is what makes a resharded arm scoreable
        # and shippable through
        # `submission/inference.py` without being told the box.
        stored_grid = checkpoint.get("grid")
        if stored_grid:
            trained_on = BeamletGrid.from_dict(stored_grid)
            requested = kwargs.get("grid")
            if requested is None:
                kwargs["grid"] = trained_on
            elif requested != trained_on:
                raise ValueError(
                    "checkpoint was trained with a different sampling box than "
                    f"this predictor uses: checkpoint {trained_on.as_dict()} vs "
                    f"predictor {requested.as_dict()}. Entry depth anchors the "
                    "box, so the dose would land in the wrong place without "
                    "failing."
                )

        # A checkpoint trained on thresholded labels has learned to emit ~0
        # below that cutoff. That is harmless while the platform's cutoff is at
        # least as large -- the clamp erases the same band on both sides -- and
        # silently destructive when it is smaller, because the blanked band is
        # still scored. Recorded so the pairing is checkable; `predict` cannot
        # check it, since the platform's value arrives per beamlet at call time.
        label_cutoff = float(checkpoint.get("label_cutoff", 0.0) or 0.0)

        # No `arch` means a checkpoint written before there was a second
        # architecture, and those are all U-Nets.
        net = build_network(
            str(checkpoint.get("arch") or config.get("arch") or "unet"),
            in_channels=in_channels,
            base_features=int(config.get("base_features", 24)),
            levels=int(config.get("levels", 4)),
            **({"condition_lateral": True} if with_lateral else {}),
            **({"condition_depth": True, "bragg_index": channels.index("bragg")}
               if condition_depth else {}),
        )
        # `strict=True` is the whole guard for the conditioned arms: a
        # conditioned checkpoint carries `prior_scale`/`prior_bias`, which an
        # unconditioned build has nowhere to put. Loosen this and the arm loads
        # cleanly, drops the two parameters, and predicts a flat-started curve.
        net.load_state_dict(checkpoint["model"])
        net.eval().to(device)
        if label_cutoff > 0:
            print(f"[predictor] checkpoint trained on labels thresholded at "
                  f"{label_cutoff:g} Gy -- valid only where the platform's "
                  f"minimum_cutoff is >= that", flush=True)

        dose_scale = float(checkpoint.get("dose_scale", 1.0e-3))
        return cls(
            model=TorchModel(net, device, dose_scale, autocast=infer_dtype,
                             cudnn_benchmark=cudnn_benchmark,
                             channels_last=channels_last,
                             compile_mode=compile_mode,
                             aoti_package=os.environ.get("AOTI_PACKAGE") or None,
                             # Read from the checkpoint, never from a flag: a
                             # package is only valid for the weights it was
                             # exported from, and a caller cannot know which.
                             # `step`/`epoch`, not `steps`/`config.epochs`.
                             # The config records what was ASKED FOR -- 240
                             # epochs -- while the checkpoint records what was
                             # kept, epoch 229 at step 465750. Keying on the
                             # request makes every run of one config share an
                             # identity, which is precisely the discrimination
                             # this guard exists to provide.
                             aoti_identity={
                                 "step": checkpoint.get("step", ""),
                                 "epoch": checkpoint.get("epoch", ""),
                                 "geometry_sha": checkpoint.get("geometry_sha256", "")
                                 or (config or {}).get("geometry_sha256", ""),
                             }), **kwargs
        )

    def resolve_entry(
        self,
        ct: np.ndarray,
        geom: VolumeGeometry,
        source: np.ndarray,
        target: np.ndarray,
    ) -> float | None:
        """Entry depth for one beamlet: where the sampling *box* reaches tissue.

        One rule, applied to every beamlet. The central axis is not consulted at
        all -- it used to decide first, with the box as a fallback for rays that
        graze the patient, and that gate is what this replaces. A gate buys a
        bit-identity invariant for the 99.5% of beamlets whose axis resolves, but
        it also keeps a discontinuity in the anchor, an out-of-distribution class
        of "tangential" beamlets, and a counter and an assert to police them.
        Anchoring on the box collapses all of that into one continuous function
        of geometry.

        The anchor is a box corner rather than the beam core, which is less
        physically meaningful but *consistent*, and consistency is what a
        registration anchor needs. It lives here rather than inline in
        :meth:`predict` so it can be measured directly instead of being
        reimplemented by whatever is checking it.

        **On CUDA the walk runs on the device**, because it is the largest
        host-side term left in ``predict`` and the host is what ``predict``
        waits on: ~14 ms per call on every card measured, with the GPU idle
        throughout. The CT is already there for the sampling
        that follows, so this costs no transfer.

        **There are now two implementations of the box anchor**, which is the
        shape of bug `models/geometry_torch.py` exists to warn about — so the
        numpy one stays the definition (preprocessing builds every shard with
        it) and the device one is held to it by equality, not by argument:
        `tests/test_entry_walk.py` on synthetic geometry and
        `tests/test_geometry_torch.py` on real beamlets and real CUDA. A
        difference of one millimetre is a different box for that beamlet, so
        those tests compare with ``==`` and must keep doing so.
        """
        device = getattr(self.model, "device", None)
        if device is not None and str(device).startswith("cuda"):
            return GT.find_entry_depth_box(
                self._ct_on_device(ct, device), geom, source, target, self.grid
            )
        return find_entry_depth_box(ct, geom, source, target, self.grid)

    def _to_host_volume(self, block, lo, hi, shape) -> np.ndarray:
        """A full-grid dose map, moving only the box across the bus.

        The copy back to the host was the largest single cost in this path, and
        it moved a whole CT grid -- ~85 MB -- of which the beamlet's box is a
        median **5.5%** and everything else is **exactly zero by construction**
        (`geometry_torch.render_block` performs the only write). So the transfer
        carries the block and the zeros are made here, where they are free:
        ``np.zeros`` is a ``calloc``, and untouched pages cost nothing to
        allocate and almost nothing to read back.

        Three copies were measured on an A100 for a full 85 MB frame -- pageable
        39.7 ms, pinned 9.6 -- and this replaces the payload rather than the
        mechanism, so it composes with the pinning rather than competing.

        **The staging buffer is reused, and that is safe here for the reason it
        was not before.** When the pinned buffer *was* the returned array a
        shared one would have aliased every result in a chunk; here it is only a
        landing pad and the value is copied straight into ``out``, which the
        caller owns. Sized to the largest block seen, so the pinned allocation
        happens a handful of times rather than per beamlet -- block extents vary
        per beamlet, and a fresh pinned tensor each time would ask the kernel to
        page-lock a new size 500 times a job.

        **CPU-only skips the staging entirely**: pinning needs a CUDA context
        and the stub and every laptop test run without one.
        """
        import torch

        # The allocation itself is 1.47 of the 2.61 ms this line used to cost --
        # a calloc plus ~2700 page faults for a block that covers 5.5% of the
        # frame. A pooled frame arrives already faulted and already zero
        # everywhere but its last window (:class:`FramePool`).
        out = (self.frames.take(shape) if self.frames is not None
               else np.zeros(shape, dtype=np.float32))
        if block is None or block.numel() == 0:
            return out
        window = block_window(lo, hi)
        # The mark is taken from the window the write below uses, never
        # recomputed: it is what the *next* beamlet to get this frame will zero,
        # so anything written outside it would survive into that beamlet's dose.
        if self.frames is not None:
            self.frames.mark(out, window)
        block = block.to(torch.float32)
        if not block.is_cuda:
            out[window] = block.numpy()
            return out

        flat = block.reshape(-1)
        if self._staging is None or self._staging.numel() < flat.numel():
            self._staging = torch.empty(
                flat.numel(), dtype=torch.float32, pin_memory=True
            )
        landing = self._staging[: flat.numel()]
        landing.copy_(flat)
        out[window] = landing.numpy().reshape(tuple(block.shape))
        return out

    def _issue_host_volume(self, block, lo, hi, shape, slot: int):
        """:meth:`_to_host_volume`, split so the copy can finish LATER.

        **The point is the schedule, not the copy.** ``predict`` is serialised
        — the host launches a chunk's forward pass and then stands still for
        ~185 ms of A10G before it may touch the result — while the copy-out it
        owes for the *previous* chunk is pure host and device-to-host work that
        depends on nothing the current forward produces. Measured inside the
        shipping image: **~1.5 ms/beamlet** of it, against a forward of 23.2.

        So this half issues the device-to-host copy (**asynchronous**, into a
        pinned buffer this chunk owns) and returns what :meth:`_finish_host_volume`
        needs to complete it; the caller finishes the previous chunk after
        launching the next forward, where the wait is free.

        **THE BUFFER MUST OUTLIVE THE COPY, AND SO MUST THE BLOCK.** An async
        D2H reads the device tensor after this returns, so ``block`` is kept in
        the returned tuple — dropping it would let the caching allocator hand
        its memory to the next kernel *while the copy is still reading it*, which
        is the silent-wrong class again. One generation of buffers per chunk,
        never reused until its event has been waited on.

        CPU tensors go through the same two halves rather than a special case:
        the laptop has no CUDA, and a pipeline that is only exercised on the box
        is a pipeline whose ordering is untested.
        """
        import torch

        out = (self.frames.take(shape) if self.frames is not None
               else np.zeros(shape, dtype=np.float32))
        if block is None or block.numel() == 0:
            return out, None
        window = block_window(lo, hi)
        if self.frames is not None:
            self.frames.mark(out, window)
        block = block.to(torch.float32)
        if not block.is_cuda:
            return out, (None, window, tuple(block.shape), block)

        flat = block.reshape(-1)
        landing = self._staging_slot(slot, flat.numel())
        landing.copy_(flat, non_blocking=True)
        return out, (landing, window, tuple(block.shape), block)

    def _staging_slot(self, slot: int, numel: int):
        """One pinned landing pad per in-flight beamlet, grown to the largest.

        The synchronous path can share a single buffer because the copy is over
        before the next beamlet starts. This one cannot: a chunk's copies are all
        in flight at once, so each needs its own, and the *previous* chunk's are
        still being read by the host. Hence ``2 x batch`` slots, ~2.5 MB each on
        the shipping box -- 40 MB of pinned memory against a 1.9 GB peak.
        """
        import torch

        held = self._staging_pool.get(slot)
        if held is None or held.numel() < numel:
            held = torch.empty(numel, dtype=torch.float32, pin_memory=True)
            self._staging_pool[slot] = held
        return held[:numel]

    @staticmethod
    def _finish_host_volume(pending) -> None:
        """The host half: land each block in its frame. After the event."""
        for out, item in pending:
            if item is None:
                continue
            landing, window, block_shape, block = item
            source = block.numpy() if landing is None else landing.numpy()
            out[window] = source.reshape(block_shape)

    def _ct_on_device(self, ct: np.ndarray, device):
        """The CT on ``device``, uploaded once per image rather than per call.

        Uploading inside `_predict_on_device` instead would run once per
        ``predict`` call -- and the container calls ``predict`` per *batch*, so
        an 85 MB host-to-device copy would pay for every 8 beamlets. Caching is
        bit-identical (same bytes, same tensor) and it is also what lets the
        entry walk run on the device at all: `resolve_entry` needs the CT there
        **before** the beamlet loop, not inside it.

        Keyed on the array's own identity, holding the reference so no recycled
        ``id()`` can match a dead array -- the same rule the entry cache uses,
        and for the same reason.
        """
        import torch

        if self._ct_device is None or self._ct_device_src is not ct:
            self._ct_device = torch.as_tensor(
                np.ascontiguousarray(ct), device=device
            )
            self._ct_device_src = ct
        return self._ct_device

    def _body_on_device(self, ct: np.ndarray, device):
        """The filled body contour, cached per image exactly as the CT is.

        **Per IMAGE, not per beamlet.** :func:`body_contour` labels the whole
        volume (211-473 ms) and a job serves several beamlets per image, so
        computing it inside the loop would put it on every one of them. Keyed on
        the array's own identity and holding the reference, the same rule as
        ``_ct_on_device`` and for the same reason.
        """
        import torch

        if self._body_device is None or self._body_device_src is not ct:
            self._body_device = torch.as_tensor(
                np.ascontiguousarray(body_contour(ct, AIR_HU)), device=device
            )
            self._body_device_src = ct
        return self._body_device

    def _render_coords(self, geom: VolumeGeometry, source, target, entry: float, device):
        """``(lo, hi, coords)`` for this ray, reused across the ray's beamlets.

        **The same argument as the entry memo, one stage later.** The map from
        CT voxel to prediction cuboid is a function of the *ray* and the entry
        depth — energy is not one of its arguments — so the beamlets sharing a
        ray share this exactly. Measured 2026-08-24: building it is **9.1 of
        `render_block`'s 10.2 ms/beamlet of device time**, the largest term
        after the forward pass, and the sampling it feeds is under 1 ms/beamlet.

        **One entry, not a dict.** The coordinates are one float64 triple per
        CT voxel of the box's bounding box — tens of megabytes — so a per-image
        cache the size of the entry memo would be gigabytes. The pairs that
        share a ray are adjacent (`predict`), so depth one collects the same
        hits at bounded cost. Do not "improve" this into a dict.

        Bit-exact: the *same tensor*, not an equivalent one.
        """
        key = (tuple(np.asarray(source).tolist()), tuple(np.asarray(target).tolist()),
               float(entry))
        if self._render_cache is not None:
            cached_key, cached_geom, built = self._render_cache
            # `geom` is compared by IDENTITY and the reference is HELD, never
            # keyed on `id()`. CPython recycles ids, so a dead geometry's id can
            # come back on a live one with a different origin or spacing, and
            # these coordinates would then map a beamlet onto the wrong voxels
            # -- the same trap `tests/test_predictor_entry_cache.py` names for
            # the entry memo. A held reference cannot be recycled.
            if cached_key == key and cached_geom is geom:
                return built
        built = GT.render_coordinates(source, target, entry, self.grid, geom, device)
        self._render_cache = (key, geom, built)
        return built

    def _cuboid(self, ct_device, geom, source, target, entry, device):
        """The CT sampled on this beamlet's box, reused across the ray's beamlets.

        **The third memo with the same argument, and the largest.** ``sample_ct``
        and the ``grid_points_world`` that feeds it are functions of the *ray*
        and the entry depth — **energy is not an argument of either**, so two
        beamlets on one ray get the same box by construction rather than by
        luck, exactly as the entry walk and the render coordinates do. Together
        they are **~2.3 ms/beamlet** on the released 384×64×24 box, against ~0.8 for the
        render coordinates this copies.

        **What makes it worth more than the other two is a clinical plan's
        shape**: its rays carry several energy layers each,
        and `predict` receives beamlets in `idx_in_output` order, which nests
        them under their ray — so a depth-one memo sees one ray many times in a
        row, and a *batch* of 8 asks for it 8 times inside a single chunk.

        **One entry, not a dict**, for the reason :meth:`_render_coords`
        gives: the box is 384 × 64 × 24 floats and a per-image dict of them
        would be gigabytes. Depth one collects the same hits at 2.4 MB.

        **Keyed on the CT TENSOR's identity as well as the geometry**, both
        held by reference. Coordinates do not depend on the CT and the render
        memo therefore does not check it; this one is *sampled from* it, so a
        second patient with the same ray geometry would otherwise be served the
        first one's anatomy — the trap `tests/test_predictor_entry_cache.py`
        names, one stage further along.

        Sharing one tensor across a chunk is safe because every consumer only
        reads it: ``build_network_input`` clamps out of place, the WEPL and
        Bragg channels build new tensors, and `models/physics.py` contains no
        in-place operator at all.
        """
        if not self.cuboid_memo:
            return GT.sample_ct(
                ct_device, geom,
                GT.grid_points_world(source, target, entry, self.grid, device),
            )
        key = (tuple(np.asarray(source).tolist()),
               tuple(np.asarray(target).tolist()), float(entry))
        if self._cuboid_cache is not None:
            cached_key, cached_geom, cached_ct, built = self._cuboid_cache
            if cached_key == key and cached_geom is geom and cached_ct is ct_device:
                return built
        built = GT.sample_ct(
            ct_device, geom,
            GT.grid_points_world(source, target, entry, self.grid, device),
        )
        self._cuboid_cache = (key, geom, ct_device, built)
        return built

    def disable_compilation(self) -> bool:
        """Ask the model to drop its compiled wrapper; False if it had none.

        Delegated, because `net` belongs to :class:`TorchModel` and the stub has
        no network at all -- and the caller is `submission/inference.py`, which
        holds a predictor rather than a model.
        """
        model = getattr(self, "model", None)
        drop = getattr(model, "disable_compilation", None)
        return bool(drop()) if drop is not None else False

    def release_frame(self, frame) -> bool:
        """Hand a dose map back for reuse; False if it was not pooled.

        **Only after nothing will read it again.** The caller owns the frame
        until this call and must not own it after -- `submission/inference.py`
        calls it on the writer thread, once ``StreamingStackWriter.add`` has
        returned and the frame is bytes on disk. Releasing early is not an
        error, it is the next beamlet's dose landing in this one's map.
        """
        return self.frames.release(frame) if self.frames is not None else False

    def reset_entry_cache(self) -> None:
        """Forget the memoised entry depths.

        ``predict`` drops them on its own when a new image arrives, so nothing in
        the production path needs this. **A benchmark does.** Timing the same
        image twice with one predictor measures a hit rate the container can
        never have: on 1ABB020 the first pass cost 21.7 ms/beamlet and every pass
        after it 16.4, and a harness that discards its first round as warmup
        would have reported the 16.4.
        """
        self._entry_cache = {}
        self._entry_cache_ct = None
        self._entry_cache_geom = None
        # The upload too, or a benchmark's second round measures a resident
        # CT the container never has: it sees each image once.
        self._ct_device = None
        self._ct_device_src = None
        self._body_device = None
        self._body_device_src = None
        self._render_cache = None
        self._cuboid_cache = None
        # And the Bragg depth profile, which is memoised on the cuboid this
        # predictor hands it (`models/physics.py`, `_wepl_and_body`).
        from . import physics
        physics.reset_bragg_memo()
        # And the frames, for the same reason: a pool warmed by round one is
        # a hit rate the container -- one process per job -- never sees.
        if self.frames is not None:
            self.frames.clear()

    def _predict_on_device(self, ct, geom, prepared, results) -> None:
        """The whole beamlet on the GPU: sample -> forward -> render.

        Everything geometric used to run in scipy on the CPU -- ~105 ms per
        beamlet against a forward pass that batching showed to be nearly free.
        Moving it here is worth more than the arithmetic alone, because the CT
        is uploaded **once per image** and the box never returns to the host
        between sampling and rendering; only the finished dose map comes back.

        Numerically this is the same operation as the scipy path, and that is
        asserted rather than assumed -- `tests/test_geometry_torch.py` compares
        every function against its counterpart on CPU and CUDA. It is also the
        *same* code the shards were built with, which is what keeps training and
        inference symmetric (`models/geometry.py`).
        """
        import torch

        device = getattr(self.model, "device", "cpu")
        forward_device = getattr(self.model, "forward_device", None)
        ct_device = self._ct_on_device(ct, device)
        body_device = self._body_on_device(ct, device) if self.body_mask else None
        # **ONE CHUNK OF SLACK, AND THE SLACK IS THE FORWARD PASS.** The host
        # launches a chunk's forward and then has ~185 ms of A10G to wait for
        # (batch 8, the released box), while the copy-out it owes for the PREVIOUS chunk
        # depends on nothing that forward produces. So the copies are issued
        # asynchronously and finished one iteration later, right after the next
        # forward is launched — where the host was standing still anyway.
        # Two generations of pinned buffers, alternating, because the previous
        # chunk's are still being read while this chunk's are being filled.
        pipelined = self.copy_ahead and len(prepared) > self.batch_size
        pending: list = []
        pending_event = None
        generation = 0
        with torch.inference_mode():
            for start in range(0, len(prepared), self.batch_size):
                chunk = prepared[start : start + self.batch_size]
                cuboids = [
                    self._cuboid(ct_device, geom, source, target, entry, device)
                    for _, source, target, entry, _ in chunk
                ]
                inputs = [
                    GT.build_network_input(cuboid, request.energy)
                    for cuboid, (_, _, _, _, request) in zip(cuboids, chunk)
                ]
                if self.with_wepl:
                    inputs = [
                        torch.cat(
                            [
                                inp,
                                GT.build_wepl_channel(
                                    cuboid, self.grid.depth_spacing
                                )[None],
                            ]
                        )
                        for inp, cuboid in zip(inputs, cuboids)
                    ]
                if self.with_bragg:
                    from . import physics

                    inputs = [
                        torch.cat(
                            [
                                inp,
                                physics.build_bragg_channel(
                                    cuboid,
                                    request.energy,
                                    self.grid.depth_spacing,
                                    # Degrade, never raise: an energy the machine
                                    # does not tabulate costs a slightly worse
                                    # input channel, while an exception costs the
                                    # submission (`models/physics.py`).
                                    strict_energy=False,
                                    rsp=self.rsp,
                                )[None],
                            ]
                        )
                        for inp, cuboid, (_, _, _, _, request) in zip(
                            inputs, cuboids, chunk
                        )
                    ]
                if self.with_lateral:
                    from . import physics

                    inputs = [
                        torch.cat([
                            inp,
                            physics.build_lateral_prior(
                                cuboid, request.energy, self.grid.depth_spacing,
                                rsp=self.rsp, strict_energy=False)[None],
                        ])
                        for inp, cuboid, (_, _, _, _, request) in zip(
                            inputs, cuboids, chunk)
                    ]
                stacked = torch.stack(inputs)
                if forward_device is not None:
                    predicted = forward_device(stacked)
                    # HERE, AND NOT ONE LINE EARLIER. The launch above is
                    # asynchronous, so the GPU is now busy for the whole forward
                    # and this host thread is free — which is the only moment in
                    # `predict` where finishing the previous chunk's copies is
                    # free. Before the launch it would delay the forward; after
                    # the renders below it would be the serial schedule again.
                    if pending:
                        if pending_event is not None:
                            pending_event.synchronize()
                        self._finish_host_volume(pending)
                        pending = []
                else:
                    # A model that only honours the numpy contract still works;
                    # it just pays a round trip the device path avoids.
                    predicted = torch.as_tensor(
                        np.asarray(self.model(stacked.cpu().numpy()), dtype=np.float32),
                        device=device,
                    )
                    if predicted.ndim == 5:
                        predicted = predicted[:, 0]

                for offset, (position, source, target, entry, request) in enumerate(
                    chunk
                ):
                    # The block, not the volume: everything outside the box is
                    # exactly zero and there is no reason to move 85 MB of it
                    # across the bus (`_to_host_volume`). The clamps below follow
                    # it for the same reason -- they ran over 21M voxels to leave
                    # 95% of them at the zero they already were.
                    lo, hi, coords = self._render_coords(
                        geom, source, target, entry, device
                    )
                    block, lo, hi = GT.render_block(
                        predicted[offset], source, target, entry, self.grid, geom,
                        coords=coords, bounds=(lo, hi),
                    )
                    if block is not None:
                        if self.body_mask:
                            # Our head is strictly positive everywhere, so
                            # without this we emit a small positive floor over
                            # most of the volume. `idd_distance` is a lateral
                            # integral and accumulates it; `beam_mae` and gamma
                            # mask to >=10% of GT peak and never see it.
                            #
                            # THE CONTOUR, NEVER `ct > AIR_HU`. The bare
                            # threshold deletes internal air stored at the
                            # reconstruction floor -- bowel gas, trachea -- where
                            # their contour-masked labels HAVE dose, which the
                            # `idd` metric charges for.
                            block = block * body_device[block_window(lo, hi)]
                        # The platform counts any value below a beam's declared
                        # cutoff as an implementation error, so clamp before
                        # handing it back. Zeros outside the block are already
                        # below any cutoff and stay zero either way.
                        # The threshold is NOT the cutoff -- see
                        # `clamp_threshold`, which is where the float32 and
                        # float64 comparisons are reconciled.
                        if request.minimum_cutoff > 0:
                            # THREE-WAY when snap_alpha > 0. Only `0` or `>= c`
                            # is legal, so against an UNthresholded truth the bottom
                            # of the sub-cutoff band is cheaper at zero and the TOP
                            # is cheaper at the threshold (off in the submission). Snapping to `t` and not
                            # to `c`: `t` is what the zero-arm already tests, so a
                            # snapped voxel provably passes the same check, with
                            # the same margin argument behind it.
                            block = apply_cutoff_policy(
                                block, request.minimum_cutoff, self.snap_alpha
                            )
                        block = block.clamp(min=0.0)
                    if pipelined:
                        frame, item = self._issue_host_volume(
                            block, lo, hi, geom.shape,
                            generation * self.batch_size + offset,
                        )
                        results[position] = frame
                        pending.append((frame, item))
                    else:
                        results[position] = self._to_host_volume(
                            block, lo, hi, geom.shape
                        )
                if pipelined:
                    # The event, not a `synchronize()`: what the next
                    # iteration must wait for is *these* copies, not whatever
                    # the GPU is doing by then — which will be the next forward.
                    if str(device).startswith("cuda"):
                        pending_event = torch.cuda.Event()
                        pending_event.record()
                    generation ^= 1

            # The last chunk has no next forward to hide behind, so it pays
            # what every chunk used to. Reached on every path out of the loop:
            # `predict` may not return a frame whose voxels are still in a
            # pinned buffer.
            if pending:
                if pending_event is not None:
                    pending_event.synchronize()
                self._finish_host_volume(pending)

    def predict(
        self,
        ct: np.ndarray,
        geom: VolumeGeometry,
        requests: list[BeamletRequest],
    ) -> list[np.ndarray]:
        """Dose volumes on ``geom``'s grid, one per request, in request order.

        Requests whose ray misses the patient yield an all-zero volume rather
        than being dropped: the output contract requires every declared dose map
        to exist, and a gap would shift `idx_in_output` for everything after it.

        **Entry depths are memoised per image.** ``resolve_entry`` is a function
        of the ray and the box, and **energy is not one of its arguments** -- so
        two beamlets sharing a ray have the same answer by construction, not by
        luck. In the training plans that is *exactly half* of them (1,080
        beamlets, 540 distinct rays, and the pairs are adjacent), and the walk
        costs 11.7 ms/beamlet, the largest single term left in this path. The
        test plans need not share that shape -- per-level counts are "not fix for
        testing" -- so this buys 50% *there* only if the same
        pairing holds. A dict lookup either way; nothing is banked on it.

        The cache belongs to one image and is dropped when a different ``ct``
        arrives. Keyed on the array's **identity**, holding the reference so no
        recycled ``id()`` can ever match a dead array -- the caller owns that
        array for the duration anyway, so nothing is copied and nothing is kept
        alive that was not already.
        """
        results: list[np.ndarray] = [None] * len(requests)  # type: ignore[list-item]

        if ct is not self._entry_cache_ct or geom is not self._entry_cache_geom:
            self._entry_cache = {}
            self._entry_cache_ct = ct
            self._entry_cache_geom = geom

        prepared, skipped = [], []
        for position, request in enumerate(requests):
            source = np.asarray(request.ray_source, dtype=float)
            target = np.asarray(request.ray_target, dtype=float)
            try:
                beam_frame(source, target)      # raises on a degenerate ray
            except ValueError:
                self.n_zero_frame_error += 1
                skipped.append(position)
                continue
            key = (tuple(source.tolist()), tuple(target.tolist()))
            if key in self._entry_cache:
                entry = self._entry_cache[key]
            else:
                entry = self.resolve_entry(ct, geom, source, target)
                self._entry_cache[key] = entry
            if entry is None:
                self.n_zero_no_entry += 1
                skipped.append(position)
                continue
            prepared.append((position, source, target, entry, request))

        for position in skipped:
            results[position] = (self.frames.take(geom.shape)
                                 if self.frames is not None
                                 else np.zeros(geom.shape, dtype=np.float32))

        if skipped:
            # Development measured zero genuine misses in real data on either side
            # of the split, so a non-zero reading here is evidence of a defect
            # in the fix, not a rare event that got logged. Silence is what let
            # the tangential defect survive months of full sweeps.
            print(
                f"[predictor] emitted {len(skipped)} all-zero map(s): "
                f"{self.n_zero_frame_error} frame error, "
                f"{self.n_zero_no_entry} no entry (cumulative)",
                flush=True,
            )

        # Unconditional: there is one resampling implementation on the
        # production path, and it is the torch one. A model without a device
        # (the stub, or any plain callable) still goes through it -- torch runs
        # on CPU perfectly well, so a scipy fallback would buy nothing and cost
        # the guarantee that preprocessing and inference resample identically.
        self._predict_on_device(ct, geom, prepared, results)
        return results
