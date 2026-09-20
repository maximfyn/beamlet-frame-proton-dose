#!/usr/bin/env python3
"""Train the beamlet dose network.

Single-GPU by default; multi-GPU by launching under ``torch.distributed.run``,
which the script detects from the environment rather than needing a flag.

Metric naming
-------------
Everything this script measures lives in the **beamlet frame** -- the
cuboid the shards were built with, 384x64x24 for the released models -- and is a different quantity from the ranked ``beam_mae``,
which is computed after rendering back onto the CT grid
(``evaluation/metrics.py``, emitted as ``beam_mae_mean``). The two are close
enough to be mistaken for each other: one development run carried 0.010075 here against a
CT-frame 0.01020. Two unrelated numbers that agree to three decimals are worse
than two that obviously differ, because nobody checks a number that looks
right.

So every emission carries the ``_boxframe`` qualifier and ``assert_boxframe``
refuses anything that does not -- the trainer keeps this guard local rather than importing the evaluation
stack. **A bare
leaderboard name means the ranked quantity, always.**

Loss
----
Plain MSE over the cuboid would be dominated by the empty volume -- the target
is zero across most voxels, so a model that predicts nothing scores well. The
default here weights each voxel by its own dose, which concentrates the
gradient on the region the Level 1 metric actually scores (voxels above 10% of
the beam's peak) without discarding the zero region entirely.

Usage
-----
    # smoke test, one GPU, a few steps
    python scripts/train/train_doserad.py --config configs/base_e240.yaml \
        --data-root <SHARDS> --max-steps 50

    # the two runs behind the submitted model (2 GPUs, effective batch 40)
    python -m torch.distributed.run --nproc_per_node=2 \
        scripts/train/train_doserad.py --config configs/base_e240.yaml \
        --data-root <SHARDS>
    python -m torch.distributed.run --nproc_per_node=2 \
        scripts/train/train_doserad.py --config configs/finetune_rbias_w32.yaml \
        --data-root <SHARDS>

    Some multi-GPU nodes need NCCL_NVLS_ENABLE=0: without it a launch can die
    within ~20 s on "Failed to bind NVLink SHARP (NVLS) Multicast memory
    ... CUDA error 401", which reads like a fault in this file and is not.

    The EFFECTIVE batch is ``batch_size`` x world size x ``--grad-accum``, so
    the GPU count is a training hyperparameter and not just a placement choice.
    Changing --nproc_per_node without compensating changes the run, and two arms
    compared across different world sizes are not paired. ``--grad-accum`` is
    what lets the 2-GPU cap keep an arm's effective batch: 20 x 2 x 2 is the
    same 80 as 20 x 4 x 1. It is an optimisation identity, pinned by
    tests/test_grad_accum.py, not assumed.

    There is NO resume. ``--init-from`` warm-starts the weights only (the
    fine-tune config uses it); it restores no optimiser or OneCycleLR state and
    no start epoch, so an interrupted run restarts from step 0.
    ``--save-state-every`` writes that state to ``<run>_state.pt``, but nothing
    here reads it back. Budget a run to finish inside the window it has.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import contextlib
import signal
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.dataset import BeamletDataset, DOSE_SCALE  # noqa: E402
from models.geometry import BeamletGrid  # noqa: E402
from models.network import ARCHITECTURES, build_network  # noqa: E402

try:
    import yaml
except ImportError:
    yaml = None

DEFAULT_DATA_ROOT = Path("data/dataset_beamlet_tall24_it32/proton")


# ---------------------------------------------------------------------------
# metric naming -- see the module docstring
# ---------------------------------------------------------------------------

# A key whose stem is exactly one of these names the *ranked* quantity. Nothing
# this script measures is one, so nothing it emits may be called one.
LEADERBOARD_NAMES = frozenset(
    {"beam_mae", "idd_distance", "plan_mae", "gamma", "dvh_score"}
)

BOXFRAME_SUFFIX = "_boxframe"

# The hidden test set is 14 thoracic / 26 abdominal; our splits
# are all ~50/50. Selecting a checkpoint on an unweighted val mean therefore
# optimises a mixture the leaderboard does not score. Reweighting costs ~0.5
# effective patients of six (Kish) and removes a systematic bias.
THORACIC_SHARE = 14 / 40

# `val/`, `train/`, `val_`, `best_val_` route a key to a section of W&B or a log
# line; they do not change what it measures. `best_val_beam_mae` claims the
# ranked name exactly as `beam_mae` does, which is why the stem is what is
# checked. Longest first, so `best_val_` is not eaten by `best_`.
_ROUTING_PREFIXES = ("best_val_", "best_", "val_", "train_")

# The checkpoint field, newest name first. Checkpoints written before
# 2026-08-15 carry the second and must still load and still build a container,
# so readers accept both -- but writers emit only the first. A tuple, not a
# dict literal, which is why the legacy name here is not itself an emission.
CHECKPOINT_VAL_KEYS = ("val_beam_mae_boxframe", "val_beam_mae")


def metric_stem(key: str) -> str:
    """The part of an emission key that says *what was measured*."""
    stem = key.rsplit("/", 1)[-1]
    for prefix in _ROUTING_PREFIXES:
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem


def assert_boxframe(keys) -> None:
    """Refuse to emit a beamlet-frame number under a bare leaderboard name.

    The suffix is what survives being copied into a table, a run JSON or a
    W&B panel -- which is the only place the confusion has ever done damage.
    """
    for key in keys:
        if metric_stem(key) in LEADERBOARD_NAMES:
            raise ValueError(
                f"{key!r} is a leaderboard metric name, and this script measures "
                f"the beamlet frame. Emit {key + BOXFRAME_SUFFIX!r} instead; the "
                "bare name belongs to evaluation/metrics.py's CT-frame number."
            )


def emit(payload: dict) -> dict:
    """Guard a structured emission and return it, so it wraps the call in place."""
    assert_boxframe(payload)
    return payload


def val_beam_mae_boxframe(checkpoint) -> float | None:
    """The selector value a checkpoint was saved on, under either name.

    Accepts both keys because existing checkpoints keep the old one; ``None``
    when the checkpoint records neither. ``CHECKPOINT_VAL_KEYS`` is the order
    they are tried in.
    """
    for key in CHECKPOINT_VAL_KEYS:
        if key in checkpoint:
            return float(checkpoint[key])
    return None


# ---------------------------------------------------------------------------
# distributed helpers
# ---------------------------------------------------------------------------

def channel_names(args) -> list[str]:
    """The input channels this run feeds, in order.

    One source of truth on purpose. The order is a contract with three readers --
    `models/dataset.py` builds it, the checkpoint records it, and
    `models/predictor.py` refuses a checkpoint that disagrees -- and
    `condition_depth` now needs the *index* of the Bragg channel, which moves
    with `with_wepl`. Deriving that index a second time somewhere else is how it
    would silently come to mean a different channel.
    """
    return (["ct", "energy"]
            + (["wepl"] if args.with_wepl else [])
            + (["bragg"] if args.with_bragg else [])
            + (["lateral"] if args.with_lateral else []))


def dist_info() -> tuple[int, int, int]:
    """(rank, world_size, local_rank), all zero when launched standalone."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return (
            int(os.environ["RANK"]),
            int(os.environ["WORLD_SIZE"]),
            int(os.environ.get("LOCAL_RANK", 0)),
        )
    return 0, 1, 0


def is_primary(rank: int) -> bool:
    return rank == 0


# ---------------------------------------------------------------------------
# loss
# ---------------------------------------------------------------------------

def geometry_sha256() -> str:
    """Hash the code that decides what the network sees, at train and inference.

    `models/geometry.py` permits exactly one resampling implementation on the
    production path, and entry depth anchors the box -- so geometry that changed
    between training and inference moves every beamlet's dose without raising.
    The `grid` dict below catches a changed *box*; this catches changed *code*.
    A checkpoint is a durable artifact, and those record a hash of what produced
    them.

    `models/physics.py` is on the list for the same reason, so the name is a
    little narrow for what it covers. The Bragg
    prior is an **input channel** built identically by `models/dataset.py` and
    `models/predictor.py`; change how it is computed and a checkpoint trained on
    the old definition is served the new one, silently. That is the same failure
    class as a changed box, so it belongs under the same hash rather than a
    second one nobody would check. Older checkpoints carry a two-file digest,
    so a hash mismatch against them means "older", not "wrong".

    **AND IT TRIPS ON ANY EDIT AT ALL, WHICH IS ITS MAIN FALSE ALARM.** This
    hashes *bytes*, so a comment, a docstring or an equivalence-preserving
    refactor changes it while the behaviour is pinned by
    `tests/test_entry_walk.py` and `tests/test_render_coords.py`. The two
    released checkpoints already record different digests for that reason, and
    neither matches this tree. ⇒ **nothing reads this value back**: it is
    provenance recorded in the checkpoint, not a gate. On a mismatch, run those
    tests rather than assuming the geometry moved. A guard that cries wolf gets
    ignored, and then it will not catch the case it exists for.
    """
    digest = hashlib.sha256()
    for rel in ("models/geometry.py", "models/geometry_torch.py", "models/physics.py"):
        path = Path(__file__).resolve().parents[2] / rel
        # Sorted, named, and length-prefixed so the digest cannot shift just
        # because a file was added, renamed or reordered.
        digest.update(rel.encode())
        digest.update(path.read_bytes() if path.is_file() else b"<missing>")
    return digest.hexdigest()


def shard_label_cutoff(data_root: Path) -> float:
    """The label cutoff the shards under ``data_root`` were built with.

    Read from the shards' own ``_COMPLETE`` markers rather than taken as a flag,
    so the checkpoint records what it was *actually* trained on. A flag can
    disagree with the data; a marker cannot.

    Three ways this refuses rather than guesses, all of them failures that
    otherwise reach the checkpoint as a quietly wrong number:

    * **shards disagreeing** -- a cohort mixing raw and thresholded labels trains
      a model matching neither;
    * **a marker that will not parse** -- unknown is not 0.0. A *missing*
      ``label_cutoff`` key does mean raw labels, because no build before
      2026-08-15 wrote one; an unreadable marker means we do not know;
    * **a built-looking shard with no marker** -- ``meta.json`` present and
      ``_COMPLETE`` absent is a patient that died between writing its arrays and
      being marked done. `BeamletDataset` only *prints* when it skips those, so a
      half-finished reshard would otherwise train on a smaller cohort with one
      line in a long log to say so.
    """
    root = Path(data_root)
    values, unmarked = set(), []
    for shard in sorted(p for p in root.iterdir() if p.is_dir()):
        marker = shard / "_COMPLETE"
        if not marker.exists():
            if (shard / "meta.json").exists() or (shard / "label_f16.npy").exists():
                unmarked.append(shard.name)
            continue
        try:
            record = json.loads(marker.read_text())
        except (ValueError, OSError) as exc:
            raise SystemExit(f"{marker}: unreadable ({exc}); refusing to train")
        values.add(float(record.get("label_cutoff", 0.0) or 0.0))

    if unmarked:
        raise SystemExit(
            f"{len(unmarked)} shard(s) have data but no _COMPLETE "
            f"(e.g. {unmarked[:5]}) -- finish or delete them; training on the "
            "remainder would silently use a smaller cohort"
        )
    if len(values) > 1:
        raise SystemExit(f"shards disagree on label_cutoff {sorted(values)}; refusing to train")
    return values.pop() if values else 0.0


def shard_grid(data_root: Path) -> BeamletGrid:
    """The sampling box the shards under ``data_root`` were built with.

    Same argument as ``shard_label_cutoff`` above, for a parameter that must be
    read rather than assumed. Writing ``BeamletGrid()`` into every checkpoint's
    ``grid`` and handing ``BeamletDataset`` no box would hold only while one box
    existed. A second box makes it false: the arrays would be read at their
    real shape (they come off disk that way) while ``depth_spacing`` stayed at the
    default, so the Bragg prior -- the heaviest-lifting input channel -- would be sampled at
    twice its true depth, and the checkpoint would then *record* the box it was
    not trained with, disarming the predictor's own mismatch guard.

    Refuses on disagreement rather than picking one. A cohort mixing two
    boxes trains a model that matches neither, and the shards are the only
    honest witness -- a ``--n-depth`` flag can be wrong about its own data.
    """
    root = Path(data_root)
    boxes: dict[str, BeamletGrid] = {}
    for shard in sorted(p for p in root.iterdir() if p.is_dir()):
        marker = shard / "_COMPLETE"
        if not marker.exists():
            continue
        try:
            record = json.loads(marker.read_text()).get("grid")
        except (ValueError, OSError) as exc:
            raise SystemExit(f"{marker}: unreadable ({exc}); refusing to train")
        if record:
            boxes[shard.name] = BeamletGrid.from_dict(record)

    distinct = set(boxes.values())
    if len(distinct) > 1:
        raise SystemExit(
            f"shards under {root} disagree on the sampling box: "
            + "; ".join(f"{pid}={g.as_dict()}" for pid, g in sorted(boxes.items()))
            + " -- refusing to train"
        )
    # No marker carries one only for shard sets predating the field, and every
    # one of those is the default lattice.
    return distinct.pop() if distinct else BeamletGrid()


def dose_weighted_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    floor: float = 0.05,
    peak_relative: bool = False,
) -> torch.Tensor:
    """MSE weighted toward voxels that actually carry dose.

    ``floor`` keeps the empty region contributing a little, so the model still
    learns where dose is absent; without it, predictions bleed outside the beam.
    Weights are normalized per sample so a bright beamlet does not dominate the
    batch purely by being bright. **That normalises the WEIGHT field, not the
    residual, which is still absolute** — so a beamlet with 3x the peak still
    contributes ~9x the loss at equal *relative* accuracy.

    ``peak_relative`` divides the residual by the beamlet's own peak, making the
    objective a relative error. **That is what `beam_mae` scores** (it divides by
    each map's own peak), and the default weights beamlets by peak² where the
    metric weights them by 1/peak. Measured consequence of the default: our
    absolute error is flat in peak height while `beam_mae` runs 0.01070 on the
    lowest peak quartile against 0.00541 on the highest.
    Scale-safe only because ``target`` is already ``label / dose_scale``, so
    peaks are ~0.7-1.7 and the loss magnitude moves ~40%, not 1000x.
    """
    with torch.no_grad():
        peak = target.amax(dim=(1, 2, 3, 4), keepdim=True).clamp_min(1e-8)
        weight = (target / peak).clamp(min=floor)
        weight = weight / weight.mean(dim=(1, 2, 3, 4), keepdim=True)
    residual = prediction - target
    if peak_relative:
        residual = residual / peak
    return (weight * residual ** 2).mean()


def idd_distance_boxframe(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """The IDD metric's own shape, in the beamlet frame -- and differentiable.

    ``idd_distance`` scores the depth-dose curve, and in the beamlet frame the
    box is ray-aligned, so that curve is exactly ``sum(dim=(u, v))``: no
    resampling, no rotation, no interpolation, and a clean gradient. That is the
    whole reason this can be a loss term at all.

    Same arithmetic as ``evaluation/metrics.py:idd_curve_distance`` -- both
    curves divided by the **ground truth** curve's peak, then RMS along depth --
    but **it is not that number**. The ranked one is computed after rendering
    onto the CT grid, where the curve is resampled along the beam axis through a
    rotation. This is a proxy that moves with it, which is why it carries
    ``_boxframe`` like everything else here.

    Why it exists as a *loss* and not only a metric: a per-voxel objective does
    not control a projection. Errors along the summed axis are free to cancel or
    to accumulate, so a model can be pointwise good and still misplace the range
    -- which is precisely what ``idd_distance`` punishes, and where the headroom
    is (``beam_mae`` is ~40% irreducible label noise).
    """
    idd_pred = prediction.sum(dim=(3, 4))
    idd_true = target.sum(dim=(3, 4))
    peak = idd_true.amax(dim=2, keepdim=True).clamp_min(1e-8)
    squared = ((idd_pred - idd_true) / peak) ** 2
    return squared.mean(dim=2).sqrt().mean()


def conditional_bias_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    orders: tuple[int, ...] = (1, 2, 3),
) -> torch.Tensor:
    """The **coherent** part of the residual -- the only part `dvh_score` reads.

    **WHY A MEAN AND NOT AN ABSOLUTE.** `D98` and `V95` are order statistics of
    the *composed* plan, where hundreds of beamlets sum. Independent residuals
    average down as sqrt(N) and leave the DVH intact; a residual that is
    systematic at a given dose level survives composition undiminished. So this
    squares dose-weighted **means** of the residual, which are ~0 for scatter
    however large and nonzero only for bias. ⇒ deliberately blind to what
    `dose_weighted_loss` already handles, and it adds no second opinion about it.

    **WHY MOMENTS AND NOT BINS.** The condition wanted is conditional
    unbiasedness, ``E[residual | dose level] = 0``; indicator bins are the crudest
    basis for it and this term was written that way first. Two defects killed it.
    (1) A bin's mean averages scatter down only as ``1/n``, and the sparse bins are
    the HIGH-dose ones -- on a small box an sd-0.05 scatter and a +1% coherent
    offset scored **identically, 1.0x**, with nothing raising. (2) A post-hoc
    correction of the same shape -- a dose-binned curve per patient -- generalised
    worse than a single per-patient scale in our experiments: more parameters,
    strictly worse. A binned loss term would be copying that.
    ⇒ ``m_k = sum(r * t^k) / sum(t^k)`` for k = 1, 2, 3: no edges, every voxel
    contributing smoothly, effective ``n`` large at every order. ``k=1`` is the
    dose-weighted net bias, the quantity a per-patient scale would remove; ``k=2,3``
    carry the TILT of bias against dose, which a net-only term cannot see -- a
    beamlet hot at its peak and cold in its plateau has zero net bias and still
    biases a composed plan, because a target voxel samples mostly plateau.

    Residual and weights are divided by the beamlet's **own peak**, so the term
    is scale-free and matches `beam_mae`'s denominator. Weights come from the
    TARGET, never the prediction: keyed on the prediction the network could shrink
    a region instead of correcting it.
    """
    peak = target.amax(dim=(1, 2, 3, 4), keepdim=True).clamp_min(1e-8)
    rel = target / peak
    residual = (prediction - target) / peak
    dims = (1, 2, 3, 4)
    total = prediction.new_zeros(())
    for k in orders:
        with torch.no_grad():
            weight = rel ** k
            norm = weight.sum(dim=dims).clamp_min(1e-8)
        total = total + (((residual * weight).sum(dim=dims) / norm) ** 2).mean()
    return total


def rendered_bias_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    geometry: dict,
    grid,
    orders: tuple[int, ...] = (1, 2, 3),
) -> torch.Tensor:
    """:func:`conditional_bias_loss`, but in the frame the metric is scored in.

    **WHY THE BOX-FRAME VERSION FAILED, WHICH IS THE ONLY REASON THIS EXISTS.**
    The box-frame term halved beamlet-frame coherent bias (0.347% -> 0.142% on
    held-out beamlets) and did **not** make the composed plan dose less biased.
    The term moved exactly what it was built to move, in the wrong frame. The
    mechanism is the frame: the network trains against a
    **precompensated** label carrying `amp_max` 1.207, and that label is not
    dose -- it is what ``render`` inverts back into dose. Driving the residual to
    conditional zero *against the label* is not being unbiased in **rendered**
    dose, because ``render . precompensate`` is not the identity.
    => the term was aimed one transform too early in the chain, and this moves it.

    **AND THE DEFECT IT AIMS AT IS THE MODEL'S, NOT THE LABEL PIPELINE'S.**
    Rendering ``precompensate(gt)`` -- the dose a *flawless* network would
    deliver -- composes to almost no offset, far below the trained model's. So
    the bias this term targets is the network's to fix, not the round trip's.

    **WHAT IS AND IS NOT CHANGED FROM THE BOX-FRAME TERM.** The statistic is
    identical and deliberately so: dose-weighted **moments** ``sum(r*t^k)/sum(t^k)``
    for k = 1,2,3, reduced **per beamlet** and then squared -- so the fixed point
    is per-beamlet, hence per-patient, zero bias rather than a pooled mean, which
    is what makes the target per-patient zero bias instead of a cohort scalar.
    Bins were refuted twice over and are not revisited (see
    :func:`conditional_bias_loss`). The ONLY change is which tensors it is fed.

    **THE WEIGHTS COME FROM THE RENDERED TARGET, AND THAT IS THE POINT.** In
    the box frame the weighting field is the precompensated label, which
    concentrates amplification into the Bragg peak (``amp_max`` ~1.2) -- so ``t^3``
    there weights a sharpened artefact, not delivered dose. Rendered, ``t`` is
    the dose itself. Still keyed on the TARGET and never the prediction:
    keyed on the prediction the network could shrink a region instead of
    correcting it.

    **PER-BEAMLET PYTHON LOOP, NOT A BATCHED OP.** Every beamlet has its own
    ray, so ``render_coordinates`` differs per sample and there is nothing to
    batch. The rendered TARGET is built under ``no_grad`` -- it is a constant of
    the data -- so only one of the two renders carries a graph.

    ``geometry`` carries ``ray`` (source[3], target[3], entry_depth) and the
    patient's ``vol_origin``/``vol_spacing``/``vol_shape``, from
    ``BeamletDataset(with_geometry=True)``.
    """
    from models.geometry import VolumeGeometry
    from models import geometry_torch as GT

    ray = geometry["ray"].detach().cpu().numpy()
    origin = geometry["vol_origin"].detach().cpu().numpy()
    spacing = geometry["vol_spacing"].detach().cpu().numpy()
    shape = geometry["vol_shape"].detach().cpu().numpy()

    total = prediction.new_zeros(())
    counted = 0
    for i in range(prediction.shape[0]):
        source, tgt, entry = ray[i, 0:3], ray[i, 3:6], float(ray[i, 6])
        geom = VolumeGeometry(
            origin=np.asarray(origin[i], dtype=float),
            spacing=np.asarray(spacing[i], dtype=float),
            shape=tuple(int(v) for v in shape[i]),
        )
        # ONE set of coordinates for both renders. Rebuilding them per call
        # would be pure cost, and any drift between the two would show up as a
        # bias this term would then chase.
        lo, hi, coords = GT.render_coordinates(
            source, tgt, entry, grid, geom, prediction.device
        )
        if coords is None:
            # The box misses the volume. Skipped, never counted as zero bias:
            # a zero here would dilute the mean toward "unbiased" on exactly the
            # beamlets the term has no opinion about.
            continue
        with torch.no_grad():
            t_r = GT._sample_trilinear(target[i, 0], coords, 0.0)
            peak = t_r.amax().clamp_min(1e-8)
            rel = t_r / peak
        p_r = GT._sample_trilinear(prediction[i, 0], coords, 0.0)
        residual = (p_r - t_r) / peak
        for k in orders:
            with torch.no_grad():
                weight = rel ** k
                norm = weight.sum().clamp_min(1e-8)
            total = total + ((residual * weight).sum() / norm) ** 2
        counted += 1
    if counted == 0:
        return total
    return total / counted


@torch.no_grad()
def beam_mae_proxy(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The Level 1 metric evaluated in the beamlet frame.

    Not identical to the official score -- that is computed after rendering back
    onto the CT grid -- but it moves with it and needs no geometry, so it is
    cheap enough to log every step.
    """
    peak = target.amax(dim=(1, 2, 3, 4), keepdim=True).clamp_min(1e-8)
    mask = target >= 0.1 * peak
    error = (prediction - target).abs() * mask
    denom = (mask.sum(dim=(1, 2, 3, 4)).clamp_min(1) * peak.squeeze()).clamp_min(1e-8)
    return (error.sum(dim=(1, 2, 3, 4)) / denom).mean()


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def total_loss(
    prediction: torch.Tensor, target: torch.Tensor, idd_weight: float,
    peak_relative: bool = False, bias_weight: float = 0.0,
    bias_frame: str = "box", geometry: dict | None = None, grid=None,
) -> tuple[torch.Tensor, dict]:
    """``dose_weighted_loss + idd_weight * idd + bias_weight * conditional_bias``.

    Returns the parts as well as the sum, because a combined loss whose split
    nobody logs is a combined loss nobody can debug -- and the two terms have
    genuinely different scales (the dose term is a normalised MSE, the IDD term
    a normalised RMS, so at init they sit roughly an order of magnitude apart).
    """
    dose = dose_weighted_loss(prediction, target, peak_relative=peak_relative)
    # `rendered` is the frame the metric is scored in and `box` is the one
    # that failed in our experiments; the flag exists so the two can be run as arms, not
    # because either is a default worth defending.
    if bias_weight <= 0:
        bias = None
    elif bias_frame == "rendered":
        if geometry is None or grid is None:
            raise ValueError(
                "bias_frame='rendered' needs `geometry` and `grid`; without them "
                "there is no way to render, and silently falling back to the box "
                "frame would run the box-frame variant under the rendered name."
            )
        bias = rendered_bias_loss(prediction, target, geometry, grid)
    else:
        bias = conditional_bias_loss(prediction, target)
    # `.detach()` before reading the value: `float()` on a tensor that still
    # carries grad works but warns, and a warning printed once per rank into a
    # log nobody greps is how a real one gets missed later.
    parts = {"loss_dose": dose.detach().item()}
    out = dose
    if bias is not None:
        parts[f"loss_bias_{bias_frame}"] = bias.detach().item()
        out = out + bias_weight * bias
    if idd_weight <= 0:
        return out, parts
    idd = idd_distance_boxframe(prediction, target)
    parts["loss_idd_boxframe"] = idd.detach().item()
    return out + idd_weight * idd, parts


def evaluate(model, loader, device, amp_dtype) -> dict:
    """Score every beamlet the loader yields.

    The loader is capped by construction (see ``--val-beamlets``), never here:
    a cap applied at this level truncates whatever order the loader happens to
    have, which is patient-major.
    """
    model.eval()
    loss_sum = mae_sum = idd_sum = 0.0
    seen = 0
    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                prediction = model(inputs)
            # Both reduce with .mean() over the batch, so a plain mean of batch
            # means would overweight a short trailing batch.
            n = int(target.shape[0])
            loss_sum += float(dose_weighted_loss(prediction.float(), target)) * n
            mae_sum += float(beam_mae_proxy(prediction.float(), target)) * n
            # Reported unconditionally, whatever the loss weights: it is the
            # axis with the headroom, so an arm that does not train on it still
            # has to be comparable to one that does.
            idd_sum += float(idd_distance_boxframe(prediction.float(), target)) * n
            seen += n
    model.train()
    return emit({
        "loss": loss_sum / max(seen, 1),
        "beam_mae_boxframe": mae_sum / max(seen, 1),
        "idd_distance_boxframe": idd_sum / max(seen, 1),
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--augment-flip", action="store_true",
                        help="mirror the beamlet box laterally, at random, on "
                             "the TRAIN split only. The lateral samples sit at "
                             "`(arange(n) - (n-1)/2) * spacing`, exactly "
                             "symmetric about the beam axis, so the flip is a "
                             "relabelling rather than an interpolation and the "
                             "physics is unchanged. Depth is never flipped.")
    parser.add_argument("--ema-decay", type=float, default=0.0, metavar="D",
                        help="keep an exponential moving average of the weights "
                             "and VALIDATE AND SAVE FROM IT (0 = off). A single "
                             "model, so it costs nothing at inference -- unlike "
                             "the ensemble that measured the same variance. "
                             "0.9999 is a ~10k-step horizon, ~6 epochs here.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--base-features", type=int, default=24)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--arch", default="unet", choices=sorted(ARCHITECTURES),
                        help="'factorised' predicts IDD(depth) x lateral kernel")
    parser.add_argument("--with-wepl", action="store_true")
    parser.add_argument("--with-bragg", action="store_true",
                        help="append the Generic-machine Bragg prior channel")
    parser.add_argument("--with-lateral", action="store_true",
                        help="append the analytic double-Gaussian lateral kernel "
                             "and condition the softmax head on it")
    parser.add_argument("--with-depth", action="store_true",
                        help="condition the depth branch on the Bragg prior: "
                             "IDD = relu(a*Z + b + residual), so the curve "
                             "starts at the prior instead of being rebuilt from "
                             "trunk features. Requires --with-bragg")
    parser.add_argument("--rsp", default="linear", choices=["linear", "hlut"],
                        help="the HU conversion behind the Bragg prior's WEPL, "
                             "used in place of a stopping-power curve. 'hlut' is "
                             "the dataset's own HU-to-density table "
                             "(beam_parameters.json); 'linear' is 1 + HU/1000")
    parser.add_argument("--peak-relative-loss", action="store_true",
                        help="divide the residual by each beamlet's own peak, so "
                             "the objective is RELATIVE error -- what beam_mae "
                             "scores. The default weights a beamlet by peak^2 "
                             "where the metric weights it by 1/peak. Selection and the reported val loss are "
                             "unaffected: `best` is chosen on "
                             "beam_mae_boxframe_testmix, a metric, so arms stay "
                             "comparable.")
    parser.add_argument("--init-from", default=None, metavar="CKPT",
                        help="warm-start the WEIGHTS from a checkpoint. Not a "
                             "resume: no optimiser state, no schedule phase, no "
                             "start epoch -- see the module docstring. Use a low "
                             "--lr, or the fresh OneCycle warm-up undoes it.")
    parser.add_argument("--bias-weight", type=float, default=0.0,
                        help="weight on `conditional_bias_loss` -- the squared "
                             "dose-weighted moments (orders 1-3) of the signed "
                             "residual, the coherent component `dvh_score` reads. 0 "
                             "reproduces every arm trained before it exists.")
    parser.add_argument("--save-every-epoch", action="store_true",
                        help="also write `<run>_ep<NNN>.pt` every validation, "
                             "beside `_best.pt`. For any arm whose target is "
                             "NOT the `beam_mae` selector -- a bias arm above "
                             "all -- because `best` would otherwise pick the "
                             "epoch that is best at what the arm does not move. "
                             "~213 MB per epoch.")
    parser.add_argument("--bias-frame", default="box", choices=("box", "rendered"),
                        help="WHICH FRAME the bias term is computed in, and it is "
                             "the whole result. `box` scores the residual against "
                             "the PRECOMPENSATED label -- that variant halved "
                             "beamlet-frame bias without making the composed dose "
                             "less biased, because "
                             "`render . precompensate` is not the identity. "
                             "`rendered` maps both sides onto the CT lattice "
                             "first, which is the frame `dvh_score` is scored in. "
                             "`rendered` forces `with_geometry` on the loaders "
                             "and is therefore incompatible with --augment-flip.")
    parser.add_argument("--idd-weight", type=float, default=0.0,
                        help="weight on idd_distance_boxframe in the loss. "
                             "Defaults OFF so an existing config reproduces the "
                             "run it described; the arms set it explicitly")
    parser.add_argument(
        "--grad-accum", type=int, default=1, metavar="N",
        help="micro-batches per optimiser step. Effective batch = batch_size x "
             "world_size x N, and it is the EFFECTIVE batch that arms are "
             "matched on. This exists because the "
             "resized box costs ~10.2 GiB per unit of batch, so the "
             "effective batch an arm must hold no longer divides evenly into "
             "whatever cards happen to be free -- without it, only an H200 can "
             "run these arms at all. Equivalence to an un-accumulated run is "
             "pinned by tests/test_grad_accum.py, not assumed.")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="stop early; for smoke tests")
    parser.add_argument("--train-patients", nargs="*", default=None,
                        help="override the train split; for smoke tests while "
                             "preprocessing is still running")
    parser.add_argument("--val-patients", nargs="*", default=None,
                        help="override the val split. Only for smoke tests -- "
                             "pointing this at train patients measures nothing")
    parser.add_argument("--val-beamlets", type=int, default=2000,
                        help="beamlets per validation pass, spread evenly over "
                             "every val patient; 0 = the whole val split")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"])
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    # RESUME INSURANCE, not a shippable checkpoint. `<run>_best.pt` stores
    # weights only, so a killed run restarts from step 0. This writes the
    # OPTIMISER and OneCycleLR state too, every N OPTIMISER steps (0 = off), so
    # a resume has what it needs; no resume path is implemented here. The file is ~3x the weights (Adam keeps two moments per
    # parameter) and is rewritten in place, so the cost is one ~640 MB write per
    # interval, not per-interval growth. It is written atomically (tmp +
    # os.replace): a kill mid-write leaves the previous state intact rather than
    # a truncated file that loads and then explodes.
    parser.add_argument("--save-state-every", type=int, default=0, metavar="STEPS")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    if args.config:
        if yaml is None:
            raise SystemExit("pyyaml required for --config")
        cfg = yaml.safe_load(Path(args.config).read_text()) or {}
        # Only fill values the command line did not set, so an explicit flag
        # always beats the config file.
        explicit = {
            a.lstrip("-").replace("-", "_")
            for a in sys.argv[1:] if a.startswith("--")
        }
        # A key the parser does not know is a typo, and a typo that is skipped
        # silently runs the default instead -- a different experiment that
        # looks like the configured one. Refuse it.
        unknown = sorted(k for k in cfg if not hasattr(args, k.replace("-", "_")))
        if unknown:
            raise SystemExit(f"{args.config}: unknown config keys {unknown}")
        for key, value in cfg.items():
            key = key.replace("-", "_")
            if key not in explicit:
                setattr(args, key, value)

    rank, world_size, local_rank = dist_info()
    distributed = world_size > 1
    if distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}[args.amp]
    if device.type != "cuda":
        amp_dtype = None
    elif amp_dtype is torch.bfloat16 and torch.cuda.get_device_capability(local_rank)[0] < 8:
        # Native bf16 needs Ampere (compute capability 8.0+). Do NOT trust
        # torch.cuda.is_bf16_supported(): it returns True on Turing, where bf16
        # is emulated -- measured on a Quadro RTX 6000 (7.5) at 8071 ms/step
        # against 882 ms/step for fp16, a 9.2x penalty for silently taking the
        # slow path.
        if is_primary(rank):
            print(f"[amp] {torch.cuda.get_device_name(local_rank)} is compute "
                  f"{torch.cuda.get_device_capability(local_rank)}; bf16 would be "
                  "emulated, using fp16 instead", flush=True)
        amp_dtype = torch.float16

    run_name = args.run_name or f"doserad_{time.strftime('%Y%m%d_%H%M%S')}"
    data_root = Path(args.data_root) if args.data_root else DEFAULT_DATA_ROOT

    from models.splits import get_splits
    splits = get_splits()
    train_patients = args.train_patients or splits["train"]
    val_patients = args.val_patients or splits["val"]
    # **GATE ON THE OVERLAP, NEVER ON WHICH FLAG WAS TYPED.** This used to
    # test `args.val_patients and ...`, so a run that passed `--train-patients`
    # covering all 75 and left `--val-patients` alone got the registry's val 6
    # -- entirely inside its own training set -- and the guard could not fire.
    # The released base is such a run: it predates this guard, so its checkpoint
    # does not record the overlap, and its epoch was chosen on patients it had
    # seen (paper §2.3). The overlap is a property of the two LISTS; how
    # they arrived is not part of the question.
    val_overlap = sorted(set(val_patients) & set(train_patients))
    if val_overlap:
        print(f"[WARNING] {len(val_overlap)} of {len(val_patients)} val patients "
              f"are in train -- `best` selects on MEMORISED data and the val "
              f"number is not comparable to any run without this overlap: "
              f"{', '.join(val_overlap)}", flush=True)
    # Read off the shards, never assumed: see `shard_grid`.
    grid = shard_grid(data_root)
    channels = dict(with_wepl=args.with_wepl, with_bragg=args.with_bragg,
                    with_lateral=args.with_lateral, rsp=args.rsp, grid=grid)
    # The augmentation goes to TRAIN ONLY. `best` selects on val, so a val
    # set that mirrors at random moves the selector for a reason that is not the
    # model.
    # Only the TRAIN loader carries geometry: the rendered term is a training
    # signal, and `evaluate` scores `beam_mae_boxframe`, which needs none of it.
    # Handing it to val would pay the meta.json walk for nothing.
    want_geom = args.bias_frame == "rendered" and args.bias_weight > 0
    train_set = BeamletDataset(data_root, train_patients,
                               augment_flip=args.augment_flip,
                               with_geometry=want_geom, **channels)
    val_set = BeamletDataset(data_root, val_patients, **channels)

    # The val index is patient-major (models/dataset.py), so a subset taken off
    # its head is one patient. Capping validation by *batch* count would do
    # exactly that -- and would score a different number of beamlets per arm
    # whenever batch_size differed, inside a comparison built to hold effective
    # batch constant. A float stride spans
    # every patient, is identical every epoch, and does not move with
    # batch_size, so val curves stay comparable across runs.
    if args.val_beamlets and args.val_beamlets < len(val_set):
        stride = len(val_set) / args.val_beamlets
        val_subset = Subset(val_set, [int(i * stride) for i in range(args.val_beamlets)])
    else:
        val_subset = val_set

    if is_primary(rank):
        print(f"run        : {run_name}")
        print(f"data       : {data_root}")
        print(f"box        : {grid.shape} @ ({grid.depth_spacing}, "
              f"{grid.lat_u_spacing}, {grid.lat_v_spacing}) mm  "
              f"entry_margin={grid.entry_margin_mm} mm")
        print(f"train/val  : {len(train_set)} / {len(val_set)} beamlets")
        print(f"val/pass   : {len(val_subset)} beamlets over {len(val_patients)} patients")
        print(f"arch       : {args.arch}  idd_weight={args.idd_weight}")
        print(f"channels   : {train_set.n_channels}  "
              f"(wepl={args.with_wepl} bragg={args.with_bragg} lateral={args.with_lateral} "
              f"depth={args.with_depth} rsp={args.rsp})")
        print(f"world size : {world_size}   device: {device}   amp: {args.amp}")
        # Resolved, not the argument: the number that has to
        # match across arms is this product, and reading it off three separate
        # values is how two arms come to differ without anyone noticing.
        print(f"eff. batch : {args.batch_size} x {world_size} x {args.grad_accum} "
              f"= {args.batch_size * world_size * args.grad_accum}")
        sys.stdout.flush()

    train_sampler = DistributedSampler(train_set) if distributed else None
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=train_sampler is None,
        sampler=train_sampler, num_workers=args.workers, pin_memory=True,
        drop_last=True, persistent_workers=args.workers > 0,
    )
    # Scored as two subsets rather than one, so the selector can weight them to
    # the deployment mixture. Splitting here (not inside `evaluate`) keeps the
    # loss functions batch-reducing, which is all they were ever asked to be.
    def _site_subset(is_thoracic: bool):
        rows = getattr(val_subset, "indices", range(len(val_set)))
        idx = [i for i, row in enumerate(rows)
               if val_set.shards[val_set.index[row][0]].startswith("1THB") is is_thoracic]
        return Subset(val_subset, idx) if idx else None

    val_thoracic, val_abdominal = _site_subset(True), _site_subset(False)

    def _val_loader(subset):
        return None if subset is None else DataLoader(
            subset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )

    site_loaders = {"thoracic": _val_loader(val_thoracic),
                    "abdominal": _val_loader(val_abdominal)}

    val_loader = DataLoader(
        val_subset, batch_size=args.batch_size, shuffle=False,
        num_workers=max(args.workers // 2, 1), pin_memory=True,
    )

    channels = channel_names(args)
    if args.with_depth and not args.with_bragg:
        raise SystemExit(
            "--with-depth conditions the depth curve on the Bragg prior, which "
            "--with-bragg is what supplies. Refusing rather than silently "
            "training the unconditioned arm under the conditioned arm's name."
        )
    if len(channels) != train_set.n_channels:
        raise SystemExit(
            f"the flags describe channels {channels} but the dataset yields "
            f"{train_set.n_channels}; one of them is wrong and the checkpoint "
            "would record the other"
        )
    model = build_network(
        args.arch,
        in_channels=train_set.n_channels,
        base_features=args.base_features,
        levels=args.levels,
        **({"condition_lateral": True} if args.with_lateral else {}),
        **({"condition_depth": True, "bragg_index": channels.index("bragg")}
           if args.with_depth else {}),
    ).to(device)
    if args.init_from:
        # **A WARM START IS NOT A RESUME, AND THE DOCSTRING'S REFUSAL IS ABOUT
        # RESUME.** No optimiser moments and no OneCycleLR phase are restored --
        # this loads WEIGHTS ONLY and the schedule starts fresh at step 0, which
        # is the whole point when the objective has changed. So it is only
        # sound at a low `--lr`: at the original 3e-4 the fresh OneCycle warm-up
        # would walk the loaded weights straight back out of their basin.
        # Refuses on any architecture mismatch rather than filtering keys: a
        # partially-loaded net trains happily and silently is not the model whose
        # name the checkpoint carries.
        blob = torch.load(args.init_from, map_location="cpu", weights_only=True)
        state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise SystemExit(
                f"--init-from {args.init_from}: {len(missing)} missing and "
                f"{len(unexpected)} unexpected tensors. This is a different "
                f"architecture, not a warm start.\n  missing: {list(missing)[:5]}"
                f"\n  unexpected: {list(unexpected)[:5]}"
            )
        if is_primary(rank):
            print(f"init_from  : {args.init_from} "
                  f"(epoch {blob.get('epoch') if isinstance(blob, dict) else '?'}, "
                  f"weights only -- optimiser and schedule start fresh)", flush=True)
    if is_primary(rank):
        print(f"parameters : {model.parameter_count() / 1e6:.2f}M\n", flush=True)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank]
        )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # OPTIMISER steps, not micro-batches -- `OneCycleLR` anneals over the
    # steps it is actually stepped on, and dividing by `grad_accum` is what
    # keeps the schedule identical to an un-accumulated run of the same
    # effective batch. Getting this wrong anneals `grad_accum`x too slowly and
    # the run simply ends mid-schedule, which looks like undertraining.
    total_steps = args.max_steps or args.epochs * (len(train_loader) // args.grad_accum)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=max(total_steps, 1), pct_start=0.1
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype is torch.float16)

    run = None
    if args.wandb and is_primary(rank):
        try:
            import wandb
            run = wandb.init(project="doserad2026", name=run_name,
                             config=vars(args), save_code=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[wandb] disabled: {exc}", flush=True)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    stop = {"now": False}

    # **THE EMA IS THE THING VALIDATED AND SAVED, OR IT IS NOTHING.** Keeping
    # a shadow copy and still selecting on the live weights would cost the memory
    # and buy nothing -- and the failure is silent, because both sets of weights
    # train fine. Parameters only: the network is GroupNorm throughout, so there
    # are no running statistics to average (a BatchNorm layer would need its
    # buffers copied too, and this would be wrong for it).
    ema_module = model.module if distributed else model
    ema = ({k: v.detach().clone().float() for k, v in ema_module.state_dict().items()}
           if args.ema_decay > 0 else None)

    def ema_update():
        d = args.ema_decay
        with torch.no_grad():
            for k, v in ema_module.state_dict().items():
                if ema[k].is_floating_point():
                    ema[k].mul_(d).add_(v.detach().float(), alpha=1.0 - d)
                else:
                    ema[k].copy_(v.detach())

    @contextlib.contextmanager
    def ema_weights():
        """Swap the EMA in for validation and the save, then put it back."""
        if ema is None:
            yield
            return
        live = {k: v.detach().clone() for k, v in ema_module.state_dict().items()}
        ema_module.load_state_dict({k: v.to(live[k].dtype) for k, v in ema.items()})
        try:
            yield
        finally:
            ema_module.load_state_dict(live)

    def handle_sigterm(signum, frame):  # noqa: ARG001
        # Stop runs with SIGTERM: finish the epoch's
        # bookkeeping and close wandb cleanly rather than dying mid-write.
        print("\n[signal] SIGTERM received, finishing current step", flush=True)
        stop["now"] = True

    signal.signal(signal.SIGTERM, handle_sigterm)

    def validate_and_checkpoint(epoch: int, step: int, best: float) -> float:
        with ema_weights():
            return _validate_and_checkpoint(epoch, step, best)

    def _validate_and_checkpoint(epoch: int, step: int, best: float) -> float:
        metrics = evaluate(model, val_loader, device, amp_dtype)

        # Per site, then recombined at the mixture the leaderboard actually
        # scores. The unweighted number stays in the log and on W&B -- it is what
        # every run before 2026-08-15 selected on, and dropping it would make old
        # and new runs silently incomparable.
        per_site = {name: evaluate(model, loader, device, amp_dtype)
                    for name, loader in site_loaders.items() if loader is not None}
        if len(per_site) == 2:
            for name, m in per_site.items():
                metrics[f"beam_mae_boxframe_{name}"] = m["beam_mae_boxframe"]
            metrics["beam_mae_boxframe_testmix"] = (
                THORACIC_SHARE * per_site["thoracic"]["beam_mae_boxframe"]
                + (1 - THORACIC_SHARE) * per_site["abdominal"]["beam_mae_boxframe"]
            )
        # Log-file names stay flat; W&B gets the `val/` prefix that puts these
        # beside their train counterparts instead of in the catch-all section.
        print(f"  [val] epoch {epoch} step {step}: " + "  ".join(
            f"val_{k}={v:.5f}" for k, v in metrics.items()), flush=True)
        if run:
            run.log(emit({f"val/{k}": v for k, v in metrics.items()}), step=step)
        # Select on the deployment mixture when both sites are present, and say
        # so in the checkpoint -- two runs selected on different mixtures are not
        # comparable and nothing else would record which was used.
        selector = "beam_mae_boxframe_testmix" if "beam_mae_boxframe_testmix" in metrics \
            else "beam_mae_boxframe"
        if metrics[selector] < best:
            best = metrics[selector]
            target_module = model.module if distributed else model
            torch.save(
                emit({
                    "model": target_module.state_dict(),
                    "config": vars(args),
                    "in_channels": train_set.n_channels,
                    # Explicit rather than left to `config`, for the same reason
                    # `in_channels` is: models/predictor.py rebuilds the network
                    # from the checkpoint alone, and loading a factorised
                    # state_dict into a U-Net raises, while the reverse -- a
                    # checkpoint with no `arch` at all -- must still mean "unet"
                    # so every pre-2026-08-15 checkpoint keeps loading.
                    "arch": args.arch,
                    # The HU conversion the Bragg prior was built on.
                    # Inference must match it or every predicted range moves.
                    "rsp": args.rsp,
                    # Which derived input channels were appended, in order.
                    # `in_channels` alone stopped being enough the moment there
                    # was more than one optional channel: 3 could mean WEPL or
                    # Bragg, and feeding the wrong one does not raise, it
                    # silently predicts wrong dose.
                    "channels": channel_names(args),
                    # Not derivable from `channels`: depth conditioning reuses
                    # the Bragg channel rather than adding one, so the channel
                    # list looks identical to the arm without it.
                    "condition_depth": bool(args.with_depth),
                    "dose_scale": DOSE_SCALE,
                    # The box this was trained with. Entry depth is a
                    # registration anchor, so a checkpoint inferred under a
                    # different box predicts dose in the wrong place without
                    # crashing -- and every checkpoint predating 2026-08-13 had
                    # to be deleted precisely because nothing recorded this.
                    "grid": grid.as_dict(),
                    "geometry_sha256": geometry_sha256(),
                    # The label cutoff the shards were built with, read off
                    # their own _COMPLETE rather than from a flag, so it
                    # describes the data actually trained on. A model trained on
                    # thresholded labels emits ~0 below that value; served under
                    # a SMALLER platform cutoff it would blank a band that is
                    # still scored, and nothing downstream would raise.
                    "label_cutoff": shard_label_cutoff(data_root),
                    "val_selector": selector,
                    # How much of the selector was memorised. A checkpoint
                    # whose val set sits inside its train set won `best` on data
                    # it had already fitted, and a reader comparing its
                    # `val_beam_mae_boxframe` against another run's is comparing
                    # two different quantities. Recorded rather than merely
                    # warned about: a warning at epoch 0 of a 240-epoch run is
                    # gone by the time anyone reads the checkpoint.
                    "val_train_overlap": val_overlap,
                    "data_root": str(data_root),
                    "epoch": epoch,
                    "step": step,
                    # Beamlet-frame, and named so. `val_beam_mae_boxframe` is
                    # the selector this checkpoint won on; the CT-frame
                    # number is scored separately and is
                    # not in this file. Readers take either name via
                    # `val_beam_mae_boxframe()`, so older checkpoints load.
                    "val_beam_mae_boxframe": best,
                }),
                checkpoint_dir / f"{run_name}_best.pt",
            )
            print(f"  [ckpt] new best {best:.5f}", flush=True)
        if args.save_every_epoch:
            # **BECAUSE `best` SELECTS ON THE WRONG METRIC FOR A BIAS ARM.**
            # The selector is val `beam_mae_boxframe`, and `dose_weighted_loss`
            # ranks a 1% coherent offset at 0.003x a same-sized scatter -- so on
            # a run whose whole purpose is coherent bias, `best` picks the epoch
            # that is best at the thing the term does not move.
            # ⇒ keep every epoch and choose afterwards on the composed dose,
            # which `best` cannot see. The submitted model is the last of three.
            # `_best.pt` is still written on `beam_mae` so this run stays
            # comparable to every arm before it -- it is simply not what ships.
            torch.save(
                emit({
                    "model": (model.module if distributed else model).state_dict(),
                    "config": vars(args),
                    "in_channels": train_set.n_channels,
                    # `channels`, the name `models/predictor.py` reads. A
                    # later edit of this file wrote it as `channel_names`, which
                    # makes a per-epoch checkpoint unloadable for inference --
                    # the predictor falls back to the legacy 3-channel rule,
                    # guesses WEPL and raises. The released checkpoint was
                    # written before that and carries `channels`;
                    # tests/test_checkpoint_fields.py now pins the name.
                    "channels": train_set.channel_names,
                    # The same identity block `_best.pt` writes, for the same
                    # reason: `models/predictor.py` rebuilds the network from
                    # the checkpoint alone. Without `dose_scale` it falls back
                    # to its own constant, which is a silent ~1000x error the
                    # day that constant changes.
                    "arch": args.arch,
                    "rsp": args.rsp,
                    "condition_depth": bool(args.with_depth),
                    "dose_scale": DOSE_SCALE,
                    "grid": grid.as_dict(),
                    "geometry_sha256": geometry_sha256(),
                    "label_cutoff": shard_label_cutoff(data_root),
                    "val_selector": selector,
                    "val_train_overlap": val_overlap,
                    "data_root": str(data_root),
                    "epoch": epoch,
                    "step": step,
                    # THIS epoch's value, not the running best -- the whole
                    # point of the file is to be selectable against later.
                    "val_beam_mae_boxframe": metrics[selector],
                }),
                checkpoint_dir / f"{run_name}_ep{epoch:03d}.pt",
            )
        return best

    step = 0
    started = time.time()
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for micro, batch in enumerate(train_loader):
            inputs = batch["inputs"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            # Stays on the HOST. `render_coordinates` takes numpy world
            # coordinates and builds its own device tensor, so uploading these
            # would be a round trip to hand it back what it already has.
            # `None` when the loader was not built with `with_geometry`, and
            # `total_loss` raises on that rather than falling back to the box
            # frame -- a silent fallback would run the box-frame variant under
            # the rendered name.
            geometry = ({k: batch[k] for k in
                         ("ray", "vol_origin", "vol_spacing", "vol_shape")}
                        if "ray" in batch else None)
            # The last micro-batch of the group is the one that reduces and
            # steps. `drop_last=True` on the loader makes the group count exact,
            # so no group is ever short and no step sees a smaller batch than
            # the effective one the run is matched to.
            boundary = (micro + 1) % args.grad_accum == 0

            # `no_sync` on every micro-batch but the last: without it DDP
            # all-reduces 53.26M parameters `grad_accum` times per optimiser
            # step instead of once, which is the whole cost of accumulating.
            # Correctness does not depend on it -- gradients accumulate into the
            # same buffers either way -- so it is guarded rather than assumed.
            sync = model.no_sync() if (distributed and not boundary) else nullcontext()
            with sync:
                with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                    prediction = model(inputs)
                    loss, loss_parts = total_loss(
                        prediction.float(), target, args.idd_weight,
                        peak_relative=args.peak_relative_loss,
                        bias_weight=args.bias_weight,
                        bias_frame=args.bias_frame,
                        geometry=geometry, grid=grid)
                # Divided by the group size so the accumulated gradient is the
                # MEAN over the effective batch, not its sum. Without this,
                # accumulating N micro-batches multiplies the gradient by N and
                # `--grad-accum` silently becomes a learning-rate change.
                # `tests/test_grad_accum.py` pins the equivalence.
                scaled = loss / args.grad_accum
                if scaler.is_enabled():
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()

            if not boundary:
                continue

            if scaler.is_enabled():
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            # After the step, not before: `set_to_none` between micro-batches
            # would discard everything accumulated so far.
            optimizer.zero_grad(set_to_none=True)
            if scheduler.last_epoch < scheduler.total_steps - 1:
                scheduler.step()
            if ema is not None:
                ema_update()
            step += 1

            # Resume insurance. Deliberately NOT gated on `best`: the point is
            # to survive a kill, and the last good validation may be far behind.
            # Rank 0 only -- the others' optimiser state is a shard of the same
            # thing under DDP, and `_best.pt` is written the same way.
            if is_primary(rank) and args.save_state_every and step % args.save_state_every == 0:
                target_module = model.module if distributed else model
                state_path = checkpoint_dir / f"{run_name}_state.pt"
                tmp_path = state_path.with_suffix(".pt.tmp")
                torch.save(
                    {
                        "resume_only": True,   # not a shippable checkpoint
                        "model": target_module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "epoch": epoch,
                        "step": step,
                        "best": best,
                        "config": vars(args),
                    },
                    tmp_path,
                )
                os.replace(tmp_path, state_path)
                print(f"  [state] epoch {epoch} step {step} -> {state_path.name}",
                      flush=True)

            if is_primary(rank) and step % args.log_every == 0:
                mae = float(beam_mae_proxy(prediction.detach().float(), target))
                rate = step / (time.time() - started)
                print(
                    f"epoch {epoch:3d} step {step:6d}/{total_steps}  "
                    f"loss={loss.detach().item():.5f}  beam_mae_boxframe~{mae:.5f}  "
                    f"lr={scheduler.get_last_lr()[0]:.2e}  {rate:.2f} it/s",
                    flush=True,
                )
                if run:
                    run.log(emit({"train/loss": float(loss),
                                  "train/beam_mae_boxframe": mae,
                                  "train/lr": scheduler.get_last_lr()[0],
                                  **{f"train/{k}": v for k, v in loss_parts.items()}}),
                            step=step)

            if stop["now"] or (args.max_steps and step >= args.max_steps):
                break
        if stop["now"] or (args.max_steps and step >= args.max_steps):
            # Validate before leaving, so an interrupted or capped run still
            # produces a checkpoint and a smoke test actually exercises saving.
            if is_primary(rank):
                best = validate_and_checkpoint(epoch, step, best)
            break

        if is_primary(rank):
            best = validate_and_checkpoint(epoch, step, best)

    if is_primary(rank):
        print(f"\nfinished: {step} steps in {(time.time() - started) / 60:.1f} min, "
              f"best val beam_mae_boxframe {best:.5f}", flush=True)
        (checkpoint_dir / f"{run_name}_summary.json").write_text(
            json.dumps(emit({"run": run_name, "steps": step,
                             "best_val_beam_mae_boxframe": best,
                             "config": vars(args)}), indent=1, default=str)
        )
    if run:
        run.finish()
    if distributed:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
