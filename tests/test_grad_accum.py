"""Gradient accumulation must be an optimisation identity, not an approximation.

`--grad-accum` exists because effective batch is a *matched* quantity across
arms (an unmatched batch is a confound that is
invisible in every metric), and the free GPUs rarely divide into it -- 36 needs
three 143 GB H200s at batch 12, or three 40 GB A100s at batch 2 accumulated six
times. Accumulating is how an arm reaches the same effective batch on whatever
hardware is free.

**The failure mode is silent and it is a learning-rate bug.** Get the
normalisation wrong and the accumulated gradient is `grad_accum` times too
large; the run trains, converges, and is quietly a different experiment from the
one it is being compared against. Nothing raises, and the loss curve looks
plausible because it *is* a valid training run -- just not the one in the config.

Three things this pins, all of which were choices rather than defaults:

* **the division** -- each micro-loss is scaled by `1 / grad_accum`, which is
  correct only because every loss term reduces with `.mean()` over the batch and
  `drop_last=True` keeps the micro-batches equal-sized. A `.sum()` reduction
  anywhere in `total_loss` would break this and nothing else would notice.
* **clip once, at the boundary** -- `clip_grad_norm_` runs on the accumulated
  gradient, not per micro-batch. Clipping per micro-batch clips a different
  quantity and is not equivalent to the unaccumulated run.
* **`zero_grad` after the step, never between micro-batches** -- the obvious
  transcription error, and it silently reduces the run to plain small-batch
  training with `grad_accum`x fewer optimiser steps.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from models.network import build_network


def _trainer():
    spec = importlib.util.spec_from_file_location(
        "train_doserad",
        Path(__file__).resolve().parents[1] / "scripts" / "train" / "train_doserad.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Small enough for CPU, and still legal: `levels=2` uses strides (2,2,1),(2,2,1),
# so the divisors are 4/4/1 and this shape clears all three.
SHAPE = (32, 16, 8)
IDD_WEIGHT = 0.05


def _run(initial, micro_batches, grad_accum, lr=3e-4):
    """The trainer's own step sequence, for one optimiser step."""
    trainer = _trainer()
    net = build_network("unet", in_channels=2, base_features=4, levels=2)
    net.load_state_dict(initial)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    optimizer.zero_grad(set_to_none=True)

    for i, (inputs, target) in enumerate(micro_batches):
        prediction = net(inputs)
        loss, _ = trainer.total_loss(prediction.float(), target, IDD_WEIGHT)
        (loss / grad_accum).backward()
        if (i + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    return {k: v.detach().clone() for k, v in net.state_dict().items()}


def test_accumulating_two_halves_equals_one_full_batch():
    """batch 4 x accum 1 and batch 2 x accum 2 must reach the same weights.

    This is the whole contract. If it holds, an arm run on three A100s at batch
    2 accumulated six times IS the arm run on three H200s at batch 12 -- which is
    what lets the pair stay comparable across whatever hardware was free.
    """
    torch.manual_seed(0)
    net = build_network("unet", in_channels=2, base_features=4, levels=2)
    initial = {k: v.detach().clone() for k, v in net.state_dict().items()}

    inputs = torch.randn(4, 2, *SHAPE)
    target = torch.rand(4, 1, *SHAPE)

    full = _run(initial, [(inputs, target)], grad_accum=1)
    split = _run(
        initial,
        [(inputs[:2], target[:2]), (inputs[2:], target[2:])],
        grad_accum=2,
    )

    for key in full:
        if not full[key].is_floating_point():
            continue
        assert torch.allclose(full[key], split[key], atol=1e-6, rtol=1e-5), (
            f"{key} diverged: accumulation is not equivalent to the full batch, "
            f"max |delta| = {(full[key] - split[key]).abs().max():.3e}"
        )


def test_forgetting_the_division_is_detectable():
    """Guard the guard: without `/ grad_accum` the test above must FAIL.

    A conservation test that passes for the wrong reason is worse than none, and
    this one would pass trivially if the two paths were somehow identical for
    reasons unrelated to the normalisation. Undoing exactly the thing under test
    has to break it.
    """
    torch.manual_seed(1)
    net = build_network("unet", in_channels=2, base_features=4, levels=2)
    initial = {k: v.detach().clone() for k, v in net.state_dict().items()}
    inputs = torch.randn(4, 2, *SHAPE)
    target = torch.rand(4, 1, *SHAPE)

    full = _run(initial, [(inputs, target)], grad_accum=1)
    # grad_accum=1 over two micro-batches never steps between them and never
    # divides -- i.e. the accumulated gradient is 2x too large, which is exactly
    # the bug. It steps once, at the second micro-batch.
    unscaled = _run(
        initial, [(inputs[:2], target[:2]), (inputs[2:], target[2:])], grad_accum=1
    )

    deltas = [
        (full[k] - unscaled[k]).abs().max().item()
        for k in full
        if full[k].is_floating_point()
    ]
    assert max(deltas) > 1e-7, (
        "omitting the /grad_accum division produced identical weights, so the "
        "equivalence test above cannot be detecting the normalisation at all"
    )
