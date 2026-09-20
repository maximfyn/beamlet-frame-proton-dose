"""`conditional_bias_loss` must see coherent bias and NOT scatter.

**THE PROPERTY THIS PINS IS INVISIBLE TO ANY ORDINARY CHECK.** The term is a
dose-weighted mean residual, and a mean averages scatter down only as ``1/n``.
An earlier bin-shaped version therefore scored zero-mean noise exactly as highly
as a real bias -- on a 16x8x8 toy an sd-0.05 scatter and a +1% coherent offset
both scored 1.8e-4, a separation of **1.0x**. Nothing raises, the gradient is
finite, the loss goes down, and the term is measuring the wrong thing. ⇒ these
tests are what say it is not.
"""
import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]


def _train_module():
    spec = importlib.util.spec_from_file_location(
        "train_doserad", REPO / "scripts" / "train" / "train_doserad.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bragg_box(batch=4, d=192, u=32, v=24):
    """A beamlet-shaped target: sharp peak, broad plateau, near-empty halo.

    The SHAPE matters, not the realism -- band occupancy is what the test is
    about, and a uniform-random target would put every band at the same count and
    hide the defect this file exists for.
    """
    z = torch.linspace(0, 1, d).view(1, 1, d, 1, 1)
    bragg = torch.exp(-((z - 0.75) ** 2) / 0.002) + 0.25 * (z < 0.78)
    lat = (torch.exp(-(torch.linspace(-1, 1, u).view(1, 1, 1, u, 1) ** 2) / 0.08)
           * torch.exp(-(torch.linspace(-1, 1, v).view(1, 1, 1, 1, v) ** 2) / 0.08))
    t = (bragg * lat).repeat(batch, 1, 1, 1, 1).clamp_min(0)
    return t / t.amax(dim=(1, 2, 3, 4), keepdim=True)


def test_zero_on_a_perfect_prediction():
    tr = _train_module()
    t = _bragg_box()
    assert tr.conditional_bias_loss(t.clone(), t).item() == 0.0


def test_separates_coherent_bias_from_scatter():
    """The whole reason the term exists, and the thing equal band weights break."""
    tr = _train_module()
    torch.manual_seed(0)
    t = _bragg_box()
    scatter = tr.conditional_bias_loss(t + torch.randn_like(t) * 0.01, t).item()
    coherent = tr.conditional_bias_loss(t * 1.01, t).item()
    assert coherent > 20 * scatter, (
        f"separation only {coherent / max(scatter, 1e-30):.1f}x -- the term is "
        "reading noise, which is what an unweighted bin basis does")


def test_the_existing_loss_is_blind_to_what_this_one_sees():
    """The justification for adding a term at all: `dose_weighted_loss` ranks a
    1% coherent offset FAR below scatter of the same size, so no weighting of it
    reaches the defect `dvh_score` reads."""
    tr = _train_module()
    torch.manual_seed(0)
    t = _bragg_box()
    noisy, biased = t + torch.randn_like(t) * 0.01, t * 1.01
    mse_ratio = (tr.dose_weighted_loss(biased, t) / tr.dose_weighted_loss(noisy, t)).item()
    bias_ratio = (tr.conditional_bias_loss(biased, t)
                  / tr.conditional_bias_loss(noisy, t)).item()
    assert mse_ratio < 0.1 < 10 < bias_ratio


def test_gradient_is_finite_and_nonzero():
    tr = _train_module()
    t = _bragg_box()
    x = t.clone().requires_grad_()
    tr.conditional_bias_loss(x * 1.01, t).backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_off_by_default_reproduces_the_existing_objective():
    """Every arm trained before this term existed must be reproducible."""
    tr = _train_module()
    torch.manual_seed(0)
    t = _bragg_box(batch=2, d=64, u=16, v=12)
    p = t + torch.randn_like(t) * 0.01
    a, parts_a = tr.total_loss(p, t, idd_weight=0.05)
    b, parts_b = tr.total_loss(p, t, idd_weight=0.05, bias_weight=0.0)
    assert torch.equal(a, b) and "loss_cond_bias" not in parts_a and "loss_cond_bias" not in parts_b
