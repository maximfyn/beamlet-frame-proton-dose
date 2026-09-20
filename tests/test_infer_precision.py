"""The forward pass's dtype, and the four ways asking for one can lie.

`models.predictor.resolve_autocast_dtype` is the whole policy: everything else
is a `torch.autocast` context. It refuses rather than falls back, and each
refusal here corresponds to a measurement that would otherwise have been
reported as a result --

* **bf16 below compute 8.0** is emulated and 9.2x slower than fp16. A fallback would report the emulation as "bf16 is no
  faster", which is true of the number and false of the hardware.
* **autocast on CPU** silently computes in float32, so a proxy arm would
  report a precision change that never happened.
* An **unknown name** ("f16", "half") would otherwise resolve to fp32 by
  falling through, which is the same null with a typo behind it.

The device cases run without CUDA on purpose: the resolver takes ``torch`` as
an argument precisely so its policy can be tested on a laptop, and a test that
needs an A100 to run is a test that does not run.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from models.predictor import resolve_autocast_dtype

FP16, BF16 = "<fp16>", "<bf16>"


def fake_torch(available: bool = True, capability: tuple[int, int] = (8, 6)):
    """Just the surface the resolver touches."""
    return SimpleNamespace(
        float16=FP16,
        bfloat16=BF16,
        cuda=SimpleNamespace(
            is_available=lambda: available,
            get_device_capability=lambda: capability,
        ),
    )


@pytest.mark.parametrize("name", ["fp32", "float32", "none", "", None])
def test_fp32_spellings_mean_no_autocast(name):
    assert resolve_autocast_dtype(name, "cuda", fake_torch()) is None


@pytest.mark.parametrize(
    "name,expected", [("fp16", FP16), ("float16", FP16), ("bf16", BF16), ("bfloat16", BF16)]
)
def test_supported_dtypes_resolve(name, expected):
    assert resolve_autocast_dtype(name, "cuda", fake_torch()) == expected


@pytest.mark.parametrize("name", ["f16", "half", "int8", "fp8"])
def test_an_unknown_name_raises_rather_than_defaulting_to_fp32(name):
    with pytest.raises(ValueError, match="unknown inference precision"):
        resolve_autocast_dtype(name, "cuda", fake_torch())


def test_bf16_is_refused_below_compute_80():
    """The 9.2x trap: `is_bf16_supported()` says True on 7.5 and lies."""
    with pytest.raises(ValueError, match="emulated below compute 8.0"):
        resolve_autocast_dtype("bf16", "cuda", fake_torch(capability=(7, 5)))


def test_bf16_is_allowed_on_ampere():
    assert resolve_autocast_dtype("bf16", "cuda", fake_torch(capability=(8, 0))) == BF16


@pytest.mark.parametrize("device,available", [("cpu", True), ("cuda", False)])
def test_half_precision_without_a_cuda_device_raises(device, available):
    with pytest.raises(ValueError, match="needs a CUDA device"):
        resolve_autocast_dtype("fp16", device, fake_torch(available=available))


def test_fp32_is_fine_without_any_device():
    """The default must never need a GPU: the stub path and every laptop test."""
    assert resolve_autocast_dtype("fp32", "cpu", fake_torch(available=False)) is None
