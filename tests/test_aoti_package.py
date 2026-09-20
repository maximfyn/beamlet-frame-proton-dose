"""A precompiled forward may be refused, but it may never be wrong.

`models/aoti.py` moves compilation to build time, which removes the three costs
`torch.compile` carries (per-job compile before `/health`, a write-then-execute
on the scratch, and a batch of exactly 1 that recompiles inside the timed
window). What it adds is a *file containing weights* — so a package served
beside the wrong checkpoint loads, runs, and predicts the wrong dose without
raising. That is the failure class of `models/geometry.py`'s box mismatch, and
it is guarded the same way: the package records what produced it, and anything
that disagrees is refused in favour of eager.

**And a load is not a run.** A bare `.so` `dlopen`s cleanly while its CUDA
kernels are still unresolved, so "it loaded" says nothing about whether the
first forward works -- which is precisely how 2026-08-26 was spent. The loader
therefore calls the thing before returning it, and these tests pin that a
failure there is eager rather than an exception out of the model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.aoti import load_package  # noqa: E402


# What the export records about the tensor the package was built for. Without
# it a package cannot be smoke-tested, and `load_package` refuses it.
SPEC = {"shape": [8, 3, 32, 32, 32], "stride": [98304, 32768, 1024, 32, 1],
        "dtype": "torch.float32", "device": "cuda:0", "min_batch": 2}


def written(tmp_path, recorded: dict) -> Path:
    path = tmp_path / "net.so"
    path.write_bytes(b"not a real package")
    Path(str(path) + ".json").write_text(json.dumps(dict(recorded, input=SPEC)))
    return path


IDENTITY = {"checkpoint_step": "92232", "geometry_sha": "7a8f", "torch": "2.9.1"}


def test_a_package_from_another_checkpoint_is_refused(tmp_path, capsys):
    """The case that would otherwise be silent and wrong."""
    stale = dict(IDENTITY, checkpoint_step="189405")
    assert load_package(written(tmp_path, stale), IDENTITY) is None
    out = capsys.readouterr().out
    assert "REFUSING" in out and "checkpoint_step" in out


def test_a_package_for_another_gpu_is_refused(tmp_path, capsys):
    """Kernels are per architecture; sm_86's are not sm_90's."""
    identity = dict(IDENTITY, gpu_arch="sm_86")
    recorded = dict(identity, gpu_arch="sm_90")
    assert load_package(written(tmp_path, recorded), identity) is None
    assert "REFUSING" in capsys.readouterr().out


def test_a_package_from_another_torch_is_refused(tmp_path):
    assert load_package(written(tmp_path, dict(IDENTITY, torch="2.8.0")), IDENTITY) is None


def test_a_missing_package_is_not_an_error(tmp_path):
    """No package is the normal case: the container then compiles, or stays eager."""
    assert load_package(tmp_path / "absent.pt2", IDENTITY) is None


def test_a_package_without_its_identity_is_refused(tmp_path, capsys):
    """An unlabelled package cannot be checked, so it cannot be trusted."""
    path = tmp_path / "net.so"
    path.write_bytes(b"not a real package")
    assert load_package(path, IDENTITY) is None
    assert "unreadable" in capsys.readouterr().out


def test_a_package_recording_no_input_shape_is_refused(tmp_path, capsys):
    """An object that cannot be smoke-tested is not shipped on trust.

    It is also how an image built before the spec existed announces itself,
    instead of loading and failing inside the timed window.
    """
    path = tmp_path / "net.so"
    path.write_bytes(b"not a real package")
    Path(str(path) + ".json").write_text(json.dumps(IDENTITY))
    assert load_package(path, IDENTITY) is None
    assert "no input shape" in capsys.readouterr().out


def test_a_matching_package_is_attempted(tmp_path, capsys):
    """And a load failure is still eager, never an exception out of the model."""
    assert load_package(written(tmp_path, IDENTITY), IDENTITY) is None
    assert "load failed" in capsys.readouterr().out


def loads_as(monkeypatch, forward):
    """Stand in for `aot_load`, which needs a real object and a real GPU."""
    import torch._export as export_api

    monkeypatch.setattr(export_api, "aot_load", lambda path, device: forward)


def test_a_package_that_loads_but_cannot_run_is_refused(monkeypatch, tmp_path, capsys):
    """THE 2026-08-26 FAILURE, pinned.

    A bare `.so` whose CUDA kernels are not embedded `dlopen`s perfectly and
    dies on the first forward -- inside `/invoke`, if nothing calls it earlier.
    Here it costs the speedup and a log line instead.
    """
    def explode(_x):
        raise RuntimeError("run_func_(...) API call failed at "
                           ".../model_container_runner.cpp:145")

    loads_as(monkeypatch, explode)
    assert load_package(written(tmp_path, IDENTITY), IDENTITY, device="cpu") is None
    out = capsys.readouterr().out
    assert "will not RUN" in out and "stderr" in out


def test_a_working_package_is_run_at_both_ends_of_the_range(monkeypatch, tmp_path,
                                                            capsys):
    """**Both ends means the exported batch and the FLOOR, not batch 1.**
    `torch.export` specialises 0 and 1, so the artefact's range starts at 2 and
    probing 1 would fail a package that is perfectly good -- which is what
    2026-08-26 measured on the A10G."""
    seen = []

    def record(x):
        seen.append(tuple(x.shape))
        return x

    loads_as(monkeypatch, record)
    loaded = load_package(written(tmp_path, IDENTITY), IDENTITY, device="cpu")
    assert loaded is record
    assert seen == [(8, 3, 32, 32, 32), (2, 3, 32, 32, 32)]
    out = capsys.readouterr().out
    assert "ran it at batch 8/2" in out and "under 2 stays eager" in out


def test_the_floor_travels_with_the_package(monkeypatch, tmp_path):
    """The caller cannot route what it cannot see, and a hardcoded 2 would drift
    from whatever the next torch specialises."""
    loads_as(monkeypatch, lambda x: x)
    loaded = load_package(written(tmp_path, IDENTITY), IDENTITY, device="cpu")
    assert loaded.min_batch == 2


def test_the_no_grad_retry_asks_about_the_size_that_failed(monkeypatch, tmp_path,
                                                           capsys):
    """Retrying at a size that works reports `inference_mode` as the cause of
    a failure that was about the batch — which is exactly what it did once."""
    def only_big(x):
        if x.shape[0] < 8:
            raise RuntimeError("dim value is too small at 0")
        return x

    loads_as(monkeypatch, only_big)
    assert load_package(written(tmp_path, IDENTITY), IDENTITY, device="cpu") is None
    out = capsys.readouterr().out
    assert "will not RUN at batch 2" in out and "no_grad" not in out
