"""The warmup runs before /health, so what it must not do is matter afterwards.

Every job on the platform is a fresh container and everything before /health is
untimed, so the first `/invoke` was paying for a cold CUDA context, cuDNN's
algorithm search and the allocator's first arena — on every job. `warm_up`
moves that into the untimed window.

Two properties make it safe rather than merely faster, and both are here:

* **Nothing synthetic survives it.** The predictor memoises entry depths, the
  uploaded CT and the render coordinates per image, all keyed on the fake
  geometry the warmup invents. A leak there would anchor a real beamlet's box
  against a phantom.
* **It cannot take the container down.** A warmup is an optimisation; failing it
  should cost the seconds it would have saved and nothing more. The alternative
  is a container that refuses to start over a call whose output is discarded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "submission"))

from submission import inference  # noqa: E402


class RecordingPredictor:
    batch_size = 8

    def __init__(self) -> None:
        self.calls: list[int] = []
        self.resets = 0

    def predict(self, ct, geom, requests):
        self.calls.append(len(requests))
        return [np.zeros(ct.shape, dtype=np.float32) for _ in requests]

    def reset_entry_cache(self) -> None:
        self.resets += 1


def test_it_warms_the_full_batch_a_short_one_and_a_single():
    """One shape is not enough, and the missing ones are the expensive ones.

    A 500-beamlet job is 62 full batches and one short batch. Everything that
    specialises per shape — cuDNN's autotuner, `torch.compile`'s graphs — meets
    that short batch for the first time *inside the timed window*, at a measured
    100.3 ms/beamlet against a 27.8 steady state. Warming
    only the full batch leaves exactly the cost the warmup exists to remove.

    **And a batch of ONE is a third shape, not a case of the second**: a
    dynamic batch dimension covers every size except 0 and 1, which PyTorch
    specialises unconditionally. Measured on the A10G, the jobs holding a
    depth-1 stack paid **~8 s of `predict`** and the rest paid nothing — and a small job
    cannot avoid that shape.
    """
    predictor = RecordingPredictor()
    inference.warm_up(predictor)
    assert predictor.calls == [RecordingPredictor.batch_size,
                               RecordingPredictor.batch_size // 2,
                               1]


def test_a_batch_of_two_does_not_warm_a_single_twice():
    """`batch // 2` already IS one, so a third call would repeat it."""
    class Two(RecordingPredictor):
        batch_size = 2

    predictor = Two()
    inference.warm_up(predictor)
    assert predictor.calls == [2, 1]


def test_a_batch_of_one_warms_once_and_does_not_repeat_itself():
    """There is no short batch to warm when the batch is already one."""
    class Single(RecordingPredictor):
        batch_size = 1

    predictor = Single()
    inference.warm_up(predictor)
    assert predictor.calls == [1]


class CompiledPredictor(RecordingPredictor):
    """Fails while compiled, works once compilation is abandoned.

    Stands in for the real failure mode: Inductor writes a kernel to the scratch
    mount and cannot execute it, which surfaces on the first forward — inside
    the warmup if we are lucky, inside /invoke if we are not.
    """

    def __init__(self, fail_before_predict: bool = False) -> None:
        super().__init__()
        self.compiled = True
        self.disabled = 0
        self.fail_before_predict = fail_before_predict

    def predict(self, ct, geom, requests):
        if self.compiled:
            raise RuntimeError("ImportError: failed to map segment from shared object")
        return super().predict(ct, geom, requests)

    def disable_compilation(self) -> bool:
        self.disabled += 1
        was, self.compiled = self.compiled, False
        return was


def test_a_compiled_warmup_that_fails_abandons_compilation_and_serves_eager(capsys):
    """The container must not carry a compile failure past /health.

    Swallowing the exception is right for an eager warmup and wrong for a
    compiled one: the compiled module stays installed and the first real job
    meets the same failure inside the timed window, where the platform sees a
    job return 0 instead of 201. That is how the first `torch.compile` run died, and the final phase returns no logs to diagnose it with.
    """
    predictor = CompiledPredictor()
    inference.warm_up(predictor)

    assert predictor.disabled == 1
    assert not predictor.compiled
    # And it re-warmed eagerly, so /health is not reached on a cold path either.
    assert predictor.calls == [RecordingPredictor.batch_size]
    assert "COMPILATION ABANDONED" in capsys.readouterr().out


def test_an_eager_warmup_that_fails_is_still_only_a_warning(capsys):
    """A warmup is an optimisation and may cost only itself."""
    class Failing(RecordingPredictor):
        def predict(self, ct, geom, requests):
            raise RuntimeError("no GPU today")

    inference.warm_up(Failing())
    out = capsys.readouterr().out
    assert "skipped, ignoring" in out and "ABANDONED" not in out


def test_the_fallback_survives_a_failure_before_the_batch_exists(monkeypatch, capsys):
    """The handler refers to names bound inside the `try` it is catching.

    If the warmup dies before those exist — a geometry constructor, an import —
    the retry would raise NameError *from the exception handler*, and a
    diagnostic may never be the reason a submission errors.
    """
    monkeypatch.setattr(inference, "VolumeGeometry",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("early")))
    predictor = CompiledPredictor()
    inference.warm_up(predictor)

    assert predictor.disabled == 1
    assert "eager warmup also failed" in capsys.readouterr().out


def test_nothing_synthetic_survives_into_the_first_real_image():
    predictor = RecordingPredictor()
    inference.warm_up(predictor)
    assert predictor.resets == 1


def test_a_failing_warmup_does_not_stop_the_container(capsys):
    """The output is discarded; refusing to start over it would not be."""
    class Broken(RecordingPredictor):
        def predict(self, ct, geom, requests):
            raise RuntimeError("no GPU today")

    predictor = Broken()
    inference.warm_up(predictor)          # must not raise
    assert "skipped" in capsys.readouterr().out


def test_the_cache_is_still_cleared_when_the_warmup_fails():
    """The failure could be halfway through, so the state still needs dropping."""
    class HalfBroken(RecordingPredictor):
        def predict(self, ct, geom, requests):
            self.calls.append(len(requests))
            raise RuntimeError("died after touching state")

    predictor = HalfBroken()
    inference.warm_up(predictor)
    assert predictor.resets == 1


def test_a_predictor_without_a_batch_size_still_warms():
    """The stub path has no batch size and must not crash init."""
    class NoBatch:
        def predict(self, ct, geom, requests):
            assert len(requests) >= 1
            return [np.zeros(ct.shape, dtype=np.float32) for _ in requests]

    inference.warm_up(NoBatch())          # must not raise
