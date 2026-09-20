"""Loading a trained network into the inference path.

This is the seam between the two halves of the project: `submission/inference.py`
reads `CHECKPOINT_PATH` and hands it to `DosePredictor.from_checkpoint`. Until
this was implemented the call raised `NotImplementedError` for any path that
existed, so the container ran only with the stub and no trained model could
reach the leaderboard.

The assertion that matters most is the unit conversion. Training divides the
stored label by `dose_scale` (1e-3), so the network emits scaled units while
`render_to_volume` and the `minimum_cutoff` clamp both expect physical dose.
Dropping the multiply inflates every prediction ~1000x, which does not crash --
it silently produces dose that never falls below any cutoff, and the platform
counts those as implementation errors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from models.network import build_network
from models.predictor import DosePredictor, StubModel, TorchModel

# Small enough to stay fast; the network is shape-agnostic.
SHAPE = (32, 16, 8)
DOSE_SCALE = 1.0e-3


def write_checkpoint(path: Path, in_channels: int = 2, dose_scale: float = DOSE_SCALE):
    """A checkpoint in exactly the layout scripts/train/train_doserad.py saves."""
    net = build_network("unet", in_channels=in_channels, base_features=4, levels=2)
    torch.save(
        {
            "model": net.state_dict(),
            "config": {"base_features": 4, "levels": 2, "with_wepl": in_channels == 3},
            "in_channels": in_channels,
            "dose_scale": dose_scale,
            "epoch": 3,
            "step": 15984,
            "val_beam_mae": 0.0211,
        },
        path,
    )
    return net


def batch(n: int = 2, channels: int = 2) -> np.ndarray:
    rng = np.random.default_rng(0)
    ct = np.clip(rng.normal(0, 300, (n, *SHAPE)), -1000, 2000) / 1000.0
    energy = np.full_like(ct, 0.6)
    stack = [ct, energy] + ([np.abs(ct)] if channels == 3 else [])
    return np.stack(stack, axis=1).astype(np.float32)


# --------------------------------------------------------------------------
# Stub fallback -- the container must stay buildable before any checkpoint
# --------------------------------------------------------------------------


def test_none_path_yields_the_stub():
    assert isinstance(DosePredictor.from_checkpoint(None).model, StubModel)


def test_missing_path_raises_rather_than_silently_stubbing(tmp_path: Path):
    """A path that is given but missing means the weights were not baked into
    the image. Falling back to the stub would ship a container that scores as
    noise with nothing in the logs to explain it."""
    with pytest.raises(FileNotFoundError, match="mounted at"):
        DosePredictor.from_checkpoint(tmp_path / "absent.pt")


# --------------------------------------------------------------------------
# Real loading
# --------------------------------------------------------------------------


def test_checkpoint_loads_and_rebuilds_the_architecture(tmp_path: Path):
    path = tmp_path / "ckpt.pt"
    original = write_checkpoint(path)

    predictor = DosePredictor.from_checkpoint(path, device="cpu")
    assert isinstance(predictor.model, TorchModel)
    assert predictor.model.dose_scale == pytest.approx(DOSE_SCALE)

    loaded = dict(predictor.model.net.state_dict())
    for key, value in original.state_dict().items():
        assert torch.allclose(loaded[key], value), key


def test_architecture_comes_from_the_checkpoint_not_a_default(tmp_path: Path):
    """The container ships without a config file, so width/depth must be
    recoverable from the checkpoint alone."""
    path = tmp_path / "ckpt.pt"
    write_checkpoint(path)  # base_features=4, levels=2 -- not the 24/4 defaults

    net = DosePredictor.from_checkpoint(path, device="cpu").model.net
    assert net.levels == 2
    assert net.encoders[0].conv1.out_channels == 4


def test_output_is_physical_dose_not_scaled_units(tmp_path: Path):
    """Guards a ~1000x silent error: without the dose_scale multiply nothing
    crashes, but no voxel ever falls below its minimum_cutoff."""
    path = tmp_path / "ckpt.pt"
    write_checkpoint(path)
    model = DosePredictor.from_checkpoint(path, device="cpu").model

    x = batch()
    wrapped = model(x)

    with torch.inference_mode():
        raw = model.net(torch.from_numpy(x))[:, 0].numpy()

    assert np.allclose(wrapped, raw * DOSE_SCALE, rtol=1e-5, atol=1e-12)
    assert wrapped.max() < raw.max(), "dose_scale was not applied"


def test_output_magnitude_matches_the_stub_contract(tmp_path: Path):
    """Both models feed the same downstream clamp, so they must agree on units.
    Measured label peaks run 8.1e-4 to 1.74e-3."""
    path = tmp_path / "ckpt.pt"
    write_checkpoint(path)

    x = batch()
    trained = DosePredictor.from_checkpoint(path, device="cpu").model(x)
    stub = StubModel()(x)

    assert 0 < trained.max() < 1e-1, f"peak {trained.max():.3e} is not physical dose"
    # Within three orders of each other, i.e. not a unit mismatch.
    assert abs(np.log10(trained.max()) - np.log10(stub.max())) < 3


def test_output_shape_drops_the_channel_axis(tmp_path: Path):
    path = tmp_path / "ckpt.pt"
    write_checkpoint(path)
    out = DosePredictor.from_checkpoint(path, device="cpu").model(batch(n=3))
    assert out.shape == (3, *SHAPE)
    assert out.dtype == np.float32


def test_dose_scale_is_read_from_the_checkpoint(tmp_path: Path):
    """A checkpoint trained under a different scale must stay correct."""
    path = tmp_path / "ckpt.pt"
    write_checkpoint(path, dose_scale=5.0e-4)
    assert DosePredictor.from_checkpoint(path, device="cpu").model.dose_scale == \
        pytest.approx(5.0e-4)


# --------------------------------------------------------------------------
# WEPL
# --------------------------------------------------------------------------


def test_wepl_checkpoint_enables_the_third_channel(tmp_path: Path):
    """A --with-wepl model must be served with the channel it was trained on.
    Feeding it 2 channels would not raise -- it would silently emit wrong dose."""
    path = tmp_path / "wepl.pt"
    write_checkpoint(path, in_channels=3)

    predictor = DosePredictor.from_checkpoint(path, device="cpu")
    assert predictor.with_wepl is True
    assert predictor.model.net.encoders[0].conv1.in_channels == 3


def test_plain_checkpoint_leaves_wepl_off(tmp_path: Path):
    path = tmp_path / "plain.pt"
    write_checkpoint(path, in_channels=2)
    assert DosePredictor.from_checkpoint(path, device="cpu").with_wepl is False


def test_inconsistent_checkpoint_is_rejected(tmp_path: Path):
    """in_channels and config.with_wepl disagreeing means the checkpoint cannot
    be trusted to describe itself."""
    net = build_network("unet", in_channels=3, base_features=4, levels=2)
    path = tmp_path / "bad.pt"
    torch.save(
        {
            "model": net.state_dict(),
            "config": {"base_features": 4, "levels": 2, "with_wepl": False},
            "in_channels": 3,
            "dose_scale": DOSE_SCALE,
        },
        path,
    )
    with pytest.raises(ValueError, match="inconsistent"):
        DosePredictor.from_checkpoint(path, device="cpu")


def test_predict_feeds_the_model_three_channels_when_wepl_is_on():
    """Wiring check: the channel must actually reach the network, and be the
    cumulative sum the dataset builds -- not merely be requested."""
    from models.geometry import BeamletGrid, VolumeGeometry
    from reference_scipy import build_wepl_channel
    from models.predictor import BeamletRequest

    grid = BeamletGrid()
    seen: dict = {}

    def recording_model(batch: np.ndarray) -> np.ndarray:
        seen["batch"] = batch.copy()
        return np.zeros((batch.shape[0], *batch.shape[2:]), dtype=np.float32)

    # A small uniform-density block so the ray certainly enters the body.
    ct = np.full((80, 120, 120), 0.0, dtype=np.float32)
    ct[20:60, 30:90, 30:90] = 50.0
    geom = VolumeGeometry(
        origin=np.zeros(3), spacing=np.array([1.0, 1.0, 3.0]), shape=ct.shape
    )
    request = BeamletRequest(
        ray_source=(-200.0, 60.0, 120.0),
        ray_target=(200.0, 60.0, 120.0),
        energy=150.0,
        output_file_idx=0,
        idx_in_output=0,
        minimum_cutoff=0.0,
    )

    predictor = DosePredictor(model=recording_model, grid=grid, with_wepl=True)
    predictor.predict(ct, geom, [request])

    if "batch" not in seen:
        pytest.skip("synthetic ray missed the phantom; geometry-specific setup")

    batch = seen["batch"]
    assert batch.shape[1] == 3, "WEPL channel never reached the network"
    channel = batch[0, 2]
    assert np.all(np.diff(channel, axis=0) >= -1e-6), "not a cumulative sum"
    assert channel.max() <= build_wepl_channel(
        np.full(grid.shape, 2000.0), grid.depth_spacing
    ).max() + 1e-6


def test_from_checkpoint_rejects_a_box_it_was_not_trained_with(tmp_path):
    """A geometry mismatch must raise, not silently misplace the dose.

    This is the failure that cost every pre-2026-08-14 checkpoint: entry depth
    anchors the sampling box, so inferring under a different box moves every
    beamlet's dose without raising, without tripping the platform's
    implementation-error check, and without anything in the logs.
    """
    import torch

    from models.geometry import BeamletGrid
    from models.network import build_network
    from models.predictor import DosePredictor

    grid = BeamletGrid()
    net = build_network(in_channels=2, base_features=4, levels=2)
    path = tmp_path / "ckpt.pt"
    torch.save(
        {
            "model": net.state_dict(),
            "config": {"base_features": 4, "levels": 2, "with_wepl": False},
            "in_channels": 2,
            "dose_scale": 1e-3,
            "grid": {
                "shape": list(grid.shape),
                "entry_margin_mm": grid.entry_margin_mm,
                "anchor": "box",
            },
        },
        path,
    )

    # Same box: loads.
    DosePredictor.from_checkpoint(path, device="cpu")

    # A different entry margin is exactly the change that moves every anchor.
    with pytest.raises(ValueError, match="different sampling box"):
        DosePredictor.from_checkpoint(
            path,
            device="cpu",
            grid=BeamletGrid(entry_margin_mm=grid.entry_margin_mm + 8.0),
        )

    # And a different depth extent.
    with pytest.raises(ValueError, match="different sampling box"):
        DosePredictor.from_checkpoint(
            path, device="cpu", grid=BeamletGrid(n_depth=grid.n_depth - 64)
        )


# ---------------------------------------------------------------------------
# The minimum_cutoff clamp
# ---------------------------------------------------------------------------
#
# On 2026-08-14 the AWS harness defaulted this to 0.02, copied from an *example*
# in the submission instructions. A beamlet dose map peaks at ~1e-3, so 0.02 is
# ~13x its maximum and the clamp zeroed every voxel of every prediction. Nothing
# raised: the evaluator's check is `pred > 0 AND pred < cutoff`, which all-zero
# output passes while scoring near the maximum error. The organizers' own
# fixture generator uses 0.0 for normal beams and reserves `max * 2 + 1` to
# force a violation -- structurally what 0.02 was.
#
# So the clamp is exercised here, where it is free, rather than by configuring a
# run that costs GPU-hours and whose failure is invisible.


def _clamp_phantom():
    """A CT block and a ray that certainly crosses it."""
    from models.geometry import VolumeGeometry

    ct = np.full((80, 120, 120), 0.0, dtype=np.float32)
    ct[20:60, 30:90, 30:90] = 50.0
    geom = VolumeGeometry(
        origin=np.zeros(3), spacing=np.array([1.0, 1.0, 3.0]), shape=ct.shape
    )
    return ct, geom


def _predict_with_cutoff(cutoff: float) -> np.ndarray:
    from models.predictor import BeamletRequest

    ct, geom = _clamp_phantom()
    request = BeamletRequest(
        ray_source=(-200.0, 60.0, 120.0),
        ray_target=(200.0, 60.0, 120.0),
        energy=150.0,
        output_file_idx=0,
        idx_in_output=0,
        minimum_cutoff=cutoff,
    )
    predictor = DosePredictor(model=StubModel(), grid=None)
    return predictor.predict(ct, geom, [request])[0]


def test_a_cutoff_above_the_peak_annihilates_the_prediction():
    """The failure mode itself, pinned so it can never be silent again."""
    baseline = _predict_with_cutoff(0.0)
    peak = float(baseline.max())
    if peak <= 0:
        pytest.skip("synthetic ray produced no dose; geometry-specific setup")

    clamped = _predict_with_cutoff(peak * 2.0 + 1.0)
    assert not np.any(clamped), (
        "a cutoff above the beam's peak must zero the whole prediction -- if this "
        "ever passes silently in production, the submission scores as noise"
    )


def test_a_realistic_cutoff_keeps_the_beam_core():
    """A cutoff below the peak clamps the tail and nothing else."""
    baseline = _predict_with_cutoff(0.0)
    peak = float(baseline.max())
    if peak <= 0:
        pytest.skip("synthetic ray produced no dose; geometry-specific setup")

    cutoff = 0.01 * peak
    clamped = _predict_with_cutoff(cutoff)
    assert clamped.max() == pytest.approx(peak), "the core must survive"
    assert not np.any((clamped > 0) & (clamped < cutoff)), (
        "this is exactly the evaluator's implementation-error test: "
        "pred > 0 AND pred < cutoff"
    )


# --------------------------------------------------------------------------
# arch + channels -- a checkpoint has to say which network and which inputs
# --------------------------------------------------------------------------


def write_modern_checkpoint(path: Path, arch: str, channels: list[str]):
    """The layout the trainer saves once there is more than one architecture."""
    net = build_network(arch, in_channels=len(channels), base_features=4, levels=2)
    torch.save(
        {
            "model": net.state_dict(),
            "config": {
                "base_features": 4,
                "levels": 2,
                "arch": arch,
                "with_wepl": "wepl" in channels,
                "with_bragg": "bragg" in channels,
            },
            "in_channels": len(channels),
            "arch": arch,
            "channels": channels,
            "dose_scale": DOSE_SCALE,
            "epoch": 3,
            "step": 15984,
            "val_beam_mae_boxframe": 0.0101,
        },
        path,
    )
    return net


def test_a_factorised_checkpoint_rebuilds_a_factorised_network(tmp_path: Path):
    """Loading it into a U-Net would raise -- but only because the keys differ.

    Nothing guarantees that in general, so the architecture is read from the
    checkpoint rather than left to a default that happens to be wrong.
    """
    from models.network import FactorisedDoseNet

    path = tmp_path / "factorised.pt"
    write_modern_checkpoint(path, "factorised", ["ct", "energy"])
    net = DosePredictor.from_checkpoint(path, device="cpu").model.net
    assert isinstance(net, FactorisedDoseNet)


def test_a_checkpoint_without_arch_is_still_a_unet(tmp_path: Path):
    """Every checkpoint written before 2026-08-15 must keep loading."""
    from models.network import DoseUNet

    path = tmp_path / "legacy.pt"
    write_checkpoint(path)
    net = DosePredictor.from_checkpoint(path, device="cpu").model.net
    assert isinstance(net, DoseUNet)


def test_bragg_and_wepl_are_told_apart_at_three_channels(tmp_path: Path):
    """The reason `channels` exists at all.

    Both are 3-channel networks, and feeding one the other's channel does not
    raise -- it silently predicts wrong dose. The count cannot distinguish them;
    the list can.
    """
    wepl_path, bragg_path = tmp_path / "w.pt", tmp_path / "b.pt"
    write_modern_checkpoint(wepl_path, "unet", ["ct", "energy", "wepl"])
    write_modern_checkpoint(bragg_path, "unet", ["ct", "energy", "bragg"])

    wepl = DosePredictor.from_checkpoint(wepl_path, device="cpu")
    bragg = DosePredictor.from_checkpoint(bragg_path, device="cpu")

    assert (wepl.with_wepl, wepl.with_bragg) == (True, False)
    assert (bragg.with_wepl, bragg.with_bragg) == (False, True)


def test_a_channel_list_disagreeing_with_the_count_is_rejected(tmp_path: Path):
    path = tmp_path / "bad.pt"
    write_modern_checkpoint(path, "unet", ["ct", "energy", "wepl"])
    blob = torch.load(path, weights_only=False)
    blob["channels"] = ["ct", "energy"]
    torch.save(blob, path)
    with pytest.raises(ValueError, match="inconsistent"):
        DosePredictor.from_checkpoint(path, device="cpu")


def test_an_unbuildable_channel_is_refused_rather_than_guessed(tmp_path: Path):
    path = tmp_path / "bad.pt"
    write_modern_checkpoint(path, "unet", ["ct", "energy", "wepl"])
    blob = torch.load(path, weights_only=False)
    blob["channels"] = ["ct", "energy", "rsp"]
    torch.save(blob, path)
    with pytest.raises(NotImplementedError, match="channels"):
        DosePredictor.from_checkpoint(path, device="cpu")


def test_rsp_round_trips_through_the_checkpoint(tmp_path: Path):
    """A different stopping-power curve moves every predicted range, silently."""
    path = tmp_path / "hlut.pt"
    net = build_network("factorised", in_channels=3, base_features=4, levels=2)
    torch.save({"model": net.state_dict(),
                "config": {"base_features": 4, "levels": 2, "arch": "factorised",
                           "with_bragg": True, "rsp": "hlut"},
                "in_channels": 3, "arch": "factorised",
                "channels": ["ct", "energy", "bragg"], "rsp": "hlut",
                "dose_scale": DOSE_SCALE}, path)
    assert DosePredictor.from_checkpoint(path, device="cpu").rsp == "hlut"


def test_a_checkpoint_without_rsp_means_linear(tmp_path: Path):
    path = tmp_path / "legacy.pt"
    write_checkpoint(path)
    assert DosePredictor.from_checkpoint(path, device="cpu").rsp == "linear"


def test_lateral_must_be_the_last_channel(tmp_path: Path):
    """`condition_lateral` reads x[:, -1]; any other position conditions on junk."""
    path = tmp_path / "bad.pt"
    write_modern_checkpoint(path, "factorised", ["ct", "energy", "lateral", "bragg"])
    with pytest.raises(ValueError, match="must be LAST"):
        DosePredictor.from_checkpoint(path, device="cpu")


def test_disable_compilation_restores_the_exact_eager_module():
    """Abandoning the fast path must be exact, not approximate.

    `torch.compile` returns an ``OptimizedModule`` holding the original as
    ``_orig_mod`` — the same parameter objects, not a copy — so dropping the
    wrapper cannot change a single weight. `submission/inference.py` calls this
    when a compiled warmup fails, and a container that served slightly different
    numbers after falling back would be worse than one that failed loudly.
    """
    net = build_network(base_features=4, levels=2, in_channels=2)
    model = TorchModel(net, device="cpu", dose_scale=1.0)
    eager = model.net
    model.net = type("OptimizedModule", (), {"_orig_mod": eager})()
    model.compile_mode = "max-autotune"

    assert model.disable_compilation() is True
    assert model.net is eager
    assert model.compile_mode is None


def test_disable_compilation_says_no_when_nothing_was_compiled():
    """The caller branches on this, so "nothing to undo" must not read as done."""
    model = TorchModel(build_network(base_features=4, levels=2, in_channels=2),
                       device="cpu", dose_scale=1.0)
    assert model.disable_compilation() is False


def test_the_predictor_delegates_to_its_model_and_tolerates_a_stub():
    """The warmup holds a predictor; the stub has no network to un-compile."""
    assert DosePredictor(model=StubModel(), grid=None).disable_compilation() is False

    class Model:
        def disable_compilation(self):
            return True

    assert DosePredictor(model=Model(), grid=None).disable_compilation() is True

def test_the_box_record_carries_its_spacings(tmp_path):
    """A box is not its shape, and the guard used to compare only the shape.

    Until 2026-08-23 the checkpoint stored ``{shape, entry_margin_mm, anchor}``
    and `from_checkpoint` compared exactly that dict -- so ``(768, 128, 40)`` at
    0.5 mm depth and the same counts at 1.0 mm were indistinguishable. That is
    the fine-spacing box family a reshard would use, and getting it
    wrong halves the box's reach and reads every Bragg curve at twice its true
    depth, silently. The mismatch is invisible in the shape, which is why it
    needs a test rather than a reviewer.
    """
    import torch

    from models.geometry import BeamletGrid
    from models.network import build_network
    from models.predictor import DosePredictor

    fine = BeamletGrid(n_depth=768, depth_spacing=0.5, n_lat_u=128, n_lat_v=40)
    coarse = BeamletGrid(n_depth=768, depth_spacing=1.0, n_lat_u=128, n_lat_v=40)
    assert fine.shape == coarse.shape, "the point of the test is that these agree"

    net = build_network(in_channels=2, base_features=4, levels=2)
    path = tmp_path / "ckpt.pt"
    torch.save(
        {
            "model": net.state_dict(),
            "config": {"base_features": 4, "levels": 2, "with_wepl": False},
            "in_channels": 2,
            "dose_scale": 1e-3,
            "grid": fine.as_dict(),
        },
        path,
    )

    # Caller says nothing: the checkpoint's box wins, rather than the default.
    # This is what lets `submission/inference.py`
    # serve a resharded arm without being told the box.
    assert DosePredictor.from_checkpoint(path, device="cpu").grid == fine

    # Caller names the same box: fine. Caller names the same *shape* at another
    # spacing: refused, which the old dict comparison could not do.
    DosePredictor.from_checkpoint(path, device="cpu", grid=fine)
    with pytest.raises(ValueError, match="different sampling box"):
        DosePredictor.from_checkpoint(path, device="cpu", grid=coarse)


def test_a_checkpoint_from_before_spacings_were_recorded_still_loads(tmp_path):
    """Every box predating the field is the default lattice, so defaults fill it.

    `e120` and every other shipping checkpoint carries the three-key record. If
    reading one raised -- or worse, resolved to some other box -- the reshard
    would take the current candidate down with it.
    """
    import torch

    from models.geometry import BeamletGrid
    from models.network import build_network
    from models.predictor import DosePredictor

    net = build_network(in_channels=2, base_features=4, levels=2)
    path = tmp_path / "old.pt"
    torch.save(
        {
            "model": net.state_dict(),
            "config": {"base_features": 4, "levels": 2, "with_wepl": False},
            "in_channels": 2,
            "dose_scale": 1e-3,
            # Exactly what train_doserad.py wrote before 2026-08-23.
            "grid": {
                "shape": list(BeamletGrid().shape),
                "entry_margin_mm": BeamletGrid().entry_margin_mm,
                "anchor": "box",
            },
        },
        path,
    )
    assert DosePredictor.from_checkpoint(path, device="cpu").grid == BeamletGrid()
