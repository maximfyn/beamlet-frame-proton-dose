"""Training set over the preprocessed beamlet shards.

Reads what `scripts/data/preprocess_beamlets.py` writes: one memory-mappable
`ct_i16.npy` / `label_f16.npy` pair per patient plus a `meta.json` sidecar.
Shards are opened lazily per worker and left as memmaps, so with the dataset in
page cache the loader costs little more than a copy.

**Inputs are assembled by `models.geometry_torch.build_network_input`, the same call
the inference path makes.** Training and inference therefore cannot drift in how
a beamlet is presented to the network, which two separate implementations of
exactly this would permit.

Normalization
-------------
Labels are scaled by a single global constant rather than per-sample peak.
Measured over 31,968 beamlets, the pre-compensated peak distribution is tight
(p1 8.1e-4, median 1.26e-3, p99 1.74e-3), so a global scale keeps every sample
O(1) while preserving the *relative* magnitudes between beamlets -- which the
plan-level metrics depend on, since Level 2 sums beamlets with clinical weights.
Per-sample normalization would discard exactly that information and force the
network to regress a separate scalar peak.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import geometry_torch as GT
from .geometry import BeamletGrid

# Chosen so typical peaks land near 1.0 (median label peak is 1.26e-3).
DOSE_SCALE = 1.0e-3


class BeamletDataset(Dataset):
    """Beamlet crops for one split.

    Parameters
    ----------
    root:
        Directory holding ``<PID>/`` shards.
    patients:
        Patient ids to include. Shards without a ``_COMPLETE`` marker are
        skipped with a warning rather than silently yielding partial data.
    with_wepl:
        Append a water-equivalent-path-length channel. Because the cuboid's
        first axis *is* the beam direction, this is a cumulative sum along that
        axis -- exact, and far easier for the network than inferring range from
        a limited receptive field. Costs nothing to store since it is derived.
    with_bragg:
        Append the Generic-machine Bragg prior (`models.physics`). Also derived,
        so also free to store; it is WEPL pushed one step further, through the
        depth-dose curve the dataset's own beam parameters came from.

    **Channel order is ``[ct, energy, wepl?, bragg?]`` and it is a contract.**
    The count alone stopped identifying the channels once there were two
    optional ones -- 3 channels can mean either, and feeding a Bragg-trained
    network a WEPL channel does not raise. The checkpoint records the list;
    ``models/predictor.py`` refuses a mismatch.
    """

    def __init__(
        self,
        root: str | Path,
        patients: list[str],
        grid: BeamletGrid | None = None,
        with_wepl: bool = False,
        with_bragg: bool = False,
        with_lateral: bool = False,
        rsp: str = "linear",
        dose_scale: float = DOSE_SCALE,
        augment_flip: bool = False,
        with_geometry: bool = False,
    ) -> None:
        self.root = Path(root)
        self.grid = grid or BeamletGrid()
        self.with_wepl = with_wepl
        self.with_bragg = with_bragg
        self.with_lateral = with_lateral
        # The HU conversion behind the WEPL the Bragg prior is read at: the
        # dataset's HU-to-density table, or the linear approximation.
        # Recorded in the checkpoint because inference must use the same one:
        # a different curve moves every predicted range and nothing raises.
        self.rsp = rsp
        self.dose_scale = dose_scale
        # TRAIN ONLY. A validation set that augments is a validation set whose
        # number moves for a reason unrelated to the model, and `best` selects on
        # it. `train_doserad.py` passes this to the train split and never to val.
        self.augment_flip = augment_flip
        # Per-beamlet ray geometry plus the patient's CT volume geometry -- what
        # `geometry_torch.render_block` needs to map a box back onto the CT
        # lattice. It costs 7 floats and a shape per sample and comes from the
        # SAME `meta.json` the energies do, so a shard that can feed the network
        # can always feed this too.
        self.with_geometry = with_geometry
        # **THE FLIP AND THE RENDER ARE MUTUALLY EXCLUSIVE, AND SILENTLY SO.**
        # `augment_flip` mirrors the cuboid about the beam axis; the ray
        # coordinates it would then be rendered through are NOT mirrored, so the
        # block lands in the wrong half of the patient while every shape still
        # matches and nothing raises. => refuse rather than document.
        if augment_flip and with_geometry:
            raise ValueError(
                "augment_flip=True with with_geometry=True: the flip mirrors the "
                "box but not the ray, so anything rendered through that geometry "
                "is wrong and no shape check would catch it. Turn one off."
            )

        self.shards: list[str] = []
        self.index: list[tuple[int, int]] = []
        self.energies: list[np.ndarray] = []
        self.rays: list[np.ndarray] = []
        self.volumes: list[tuple] = []
        skipped = []

        for pid in patients:
            shard = self.root / pid
            if not (shard / "_COMPLETE").exists():
                skipped.append(pid)
                continue
            # The arrays are read raw and reshaped by the box we were given,
            # so a shard built under a different one does not raise -- it feeds
            # the network a cuboid whose axes mean something else. The spacings
            # are the silent half: `with_bragg` reads the depth-dose curve at
            # `grid.depth_spacing`, so 0.5 mm shards under the 1.0 mm default
            # put every Bragg peak at twice its true depth, and that prior does
            # the heaviest lifting of any input channel.
            stored = json.loads((shard / "_COMPLETE").read_text()).get("grid")
            if stored and BeamletGrid.from_dict(stored) != self.grid:
                raise ValueError(
                    f"shard {pid} was built with a different sampling box than "
                    f"this dataset reads: shard {stored} vs dataset "
                    f"{self.grid.as_dict()}."
                )
            meta = json.loads((shard / "meta.json").read_text())
            count = int(meta["count"])
            shard_id = len(self.shards)
            self.shards.append(pid)
            self.energies.append(
                np.array([b["energy"] for b in meta["beamlets"]], dtype=np.float32)
            )
            if self.with_geometry:
                bl = meta["beamlets"]
                self.rays.append(np.array(
                    [list(b["ray_source"]) + list(b["ray_target"])
                     + [b["entry_depth"]] for b in bl], dtype=np.float64))
                vol = meta["volume"]
                self.volumes.append((
                    np.asarray(vol["origin"], dtype=np.float64),
                    np.asarray(vol["spacing"], dtype=np.float64),
                    tuple(int(v) for v in vol["shape"]),
                ))
            self.index.extend((shard_id, row) for row in range(count))

        if skipped:
            print(
                f"[BeamletDataset] skipping {len(skipped)} shard(s) without "
                f"_COMPLETE: {skipped[:5]}{'...' if len(skipped) > 5 else ''}"
            )
        if not self.index:
            raise RuntimeError(f"no complete shards for {len(patients)} patients under {self.root}")

        # Opened lazily so each dataloader worker gets its own handles.
        self._ct: dict[int, np.ndarray] = {}
        self._label: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.index)

    @property
    def channel_names(self) -> list[str]:
        return (["ct", "energy"]
                + (["wepl"] if self.with_wepl else [])
                + (["bragg"] if self.with_bragg else [])
                # Last by contract: FactorisedDoseNet(condition_lateral=True)
                # reads x[:, -1] as the analytic kernel.
                + (["lateral"] if self.with_lateral else []))

    @property
    def n_channels(self) -> int:
        return len(self.channel_names)

    def _arrays(self, shard_id: int) -> tuple[np.ndarray, np.ndarray]:
        if shard_id not in self._ct:
            shard = self.root / self.shards[shard_id]
            self._ct[shard_id] = np.load(shard / "ct_i16.npy", mmap_mode="r")
            self._label[shard_id] = np.load(shard / "label_f16.npy", mmap_mode="r")
        return self._ct[shard_id], self._label[shard_id]

    def __getitem__(self, item: int) -> dict:
        shard_id, row = self.index[item]
        ct_arr, label_arr = self._arrays(shard_id)

        ct_cuboid = torch.from_numpy(np.asarray(ct_arr[row], dtype=np.float32))
        energy = float(self.energies[shard_id][row])
        target_arr = np.asarray(label_arr[row], dtype=np.float32)

        # **FLIP THE CT CUBOID, NOT THE CHANNELS.** Every channel below is a
        # deterministic function of this cuboid, so flipping it first makes the
        # whole stack equivariant by CONSTRUCTION rather than by an argument that
        # has to hold for each builder separately -- and one of them,
        # `build_lateral_prior`, reads the lateral axes explicitly.
        # **The flip is EXACT, not an approximation.** `models/geometry.py`
        # places the lateral samples at `(arange(n) - (n-1)/2) * spacing`, exactly
        # symmetric about the beam axis, so reversing an axis maps each sample
        # onto the one at minus its own offset. No interpolation, no half-voxel
        # drift. `precompensate` is linear up to a pointwise clamp at zero, and
        # both commute with the flip, so the label transforms the same way.
        # Axis 0 is DEPTH and is never flipped -- protons travel one way.
        if self.augment_flip:
            flip = int(torch.randint(0, 4, (1,)).item())
            axes = [a for a, bit in ((1, 1), (2, 2)) if flip & bit]
            if axes:
                ct_cuboid = torch.flip(ct_cuboid, dims=axes)
                # `np.flip` returns a negative-stride view and `torch.from_numpy`
                # refuses those, so copy rather than hand on the view.
                # SAME axis indices as the cuboid: both `ct_arr[row]` and
                # `label_arr[row]` are `(depth, u, v)`. The channel dimension is
                # added later, to `inputs` by the builders and to `target` by the
                # `[None]` below, so it is not present here for either.
                target_arr = np.ascontiguousarray(np.flip(target_arr, axis=axes))

        # The *same function* inference calls, not merely the same formula.
        inputs = GT.build_network_input(ct_cuboid, energy)

        if self.with_wepl:
            wepl = GT.build_wepl_channel(ct_cuboid, self.grid.depth_spacing)
            inputs = torch.cat([inputs, wepl[None]], dim=0)

        if self.with_bragg:
            from . import physics

            bragg = physics.build_bragg_channel(
                ct_cuboid, energy, self.grid.depth_spacing, rsp=self.rsp)
            inputs = torch.cat([inputs, bragg[None]], dim=0)

        if self.with_lateral:
            from . import physics

            lateral = physics.build_lateral_prior(
                ct_cuboid, energy, self.grid.depth_spacing, rsp=self.rsp)
            inputs = torch.cat([inputs, lateral[None]], dim=0)

        target = target_arr / self.dose_scale

        out = {
            "inputs": inputs.contiguous(),
            "target": torch.from_numpy(target)[None],
            "energy": energy,
            "patient": self.shards[shard_id],
            "row": row,
        }
        if self.with_geometry:
            # float64 throughout: `render_coordinates` works in world mm along a
            # ~1000 mm ray while the box samples sit 0.5-3 mm apart, so float32
            # here would quantise the sample POSITIONS rather than the dose.
            r = self.rays[shard_id][row]
            origin, spacing, shape = self.volumes[shard_id]
            out["ray"] = torch.from_numpy(np.ascontiguousarray(r))
            out["vol_origin"] = torch.from_numpy(origin)
            out["vol_spacing"] = torch.from_numpy(spacing)
            out["vol_shape"] = torch.tensor(shape, dtype=torch.int64)
        return out


def load_split(
    root: str | Path,
    split: str,
    **kwargs,
) -> BeamletDataset:
    """Convenience constructor using the fixed registry in `models.splits`."""
    from models.splits import get_splits

    return BeamletDataset(root, get_splits()[split], **kwargs)
