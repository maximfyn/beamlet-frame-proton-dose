#!/usr/bin/env python3
"""Print the `--build-arg` flags that label an image with its checkpoint.

The container records which weights it was built for as Docker labels, so that
`docker inspect` on a running image answers it. Those values live in the
checkpoint, and typing them by hand is how they go stale:

    docker build -f submission/Dockerfile -t doserad2026-task3 \\
        --build-arg GIT_SHA=$(git rev-parse --short HEAD) \\
        $(python scripts/checkpoint_build_args.py model/checkpoint.pt) .

`geometry_sha256` is provenance, not a gate: it hashes the bytes of
`models/geometry.py`, `geometry_torch.py` and `physics.py`, so a comment moves
it, and nothing reads it back (see `geometry_sha256` in the trainer).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

FIELDS = (("CHECKPOINT_STEP", "step"), ("CHECKPOINT_EPOCH", "epoch"),
          ("CHECKPOINT_GEOMETRY_SHA", "geometry_sha256"))


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"{path} not found", file=sys.stderr)
        return 1
    blob = torch.load(path, map_location="cpu", weights_only=True)
    config = blob.get("config") or {}
    args = []
    for flag, key in FIELDS:
        value = blob.get(key, config.get(key, ""))
        if value == "":
            print(f"{path.name} records no {key}", file=sys.stderr)
        args.append(f"--build-arg {flag}={value}")
    print(" ".join(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
