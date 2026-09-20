"""Put ``tests/`` itself on the path so the scipy oracle is importable.

``reference_scipy`` is deliberately not part of the ``models`` package -- it is
the independent implementation the production one is checked against, and
keeping it out of the importable production tree is the point.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# `submission/inference.py` reads its settings from the environment with **no
# defaults**, so a build that drops an ENV line fails at import instead of
# silently falling back to a number written somewhere else. The test session is
# another deployment context and has to declare them too -- `setdefault`, so a
# test that wants to vary one still can.
os.environ.setdefault("ZLIB_LEVEL", "1")
os.environ.setdefault("BATCH_SIZE", "8")
os.environ.setdefault("BODY_MASK", "1")
os.environ.setdefault("SNAP_ALPHA", "0")
os.environ.setdefault("ZLIB_BLOCKS", "32")
