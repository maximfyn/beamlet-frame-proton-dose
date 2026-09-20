"""No undefined names in the code that runs far from a keyboard.

A NameError inside a function is invisible until that line executes, and the
lines that execute latest here are the expensive ones: a training run that dies
at its first checkpoint, or a container that builds, starts, and fails on its
first job, having produced nothing.

This is not a style check and must not become one. Undefined names only;
pyflakes' other opinions are filtered out deliberately, so that a real finding
is never buried in noise nobody reads.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pyflakes_api = pytest.importorskip("pyflakes.api")
pyflakes_reporter = pytest.importorskip("pyflakes.reporter")

ROOT = Path(__file__).resolve().parent.parent

# The code that fails late: a long run, or shipped inside an image.
CHECKED = [
    "evaluation",
    "submission",
    "models",
    "scripts/data",
    "scripts/train",
    "examples",
]


class UndefinedOnly:
    """Collect only the messages that mean "this name does not exist"."""

    def __init__(self) -> None:
        self.found: list[str] = []

    def unexpectedError(self, filename, msg):        # noqa: N802
        self.found.append(f"{filename}: {msg}")

    def syntaxError(self, filename, msg, lineno, offset, text):  # noqa: N802
        self.found.append(f"{filename}:{lineno}: syntax error: {msg}")

    def flake(self, message) -> None:
        if type(message).__name__ in (
            "UndefinedName", "UndefinedLocal", "UndefinedExport"
        ):
            self.found.append(str(message))


@pytest.mark.parametrize("target", CHECKED)
def test_no_undefined_names(target):
    reporter = UndefinedOnly()
    directory = ROOT / target
    if not directory.is_dir():
        pytest.skip(f"{target} not present in this checkout")
    for path in sorted(directory.glob("*.py")):
        pyflakes_api.checkPath(str(path), reporter)
    assert not reporter.found, "\n".join(reporter.found)
