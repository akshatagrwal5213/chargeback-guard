"""Friendly failures for the ML extras.

Two things go wrong on other people's machines, and both produce tracebacks
that read like a bug in this project rather than a missing dependency:

  1. `make install` was run but not `make install-ml`, so lightgbm is absent.
  2. On macOS, the lightgbm wheel does not bundle the OpenMP runtime, so the
     import dies inside ctypes with a dlopen error about libomp.dylib.

Anyone reviewing this repo will hit one of them. Catch both and say what to do.
"""
from __future__ import annotations

import platform

MISSING = """
LightGBM is not installed.

    make install-ml
"""

NO_OPENMP = """
LightGBM is installed but cannot load its native library: the OpenMP runtime
is missing. The macOS wheel does not bundle it.

    brew install libomp

or, staying inside conda:

    conda install -c conda-forge lightgbm -y

Nothing else in this project needs OpenMP — the API, the tests and the
evidence layer all run without it.
"""


def require_lightgbm():
    """Import lightgbm, or exit with something a human can act on."""
    try:
        import lightgbm as lgb

        return lgb
    except ImportError:
        raise SystemExit(MISSING)
    except OSError as exc:
        if "libomp" in str(exc) or "OpenMP" in str(exc):
            raise SystemExit(NO_OPENMP)
        raise SystemExit(f"\nLightGBM failed to load:\n  {exc}\n")


def openmp_hint() -> str:
    """One-line hint for logs, where raising would be wrong."""
    if platform.system() == "Darwin":
        return "missing OpenMP runtime — run: brew install libomp"
    return "native library failed to load"
