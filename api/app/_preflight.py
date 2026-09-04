"""Fail early and legibly on an unsupported interpreter.

Without this, Python 3.10 or older dies on `from enum import StrEnum` deep in
an import chain, which reads as a bug in this project rather than a wrong
virtualenv. Imported first thing by app/__init__.py.
"""
from __future__ import annotations

import sys

MINIMUM = (3, 11)


def check() -> None:
    if sys.version_info >= MINIMUM:
        return

    have = ".".join(str(n) for n in sys.version_info[:3])
    want = ".".join(str(n) for n in MINIMUM)
    raise SystemExit(
        f"\nchargeback-guard needs Python {want}+, but this interpreter is {have}\n"
        f"  {sys.executable}\n\n"
        "On macOS, `python3` outside conda is usually the system 3.9. Create the\n"
        "environment from a newer interpreter instead — any one of:\n\n"
        "  conda create -n cbg python=3.12 -y && conda activate cbg\n"
        "  /opt/anaconda3/bin/python3 -m venv .venv && source .venv/bin/activate\n"
        "  brew install python@3.12 && python3.12 -m venv .venv && source .venv/bin/activate\n\n"
        "Then:  python -m pip install --upgrade pip && make install\n"
    )


check()
