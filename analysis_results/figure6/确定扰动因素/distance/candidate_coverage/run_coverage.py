#!/usr/bin/env python3
"""Reproduce the ico4 within/across candidate-coverage pilot."""

from pathlib import Path
import runpy


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "summarize_ico4_within_across_coverage.py"
)
runpy.run_path(str(SCRIPT), run_name="__main__")
