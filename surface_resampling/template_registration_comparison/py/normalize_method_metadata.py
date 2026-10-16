#!/usr/bin/env python
"""Normalize method labels in existing surface-resampling outputs."""

from __future__ import annotations

import argparse
import io
import json
import os
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np


METHOD_BY_ROOT = {
    "o2_column_native_profiles": "O2-Column",
    "o3_coordinate_profiles": "O3-Coordinate",
}


def replace_npz_method(path: Path, method: str) -> bool:
    """Replace only the method metadata, preserving all numerical arrays."""
    with ZipFile(path, "r") as source:
        with source.open("method.npy") as handle:
            current = np.load(handle, allow_pickle=True).item()
    if current == method:
        return False

    replacement = io.BytesIO()
    np.save(replacement, np.asarray(method), allow_pickle=False)
    replacement_bytes = replacement.getvalue()

    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp.npz",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)

    try:
        with ZipFile(path, "r") as source, ZipFile(
            temporary, "w", compression=ZIP_DEFLATED
        ) as destination:
            for info in source.infolist():
                if info.filename == "method.npy":
                    data = replacement_bytes
                else:
                    data = source.read(info)
                destination.writestr(info, data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def normalize_root(root: Path, method: str) -> int:
    """Normalize NPZ and JSON metadata below one result root."""
    changed = 0
    npz_paths = sorted(
        path for path in root.rglob("*.npz") if ".tmp.npz" not in path.name
    )
    for npz_path in npz_paths:
        changed_npz = replace_npz_method(npz_path, method)
        qc_path = npz_path.with_suffix(".qc.json")
        if qc_path.exists():
            payload = json.loads(qc_path.read_text())
            if payload.get("method") != method:
                payload["method"] = method
                qc_path.write_text(json.dumps(payload, indent=2) + "\n")
                changed_npz = True
        if changed_npz:
            changed += 1
        print(f"PROGRESS {changed} file={npz_path}", flush=True)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    args = parser.parse_args()

    root_name = args.results_root.name
    try:
        method = METHOD_BY_ROOT[root_name]
    except KeyError as error:
        valid = ", ".join(sorted(METHOD_BY_ROOT))
        raise SystemExit(f"Unsupported result root {root_name!r}; use {valid}") from error

    count = normalize_root(args.results_root, method)
    print(f"Updated {count} NPZ metadata files as {method}")


if __name__ == "__main__":
    main()
