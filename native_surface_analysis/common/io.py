from __future__ import annotations

import gzip
from pathlib import Path
import numpy as np
import pandas as pd
from nibabel.freesurfer.io import read_annot, read_geometry
from .config import DEPTHS

def read_surface(subjects_dir: Path, subject: str, hemi: str, surface: str) -> np.ndarray:
    path = subjects_dir / subject / "surf" / f"{hemi}.{surface}"
    if not path.is_file(): raise FileNotFoundError(path)
    vertices, _ = read_geometry(str(path))
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError(f"Invalid {surface} surface: {path}")
    return vertices

def read_aparc(subject_dir: Path, hemi: str) -> tuple[np.ndarray, dict[int, str]]:
    labels, _, names = read_annot(str(subject_dir / "label" / f"{hemi}.aparc.annot"))
    decoded = [x.decode() if isinstance(x, bytes) else str(x) for x in names]
    return np.asarray(labels, dtype=np.int64), dict(enumerate(decoded))

def mapping_files(mapping_root: Path, subject: str, hemi: str) -> list[Path]:
    return sorted((mapping_root / subject / hemi / "regions").glob(
        f"{subject}_{hemi}_*_native_to_fsaverage.csv.gz"))

def mapping_frame(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8") as handle: frame = pd.read_csv(handle)
    required = {"subject_id","hemisphere","region","native_vertex_id","column_valid","template_vertex_id"}
    missing = required - set(frame.columns)
    if missing: raise ValueError(f"{path}: missing fields {sorted(missing)}")
    if frame["native_vertex_id"].duplicated().any(): raise ValueError(f"{path}: duplicate native IDs")
    return frame

def profile_path(profiles_root: Path, subject: str, hemi: str, region: str) -> Path:
    return profiles_root / subject / "profiles" / f"{subject}_{hemi}_{region}_md.csv"

def read_md_profile(path: Path) -> np.ndarray:
    if not path.is_file(): raise FileNotFoundError(path)
    if path.stat().st_size == 0: return np.empty((0, DEPTHS), dtype=float)
    values = np.asarray(np.loadtxt(path, delimiter=",", ndmin=2), dtype=float)
    if values.ndim != 2 or values.shape[1] != DEPTHS:
        raise ValueError(f"{path}: expected N x {DEPTHS}, found {values.shape}")
    return values
