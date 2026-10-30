#!/usr/bin/env python3
"""Build Figure 2A/B method-comparison tables for fsaverage profiles."""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from nibabel.freesurfer.io import read_annot, read_label
from scipy.stats import spearmanr, wilcoxon


CANONICAL_DEPTHS = np.linspace(0.0, 1.0, 21)
METHOD_CHOICES = ("O2-Column", "O2-FS", "O3")
EXCLUDED_ANNOTATION_NAMES = {
    "unknown",
    "medial_wall",
    "corpuscallosum",
    "background",
    "???",
}


@dataclass
class ProfileData:
    """One method's profiles aligned to the full template vertex index."""

    target_vertex_id: np.ndarray
    profiles: np.ndarray
    declared_valid: np.ndarray
    finite_profile: np.ndarray
    depths: np.ndarray
    depth_direction: str


@dataclass
class PairData:
    """Two methods and their common vertex-level masks."""

    method_a: ProfileData
    method_b: ProfileData
    pairwise_valid: np.ndarray
    cortex_mask: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-a", choices=METHOD_CHOICES, required=True)
    parser.add_argument("--method-b", choices=METHOD_CHOICES, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--freesurfer-subjects-dir", type=Path, required=True)
    parser.add_argument("--subjects-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--template", default="fsaverage")
    parser.add_argument(
        "--run-both",
        action="store_true",
        help=(
            "Also run the predefined O2-Column vs O3 and O2-FS vs O3 "
            "comparisons with shared curvature bins."
        ),
    )
    return parser.parse_args()


def read_subjects(path: Path) -> list[str]:
    subjects = []
    for line in path.read_text().splitlines():
        subject = line.strip()
        if subject and not subject.startswith("#"):
            subjects.append(subject)
    return subjects


def scalar_value(array: np.ndarray) -> object:
    value = array.item()
    if isinstance(value, bytes):
        return value.decode()
    return value


def decode_annotation_name(name: bytes | str) -> str:
    if isinstance(name, bytes):
        return name.decode(errors="replace")
    return str(name)


def load_cortex_mask(
    subjects_dir: Path,
    template: str,
    hemisphere: str,
) -> np.ndarray:
    """Return fsaverage cortex intersected with valid DK annotation labels."""

    label_path = subjects_dir / template / "label" / f"{hemisphere}.cortex.label"
    annot_path = subjects_dir / template / "label" / f"{hemisphere}.aparc.annot"
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing cortex label: {label_path}")
    if not annot_path.is_file():
        raise FileNotFoundError(f"Missing DK annotation: {annot_path}")

    cortex_vertices = read_label(str(label_path), read_scalars=False).astype(int)
    annotation_ids, _, names = read_annot(str(annot_path))
    mask = np.zeros(annotation_ids.size, dtype=bool)
    mask[cortex_vertices] = True

    valid_annotation = annotation_ids >= 0
    annotation_names = {
        index: decode_annotation_name(name).lower()
        for index, name in enumerate(names)
    }
    for index, name in annotation_names.items():
        if name in EXCLUDED_ANNOTATION_NAMES:
            valid_annotation &= annotation_ids != index

    return mask & valid_annotation


def ensure_target_ids(
    target_vertex_id: np.ndarray,
    profiles: np.ndarray,
    declared_valid: np.ndarray,
    n_vertices: int,
) -> ProfileData:
    target_vertex_id = np.asarray(target_vertex_id, dtype=np.int64)
    profiles = np.asarray(profiles, dtype=float)
    declared_valid = np.asarray(declared_valid, dtype=bool)

    if profiles.ndim != 2 or profiles.shape[1] != 21:
        raise ValueError(f"Expected n_vertices x 21 profiles, found {profiles.shape}")
    if target_vertex_id.size != profiles.shape[0]:
        raise ValueError("target_vertex_id does not match profile rows")
    if declared_valid.size != target_vertex_id.size:
        raise ValueError("Validity mask does not match profile rows")
    if np.any(target_vertex_id < 0) or np.any(target_vertex_id >= n_vertices):
        raise ValueError("Target vertex IDs are outside the template vertex range")
    if np.unique(target_vertex_id).size != target_vertex_id.size:
        raise ValueError("Duplicate target_vertex_id values are not allowed")

    order = np.argsort(target_vertex_id)
    target_vertex_id = target_vertex_id[order]
    profiles = profiles[order]
    declared_valid = declared_valid[order]
    finite_profile = np.isfinite(profiles).all(axis=1)

    aligned_profiles = np.full((n_vertices, 21), np.nan, dtype=float)
    aligned_declared_valid = np.zeros(n_vertices, dtype=bool)
    aligned_profiles[target_vertex_id] = profiles
    aligned_declared_valid[target_vertex_id] = declared_valid

    return ProfileData(
        target_vertex_id=np.arange(n_vertices, dtype=np.int64),
        profiles=aligned_profiles,
        declared_valid=aligned_declared_valid,
        finite_profile=np.isfinite(aligned_profiles).all(axis=1),
        depths=CANONICAL_DEPTHS.copy(),
        depth_direction="pial_to_white",
    )


def align_scalar_values(
    target_vertex_id: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
    n_vertices: int,
) -> np.ndarray:
    target_vertex_id = np.asarray(target_vertex_id, dtype=np.int64)
    values = np.asarray(values, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if values.ndim != 1 or values.size != target_vertex_id.size:
        raise ValueError("Scalar values do not match target_vertex_id")
    if valid.size != target_vertex_id.size:
        raise ValueError("Scalar validity mask does not match target_vertex_id")
    if np.any(target_vertex_id < 0) or np.any(target_vertex_id >= n_vertices):
        raise ValueError("Scalar target vertex IDs are out of range")
    if np.unique(target_vertex_id).size != target_vertex_id.size:
        raise ValueError("Duplicate scalar target vertex IDs are not allowed")

    aligned = np.full(n_vertices, np.nan, dtype=float)
    aligned_valid = np.zeros(n_vertices, dtype=bool)
    aligned[target_vertex_id] = values
    aligned_valid[target_vertex_id] = valid
    aligned[~aligned_valid] = np.nan
    return aligned


def canonicalize_depths(
    profiles: np.ndarray,
    depths: np.ndarray,
    direction: str,
) -> tuple[np.ndarray, np.ndarray]:
    direction = direction.lower().replace("-", "_")
    if direction not in {"pial_to_white", "white_to_pial"}:
        raise ValueError(f"Cannot determine depth direction: {direction}")

    depths = np.asarray(depths, dtype=float)
    if depths.size != 21:
        raise ValueError(f"Expected 21 depths, found {depths.size}")

    if direction == "white_to_pial":
        profiles = profiles[:, ::-1]
        depths = 1.0 - depths[::-1]

    if not np.allclose(depths, CANONICAL_DEPTHS, atol=1e-8, rtol=0.0):
        raise ValueError("Depth values do not match the canonical 0.00 to 1.00 grid")

    return profiles, CANONICAL_DEPTHS.copy()


def load_npz_profiles(
    path: Path,
    n_vertices: int,
) -> ProfileData:
    with np.load(path, allow_pickle=True) as data:
        profile_key = "md_profiles" if "md_profiles" in data.files else "profiles"
        if profile_key not in data.files:
            raise ValueError(f"No MD profile array found in {path}")
        target_vertex_id = np.asarray(data["target_vertex_id"], dtype=np.int64)
        profiles = np.asarray(data[profile_key], dtype=float)
        depth_key = "depths" if "depths" in data.files else "depth_fractions"
        if depth_key not in data.files:
            raise ValueError(f"Missing depth metadata in {path}")
        depths = np.asarray(data[depth_key], dtype=float)
        if "depth_direction" in data.files:
            direction = str(scalar_value(data["depth_direction"]))
        elif "method" in data.files and str(scalar_value(data["method"])) == (
            "O3-Coordinate"
        ):
            direction = "pial_to_white"
        else:
            raise ValueError(f"Missing depth direction metadata in {path}")

        declared_valid = np.ones(target_vertex_id.size, dtype=bool)
        one_dimensional_masks = (
            "valid_mask",
            "sampling_valid",
            "mapping_valid",
            "in_analysis_mask",
            "column_valid",
            "weights_valid",
            "source_profile_available",
        )
        depth_masks = (
            "coordinate_in_bounds_mask",
            "raw_in_bounds_mask",
            "sampled_finite_mask",
            "finite_mask",
        )
        for key in one_dimensional_masks:
            if key in data.files:
                mask = np.asarray(data[key], dtype=bool)
                if mask.ndim == 1 and mask.size == target_vertex_id.size:
                    declared_valid &= mask
        for key in depth_masks:
            if key in data.files:
                mask = np.asarray(data[key], dtype=bool)
                if mask.shape == (target_vertex_id.size, 21):
                    declared_valid &= mask.all(axis=1)

    profiles, depths = canonicalize_depths(profiles, depths, direction)
    result = ensure_target_ids(
        target_vertex_id,
        profiles,
        declared_valid,
        n_vertices,
    )
    result.depths = depths
    return result


def load_o2_fs_profiles(
    results_root: Path,
    subject: str,
    hemisphere: str,
    template: str,
    n_vertices: int,
) -> ProfileData:
    output_dir = results_root / "qc" / "o2_fs" / subject / template / hemisphere
    manifest_path = output_dir / "vol2surf_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing O2-FS manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    direction = manifest.get("depth_direction")
    output_by_depth = {
        round(float(entry["depth"]), 8): Path(entry["output"])
        for entry in manifest["outputs"]
        if entry.get("status") == "completed"
    }
    profiles = []
    for depth in CANONICAL_DEPTHS:
        path = output_by_depth.get(round(float(depth), 8))
        if path is None:
            path = output_dir / (
                f"{subject}_{hemisphere}_md_depth-{depth:.2f}.mgh"
            )
        if not path.is_file():
            raise FileNotFoundError(f"Missing O2-FS depth map: {path}")
        profiles.append(nib.load(str(path)).get_fdata(dtype=np.float64).reshape(-1))

    profile_matrix = np.column_stack(profiles)
    if profile_matrix.shape[0] != n_vertices:
        raise ValueError(
            f"O2-FS has {profile_matrix.shape[0]} vertices; expected {n_vertices}"
        )
    declared_valid = np.ones(n_vertices, dtype=bool)
    profile_matrix, depths = canonicalize_depths(
        profile_matrix,
        CANONICAL_DEPTHS,
        direction,
    )
    return ensure_target_ids(
        np.arange(n_vertices, dtype=np.int64),
        profile_matrix,
        declared_valid,
        n_vertices,
    )


def method_profile_path(
    results_root: Path,
    method: str,
    subject: str,
    hemisphere: str,
    template: str,
) -> Path:
    if method == "O2-Column":
        combined = (
            results_root
            / "qc"
            / "template_registration_comparison"
            / "o2_column"
            / template
            / subject
            / f"{subject}_{hemisphere}_profiles.npz"
        )
        if combined.is_file():
            return combined
        return (
            results_root
            / "qc"
            / "o2_column"
            / subject
            / template
            / "o2_column"
            / f"{subject}_{hemisphere}_profiles.npz"
        )
    if method == "O3":
        return (
            results_root
            / "fsaverage_result"
            / "columns"
            / subject
            / "profiles"
            / f"{subject}_{hemisphere}_md_profiles.npz"
        )
    raise ValueError(f"No NPZ profile path is defined for {method}")


def load_method_profiles(
    results_root: Path,
    method: str,
    subject: str,
    hemisphere: str,
    template: str,
    n_vertices: int,
) -> ProfileData:
    if method == "O2-FS":
        return load_o2_fs_profiles(
            results_root,
            subject,
            hemisphere,
            template,
            n_vertices,
        )
    path = method_profile_path(
        results_root,
        method,
        subject,
        hemisphere,
        template,
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing {method} profile file: {path}")
    return load_npz_profiles(path, n_vertices)


def load_curvature(
    results_root: Path,
    subject: str,
    hemisphere: str,
    template: str,
    n_vertices: int,
) -> np.ndarray:
    path = (
        results_root
        / "qc"
        / "template_registration_comparison"
        / "curvature"
        / "shared_barycentric"
        / subject
        / template
        / f"{subject}_{hemisphere}_curvature.npz"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing shared curvature file: {path}")
    with np.load(path, allow_pickle=True) as data:
        target_vertex_id = np.asarray(data["target_vertex_id"], dtype=np.int64)
        curvature = np.asarray(data["absolute_mean_curvature"], dtype=float)
        valid = np.asarray(data["valid_mask"], dtype=bool)
    return align_scalar_values(
        target_vertex_id,
        curvature,
        valid,
        n_vertices,
    )


def build_pair_data(
    method_a: ProfileData,
    method_b: ProfileData,
    cortex_mask: np.ndarray,
) -> PairData:
    if not np.array_equal(method_a.target_vertex_id, method_b.target_vertex_id):
        raise ValueError("Methods could not be aligned to the same target vertices")
    if not np.array_equal(method_a.depths, method_b.depths):
        raise ValueError("Method depth grids differ after canonicalization")
    pairwise_valid = (
        cortex_mask
        & method_a.declared_valid
        & method_b.declared_valid
        & method_a.finite_profile
        & method_b.finite_profile
    )
    return PairData(method_a, method_b, pairwise_valid, cortex_mask)


def profile_correlation(
    profiles_a: np.ndarray,
    profiles_b: np.ndarray,
) -> np.ndarray:
    centered_a = profiles_a - profiles_a.mean(axis=1, keepdims=True)
    centered_b = profiles_b - profiles_b.mean(axis=1, keepdims=True)
    numerator = np.sum(centered_a * centered_b, axis=1)
    denominator = np.sqrt(
        np.sum(centered_a**2, axis=1) * np.sum(centered_b**2, axis=1)
    )
    result = np.full(profiles_a.shape[0], np.nan, dtype=float)
    nonconstant = denominator > 0.0
    result[nonconstant] = numerator[nonconstant] / denominator[nonconstant]
    return result


def calculate_vertex_metrics(
    pair: PairData,
    curvature: np.ndarray,
    three_method_common_valid: np.ndarray | None = None,
) -> pd.DataFrame:
    profiles_a = pair.method_a.profiles
    profiles_b = pair.method_b.profiles
    profile_r = np.full(profiles_a.shape[0], np.nan, dtype=float)
    profile_mae = np.full(profiles_a.shape[0], np.nan, dtype=float)
    profile_snmae_pct = np.full(profiles_a.shape[0], np.nan, dtype=float)
    signed_difference = np.full(profiles_a.shape[0], np.nan, dtype=float)
    sd_a = np.full(profiles_a.shape[0], np.nan, dtype=float)
    sd_b = np.full(profiles_a.shape[0], np.nan, dtype=float)

    valid = pair.pairwise_valid
    difference = profiles_b[valid] - profiles_a[valid]
    profile_r[valid] = profile_correlation(
        profiles_a[valid],
        profiles_b[valid],
    )
    profile_mae[valid] = np.mean(np.abs(difference), axis=1)
    snmae_denominator = np.median(np.abs(profiles_a[valid]), axis=1)
    snmae_denominator += np.median(np.abs(profiles_b[valid]), axis=1)
    profile_snmae_pct[valid] = np.divide(
        200.0 * profile_mae[valid],
        snmae_denominator,
        out=np.full(snmae_denominator.shape, np.nan, dtype=float),
        where=snmae_denominator > 0.0,
    )
    signed_difference[valid] = np.mean(difference, axis=1)
    sd_a[valid] = np.std(profiles_a[valid], axis=1, ddof=1)
    sd_b[valid] = np.std(profiles_b[valid], axis=1, ddof=1)

    if three_method_common_valid is None:
        three_method_common_valid = np.zeros(profiles_a.shape[0], dtype=bool)

    return pd.DataFrame(
        {
            "target_vertex_id": pair.method_a.target_vertex_id,
            "profile_r": profile_r,
            "profile_mae": profile_mae,
            "profile_snmae_pct": profile_snmae_pct,
            "profile_mean_signed_difference_B_minus_A": signed_difference,
            "mean_signed_difference_O3_minus_other": signed_difference,
            "profile_sd_method_A": sd_a,
            "profile_sd_method_B": sd_b,
            "profile_sd_other": sd_a,
            "profile_sd_O3": sd_b,
            "absolute_mean_curvature": curvature,
            "valid": pair.pairwise_valid,
            "pairwise_valid": pair.pairwise_valid,
            "three_method_common_valid": three_method_common_valid,
            "method_A_declared_valid": pair.method_a.declared_valid,
            "method_B_declared_valid": pair.method_b.declared_valid,
            "method_A_finite_profile": pair.method_a.finite_profile,
            "method_B_finite_profile": pair.method_b.finite_profile,
        }
    )


def coverage_row(
    subject: str,
    hemisphere: str,
    comparison: str,
    pair: PairData,
    three_method_common_valid: np.ndarray,
) -> dict[str, object]:
    cortex = pair.cortex_mask
    method_a_valid = pair.method_a.declared_valid & pair.method_a.finite_profile
    method_b_valid = pair.method_b.declared_valid & pair.method_b.finite_profile
    both_declared = pair.method_a.declared_valid & pair.method_b.declared_valid
    nonfinite = both_declared & ~(
        pair.method_a.finite_profile & pair.method_b.finite_profile
    )
    return {
        "subject_id": subject,
        "hemisphere": hemisphere,
        "comparison": comparison,
        "n_template_vertices": int(cortex.size),
        "n_anatomical_mask_vertices": int(cortex.sum()),
        "n_analysis_mask_vertices": int(cortex.sum()),
        "n_valid_method_A": int((cortex & method_a_valid).sum()),
        "n_O3_valid": int((cortex & method_b_valid).sum()),
        "n_valid_method_B": int((cortex & method_b_valid).sum()),
        "n_pairwise_valid": int(pair.pairwise_valid.sum()),
        "n_pairwise_valid_vertices": int(pair.pairwise_valid.sum()),
        "pairwise_valid_percentage": float(
            100.0 * pair.pairwise_valid.sum() / max(cortex.sum(), 1)
        ),
        "n_excluded_due_to_A": int(
            (cortex & ~method_a_valid & method_b_valid).sum()
        ),
        "n_excluded_due_to_B": int(
            (cortex & method_a_valid & ~method_b_valid).sum()
        ),
        "n_excluded_due_to_nonfinite_profile": int((cortex & nonfinite).sum()),
        "n_nonfinite_profiles": int((cortex & nonfinite).sum()),
        "n_three_method_common_valid": int(three_method_common_valid.sum()),
        "n_undefined_profile_correlation": 0,
        "exclusion_reason": "",
    }


def write_parquet_chunks(
    path: Path,
    chunks: Iterable[pd.DataFrame],
) -> None:
    writer = None
    try:
        for chunk in chunks:
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def comparison_slug(method_a: str, method_b: str) -> str:
    return f"{method_a.replace('-', '')}_vs_{method_b.replace('-', '')}"


def load_pair(
    args: argparse.Namespace,
    subject: str,
    hemisphere: str,
    cortex_mask: np.ndarray,
) -> tuple[PairData, np.ndarray]:
    method_a = load_method_profiles(
        args.results_root,
        args.method_a,
        subject,
        hemisphere,
        args.template,
        cortex_mask.size,
    )
    method_b = load_method_profiles(
        args.results_root,
        args.method_b,
        subject,
        hemisphere,
        args.template,
        cortex_mask.size,
    )
    pair = build_pair_data(method_a, method_b, cortex_mask)
    curvature = load_curvature(
        args.results_root,
        subject,
        hemisphere,
        args.template,
        cortex_mask.size,
    )
    return pair, curvature


def load_three_method_common_mask(
    args: argparse.Namespace,
    subject: str,
    hemisphere: str,
    pair: PairData,
) -> np.ndarray:
    remaining_methods = set(METHOD_CHOICES) - {args.method_a, args.method_b}
    if len(remaining_methods) != 1:
        raise ValueError("Three-method common mask requires two distinct methods")
    third_method = remaining_methods.pop()
    third = load_method_profiles(
        args.results_root,
        third_method,
        subject,
        hemisphere,
        args.template,
        pair.cortex_mask.size,
    )
    return (
        pair.pairwise_valid
        & third.declared_valid
        & third.finite_profile
    )


def make_cross_subject_summary(
    r_matrix: dict[str, np.ndarray],
) -> dict[str, pd.DataFrame]:
    summaries = {}
    for hemisphere, values in r_matrix.items():
        valid = np.isfinite(values)
        n_subjects = valid.sum(axis=0)
        clipped = np.clip(values, -0.999999, 0.999999)
        z_values = np.where(valid, np.arctanh(clipped), np.nan)
        mean_z = np.full(values.shape[1], np.nan, dtype=float)
        has_values = n_subjects > 0
        mean_z[has_values] = np.nanmean(z_values[:, has_values], axis=0)
        median = np.full(values.shape[1], np.nan, dtype=float)
        q25 = np.full(values.shape[1], np.nan, dtype=float)
        q75 = np.full(values.shape[1], np.nan, dtype=float)
        median[has_values] = np.nanmedian(values[:, has_values], axis=0)
        q25[has_values] = np.nanpercentile(values[:, has_values], 25, axis=0)
        q75[has_values] = np.nanpercentile(values[:, has_values], 75, axis=0)
        summary = pd.DataFrame(
            {
                "hemisphere": hemisphere,
                "target_vertex_id": np.arange(values.shape[1]),
                "mean_profile_r_fisher": np.tanh(mean_z),
                "median_profile_r": median,
                "n_subjects_valid": n_subjects,
                "valid_subject_fraction": n_subjects / values.shape[0],
                "q25_profile_r": q25,
                "q75_profile_r": q75,
            }
        )
        summaries[hemisphere] = summary
    return summaries


def run_comparison(
    args: argparse.Namespace,
    subjects: list[str],
    cortex_masks: dict[str, np.ndarray],
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], list[dict[str, object]]]:
    slug = comparison_slug(args.method_a, args.method_b)
    vertex_path = output_dir / f"figure2_{slug}_vertex_metrics.parquet"
    if vertex_path.exists():
        vertex_path.unlink()
    coverage_rows = []
    manifest_rows = []
    r_matrix = {
        hemisphere: np.full((len(subjects), mask.size), np.nan, dtype=float)
        for hemisphere, mask in cortex_masks.items()
    }

    def chunks() -> Iterable[pd.DataFrame]:
        for subject_index, subject in enumerate(subjects):
            subject_pairwise = []
            for hemisphere in ("lh", "rh"):
                logging.info(
                    "[%s] subject %d/%d %s %s",
                    slug,
                    subject_index + 1,
                    len(subjects),
                    subject,
                    hemisphere,
                )
                cortex_mask = cortex_masks[hemisphere]
                try:
                    pair, curvature = load_pair(
                        args,
                        subject,
                        hemisphere,
                        cortex_mask,
                    )
                    three_method_common_valid = load_three_method_common_mask(
                        args,
                        subject,
                        hemisphere,
                        pair,
                    )
                    metrics = calculate_vertex_metrics(
                        pair,
                        curvature,
                        three_method_common_valid,
                    )
                    metrics.insert(0, "target_vertex_id", metrics.pop("target_vertex_id"))
                    metrics.insert(0, "hemisphere", hemisphere)
                    metrics.insert(0, "subject_id", subject)
                    metrics.insert(0, "comparison", slug)
                    coverage = coverage_row(
                        subject,
                        hemisphere,
                        slug,
                        pair,
                        three_method_common_valid,
                    )
                    undefined = metrics["valid"] & metrics["profile_r"].isna()
                    coverage["n_undefined_profile_correlation"] = int(undefined.sum())
                    coverage_rows.append(coverage)
                    r_matrix[hemisphere][subject_index] = metrics[
                        "profile_r"
                    ].to_numpy()
                    subject_pairwise.append(int(metrics["valid"].sum()))
                    manifest_rows.append(
                        {
                            "subject_id": subject,
                            "hemisphere": hemisphere,
                            "comparison": slug,
                            "subject_included": True,
                            "hemisphere_included": bool(metrics["valid"].any()),
                            "n_pairwise_valid_vertices": int(metrics["valid"].sum()),
                            "exclusion_reason": "",
                        }
                    )
                    yield metrics
                except (FileNotFoundError, ValueError) as error:
                    coverage_rows.append(
                        {
                            "subject_id": subject,
                            "hemisphere": hemisphere,
                            "comparison": slug,
                            "n_template_vertices": int(cortex_mask.size),
                            "n_anatomical_mask_vertices": int(cortex_mask.sum()),
                            "n_analysis_mask_vertices": int(cortex_mask.sum()),
                            "n_valid_method_A": 0,
                            "n_O3_valid": 0,
                            "n_valid_method_B": 0,
                            "n_pairwise_valid": 0,
                            "n_pairwise_valid_vertices": 0,
                            "pairwise_valid_percentage": 0.0,
                            "n_excluded_due_to_A": 0,
                            "n_excluded_due_to_B": 0,
                            "n_excluded_due_to_nonfinite_profile": 0,
                            "n_nonfinite_profiles": 0,
                            "n_three_method_common_valid": 0,
                            "n_undefined_profile_correlation": 0,
                            "exclusion_reason": str(error),
                        }
                    )
                    manifest_rows.append(
                        {
                            "subject_id": subject,
                            "hemisphere": hemisphere,
                            "comparison": slug,
                            "subject_included": False,
                            "hemisphere_included": False,
                            "n_pairwise_valid_vertices": 0,
                            "exclusion_reason": str(error),
                        }
                    )
                    logging.warning(
                        "Skipping %s %s %s: %s",
                        subject,
                        hemisphere,
                        slug,
                        error,
                    )
            if subject_pairwise and max(subject_pairwise) > 0:
                for row in manifest_rows:
                    if row["subject_id"] == subject and row["comparison"] == slug:
                        row["subject_included"] = True

    write_parquet_chunks(vertex_path, chunks())
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(
        output_dir / "figure2_pairwise_coverage_qc.csv",
        mode="a",
        header=not (output_dir / "figure2_pairwise_coverage_qc.csv").exists(),
        index=False,
    )
    return coverage, r_matrix, manifest_rows


def assign_curvature_bins(
    values: np.ndarray,
    n_bins: int = 10,
) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.nanquantile(values, quantiles))
    if edges.size < 2:
        raise ValueError("Curvature distribution does not support two bins")
    if edges.size != n_bins + 1:
        logging.warning(
            "Requested %d curvature bins but only %d unique bins are possible",
            n_bins,
            edges.size - 1,
        )
    return edges


def bin_index(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(edges, values, side="right") - 1
    indices[values == edges[-1]] = edges.size - 2
    return indices


def build_subject_curvature_bins(
    args: argparse.Namespace,
    subjects: list[str],
    cortex_masks: dict[str, np.ndarray],
    n_bins: int = 10,
) -> pd.DataFrame:
    """Summarize MAE in subject-specific curvature quantile bins.

    Hemispheres are pooled within subject before binning, so every subject can
    contribute at most one median MAE to each group-level bin.
    """
    slug = comparison_slug(args.method_a, args.method_b)
    rows = []
    for subject in subjects:
        logging.info(
            "[%s] curvature bins subject %s",
            slug,
            subject,
        )
        subject_maes = []
        subject_curvatures = []
        for hemisphere in ("lh", "rh"):
            try:
                pair, curvature = load_pair(
                    args,
                    subject,
                    hemisphere,
                    cortex_masks[hemisphere],
                )
            except (FileNotFoundError, ValueError):
                continue
            metrics = calculate_vertex_metrics(pair, curvature)
            valid = metrics["valid"].to_numpy() & np.isfinite(curvature)
            subject_maes.append(metrics.loc[valid, "profile_mae"].to_numpy())
            subject_curvatures.append(curvature[valid])
        if not subject_maes:
            continue
        maes = np.concatenate(subject_maes)
        curvatures = np.concatenate(subject_curvatures)
        edges = assign_curvature_bins(curvatures, n_bins=n_bins)
        indices = bin_index(curvatures, edges)
        for current_bin in np.unique(indices):
            in_bin = indices == current_bin
            rows.append(
                {
                    "comparison": slug,
                    "subject_id": subject,
                    "curvature_bin": int(current_bin),
                    "median_profile_mae": float(np.median(maes[in_bin])),
                    "median_absolute_mean_curvature": float(
                        np.median(curvatures[in_bin])
                    ),
                    "n_vertices": int(in_bin.sum()),
                    "n_curvature_bins_subject": int(edges.size - 1),
                }
            )
    return pd.DataFrame(rows)


def build_subject_curvature_spearman(
    args: argparse.Namespace,
    subjects: list[str],
    cortex_masks: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Calculate one within-subject curvature/MAE Spearman rho per subject."""
    slug = comparison_slug(args.method_a, args.method_b)
    rows = []
    for subject in subjects:
        subject_maes = []
        subject_curvatures = []
        for hemisphere in ("lh", "rh"):
            try:
                pair, curvature = load_pair(
                    args, subject, hemisphere, cortex_masks[hemisphere]
                )
            except (FileNotFoundError, ValueError):
                continue
            metrics = calculate_vertex_metrics(pair, curvature)
            valid = metrics["valid"].to_numpy() & np.isfinite(curvature)
            subject_maes.append(metrics.loc[valid, "profile_mae"].to_numpy())
            subject_curvatures.append(curvature[valid])
        if not subject_maes:
            continue
        maes = np.concatenate(subject_maes)
        curvatures = np.concatenate(subject_curvatures)
        rho = spearmanr(curvatures, maes).statistic
        rows.append(
            {
                "comparison": slug,
                "subject_id": subject,
                "spearman_rho": float(rho),
                "n_valid_vertices": int(maes.size),
            }
        )
    return pd.DataFrame(rows)


def summarize_subject_spearman(
    subject_spearman: pd.DataFrame,
    n_bootstrap: int = 10_000,
) -> pd.DataFrame:
    """Summarize independent subject-level Spearman coefficients."""
    rows = []
    rng = np.random.default_rng(20261021)
    for comparison, table in subject_spearman.groupby("comparison"):
        values = table["spearman_rho"].dropna().to_numpy()
        if values.size == 0:
            continue
        bootstrap_medians = np.median(
            rng.choice(values, size=(n_bootstrap, values.size), replace=True), axis=1
        )
        nonzero = values[values != 0.0]
        if nonzero.size:
            test = wilcoxon(values, alternative="two-sided", method="auto")
            statistic = float(test.statistic)
            p_value = float(test.pvalue)
        else:
            statistic = np.nan
            p_value = np.nan
        rows.append(
            {
                "comparison": comparison,
                "n_subjects": int(values.size),
                "median_spearman_rho": float(np.median(values)),
                "q25_spearman_rho": float(np.quantile(values, 0.25)),
                "q75_spearman_rho": float(np.quantile(values, 0.75)),
                "bootstrap_ci95_low": float(np.quantile(bootstrap_medians, 0.025)),
                "bootstrap_ci95_high": float(np.quantile(bootstrap_medians, 0.975)),
                "wilcoxon_statistic": statistic,
                "wilcoxon_p_value": p_value,
                "n_bootstrap": n_bootstrap,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.method_a == args.method_b:
        raise ValueError("method-a and method-b must be different")
    if args.template != "fsaverage":
        raise ValueError("This Figure 2 implementation currently requires fsaverage")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    subjects = read_subjects(args.subjects_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cortex_masks = {
        hemisphere: load_cortex_mask(
            args.freesurfer_subjects_dir,
            args.template,
            hemisphere,
        )
        for hemisphere in ("lh", "rh")
    }
    n_vertices = cortex_masks["lh"].size
    if cortex_masks["rh"].size != n_vertices:
        raise ValueError("Left and right template vertex counts differ")

    coverage_path = args.output_dir / "figure2_pairwise_coverage_qc.csv"
    if coverage_path.exists():
        coverage_path.unlink()

    all_manifest_rows = []
    all_r_matrices = {}
    if args.run_both:
        comparison_results = [
            ("O2-Column", "O3"),
            ("O2-FS", "O3"),
        ]
    else:
        comparison_results = [(args.method_a, args.method_b)]
    for method_a, method_b in comparison_results:
        comparison_args = argparse.Namespace(**vars(args))
        comparison_args.method_a = method_a
        comparison_args.method_b = method_b
        logging.info("Running %s vs %s", method_a, method_b)
        _, r_matrix, manifest_rows = run_comparison(
            comparison_args,
            subjects,
            cortex_masks,
            args.output_dir,
        )
        slug = comparison_slug(method_a, method_b)
        all_r_matrices[slug] = r_matrix
        all_manifest_rows.extend(manifest_rows)
        summaries = make_cross_subject_summary(r_matrix)
        for hemisphere, summary in summaries.items():
            summary.to_csv(
                args.output_dir
                / f"figure2_{slug}_vertex_agreement_map_{hemisphere}.csv",
                index=False,
            )

    pd.DataFrame(
        {
            "curvature_bin": np.arange(10),
            "binning_method": "subject_specific_equal_frequency",
            "n_bins_requested": 10,
        }
    ).to_csv(args.output_dir / "figure2_curvature_bins.csv", index=False)

    subject_bin_tables = []
    subject_spearman_tables = []
    for method_a, method_b in comparison_results:
        comparison_args = argparse.Namespace(**vars(args))
        comparison_args.method_a = method_a
        comparison_args.method_b = method_b
        subject_bins = build_subject_curvature_bins(
            comparison_args,
            subjects,
            cortex_masks,
        )
        slug = comparison_slug(method_a, method_b)
        subject_bins.to_csv(
            args.output_dir
            / f"figure2_{slug}_curvature_subject_bins.csv",
            index=False,
        )
        subject_bin_tables.append(subject_bins)
        subject_spearman = build_subject_curvature_spearman(
            comparison_args,
            subjects,
            cortex_masks,
        )
        subject_spearman.to_csv(
            args.output_dir / f"figure2_{slug}_subject_spearman.csv",
            index=False,
        )
        subject_spearman_tables.append(subject_spearman)

    subject_bins = pd.concat(subject_bin_tables, ignore_index=True)
    cross_summary = (
        subject_bins.groupby(["comparison", "curvature_bin"], as_index=False)
        .agg(
            n_subjects=("subject_id", "nunique"),
            median_mae_across_subjects=("median_profile_mae", "median"),
            q25_mae_across_subjects=("median_profile_mae", lambda x: x.quantile(0.25)),
            q75_mae_across_subjects=("median_profile_mae", lambda x: x.quantile(0.75)),
            median_curvature_across_subjects=(
                "median_absolute_mean_curvature",
                "median",
            ),
            q25_curvature_across_subjects=(
                "median_absolute_mean_curvature",
                lambda x: x.quantile(0.25),
            ),
            q75_curvature_across_subjects=(
                "median_absolute_mean_curvature",
                lambda x: x.quantile(0.75),
            ),
        )
    )
    cross_summary.to_csv(
        args.output_dir / "figure2_curvature_cross_subject_summary.csv",
        index=False,
    )
    subject_spearman = pd.concat(subject_spearman_tables, ignore_index=True)
    subject_spearman.to_csv(
        args.output_dir / "figure2_subject_spearman.csv", index=False
    )
    summarize_subject_spearman(subject_spearman).to_csv(
        args.output_dir / "figure2_subject_spearman_summary.csv", index=False
    )

    manifest = pd.DataFrame(all_manifest_rows)
    manifest.to_csv(args.output_dir / "figure2_analysis_manifest.csv", index=False)
    manifest.to_csv(args.output_dir / "figure2_input_manifest.csv", index=False)
    manifest.loc[~manifest["hemisphere_included"]].to_csv(
        args.output_dir / "figure2_exclusions.csv",
        index=False,
    )
    run_config = {
        "template": args.template,
        "subjects_file": str(args.subjects_file),
        "n_subjects_requested": len(subjects),
        "comparisons": [
            f"{method_a} vs {method_b}"
            for method_a, method_b in comparison_results
        ],
        "canonical_depth_direction": "pial_to_white",
        "canonical_depths": CANONICAL_DEPTHS.tolist(),
        "n_curvature_bins_requested": 10,
        "curvature_binning": "subject_specific_equal_frequency",
        "curvature_reference": (
            "O3 target absolute mean curvature within the fsaverage "
            "anatomical analysis mask"
        ),
    }
    (args.output_dir / "figure2_run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n"
    )
    logging.info("Saved Figure 2A/B tables to %s", args.output_dir)


if __name__ == "__main__":
    main()
