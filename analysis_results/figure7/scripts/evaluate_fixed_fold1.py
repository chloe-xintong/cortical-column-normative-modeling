#!/usr/bin/env python3
"""Evaluate all Figure 7 participants with the frozen Fold 1 prediction functions.

Description
-----------
Evaluate all Figure 7 participants with the frozen Fold 1 prediction functions.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--figure6-root`` command-line input
- ``--input-dir`` command-line input
- ``--output-dir`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure7/scripts/evaluate_fixed_fold1.py --help
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


_CONFIG: dict[str, str] = {}
_INDEX: pd.DataFrame | None = None
_ROSTER: pd.DataFrame | None = None
_MD: np.ndarray | None = None
_THICKNESS: np.ndarray | None = None

DEPTH_BANDS = {
    "outer": tuple(range(0, 7)),
    "middle": tuple(range(7, 14)),
    "inner": tuple(range(14, 21)),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--figure6-root", type=Path, required=True)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--outer-fold", type=int, default=1)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--z-threshold", type=float, default=2.0)
    return p.parse_args()


def initialize(config: dict[str, str]) -> None:
    global _CONFIG, _INDEX, _ROSTER, _MD, _THICKNESS
    _CONFIG = config
    figure6 = Path(config["figure6_root"])
    input_dir = Path(config["input_dir"])
    _INDEX = pd.read_parquet(figure6 / "oof" / "prediction_file_index.parquet")
    _ROSTER = pd.read_csv(input_dir / "final_analysis_roster.csv", dtype={"subject_id": str})
    _MD = np.load(input_dir / "X_figure7_ico4_md.npy", mmap_mode="r")
    _THICKNESS = np.load(input_dir / "X_figure7_local_freesurfer_thickness.npy", mmap_mode="r")


def warp_forward(x: np.ndarray, epsilon: float, log_b: float) -> np.ndarray:
    b = np.exp(log_b)
    a = -epsilon * b
    return np.sinh(b * np.arcsinh(x) - a)


def mean_features(age_z: np.ndarray, sex_indicator: np.ndarray,
                  thickness_z: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(age_z)), age_z, sex_indicator, thickness_z])


def variance_features(age_z: np.ndarray, sex_indicator: np.ndarray,
                      thickness_z: np.ndarray) -> np.ndarray:
    """Independent quadratic terms for binary sex and continuous age/thickness."""
    return np.column_stack([
        np.ones(len(age_z)), age_z, sex_indicator, thickness_z,
        age_z ** 2, thickness_z ** 2, age_z * sex_indicator,
        age_z * thickness_z, sex_indicator * thickness_z,
    ])


def worker(node_index: int) -> dict[str, object]:
    started = time.monotonic()
    try:
        figure6 = Path(_CONFIG["figure6_root"])
        if _INDEX is None or _ROSTER is None or _MD is None or _THICKNESS is None:
            raise RuntimeError("Worker was not initialized")
        roster, md, thickness = _ROSTER, _MD, _THICKNESS
        hit = _INDEX[(_INDEX.fold == int(_CONFIG["outer_fold"])) & (_INDEX.node_index == node_index)]
        if len(hit) != 1:
            raise RuntimeError(f"Expected one Fold 1 OOF prediction shard for node {node_index}")
        relative = str(hit.relative_path.iloc[0])
        saved = pd.read_parquet(figure6 / relative)
        diagnostics = pd.read_parquet(figure6 / relative.replace("predictions", "diagnostics"))
        if set(saved.outer_fold.unique()) != {int(_CONFIG["outer_fold"])}:
            raise RuntimeError("Mixed-fold prediction shard")
        heldout = set(roster.loc[roster.model_role.eq("fold1_heldout_cn"), "subject_id"])
        threshold = float(_CONFIG["z_threshold"])
        n = len(roster)
        n_valid = np.zeros(n, dtype=np.int32)
        n_positive = np.zeros(n, dtype=np.int32)
        n_negative = np.zeros(n, dtype=np.int32)
        band_totals = {
            band: {name: np.zeros(n, dtype=np.int32)
                   for name in ("n_valid", "n_positive", "n_negative")}
            for band in DEPTH_BANDS
        }
        validations = []
        for depth in range(21):
            band = next(name for name, indices in DEPTH_BANDS.items() if depth in indices)
            drow = diagnostics.loc[diagnostics.depth_index.eq(depth)]
            if len(drow) != 1 or not str(drow.status.iloc[0]).startswith("complete"):
                raise RuntimeError(f"Missing complete diagnostic n{node_index} d{depth}")
            d = drow.iloc[0]
            reference = saved.loc[saved.depth_index.eq(depth)].copy()
            if len(reference) < 1:
                raise RuntimeError(f"No saved held-out design point n{node_index} d{depth}")
            age_z = (reference.age.to_numpy(float) - float(d.age_mean)) / float(d.age_scale)
            sex = (reference.sex_code.to_numpy(int) == 2).astype(float)
            thickness_z = (reference.local_thickness.to_numpy(float) - float(d.thickness_mean)) / float(d.thickness_scale)
            phi = mean_features(age_z, sex, thickness_z)
            qphi = variance_features(age_z, sex, thickness_z)
            median_scaled = (reference.pred_median.to_numpy(float) - float(d.response_mean)) / float(d.response_scale)
            latent_mean = warp_forward(median_scaled, float(d.warp_epsilon), float(d.warp_log_b))
            lower = warp_forward(
                (reference.lower_50.to_numpy(float) - float(d.response_mean)) / float(d.response_scale),
                float(d.warp_epsilon), float(d.warp_log_b),
            )
            upper = warp_forward(
                (reference.upper_50.to_numpy(float) - float(d.response_mean)) / float(d.response_scale),
                float(d.warp_epsilon), float(d.warp_log_b),
            )
            latent_sd = (upper - lower) / (2 * stats.norm.ppf(0.75))
            latent_var = latent_sd ** 2
            mean_coef, _, mean_rank, _ = np.linalg.lstsq(phi, latent_mean, rcond=None)
            var_coef, _, var_rank, _ = np.linalg.lstsq(qphi, latent_var, rcond=None)
            recovered_mean = phi @ mean_coef
            recovered_var = qphi @ var_coef
            observed_scaled = (reference.observed_md.to_numpy(float) - float(d.response_mean)) / float(d.response_scale)
            recovered_z = (
                warp_forward(observed_scaled, float(d.warp_epsilon), float(d.warp_log_b)) - recovered_mean
            ) / np.sqrt(recovered_var)
            mean_error = float(np.max(np.abs(recovered_mean - latent_mean)))
            var_error = float(np.max(np.abs(recovered_var - latent_var)))
            z_error = float(np.max(np.abs(recovered_z - reference.z.to_numpy(float))))
            if mean_rank < 4 or var_rank < 9:
                # The serialized Figure 6 artifacts do not contain enough held-out design
                # points to recover the target prediction function. Preserve available CN
                # OOF z-scores, but do not invent or refit target z-scores.
                z = np.full(n, np.nan)
                saved_z = reference.set_index("subject_id").z
                for sid, value in saved_z.items():
                    z[int(roster.index[roster.subject_id.eq(sid)][0])] = float(value)
                n_valid += np.isfinite(z)
                n_positive += z > threshold
                n_negative += z < -threshold
                band_totals[band]["n_valid"] += np.isfinite(z)
                band_totals[band]["n_positive"] += z > threshold
                band_totals[band]["n_negative"] += z < -threshold
                validations.append({
                    "node_index": node_index, "depth_index": depth,
                    "recovery_status": "target_unavailable_insufficient_saved_design_rank",
                    "n_saved_heldout": len(reference), "mean_design_rank": int(mean_rank),
                    "variance_design_rank": int(var_rank),
                    "max_abs_latent_mean_recovery_error": mean_error,
                    "max_abs_latent_variance_recovery_error": var_error,
                    "max_abs_saved_z_recovery_error": z_error,
                    "max_abs_heldout_input_z_error": np.nan,
                    "n_target_valid": int(np.isfinite(z).sum()),
                    "minimum_target_latent_variance": np.nan,
                })
                continue
            if min(recovered_var) <= 0 or z_error > 1e-8:
                raise RuntimeError(
                    f"Fixed model recovery failed n{node_index} d{depth}: ranks={mean_rank}/{var_rank}, zerr={z_error}"
                )

            y = np.asarray(md[:, node_index * 21 + depth], float)
            t = np.asarray(thickness[:, node_index], float)
            valid = np.isfinite(y) & np.isfinite(t)
            target_age_z = (roster.age_at_scan.to_numpy(float) - float(d.age_mean)) / float(d.age_scale)
            target_sex = (roster.SUBJECT_SEX.to_numpy(int) == 2).astype(float)
            target_thickness_z = (t - float(d.thickness_mean)) / float(d.thickness_scale)
            target_phi = mean_features(target_age_z, target_sex, target_thickness_z)
            target_qphi = variance_features(target_age_z, target_sex, target_thickness_z)
            target_var = target_qphi @ var_coef
            valid &= np.isfinite(target_var) & (target_var > 0)
            z = np.full(n, np.nan)
            y_scaled = (y[valid] - float(d.response_mean)) / float(d.response_scale)
            z[valid] = (
                warp_forward(y_scaled, float(d.warp_epsilon), float(d.warp_log_b)) -
                target_phi[valid] @ mean_coef
            ) / np.sqrt(target_var[valid])

            saved_z = reference.set_index("subject_id").z
            heldout_rows = roster.subject_id.isin(heldout).to_numpy()
            heldout_valid = heldout_rows & valid
            expected_ids = set(roster.loc[heldout_valid, "subject_id"])
            if expected_ids != set(saved_z.index):
                raise RuntimeError(f"Held-out availability mismatch n{node_index} d{depth}")
            computed_cn = pd.Series(z[heldout_valid], index=roster.loc[heldout_valid, "subject_id"])
            input_z_error = float(np.max(np.abs(computed_cn.loc[saved_z.index].to_numpy() - saved_z.to_numpy())))
            if input_z_error > 1e-6:
                raise RuntimeError(f"Figure 7 held-out input mismatch n{node_index} d{depth}: {input_z_error}")
            # Requirement: CN burdens use the saved OOF z-score, not recomputed values.
            for sid, value in saved_z.items():
                z[int(roster.index[roster.subject_id.eq(sid)][0])] = float(value)
            n_valid += np.isfinite(z)
            n_positive += z > threshold
            n_negative += z < -threshold
            band_totals[band]["n_valid"] += np.isfinite(z)
            band_totals[band]["n_positive"] += z > threshold
            band_totals[band]["n_negative"] += z < -threshold
            validations.append({
                "node_index": node_index, "depth_index": depth,
                "recovery_status": "complete",
                "n_saved_heldout": len(reference), "mean_design_rank": int(mean_rank),
                "variance_design_rank": int(var_rank), "max_abs_latent_mean_recovery_error": mean_error,
                "max_abs_latent_variance_recovery_error": var_error,
                "max_abs_saved_z_recovery_error": z_error,
                "max_abs_heldout_input_z_error": input_z_error,
                "n_target_valid": int(np.isfinite(z).sum()),
                "minimum_target_latent_variance": float(np.nanmin(target_var[valid])),
            })
        return {"status": "complete", "node_index": node_index, "n_valid": n_valid,
                "n_positive": n_positive, "n_negative": n_negative,
                "band_totals": band_totals,
                "validations": validations, "elapsed_seconds": time.monotonic() - started}
    except Exception as error:
        return {"status": "failed", "node_index": node_index, "error": repr(error),
                "traceback": traceback.format_exc(), "elapsed_seconds": time.monotonic() - started}


def main() -> None:
    a = parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    roster = pd.read_csv(a.input_dir / "final_analysis_roster.csv", dtype={"subject_id": str})
    n = len(roster)
    totals = {name: np.zeros(n, dtype=np.int64) for name in ("n_valid", "n_positive", "n_negative")}
    band_totals = {
        band: {name: np.zeros(n, dtype=np.int64)
               for name in ("n_valid", "n_positive", "n_negative")}
        for band in DEPTH_BANDS
    }
    validations, failures = [], []
    config = {"figure6_root": str(a.figure6_root), "input_dir": str(a.input_dir),
              "outer_fold": str(a.outer_fold), "z_threshold": str(a.z_threshold)}
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=a.n_jobs, initializer=initialize, initargs=(config,)) as pool:
        futures = {pool.submit(worker, node): node for node in range(4687)}
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result["status"] == "failed":
                failures.append(result)
            else:
                for name in totals:
                    totals[name] += result[name]
                for band in DEPTH_BANDS:
                    for name in totals:
                        band_totals[band][name] += result["band_totals"][band][name]
                validations.extend(result["validations"])
            if done % 100 == 0 or done == len(futures):
                print(json.dumps({"completed_nodes": done, "total_nodes": 4687,
                                  "failures": len(failures)}), flush=True)
    (a.output_dir / "model_recovery_failures.json").write_text(
        json.dumps(failures, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    if failures:
        raise RuntimeError(f"{len(failures)} fixed Fold 1 node evaluations failed")
    validation = pd.DataFrame(validations).sort_values(["node_index", "depth_index"])
    validation.to_parquet(a.output_dir / "model_recovery_validation_by_position.parquet", index=False)
    burden = roster.copy()
    burden["n_valid_positions"] = totals["n_valid"]
    burden["n_positive_z_gt_2"] = totals["n_positive"]
    burden["n_negative_z_lt_minus_2"] = totals["n_negative"]
    burden["n_total_abs_z_gt_2"] = totals["n_positive"] + totals["n_negative"]
    if (burden.n_valid_positions <= 0).any():
        raise RuntimeError("At least one participant has no valid z-score position")
    burden["positive_deviation_burden"] = burden.n_positive_z_gt_2 / burden.n_valid_positions
    burden["negative_deviation_burden"] = burden.n_negative_z_lt_minus_2 / burden.n_valid_positions
    burden["total_deviation_burden"] = burden.n_total_abs_z_gt_2 / burden.n_valid_positions
    burden.to_csv(a.output_dir / "participant_deviation_burden.csv", index=False)
    band_rows = []
    for band, indices in DEPTH_BANDS.items():
        frame = roster.copy()
        frame["depth_band"] = band
        frame["depth_indices"] = f"{min(indices)}-{max(indices)}"
        frame["n_valid_positions"] = band_totals[band]["n_valid"]
        frame["n_positive_z_gt_2"] = band_totals[band]["n_positive"]
        frame["n_negative_z_lt_minus_2"] = band_totals[band]["n_negative"]
        frame["n_total_abs_z_gt_2"] = frame.n_positive_z_gt_2 + frame.n_negative_z_lt_minus_2
        if (frame.n_valid_positions <= 0).any():
            raise RuntimeError(f"At least one participant has no valid {band}-band position")
        frame["positive_deviation_burden"] = frame.n_positive_z_gt_2 / frame.n_valid_positions
        frame["negative_deviation_burden"] = frame.n_negative_z_lt_minus_2 / frame.n_valid_positions
        frame["total_deviation_burden"] = frame.n_total_abs_z_gt_2 / frame.n_valid_positions
        band_rows.append(frame)
    pd.concat(band_rows, ignore_index=True).to_csv(
        a.output_dir / "participant_depth_band_deviation_burden.csv", index=False
    )
    report = {
        "status": "complete", "outer_fold": a.outer_fold,
        "fold_numbering": "zero-based; selected Fold 1 is outer_fold=1", "n_participants": len(burden),
        "n_nodes": 4687, "n_depths": 21, "fixed_mask_positions": 4687 * 21,
        "depth_bands": {name: list(indices) for name, indices in DEPTH_BANDS.items()},
        "z_threshold": a.z_threshold, "model_refit": False, "folds_mixed": False,
        "heldout_cn_z_source": "saved Figure 6 Fold 1 OOF z-scores copied directly",
        "target_z_source": "fixed latent mean/variance functions recovered from saved Fold 1 OOF medians, intervals, transformations, and warp",
        "recovery_positions": int(validation.recovery_status.eq("complete").sum()),
        "target_unavailable_positions_insufficient_saved_design_rank": int(
            validation.recovery_status.ne("complete").sum()),
        "maximum_saved_z_recovery_error": float(validation.max_abs_saved_z_recovery_error.max()),
        "maximum_heldout_input_z_error": float(validation.max_abs_heldout_input_z_error.max(skipna=True)),
        "minimum_target_latent_variance": float(validation.minimum_target_latent_variance.min(skipna=True)),
        "elapsed_seconds": time.monotonic() - started,
    }
    (a.output_dir / "model_recovery_validation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
