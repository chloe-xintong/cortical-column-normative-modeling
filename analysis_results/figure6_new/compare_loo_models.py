#!/usr/bin/env python3
"""Compare young-CN LOO models A, B, and C at ico4 node-depth positions.

Model A: OLS raw MD ~ categorical sex + local FreeSurfer thickness.
Model B: PCNToolkit 1.3.0 homoskedastic WarpSinhArcsinh BLR with the same
         covariates and the historical Figure 6 training-fold scaling.
Model C: OLS raw MD ~ categorical sex.

Every held-out participant has a separately fitted training model. A/B and
A/C comparisons use the same training observations and only score positions
jointly valid for the corresponding pair.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pcntoolkit import BLR
import pcntoolkit


REPO = Path(__file__).resolve().parents[2]
NEW_ROOT = REPO / "analysis_results/figure6_new"
FEATURE_ROOT = Path(
    "/mnt/newStor/paros/paros_WORK/xintong/results/"
    "figure5_age_prediction/outputs/features"
)
FIGURE6_ROOT = REPO / "analysis_results/figure6/ico4_warped_m2_186"
N_NODES = 4687
N_DEPTHS = 21
DEPTH_BANDS = {
    "outer": tuple(range(0, 7)),
    "middle": tuple(range(7, 14)),
    "inner": tuple(range(14, 21)),
}
BAND_INDEX = {depth: band for band, depths in DEPTH_BANDS.items() for depth in depths}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("task", "summarize", "smoke"), required=True)
    parser.add_argument("--node-index", type=int)
    parser.add_argument("--smoke-nodes", type=int, default=1)
    parser.add_argument("--z-threshold", type=float, default=2.0)
    parser.add_argument("--minimum-training-n", type=int, default=10)
    parser.add_argument(
        "--reference-roster", type=Path,
        default=NEW_ROOT / "outputs/full/young_reference_roster.csv",
    )
    parser.add_argument(
        "--ico4-md", type=Path,
        default=FEATURE_ROOT / "matrices/X_ico4_md.npy",
    )
    parser.add_argument(
        "--local-thickness", type=Path,
        default=FIGURE6_ROOT / "inputs/X_ico4_local_freesurfer_thickness.npy",
    )
    parser.add_argument(
        "--node-metadata", type=Path,
        default=FIGURE6_ROOT / "ico4_node_metadata.csv",
    )
    parser.add_argument(
        "--smoke-node-list", type=Path,
        default=FIGURE6_ROOT / "smoke_nodes.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=NEW_ROOT / "outputs/loo_model_comparison_full",
    )
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def fit_ols_predict(
    y_train: np.ndarray,
    sex_train: np.ndarray,
    thickness_train: np.ndarray,
    y_test: float,
    sex_test: float,
    thickness_test: float,
    include_thickness: bool,
) -> tuple[float, float, float]:
    columns = [np.ones(len(y_train)), sex_train]
    test = [1.0, sex_test]
    if include_thickness:
        columns.append(thickness_train)
        test.append(thickness_test)
    design = np.column_stack(columns)
    n_parameters = design.shape[1]
    if len(y_train) <= n_parameters or np.linalg.matrix_rank(design) != n_parameters:
        raise RuntimeError("Rank-deficient OLS training design")
    beta, _, _, _ = np.linalg.lstsq(design, y_train, rcond=None)
    fitted = design @ beta
    residual_sd = float(np.sqrt(np.sum((y_train - fitted) ** 2) / (len(y_train) - n_parameters)))
    prediction = float(np.asarray(test) @ beta)
    if not np.isfinite(residual_sd) or residual_sd <= 0:
        raise RuntimeError("Invalid OLS training residual SD")
    return prediction, residual_sd, float((y_test - prediction) / residual_sd)


def fit_warped_z(
    y_train: np.ndarray,
    sex_train: np.ndarray,
    thickness_train: np.ndarray,
    y_test: float,
    sex_test: float,
    thickness_test: float,
    name: str,
) -> tuple[float, float, float, bool]:
    thickness_mean = float(thickness_train.mean())
    thickness_scale = float(thickness_train.std(ddof=0))
    response_mean = float(y_train.mean())
    response_scale = float(y_train.std(ddof=0))
    scales = np.asarray([thickness_scale, response_scale])
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise RuntimeError("Invalid warped-model training scale")
    x_train = np.column_stack([
        sex_train,
        (thickness_train - thickness_mean) / thickness_scale,
    ])
    x_test = np.asarray([[
        sex_test,
        (thickness_test - thickness_mean) / thickness_scale,
    ]])
    response_train = (y_train - response_mean) / response_scale
    response_test = np.asarray([(y_test - response_mean) / response_scale])
    be_train = xr.DataArray(
        np.zeros((len(y_train), 1), dtype=int),
        dims=("observations", "batch_effect_dims"),
    )
    be_test = xr.DataArray(
        np.zeros((1, 1), dtype=int), dims=("observations", "batch_effect_dims")
    )
    model = BLR(
        name=name, heteroskedastic=False, ard=False, optimizer="l-bfgs-b",
        n_iter=100, tol=1e-3, warp_name="WarpSinhArcsinh", warp_reparam=True,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(
            xr.DataArray(x_train, dims=("observations", "covariates")),
            be_train,
            {"dummy_batch_effect": {"0": 0}},
            xr.DataArray(response_train, dims=("observations",)),
        )
    if type(model.warp).__name__ != "WarpSinhArcsinh" or model.heteroskedastic:
        raise RuntimeError("Warped model silently changed family")
    z = float(np.asarray(model.forward(
        xr.DataArray(x_test, dims=("observations", "covariates")),
        be_test,
        xr.DataArray(response_test, dims=("observations",)),
    ).values)[0])
    gamma = np.asarray(model.gamma, dtype=float)
    if not np.isfinite(z) or not np.isfinite(gamma).all() or not np.isfinite(model.nlZ):
        raise RuntimeError("Nonfinite warped-model result")
    if float(model.warp.df(response_test, gamma)[0]) <= 0:
        raise RuntimeError("Invalid WarpSinhArcsinh Jacobian")
    return z, float(gamma[0]), float(gamma[1]), bool(caught)


def load_inputs(args: argparse.Namespace):
    required = [
        args.reference_roster, args.ico4_md, args.local_thickness,
        args.node_metadata,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))
    roster = pd.read_csv(args.reference_roster, dtype={"subject_id": str})
    if len(roster) != 37 or roster.subject_id.nunique() != 37:
        raise RuntimeError("Expected the exact 37-person young-CN roster")
    if not np.array_equal(roster.loo_row_index.to_numpy(), np.arange(37)):
        raise RuntimeError("Young-CN loo_row_index is not contiguous")
    if not roster.actual_age.lt(50).all() or set(roster.diagnosis_at_scan) != {"Normal"}:
        raise RuntimeError("Reference roster is not exclusively CN younger than 50")
    md = np.load(args.ico4_md, mmap_mode="r")
    thickness = np.load(args.local_thickness, mmap_mode="r")
    nodes = pd.read_csv(args.node_metadata)
    if md.shape[1] != N_NODES * N_DEPTHS or thickness.shape[1] != N_NODES:
        raise RuntimeError("Unexpected ico4 input dimensions")
    if len(nodes) != N_NODES or not np.array_equal(nodes.node_index, np.arange(N_NODES)):
        raise RuntimeError("Unexpected ico4 node order")
    return roster, md, thickness, nodes


def empty_accumulators(n_subjects: int) -> dict[str, dict[str, np.ndarray]]:
    names = (
        "ab_valid", "a_ab_positive", "a_ab_negative", "b_positive", "b_negative",
        "ac_valid", "a_squared_error", "c_squared_error", "a_ac_abs_gt2",
        "c_abs_gt2",
    )
    return {
        band: {
            name: np.zeros(n_subjects, dtype=float if "squared" in name else np.int64)
            for name in names
        }
        for band in DEPTH_BANDS
    }


def run_node(args: argparse.Namespace, node_index: int, output_dir: Path) -> dict[str, object]:
    if not 0 <= node_index < N_NODES:
        raise ValueError(f"node_index outside 0..{N_NODES - 1}")
    shard_dir = output_dir / "shards"
    status_dir = output_dir / "status"
    shard_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"node_{node_index:04d}.parquet"
    status_path = status_dir / f"node_{node_index:04d}.json"
    if shard_path.is_file() and status_path.is_file():
        previous = json.loads(status_path.read_text())
        if previous.get("status") == "complete":
            return previous | {"already_complete": True}

    roster, md, thickness, nodes = load_inputs(args)
    rows = roster.matrix_row_index.astype(int).to_numpy()
    sex = roster.SUBJECT_SEX.eq(2).to_numpy(dtype=float)
    node_md = np.asarray(
        md[rows, node_index * N_DEPTHS:(node_index + 1) * N_DEPTHS], float
    )
    node_thickness = np.asarray(thickness[rows, node_index], float)
    accum = empty_accumulators(len(roster))
    attempted_warped = successful_warped = warning_warped = 0
    failed_warped: list[dict[str, object]] = []
    gamma_epsilon: list[float] = []
    gamma_log_b: list[float] = []
    started = time.monotonic()

    for depth in range(N_DEPTHS):
        band = BAND_INDEX[depth]
        y = node_md[:, depth]
        position_valid = np.isfinite(y) & np.isfinite(node_thickness) & np.isfinite(sex)
        for held_out in np.flatnonzero(position_valid):
            train = position_valid.copy()
            train[held_out] = False
            if train.sum() < args.minimum_training_n:
                continue
            y_train = y[train]
            sex_train = sex[train]
            thickness_train = node_thickness[train]
            try:
                a_prediction, _, a_z = fit_ols_predict(
                    y_train, sex_train, thickness_train,
                    float(y[held_out]), float(sex[held_out]),
                    float(node_thickness[held_out]), True,
                )
                c_prediction, _, c_z = fit_ols_predict(
                    y_train, sex_train, thickness_train,
                    float(y[held_out]), float(sex[held_out]),
                    float(node_thickness[held_out]), False,
                )
            except Exception:
                continue

            ac = accum[band]
            ac["ac_valid"][held_out] += 1
            ac["a_squared_error"][held_out] += (y[held_out] - a_prediction) ** 2
            ac["c_squared_error"][held_out] += (y[held_out] - c_prediction) ** 2
            ac["a_ac_abs_gt2"][held_out] += abs(a_z) > args.z_threshold
            ac["c_abs_gt2"][held_out] += abs(c_z) > args.z_threshold

            attempted_warped += 1
            try:
                b_z, epsilon, log_b, warned = fit_warped_z(
                    y_train, sex_train, thickness_train,
                    float(y[held_out]), float(sex[held_out]),
                    float(node_thickness[held_out]),
                    f"young_B_n{node_index}_d{depth}_loo{held_out}",
                )
                successful_warped += 1
                warning_warped += int(warned)
                gamma_epsilon.append(epsilon)
                gamma_log_b.append(log_b)
                ab = accum[band]
                ab["ab_valid"][held_out] += 1
                ab["a_ab_positive"][held_out] += a_z > args.z_threshold
                ab["a_ab_negative"][held_out] += a_z < -args.z_threshold
                ab["b_positive"][held_out] += b_z > args.z_threshold
                ab["b_negative"][held_out] += b_z < -args.z_threshold
            except Exception as error:
                failed_warped.append({
                    "held_out": int(held_out), "depth_index": depth,
                    "error": repr(error), "traceback": traceback.format_exc(),
                })

    records = []
    for subject in roster.itertuples(index=False):
        i = int(subject.loo_row_index)
        for band in DEPTH_BANDS:
            row = {
                "node_index": node_index,
                "hemisphere": str(nodes.iloc[node_index].hemisphere),
                "subject_id": subject.subject_id,
                "loo_row_index": i,
                "depth_band": band,
            }
            row.update({name: values[i] for name, values in accum[band].items()})
            records.append(row)
    atomic_parquet(pd.DataFrame(records), shard_path)
    report = {
        "status": "complete",
        "node_index": node_index,
        "attempted_warped_fits": attempted_warped,
        "successful_warped_fits": successful_warped,
        "failed_warped_fits": len(failed_warped),
        "warning_warped_fits": warning_warped,
        "warp_epsilon_range": [min(gamma_epsilon), max(gamma_epsilon)] if gamma_epsilon else None,
        "warp_log_b_range": [min(gamma_log_b), max(gamma_log_b)] if gamma_log_b else None,
        "failures": failed_warped,
        "elapsed_seconds": time.monotonic() - started,
        "pcntoolkit_version": pcntoolkit.__version__,
    }
    write_json(status_path, report)
    return report


def quantile_summary(values: pd.Series) -> tuple[float, float, float]:
    clean = values.dropna().to_numpy(float)
    if len(clean) == 0:
        return np.nan, np.nan, np.nan
    q25, median, q75 = np.quantile(clean, [0.25, 0.5, 0.75])
    return float(median), float(q25), float(q75)


def summarize(args: argparse.Namespace, output_dir: Path) -> dict[str, object]:
    shard_paths = sorted((output_dir / "shards").glob("node_*.parquet"))
    status_paths = sorted((output_dir / "status").glob("node_*.json"))
    expected = N_NODES if args.action == "summarize" else args.smoke_nodes
    if len(shard_paths) != expected or len(status_paths) != expected:
        raise RuntimeError(
            f"Expected {expected} completed node shards/status files; "
            f"found {len(shard_paths)}/{len(status_paths)}"
        )
    status = [json.loads(path.read_text()) for path in status_paths]
    if any(item.get("status") != "complete" for item in status):
        raise RuntimeError("At least one node task is incomplete")
    data = pd.concat((pd.read_parquet(path) for path in shard_paths), ignore_index=True)
    numeric = [
        "ab_valid", "a_ab_positive", "a_ab_negative", "b_positive", "b_negative",
        "ac_valid", "a_squared_error", "c_squared_error", "a_ac_abs_gt2", "c_abs_gt2",
    ]
    paired = data.groupby(
        ["subject_id", "loo_row_index", "depth_band"], observed=True, as_index=False
    )[numeric].sum()
    roster = pd.read_csv(args.reference_roster, dtype={"subject_id": str})
    paired = paired.merge(
        roster[["subject_id", "actual_age", "SUBJECT_SEX", "coverage_fraction"]],
        on="subject_id", how="left", validate="many_to_one",
    )
    for model in ("a", "b"):
        positive = paired["a_ab_positive" if model == "a" else "b_positive"]
        negative = paired["a_ab_negative" if model == "a" else "b_negative"]
        paired[f"{model}_positive_tail_fraction"] = positive / paired.ab_valid
        paired[f"{model}_negative_tail_fraction"] = negative / paired.ab_valid
        paired[f"{model}_coverage_minus2_to_2"] = 1.0 - (positive + negative) / paired.ab_valid
        paired[f"{model}_positive_minus_negative_pp"] = (
            paired[f"{model}_positive_tail_fraction"]
            - paired[f"{model}_negative_tail_fraction"]
        ) * 100.0
    paired.to_csv(output_dir / "a_vs_b_participant_depth_band.csv", index=False)

    ac = paired.groupby(
        ["subject_id", "loo_row_index"], as_index=False
    )[["ac_valid", "a_squared_error", "c_squared_error"]].sum()
    ac["a_rmse_md"] = np.sqrt(ac.a_squared_error / ac.ac_valid)
    ac["c_rmse_md"] = np.sqrt(ac.c_squared_error / ac.ac_valid)
    ac["a_minus_c_rmse_md"] = ac.a_rmse_md - ac.c_rmse_md
    outer = paired.loc[paired.depth_band.eq("outer"), [
        "subject_id", "ac_valid", "a_ac_abs_gt2", "c_abs_gt2"
    ]].copy()
    outer["a_outer_burden"] = outer.a_ac_abs_gt2 / outer.ac_valid
    outer["c_outer_burden"] = outer.c_abs_gt2 / outer.ac_valid
    outer["a_minus_c_outer_burden_pp"] = (
        outer.a_outer_burden - outer.c_outer_burden
    ) * 100.0
    ac = ac.merge(
        outer[["subject_id", "a_outer_burden", "c_outer_burden",
               "a_minus_c_outer_burden_pp"]],
        on="subject_id", validate="one_to_one",
    ).merge(
        roster[["subject_id", "actual_age", "SUBJECT_SEX", "coverage_fraction"]],
        on="subject_id", validate="one_to_one",
    )
    ac.to_csv(output_dir / "a_vs_c_participant_paired.csv", index=False)

    summary_rows: list[dict[str, object]] = []
    for band in DEPTH_BANDS:
        subset = paired.loc[paired.depth_band.eq(band)]
        for model in ("a", "b"):
            for metric in (
                "positive_tail_fraction", "negative_tail_fraction",
                "coverage_minus2_to_2",
            ):
                column = f"{model}_{metric}"
                median, q25, q75 = quantile_summary(subset[column] * 100.0)
                summary_rows.append({
                    "comparison": "A_vs_B", "depth_band": band,
                    "model": model.upper(), "metric": metric + "_percent",
                    "n_participants": int(subset[column].notna().sum()),
                    "median": median, "q25": q25, "q75": q75,
                })
            median, q25, q75 = quantile_summary(subset.ab_valid)
            summary_rows.append({
                "comparison": "A_vs_B", "depth_band": band,
                "model": model.upper(), "metric": "common_valid_positions",
                "n_participants": len(subset), "median": median,
                "q25": q25, "q75": q75,
            })
    for model, column in (("A", "a_rmse_md"), ("C", "c_rmse_md")):
        median, q25, q75 = quantile_summary(ac[column])
        summary_rows.append({
            "comparison": "A_vs_C", "depth_band": "all", "model": model,
            "metric": "loo_prediction_rmse_raw_md", "n_participants": len(ac),
            "median": median, "q25": q25, "q75": q75,
        })
    for metric in ("a_minus_c_rmse_md", "a_minus_c_outer_burden_pp"):
        median, q25, q75 = quantile_summary(ac[metric])
        summary_rows.append({
            "comparison": "A_vs_C", "depth_band": "outer" if "burden" in metric else "all",
            "model": "A_minus_C", "metric": metric,
            "n_participants": len(ac), "median": median, "q25": q25, "q75": q75,
        })
    summary_table = pd.DataFrame(summary_rows)
    summary_table.to_csv(output_dir / "loo_model_comparison_summary.csv", index=False)

    outer_ab = paired.loc[paired.depth_band.eq("outer")].copy()
    outer_direction = {
        model.upper(): {
            "n_positive_tail_gt_negative_tail": int((
                outer_ab[f"{model}_positive_tail_fraction"]
                > outer_ab[f"{model}_negative_tail_fraction"]
            ).sum()),
            "n_equal": int((
                outer_ab[f"{model}_positive_tail_fraction"]
                == outer_ab[f"{model}_negative_tail_fraction"]
            ).sum()),
            "positive_minus_negative_pp_median_q25_q75": list(quantile_summary(
                outer_ab[f"{model}_positive_minus_negative_pp"]
            )),
        }
        for model in ("a", "b")
    }
    report = {
        "status": "complete",
        "mode": "full" if expected == N_NODES else "smoke",
        "reference_n": 37,
        "n_nodes": expected,
        "n_depths": N_DEPTHS,
        "models": {
            "A": "OLS raw MD ~ categorical sex + local FreeSurfer thickness",
            "B": "PCNToolkit 1.3.0 homoskedastic WarpSinhArcsinh BLR; same covariates",
            "C": "OLS raw MD ~ categorical sex",
        },
        "age_use": "reference selection only",
        "pairing": "same held-out participant, same training observations, pairwise common valid positions",
        "warped_z": "held-out predictive percentile mapped to standard-normal z by model.forward",
        "attempted_warped_fits": int(sum(x["attempted_warped_fits"] for x in status)),
        "successful_warped_fits": int(sum(x["successful_warped_fits"] for x in status)),
        "failed_warped_fits": int(sum(x["failed_warped_fits"] for x in status)),
        "warning_warped_fits": int(sum(x["warning_warped_fits"] for x in status)),
        "outer_tail_direction": outer_direction,
        "pcntoolkit_version": pcntoolkit.__version__,
        "python": platform.python_version(),
        "outputs": {
            "a_vs_b_participant_depth_band": "a_vs_b_participant_depth_band.csv",
            "a_vs_c_participant_paired": "a_vs_c_participant_paired.csv",
            "summary": "loo_model_comparison_summary.csv",
        },
    }
    write_json(output_dir / "loo_model_comparison_report.json", report)
    return report


def main() -> None:
    args = parse_args()
    if args.z_threshold <= 0 or args.minimum_training_n <= 3:
        raise ValueError("Invalid threshold or minimum training N")
    if args.action == "task":
        if args.node_index is None:
            raise ValueError("--node-index is required for --action task")
        report = run_node(args, args.node_index, args.output_dir)
        print(json.dumps({k: report.get(k) for k in (
            "status", "node_index", "attempted_warped_fits",
            "successful_warped_fits", "failed_warped_fits", "elapsed_seconds",
        )}))
        return
    if args.action == "smoke":
        smoke_output = args.output_dir
        nodes = pd.read_csv(args.smoke_node_list).node_index.astype(int).tolist()[:args.smoke_nodes]
        if len(nodes) != args.smoke_nodes:
            raise RuntimeError("Insufficient smoke nodes")
        for node in nodes:
            run_node(args, node, smoke_output)
        report = summarize(args, smoke_output)
    else:
        report = summarize(args, args.output_dir)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
