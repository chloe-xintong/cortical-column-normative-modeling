#!/usr/bin/env python3
"""Fit and score the Figure 6 young-CN linear normative-reference pilot.

The young reference is selected from the pre-age-filter Figure 5/6 candidate
matrix.  At every selected ico4 node and cortical depth the model is

    raw MD ~ categorical sex code + node-local FreeSurfer thickness.

Age is used only to select CN younger than 50 years.  Young-reference
validation is leave-one-subject-out (LOO); the full young reference is then
refit and used to score older CN and the frozen age-supported MCI/AD targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
FEATURE_ROOT = Path(
    "/mnt/newStor/paros/paros_WORK/xintong/results/"
    "figure5_age_prediction/outputs/features"
)
FIGURE6_ROOT = REPO / "analysis_results/figure6/ico4_warped_m2_186"
FIGURE7_ROOT = Path(
    "/mnt/newStor/paros/paros_WORK/xintong/result/"
    "cortical_trajectory_analysis/analysis_results/figure7"
)

DEPTH_BANDS = {
    "outer": tuple(range(0, 7)),
    "middle": tuple(range(7, 14)),
    "inner": tuple(range(14, 21)),
}
SOURCE_IMAGE_EXCLUSIONS = {"D0271", "D0285"}
RETAINED_MCI_PRE_AGE = {
    "D0052", "D0126", "D0129", "D0139", "D0156", "D0183", "D0185",
    "D0190", "D0204", "D0242", "D0243", "D0257", "D0259", "D0290",
    "D0315", "D0316", "D0354", "D0357", "D0435", "D0439",
}
RETAINED_AD_PRE_AGE = {
    "D0029", "D0064", "D0070", "D0083", "D0176", "D0247", "D0370",
}
AGE_SUPPORT_MIN = 41.0
AGE_SUPPORT_MAX = 78.0
N_NODES = 4687
N_DEPTHS = 21
N_FEATURES = N_NODES * N_DEPTHS
PREDICTOR_COUNT = 3  # intercept, sex-code-2 indicator, local thickness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--smoke-nodes", type=int, default=4)
    parser.add_argument("--node-list", type=Path)
    parser.add_argument("--z-threshold", type=float, default=2.0)
    parser.add_argument("--minimum-training-n", type=int, default=10)
    parser.add_argument(
        "--cohort-flow", type=Path,
        default=FIGURE7_ROOT / "inputs/cohort_flow_coverage95_n313.csv",
    )
    parser.add_argument(
        "--subject-order", type=Path,
        default=FEATURE_ROOT / "subject_order.csv",
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
        "--node-depth-metadata", type=Path,
        default=FIGURE6_ROOT / "ico4_node_depth_metadata.csv",
    )
    parser.add_argument(
        "--figure6-smoke-nodes", type=Path,
        default=FIGURE6_ROOT / "smoke_nodes.csv",
    )
    parser.add_argument(
        "--figure7-roster", type=Path,
        default=FIGURE7_ROOT / "inputs/final_analysis_roster.csv",
    )
    parser.add_argument(
        "--figure7-md", type=Path,
        default=FIGURE7_ROOT / "inputs/X_figure7_ico4_md.npy",
    )
    parser.add_argument(
        "--figure7-thickness", type=Path,
        default=FIGURE7_ROOT / "inputs/X_figure7_local_freesurfer_thickness.npy",
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))


def validate_inputs(
    flow: pd.DataFrame,
    source_order: pd.DataFrame,
    source_md: np.ndarray,
    source_thickness: np.ndarray,
    node_meta: pd.DataFrame,
    depth_meta: pd.DataFrame,
    figure7_roster: pd.DataFrame,
    figure7_md: np.ndarray,
    figure7_thickness: np.ndarray,
) -> None:
    if len(flow) != 313 or flow.subject_id.nunique() != 313:
        raise RuntimeError("Expected the coverage>=95% N=313 cohort flow")
    if not flow.passes_minimum_coverage.all() or not flow.option3_qc_pass.all():
        raise RuntimeError("The supplied cohort flow contains a failed coverage/Option-3 row")
    if flow.manual_registration_qc_excluded.any():
        raise RuntimeError("The supplied cohort flow contains a registration-QC exclusion")
    if len(source_order) != 196 or source_order.subject_id.nunique() != 196:
        raise RuntimeError("Expected the pre-age-filter Figure 5/6 N=196 subject order")
    if not np.array_equal(source_order.row_index.to_numpy(), np.arange(196)):
        raise RuntimeError("Figure 5/6 source row_index is not contiguous")
    if source_md.shape != (196, N_FEATURES):
        raise RuntimeError(f"Unexpected Figure 5/6 MD shape: {source_md.shape}")
    if source_thickness.shape != (196, N_NODES):
        raise RuntimeError(f"Unexpected Figure 6 thickness shape: {source_thickness.shape}")
    if len(node_meta) != N_NODES or not np.array_equal(
        node_meta.node_index.to_numpy(), np.arange(N_NODES)
    ):
        raise RuntimeError("Unexpected ico4 node metadata/order")
    if len(depth_meta) != N_FEATURES or not np.array_equal(
        depth_meta.feature_index.to_numpy(), np.arange(N_FEATURES)
    ):
        raise RuntimeError("Unexpected ico4 node-depth metadata/order")
    expected_depths = np.tile(np.arange(N_DEPTHS), N_NODES)
    if not np.array_equal(depth_meta.depth_index.to_numpy(), expected_depths):
        raise RuntimeError("The node-depth metadata is not node-major/depth-minor")
    if figure7_roster.subject_id.nunique() != len(figure7_roster):
        raise RuntimeError("Figure 7 roster subject key is not unique")
    if figure7_md.shape != (len(figure7_roster), N_FEATURES):
        raise RuntimeError(f"Unexpected Figure 7 MD shape: {figure7_md.shape}")
    if figure7_thickness.shape != (len(figure7_roster), N_NODES):
        raise RuntimeError(f"Unexpected Figure 7 thickness shape: {figure7_thickness.shape}")


def build_rosters(
    flow: pd.DataFrame,
    source_order: pd.DataFrame,
    figure7_roster: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = source_order.merge(
        flow, on="subject_id", how="left", validate="one_to_one"
    )
    if source[["age_at_scan_y", "diagnosis_at_scan", "SUBJECT_SEX"]].isna().any().any():
        raise RuntimeError("A Figure 5/6 candidate is missing cohort metadata")
    if not np.allclose(source.age_at_scan_x, source.age_at_scan_y):
        raise RuntimeError("Subject-order and cohort-flow ages disagree")
    source = source.rename(columns={"age_at_scan_y": "actual_age"})
    source = source.drop(columns=["age_at_scan_x"])

    common_qc = (
        source.passes_minimum_coverage
        & source.option3_qc_pass
        & ~source.manual_registration_qc_excluded
        & source.figure5_primary_eligible
        & source.SUBJECT_SEX.isin([1, 2])
        & ~source.subject_id.isin(SOURCE_IMAGE_EXCLUSIONS)
    )
    reference = source.loc[
        common_qc
        & source.diagnosis_at_scan.eq("Normal")
        & source.actual_age.lt(50)
    ].copy()
    reference["reference_or_target"] = "young_reference"
    reference["model_role"] = "young_cn_reference"
    reference["matrix_source"] = "figure5_6"
    reference["matrix_row_index"] = reference.row_index.astype(int)
    reference = reference.sort_values("subject_id").reset_index(drop=True)
    reference.insert(0, "loo_row_index", np.arange(len(reference), dtype=int))
    if len(reference) != 37:
        raise RuntimeError(f"Expected 37 eligible young CN, found {len(reference)}")
    if set(reference.SUBJECT_SEX) != {1, 2}:
        raise RuntimeError("Young reference does not contain both sex-code levels")

    older_cn = source.loc[
        common_qc
        & source.diagnosis_at_scan.eq("Normal")
        & source.actual_age.ge(50)
    ].copy()
    older_cn["reference_or_target"] = "target"
    older_cn["model_role"] = "older_cn_target"
    older_cn["matrix_source"] = "figure5_6"
    older_cn["matrix_row_index"] = older_cn.row_index.astype(int)

    age_supported_mci = RETAINED_MCI_PRE_AGE - {"D0183", "D0290"}
    age_supported_ad = RETAINED_AD_PRE_AGE - {"D0176"}
    target_ids = age_supported_mci | age_supported_ad
    clinical = figure7_roster.loc[figure7_roster.subject_id.isin(target_ids)].copy()
    if set(clinical.subject_id) != target_ids:
        missing = sorted(target_ids - set(clinical.subject_id))
        raise RuntimeError(f"Frozen MCI/AD targets missing from Figure 7 arrays: {missing}")
    if not clinical.age_at_scan.between(
        AGE_SUPPORT_MIN, AGE_SUPPORT_MAX, inclusive="both"
    ).all():
        raise RuntimeError("A frozen MCI/AD target is outside age support")
    expected_dx = {
        **{subject: "MCI" for subject in age_supported_mci},
        **{subject: "AD" for subject in age_supported_ad},
    }
    if any(
        row.diagnosis_at_scan != expected_dx[row.subject_id]
        for row in clinical.itertuples(index=False)
    ):
        raise RuntimeError("A frozen target diagnosis disagrees with Figure 7")
    clinical = clinical.rename(
        columns={"age_at_scan": "actual_age", "analysis_row_index": "matrix_row_index"}
    )
    clinical["reference_or_target"] = "target"
    clinical["model_role"] = clinical.diagnosis_at_scan.map(
        {"MCI": "mci_target", "AD": "ad_target"}
    )
    clinical["matrix_source"] = "figure7"
    clinical["row_index"] = np.nan

    common_columns = sorted(set(older_cn.columns) | set(clinical.columns))
    targets = pd.concat(
        [older_cn.reindex(columns=common_columns), clinical.reindex(columns=common_columns)],
        ignore_index=True,
    ).sort_values(["diagnosis_at_scan", "subject_id"]).reset_index(drop=True)
    targets.insert(0, "scoring_row_index", np.arange(len(targets), dtype=int))
    if targets.subject_id.duplicated().any():
        raise RuntimeError("Target roster contains duplicate subjects")
    counts = targets.diagnosis_at_scan.value_counts().to_dict()
    if counts != {"Normal": 156, "MCI": 18, "AD": 6}:
        raise RuntimeError(f"Unexpected target counts after frozen QC: {counts}")

    candidate_audit = flow.copy()
    candidate_audit["in_pre_age_figure5_6_matrix"] = candidate_audit.subject_id.isin(
        source_order.subject_id
    )
    candidate_audit["source_image_qc_excluded"] = candidate_audit.subject_id.isin(
        SOURCE_IMAGE_EXCLUSIONS
    )
    candidate_audit["young_reference_included"] = candidate_audit.subject_id.isin(
        reference.subject_id
    )
    candidate_audit["young_reference_exclusion_reason"] = ""
    is_cn_young = candidate_audit.diagnosis_at_scan.eq("Normal") & candidate_audit.age_at_scan.lt(50)
    candidate_audit.loc[~is_cn_young, "young_reference_exclusion_reason"] = "not_cn_age_lt_50"
    candidate_audit.loc[
        is_cn_young & ~candidate_audit.in_pre_age_figure5_6_matrix,
        "young_reference_exclusion_reason",
    ] = "not_in_pre_age_figure5_6_primary_candidate_matrix"
    candidate_audit.loc[
        is_cn_young & candidate_audit.SUBJECT_SEX.isin([1, 2]).eq(False),
        "young_reference_exclusion_reason",
    ] = "sex_code_not_1_or_2"
    candidate_audit.loc[
        candidate_audit.young_reference_included, "young_reference_exclusion_reason"
    ] = "included"
    return reference, targets, candidate_audit


def choose_nodes(args: argparse.Namespace, node_meta: pd.DataFrame) -> np.ndarray:
    if args.node_list is not None:
        table = pd.read_csv(args.node_list)
        if "node_index" not in table:
            raise RuntimeError("--node-list must contain node_index")
        nodes = np.sort(table.node_index.astype(int).unique())
    elif args.mode == "full":
        nodes = np.arange(N_NODES, dtype=int)
    else:
        if args.smoke_nodes < 1:
            raise ValueError("--smoke-nodes must be positive")
        smoke = pd.read_csv(args.figure6_smoke_nodes)
        nodes = np.sort(smoke.node_index.astype(int).unique())[: args.smoke_nodes]
    if len(nodes) < 1 or np.any(nodes < 0) or np.any(nodes >= N_NODES):
        raise RuntimeError("Selected node indices are empty or out of range")
    if len(nodes) != len(np.unique(nodes)):
        raise RuntimeError("Selected node list contains duplicates")
    expected = set(node_meta.node_index)
    if not set(nodes).issubset(expected):
        raise RuntimeError("Selected node list is not contained in ico4 metadata")
    return nodes


def extract_target_node(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    targets: pd.DataFrame,
    node_index: int,
    is_thickness: bool,
) -> np.ndarray:
    width = 1 if is_thickness else N_DEPTHS
    result = np.full((len(targets), width), np.nan, dtype=float)
    column = node_index if is_thickness else slice(node_index * N_DEPTHS, (node_index + 1) * N_DEPTHS)
    for source, matrix in (("figure5_6", matrix_a), ("figure7", matrix_b)):
        mask = targets.matrix_source.eq(source).to_numpy()
        rows = targets.loc[mask, "matrix_row_index"].astype(int).to_numpy()
        values = np.asarray(matrix[rows, column], dtype=float)
        if is_thickness:
            values = values[:, None]
        result[mask] = values
    return result[:, 0] if is_thickness else result


def fit_position(
    y: np.ndarray,
    sex_indicator: np.ndarray,
    thickness: np.ndarray,
    minimum_training_n: int,
) -> dict[str, object] | None:
    valid = np.isfinite(y) & np.isfinite(thickness) & np.isfinite(sex_indicator)
    indices = np.flatnonzero(valid)
    if len(indices) < minimum_training_n:
        return None
    design = np.column_stack(
        [np.ones(len(indices)), sex_indicator[indices], thickness[indices]]
    )
    if np.linalg.matrix_rank(design) != PREDICTOR_COUNT:
        return None
    response = y[indices]
    xtx_inverse = np.linalg.inv(design.T @ design)
    beta = xtx_inverse @ design.T @ response
    residual = response - design @ beta
    sse = float(residual @ residual)
    degrees_freedom = len(indices) - PREDICTOR_COUNT
    if degrees_freedom <= 0 or not np.isfinite(sse) or sse <= 0:
        return None
    residual_sd = float(np.sqrt(sse / degrees_freedom))
    leverage = np.einsum("ij,jk,ik->i", design, xtx_inverse, design)
    one_minus_h = 1.0 - leverage
    loo_df = len(indices) - 1 - PREDICTOR_COUNT
    with np.errstate(invalid="ignore", divide="ignore"):
        deleted_sse = sse - residual ** 2 / one_minus_h
        deleted_sd = np.sqrt(deleted_sse / loo_df)
        deleted_prediction_error = residual / one_minus_h
        loo_z = deleted_prediction_error / deleted_sd
    loo_ok = (
        (loo_df > 0)
        & np.isfinite(one_minus_h)
        & (one_minus_h > 1e-10)
        & np.isfinite(deleted_sse)
        & (deleted_sse > 0)
        & np.isfinite(deleted_sd)
        & (deleted_sd > 0)
        & np.isfinite(loo_z)
    )
    full_loo = np.full(len(y), np.nan, dtype=float)
    full_loo[indices[loo_ok]] = loo_z[loo_ok]
    return {
        "beta": beta,
        "residual_sd": residual_sd,
        "n_train": len(indices),
        "rank": PREDICTOR_COUNT,
        "loo_z": full_loo,
        "max_leverage": float(np.max(leverage)),
        "minimum_deleted_residual_sd": float(np.nanmin(deleted_sd[loo_ok]))
        if np.any(loo_ok) else np.nan,
    }


def explicit_loo_check(
    y: np.ndarray,
    sex_indicator: np.ndarray,
    thickness: np.ndarray,
    analytic_loo_z: np.ndarray,
) -> float:
    errors: list[float] = []
    for left_out in np.flatnonzero(np.isfinite(analytic_loo_z))[:5]:
        valid = (
            np.isfinite(y) & np.isfinite(thickness) & np.isfinite(sex_indicator)
        )
        train = valid.copy()
        train[left_out] = False
        design = np.column_stack(
            [np.ones(train.sum()), sex_indicator[train], thickness[train]]
        )
        beta, _, rank, _ = np.linalg.lstsq(design, y[train], rcond=None)
        if rank != PREDICTOR_COUNT:
            continue
        residual = y[train] - design @ beta
        sd = float(np.sqrt(np.sum(residual ** 2) / (train.sum() - PREDICTOR_COUNT)))
        test_x = np.array([1.0, sex_indicator[left_out], thickness[left_out]])
        explicit = float((y[left_out] - test_x @ beta) / sd)
        errors.append(abs(explicit - analytic_loo_z[left_out]))
    return max(errors, default=np.nan)


def main() -> None:
    args = parse_args()
    if args.z_threshold <= 0:
        raise ValueError("--z-threshold must be positive")
    if args.minimum_training_n <= PREDICTOR_COUNT + 1:
        raise ValueError("--minimum-training-n is too small for exact LOO residual SD")
    output = args.output_dir or (Path(__file__).resolve().parent / "outputs" / args.mode)
    output.mkdir(parents=True, exist_ok=True)
    require_files([
        args.cohort_flow, args.subject_order, args.ico4_md, args.local_thickness,
        args.node_metadata, args.node_depth_metadata, args.figure7_roster,
        args.figure7_md, args.figure7_thickness,
    ] + ([args.figure6_smoke_nodes] if args.mode == "smoke" and args.node_list is None else []))

    started = time.monotonic()
    flow = pd.read_csv(args.cohort_flow, dtype={"subject_id": str})
    source_order = pd.read_csv(args.subject_order, dtype={"subject_id": str})
    node_meta = pd.read_csv(args.node_metadata)
    depth_meta = pd.read_csv(args.node_depth_metadata)
    figure7_roster = pd.read_csv(args.figure7_roster, dtype={"subject_id": str})
    source_md = np.load(args.ico4_md, mmap_mode="r")
    source_thickness = np.load(args.local_thickness, mmap_mode="r")
    figure7_md = np.load(args.figure7_md, mmap_mode="r")
    figure7_thickness = np.load(args.figure7_thickness, mmap_mode="r")
    validate_inputs(
        flow, source_order, source_md, source_thickness, node_meta, depth_meta,
        figure7_roster, figure7_md, figure7_thickness,
    )
    reference, targets, candidate_audit = build_rosters(
        flow, source_order, figure7_roster
    )
    nodes = choose_nodes(args, node_meta)
    selected_depth_meta = pd.concat(
        [depth_meta.iloc[node * N_DEPTHS:(node + 1) * N_DEPTHS] for node in nodes],
        ignore_index=True,
    )
    selected_depth_meta.insert(
        0, "output_position", np.arange(len(selected_depth_meta), dtype=int)
    )
    selected_depth_meta["depth_band"] = pd.cut(
        selected_depth_meta.depth_index,
        bins=[-1, 6, 13, 20], labels=["outer", "middle", "inner"],
    ).astype(str)
    selected_depth_meta.to_csv(output / "scored_node_depth_metadata.csv", index=False)
    node_meta.loc[node_meta.node_index.isin(nodes)].to_csv(
        output / "scored_nodes.csv", index=False
    )

    reference.to_csv(output / "young_reference_roster.csv", index=False)
    targets.to_csv(output / "scoring_target_roster.csv", index=False)
    candidate_audit.to_csv(output / "young_reference_candidate_audit.csv", index=False)

    reference_rows = reference.matrix_row_index.astype(int).to_numpy()
    reference_sex = reference.SUBJECT_SEX.eq(2).to_numpy(dtype=float)
    target_sex = targets.SUBJECT_SEX.eq(2).to_numpy(dtype=float)
    n_positions = len(nodes) * N_DEPTHS
    z_path = output / "target_z_young.npy"
    z_map = np.lib.format.open_memmap(
        z_path, mode="w+", dtype=np.float32, shape=(len(targets), n_positions)
    )
    z_map[:] = np.nan
    loo_z_path = output / "young_reference_loo_z_young.npy"
    loo_z_map = np.lib.format.open_memmap(
        loo_z_path, mode="w+", dtype=np.float32,
        shape=(len(reference), n_positions),
    )
    loo_z_map[:] = np.nan

    betas = np.full((n_positions, PREDICTOR_COUNT), np.nan, dtype=float)
    residual_sd = np.full(n_positions, np.nan, dtype=float)
    n_train = np.zeros(n_positions, dtype=np.int16)
    fit_rank = np.zeros(n_positions, dtype=np.int8)
    max_leverage = np.full(n_positions, np.nan, dtype=float)
    min_deleted_sd = np.full(n_positions, np.nan, dtype=float)
    loo_counts = {
        band: {"valid": np.zeros(len(reference), dtype=np.int64),
               "positive": np.zeros(len(reference), dtype=np.int64),
               "negative": np.zeros(len(reference), dtype=np.int64),
               "gt2": np.zeros(len(reference), dtype=np.int64)}
        for band in DEPTH_BANDS
    }
    target_counts = {
        band: {"valid": np.zeros(len(targets), dtype=np.int64),
               "gt2": np.zeros(len(targets), dtype=np.int64)}
        for band in DEPTH_BANDS
    }
    explicit_errors: list[float] = []

    for node_number, node in enumerate(nodes):
        reference_y = np.asarray(
            source_md[reference_rows, node * N_DEPTHS:(node + 1) * N_DEPTHS],
            dtype=float,
        )
        reference_t = np.asarray(source_thickness[reference_rows, node], dtype=float)
        target_y = extract_target_node(
            source_md, figure7_md, targets, node, is_thickness=False
        )
        target_t = extract_target_node(
            source_thickness, figure7_thickness, targets, node, is_thickness=True
        )
        target_design = np.column_stack(
            [np.ones(len(targets)), target_sex, target_t]
        )
        for depth in range(N_DEPTHS):
            position = node_number * N_DEPTHS + depth
            band = next(name for name, values in DEPTH_BANDS.items() if depth in values)
            fitted = fit_position(
                reference_y[:, depth], reference_sex, reference_t,
                args.minimum_training_n,
            )
            if fitted is None:
                continue
            beta = np.asarray(fitted["beta"], dtype=float)
            betas[position] = beta
            residual_sd[position] = float(fitted["residual_sd"])
            n_train[position] = int(fitted["n_train"])
            fit_rank[position] = int(fitted["rank"])
            max_leverage[position] = float(fitted["max_leverage"])
            min_deleted_sd[position] = float(fitted["minimum_deleted_residual_sd"])

            loo_z = np.asarray(fitted["loo_z"], dtype=float)
            loo_z_map[:, position] = loo_z.astype(np.float32)
            loo_valid = np.isfinite(loo_z)
            loo_positive = loo_valid & (loo_z > args.z_threshold)
            loo_negative = loo_valid & (loo_z < -args.z_threshold)
            loo_counts[band]["valid"] += loo_valid
            loo_counts[band]["positive"] += loo_positive
            loo_counts[band]["negative"] += loo_negative
            loo_counts[band]["gt2"] += loo_positive | loo_negative
            if len(explicit_errors) < 20:
                error = explicit_loo_check(
                    reference_y[:, depth], reference_sex, reference_t, loo_z
                )
                if np.isfinite(error):
                    explicit_errors.append(error)

            target_valid = (
                np.isfinite(target_y[:, depth])
                & np.isfinite(target_t)
                & np.all(np.isfinite(target_design), axis=1)
            )
            z = np.full(len(targets), np.nan, dtype=float)
            z[target_valid] = (
                target_y[target_valid, depth]
                - target_design[target_valid] @ beta
            ) / float(fitted["residual_sd"])
            z_map[:, position] = z.astype(np.float32)
            target_counts[band]["valid"] += np.isfinite(z)
            target_counts[band]["gt2"] += np.isfinite(z) & (
                np.abs(z) > args.z_threshold
            )
        z_map.flush()
        loo_z_map.flush()
        print(json.dumps({
            "completed_nodes": node_number + 1,
            "total_nodes": len(nodes),
            "node_index": int(node),
        }), flush=True)
    z_map.flush()
    loo_z_map.flush()

    parameter_table = selected_depth_meta.copy()
    parameter_table["intercept"] = betas[:, 0]
    parameter_table["sex_code_2_coefficient"] = betas[:, 1]
    parameter_table["local_thickness_coefficient"] = betas[:, 2]
    parameter_table["residual_sd"] = residual_sd
    parameter_table["n_training_reference"] = n_train
    parameter_table["design_rank"] = fit_rank
    parameter_table["max_leverage"] = max_leverage
    parameter_table["minimum_loo_training_residual_sd"] = min_deleted_sd
    parameter_table["fit_valid"] = (
        np.all(np.isfinite(betas), axis=1) & np.isfinite(residual_sd) & (residual_sd > 0)
    )
    parameter_table.to_parquet(output / "young_reference_parameters.parquet", index=False)
    np.savez_compressed(
        output / "young_reference_parameters.npz",
        node_index=parameter_table.node_index.to_numpy(int)
        if "node_index" in parameter_table else
        (parameter_table.feature_index.to_numpy(int) // N_DEPTHS),
        depth_index=parameter_table.depth_index.to_numpy(int),
        beta=betas,
        residual_sd=residual_sd,
        n_train=n_train,
        fit_valid=parameter_table.fit_valid.to_numpy(bool),
    )

    participant_keys = [
        "subject_id", "PTID", "PTID_clean", "visit_id", "matched_visit_date"
    ]
    loo_burden = reference[[
        *participant_keys, "actual_age", "diagnosis_at_scan", "SUBJECT_SEX",
        "coverage_fraction", "reference_or_target", "model_role",
    ]].copy()
    target_burden = targets[[
        "scoring_row_index", *participant_keys, "actual_age", "diagnosis_at_scan",
        "SUBJECT_SEX", "coverage_fraction", "reference_or_target", "model_role",
    ]].copy()
    for band in DEPTH_BANDS:
        for frame, counts in (
            (loo_burden, loo_counts[band]), (target_burden, target_counts[band])
        ):
            frame[f"{band}_n_valid_positions"] = counts["valid"]
            frame[f"{band}_n_abs_z_young_gt_2"] = counts["gt2"]
            frame[f"{band}_abs_z_young_gt_2_fraction"] = np.divide(
                counts["gt2"], counts["valid"],
                out=np.full(len(frame), np.nan, dtype=float),
                where=counts["valid"] > 0,
            )
        loo_burden[f"{band}_n_z_young_gt_2"] = loo_counts[band]["positive"]
        loo_burden[f"{band}_z_young_gt_2_fraction"] = np.divide(
            loo_counts[band]["positive"], loo_counts[band]["valid"],
            out=np.full(len(loo_burden), np.nan, dtype=float),
            where=loo_counts[band]["valid"] > 0,
        )
        loo_burden[f"{band}_n_z_young_lt_minus_2"] = loo_counts[band]["negative"]
        loo_burden[f"{band}_z_young_lt_minus_2_fraction"] = np.divide(
            loo_counts[band]["negative"], loo_counts[band]["valid"],
            out=np.full(len(loo_burden), np.nan, dtype=float),
            where=loo_counts[band]["valid"] > 0,
        )
        if not np.array_equal(
            loo_counts[band]["positive"] + loo_counts[band]["negative"],
            loo_counts[band]["gt2"],
        ):
            raise RuntimeError(
                f"{band} positive plus negative LOO tails do not equal absolute tail"
            )
    for frame, all_counts in ((loo_burden, loo_counts), (target_burden, target_counts)):
        total_valid = sum(all_counts[band]["valid"] for band in DEPTH_BANDS)
        total_gt2 = sum(all_counts[band]["gt2"] for band in DEPTH_BANDS)
        frame["n_valid_positions"] = total_valid
        frame["n_abs_z_young_gt_2"] = total_gt2
        frame["abs_z_young_gt_2_fraction"] = np.divide(
            total_gt2, total_valid,
            out=np.full(len(frame), np.nan, dtype=float), where=total_valid > 0,
        )
    loo_burden.to_csv(output / "young_reference_loo_depth_band_burden.csv", index=False)
    target_burden.to_csv(output / "participant_depth_band_burden.csv", index=False)

    cohort_report = {
        "status": "complete",
        "pre_age_figure5_6_candidate_n": int(len(source_order)),
        "coverage95_source_pool_n": int(len(flow)),
        "young_reference_definition": (
            "Normal, actual age <50, Figure5 primary eligible, coverage>=95%, "
            "Option-3 QC pass, registration-QC pass, sex code 1/2, and not D0271/D0285"
        ),
        "young_reference_n": int(len(reference)),
        "young_reference_age_range": [
            float(reference.actual_age.min()), float(reference.actual_age.max())
        ],
        "young_reference_sex_code_counts": {
            str(int(key)): int(value)
            for key, value in reference.SUBJECT_SEX.value_counts().sort_index().items()
        },
        "target_counts": {
            key: int(value)
            for key, value in targets.diagnosis_at_scan.value_counts().items()
        },
        "mci_ad_age_support_inclusive": [AGE_SUPPORT_MIN, AGE_SUPPORT_MAX],
        "source_image_exclusions": sorted(SOURCE_IMAGE_EXCLUSIONS),
        "mci_ad_policy": (
            "Frozen relaxed downstream cohort: D0142 excluded; D0183, D0290, "
            "and D0176 excluded by age support"
        ),
    }
    write_json(output / "cohort_report.json", cohort_report)

    valid_fit = parameter_table.fit_valid.to_numpy(bool)
    run_report = {
        "status": "complete",
        "mode": args.mode,
        "output_dir": str(output.resolve()),
        "model": "raw_MD ~ intercept + I(sex_code == 2) + local_FreeSurfer_thickness",
        "age_use": "reference selection only; age is not a model predictor",
        "young_reference_n": int(len(reference)),
        "target_n": int(len(targets)),
        "target_counts": cohort_report["target_counts"],
        "n_nodes": int(len(nodes)),
        "n_depths": N_DEPTHS,
        "n_positions": int(n_positions),
        "n_valid_full_reference_fits": int(valid_fit.sum()),
        "n_invalid_full_reference_fits": int((~valid_fit).sum()),
        "minimum_training_n": int(args.minimum_training_n),
        "observed_training_n_range": [
            int(n_train[valid_fit].min()), int(n_train[valid_fit].max())
        ] if valid_fit.any() else None,
        "z_definition": (
            "signed (observed raw MD - predicted raw MD) / young-reference training residual SD"
        ),
        "loo_rule": (
            "each young CN is deleted from both coefficient and residual-SD estimation"
        ),
        "z_threshold": float(args.z_threshold),
        "depth_bands": {name: list(indices) for name, indices in DEPTH_BANDS.items()},
        "maximum_analytic_vs_explicit_loo_z_error": float(max(explicit_errors))
        if explicit_errors else None,
        "target_z_map": {
            "path": str(z_path.resolve()),
            "shape": [int(len(targets)), int(n_positions)],
            "dtype": "float32",
            "row_index": "scoring_target_roster.csv: scoring_row_index",
            "column_index": "scored_node_depth_metadata.csv: output_position",
        },
        "young_reference_loo_z_map": {
            "path": str(loo_z_path.resolve()),
            "shape": [int(len(reference)), int(n_positions)],
            "dtype": "float32",
            "row_index": "young_reference_roster.csv: loo_row_index",
            "column_index": "scored_node_depth_metadata.csv: output_position",
            "invalid_positions": "NaN",
        },
        "input_sha256": {
            "cohort_flow": sha256(args.cohort_flow),
            "subject_order": sha256(args.subject_order),
            "node_metadata": sha256(args.node_metadata),
            "node_depth_metadata": sha256(args.node_depth_metadata),
            "figure7_roster": sha256(args.figure7_roster),
        },
        "elapsed_seconds": float(time.monotonic() - started),
    }
    write_json(output / "run_report.json", run_report)
    print(json.dumps(run_report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
