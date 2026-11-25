#!/usr/bin/env python3
"""Fit the selected full-cohort regional BLR and export Figure 3-linked centile curves.

Description
-----------
Fit the selected full-cohort regional BLR and export Figure 3-linked centile curves.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--subject-order`` command-line input
- ``--cohort-audit`` command-line input
- ``--cohort-manifest`` command-line input
- ``--regional-profile`` command-line input
- ``--regional-thickness`` command-line input
- ``--profile-metadata`` command-line input
- ``--thickness-metadata`` command-line input
- ``--figure3-ranking`` command-line input
- ``--output-dir`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure6/scripts/fit_selected_centiles.py --help
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pcntoolkit import BLR
from scipy.stats import norm

from pilot_utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-order", type=Path, required=True)
    parser.add_argument("--cohort-audit", type=Path, required=True)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--regional-profile", type=Path, required=True)
    parser.add_argument("--regional-thickness", type=Path, required=True)
    parser.add_argument("--profile-metadata", type=Path, required=True)
    parser.add_argument("--thickness-metadata", type=Path, required=True)
    parser.add_argument("--figure3-ranking", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selected-model", choices=["M0", "M2"], default="M2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subject_order = pd.read_csv(args.subject_order, dtype={"subject_id": str})
    audit = pd.read_csv(args.cohort_audit, dtype={"subject_id": str}).query("included == True")
    cohort = pd.read_csv(args.cohort_manifest, dtype={"subject_id": str})
    subjects = audit.merge(subject_order[["subject_id", "row_index"]], on="subject_id", validate="one_to_one").merge(
        cohort[["subject_id", "SUBJECT_SEX"]], on="subject_id", validate="one_to_one"
    )
    profiles = np.load(args.regional_profile, mmap_mode="r")
    thickness = np.load(args.regional_thickness, mmap_mode="r")
    profile_meta = pd.read_csv(args.profile_metadata)
    thickness_meta = pd.read_csv(args.thickness_metadata)
    selected_names = pd.read_csv(args.figure3_ranking).query("selected_for_figure3 == True").region_name.tolist()
    selected_regions = thickness_meta.index[thickness_meta.dk_label_name.isin(selected_names)].tolist()
    if len(selected_regions) != 6:
        raise RuntimeError("Expected both hemispheres for the three frozen Figure 3 regions")
    representative_depths = [0.20, 0.50, 0.80]
    subject_rows = subjects.row_index.to_numpy(int)
    age_mean, age_scale = subjects.age.mean(), subjects.age.std(ddof=0)
    sex_weights = subjects.SUBJECT_SEX.value_counts(normalize=True).sort_index().to_dict()
    grid_age = np.arange(40.0, 80.0, 0.5)
    be_train = xr.DataArray(np.zeros((len(subjects), 1), dtype=int), dims=("observations", "batch_effect_dims"))
    be_maps = {"dummy_batch_effect": {"0": 0}}
    rows: list[dict[str, object]] = []
    for region_index in selected_regions:
        region = thickness_meta.iloc[region_index]
        local_thickness = np.asarray(thickness[subject_rows, region_index], float)
        thickness_mean, thickness_scale = local_thickness.mean(), local_thickness.std(ddof=0)
        x_parts = [((subjects.age.to_numpy(float) - age_mean) / age_scale)[:, None],
                   (subjects.SUBJECT_SEX.to_numpy(int) == 2).astype(float)[:, None]]
        if args.selected_model == "M2":
            x_parts.append(((local_thickness - thickness_mean) / thickness_scale)[:, None])
        x_train = np.column_stack(x_parts)
        feature_start = region_index * 21
        y = np.asarray(profiles[subject_rows, feature_start:feature_start + 21], float)
        for depth in representative_depths:
            depth_index = int(round(depth * 20))
            observed = y[:, depth_index]
            y_mean, y_scale = observed.mean(), observed.std(ddof=0)
            blr = BLR(name=f"selected_{args.selected_model}_r{region_index}_d{depth_index}",
                      heteroskedastic=False, ard=False, optimizer="l-bfgs-b", n_iter=100, tol=1e-3)
            blr.fit(xr.DataArray(x_train, dims=("observations", "covariates")), be_train, be_maps,
                    xr.DataArray((observed - y_mean) / y_scale, dims=("observations",)))
            for subject_i, subject in subjects.reset_index(drop=True).iterrows():
                rows.append({"row_type": "observed", "selected_model": args.selected_model,
                             "subject_id": subject.subject_id, "age": float(subject.age),
                             "sex_code": int(subject.SUBJECT_SEX), "hemisphere": region.hemisphere,
                             "dk_label_id": int(region.dk_label_id), "dk_label_name": region.dk_label_name,
                             "depth": depth, "local_thickness": float(local_thickness[subject_i]),
                             "observed_md": float(observed[subject_i]), "pred_median": np.nan,
                             "pred_p10": np.nan, "pred_p90": np.nan})
            conditional_mean: list[np.ndarray] = []
            conditional_var: list[np.ndarray] = []
            weights: list[float] = []
            for sex_code in (1, 2):
                curve_parts = [((grid_age - age_mean) / age_scale)[:, None],
                               np.full((len(grid_age), 1), float(sex_code == 2))]
                if args.selected_model == "M2":
                    curve_parts.append(np.zeros((len(grid_age), 1)))
                mean_scaled, var_scaled = blr.ys_s2(np.column_stack(curve_parts), np.zeros((len(grid_age), 1), dtype=int))
                conditional_mean.append(mean_scaled * y_scale + y_mean)
                conditional_var.append(var_scaled * y_scale ** 2)
                weights.append(float(sex_weights[sex_code]))
            weights_array = np.asarray(weights)[:, None]
            mean_array = np.asarray(conditional_mean)
            var_array = np.asarray(conditional_var)
            marginal_mean = np.sum(weights_array * mean_array, axis=0)
            marginal_var = np.sum(weights_array * (var_array + np.square(mean_array)), axis=0) - np.square(marginal_mean)
            for age_i, age in enumerate(grid_age):
                rows.append({"row_type": "curve", "selected_model": args.selected_model, "subject_id": "",
                             "age": age, "sex_code": "empirical_marginal", "hemisphere": region.hemisphere,
                             "dk_label_id": int(region.dk_label_id), "dk_label_name": region.dk_label_name,
                             "depth": depth, "local_thickness": thickness_mean, "observed_md": np.nan,
                             "pred_median": float(marginal_mean[age_i]),
                             "pred_p10": float(marginal_mean[age_i] + norm.ppf(.10) * np.sqrt(marginal_var[age_i])),
                             "pred_p90": float(marginal_mean[age_i] + norm.ppf(.90) * np.sqrt(marginal_var[age_i]))})
    pd.DataFrame(rows).to_csv(args.output_dir / "selected_region_centile_data.csv", index=False)
    write_json(args.output_dir / "selected_centile_metadata.json", {
        "selected_model": args.selected_model, "selected_region_names": selected_names,
        "region_source": str(args.figure3_ranking), "representative_depths": representative_depths,
        "age_grid": [40.0, 79.5, 0.5],
        "sex_curve_handling": f"Moment-marginalized with empirical weights {sex_weights}; code labels are not interpreted as male/female.",
        "thickness_curve_handling": "Held at the regional cohort mean for M2.",
        "interval": "10th-90th Gaussian predictive interval using full PCNToolkit variance, with between-sex mean variation included.",
    })
    print(json.dumps({"status": "complete", "selected_model": args.selected_model, "rows": len(rows)}))


if __name__ == "__main__":
    main()
