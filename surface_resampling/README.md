# Surface resampling pilot

This is an independent surface-based workflow. It does not modify the native
column pipeline. Tasks 1--4 sample MD to a FreeSurfer template, stack the 21
depths, optionally resample thickness, and validate the resulting profiles.

Example depth list:

```text
0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00
```

All commands require an active FreeSurfer environment. Outputs are written to
an external output directory and should be kept separate for `fsaverage` and
`fsaverage6`.

## Method labels

The formal template comparison uses the following display names:

- **Native:** the original native white-to-pial column pipeline.
- **FS DepthMap:** FreeSurfer standard `mri_vol2surf` depth-map sampling and
  template resampling.
- **Profile Interp.:** barycentric interpolation of existing native-column MD
  profiles using validated source vertices and weights.
- **Coordinate Interp.:** barycentric interpolation of native white/pial
  coordinates, construction of a new column, and MD-volume sampling.

The corresponding formal result roots are:

- `o2_fs_volume_resampling`
- `o2_column_native_profiles`
- `o3_coordinate_profiles`

Within the two profile-based roots, the method-specific output directory names
are `o2_column` and `o3_coordinate`. The interpolation-order decision
notebook described below is restricted to `fsaverage` and uses the
common-subject manifest
`manifests/method_comparison_common_30_subjects.csv`.

The earlier native nearest-vertex mapping remains available as a spatial
correspondence result, but it is not one of the three formal comparison
methods above.

## O3 template-resolution comparison

`analysis/py/compare_o3_template_resolution.py` compares the existing
O3-Coordinate profiles on `fsaverage` and `fsaverage6`. It builds a fixed
nearest correspondence from normalized `sphere.reg` directions, summarizes
within-cell variation, compares adjacent profiles, and evaluates spatial
groups matched by approximate white-surface area. The default subject set is
the paired 30-subject manifest
`manifests/method_comparison_common_30_subjects.csv`. It does not rerun volume
sampling or surface mapping.

Run the calculation on a compute node with:

```bash
python -u surface_resampling/analysis/py/compare_o3_template_resolution.py \
    --output-root /mnt/newStor/paros/paros_WORK/xintong/results/o3_template_resolution_comparison
```

The summary notebook is
`analysis/notebooks/compare_o3_template_resolution.ipynb`. It reads the
generated tables and writes comparison figures under the output `figures`
directory.

The paired comparison is run with:

```bash
sbatch slurm/submit_method_comparison_o2_column_o3_coordinate.sbatch
```

The default run uses `fsaverage6`. Set `TEMPLATE=fsaverage` and a separate
`OUTPUT_ROOT` to run the same comparison on `fsaverage`. It writes method QC
and depth-wise tables under the selected external output directory.
Only vertices valid in both methods are included in the paired metrics.

The extended interpolation-order calculation is owned by
`surface_resampling/analysis/py/compare_o2_column_o3_coordinate.py`. Submit the
fsaverage calculation with:

```bash
sbatch slurm/submit_interpolation_order_analysis.sbatch
```

It writes independent outputs under
`results/interpolation_order_comparison_fsaverage_python`, including QC,
depth/profile metrics, cross-subject metrics, fixed spatial-group metrics, and
50% depth difference overlays. The script reuses existing O2/O3 profiles and
fixed group memberships when available; it does not rerun surface mapping or
MD-volume sampling.

The summary-only notebook is
`surface_resampling/analysis/notebooks/compare_md_interpolation_order_fsaverage.ipynb`.
It reads the Python-generated CSV tables and overlays and produces the three
main figures plus supplementary spatial-structure and O2-FS comparisons. The
supplementary calculations are implemented in
`surface_resampling/analysis/py/supplementary_interpolation_order_analysis.py`.
The previous calculation-heavy notebook is preserved as
`surface_resampling/analysis/archive/compare_md_interpolation_order_fsaverage.pre_python_backup.ipynb`.

## Regional profile method comparison

`analysis/py/compare_method_regional_profiles.py` compares four regional
MD profile methods: Native, FS DepthMap, Profile Interp., and Coordinate
Interp. It aggregates fsaverage maps to the 34 cortical DK regions per
hemisphere and reports the resulting 68 bilateral regions across 21 depths.
The formal common-30 run uses
`manifests/method_comparison_common_30_subjects.csv`, which includes D0007 as
the replacement for D0178 because D0178 has no final cleaned native profile
directory.

The formal pair order is the three template methods versus Native, followed by
Coordinate Interp. versus Profile Interp. No additional pairwise comparisons
are included in the formal summary.

Submit the calculation with:

```bash
sbatch slurm/submit_method_regional_comparison_fsaverage.sbatch
```

The summary-only notebook is
`surface_resampling/analysis/notebooks/compare_method_regional_profiles_fsaverage.ipynb`.
Formal outputs are written to
`/mnt/newStor/paros/paros_WORK/xintong/results/method_comparison_regional_profiles_fsaverage_common30`.
The previous output root with the old method labels is retained for traceability
but should not be used for the updated formal figures.

## Option 3 cohort-level QC

`analysis/py/cohort_option3_qc.py` summarizes the existing Option 3 profiles for
the common 30-subject fsaverage cohort. It does not rerun mapping or MD
sampling. The analysis mask is the fsaverage `cortex.label` intersected with
non-excluded DK labels. A profile is formally valid only when all 21 MD values
are finite and nonzero, all 21 sampling points are in bounds, the barycentric
mapping is valid, and the white-to-pial line has nonzero length.

The cohort QC separates zero MD, nonfinite MD, out-of-bounds coordinates,
zero-length geometry, mixed-label mappings, and the combined invalid-profile
flag. It writes depth-wise MD validity tables, masked fsaverage overlays for
these flags, and a surface-hop distance from mixed-label vertices to the
nearest DK label boundary. Values outside the structural analysis mask are
stored as NaN in frequency overlays.

Submit the cohort QC with:

```bash
sbatch slurm/submit_cohort_option3_qc.sbatch
```

The summary notebook is
`analysis/notebooks/cohort_option3_qc.ipynb`. Outputs are written to
`/mnt/newStor/paros/paros_WORK/xintong/results/o3_cohort_qc_fsaverage`,
including subject-by-hemisphere QC, vertex flags, cohort frequency tables,
structural and 95%-reliability masks, FreeSurfer overlays, and cohort QC
figures.

The additional `08_cohort_level_summary.png` merges the two hemispheres only
for presentation. Coverage is vertex-count weighted, zero-length values are
shown as paired full-mesh versus analysis-mask measurements, invalid MD is
reported across depth, and length--thickness correlation is pooled within each
subject. Hemisphere-specific tables and figures remain available for tracing
outliers.

For manual T1--DWI registration checks, create compact DWI-space trajectory
point overlays for the highest-OOB, lowest-coverage, and representative normal
subject-by-hemisphere cases:

```bash
python -u surface_resampling/analysis/py/make_o3_registration_overlays.py \
    --qc-root /mnt/newStor/paros/paros_WORK/xintong/results/o3_cohort_qc_fsaverage \
    --o3-root /mnt/newStor/paros/paros_WORK/xintong/results/o3_coordinate_profiles \
    --output-root /mnt/newStor/paros/paros_WORK/xintong/results/o3_registration_qc_overlays
```

The script does not resample MD. It writes a binary trajectory-point NIfTI, a
depth-coded point NIfTI, the native white/pial surface paths, and a
`freeview_command.sh` for each selected case.
