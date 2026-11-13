# Cortical Trajectory Analysis

This repository implements a template-driven cortical-column pipeline for
sampling diffusion MRI across cortical depth. It covers subject inventory,
FreeSurfer reconstruction, DWI preprocessing, DWI-to-T1 registration,
column construction, multi-metric profile sampling, quality control, and the
analyses used for Figures 2, 4, and 5.

## Pipeline overview

```text
raw T1 ──> FreeSurfer surfaces and parcellation ─┐
                                                 ├─> template-driven columns
raw DWI ──> preprocessing and tensor metrics ────┤   in native T1 space
                                                 └─> DWI-to-T1 registration
                                                          │
                                                          v
                                              depth profiles and regional means
```

The current column definition is target-driven. For every vertex on the
selected FreeSurfer template (`fsaverage` or `fsaverage6`), the nearest point
on the subject's registered spherical mesh is found with
`vtkStaticCellLocator`. The source triangle and barycentric weights are then
applied to the native white and pial surfaces to create a subject-specific
column.

Each column has 21 points in **pial-to-white** order:

```text
depth_index 0   / depth_fraction 0.00 = pial surface
depth_index 20  / depth_fraction 1.00 = white surface (GM/WM boundary)
```
<img width="1061" height="556" alt="image" src="https://github.com/user-attachments/assets/8aee232d-3cfe-46f8-8629-3f20b9f35201" />



## Main entry points

| Task | Entry point |
| --- | --- |
| Complete single-subject workflow | `00_run_pipeline/run_subject.py` |
| Preprocessing only | `00_run_pipeline/run_preprocess.py` |
| Columns only, with optional metric sampling | `00_run_pipeline/run_column_calculation.py` |
| Analysis from existing FreeSurfer and DWI metrics | `00_run_pipeline/run_analysis.py` |
| ADRC batch processing | `submit_adrc_template_columns_all.sbatch` |
| Build template columns directly | `03_column/build_template_columns.py` |
| Check required FreeSurfer outputs | `qc/check_freesurfer_outputs.py` |
| Check generated columns | `qc/check_columns.py` |
| Full-cohort preflight and report | `qc/submit_full_cohort.sh` |

The complete runner automatically selects ADRC preprocessing for `Dxxxx`
subjects and ADD preprocessing for `Sxxxx` subjects. Example:

```bash
python 00_run_pipeline/run_subject.py \
    --subject-id D0007 \
    --t1 /path/to/T1.nii.gz \
    --dwi /path/to/dwi.nii.gz \
    --bvec /path/to/D0007.bvec \
    --bval /path/to/D0007.bval \
    --reverse-phase /path/to/reverse_b0.nii.gz \
    --pe-dir PA \
    --readout-time 0.076191 \
    --output-dir /path/to/output/D0007
```

Use `--skip-freesurfer` or `--skip-dwi-preprocess` to resume from existing
outputs. Run `python 00_run_pipeline/run_subject.py --help` for all cohort-specific
options.

## Calculate columns without requiring MD

The short runner stops after the two fsaverage-indexed hemisphere column
files. It needs an existing FreeSurfer reconstruction but does not require a
DWI or MD image:

```bash
python 00_run_pipeline/run_column_calculation.py \
    --subject-id D0007 \
    --subjects-dir /path/to/freesurfer_subjects \
    --output-dir /path/to/column_output/D0007
```

To sample an optional scalar image, add `--metric-image`, `--metric-name`,
`--registration`, and `--dwi-reference`. The image can be any 3D metric in
the registration-reference grid; it is not restricted to MD.

## Run analysis from prepared inputs

When FreeSurfer reconstruction and tensor metrics already exist:

```bash
python 00_run_pipeline/run_analysis.py \
    --subject-id D0007 \
    --subjects-dir /path/to/freesurfer_subjects \
    --md-image /path/to/md.nii.gz \
    --fa-image /path/to/fa.nii.gz \
    --ad-image /path/to/ad.nii.gz \
    --rd-image /path/to/rd.nii.gz \
    --registration-reference /path/to/masked_dwi.nii.gz \
    --analysis-dir /path/to/analysis/D0007
```

MD is required; FA, AD, and RD are optional. The analysis runner performs
registration, column generation, metric sampling, shared-mask cleaning, and
MD aggregation over the template Desikan--Killiany regions.

The output layout is:

```text
<analysis_dir>/
├── registration/             # bbregister DWI-to-T1 transform
├── columns/                  # hemisphere NPZ files and mapping-QC JSON
├── profiles_before_clean/    # raw metric profiles
├── profiles/                 # profiles after the shared validity mask
└── regional_profiles/
    ├── region_mean/          # 68 regional MD profiles
    └── <subject>_profiles_QA_md.csv
```

Column NPZ files retain template vertex and annotation identifiers, native
triangle vertices, barycentric weights, pial/white endpoints, 21-point
coordinates, cortical thickness, column length, mapping errors, and validity
masks. Profile NPZ files retain sampled values and DWI voxel coordinates so
that mapping and sampling decisions remain auditable.

## Data preparation and preprocessing

ADRC input inventories are built with
`data_preparation/data_inventory/build_adrc_data_inventory.py`. The inventory
records T1, DWI, gradient, reverse-b0, and JSON inputs consumed by the batch
runner.

FreeSurfer reconstruction can also be run directly:

```bash
bash 01_preprocessing/t1/run_freesurfer.sh \
    <subject_id> <t1_image> <subjects_dir> [threads]
```

DWI preprocessing is coordinated by `00_run_pipeline/run_preprocess.py` and delegates
to the protocol-specific workflows under `01_preprocessing/dwi/ADRC/` and
`01_preprocessing/dwi/ADD/`. Expected tensor products include MD/ADC, FA, AD, and
RD images.

The pipeline expects the required FreeSurfer, FSL, MRtrix, ANTs, and Python
environments to be loaded by the caller. Python analysis code uses packages
including NumPy, SciPy, pandas, nibabel, VTK, scikit-learn, matplotlib, and
seaborn; individual workflows may use only a subset.

## Quality control and cohort execution

Current QC entry points and full-cohort tooling live under [`qc/`](qc/README.md):

- `qc/check_freesurfer_outputs.py` validates surfaces, thickness, and labels.
- `qc/check_columns.py` validates template-column geometry and mapping.
- `qc/option3/` contains alignment QC for the active template-driven method.
- `qc/full_cohort/` contains preflight and cohort-report scripts.
- `qc/archive/` contains superseded experiments and validation workflows.

Frozen cohort lists and exclusion tables are stored in `manifests/`. See
[`qc/FULL_COHORT_EXECUTION.md`](qc/FULL_COHORT_EXECUTION.md) for the cohort
execution procedure.

## Figure analyses

Figure-specific code, notebooks, configurations, and SLURM scripts are grouped
under [`analysis_results/`](analysis_results/README.md):

- `figure2/`: mapping validation, method comparisons, curvature, and
  registration QC.
- `figure4/`: spatial similarity, null models, correlation length, and matched
  cortical-boundary analyses.
- `figure5/`: age-prediction feature construction, model evaluation, and
  ablation analyses.
- `cohort_audit/`: cross-figure participant flow and exclusion auditing.

The former `qc/figure4`, `qc/figure5`, and
`qc/template_registration_comparison` paths are compatibility symlinks.

## Repository layout

| Directory | Purpose |
| --- | --- |
| `01_preprocessing/` | Cohort-specific DWI and T1 workflows |
| `02_registration/` | DWI-to-T1 registration |
| `03_column/` | Template-to-subject mapping and cortical-column construction |
| `04_profile_sampling/` | T1-to-DWI coordinate conversion and optional metric sampling |
| `00_run_pipeline/` | Numbered `run_*.py` workflow entry points |
| `qc/` | Active QC, cohort execution, and archived validation |
| `analysis_results/` | Figure analyses and cohort audit code |
| `manifests/` | Frozen subject lists and QC inclusion/exclusion tables |
| `visualization/` | Column and method-figure visualization utilities |
| `presentations/` | Reproducible presentation-building scripts and assets |

External result locations and the distinction between current production and
legacy native-space outputs are documented in
[`RESULTS_LAYOUT.md`](RESULTS_LAYOUT.md). Large imaging data, generated outputs,
logs, and subject-level surface files are intentionally excluded from Git.
