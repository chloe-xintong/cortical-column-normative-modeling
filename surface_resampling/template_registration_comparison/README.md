# Template registration comparison

This directory contains the formal comparisons of template-aligned cortical MD
profiles. It is separate from the profile-generation pipeline and from the
cohort-level Option 3 QC.

## Contents

- `py/`: calculation and plotting scripts;
- `notebooks/`: summary-only notebooks that read Python-generated tables and
  overlays;
- Slurm submission scripts remain in the repository-level `slurm/` directory.

The main comparisons are:

1. O2-Column versus O3-Coordinate across cortical depth;
2. regional profiles for Native, FS DepthMap, Profile Interp., and Coordinate
   Interp.;
3. fsaverage versus fsaverage6 template resolution for O3-Coordinate.

The notebooks do not regenerate surface mappings, construct columns, or sample
MD volumes. Large outputs are written outside the repository under the paths
specified by the corresponding Slurm scripts.

## Main entry points

```bash
python -u surface_resampling/template_registration_comparison/py/compare_o2_column_o3_coordinate.py
python -u surface_resampling/template_registration_comparison/py/compare_method_regional_profiles.py
python -u surface_resampling/template_registration_comparison/py/compare_o3_template_resolution.py
```

Use the existing Slurm scripts for the complete argument sets:

```bash
sbatch slurm/submit_interpolation_order_analysis.sbatch
sbatch slurm/submit_method_regional_comparison_fsaverage.sbatch
```
