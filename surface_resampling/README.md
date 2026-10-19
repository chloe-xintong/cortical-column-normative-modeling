# Template registration comparison

The production cortical-column pipeline is implemented in
`trajectory_processing/` and `profile_sampling/`. This directory contains
only the downstream template-method comparison analyses.

The retained summary notebooks are:

- `template_registration_comparison/notebooks/compare_md_interpolation_order_fsaverage.ipynb`
- `template_registration_comparison/notebooks/compare_method_regional_profiles_fsaverage.ipynb`
- `template_registration_comparison/notebooks/compare_o3_template_resolution.ipynb`

Expensive calculations are implemented in
`template_registration_comparison/py/`; the notebooks read those generated
tables and profiles and produce summary figures.

The three comparisons are:

1. O2-Column versus O3-Coordinate across cortical depth;
2. regional profile comparison for Native, FS DepthMap, Profile Interp., and
   Coordinate Interp.;
3. fsaverage versus fsaverage6 template resolution for O3-Coordinate.

All large outputs are stored outside the repository under the configured
results directory.
