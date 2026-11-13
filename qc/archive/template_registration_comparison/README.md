# Archived template-registration comparisons

These analyses were completed while comparing template-alignment methods.
Option 3 has since been selected for the production column definition, so the
following files are retained as validation evidence and are not current
production code.

The archive is separated by task:

- `interpolation_order/`: depth-wise agreement and supplementary comparisons
  of O2-Column, O3-Coordinate, and the FreeSurfer depth-map reference;
- `regional_profiles/`: regional aggregation and method-profile comparison;
- `template_resolution/`: O3-Coordinate comparison of fsaverage and
  fsaverage6 resolution;
- `option1/`: supplementary Option 1 boundary-disagreement and collision
  profile QC.

Each task keeps its own notebooks and Python scripts. The notebooks are
summary notebooks; expensive calculations and table generation are in the
corresponding `py/` directory. Existing large outputs remain outside the
repository.

## Archived calculation entry points

```bash
python -u qc/archive/template_registration_comparison/interpolation_order/py/compare_o2_column_o3_coordinate.py
python -u qc/archive/template_registration_comparison/regional_profiles/py/compare_method_regional_profiles.py
python -u qc/archive/template_registration_comparison/template_resolution/py/compare_o3_template_resolution.py
```

These commands are preserved for traceability only. They should not be used
to replace the current Option 3 production pipeline without a new review.

The historical Slurm launchers are stored beside their corresponding task:

- `interpolation_order/submit_interpolation_order_analysis.sbatch`;
- `regional_profiles/submit_method_regional_comparison_fsaverage.sbatch`.
