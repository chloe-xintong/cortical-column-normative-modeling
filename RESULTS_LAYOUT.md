# Results layout

The external results root is:

`/mnt/newStor/paros/paros_WORK/xintong/results`

## Current production result

- `fsaverage_result/columns/`: the only current formal result, containing the
  template-driven cortical columns and profiles.

## Native-space legacy results

- `native_unmapped_column_results/column_analysis/`: older native-column
  results that are not mapped to a template.
- `native_unmapped_column_results/column_profiles/`: older native profile
  outputs.

The top-level `column_analysis` and `column_profiles` paths are compatibility
links to these native-space directories. They are retained so older scripts and
manifests do not silently break.

## QC

- `qc/template_mapping/`: template mapping and label-agreement QC.
- `qc/o2_fs/`: FreeSurfer depth-map resampling outputs and validation tables.
- `qc/template_registration_comparison/o2_column/fsaverage/`: the combined
  50-subject fsaverage O2-Column profiles and QC sidecars.
- `qc/o3_coordinate/`: O3-Coordinate outputs, registration overlays, and QC.
- `qc/native_column/`: native-column collision and spatial-similarity QC.

## Archive

`archive/` contains superseded experiments and historical outputs. It is not a
source for the current formal production analysis.
