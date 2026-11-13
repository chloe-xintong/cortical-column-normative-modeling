# Results layout

## Generated analysis results

The new results directory is a sibling of `code/`:

`/mnt/newStor/paros/paros_WORK/xintong/result/cortical_trajectory_analysis/`

Generated artifacts formerly stored inside this repository now mirror their
relative source layout beneath that directory, for example:

```text
result/cortical_trajectory_analysis/
├── analysis_results/
│   ├── figure4/outputs/
│   ├── figure5/outputs/
│   ├── figure6/
│   └── figure7/
└── visualization/outputs/
```

Scripts, notebooks, configuration, and documentation remain in the repository.
Moved result locations have local compatibility symlinks excluded by
`.gitignore`. Explicit source paths are updated where identifiable; computed
legacy paths still resolve through these links. Keep the links when running
historical notebooks. These links are local conveniences, not portable Git
artifacts. On another checkout, supply the external output path using each
script's existing CLI or recreate the links for historical notebooks.

Migration mapping: `result/cortical_trajectory_analysis/migration_manifest.json`.
No generated data were deleted. Existing result metadata retain their original
provenance paths, which continue to resolve through compatibility links.

## Existing production imaging and legacy results

The previously established external results root remains:

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
