# Native cortical columns to fsaverage correspondence

`native_columns_to_fsaverage.py` creates the master spatial identity table for
native cortical columns. It does **not** resample, interpolate, average, or
modify MD values. It does not modify existing profile CSVs, MAT files, or
cleaning masks.

## Input layout

- `--subjects-dir/<subject>/surf/{lh,rh}.{white,pial,sphere.reg}`
- `--subjects-dir/<subject>/label/{lh,rh}.aparc.annot`
- `--fsaverage-dir/surf/{lh,rh}.{white,pial,sphere.reg}`
- `--profiles-root/<subject>/profiles/<subject>_<hemi>_<region>_{md,fa,ad,rd}.csv`
- `--profiles-root/<subject>/profiles/bad_trajectory_mask.json`

The repository's cleaning output is named `bad_trajectory_mask.json`, rather
than `bad_columns_master.json`. Its keys are `lh_<region>` or `rh_<region>` and
its `bad_indices` are 0-based region-row indices before cleaning.

## Row reconstruction

The current primary pipeline is template-driven. It creates one column for
each fsaverage vertex and stores the original source triangle and barycentric
mapping provenance in the column NPZ files. Older region-row reconstruction
described below applies only to legacy native-column outputs.

For a legacy region output, native IDs were recovered as:

```python
native_vertex_ids = np.where(labels == region_index)[0]
```

The mask is applied to these original region rows. The program stops if:

```
cleaned MD rows != original region vertices - bad rows
```

Each output row preserves the original region row. Coordinates and template
identity are written once per column, followed by `MD_depth_*`, `FA_depth_*`,
`AD_depth_*`, and `RD_depth_*`. Invalid/deleted columns are retained with
`column_valid=False`, `cleaned_region_row=-1`, and all metric depths set to
NaN. The same mask must recover the same retained row count in every metric.

## Coordinates and mapping

Native and fsaverage `sphere.reg` coordinates are normalized to unit spheres.
A `scipy.spatial.cKDTree` nearest-neighbour query maps every native vertex to
one fsaverage vertex of the same hemisphere. `registered_sphere_*` and
`template_sphere_*` are the normalized unit-sphere coordinates. White, pial,
and optional inflated coordinates retain FreeSurfer surface-RAS coordinates.

This is a many-to-one lookup only: collisions are retained, and no MD values
or other metric values are changed or resampled. It is not a cross-subject
physical-coordinate model.

## Dry run on one subject

```bash
python surface_alignment/native_columns_to_fsaverage.py \
  --subject-id D0007 \
  --subjects-dir /path/to/t1_freesurfer \
  --fsaverage-dir /path/to/fsaverage \
  --profiles-root /path/to/results/column_profiles \
  --output-root /path/to/results/native_column_fsaverage \
  --dry-run
```

Remove `--dry-run` only after the row checks pass. Add `--include-inflated` if
the fsaverage inflated surface is wanted in the table.

## Batch run

`subjects.txt` may contain one ID per line, or a first CSV column of IDs.

```bash
python surface_alignment/native_columns_to_fsaverage.py \
  --subjects-file subjects.txt \
  --subjects-dir /path/to/t1_freesurfer \
  --fsaverage-dir /path/to/fsaverage \
  --profiles-root /path/to/results/column_profiles \
  --output-root /path/to/results/native_column_fsaverage
```

Outputs are compressed per `subject × hemisphere × region` CSV files plus
subject- and region-level QC CSVs. Existing mapping files are never overwritten
unless `--overwrite` is explicitly supplied.

## Cross-subject spatial coverage QC

`cross_subject_fsaverage_coverage.py` is a separate, non-resampling analysis.
It reports how many subjects and valid native columns map to each
`hemisphere × region × template_vertex_id`, together with collision rates and
incomplete-subject reasons. It reads no metric values.

For a robust Slurm hand-off, submit it after the mapping array with
`afterany`, not `afterok`: it then runs when the array has finished even if a
small number of mapping tasks failed, and records skipped subjects instead of
aborting the cohort result.

## Metric resampling

After correspondence tables exist, resample the four metric profiles onto
fsaverage vertices. The default is a per-vertex, per-depth arithmetic mean;
only rows with `column_valid=True` participate. Native columns mapping to the
same vertex are reported by `n_native_columns` and `collision`.

```bash
python surface_alignment/resample_profiles_to_fsaverage.py \
  --mapping-root /path/to/results/native_column_fsaverage \
  --output-root /path/to/results/fsaverage_profiles \
  --subject-id D0007 --method mean
```

Use `--method median` for robust collision reduction, or `--method retain` to
keep one output row per native column while recording the collision group.
