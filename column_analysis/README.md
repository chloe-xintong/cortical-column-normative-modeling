# One-column-per-region baseline

This task creates one baseline row for each subject, hemisphere, and cortical
region.  The row contains the mean MD at each of the 21 sampled cortical
depths and regional cortical-thickness summaries.

Depth 0 is the white-matter surface and depth 20 is the pial surface, matching
the existing profile-sampling convention.  MD values are averaged over valid
native columns within each subject, hemisphere, and region.  The script also
reports FreeSurfer vertex thickness sampled at the white-surface anchor and
the length of the sampled trajectory as a separate thickness check.

The script validates that each region trajectory contains exactly
`n_columns * 21` points for the corresponding profile table.  Cases that fail
validation are written to a separate failure CSV and are not silently included.

Example for one subject:

```bash
python column_analysis/one_column_per_region_baseline.py \
  --analysis-root /mnt/newStor/paros/paros_WORK/xintong/results/column_analysis \
  --freesurfer-root /mnt/newStor/paros/paros_WORK/xintong/data/processed/t1_freesurfer \
  --output-dir /mnt/newStor/paros/paros_WORK/xintong/results/column_analysis/one_column_per_region_baseline \
  --subject-id D0097
```

Outputs are:

- `one_column_per_region_baseline.csv`
- `one_column_per_region_baseline_failures.csv`
