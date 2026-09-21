# fsaverage coverage visualization

This module converts existing native-to-fsaverage Option-A mapping tables into
categorical overlays. It does not alter mapping tables or diffusion profiles.

Status values are: 0 = outside the analysis mask, 1 = analysis vertex without
a mapped native column, 2 = exactly one mapped native column, and 3 = collision
(two or more mapped native columns). Coverage percentages use only the analysis
mask. The same output is produced separately for `lh` and `rh`.

Example:

```bash
python visualization/create_fsaverage_coverage_map.py \
  --subject-id D0072 --hemisphere lh \
  --mapping-root /mnt/newStor/paros/paros_WORK/xintong/results/native_column_fsaverage \
  --fsaverage-dir "$SUBJECTS_DIR/fsaverage" \
  --output-dir /mnt/newStor/paros/paros_WORK/xintong/results/fsaverage_coverage_visualization/D0072 \
  --dry-run
```

Remove `--dry-run` to write `.annot`, `.mgh`, and summary `.csv` files. View
the annotation with `freeview -f $SUBJECTS_DIR/fsaverage/surf/lh.inflated:annot=<file>`.
