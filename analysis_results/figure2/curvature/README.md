# Native curvature mapping

This is an independent curvature workflow. It is not called by the column or
profile production pipeline.

The native input is the FreeSurfer white-surface curvature file:

```text
<subjects_dir>/<subject>/surf/lh.curv
<subjects_dir>/<subject>/surf/rh.curv
```

`map_barycentric_curvature.py` applies the saved native triangle IDs and
barycentric weights from a validated fsaverage column file. It writes both:

- `signed_mean_curvature`: weighted signed native curvature;
- `absolute_mean_curvature`: weighted mean of absolute native curvature.

The second quantity is defined as:

```text
w1 * abs(H1) + w2 * abs(H2) + w3 * abs(H3)
```

It is not `abs(w1 * H1 + w2 * H2 + w3 * H3)`, so sulcal and gyral signs do
not cancel in the magnitude measure.

`run_o2_fs_curvature.py` is the separate O2-FS reference implementation. It
uses `mri_surf2surf` on the native surface scalar and does not use
`mri_vol2surf`.

To run both curvature mappings for the current 50-subject fsaverage set:

```bash
sbatch analysis_results/figure2/curvature/submit_curvature_fsaverage_50.sbatch
```

Outputs are written separately under:

```text
results/qc/template_registration_comparison/curvature/o2_fs/
results/qc/template_registration_comparison/curvature/shared_barycentric/
```

The barycentric output is shared by O2-Column and O3 because both methods use
the same native triangle and weights for the target vertex. O3 white/pial
endpoints are not used to calculate curvature.
