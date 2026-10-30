# O2-FS: FreeSurfer depth-map resampling

This directory contains the restored O2-FS implementation.

`run_vol2surf.py` runs `mri_vol2surf` independently at 21 cortical depth
fractions and validates that each output contains the complete target surface.
The method label is **O2-FS**, or **FS DepthMap** in figures.

Project depth is stored as pial -> white: depth 0.00 is pial and depth 1.00
is white. Because FreeSurfer `--projfrac` is white -> pial, the runner passes
`--projfrac 1 - depth` and records both values in `vol2surf_manifest.json`.

Submission scripts are kept with this method:

- `submit_o2_fs_fsaverage_additional_19.sbatch`: earlier 19-subject pilot;
- `submit_o2_fs_fsaverage_50.sbatch`: current 50-subject fsaverage run.

Submit the current run from the repository root with:

```bash
sbatch analysis_results/figure2/o2_fs/submit_o2_fs_fsaverage_50.sbatch
```

O2-FS outputs are retained under `results/qc/o2_fs/` for method validation and
are not part of the formal production result. The formal result is under
`results/fsaverage_result/columns/`.
