# O3 full-cohort QC

`scripts/compute_option3_qc.py` reads existing fsaverage O3 columns and MD
profiles. It
does not rebuild columns, resample MD, overwrite transforms, or modify any
production output.

The cached subject-level outputs cover:

- spherical mapping and barycentric weights;
- mapping, geometry, and complete-profile coverage;
- zero-length columns and pial-to-white coordinate construction;
- DWI voxel-grid bounds and MD validity at all 21 depths;
- native triangle label composition and fsaverage DK-boundary distance;
- column length versus interpolated FreeSurfer thickness;
- existing bbregister cost and registration-review inputs.

The SLURM array writes only to `subjects/<subject_id>/`. Cohort tables are made
after the array by `scripts/summarize_option3_qc.py`, including explicit failed
and
incomplete subject lists.

The notebook under `notebooks/` is not executed by any submission script. Run
it manually after reviewing `full_cohort_status.csv`.

## Revised alignment report

Compute all metrics and figures outside the notebook:

```bash
python qc/option3/scripts/run_option3_alignment_qc.py
```

Use `--manifest`, `--subjects-dir`, `--output-dir`, or `--covariates` to override
inputs. Validate a small run with `--limit 1 --output-dir /tmp/option3_qc_check`.
The script reads production data without modifying it and writes CSV tables,
publication figures, and a completion metadata file. Missing and failed
subjects remain explicit. The default destination is
`/mnt/newStor/paros/paros_WORK/xintong/results/qc/full_cohort_spatial_scale/option3_alignment_qc_v2/`.

`notebooks/option3_alignment_qc_v2.ipynb` is an English-only summary notebook.
It reads the precomputed tables, performs only lightweight presentation
summaries, and contains the plotting code for the cohort-level QC figures. It
does not load production surfaces or profiles or recompute subject-level QC.
Surface and volume overlays produced by the external script are displayed as
cached images. Before the external calculation finishes, pending tables and
figures are reported without stopping notebook execution. Registration manual-review
records can be supplied in `registration_manual_review.csv` in that directory
before rerunning the script. Missing boundary-offset measurements and covariates
remain explicit rather than being treated as passing QC.
