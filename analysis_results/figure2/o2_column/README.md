# O2-Column: native-profile interpolation

`generate_o2_column_profiles.py` restores the O2-Column generation step.
It reads cleaned native MD profiles by native vertex ID and applies the saved
barycentric weights from the validated coordinate mapping.

The legacy native profiles are white -> pial. The generator reverses that
depth axis once on input and saves O2-Column profiles in the project-wide
pial -> white convention, with depth 0.00 at pial and depth 1.00 at white.

The script does not use row order, does not fill missing source profiles with
zero, and writes a complete QC sidecar for every generated profile file.

The supplementary fsaverage run reads the validated triangle and barycentric
weights from the formal subject column files and uses the legacy native profile
mapping only as the native MD source:

```bash
sbatch analysis_results/figure2/o2_column/submit_o2_column_fsaverage_additional_20.sbatch
```

This is a method-comparison workflow. The 50-subject fsaverage outputs are
kept together under:

```text
results/qc/template_registration_comparison/o2_column/fsaverage/<subject>/
```

Each subject directory contains the left- and right-hemisphere profiles and
their QC sidecars. The older fsaverage6 outputs remain separate and are not
part of the current fsaverage comparison. The formal production columns
remain under `results/fsaverage_result/columns/`.
