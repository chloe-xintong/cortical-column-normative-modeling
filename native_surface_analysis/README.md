# Native surface analyses

New analyses built on existing native-column mappings and cleaned MD profiles.
Existing mapping, profiles, and QC results are read-only inputs.

External inputs:
- Native subjects: /mnt/newStor/paros/paros_WORK/xintong/data/processed/t1_freesurfer
- Cleaned profiles: /mnt/newStor/paros/paros_WORK/xintong/results/column_profiles
- fsaverage6 mappings: /mnt/newStor/paros/paros_WORK/xintong/results/native_column_fsaverage6
- FreeSurfer templates: /home/apps/freesurfer/subjects

Each analysis writes to an external result directory with tables, figures,
overlays, and logs. No source data are modified.
