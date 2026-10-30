# bbregister QC visualization package

This package contains the 22 robust bbregister cost outliers and the three
non-outlier subjects whose costs are closest to the full-cohort median
(`0.123435`).

- `cost_outliers/`: filenames begin with `subjectid_cost(value)_`.
- `median_controls/`: filenames begin with `subjectid_control_`.
- Each subject has `orig.mgz`, `dwi_masked.nii.gz`, `DWI2T1.dat`, `lh.white`,
  and `rh.white`.
- `selection_manifest.csv` records the exact cost and group for every subject.

On a local machine with FreeSurfer configured, run:

```bash
bash freeview_commands.sh
```

Freeview opens one subject at a time; close its window to advance to the next
subject. Review whether the white surfaces follow the expected tissue boundary
on the registered diffusion image in axial, coronal, and sagittal views.

For the confirmed D0388 failure, run:

```bash
bash review_D0388.sh
```

This opens a median-cost control first and D0388 second. Yellow contours are
white surfaces and red contours are pial surfaces. Toggle the registered DWI
layer and vary its opacity in all three orthogonal views. D0388 should remain
excluded if the DWI brain outline and cortical ribbon are globally displaced or
scaled relative to the T1 surfaces; isolated susceptibility distortion alone is
not sufficient evidence for a whole-subject exclusion.
