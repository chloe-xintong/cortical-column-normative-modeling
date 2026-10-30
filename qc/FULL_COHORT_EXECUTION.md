# Full-cohort O3 QC and Figure 4 execution

The workflow is submitted with:

```bash
bash qc/submit_full_cohort.sh
```

The current production O3 directory contains 327 subject directories. The
input inventory currently identifies 325 subjects with every required O3,
FreeSurfer, MD, and registration input. The array length is read from the
manifest rather than hard-coded to 300, so complete available subjects are not
silently discarded.

## Dependency graph

```text
J0  Two-subject end-to-end automatic preflight

J1  Full-cohort O3 QC array              afterok:J0
J2  Full fsaverage geodesic pair tables  afterok:J0
J3  Full-cohort Figure 4 real array      afterok:J2
J4  Full-cohort interpolation null       afterok:J0:J2

J5  O3 QC summary                        afterany:J1
J6  Real curves and length fits          afterany:J3
J7  Null summary                         afterany:J4
J8  Final machine-readable status        afterany:J5:J6:J7
```

There is no 30-subject manual pause. Full arrays can start only after the
two-subject preflight succeeds. Individual array tasks write independent
completion or failure JSON files, and the summaries retain the requested
denominator.

The workflow does not execute either Jupyter notebook. Notebooks are run
manually after the final status tables have been reviewed.
