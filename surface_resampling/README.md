# Surface resampling pilot

This is an independent surface-based workflow. It does not modify the native
column pipeline. Tasks 1--4 sample MD to a FreeSurfer template, stack the 21
depths, optionally resample thickness, and validate the resulting profiles.

Example depth list:

```text
0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00
```

All commands require an active FreeSurfer environment. Outputs are written to
an external output directory and should be kept separate for `fsaverage` and
`fsaverage6`.
