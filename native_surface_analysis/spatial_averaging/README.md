# Fixed-depth spatial averaging pilot

This analysis compares whole-region, native-surface spatial grouping, and
random grouping for MD at `MD_depth_10` (50% of the 21 sampled depths).
It uses identical 5-fold column-level cross-validation folds for all methods.

The pilot defaults to five representative subjects and four regions. Results
are written outside the repository under `results/native_surface_analysis/
spatial_averaging_pilot/`, split into `tables`, `overlays`, `figures`, and
`logs`.

Spatial groups are generated only from native white-surface mesh adjacency and
ROI geometry. They are not determined from MD. A multi-source shortest-path
assignment provides a surface-contiguous approximation; disconnected ROI
components and group-size/connectedness diagnostics are reported.

Example:

```bash
python native_surface_analysis/spatial_averaging/analyze_spatial_averaging.py \
  --subjects D0231 D0330 D0097 D0388 D0163 \
  --regions precentral superiorfrontal supramarginal superiortemporal \
  --output-dir /mnt/newStor/paros/paros_WORK/xintong/results/native_surface_analysis/spatial_averaging_pilot
```
