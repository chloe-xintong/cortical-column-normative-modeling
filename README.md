

## Trajectory Level Data Structure

This project organizes cortical trajectory data into two linked tables. The first table stores trajectory level anatomical and geometric information, while the second table stores depth-resolved imaging measurements sampled along each cortical trajectory. Together, these tables support cortical depth profiling, sampling experiments, and trajectory level graph construction.

## Table 1 — Trajectory Level Information

Each row represents one cortical trajectory.

| Field                | Description                                            |
| -------------------- | ------------------------------------------------------ |
| `subject_id`         | Subject identifier                                     |
| `hemisphere`         | Hemisphere (`lh` / `rh`)                               |
| `trajectory_id`      | Stable trajectory identifier within each hemisphere    |
| `region_label`       | Cortical anatomical region                             |
| `white_x`            | X coordinate at the gray matter/white matter interface |
| `white_y`            | Y coordinate at the gray matter/white matter interface |
| `white_z`            | Z coordinate at the gray matter/white matter interface |
| `pial_x`             | X coordinate at the pial surface                       |
| `pial_y`             | Y coordinate at the pial surface                       |
| `pial_z`             | Z coordinate at the pial surface                       |
| `cortical_thickness` | Cortical thickness associated with the trajectory      |
| `local_surface_area` | Local pial surface area associated with the trajectory |
| `coordinate_space`   | Coordinate space of the trajectory coordinates         |

The trajectory identifier is defined by:

```text
subject_id + hemisphere + trajectory_id
```

Example:

```text
S00775 + lh + 154
```

## Table 2 — Depth-Level Information

Each row represents one sampling point along a cortical trajectory.

| Field              | Description                                        |
| ------------------ | -------------------------------------------------- |
| `subject_id`       | Subject identifier                                 |
| `hemisphere`       | Hemisphere (`lh` / `rh`)                           |
| `trajectory_id`    | Corresponding trajectory identifier                |
| `depth_index`      | Sampling-point index along the trajectory (`0–20`) |
| `normalized_depth` | Normalized cortical depth (`0.00–1.00`)            |
| `x`                | X coordinate of the depth sampling point           |
| `y`                | Y coordinate of the depth sampling point           |
| `z`                | Z coordinate of the depth sampling point           |
| `ADC`              | Apparent diffusion coefficient                     |
| `FA`               | Fractional anisotropy                              |
| `AD`               | Axial diffusivity                                  |
| `RD`               | Radial diffusivity                                 |
| `QSM`              | Quantitative susceptibility mapping value          |

### Depth Convention

```text
normalized_depth = 0.00   GM/WM interface
normalized_depth = 1.00   Pial surface
```


