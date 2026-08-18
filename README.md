# Cortical trajectory analysis

## Data Preparation

The `data_preparation` module identifies and verifies MRI acquisitions before cortical reconstruction, diffusion processing, and cortical trajectory generation.

Raw subject directories may contain multiple MRI acquisitions without standardized filenames. This module uses BXH metadata and image-level comparison to identify candidate T1-weighted, diffusion, QSM, and other MRI acquisitions.

### Scripts

```text
data_preparation/
├── identify_scans.py
├── inspect_bxh_details.py
└── compare_images.py
```

### `identify_scans.py`

Scans a raw subject directory for `.bxh` files, extracts key acquisition metadata, and assigns a preliminary scan classification.

Run:

```bash
python data_preparation/identify_scans.py \
/path/to/raw_subject
```

To save the scan inventory:

```bash
python data_preparation/identify_scans.py \
/path/to/raw_subject \
--output /path/to/scan_inventory.csv
```

The output includes metadata such as:

```text
file
description
series_number
protocol_name
TR
TE
TI
flip_angle
matrix
voxel_size
scan_options
classification
confidence
status
```

Temporary files beginning with `._` are ignored.

### `inspect_bxh_details.py`

Displays detailed BXH metadata for one acquisition or compares the metadata of two acquisitions.

Run with one BXH file:

```bash
python data_preparation/inspect_bxh_details.py \
/path/to/scan.bxh
```

Compare two BXH files:

```bash
python data_preparation/inspect_bxh_details.py \
/path/to/scan_1.bxh \
/path/to/scan_2.bxh
```

This script is useful when multiple candidate acquisitions are detected and additional metadata are needed to determine their relationship.

### `compare_images.py`

Compares two MRI image files using image geometry and voxel intensities.

Run:

```bash
python data_preparation/compare_images.py \
/path/to/image_1.nii.gz \
/path/to/image_2.nii.gz
```

The comparison includes:

```text
shape
voxel size
orientation
affine
intensity statistics
voxel-wise differences
```

Voxel-wise comparison is performed only when the image dimensions match.

The script uses `nibabel` and can also read supported FreeSurfer image formats such as `.mgz` and `.mgh`.

### Workflow

```text
Raw subject directory
        |
        v
identify_scans.py
        |
        v
Scan inventory
        |
        +----------------------+
        |                      |
        v                      v
clear candidate         ambiguous candidates
                               |
                               v
                  inspect_bxh_details.py
                               |
                               v
                      compare_images.py
                               |
                               v
                   select input images
```

The raw MRI files are not modified during data preparation.

The selected structural T1 image is passed to cortical reconstruction, while the selected diffusion acquisition is passed to diffusion processing.

## Cortical Reconstruction

Cortical surfaces and anatomical parcellations are generated using FreeSurfer.

This step must be run on a system with FreeSurfer installed and configured. In the current workflow, FreeSurfer reconstruction is performed on a local macOS workstation, and the resulting subject directory is transferred back to the compute server for downstream processing.

Run:

```bash
bash cortical_reconstruction/run_freesurfer.sh \
    <subject_id> \
    <t1_image> \
    <subjects_dir> \
    [threads]

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


