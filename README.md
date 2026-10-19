# Cortical Column Analysis

This repository contains the current implementation of the cortical column
processing pipeline, including MRI data preparation, cortical reconstruction,
diffusion processing, registration, and template-driven column generation.

## Data Preparation

The `data_preparation` module is used to identify and inspect MRI acquisitions before downstream processing.

### `identify_scans.py`

Identifies candidate MRI scans from BXH metadata and provides a preliminary classification of structural, diffusion, QSM, and other acquisitions.

```bash
python data_preparation/identify_scans.py \
    /path/to/raw_subject
```

Optional output:

```bash
python data_preparation/identify_scans.py \
    /path/to/raw_subject \
    --output /path/to/scan_inventory.csv
```

### `inspect_bxh_details.py`

Displays detailed BXH metadata for one acquisition or compares two BXH files.

```bash
python data_preparation/inspect_bxh_details.py \
    /path/to/scan.bxh
```

### `compare_images.py`

Compares image geometry and voxel intensities between two MRI images.

```bash
python data_preparation/compare_images.py \
    /path/to/image_1.nii.gz \
    /path/to/image_2.nii.gz
```

### `extract_diffusion_gradients.py`

Extracts diffusion b-values and gradient vectors from BXH metadata and saves them in FSL `.bval` and `.bvec` format.

```bash
python data_preparation/extract_diffusion_gradients.py \
    /path/to/diffusion.bxh \
    --nifti /path/to/diffusion.nii.gz \
    --output-prefix /path/to/output_prefix
```

The gradient coordinate convention should be validated before tensor fitting.

## Cortical Reconstruction

Cortical surfaces and anatomical parcellations are generated using FreeSurfer.

```bash
bash cortical_reconstruction/run_freesurfer.sh \
    <subject_id> \
    <t1_image> \
    <subjects_dir> \
    [threads]
```

The main outputs used by the column pipeline include the white and pial
surfaces, cortical thickness, and anatomical parcellation.

FreeSurfer outputs can be checked with:

```bash
python qc/check_freesurfer_outputs.py \
    <freesurfer_subject_dir>
```

## Diffusion Processing

### Initial Preprocessing

The current preprocessing script performs DWI denoising, b0 extraction, and brain-mask generation.

```bash
bash diffusion_processing/preprocess_dwi.sh \
    <input_dwi.mif> \
    <output_dir>
```

The complete preprocessing strategy for the MUSE diffusion data is still being validated.

### Diffusion Tensor Metrics

After a validated preprocessed DWI and diffusion gradients are available, tensor fitting can be performed with:

```bash
bash diffusion_processing/fit_tensor_and_metrics.sh \
    --dwi <preprocessed_dwi> \
    --bvec <bvec_file> \
    --bval <bval_file> \
    --mask <brain_mask> \
    --output-dir <output_directory>
```

The script generates:

```text
tensor.mif
adc.nii.gz
fa.nii.gz
ad.nii.gz
rd.nii.gz
```

## Registration

Diffusion MRI is registered to the FreeSurfer subject using boundary-based registration.

```bash
bash registration/run_bbregister.sh \
    <subject_id> \
    <freesurfer_subjects_dir> \
    <masked_dwi.nii.gz> \
    <output_transform.dat>
```

The resulting transformation is used to relate native-T1 column coordinates to
diffusion-image space.

## Template-Driven Cortical Column Generation

The primary column definition is driven by the selected FreeSurfer template.
For each template vertex, `vtkStaticCellLocator` finds the closest point on
the subject's registered spherical mesh. The corresponding source triangle and
its barycentric weights are then applied to the subject's native white and
pial surfaces. The interpolated endpoints define one subject-specific column
in native T1 coordinates.

Each column contains 21 equally spaced points from pial to white, including
both endpoints. The saved column file also retains the source face, source
vertex IDs, barycentric weights, sphere QC, and mapping validity.

```bash
python trajectory_processing/build_template_columns.py \
    --subject-id <subject_id> \
    --subjects-dir <freesurfer_subjects_dir> \
    --template fsaverage \
    --output-dir <columns_output_dir>
```

Diffusion profiles are sampled from these native-T1 columns after the DWI to
T1 registration. The end-to-end runner is:

```bash
python pipeline/run_analysis.py \
    --subject-id <subject_id> \
    --subjects-dir <freesurfer_subjects_dir> \
    --md-image <md.nii.gz> \
    --registration-reference <masked_dwi.nii.gz> \
    --analysis-dir <analysis_dir>
```

The output structure is:

```text
<analysis_dir>/
├── registration/
├── columns/
├── profiles_before_clean/
├── profiles/
└── regional_profiles/
```

## Image Space QC

Spatial geometry between two NIfTI images can be compared using:

```bash
python visualization/check_image_space.py \
    --image1 <image_1> \
    --image2 <image_2>
```

The script compares image shape, voxel size, orientation, and affine transformation.

## Column-Level Data Structure

This project organizes cortical column data into two linked tables. The first
table stores column-level anatomical and geometric information, while the
second table stores depth-resolved imaging measurements sampled along each
column. Together, these tables support cortical depth profiling, sampling
experiments, and column-level graph construction.

## Table 1 — Column-Level Information

Each row represents one cortical column.

| Field                | Description                                            |
| -------------------- | ------------------------------------------------------ |
| `subject_id`         | Subject identifier                                     |
| `hemisphere`         | Hemisphere (`lh` / `rh`)                               |
| `column_id`          | Stable column identifier within each hemisphere        |
| `region_label`       | Cortical anatomical region                             |
| `white_x`            | X coordinate at the gray matter/white matter interface |
| `white_y`            | Y coordinate at the gray matter/white matter interface |
| `white_z`            | Z coordinate at the gray matter/white matter interface |
| `pial_x`             | X coordinate at the pial surface                       |
| `pial_y`             | Y coordinate at the pial surface                       |
| `pial_z`             | Z coordinate at the pial surface                       |
| `cortical_thickness` | Cortical thickness associated with the column          |
| `local_surface_area` | Local pial surface area associated with the column     |
| `coordinate_space`   | Coordinate space of the column coordinates             |

The column identifier is defined by:

```text
subject_id + hemisphere + column_id
```

Example:

```text
S00775 + lh + 154
```

## Table 2 — Column Depth-Level Information

Each row represents one sampling point along a cortical column.

| Field              | Description                                        |
| ------------------ | -------------------------------------------------- |
| `subject_id`       | Subject identifier                                 |
| `hemisphere`       | Hemisphere (`lh` / `rh`)                           |
| `column_id`        | Corresponding column identifier                    |
| `depth_index`      | Sampling-point index along the column (`0–20`)      |
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
