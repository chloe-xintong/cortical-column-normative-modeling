#!/usr/bin/env python3

"""
Identify MRI acquisitions from BXH metadata.

The script scans a subject directory for BXH files, extracts key
acquisition metadata, and assigns a conservative scan classification.

It is intended as the first stage of the cortical trajectory pipeline.
Raw image files are never modified.
"""

from pathlib import Path
import argparse
import csv
import xml.etree.ElementTree as ET


def strip_namespace(tag):
    """Remove an XML namespace from a tag."""
    return tag.split("}")[-1]


def get_all_text(root):
    """Collect non-empty XML elements into a dictionary."""
    metadata = {}

    for elem in root.iter():
        tag = strip_namespace(elem.tag)

        if elem.text:
            value = elem.text.strip()

            if value:
                metadata.setdefault(tag, []).append(value)

    return metadata


def first_value(metadata, possible_keys):
    """Return the first value found among possible XML tag names."""
    for key in possible_keys:
        if key in metadata and metadata[key]:
            return metadata[key][0]

    return ""


def get_values(metadata, possible_keys):
    """Return all values found for the first matching XML tag."""
    for key in possible_keys:
        if key in metadata:
            return metadata[key]

    return []


def join_values(values):
    """Convert multiple XML values to a compact string."""
    return " | ".join(values)


def classify_scan(description, scan_options):
    """
    Classify an MRI acquisition conservatively.

    Classification is based primarily on the BXH description and
    reconstruction information. The rules can be expanded after
    validation on additional subjects.
    """

    text = description.lower().strip()
    options = scan_options.lower().strip()

    if not text:
        return "UNKNOWN", "LOW"

    if "localizer" in text:
        return "LOCALIZER", "HIGH"

    if text.startswith("trace:"):
        return "DWI_DERIVED_TRACE", "HIGH"

    if text.startswith("fa:"):
        return "DWI_DERIVED_FA", "HIGH"

    if text.startswith("avdc"):
        return "DWI_DERIVED_ADC", "HIGH"

    if text.startswith("cbf:"):
        return "ASL_DERIVED_CBF", "HIGH"

    if text == "muse":
        return "DWI_RAW_CANDIDATE", "HIGH"

    if (
        "diffusion" in text
        or "dwi" in text
        or "dti" in text
    ):
        return "DWI_RAW_CANDIDATE", "HIGH"

    if "t1" in text and (
        "mprage" in text
        or "bravo" in text
        or "fspgr" in text
    ):
        if text.startswith("orig"):
            return "T1_ORIGINAL_CANDIDATE", "HIGH"

        if "filtered_gems" in options:
            return "T1_FILTERED_RECONSTRUCTION", "HIGH"

        return "T1_RECONSTRUCTION", "MEDIUM"

    if "qsm" in text or "multi-echo" in text:
        return "QSM_CANDIDATE", "HIGH"

    if "flair" in text:
        return "T2_FLAIR", "HIGH"

    if "asl" in text:
        return "ASL", "HIGH"

    return "UNKNOWN", "LOW"


def parse_bxh(bxh_file):
    """Parse one BXH file and extract relevant acquisition metadata."""

    try:
        tree = ET.parse(bxh_file)
        root = tree.getroot()

    except ET.ParseError as error:
        return {
            "file": bxh_file.name,
            "description": "",
            "series_number": "",
            "protocol_name": "",
            "TR": "",
            "TE": "",
            "TI": "",
            "flip_angle": "",
            "matrix": "",
            "voxel_size": "",
            "scan_options": "",
            "classification": "INVALID_BXH",
            "confidence": "LOW",
            "status": f"XML ERROR: {error}",
        }

    metadata = get_all_text(root)

    description = first_value(
        metadata,
        [
            "description",
            "seriesdescription",
            "protocolname",
            "acquisitiondescription",
        ],
    )

    series_number = first_value(
        metadata,
        [
            "seriesnumber",
            "series_number",
        ],
    )

    protocol_name = first_value(
        metadata,
        [
            "protocolname",
            "protocol_name",
        ],
    )

    tr = first_value(
        metadata,
        [
            "tr",
            "repetitiontime",
        ],
    )

    te = first_value(
        metadata,
        [
            "te",
            "echotime",
        ],
    )

    ti = first_value(
        metadata,
        [
            "ti",
            "inversiontime",
        ],
    )

    flip_angle = first_value(
        metadata,
        [
            "flipangle",
            "flip_angle",
        ],
    )

    acquisition_matrix = first_value(
        metadata,
        [
            "acquisitionmatrix",
        ],
    )

    size_values = get_values(
        metadata,
        [
            "size",
        ],
    )

    if acquisition_matrix and size_values:
        matrix = (
            f"{acquisition_matrix} "
            f"{size_values[-1]}"
        )
    elif acquisition_matrix:
        matrix = acquisition_matrix
    else:
        matrix = join_values(size_values)

    prescribed_pixel_size = first_value(
        metadata,
        [
            "prescribedpixelsize",
        ],
    )

    slice_spacing = first_value(
        metadata,
        [
            "spacing",
            "spacingbetweenslices",
            "prescribedslicespacing",
        ],
    )

    if prescribed_pixel_size and slice_spacing:
        voxel_size = (
            f"{prescribed_pixel_size} "
            f"{slice_spacing}"
        )
    else:
        voxel_size = prescribed_pixel_size or slice_spacing

    scan_options = first_value(
        metadata,
        [
            "scanoptions",
        ],
    )

    classification, confidence = classify_scan(
        description,
        scan_options,
    )

    return {
        "file": bxh_file.name,
        "description": description,
        "series_number": series_number,
        "protocol_name": protocol_name,
        "TR": tr,
        "TE": te,
        "TI": ti,
        "flip_angle": flip_angle,
        "matrix": matrix,
        "voxel_size": voxel_size,
        "scan_options": scan_options,
        "classification": classification,
        "confidence": confidence,
        "status": "OK",
    }


def scan_subject(subject_dir):
    """Find and parse all valid BXH files in a subject directory."""

    bxh_files = sorted(
        path
        for path in subject_dir.rglob("*.bxh")
        if not path.name.startswith("._")
    )

    if not bxh_files:
        raise FileNotFoundError(
            f"No valid BXH files found in:\n{subject_dir}"
        )

    return [
        parse_bxh(bxh_file)
        for bxh_file in bxh_files
    ]


def print_inventory(results):
    """Print detailed scan information."""

    print()
    print("SCAN INVENTORY")
    print()

    for index, scan in enumerate(results, start=1):
        print(f"[SCAN {index}]")
        print(f"File          : {scan.get('file', '')}")
        print(f"Description   : {scan.get('description', '')}")
        print(f"Series number : {scan.get('series_number', '')}")
        print(f"Protocol      : {scan.get('protocol_name', '')}")
        print(f"TR            : {scan.get('TR', '')}")
        print(f"TE            : {scan.get('TE', '')}")
        print(f"TI            : {scan.get('TI', '')}")
        print(f"Flip angle    : {scan.get('flip_angle', '')}")
        print(f"Matrix        : {scan.get('matrix', '')}")
        print(f"Voxel size    : {scan.get('voxel_size', '')}")
        print(f"Scan options  : {scan.get('scan_options', '')}")
        print(f"Classification: {scan.get('classification', '')}")
        print(f"Confidence    : {scan.get('confidence', '')}")
        print(f"Status        : {scan.get('status', '')}")
        print()


def print_summary(results):
    """Print a compact scan classification summary."""

    groups = {}

    for scan in results:
        classification = scan.get(
            "classification",
            "UNKNOWN",
        )

        groups.setdefault(
            classification,
            [],
        ).append(scan["file"])

    print("SUMMARY")
    print()

    for classification, files in sorted(groups.items()):
        print(f"{classification}: {len(files)}")

        for filename in files:
            print(f"  {filename}")

        print()


def save_inventory(results, output_csv):
    """Save scan inventory as CSV."""

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "file",
        "description",
        "series_number",
        "protocol_name",
        "TR",
        "TE",
        "TI",
        "flip_angle",
        "matrix",
        "voxel_size",
        "scan_options",
        "classification",
        "confidence",
        "status",
    ]

    with open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    key: result.get(key, "")
                    for key in fieldnames
                }
            )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect BXH metadata and identify MRI acquisitions."
        )
    )

    parser.add_argument(
        "subject_dir",
        type=Path,
        help="Raw subject directory containing BXH files",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for scan_inventory.csv",
    )

    args = parser.parse_args()

    subject_dir = args.subject_dir.resolve()

    if not subject_dir.exists():
        raise FileNotFoundError(
            f"Subject directory does not exist:\n{subject_dir}"
        )

    if not subject_dir.is_dir():
        raise NotADirectoryError(
            f"Input is not a directory:\n{subject_dir}"
        )

    print()
    print(f"Subject directory: {subject_dir}")

    results = scan_subject(subject_dir)

    print(f"BXH files found: {len(results)}")

    print_inventory(results)
    print_summary(results)

    if args.output:
        output_csv = args.output.resolve()

        save_inventory(
            results,
            output_csv,
        )

        print(f"Inventory saved to: {output_csv}")


if __name__ == "__main__":
    main()