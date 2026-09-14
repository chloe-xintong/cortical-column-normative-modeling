#!/usr/bin/env python3
"""Build a file-level inventory for ADRC DWI/T1 preprocessing inputs.

This script only reads raw folders and optional Excel metadata. It does not
run preprocessing or alter any files.
"""

import argparse
import csv
import json
from pathlib import Path


# These are the inputs needed by the current ADRC dwifslpreproc command.
REQUIRED_FIELDS = (
    "has_t1", "has_dwi", "has_bval", "has_bvec", "has_reverse_b0",
    "has_dwi_json", "has_readout_time", "has_phase_encoding",
)
DIAGNOSIS_LABELS = ("NORMCOG", "IMPNOMCI", "DEMENTED", "ALZDIS")


def load_metadata(metadata_path):
    if metadata_path is None or not metadata_path.is_file():
        return {}
    if metadata_path.suffix.lower() == ".csv":
        with metadata_path.open(newline="") as handle:
            records = list(csv.DictReader(handle))
        id_key = "PTID" if records and "PTID" in records[0] else "SubjectID"
        def value(record, key):
            return record.get(key, "")
    else:
        try:
            import openpyxl
        except ImportError:
            return {}
        sheet = openpyxl.load_workbook(
            metadata_path, read_only=True, data_only=True
        ).active
        rows = sheet.iter_rows(values_only=True)
        headers = list(next(rows, ()))
        id_key = "SubjectID" if "SubjectID" in headers else "PTID"
        if id_key not in headers:
            return {}
        records = [dict(zip(headers, row)) for row in rows]
        def value(record, key):
            return record.get(key, "")

    result = {}
    for record in records:
        sid = value(record, id_key)
        if sid is None or str(sid).strip() == "":
            continue
        sid = str(sid).strip()
        current = result.setdefault(sid, {"has_imaging_metadata": ""})
        if value(record, "HAS_IMAGING_METADATA") not in (None, ""):
            current["has_imaging_metadata"] = value(record, "HAS_IMAGING_METADATA")
        for name in DIAGNOSIS_LABELS:
            key = f"has_{name.lower()}_label"
            current[key] = current.get(key, False) or (
                value(record, name) is not None
                and str(value(record, name)).strip() != ""
            )
    return result


def first_file(folder, patterns, exclude=()):
    candidates = []
    for pattern in patterns:
        candidates.extend(folder.glob(pattern))
    candidates = [p for p in candidates if p.is_file() and not any(x in p.name for x in exclude)]
    return sorted(set(candidates))[0] if candidates else None


def describe(path):
    return (str(path), path.stat().st_size) if path else ("", 0)


def inspect_subject(folder, metadata, metadata_match):
    subject = folder.name
    dwi = first_file(folder, ("*.nii.gz",), exclude=("T1", "revphase"))
    reverse = first_file(folder, ("*revphase*.nii.gz",))
    t1 = folder / "T1.nii.gz"
    # pathlib.Path.stem for foo.nii.gz is foo.nii; strip the full suffix
    # explicitly so the sidecar lookup remains correct.
    dwi_stem = dwi.name[:-7] if dwi and dwi.name.endswith(".nii.gz") else ""
    json_path = (folder / f"{dwi_stem}.json") if dwi_stem else None
    if json_path and not json_path.is_file():
        json_path = first_file(folder, ("*.json",), exclude=("T1", "revphase"))

    stem = dwi.name[:-7] if dwi and dwi.name.endswith(".nii.gz") else ""
    bval = folder / f"{stem}.bval" if stem else None
    bvec = folder / f"{stem}.bvec" if stem else None
    if bval and not bval.is_file():
        bval = first_file(folder, ("*.bval",))
    if bvec and not bvec.is_file():
        bvec = first_file(folder, ("*.bvec",))

    pe_direction = ""
    readout_time = ""
    if json_path and json_path.is_file():
        try:
            data = json.loads(json_path.read_text())
            pe_direction = data.get("PhaseEncodingDirection", "")
            readout_time = data.get("TotalReadoutTime", "")
        except (OSError, json.JSONDecodeError):
            pass

    flags = {
        "has_t1": t1.is_file(),
        "has_dwi": dwi is not None,
        "has_bval": bval is not None and bval.is_file(),
        "has_bvec": bvec is not None and bvec.is_file(),
        "has_reverse_b0": reverse is not None,
        "has_dwi_json": json_path is not None and json_path.is_file(),
        "has_readout_time": isinstance(readout_time, (int, float)) and readout_time > 0,
        "has_phase_encoding": bool(pe_direction),
    }
    ready = all(flags[key] for key in REQUIRED_FIELDS)
    missing = [key[4:] for key in REQUIRED_FIELDS if not flags[key]]

    label_flags = {
        f"has_{name.lower()}_label": bool(metadata.get(f"has_{name.lower()}_label", False))
        for name in DIAGNOSIS_LABELS
    }
    row = {
        "subject": subject,
        "metadata_subject_id": f"ADRC{subject[1:]}" if subject.startswith("D") else "",
        "metadata_match": metadata_match,
        "has_imaging_metadata": metadata.get("has_imaging_metadata", ""),
        "has_any_diagnosis_label": any(label_flags.values()),
        **flags,
        "preprocessing_ready": ready,
        "missing_required": ";".join(missing),
        "pe_direction_json": pe_direction,
        "readout_time": readout_time,
    }
    for label, path in (("t1", t1), ("dwi", dwi), ("bval", bval), ("bvec", bvec), ("reverse_b0", reverse), ("dwi_json", json_path)):
        row[f"{label}_path"], row[f"{label}_bytes"] = describe(path) if path and path.is_file() else ("", 0)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()

    metadata = load_metadata(args.metadata)
    rows = [
        inspect_subject(
            folder,
            metadata.get(f"ADRC{folder.name[1:]}", {}),
            f"ADRC{folder.name[1:]}" in metadata,
        )
        for folder in sorted(args.raw_root.iterdir())
        if folder.is_dir() and folder.name.startswith("D")
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["subject"]
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} subjects to {args.output}")
    print(f"Preprocessing-ready: {sum(bool(row.get('preprocessing_ready')) for row in rows)}")
    without_normcog = sum(
        not bool(metadata.get(f"ADRC{row['subject'][1:]}", {}).get("has_normcog_label", False))
        for row in rows
    )
    print(f"Without NORMCOG label: {without_normcog}")


if __name__ == "__main__":
    main()
