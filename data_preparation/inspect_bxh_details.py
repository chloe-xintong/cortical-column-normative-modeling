#!/usr/bin/env python3

"""
Inspect detailed BXH metadata for one or more acquisitions.

This script prints all non-empty XML fields so that candidate scans
can be compared without assuming specific BXH tag names.
"""

from pathlib import Path
import argparse
import xml.etree.ElementTree as ET


def strip_namespace(tag):
    return tag.split("}")[-1]


def parse_bxh(path):
    tree = ET.parse(path)
    root = tree.getroot()

    rows = []

    for elem in root.iter():
        tag = strip_namespace(elem.tag)

        text = ""
        if elem.text:
            text = elem.text.strip()

        if text:
            rows.append((tag, text))

    return rows


def group_rows(rows):
    grouped = {}

    for tag, value in rows:
        grouped.setdefault(tag, [])

        if value not in grouped[tag]:
            grouped[tag].append(value)

    return grouped


def print_metadata(path):
    rows = parse_bxh(path)
    metadata = group_rows(rows)

    print()
    print(f"FILE: {path}")
    print()

    for tag in sorted(metadata):
        values = metadata[tag]

        if len(values) == 1:
            print(f"{tag}: {values[0]}")

        else:
            print(f"{tag}:")
            for value in values:
                print(f"  - {value}")


def compare_metadata(path1, path2):
    meta1 = group_rows(parse_bxh(path1))
    meta2 = group_rows(parse_bxh(path2))

    all_tags = sorted(set(meta1) | set(meta2))

    print()
    print("DIFFERENCES")
    print()

    difference_count = 0

    for tag in all_tags:
        values1 = meta1.get(tag, [])
        values2 = meta2.get(tag, [])

        if values1 != values2:
            difference_count += 1

            print(f"[{tag}]")

            print("  File 1:")
            if values1:
                for value in values1:
                    print(f"    {value}")
            else:
                print("    <missing>")

            print("  File 2:")
            if values2:
                for value in values2:
                    print(f"    {value}")
            else:
                print("    <missing>")

            print()

    print(f"Different metadata fields: {difference_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect and compare detailed BXH metadata."
    )

    parser.add_argument(
        "bxh1",
        type=Path,
        help="First BXH file",
    )

    parser.add_argument(
        "bxh2",
        type=Path,
        nargs="?",
        default=None,
        help="Optional second BXH file for comparison",
    )

    args = parser.parse_args()

    bxh1 = args.bxh1.resolve()

    if not bxh1.exists():
        raise FileNotFoundError(f"File not found: {bxh1}")

    print_metadata(bxh1)

    if args.bxh2:
        bxh2 = args.bxh2.resolve()

        if not bxh2.exists():
            raise FileNotFoundError(f"File not found: {bxh2}")

        print_metadata(bxh2)

        compare_metadata(
            bxh1,
            bxh2,
        )


if __name__ == "__main__":
    main()