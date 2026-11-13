"""Apply the repository legend and sample-annotation standard to notebooks.

Description
-----------
Apply the repository legend and sample-annotation standard to notebooks.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--check`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/apply_notebook_legend_standard.py --help
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]
LEGEND_CALL = re.compile(
    r"(?P<owner>[A-Za-z_]\w*(?:\[[^\]\n]+\])?)\.legend\("
)
STYLE_KEYS = (
    "frameon",
    "facecolor",
    "edgecolor",
    "framealpha",
    "fancybox",
    "fontsize",
    "borderpad",
    "labelspacing",
    "handlelength",
    "handletextpad",
)
UNFRAMED_EXCEPTIONS = {
    (
        "analysis_results/figure2/figure2_AB_method_comparison.ipynb",
        "figure2c-main-plot",
    ),
}


def matching_parenthesis(text: str, open_index: int) -> int:
    """Return the closing parenthesis while ignoring strings and comments."""
    depth = 0
    quote = None
    triple = False
    escaped = False
    comment = False
    index = open_index
    while index < len(text):
        char = text[index]
        if comment:
            if char == "\n":
                comment = False
            index += 1
            continue
        if quote is not None:
            if escaped:
                escaped = False
                index += 1
                continue
            if char == "\\":
                escaped = True
                index += 1
                continue
            if triple and text.startswith(quote * 3, index):
                quote = None
                triple = False
                index += 3
                continue
            if not triple and char == quote:
                quote = None
            index += 1
            continue
        if char == "#":
            comment = True
            index += 1
            continue
        if char in {'"', "'"}:
            quote = char
            triple = text.startswith(char * 3, index)
            index += 3 if triple else 1
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    raise ValueError("Unmatched legend parenthesis")


def set_keyword(call: str, key: str, value: str) -> tuple[str, bool]:
    value_pattern = r'(?:"[^"\n]*"|\'[^\'\n]*\'|[^,)\n]+)'
    pattern = re.compile(rf"(\b{key}\s*=\s*){value_pattern}")
    updated, count = pattern.subn(rf"\g<1>{value}", call, count=1)
    return updated, bool(count)


def add_keywords(call: str, keywords: list[tuple[str, str]]) -> str:
    if not keywords:
        return call
    additions = [f"{key}={value}" for key, value in keywords]
    close_index = len(call) - 1
    close_line_start = call.rfind("\n", 0, close_index) + 1
    before_close = call[close_line_start:close_index]
    if "\n" in call and not before_close.strip():
        close_indent = before_close
        argument_indent = close_indent + "    "
        prefix = call[:close_line_start].rstrip()
        if not prefix.endswith(("(", ",")):
            prefix += ","
        inserted = "\n".join(
            f"{argument_indent}{argument}," for argument in additions
        )
        return f"{prefix}\n{inserted}\n{close_indent})"
    prefix = call[:-1].rstrip()
    separator = "" if prefix.endswith(("(", ",")) else ", "
    return f"{prefix}{separator}{', '.join(additions)})"


def legend_style(*, framed: bool) -> dict[str, str]:
    common = {
        "frameon": "True" if framed else "False",
        "fontsize": "7.5",
        "labelspacing": ".35",
        "handlelength": "1.8",
        "handletextpad": ".5",
    }
    if framed:
        common.update(
            {
                "facecolor": '"white"',
                "edgecolor": '"#D0D0D0"',
                "framealpha": ".90",
                "fancybox": "False",
                "borderpad": ".35",
            }
        )
    return common


def update_legends(source: str, *, notebook: str, cell_id: str) -> str:
    matches = list(LEGEND_CALL.finditer(source))
    for match in reversed(matches):
        open_index = match.end() - 1
        close_index = matching_parenthesis(source, open_index)
        call = source[match.start():close_index + 1]
        owner = match.group("owner")
        outside_axes = owner == "fig" or "bbox_to_anchor=(1.02" in call.replace(" ", "")
        explicit_clear_area = (notebook, cell_id) in UNFRAMED_EXCEPTIONS
        style = legend_style(framed=not (outside_axes or explicit_clear_area))
        additions = []
        for key in STYLE_KEYS:
            if key not in style:
                continue
            call, found = set_keyword(call, key, style[key])
            if not found:
                additions.append((key, style[key]))
        call = add_keywords(call, additions)
        source = source[:match.start()] + call + source[close_index + 1:]
    return source


def update_sample_annotations(source: str, *, notebook: str, cell_id: str) -> str:
    # These subject-level panels can have different effective n values by
    # metric, so the in-panel n is retained but made small and neutral.
    if (
        notebook
        == "qc/archive/validation/spatial_data/column_spatial_similarity_qc_v2.ipynb"
        and cell_id == "5f338236"
    ):
        source = source.replace(
            '        fontsize=9,\n',
            '        fontsize=7,\n        color="#555555",\n',
            1,
        )
    return source


def transform_notebook(path: Path) -> bool:
    relative = path.relative_to(REPO_ROOT).as_posix()
    raw_text = path.read_text()
    lines = raw_text.splitlines()
    json_indent = (
        len(lines[1]) - len(lines[1].lstrip())
        if len(lines) > 1 and lines[1].strip()
        else 1
    )
    notebook = json.loads(raw_text)
    changed = False
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        updated = update_legends(
            source,
            notebook=relative,
            cell_id=cell.get("id", ""),
        )
        updated = update_sample_annotations(
            updated,
            notebook=relative,
            cell_id=cell.get("id", ""),
        )
        if updated != source:
            cell["source"] = updated.splitlines(keepends=True)
            if cell["source"] and not updated.endswith("\n"):
                cell["source"][-1] = cell["source"][-1].rstrip("\n")
            changed = True
    if changed:
        path.write_text(
            json.dumps(notebook, indent=json_indent, ensure_ascii=False) + "\n"
        )
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report notebooks that would change without writing them.",
    )
    args = parser.parse_args()
    changed = []
    for path in sorted(REPO_ROOT.rglob("*.ipynb")):
        if args.check:
            original = path.read_text()
            notebook = json.loads(original)
            relative = path.relative_to(REPO_ROOT).as_posix()
            would_change = False
            for cell in notebook.get("cells", []):
                if cell.get("cell_type") != "code":
                    continue
                source = "".join(cell.get("source", []))
                updated = update_sample_annotations(
                    update_legends(
                        source,
                        notebook=relative,
                        cell_id=cell.get("id", ""),
                    ),
                    notebook=relative,
                    cell_id=cell.get("id", ""),
                )
                would_change |= updated != source
            if would_change:
                changed.append(relative)
        elif transform_notebook(path):
            changed.append(path.relative_to(REPO_ROOT).as_posix())
    action = "would update" if args.check else "updated"
    print(f"{action} {len(changed)} notebook(s)")
    for path in changed:
        print(path)


if __name__ == "__main__":
    main()
