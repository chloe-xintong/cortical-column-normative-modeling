"""Create the fsaverage6 QC notebook from the standard fsaverage QC notebook."""
from pathlib import Path
import json
from rebuild_fsaverage_mapping_qc_notebook import cells

def replace(value):
    if isinstance(value, str):
        return (value.replace("fsaverage", "fsaverage6")
                .replace("163842", "40962"))
    if isinstance(value, list): return [replace(x) for x in value]
    if isinstance(value, dict): return {k: replace(v) for k, v in value.items()}
    return value

nb = replace({"cells": cells, "metadata": {"kernelspec": {"display_name": "cortical_trajectory_analysis", "language": "python", "name": "cortical_trajectory_analysis"}, "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5})
Path("surface_alignment/fsaverage6_mapping_qc.ipynb").write_text(json.dumps(nb, indent=1))
print("wrote surface_alignment/fsaverage6_mapping_qc.ipynb")
