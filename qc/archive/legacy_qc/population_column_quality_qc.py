"""Population Column Quality Qc

Description
-----------
Population Column Quality Qc

Methods
-------
1. Provide the reusable computations defined in this module.
2. Return results to the calling analysis code.

Inputs
------
- Data supplied through the module's public function arguments.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
Import this module from the calling analysis code; it has no command-line entry point.
"""

from pathlib import Path
import json

cells = [
    {"cell_type": "markdown", "metadata": {}, "source": [
        "# Population Column Data Quality QC\n", "\n",
        "This notebook evaluates raw cortical column profiles before cleaning across the full cohort. It does not interpolate, delete, or overwrite profiles.\n", "\n",
        "The analysis summarizes profile completeness, zero/NaN/Inf values, depth-specific failures, consecutive bad points, and cross-metric overlap for MD, FA, AD, and RD."
    ]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "from pathlib import Path\nimport re\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\nRESULTS_ROOT = Path('/mnt/newStor/paros/paros_WORK/xintong/results/column_analysis')\nPROFILE_ROOT = RESULTS_ROOT\nQC_ROOT = RESULTS_ROOT / 'population_column_quality_qc'\nQC_ROOT.mkdir(parents=True, exist_ok=True)\nMETRICS = ('md', 'fa', 'ad', 'rd')\nDEPTHS = 21\nprint('Reading:', PROFILE_ROOT)\nprint('Writing:', QC_ROOT)"
    ]},
    {"cell_type": "markdown", "metadata": {}, "source": [
        "## 1. Load and classify raw profiles\n\n",
        "Each CSV is one subject × hemisphere × region × metric. Empty files are retained in the QC table as unavailable rather than treated as valid zero profiles."
    ]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "pattern = re.compile(r'^(?P<subject>D\\d+)_(?P<hemi>lh|rh)_(?P<region>.+)_(?P<metric>md|fa|ad|rd)\\.csv$')\nrows = []\nprofile_cache = {}\nfor profile_dir in sorted(PROFILE_ROOT.glob('D*/profiles_before_clean')):\n    for path in sorted(profile_dir.glob('*.csv')):\n        m = pattern.match(path.name)\n        if not m:\n            continue\n        meta = m.groupdict()\n        base = {**meta, 'path': str(path), 'file_size_bytes': path.stat().st_size}\n        if path.stat().st_size == 0:\n            rows.append({**base, 'status': 'empty', 'n_columns': 0, 'n_samples': 0, 'bad_points': 0, 'zero_points': 0, 'nan_points': 0, 'inf_points': 0})\n            continue\n        try:\n            values = np.loadtxt(path, delimiter=',', ndmin=2)\n            if values.ndim != 2:\n                values = np.atleast_2d(values)\n            profile_cache[str(path)] = values\n            finite = np.isfinite(values)\n            zero = values == 0\n            bad = (~finite) | zero\n            rows.append({**base, 'status': 'valid', 'n_columns': values.shape[0], 'n_samples': values.size, 'bad_points': int(bad.sum()), 'zero_points': int(zero.sum()), 'nan_points': int(np.isnan(values).sum()), 'inf_points': int(np.isinf(values).sum())})\n        except Exception as exc:\n            rows.append({**base, 'status': 'unreadable', 'n_columns': 0, 'n_samples': 0, 'bad_points': 0, 'zero_points': 0, 'nan_points': 0, 'inf_points': 0, 'error': str(exc)})\nprofiles = pd.DataFrame(rows)\nprofiles.to_csv(QC_ROOT / 'profile_file_inventory.csv', index=False)\nprint('Files:', len(profiles), 'Subjects:', profiles.subject.nunique())\nprint(profiles.status.value_counts(dropna=False))\ndisplay(profiles.head())"
    ]},
    {"cell_type": "markdown", "metadata": {}, "source": [
        "## 2. Subject-level and metric-level summary\n\n",
        "This table reports total profile files, empty/unreadable files, total sampled points, bad points, and the percentage of affected points."
    ]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "subject_summary = (profiles.groupby('subject', as_index=False).agg(total_profiles=('path','size'), empty_profiles=('status', lambda s: int((s=='empty').sum())), unreadable_profiles=('status', lambda s: int((s=='unreadable').sum())), total_columns=('n_columns','sum'), total_samples=('n_samples','sum'), bad_points=('bad_points','sum'), zero_points=('zero_points','sum'), nan_points=('nan_points','sum'), inf_points=('inf_points','sum')))\nsubject_summary['bad_point_pct'] = 100 * subject_summary['bad_points'] / subject_summary['total_samples'].replace(0, np.nan)\nsubject_summary.to_csv(QC_ROOT / 'subject_quality_summary.csv', index=False)\nmetric_summary = (profiles.groupby('metric', as_index=False).agg(total_profiles=('path','size'), empty_profiles=('status', lambda s: int((s=='empty').sum())), total_columns=('n_columns','sum'), total_samples=('n_samples','sum'), bad_points=('bad_points','sum'), zero_points=('zero_points','sum'), nan_points=('nan_points','sum'), inf_points=('inf_points','sum')))\nmetric_summary['bad_point_pct'] = 100 * metric_summary['bad_points'] / metric_summary['total_samples'].replace(0, np.nan)\nmetric_summary.to_csv(QC_ROOT / 'metric_quality_summary.csv', index=False)\ndisplay(subject_summary); display(metric_summary)"
    ]},
    {"cell_type": "markdown", "metadata": {}, "source": [
        "## 3. Bad points by number per column\n\n",
        "For each valid column, bad points are classified as exactly 1, exactly 2, or at least 3."
    ]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "column_rows = []\ndepth_rows = []\nfor _, row in profiles[profiles.status == 'valid'].iterrows():\n    values = profile_cache[row.path]\n    bad = (~np.isfinite(values)) | (values == 0)\n    counts = bad.sum(axis=1)\n    for n in counts:\n        category = '1' if n == 1 else '2' if n == 2 else '3_or_more' if n >= 3 else '0'\n        column_rows.append({'subject': row.subject, 'hemi': row.hemi, 'region': row.region, 'metric': row.metric, 'bad_points_per_column': int(n), 'category': category})\n    for depth, count in enumerate(bad.sum(axis=0)):\n        if count:\n            depth_rows.append({'subject': row.subject, 'hemi': row.hemi, 'region': row.region, 'metric': row.metric, 'depth': depth, 'bad_point_count': int(count)})\ncolumn_detail = pd.DataFrame(column_rows)\ndepth_detail = pd.DataFrame(depth_rows)\ncolumn_detail.to_csv(QC_ROOT / 'column_bad_point_detail.csv', index=False)\ndepth_detail.to_csv(QC_ROOT / 'depth_bad_point_detail.csv', index=False)\ncategory_summary = (column_detail.groupby(['subject','metric','category'], as_index=False).size().rename(columns={'size':'column_count'}))\ncategory_summary.to_csv(QC_ROOT / 'bad_points_per_column_summary.csv', index=False)\ndisplay(category_summary.head(20)); display(depth_detail.head())"
    ]},
    {"cell_type": "markdown", "metadata": {}, "source": [
        "## 4. Consecutive bad points and cross-metric overlap\n\n",
        "A column is flagged as consecutive when it contains adjacent bad depths. Cross-metric overlap is calculated for matching subject × hemisphere × region × column × depth coordinates."
    ]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "consecutive_rows = []\nfor _, row in profiles[profiles.status == 'valid'].iterrows():\n    values = profile_cache[row.path]\n    bad = (~np.isfinite(values)) | (values == 0)\n    for col_id, mask in enumerate(bad):\n        runs = np.diff(np.r_[False, mask, False].astype(int))\n        starts, ends = np.where(runs == 1)[0], np.where(runs == -1)[0]\n        max_run = int((ends - starts).max()) if len(starts) else 0\n        consecutive_rows.append({'subject': row.subject, 'hemi': row.hemi, 'region': row.region, 'metric': row.metric, 'column_id': col_id, 'bad_points': int(mask.sum()), 'max_consecutive_bad_points': max_run, 'has_consecutive_bad_points': max_run >= 2})\nconsecutive = pd.DataFrame(consecutive_rows)\nconsecutive.to_csv(QC_ROOT / 'consecutive_bad_point_detail.csv', index=False)\nprint('Columns with consecutive bad points:', int(consecutive.has_consecutive_bad_points.sum()))\n\nkey = ['subject','hemi','region']\nvalid = profiles[profiles.status == 'valid']\noverlap_rows = []\nfor group_key, group in valid.groupby(key):\n    metric_masks = {}\n    for _, row in group.iterrows():\n        metric_masks[row.metric] = ((~np.isfinite(profile_cache[row.path])) | (profile_cache[row.path] == 0))\n    if len(metric_masks) == 4:\n        stack = np.stack([metric_masks[m] for m in METRICS])\n        overlap_rows.append(dict(zip(key, group_key), total_coordinates=int(stack.shape[1] * stack.shape[2]), bad_in_all_metrics=int(np.all(stack, axis=0).sum()), bad_in_any_metric=int(np.any(stack, axis=0).sum())))\noverlap = pd.DataFrame(overlap_rows)\nif not overlap.empty:\n    overlap['all_metric_overlap_pct'] = 100 * overlap.bad_in_all_metrics / overlap.total_coordinates\n    overlap['any_metric_bad_pct'] = 100 * overlap.bad_in_any_metric / overlap.total_coordinates\noverlap.to_csv(QC_ROOT / 'cross_metric_overlap_summary.csv', index=False)\ndisplay(overlap.head())"
    ]},
    {"cell_type": "markdown", "metadata": {}, "source": [
        "## 5. Population visualizations\n\n",
        "The main figures show bad-point percentage by subject and metric, depth-wise bad-point distribution, and the percentage of profiles that are empty or unreadable."
    ]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\nmetric_plot = profiles.groupby(['subject','metric'], as_index=False).agg(total_samples=('n_samples','sum'), bad_points=('bad_points','sum'))\nmetric_plot['bad_point_pct'] = 100 * metric_plot.bad_points / metric_plot.total_samples.replace(0, np.nan)\nfor metric, g in metric_plot.groupby('metric'):\n    axes[0].plot(g.subject, g.bad_point_pct, marker='o', label=metric.upper())\naxes[0].set_title('Bad-point percentage by subject'); axes[0].set_ylabel('Bad points (%)'); axes[0].tick_params(axis='x', rotation=90); axes[0].legend(); axes[0].grid(alpha=.25)\nif not depth_detail.empty:\n    d = depth_detail.groupby(['metric','depth'], as_index=False).bad_point_count.sum()\n    for metric, g in d.groupby('metric'):\n        axes[1].plot(g.depth, g.bad_point_count, marker='o', label=metric.upper())\naxes[1].set_title('Bad-point distribution by cortical depth'); axes[1].set_xlabel('Depth'); axes[1].set_ylabel('Count'); axes[1].legend(); axes[1].grid(alpha=.25)\nstatus = profiles.groupby(['metric','status']).size().unstack(fill_value=0)\nstatus.plot(kind='bar', ax=axes[2]); axes[2].set_title('Profile file status'); axes[2].set_ylabel('Files'); axes[2].tick_params(axis='x', rotation=0); axes[2].grid(axis='y', alpha=.25)\nfig.tight_layout(); fig.savefig(QC_ROOT / 'population_column_quality_overview.png', dpi=180); display(fig)"
    ]},
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}}, "nbformat": 4, "nbformat_minor": 5}
Path('qc/population_column_quality_qc.ipynb').write_text(json.dumps(nb, indent=1))
print('created qc/population_column_quality_qc.ipynb')
