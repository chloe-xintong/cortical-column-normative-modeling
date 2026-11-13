"""Update Badpoint Four Plots

Description
-----------
Update Badpoint Four Plots

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

import json
from pathlib import Path

p = Path('qc/column_profile_badpoint_analysis.ipynb')
nb = json.loads(p.read_text())
source = """from IPython.display import Image, display
import matplotlib.pyplot as plt
plt.close('all')

subject_summary = pd.read_csv(QC_ROOT / 'subject_badpoint_summary.csv')
depth_summary = pd.read_csv(QC_ROOT / 'depth_bad_point_summary.csv')

fig, axes = plt.subplots(2, 2, figsize=(18, 10))
axes = axes.ravel()

# 1. All subjects
all_subjects = subject_summary.sort_values('subject')
all_subjects.plot(x='subject', y='bad_point_pct', kind='bar', ax=axes[0], legend=False)
axes[0].set_title('Bad-point percentage by subject')
axes[0].set_ylabel('Bad points (%)')
axes[0].set_xlabel('Subject (labels hidden)')
axes[0].set_xticks([])

# 2. Top 20 subjects
top = subject_summary.nlargest(20, 'bad_point_pct').sort_values('bad_point_pct')
axes[1].barh(top['subject'], top['bad_point_pct'])
axes[1].set_title('Top 20 subjects by bad-point percentage')
axes[1].set_xlabel('Bad points (%)')
axes[1].grid(axis='x', alpha=0.25)

# 3. Absolute count by depth
for metric, group in depth_summary.groupby('metric'):
    group = group.sort_values('depth')
    axes[2].plot(group['depth'], group['bad_point_count'], marker='o', label=metric.upper())
axes[2].set_title('Bad-point count by depth')
axes[2].set_xlabel('Depth')
axes[2].set_ylabel('Bad-point count')
axes[2].legend()
axes[2].grid(alpha=0.25)

# 4. Normalized percentage by depth
file_summary = pd.read_csv(QC_ROOT / 'file_badpoint_summary.csv')
total_by_metric = file_summary.groupby('metric')['total_columns'].sum() * DEPTHS
for metric, group in depth_summary.groupby('metric'):
    group = group.sort_values('depth').copy()
    group['bad_point_pct'] = 100 * group['bad_point_count'] / total_by_metric.get(metric, np.nan)
    axes[3].plot(group['depth'], group['bad_point_pct'], marker='o', label=metric.upper())
axes[3].set_title('Normalized bad-point percentage by depth')
axes[3].set_xlabel('Depth')
axes[3].set_ylabel('Bad points (%)')
axes[3].legend()
axes[3].grid(alpha=0.25)

fig.tight_layout()
out = QC_ROOT / 'population_badpoint_overview.png'
fig.savefig(out, dpi=180)
plt.close(fig)
display(Image(filename=str(out)))
"""
for cell in reversed(nb['cells']):
    if cell.get('cell_type') == 'code':
        cell['source'] = source.splitlines(True)
        break
p.write_text(json.dumps(nb, indent=1))
print('updated', p)
