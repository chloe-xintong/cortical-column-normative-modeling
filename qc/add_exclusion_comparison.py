import json
from pathlib import Path

p = Path('validation/column_badpoint_pattern_analysis.ipynb')
nb = json.loads(p.read_text())
nb['cells'] = [c for c in nb['cells'] if 'Exclusion sensitivity analysis' not in ''.join(c.get('source', []))]
nb['cells'].extend([
    {'cell_type':'markdown','metadata':{},'source':['## 10. Exclusion sensitivity analysis\n\nThis compares the full cohort with the cohort after excluding the five subjects with the highest bad-point percentage. Original results are not overwritten.']},
    {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':["""excluded_subjects = retention.nlargest(5, 'bad_point_pct').subject.tolist()
comparison_root = QC_ROOT / 'excluding_top5'
comparison_root.mkdir(exist_ok=True)
retention_keep = retention[~retention.subject.isin(excluded_subjects)].copy()
detail_keep = detail[~detail.subject.isin(excluded_subjects)].copy()
file_keep = file_summary[~file_summary.subject.isin(excluded_subjects)].copy()

retention_summary = []
for label, frame in [('all_subjects', retention), ('excluding_top5', retention_keep)]:
    available_k = [k for k in range(6) if f'retention_allowed_{k}' in retention.columns]
    for k in available_k:
        col = f'retention_allowed_{k}'
        retention_summary.append({'cohort':label, 'allowed_bad_points':k,
            'median_retention_pct':frame[col].median(), 'q1_retention_pct':frame[col].quantile(.25),
            'q3_retention_pct':frame[col].quantile(.75), 'subjects':len(frame)})
pd.DataFrame(retention_summary).to_csv(comparison_root / 'retention_comparison_summary.csv', index=False)

affected_all = detail.bad_point_count[detail.bad_point_count > 0]
affected_keep = detail_keep.bad_point_count[detail_keep.bad_point_count > 0]
def distribution(series, label):
    categories = pd.cut(series, [0,1,2,3,10,20,21], labels=['1','2','3','4-10','11-20','21'])
    out = categories.value_counts(sort=False).rename_axis('bad_depths_per_column').reset_index(name='column_count')
    out['percentage_of_affected_columns'] = 100 * out.column_count / out.column_count.sum()
    out['cohort'] = label
    return out
dist_compare = pd.concat([distribution(affected_all, 'all_subjects'), distribution(affected_keep, 'excluding_top5')], ignore_index=True)
dist_compare.to_csv(comparison_root / 'bad_depth_distribution_comparison.csv', index=False)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for label, frame in [('all_subjects', retention), ('excluding_top5', retention_keep)]:
    summary = retention_summary
    sub = pd.DataFrame(summary)
    sub = sub[sub.cohort.eq(label)].sort_values('allowed_bad_points')
    axes[0].plot(sub.allowed_bad_points, sub.median_retention_pct, marker='o', label=label)
    axes[0].fill_between(sub.allowed_bad_points, sub.q1_retention_pct, sub.q3_retention_pct, alpha=.15)
axes[0].set(title='Retention comparison', xlabel='Maximum allowed bad points', ylabel='Median retention (%)', xticks=range(6)); axes[0].set_ylim(0,100); axes[0].grid(alpha=.25); axes[0].legend()
for label, group in dist_compare.groupby('cohort'):
    axes[1].plot(group.bad_depths_per_column.astype(str), group.percentage_of_affected_columns, marker='o', label=label)
axes[1].set(title='Bad depths per affected column', xlabel='Bad depths in one column', ylabel='Affected columns (%)'); axes[1].grid(alpha=.25); axes[1].legend()
valid_all = file_summary[file_summary.status.eq('valid')]
valid_keep = file_keep[file_keep.status.eq('valid')]
for label, frame, det in [('all_subjects', valid_all, detail), ('excluding_top5', valid_keep, detail_keep)]:
    den = frame.groupby(['metric','subject'], as_index=False).total_columns.sum()
    ex = det.assign(depth=det.bad_depths.fillna('').str.split(';')).explode('depth'); ex = ex[ex.depth.ne('')].copy(); ex.depth = ex.depth.astype(int)
    bad = ex.groupby(['metric','subject','depth']).size().reset_index(name='bad_count')
    d = bad.groupby(['metric','depth'], as_index=False).bad_count.sum(); totals = den.groupby('metric').total_columns.sum()
    d['bad_pct'] = d.apply(lambda r: 100*r.bad_count/(totals[r.metric]), axis=1)
    for metric, g in d.groupby('metric'): axes[2].plot(g.depth, g.bad_pct, marker='o', label=f'{label}-{metric.upper()}')
axes[2].set(title='Depth-level bad-point percentage', xlabel='Depth', ylabel='Bad points (%)'); axes[2].grid(alpha=.25); axes[2].legend(fontsize=7, ncol=2)
fig.tight_layout(); fig.savefig(comparison_root / 'excluding_top5_comparison.png', dpi=180, bbox_inches='tight'); display(fig)
print('Excluded subjects:', excluded_subjects)
print('Comparison outputs:', comparison_root)
"""]}
])
p.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print('updated', p)
