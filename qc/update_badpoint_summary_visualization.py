import json
from pathlib import Path

p = Path('validation/bad_point_cleaning/column_badpoint_pattern_analysis.ipynb')
nb = json.loads(p.read_text())
# Keep the original analysis cells and replace the generated visualization section
# instead of appending duplicate sections on every rebuild.
nb['cells'] = nb['cells'][:12]
while len(nb['cells']) < 15:
    nb['cells'].append({'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': []})

nb['cells'][12]['source'] = ["""## 6. Retention and interpolation results

This section reads cached CSV results only. It does not re-read raw profiles.
"""]
nb['cells'][13]['source'] = ["""retention_path = QC_ROOT / 'subject_column_retention_curve.csv'
population_path = QC_ROOT / 'population_column_retention_summary.csv'
interp_path = QC_ROOT / 'interpolation_validation_summary.csv'
for required in (retention_path, population_path, interp_path):
    if not required.is_file():
        raise FileNotFoundError(f'Run column_retention_interpolation.py first: {required}')
retention = pd.read_csv(retention_path)
population_retention = pd.read_csv(population_path)
interpolation = pd.read_csv(interp_path)
print('Cached retention and interpolation results loaded.')
display(population_retention)
display(interpolation)
"""]
nb['cells'][14]['source'] = ["""plt.close('all')
allowed = list(range(6))
median = [population_retention.loc[population_retention.allowed_bad_points.eq(k), 'median_retention_pct'].iloc[0] for k in allowed if k in set(population_retention.allowed_bad_points)]
q1 = [population_retention.loc[population_retention.allowed_bad_points.eq(k), 'q1_retention_pct'].iloc[0] for k in allowed if k in set(population_retention.allowed_bad_points)]
q3 = [population_retention.loc[population_retention.allowed_bad_points.eq(k), 'q3_retention_pct'].iloc[0] for k in allowed if k in set(population_retention.allowed_bad_points)]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes[0,0].plot(allowed[:len(median)], median, marker='o', linewidth=2, label='Median')
axes[0,0].fill_between(allowed[:len(median)], q1, q3, alpha=.25, label='IQR')
axes[0,0].set(title='Population column retention curve', xlabel='Maximum allowed bad points', ylabel='Retention (%)', xticks=allowed)
axes[0,0].set_ylim(0, 100); axes[0,0].grid(alpha=.25); axes[0,0].legend()
for k in allowed:
    col = f'retention_allowed_{k}'
    if col in retention: axes[0,1].boxplot(retention[col].dropna(), positions=[k], widths=.55)
axes[0,1].set(title='Retention distribution across subjects', xlabel='Maximum allowed bad points', ylabel='Retention (%)', xticks=allowed)
axes[0,1].set_ylim(0,100); axes[0,1].grid(alpha=.25)
outliers = retention.nlargest(10, 'bad_point_pct')
for _, row in outliers.iterrows():
    available = [k for k in allowed if f'retention_allowed_{k}' in retention.columns]
    axes[1,0].plot(available, [row[f'retention_allowed_{k}'] for k in available], marker='o', label=row.subject)
axes[1,0].set(title='Highest bad-point subjects', xlabel='Maximum allowed bad points', ylabel='Retention (%)', xticks=allowed)
axes[1,0].set_ylim(0,100); axes[1,0].grid(alpha=.25); axes[1,0].legend(fontsize=8, ncol=2)
if {'metric','mae_mean'}.issubset(interpolation.columns):
    for metric, group in interpolation.groupby('metric'):
        axes[1,1].plot(group.artificial_bad_points, group.nmae_mean, marker='o', label=metric.upper())
    axes[1,1].set_ylabel('Normalized mean absolute error (NMAE)')
else:
    axes[1,1].bar(interpolation.artificial_bad_points.astype(str), interpolation.mae_mean)
    axes[1,1].set_ylabel('Mean absolute error (MAE)')
axes[1,1].set(title='Interpolation validation', xlabel='Artificially removed points'); axes[1,1].grid(alpha=.25); axes[1,1].legend()
fig.tight_layout(); out = QC_ROOT / 'retention_interpolation_overview.png'; fig.savefig(out, dpi=180, bbox_inches='tight'); display(fig)
"""]

nb['cells'].extend([
    {'cell_type':'markdown','metadata':{},'source':['## 7. Bad points per affected column\n\nThis shows how many bad depths occur within columns that contain at least one bad point.']},
    {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':["""affected = detail.loc[detail.bad_point_count > 0, 'bad_point_count'].copy()
bins = pd.cut(affected, bins=[0,1,2,3,10,20,21], labels=['1','2','3','4-10','11-20','21'], include_lowest=False)
column_bad_distribution = bins.value_counts(sort=False).rename_axis('bad_depths_per_column').reset_index(name='column_count')
column_bad_distribution['percentage_of_affected_columns'] = 100 * column_bad_distribution.column_count / column_bad_distribution.column_count.sum()
total_columns_all = file_summary.loc[file_summary.status.eq('valid'), 'total_columns'].sum()
column_bad_distribution['percentage_of_all_columns'] = 100 * column_bad_distribution.column_count / total_columns_all
column_bad_distribution.to_csv(QC_ROOT / 'bad_depths_per_affected_column_summary.csv', index=False)
display(column_bad_distribution)
fig, ax = plt.subplots(figsize=(8,5)); ax.bar(column_bad_distribution.bad_depths_per_column.astype(str), column_bad_distribution.percentage_of_affected_columns)
ax.set(title='Bad depths per affected column', xlabel='Number of bad depths in a column', ylabel='Affected columns (%)'); ax.grid(axis='y', alpha=.25)
fig.tight_layout(); fig.savefig(QC_ROOT / 'bad_depths_per_affected_column.png', dpi=180); display(fig)
"""]},
    {'cell_type':'markdown','metadata':{},'source':['## 8. Region by depth heatmap\n\nThe heatmap uses the union of bad points across the four metrics and displays the cohort median percentage for each region and depth.']},
    {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':["""file_meta = pd.read_csv(FILE_SUMMARY, usecols=['subject','hemisphere','region','metric','total_columns','status'])
valid_meta = file_meta[file_meta.status.eq('valid')]
denom = valid_meta.groupby(['subject','hemisphere','region'], as_index=False).total_columns.first()
expanded = detail.assign(depth=detail.bad_depths.fillna('').str.split(';')).explode('depth')
expanded = expanded[expanded.depth.ne('')].copy(); expanded['depth'] = expanded.depth.astype(int)
union = expanded.groupby(['subject','hemisphere','region','column_id','depth']).size().reset_index(name='metric_hits')
bad_counts = union.groupby(['subject','hemisphere','region','depth'], as_index=False).size().rename(columns={'size':'bad_count'})
heat = bad_counts.merge(denom, on=['subject','hemisphere','region'], how='left')
heat['bad_pct'] = 100 * heat.bad_count / heat.total_columns
heat = heat.groupby(['region','depth'], as_index=False).bad_pct.median()
pivot = heat.pivot(index='region', columns='depth', values='bad_pct').fillna(0)
pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]
pivot.to_csv(QC_ROOT / 'region_depth_union_badpoint_heatmap.csv')
fig, ax = plt.subplots(figsize=(14, max(8, .23 * len(pivot))))
im = ax.imshow(pivot.to_numpy(), aspect='auto', cmap='magma', vmin=0, vmax=max(1, float(np.nanpercentile(pivot.to_numpy(), 99))))
ax.set(title='Cohort median bad-point percentage by region and depth', xlabel='Depth', ylabel='Region', xticks=range(len(pivot.columns)), xticklabels=pivot.columns, yticklabels=pivot.index)
fig.colorbar(im, ax=ax, label='Bad-point percentage (%)'); fig.tight_layout(); fig.savefig(QC_ROOT / 'region_depth_union_badpoint_heatmap.png', dpi=180); display(fig)
"""]}
])
p.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print('updated', p)
