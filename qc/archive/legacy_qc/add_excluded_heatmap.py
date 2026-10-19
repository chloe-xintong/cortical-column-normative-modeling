import json
from pathlib import Path

p = Path('validation/bad_point_cleaning/column_badpoint_pattern_analysis.ipynb')
nb = json.loads(p.read_text())
nb['cells'] = [c for c in nb['cells'] if 'Cohort heatmap excluding top five subjects' not in ''.join(c.get('source', []))]
nb['cells'].extend([
    {'cell_type':'markdown','metadata':{},'source':['## 11. Cohort heatmap excluding top five subjects\n\nThis heatmap shows the cohort median region-by-depth bad-point percentage after excluding the five highest-bad-point subjects.']},
    {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':["""exclude_heatmap_dir = QC_ROOT / 'excluding_top5'
exclude_heatmap_dir.mkdir(exist_ok=True)
meta_keep = file_summary[(file_summary.status.eq('valid')) & (~file_summary.subject.isin(excluded_subjects))]
denom_keep = meta_keep.groupby(['subject','hemisphere','region'], as_index=False).total_columns.first()
expanded_keep = detail_keep.assign(depth=detail_keep.bad_depths.fillna('').str.split(';')).explode('depth')
expanded_keep = expanded_keep[expanded_keep.depth.ne('')].copy(); expanded_keep['depth'] = expanded_keep.depth.astype(int)
union_keep = expanded_keep.groupby(['subject','hemisphere','region','column_id','depth']).size().reset_index(name='metric_hits')
bad_keep = union_keep.groupby(['subject','hemisphere','region','depth']).size().reset_index(name='bad_count')
heat_keep = bad_keep.merge(denom_keep, on=['subject','hemisphere','region'], how='left')
heat_keep['bad_pct'] = 100 * heat_keep.bad_count / heat_keep.total_columns
heat_keep = heat_keep.groupby(['region','depth'], as_index=False).bad_pct.median()
pivot_keep = heat_keep.pivot(index='region', columns='depth', values='bad_pct').fillna(0)
pivot_keep = pivot_keep.loc[pivot_keep.mean(axis=1).sort_values().index]
pivot_keep.to_csv(exclude_heatmap_dir / 'region_depth_union_badpoint_heatmap.csv')
fig, ax = plt.subplots(figsize=(14, max(8, .23 * len(pivot_keep))))
values = pivot_keep.to_numpy()
im = ax.imshow(values, aspect='auto', cmap='magma', vmin=0, vmax=max(1, float(np.nanpercentile(values, 99))))
ax.set_title('Cohort median bad-point percentage excluding top five subjects')
ax.set_xlabel('Depth'); ax.set_ylabel('Region')
ax.set_xticks(range(len(pivot_keep.columns))); ax.set_xticklabels(pivot_keep.columns)
ax.set_yticks(range(len(pivot_keep.index))); ax.set_yticklabels(pivot_keep.index)
fig.colorbar(im, ax=ax, label='Bad-point percentage (%)')
fig.tight_layout(); fig.savefig(exclude_heatmap_dir / 'region_depth_union_badpoint_heatmap.png', dpi=180, bbox_inches='tight'); display(fig); plt.close(fig)
print('Saved:', exclude_heatmap_dir / 'region_depth_union_badpoint_heatmap.png')
"""]}
])
p.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print('updated', p)
