import json
from pathlib import Path

p = Path('validation/column_badpoint_pattern_analysis.ipynb')
nb = json.loads(p.read_text())
nb['cells'] = [c for c in nb['cells'] if 'Top five subject-specific heatmaps' not in ''.join(c.get('source', []))]
nb['cells'].extend([
    {'cell_type':'markdown','metadata':{},'source':['## 9. Top five subject-specific heatmaps\n\nThe cohort heatmap above is a population median. This section creates one region-by-depth heatmap for each of the five subjects with the highest bad-point percentage.']},
    {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':["""from IPython.display import display
top_subjects = retention.nlargest(5, 'bad_point_pct').subject.tolist()
subject_heatmap_dir = QC_ROOT / 'subject_heatmaps'
subject_heatmap_dir.mkdir(exist_ok=True)
for subject in top_subjects:
    sub_meta = valid_meta[valid_meta.subject.eq(subject)]
    sub_denom = sub_meta.groupby(['hemisphere','region'], as_index=False).total_columns.first()
    sub_expanded = expanded[expanded.subject.eq(subject)].copy()
    sub_bad = sub_expanded.groupby(['hemisphere','region','column_id','depth']).size().reset_index(name='metric_hits')
    sub_bad = sub_bad.groupby(['hemisphere','region','depth'], as_index=False).size().rename(columns={'size':'bad_count'})
    sub_heat = sub_bad.merge(sub_denom, on=['hemisphere','region'], how='left')
    sub_heat['bad_pct'] = 100 * sub_heat.bad_count / sub_heat.total_columns
    sub_heat = sub_heat.groupby(['region','depth'], as_index=False).bad_pct.max()
    sub_pivot = sub_heat.pivot(index='region', columns='depth', values='bad_pct').fillna(0)
    sub_pivot = sub_pivot.loc[sub_pivot.mean(axis=1).sort_values().index]
    sub_pivot.to_csv(subject_heatmap_dir / f'{subject}_region_depth_union_badpoint.csv')
    fig, ax = plt.subplots(figsize=(14, max(8, .23 * len(sub_pivot))))
    values = sub_pivot.to_numpy()
    im = ax.imshow(values, aspect='auto', cmap='magma', vmin=0, vmax=max(1, float(np.nanpercentile(values, 99))))
    ax.set_title(f'{subject}: bad-point percentage by region and depth')
    ax.set_xlabel('Depth'); ax.set_ylabel('Region')
    ax.set_xticks(range(len(sub_pivot.columns))); ax.set_xticklabels(sub_pivot.columns)
    ax.set_yticks(range(len(sub_pivot.index))); ax.set_yticklabels(sub_pivot.index)
    fig.colorbar(im, ax=ax, label='Bad-point percentage (%)')
    fig.tight_layout(); fig.savefig(subject_heatmap_dir / f'{subject}_region_depth_union_badpoint.png', dpi=180)
    display(fig); plt.close(fig)
print('Saved subject heatmaps:', top_subjects)
"""]}
])
p.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print('updated', p)
