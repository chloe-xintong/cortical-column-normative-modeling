import json
from pathlib import Path

p = Path('validation/column_badpoint_pattern_analysis.ipynb')
nb = json.loads(p.read_text())
nb['cells'] = [c for c in nb['cells'] if 'Union bad-column summary' not in ''.join(c.get('source', []))]
nb['cells'].extend([
    {'cell_type':'markdown','metadata':{},'source':['## 13. Union bad-column summary\n\nA trajectory column is affected if any of MD, FA, AD, or RD contains at least one bad depth. Each trajectory column is counted only once across metrics.']},
    {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':["""union_keys = ['subject', 'hemisphere', 'region', 'column_id']
union_bad = detail[detail.bad_point_count > 0][union_keys].drop_duplicates()
valid_files = file_summary[file_summary.status.eq('valid')]
denom = valid_files.groupby(['subject','hemisphere','region'], as_index=False).total_columns.first()
affected_counts = union_bad.groupby(['subject','hemisphere','region'], as_index=False).size().rename(columns={'size':'affected_columns'})
denom = denom.merge(affected_counts, on=['subject','hemisphere','region'], how='left')
denom['affected_columns'] = denom.affected_columns.fillna(0).astype(int)
denom['unaffected_columns'] = denom.total_columns - denom.affected_columns
denom['affected_column_pct'] = 100 * denom.affected_columns / denom.total_columns.replace(0, np.nan)
denom.to_csv(QC_ROOT / 'union_bad_column_by_subject_region.csv', index=False)
subject_union = denom.groupby('subject', as_index=False).agg(total_columns=('total_columns','sum'), affected_columns=('affected_columns','sum'), unaffected_columns=('unaffected_columns','sum'))
subject_union['affected_column_pct'] = 100 * subject_union.affected_columns / subject_union.total_columns.replace(0, np.nan)
subject_union.to_csv(QC_ROOT / 'subject_union_bad_column_summary.csv', index=False)
cohort_union = pd.DataFrame([{'cohort':'all_subjects', 'total_columns':int(subject_union.total_columns.sum()), 'affected_columns':int(subject_union.affected_columns.sum()), 'unaffected_columns':int(subject_union.unaffected_columns.sum())}])
cohort_union['affected_column_pct'] = 100 * cohort_union.affected_columns / cohort_union.total_columns
cohort_union.to_csv(QC_ROOT / 'cohort_union_bad_column_summary.csv', index=False)
display(cohort_union); display(subject_union.nlargest(10, 'affected_column_pct'))
"""]}
])
p.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print('updated', p)
