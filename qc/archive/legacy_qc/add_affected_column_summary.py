import json
from pathlib import Path

p = Path('validation/bad_point_cleaning/column_badpoint_pattern_analysis.ipynb')
nb = json.loads(p.read_text())
nb['cells'] = [c for c in nb['cells'] if 'Affected column summary' not in ''.join(c.get('source', []))]
nb['cells'].extend([
    {'cell_type':'markdown','metadata':{},'source':['## 12. Affected column summary\n\nA column is affected if it contains at least one bad depth. This table reports affected columns as a percentage of all valid columns.']},
    {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':["""affected_rows = []
for label, frame in [('all_subjects', retention), ('excluding_top5', retention_keep)]:
    total = int(frame.total_columns.sum())
    affected = int(frame.bad_columns.sum())
    affected_rows.append({'cohort': label, 'total_columns': total,
                          'unaffected_columns': total - affected,
                          'affected_columns': affected,
                          'affected_column_pct': 100 * affected / total})
affected_summary = pd.DataFrame(affected_rows)
affected_summary.to_csv(QC_ROOT / 'affected_column_summary.csv', index=False)
display(affected_summary)
fig, ax = plt.subplots(figsize=(7,5))
plot = affected_summary.set_index('cohort')[['unaffected_columns','affected_columns']]
plot.plot(kind='bar', stacked=True, ax=ax, color=['#9ecae1','#de2d26'])
ax.set_title('Affected versus unaffected columns')
ax.set_xlabel('Cohort'); ax.set_ylabel('Number of columns'); ax.legend(['Unaffected (0 bad points)', 'Affected (at least 1 bad point)'])
ax.tick_params(axis='x', rotation=0); fig.tight_layout()
fig.savefig(QC_ROOT / 'affected_column_summary.png', dpi=180, bbox_inches='tight'); display(fig); plt.close(fig)
"""]}
])
p.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print('updated', p)
