import json
from pathlib import Path

p = Path('validation/column_badpoint_pattern_analysis.ipynb')
nb = json.loads(p.read_text())

nb['cells'][1]['source'] = ["""from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

QC_ROOT = Path('/mnt/newStor/paros/paros_WORK/xintong/results/column_quality_qc')
DETAIL = QC_ROOT / 'column_badpoint_detail.csv'
FILE_SUMMARY = QC_ROOT / 'column_badpoint_file_summary.csv'
DEPTHS = 21
if not DETAIL.is_file() or not FILE_SUMMARY.is_file():
    raise FileNotFoundError('请先运行 extract_column_badpoint_table.py 生成两个 CSV。')
print('读取已生成的结果，不重新读取 profiles_before_clean')
print('明细:', DETAIL)
print('文件汇总:', FILE_SUMMARY)
detail = pd.read_csv(DETAIL, usecols=['subject','hemisphere','region','metric','column_id','bad_point_count','bad_depths','bad_types','has_consecutive_bad_points'])
file_summary = pd.read_csv(FILE_SUMMARY)
print('已加载明细行:', len(detail), '文件汇总行:', len(file_summary))
"""]

nb['cells'][3]['source'] = ["""depth_path = QC_ROOT / 'depth_pattern_summary.csv'
if depth_path.is_file():
    depth_table = pd.read_csv(depth_path)
    print('读取已有 depth_pattern_summary.csv，不重新计算 depth。')
else:
    depth_rows=[]
    for row in detail.itertuples(index=False):
        if not row.bad_depths or pd.isna(row.bad_depths): continue
        for depth in str(row.bad_depths).split(';'):
            depth_rows.append({'metric':row.metric, 'depth':int(depth)})
    depth_table = pd.DataFrame(depth_rows).value_counts(['metric','depth']).rename('bad_points').reset_index()
    depth_table['total_points'] = FILE_SUMMARY.groupby('metric')['total_points'].sum().reindex(depth_table.metric).to_numpy()
    depth_table['bad_point_pct'] = 100 * depth_table.bad_points / depth_table.total_points
    depth_table.to_csv(depth_path, index=False)
display(depth_table.sort_values(['metric','depth']))
"""]

nb['cells'][5]['source'] = ["""detail['category'] = pd.cut(detail.bad_point_count, [-1,0,1,2,np.inf], labels=['0','1','2','3_or_more'])
category_table = detail.groupby(['metric','category'], observed=False).size().reset_index(name='column_count')
category_table.to_csv(QC_ROOT/'column_bad_count_summary.csv', index=False)
display(category_table)
"""]

nb['cells'][7]['source'] = ["""continuity_table = detail.groupby('metric', as_index=False).agg(
    columns_with_any_bad=('column_id','size'),
    columns_with_consecutive_bad=('has_consecutive_bad_points','sum'),
)
continuity_table['columns_all_depths_bad'] = detail.groupby('metric').bad_point_count.apply(lambda x: int((x >= DEPTHS).sum())).to_numpy()
continuity_table.to_csv(QC_ROOT/'column_continuity_summary.csv', index=False)
display(continuity_table)
"""]

nb['cells'][9]['source'] = ["""# 使用 badpoint 明细中的 subject/hemi/region/column_id/depth，避免重新读取四种 metric 原始 CSV。
keys = ['subject','hemisphere','region','column_id']
expanded = detail.assign(depth=detail.bad_depths.fillna('').str.split(';')).explode('depth')
expanded = expanded[expanded.depth.ne('')].copy()
expanded['depth'] = expanded.depth.astype(int)
overlap = expanded.groupby(keys + ['depth']).metric.nunique().reset_index(name='bad_metric_count')
overlap_summary = overlap.groupby(keys, as_index=False).agg(
    total_bad_coordinates=('depth','size'),
    bad_in_2plus_metrics=('bad_metric_count', lambda x: int((x >= 2).sum())),
    bad_in_all_4_metrics=('bad_metric_count', lambda x: int((x == 4).sum())),
)
overlap_summary.to_csv(QC_ROOT/'cross_metric_overlap_summary.csv', index=False)
display(overlap_summary.head())
"""]

nb['cells'][11]['source'] = ["""file_summary = pd.read_csv(FILE_SUMMARY)
file_summary['bad_column_pct'] = 100 * file_summary.bad_columns / file_summary.total_columns.replace(0, np.nan)
group_summary = file_summary.groupby(['subject','hemisphere','region','metric'], as_index=False).agg(
    total_columns=('total_columns','sum'), bad_points=('bad_points','sum'), bad_columns=('bad_columns','sum'),
    empty_files=('status', lambda s: int((s == 'empty').sum())))
group_summary['bad_point_pct'] = 100 * group_summary.bad_points / (group_summary.total_columns * DEPTHS).replace(0, np.nan)
group_summary['bad_column_pct'] = 100 * group_summary.bad_columns / group_summary.total_columns.replace(0, np.nan)
group_summary.to_csv(QC_ROOT/'subject_hemisphere_region_badpoint_summary.csv', index=False)
display(group_summary.sort_values('bad_point_pct', ascending=False).head(20))
"""]

p.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print('updated', p)
