"""Build Column Badpoint Pattern Notebook

Description
-----------
Build Column Badpoint Pattern Notebook

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

cells=[
{"cell_type":"markdown","metadata":{},"source":["# Column 坏点模式与可靠性 QC\n\n本 Notebook 只分析 `profiles_before_clean`，不删除、不插值、不覆盖原始 column profile。\n\n分析顺序为：\n1. 坏点位置分布；\n2. 每条 column 的坏点数量；\n3. 坏点连续性；\n4. MD、FA、AD、RD 之间的坏点重合；\n5. subject、hemisphere、region 分组。"]},
{"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":["from pathlib import Path\nfrom collections import defaultdict\nimport re, gc\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\nRESULTS_ROOT=Path('/mnt/newStor/paros/paros_WORK/xintong/results/column_analysis')\nQC_ROOT=Path('/mnt/newStor/paros/paros_WORK/xintong/results/column_quality_qc')\nQC_ROOT.mkdir(parents=True,exist_ok=True)\nPATTERN=re.compile(r'^(D\\d+)_(lh|rh)_(.+)_(md|fa|ad|rd)\\.csv$')\nMETRICS=('md','fa','ad','rd'); DEPTHS=21\nprint('读取目录:',RESULTS_ROOT)"]},
{"cell_type":"markdown","metadata":{},"source":["## 1. 坏点位于哪个 cortical depth\n\n统计每个 depth 的坏点数量和百分比。需要重点观察坏点是否集中在 depth 0 或 depth 20，还是主要出现在中间深度。\n\n注意：边界集中可能提示 mask、registration、surface sampling 或 partial-volume 问题，不应立即进行插值。"]},
{"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":["depth_bad=defaultdict(int); depth_total=defaultdict(int)\nfor folder in sorted(RESULTS_ROOT.glob('D*/profiles_before_clean')):\n  for path in folder.glob('*.csv'):\n    m=PATTERN.match(path.name)\n    if not m or path.stat().st_size==0: continue\n    try: a=np.loadtxt(path,delimiter=',',ndmin=2)\n    except Exception: continue\n    if a.shape[1]!=DEPTHS: del a; continue\n    bad=(~np.isfinite(a))|(a==0)\n    for d in range(DEPTHS):\n      depth_bad[(m.group(4),d)]+=int(bad[:,d].sum()); depth_total[(m.group(4),d)]+=int(a.shape[0])\n    del a,bad; gc.collect()\ndepth_table=pd.DataFrame([{'metric':k[0],'depth':k[1],'bad_points':v,'total_points':depth_total[k],'bad_point_pct':100*v/depth_total[k]} for k,v in depth_bad.items()])\ndepth_table.to_csv(QC_ROOT/'depth_pattern_summary.csv',index=False)\ndisplay(depth_table.sort_values(['metric','depth']).head(25))"]},
{"cell_type":"markdown","metadata":{},"source":["## 2. 每条 column 包含几个坏点\n\n将每条 column 分为 0 个、1 个、2 个和 3 个及以上坏点。\n\n如果绝大多数是单个坏点，局部插值可能可行；如果大量 column 有 3 个及以上坏点，应将这些 column 标记为低可靠性。"]},
{"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":["category_counts=defaultdict(int)\nfor folder in sorted(RESULTS_ROOT.glob('D*/profiles_before_clean')):\n  for path in folder.glob('*.csv'):\n    m=PATTERN.match(path.name)\n    if not m or path.stat().st_size==0: continue\n    try: a=np.loadtxt(path,delimiter=',',ndmin=2)\n    except Exception: continue\n    if a.shape[1]!=DEPTHS: del a; continue\n    counts=((~np.isfinite(a))|(a==0)).sum(axis=1)\n    for n in counts: category_counts[(m.group(4),'0' if n==0 else '1' if n==1 else '2' if n==2 else '3_or_more')]+=1\n    del a,counts; gc.collect()\ncategory_table=pd.DataFrame([{'metric':k[0],'category':k[1],'column_count':v} for k,v in category_counts.items()])\ncategory_table.to_csv(QC_ROOT/'column_bad_count_summary.csv',index=False)\ndisplay(category_table)"]},
{"cell_type":"markdown","metadata":{},"source":["## 3. 坏点是否连续\n\n检查同一条 column 内是否存在相邻 depth 同时为坏点。连续坏点比单个孤立坏点更难通过线性插值可靠恢复。"]},
{"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":["continuity=defaultdict(int)\nfor folder in sorted(RESULTS_ROOT.glob('D*/profiles_before_clean')):\n  for path in folder.glob('*.csv'):\n    m=PATTERN.match(path.name)\n    if not m or path.stat().st_size==0: continue\n    try: a=np.loadtxt(path,delimiter=',',ndmin=2)\n    except Exception: continue\n    if a.shape[1]!=DEPTHS: del a; continue\n    bad=(~np.isfinite(a))|(a==0)\n    continuity[(m.group(4),'columns_with_any_bad')]+=int(np.any(bad,axis=1).sum())\n    continuity[(m.group(4),'columns_with_consecutive_bad')]+=int(np.any(bad[:,:-1]&bad[:,1:],axis=1).sum())\n    continuity[(m.group(4),'columns_all_depths_bad')]+=int(np.all(bad,axis=1).sum())\n    del a,bad; gc.collect()\ncontinuity_table=pd.DataFrame([{'metric':k[0],'category':k[1],'column_count':v} for k,v in continuity.items()])\ncontinuity_table.to_csv(QC_ROOT/'column_continuity_summary.csv',index=False)\ndisplay(continuity_table)"]},
{"cell_type":"markdown","metadata":{},"source":["## 4. 坏点是否跨 metric 重合\n\n对相同 subject、hemisphere、region、column 和 depth，检查 MD、FA、AD、RD 是否在同一位置同时异常。\n\n只有一个 metric 异常，可能是局部 metric 数值问题；多个 metric 同时异常，则更需要检查 mask、registration 或 surface sampling。"]},
{"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":["overlap_rows=[]\nfor folder in sorted(RESULTS_ROOT.glob('D*/profiles_before_clean')):\n  groups=defaultdict(dict)\n  for path in folder.glob('*.csv'):\n    m=PATTERN.match(path.name)\n    if m and path.stat().st_size: groups[(m.group(1),m.group(2),m.group(3))][m.group(4)]=path\n  for key,paths in groups.items():\n    if set(paths)!=set(METRICS): continue\n    try: arrays=[((lambda a:(~np.isfinite(a))|(a==0))(np.loadtxt(paths[x],delimiter=',',ndmin=2))) for x in METRICS]\n    except Exception: continue\n    if len({a.shape for a in arrays})!=1: continue\n    stack=np.stack(arrays); overlap_rows.append({'subject':key[0],'hemisphere':key[1],'region':key[2],'total_coordinates':int(stack.shape[1]*stack.shape[2]),'bad_in_any_metric':int(np.any(stack,axis=0).sum()),'bad_in_2plus_metrics':int((stack.sum(axis=0)>=2).sum()),'bad_in_all_4_metrics':int(np.all(stack,axis=0).sum())})\n    del arrays,stack; gc.collect()\noverlap=pd.DataFrame(overlap_rows)\nif not overlap.empty:\n  overlap['all_4_metric_overlap_pct']=100*overlap.bad_in_all_4_metrics/overlap.total_coordinates\noverlap.to_csv(QC_ROOT/'cross_metric_overlap_summary.csv',index=False)\ndisplay(overlap.head())"]},
{"cell_type":"markdown","metadata":{},"source":["## 5. Subject、hemisphere 和 region 分组\n\n汇总坏点比例、含坏点 column 比例和连续坏点 column 数，寻找集中异常的 subject、半球或脑区。"]},
{"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":["file_summary=pd.read_csv(QC_ROOT/'file_badpoint_summary.csv')\nfile_summary['bad_column_pct']=100*file_summary.bad_columns/file_summary.total_columns.replace(0,np.nan)\ngroup_summary=file_summary.groupby(['subject','hemisphere','region','metric'],as_index=False).agg(total_columns=('total_columns','sum'),bad_points=('bad_points','sum'),bad_columns=('bad_columns','sum'),consecutive_columns=('consecutive_columns','sum'),empty_files=('status',lambda s:int((s=='empty').sum())))\ngroup_summary['bad_point_pct']=100*group_summary.bad_points/(group_summary.total_columns*DEPTHS).replace(0,np.nan)\ngroup_summary['bad_column_pct']=100*group_summary.bad_columns/group_summary.total_columns.replace(0,np.nan)\ngroup_summary.to_csv(QC_ROOT/'subject_hemisphere_region_badpoint_summary.csv',index=False)\ndisplay(group_summary.sort_values('bad_point_pct',ascending=False).head(20))"]}
]
Path('validation/bad_point_cleaning/column_badpoint_pattern_analysis.ipynb').write_text(json.dumps({'cells':cells,'metadata':{'kernelspec':{'display_name':'Python (cortical_trajectory_analysis)','language':'python','name':'cortical_trajectory_analysis'}},'nbformat':4,'nbformat_minor':5},indent=1))
print('created validation/bad_point_cleaning/column_badpoint_pattern_analysis.ipynb')
