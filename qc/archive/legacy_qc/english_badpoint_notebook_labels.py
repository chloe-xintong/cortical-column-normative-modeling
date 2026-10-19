import json
from pathlib import Path

p = Path('validation/bad_point_cleaning/column_badpoint_pattern_analysis.ipynb')
nb = json.loads(p.read_text())
replacements = {
    '请先运行 extract_column_badpoint_table.py 生成两个 CSV。': 'Run extract_column_badpoint_table.py first to generate the two CSV files.',
    '读取已生成的结果，不重新读取 profiles_before_clean': 'Reading cached results; raw profiles are not re-read.',
    '明细:': 'Detail:', '文件汇总:': 'File summary:', '已加载明细行:': 'Loaded detail rows:', '文件汇总行:': 'file-summary rows:',
    '读取已有 depth_pattern_summary.csv，不重新计算 depth。': 'Using existing depth_pattern_summary.csv; depth is not recomputed.',
    '使用 badpoint 明细中的 subject/hemi/region/column_id/depth，避免重新读取四种 metric 原始 CSV。': 'Using subject/hemi/region/column_id/depth from the detail table; raw metric CSVs are not re-read.',
    '## 6. Column 保留率与插值验证': '## 6. Column retention and interpolation validation',
    '本部分直接读取 `column_retention_interpolation.py` 生成的结果，不重新读取原始 profile。': 'This section reads outputs from column_retention_interpolation.py and does not re-read raw profiles.',
    '保留率定义为：允许每条 column 中最多有 0、1、2 或 3 个坏点时，仍然可以保留的 column 比例。插值验证使用原本没有坏点的 column，人为删除 1 个或 2 个内部 depth，再与真实值比较。': 'Retention is the percentage of columns kept when allowing 0, 1, 2, or 3 bad points. Interpolation validation removes one or two interior points from clean columns and compares predictions with the original values.',
    '请先运行 column_retention_interpolation.py:': 'Run column_retention_interpolation.py first:',
    '已读取 retention 和 interpolation 结果。': 'Retention and interpolation results loaded.',
    '允许的最多坏点数': 'Maximum allowed bad points',
    '人为删除的坏点数': 'Artificially removed points',
    '请先运行 extract_column_badpoint_table.py 生成两个 CSV。': 'Run extract_column_badpoint_table.py first to generate the two CSV files.',
    '读取已生成的结果，不重新读取 profiles_before_clean': 'Reading cached results; raw profiles are not re-read.',
    '明细:': 'Detail:',
    '文件汇总:': 'File summary:',
    '已加载明细行:': 'Loaded detail rows:',
    '文件汇总行:': 'file-summary rows:',
    '读取已有 depth_pattern_summary.csv，不重新计算 depth。': 'Using existing depth_pattern_summary.csv; depth is not recomputed.',
    '使用 badpoint 明细中的 subject/hemi/region/column_id/depth，避免重新读取四种 metric 原始 CSV。': 'Using subject/hemi/region/column_id/depth from the detail table; raw metric CSVs are not re-read.',
    '请先运行 column_retention_interpolation.py:': 'Run column_retention_interpolation.py first:',
    '已读取 retention 和 interpolation 结果。': 'Retention and interpolation results loaded.',
    '允许的最多坏点数': 'Maximum allowed bad points',
    '人为删除的坏点数': 'Artificially removed points',
    '人为删除的坏点数': 'Artificially removed points',
    '图像已保存:': 'Figure saved:',
}
for cell in nb['cells']:
    for old, new in replacements.items():
        cell['source'] = [line.replace(old, new) for line in cell.get('source', [])]
    for out in cell.get('outputs', []):
        if 'text' in out:
            out['text'] = [replacements.get(line, line) for line in out['text']]
p.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print('updated English labels:', p)
