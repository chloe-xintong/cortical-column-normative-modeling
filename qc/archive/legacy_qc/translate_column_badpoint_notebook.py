"""Translate Column Badpoint Notebook

Description
-----------
Translate Column Badpoint Notebook

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

p=Path('validation/bad_point_cleaning/column_badpoint_pattern_analysis.ipynb')
n=json.loads(p.read_text())
markdown={
0:"# Column 坏点模式与可靠性质量控制\n\n本 Notebook 只分析 `profiles_before_clean`，不删除、不插值、不覆盖原始 column profile。\n\n分析顺序为：\n1. 坏点所在的 cortical depth；\n2. 每条 column 的坏点数量；\n3. 坏点是否连续；\n4. MD、FA、AD、RD 之间的坏点重合；\n5. subject、半球和脑区分组。",
2:"## 1. 坏点所在的 cortical depth\n\n统计每个 depth 的坏点数量和百分比，观察坏点是否集中在 depth 0、depth 20 或中间深度。\n\n边界集中可能提示 mask、registration、surface sampling 或 partial-volume 问题，因此不能直接进行插值。",
4:"## 2. 每条 column 包含几个坏点\n\n将每条 column 分为 0 个、1 个、2 个和 3 个及以上坏点。\n\n如果大多数是单个坏点，局部插值可能可行；如果大量 column 有 3 个及以上坏点，应优先标记为低可靠性。",
6:"## 3. 坏点是否连续\n\n检查同一条 column 内是否存在相邻 depth 同时为坏点。连续坏点比单个孤立坏点更难通过线性插值可靠恢复。",
8:"## 4. 坏点是否跨 metric 重合\n\n对相同 subject、半球、脑区、column 和 depth，检查 MD、FA、AD、RD 是否在同一位置同时异常。\n\n只有一个 metric 异常，可能是局部 metric 数值问题；多个 metric 同时异常，则需要检查 mask、registration 或 surface sampling。",
10:"## 5. Subject、半球和脑区分组\n\n汇总坏点比例、含坏点 column 比例和连续坏点 column 数，用于寻找集中异常的 subject、半球或脑区。"
}
for i,text in markdown.items():
    n['cells'][i]['cell_type']='markdown'
    n['cells'][i]['source']=[line+'\n' for line in text.splitlines()]
p.write_text(json.dumps(n,indent=1))
print('translated',p)
