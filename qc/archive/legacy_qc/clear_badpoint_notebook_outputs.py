"""Clear Badpoint Notebook Outputs

Description
-----------
Clear Badpoint Notebook Outputs

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

p = Path('validation/bad_point_cleaning/column_badpoint_pattern_analysis.ipynb')
nb = json.loads(p.read_text())
for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        cell['execution_count'] = None
        cell['outputs'] = []
p.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print('cleared saved outputs:', p)
