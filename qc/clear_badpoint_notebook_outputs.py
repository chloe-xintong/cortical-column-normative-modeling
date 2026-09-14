import json
from pathlib import Path

p = Path('validation/column_badpoint_pattern_analysis.ipynb')
nb = json.loads(p.read_text())
for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        cell['execution_count'] = None
        cell['outputs'] = []
p.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print('cleared saved outputs:', p)
