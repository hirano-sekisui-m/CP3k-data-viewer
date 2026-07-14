import json

with open('src/01_correlation_参考_1.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if any('def run_analysis_action' in line for line in source):
            # Check for multiple event handlers
            for line in source:
                if 'on_click' in line:
                    print(line)
