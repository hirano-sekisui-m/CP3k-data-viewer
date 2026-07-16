import json

with open('src/01_correlation_参考_1.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if any('analyze_btn.on_click(' in line for line in source) and not any('def run_analysis_action' in line for line in source):
            print("Found events cell")
            print("".join(source))
