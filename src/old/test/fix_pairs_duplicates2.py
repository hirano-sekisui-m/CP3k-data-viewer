import json

with open('src/01_correlation_参考_1.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if any('def run_analysis_action' in line for line in source):
            new_source = []
            for line in source:
                if 'if c != b and (b, c) not in pairs:' in line:
                    new_source.append('                    if c != b and (b, c) not in pairs and (c, b) not in pairs:\n')
                else:
                    new_source.append(line)
            cell['source'] = new_source

with open('src/01_correlation_参考_1.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
