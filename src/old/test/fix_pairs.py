import json

with open('src/01_correlation_参考_1.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if any('def run_analysis_action' in line for line in source):
            new_source = []
            for line in source:
                if 'pairs = [(b, c) for b in bases for c in value_cols if c != b]' in line:
                    # Remove duplicates in pairs
                    new_source.append('            pairs = []\n')
                    new_source.append('            for b in bases:\n')
                    new_source.append('                for c in value_cols:\n')
                    new_source.append('                    if c != b and (b, c) not in pairs:\n')
                    new_source.append('                        pairs.append((b, c))\n')
                else:
                    new_source.append(line)
            cell['source'] = new_source

with open('src/01_correlation_参考_1.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
