import json

with open('src/01_correlation_参考_1.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if any('def run_analysis_action' in line for line in source):
            new_source = []
            for line in source:
                if 'pairs = []' in line:
                    new_source.append('            pairs = []\n')
                elif 'for b in bases:' in line:
                    new_source.append('            for b in bases:\n')
                elif 'for c in value_cols:' in line:
                    new_source.append('                for c in value_cols:\n')
                elif 'if c != b and (b, c) not in pairs:' in line:
                    # we also want to avoid (c,b) being processed when (b,c) was processed. But wait, it's comparing baseline against the rest.
                    # so if bases has TAT1, TAT2, then we do (TAT1, TAT3), (TAT2, TAT3), but also (TAT1, TAT2) and (TAT2, TAT1).
                    # Actually (TAT1, TAT2) and (TAT2, TAT1) are different regressions.
                    # Wait, the problem is pairs containing duplicate exact tuples like (TAT1, TAT2) multiple times.
                    # This is avoided by (b,c) not in pairs.
                    new_source.append('                    if c != b and (b, c) not in pairs:\n')
                elif 'pairs.append((b, c))' in line:
                    new_source.append('                        pairs.append((b, c))\n')
                else:
                    new_source.append(line)
            cell['source'] = new_source

with open('src/01_correlation_参考_1.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
