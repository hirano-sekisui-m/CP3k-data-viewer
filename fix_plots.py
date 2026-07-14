import json

with open('src/01_correlation_参考_1.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        new_source = []
        for line in source:
            if 'with plots_out: clear_output()' in line:
                new_source.append('    with plots_out: clear_output()\n')
                # we also need to clear status
            elif 'with status: clear_output()' in line:
                new_source.append('    with status: clear_output()\n')
            else:
                new_source.append(line)
        cell['source'] = new_source

with open('src/01_correlation_参考_1.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
