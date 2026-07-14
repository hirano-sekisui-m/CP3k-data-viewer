import json

with open('src/01_correlation_参考_1.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if any('def run_analysis_action' in line for line in source):
            new_source = []
            for line in source:
                if 'refresh_dir_btn.on_click(lambda _: refresh_input_dir_list())' in line:
                    pass # Don't need this, it's actually in another cell
                else:
                    new_source.append(line)
            cell['source'] = new_source
