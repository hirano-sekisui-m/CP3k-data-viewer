import json

with open('src/01_correlation_参考_1.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if any('color_list = []' in line for line in source):
            new_source = []
            for line in source:
                if 'if target_level == "strong_candidate": color_list.append("red")' in line:
                    new_source.append('                    if target_level == "strong_candidate": color_list.append("red")\n')
                elif 'elif target_level == "candidate": color_list.append("orange")' in line:
                    new_source.append('                    elif target_level == "candidate": color_list.append("orange")\n')
                elif 'elif target_level == "mild_candidate": color_list.append("yellow")' in line:
                    new_source.append('                    elif target_level == "mild_candidate": color_list.append("yellow")\n')
                elif 'else: color_list.append("#1f77b4")' in line:
                    new_source.append('                    else: color_list.append("#1f77b4")\n')
                else:
                    new_source.append(line)
            # cell['source'] = new_source
