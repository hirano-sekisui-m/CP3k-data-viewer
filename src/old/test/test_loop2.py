import json

with open('src/01_correlation_参考_1.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if any('summary_rows, outlier_tables' in line for line in source):
            for i, line in enumerate(source):
                if 'with status: print(f"解析完了!' in line:
                    for j in range(i-5, i+5):
                        try:
                            print(f"{j}: {source[j]}", end="")
                        except: pass
