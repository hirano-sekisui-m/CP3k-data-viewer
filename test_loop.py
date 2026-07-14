import json

with open('src/01_correlation_参考_1.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if any('summary_rows, outlier_tables' in line for line in source):
            print("Found loop cell")
            for i, line in enumerate(source):
                if 'for method in methods:' in line:
                    for j in range(i-10, i+20):
                        print(f"{j}: {source[j]}", end="")
