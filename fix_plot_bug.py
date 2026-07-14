import json

with open('src/01_correlation_参考_1.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if any('color_list = []' in line for line in source):
            print("Found loop:")
            for i, line in enumerate(source):
                if 'color_list = []' in line or 'color_list.append' in line or 'for _, row in metrics_df.iterrows():' in line or 'df_pair = apply_value_range_filter' in line or 'df_pair[[xcol, ycol]].dropna()' in line:
                    print(line.strip())
