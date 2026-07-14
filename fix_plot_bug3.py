import json

with open('src/01_correlation_参考_1.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if any('metrics_df = compute_pair_sample_metrics' in line for line in source):
            new_source = []
            for line in source:
                if 'metrics_df = compute_pair_sample_metrics(df_pair' in line:
                    # In plot_suite, x and y are computed from sub (which is dropped NA of df[xcol, ycol])
                    # And then ~is_flagged uses indices from sub.
                    # The color_list is passed to plot_suite. It MUST be the exact same length as sub
                    # But metrics_df has length of sub (since it does dropna itself inside compute_pair_sample_metrics).
                    # Actually compute_pair_sample_metrics does dropna on subset=[xcol,ycol], so it matches sub.
                    new_source.append(line)
                else:
                    new_source.append(line)
            cell['source'] = new_source
