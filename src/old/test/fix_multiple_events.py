import json

with open('src/01_correlation_参考_1.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if any('analyze_btn.on_click(run_analysis_action)' in line for line in source):
            # The issue with multiple click events is because Jupyter registers the event every time the cell is run.
            # To fix this, we can remove the callback first.
            new_source = []
            for line in source:
                if 'refresh_dir_btn.on_click(' in line:
                    new_source.append('try: refresh_dir_btn.on_click(refresh_input_dir_list, remove=True)\nexcept: pass\n')
                elif 'load_btn.on_click(' in line:
                    new_source.append('try: load_btn.on_click(load_action, remove=True)\nexcept: pass\n')
                elif 'analyze_btn.on_click(' in line:
                    new_source.append('try: analyze_btn.on_click(run_analysis_action, remove=True)\nexcept: pass\n')
                elif 'export_btn.on_click(' in line:
                    new_source.append('try: export_btn.on_click(export_action, remove=True)\nexcept: pass\n')
                elif 'tc_show_btn.on_click(' in line:
                    new_source.append('try: tc_show_btn.on_click(plot_time_course_action, remove=True)\nexcept: pass\n')
                new_source.append(line)
            cell['source'] = new_source

with open('src/01_correlation_参考_1.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
