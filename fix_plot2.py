with open("src/common/analysis_utils.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    if 'edgecolors=external_colors[is_flagged]' in line:
        new_lines.append('            edgecolors=external_colors[is_flagged] if external_colors is not None else "red",\n')
    else:
        new_lines.append(line)

with open("src/common/analysis_utils.py", "w", encoding="utf-8") as f:
    f.writelines(new_lines)
