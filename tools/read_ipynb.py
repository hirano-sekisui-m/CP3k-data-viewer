import json
from pathlib import Path

def read_ipynb(notebook_path):
    notebook_path = Path(notebook_path)

    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    content = []

    for i, cell in enumerate(notebook.get("cells", []), start=1):
        cell_type = cell.get("cell_type", "unknown")
        source = "".join(cell.get("source", []))

        content.append(
            f"\n{'=' * 80}\n"
            f"Cell {i} ({cell_type})\n"
            f"{'=' * 80}\n"
            f"{source}"
        )

    return "\n".join(content)


# Notebookのパスを指定
notebook_path = r"C:\Users\23032827\Desktop\python_pj\CP3k-data-viewer\src\01_correlation_参考_1.ipynb"

text = read_ipynb(notebook_path)

print(text)

# 必要ならテキストファイルに保存
#
# with open("notebook_contents.txt", "w", encoding="utf-8") as f:
#    f.write(text)