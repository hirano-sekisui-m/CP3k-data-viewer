import nbformat
import sys

def check_notebook(filepath):
    try:
        nb = nbformat.read(filepath, as_version=4)
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == 'code':
                print(f"Checking cell {i}...")
                try:
                    compile(cell.source, f"cell_{i}", 'exec')
                except SyntaxError as e:
                    print(f"Syntax error in cell {i}: {e}")
                    print(f"Source:\n{cell.source}")
                    return False
        print("All code cells are syntactically correct.")
        return True
    except Exception as e:
        print(f"Failed to read notebook: {e}")
        return False

if __name__ == "__main__":
    check_notebook('src/01_correlation_参考_1.ipynb')
