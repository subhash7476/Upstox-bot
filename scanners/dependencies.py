import ast
import os
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(".")  # run from project root


def extract_imports(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename=str(file_path))
    except (SyntaxError, UnicodeDecodeError):
        # Skip non-python or broken files
        return set()

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                imports.add(n.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])
    return imports


def build_dependency_map():
    deps = defaultdict(set)

    from scanners.config import EXCLUDE_DIRS, EXCLUDE_FILES

    for py_file in PROJECT_ROOT.rglob("*.py"):
        if any(p in py_file.parts for p in EXCLUDE_DIRS):
            continue
        if py_file.name in EXCLUDE_FILES:
            continue

        if "venv" in py_file.parts or "__pycache__" in py_file.parts:
            continue

        EXCLUDE_DIRS = {"venv", "trading-env", "site-packages", "__pycache__"}

        if any(p in py_file.parts for p in EXCLUDE_DIRS):
            continue

        module = py_file.relative_to(PROJECT_ROOT).as_posix()
        imports = extract_imports(py_file)
        deps[module].update(imports)

    return deps


if __name__ == "__main__":
    deps = build_dependency_map()

    for mod, imps in deps.items():
        print(mod)
        for i in sorted(imps):
            print(f"  -> {i}")
