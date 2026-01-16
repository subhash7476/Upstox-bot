import ast
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(".")


class CallVisitor(ast.NodeVisitor):
    def __init__(self):
        self.current_function = None
        self.calls = defaultdict(set)

    def visit_FunctionDef(self, node):
        prev = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = prev

    def visit_Call(self, node):
        if self.current_function:
            if isinstance(node.func, ast.Name):
                self.calls[self.current_function].add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                self.calls[self.current_function].add(node.func.attr)
        self.generic_visit(node)


def scan_calls():
    call_map = defaultdict(set)

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

        try:
            source = py_file.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source, filename=str(py_file))
        except (SyntaxError, UnicodeDecodeError):
            # Skip invalid or non-code .py files
            continue

        visitor = CallVisitor()
        visitor.visit(tree)

        for caller, callees in visitor.calls.items():
            for callee in callees:
                call_map[f"{py_file}:{caller}"].add(callee)

    return call_map
