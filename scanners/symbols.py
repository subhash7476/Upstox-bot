import ast
from scanners.walker import iter_python_files


class SymbolVisitor(ast.NodeVisitor):
    def __init__(self):
        self.classes = {}
        self.functions = []

    def visit_ClassDef(self, node):
        methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
        self.classes[node.name] = methods
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.functions.append(node.name)


def analyze_file(path):
    try:
        source = path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except SyntaxError:
        return {}, []

    v = SymbolVisitor()
    v.visit(tree)
    return v.classes, v.functions


def build_symbol_map():
    symbols = {}

    for py in iter_python_files():
        classes, functions = analyze_file(py)
        symbols[str(py)] = {
            "classes": classes,
            "functions": functions
        }

    return symbols
