from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXCLUDE_DIRS = {
    "venv",
    "trading-env",
    "site-packages",
    "__pycache__",
    ".git",
    ".idea",
    ".vscode",
    "OLD Pages",
    "Chatgpt",
    "Gemini",
    "GROK root",
}

EXCLUDE_FILES = {
    "File Map.py",
    "diagnostic queries.py",
}
