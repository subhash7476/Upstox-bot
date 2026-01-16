import os
from pathlib import Path
from scanners.config import PROJECT_ROOT, EXCLUDE_DIRS, EXCLUDE_FILES


def iter_python_files():
    for dirpath, dirnames, filenames in os.walk(PROJECT_ROOT):
        # 🔥 prune dirs BEFORE descending
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]

        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            if fname in EXCLUDE_FILES:
                continue

            yield Path(dirpath) / fname
