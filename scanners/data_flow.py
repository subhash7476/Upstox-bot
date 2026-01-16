import re
from scanners.walker import iter_python_files

DB_PATTERNS = re.compile(
    r"(ohlcv_|ehma_|live_ohlcv|FROM\s+\w+)",
    re.IGNORECASE
)

API_PATTERNS = re.compile(
    r"requests\.(get|post|put|delete)"
)


def build_data_map():
    data_map = {}

    for py in iter_python_files():
        try:
            text = py.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        db_hits = set(DB_PATTERNS.findall(text))
        api_hits = set(API_PATTERNS.findall(text))

        if db_hits or api_hits:
            data_map[str(py)] = {
                "db": sorted(db_hits),
                "api": sorted(api_hits),
            }

    return data_map
