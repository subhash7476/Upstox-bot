"""
scripts/daily_update.py - simple runner that triggers incremental appends for a list of symbols.
Usage: python scripts/daily_update.py
It reads data/symbols.txt (one symbol per line with optional segment,like "RELIANCE,NSE_EQ").
If not present, runs a demo for RELIANCE.
"""
from data_manager import fetch_and_save_data
from datetime import datetime
from pathlib import Path

SYMBOLS_FILE = Path("data/symbols.txt")

def load_symbols():
    if SYMBOLS_FILE.exists():
        lines = [l.strip() for l in SYMBOLS_FILE.read_text().splitlines() if l.strip()]
        out = []
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 1:
                out.append((parts[0],"NSE_EQ"))
            else:
                out.append((parts[0],parts[1]))
        return out
    return [("RELIANCE","NSE_EQ")]

def run_incremental_all(interval="1day"):
    syms = load_symbols()
    appended_summary = {}
    for s,seg in syms:
        try:
            appended = fetch_and_save_data(s, seg, interval, incremental=True)
            appended_summary[s] = appended
            print(f"{s}: appended {len(appended)} partitions")
        except Exception as e:
            print(f"{s}: failed -> {e}")
    return appended_summary

if __name__ == "__main__":
    print("Daily updater running at", datetime.utcnow())
    summary = run_incremental_all(interval="1day")
    print("Summary:", summary)
