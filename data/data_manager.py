# data/data_manager.py
"""
Upstox V3 data manager (Batch Size: 15 Days)
- Fetches 15 days at once for speed.
- Splits data into daily partitions for storage.
- Forces UTC -> IST conversion.
"""

from __future__ import annotations
import time
import math
import requests
import zipfile
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import List, Optional, Callable, Tuple
import pandas as pd

from core.config import get_access_token
from core.api.instruments import load_segment_instruments

# Partition root
PARTITION_ROOT = Path("data/stocks")

# -------------------------
# low-level helpers
# -------------------------
def _max_days_for(timeframe: str, interval_num: int) -> int:
    # UPDATED: Changed from 1 to 15 days for faster downloading
    if timeframe == "minutes":
        return 15 
    if timeframe == "hours":
        return 60
    if timeframe == "days":
        return 365
    return 30

def _request_with_retry(url: str, headers: dict, max_retries: int = 4, backoff_factor: float = 0.5, timeout: int = 30):
    attempt = 0
    while True:
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                raise
            wait = backoff_factor * (2 ** (attempt - 1))
            time.sleep(wait)

# -------------------------
# mapping & path helpers
# -------------------------
def map_interval_to_v3(interval: str) -> Tuple[str, int]:
    s = str(interval).lower().strip()
    if s.endswith("minute"):
        n = int(s.replace("minute", "").strip())
        return "minutes", n
    if s.endswith("hour") or s.endswith("hours"):
        n = int(s.replace("hour", "").replace("hours", "").strip())
        return "hours", n
    if s in ("1day", "day", "daily", "1d"):
        return "days", 1
    raise ValueError(f"Unsupported interval: {interval}")

def ensure_partition_dirs(symbol: str, interval: str) -> Path:
    base = PARTITION_ROOT / symbol / interval
    base.mkdir(parents=True, exist_ok=True)
    return base

def get_partition_path(symbol: str, interval: str, day_date: date) -> Path:
    p = (
        PARTITION_ROOT
        / symbol
        / interval
        / f"year={day_date.year:04d}"
        / f"month={day_date.month:02d}"
        / f"day={day_date.day:02d}"
    )
    p.mkdir(parents=True, exist_ok=True)
    return p / "data.parquet"

def save_partition(path: Path, df: pd.DataFrame, mode: str = "overwrite") -> None:
    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)

# -------------------------
# Single-chunk fetch with retries (TIMEZONE FIXED)
# -------------------------
def fetch_upstox_v3(symbol: str,
                    segment: str,
                    timeframe: str,
                    interval_num: int,
                    from_date: date,
                    to_date: date,
                    max_retries: int = 4) -> pd.DataFrame:
    seg_df = load_segment_instruments(segment)
    if seg_df is None or seg_df.empty:
        raise RuntimeError(f"No instruments for segment {segment}")

    # Robust symbol column detection
    sym_col = next((c for c in seg_df.columns if isinstance(c, str) and ("symbol" in c.lower() or "trad" in c.lower())), None)
    if not sym_col:
        if "symbol" in str(seg_df.columns[0]).lower(): 
             sym_col = seg_df.columns[0]
        else:
             raise RuntimeError("No trading symbol column in instruments")

    # Filter for the symbol
    row = seg_df[seg_df[sym_col].astype(str).str.upper() == symbol.upper()]
    if row.empty:
        row = seg_df[seg_df[sym_col].astype(str).str.contains(symbol, case=False, na=False)]
        if row.empty:
            raise RuntimeError(f"Symbol '{symbol}' not found in segment '{segment}'")

    inst_key = row.iloc[0].get("instrument_key") or row.iloc[0].get("instrumentKey")
    if not inst_key:
        raise RuntimeError("instrument_key not found")

    tok = get_access_token()
    if not tok:
        raise RuntimeError("No access token. Login first.")
    headers = {"Authorization": f"Bearer {tok}", "Accept": "application/json"}

    fd = (from_date if isinstance(from_date, date) else from_date.date()).strftime("%Y-%m-%d")
    td = (to_date if isinstance(to_date, date) else to_date.date()).strftime("%Y-%m-%d")
    
    url = f"https://api.upstox.com/v3/historical-candle/{inst_key}/{timeframe}/{interval_num}/{td}/{fd}"

    r = _request_with_retry(url, headers, max_retries=max_retries)
    j = r.json()

    candles = []
    if isinstance(j, dict):
        if "data" in j and isinstance(j["data"], dict) and "candles" in j["data"]:
            candles = j["data"]["candles"]
        elif "data" in j and isinstance(j["data"], list):
            candles = j["data"]
        elif "candles" in j:
            candles = j["candles"]

    if not candles:
        return pd.DataFrame()

    df = pd.DataFrame(candles)
    
    # Handle integer columns safely
    if df.shape[1] >= 5 and 0 in df.columns:
        cols = ["timestamp", "Open", "High", "Low", "Close", "Volume", "OI"]
        df.columns = cols[: df.shape[1]]
    else:
        mapping = {}
        for c in df.columns:
            if isinstance(c, str):
                lc = c.lower()
                if "time" in lc: mapping[c] = "timestamp"
                elif lc in ("open","o"): mapping[c] = "Open"
                elif lc in ("high","h"): mapping[c] = "High"
                elif lc in ("low","l"): mapping[c] = "Low"
                elif lc in ("close","c"): mapping[c] = "Close"
                elif "vol" in lc or lc == "v": mapping[c] = "Volume"
                elif "oi" in lc: mapping[c] = "OI"
        if mapping:
            df = df.rename(columns=mapping)

    if "timestamp" not in df.columns:
         if 0 in df.columns:
             df.rename(columns={0: "timestamp"}, inplace=True)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        
        # --- TIMEZONE FIX ---
        # Upstox returns UTC. We must convert to IST (Asia/Kolkata).
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
            
        df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Kolkata")
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
        
        df = df.set_index("timestamp").sort_index()
    
    cols = ["Open","High","Low","Close","Volume","OI"]
    keep = [c for c in cols if c in df.columns]
    df = df[keep]
    
    return df

# -------------------------
# Chunked fetch + progress + retries
# -------------------------
def fetch_historical_range(symbol: str,
                           segment: str,
                           interval: str,
                           from_date,
                           to_date,
                           force: bool = False,
                           sleep_s: float = 0.2,
                           progress_callback: Optional[Callable[[str], None]] = None,
                           max_retries_per_chunk: int = 4) -> List[str]:
    
    if isinstance(from_date, str): fd = datetime.fromisoformat(from_date).date()
    elif isinstance(from_date, datetime): fd = from_date.date()
    else: fd = from_date
        
    if isinstance(to_date, str): td = datetime.fromisoformat(to_date).date()
    elif isinstance(to_date, datetime): td = to_date.date()
    else: td = to_date

    timeframe, interval_num = map_interval_to_v3(interval)
    max_days = _max_days_for(timeframe, interval_num)

    saved_paths: List[str] = []
    cur_start = fd
    
    total_days = (td - fd).days + 1
    chunks_total = math.ceil(total_days / max_days) if max_days > 0 else 1
    chunk_count = 0

    while cur_start <= td:
        chunk_count += 1
        cur_end = min(td, cur_start + timedelta(days=max_days - 1))
        
        if progress_callback:
            progress_callback(f"Fetching {symbol} ({chunk_count}/{chunks_total}): {cur_start} to {cur_end}")

        try:
            df_chunk = fetch_upstox_v3(symbol, segment, timeframe, interval_num, cur_start, cur_end, max_retries=max_retries_per_chunk)
            
            if df_chunk is not None and not df_chunk.empty:
                # Iterate through days in the 15-day chunk and save individually
                for day, day_df in df_chunk.groupby(df_chunk.index.date):
                    day_date = pd.to_datetime(day).date()
                    path = get_partition_path(symbol, interval, day_date)
                    
                    if path.exists() and not force:
                        saved_paths.append(str(path))
                    else:
                        save_partition(path, day_df)
                        saved_paths.append(str(path))
            else:
                pass 

        except Exception as e:
            if progress_callback:
                progress_callback(f"Error fetching chunk: {e}")
            raise e

        cur_start = cur_end + timedelta(days=1)
        time.sleep(sleep_s)

    return saved_paths

def fetch_batch(symbols: List[str],
                segment: str,
                interval: str,
                from_date,
                to_date,
                force: bool = False,
                progress_callback: Optional[Callable[[str], None]] = None) -> dict:
    results = {}
    total = len(symbols)
    for i, s in enumerate(symbols, start=1):
        if progress_callback:
            progress_callback(f"[{i}/{total}] Processing {s}...")
        try:
            saved = fetch_historical_range(s, segment, interval, from_date, to_date, force=force, progress_callback=progress_callback)
            results[s] = {"ok": True, "saved_count": len(saved)}
        except Exception as e:
            results[s] = {"ok": False, "error": str(e)}
    return results

def list_partitions(symbol: str, interval: str) -> List[Path]:
    base = PARTITION_ROOT / symbol / interval
    if not base.exists():
        return []
    return sorted(base.rglob("data.parquet"))

# -------------------------
# Utilities
# -------------------------

def detect_missing_days(symbol: str, interval: str, from_date, to_date) -> List[date]:
    if isinstance(from_date, str): fd = datetime.fromisoformat(from_date).date()
    elif isinstance(from_date, datetime): fd = from_date.date()
    else: fd = from_date
        
    if isinstance(to_date, str): td = datetime.fromisoformat(to_date).date()
    elif isinstance(to_date, datetime): td = to_date.date()
    else: td = to_date

    missing = []
    d = fd
    while d <= td:
        path = get_partition_path(symbol, interval, d)
        if not path.exists():
            missing.append(d)
        d = d + timedelta(days=1)
    return missing

def preflight_plan(symbol: str, interval: str):
    parts = list_partitions(symbol, interval)
    if not parts:
        return {"last_saved": None, "next_day_to_fetch": None}
    
    def extract_date(p: Path):
        try:
            day = int(p.parent.name.replace("day=", ""))
            month = int(p.parent.parent.name.replace("month=", ""))
            year = int(p.parent.parent.parent.name.replace("year=", ""))
            return date(year, month, day)
        except Exception:
            return None
            
    dates = [extract_date(p) for p in parts]
    dates = [d for d in dates if d is not None]
    
    if not dates:
        return {"last_saved": None, "next_day_to_fetch": None}
        
    last = max(dates)
    return {"last_saved": last, "next_day_to_fetch": last + timedelta(days=1)}

def merge_partitions_to_parquet(symbol: str, interval: str, out_path: Path) -> Path:
    files = sorted([p for p in (PARTITION_ROOT / symbol / interval).rglob("data.parquet")])
    if not files:
        raise RuntimeError(f"No partitions found for {symbol} {interval} to merge")
    
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            print(f"Warning: Skipping corrupted file {f}: {e}")
            
    if not dfs:
        raise RuntimeError("No valid data found to merge")

    merged = pd.concat(dfs).sort_index()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path)
    return out_path

def merge_partitions_to_zip(symbol: str, interval: str, out_zip_path: Path) -> Path:
    tmp_parquet = out_zip_path.with_suffix(".parquet")
    merge_partitions_to_parquet(symbol, interval, tmp_parquet)
    with zipfile.ZipFile(out_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(tmp_parquet, arcname=tmp_parquet.name)
    tmp_parquet.unlink(missing_ok=True)
    return out_zip_path