from pathlib import Path
import pandas as pd
from datetime import time

# =========================================================
# CONFIG
# =========================================================
RAW_ROOT = Path("data/stocks")
DERIVED_ROOT = Path("data/derived")

MARKET_START = time(9, 15)
MARKET_END = time(15, 30)

TF_MAP = {
    "5minute": "5T",
    "15minute": "15T",
    "30minute": "30T",
    "60minute": "60T",
}

AGG = {
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last",
    "Volume": "sum",
}

# =========================================================
# CORE FUNCTIONS
# =========================================================
def list_1m_partitions(symbol: str):
    """
    Return all day-level 1-minute parquet paths for a symbol.
    """
    base = RAW_ROOT / symbol / "1minute"
    if not base.exists():
        return []

    parts = []
    for y in base.glob("year=*"):
        for m in y.glob("month=*"):
            for d in m.glob("day=*"):
                p = d / "data.parquet"
                if p.exists():
                    parts.append(p)

    return sorted(parts)


def load_1m_data(symbol: str) -> pd.DataFrame:
    """
    Load, session-filter, and merge all 1-minute data for a symbol.
    """
    parts = list_1m_partitions(symbol)
    if not parts:
        raise RuntimeError(f"No 1-minute partitions found for {symbol}")

    frames = []

    for p in parts:
        df = pd.read_parquet(p)
        if df.empty:
            continue

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # NSE session filter
        df = df.between_time(MARKET_START, MARKET_END)
        if not df.empty:
            frames.append(df)

    if not frames:
        raise RuntimeError(f"All partitions empty after session filter for {symbol}")

    merged = (
        pd.concat(frames)
        .sort_index()
        .loc[~pd.concat(frames).index.duplicated(keep="first")]
    )

    return merged


def resample_from_1m(
    df_1m: pd.DataFrame,
    target_tf: str
) -> pd.DataFrame:
    """
    Resample 1-minute dataframe into target timeframe.
    """
    if target_tf not in TF_MAP:
        raise ValueError(f"Unsupported timeframe: {target_tf}")

    rule = TF_MAP[target_tf]

    df_tf = (
        df_1m
        .resample(rule, label="left", closed="left")
        .agg(AGG)
        .dropna()
    )

    return df_tf


def build_derived_parquet(
    symbol: str,
    target_tf: str,
    overwrite: bool = False
) -> Path:
    """
    End-to-end pipeline:
    - load 1m
    - resample
    - save merged parquet
    """
    df_1m = load_1m_data(symbol)
    df_tf = resample_from_1m(df_1m, target_tf)

    if df_tf.empty:
        raise RuntimeError("Derived dataframe is empty")

    start = df_tf.index.min().strftime("%Y%m%d")
    end = df_tf.index.max().strftime("%Y%m%d")

    out_dir = DERIVED_ROOT / symbol / target_tf
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"{symbol}_{target_tf}_{start}_{end}.parquet"

    if out_file.exists() and not overwrite:
        raise FileExistsError(f"{out_file} already exists")

    df_tf.to_parquet(out_file)
    return out_file
