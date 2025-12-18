import pandas as pd
import numpy as np

def compute_supertrend(df: pd.DataFrame, atr_period: int = 10, mult: float = 3.0) -> pd.DataFrame:
    df = df.copy()
    high = df['High']; low = df['Low']; close = df['Close']
    tr = pd.concat([(high-low), (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/atr_period, adjust=False).mean()
    hl2 = (high + low) / 2
    upper = hl2 + mult * atr
    lower = hl2 - mult * atr
    trend = pd.Series(1, index=df.index)
    st_line = pd.Series(np.nan, index=df.index)
    st_line.iloc[0] = lower.iloc[0]
    for i in range(1, len(df)):
        if close.iat[i] > upper.iat[i-1]:
            trend.iat[i] = 1
        elif close.iat[i] < lower.iat[i-1]:
            trend.iat[i] = -1
        else:
            trend.iat[i] = trend.iat[i-1]
        st_line.iat[i] = lower.iat[i] if trend.iat[i] == 1 else upper.iat[i]
    df['Supertrend'] = st_line
    df['Trend'] = trend
    df['ATR'] = atr
    return df
