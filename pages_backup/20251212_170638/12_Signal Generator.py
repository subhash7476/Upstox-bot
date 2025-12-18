"""
pages/3_Simple_Supertrend_Signals.py

Modular Streamlit backtester
- Core indicators: Supertrend, EWO, Aroon
- HTF filter + ATR expansion
- SL/TP simulation (next-bar-open exit)
- TP optimizer
- Batch (multi-symbol) support
- NEW: Modular filters (toggleable) with conditional params and diagnostic columns
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Simple Supertrend — Modular Filters", layout="wide")
st.title("Simple Supertrend — Modular Filters (toggle modules on/off)")

# -------------------------
# Indicator helpers
# -------------------------

def elliott_oscillator(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['EWO'] = df['Close'].rolling(5).mean() - df['Close'].rolling(35).mean()
    return df


def aroon_oscillator(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    highs = df['High']
    lows = df['Low']

    def pos_last_high(x):
        return (len(x) - 1) - int(np.argmax(x))
    def pos_last_low(x):
        return (len(x) - 1) - int(np.argmin(x))

    aroon_up = 100 * (period - highs.rolling(period).apply(pos_last_high, raw=True)) / period
    aroon_down = 100 * (period - lows.rolling(period).apply(pos_last_low, raw=True)) / period

    df['AroonOsc'] = aroon_up - aroon_down
    return df

# -------------------------
# Supertrend
# -------------------------

# <<LOCAL compute_supertrend removed by migrate_pages.py - use core.indicators.compute_supertrenddf: pd.DataFrame, atr_period: int = 10, mult: float = 3.0) -> pd.DataFrame:
    df = df.copy()
    high = df['High']
    low = df['Low']
    close = df['Close']

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/atr_period, adjust=False).mean()

    hl2 = (high + low) / 2.0
    upper_basic = hl2 + mult * atr
    lower_basic = hl2 - mult * atr
    upper = upper_basic.copy()
    lower = lower_basic.copy()

    for i in range(1, len(df)):
        if upper_basic.iat[i] < upper.iat[i-1] or close.iat[i-1] > upper.iat[i-1]:
            upper.iat[i] = upper_basic.iat[i]
        else:
            upper.iat[i] = upper.iat[i-1]

        if lower_basic.iat[i] > lower.iat[i-1] or close.iat[i-1] < lower.iat[i-1]:
            lower.iat[i] = lower_basic.iat[i]
        else:
            lower.iat[i] = lower.iat[i-1]

    trend = pd.Series(1, index=df.index)
    st_val = pd.Series(np.nan, index=df.index)

    for i in range(1, len(df)):
        if close.iat[i] > upper.iat[i-1]:
            trend.iat[i] = 1
        elif close.iat[i] < lower.iat[i-1]:
            trend.iat[i] = -1
        else:
            trend.iat[i] = trend.iat[i-1]
        st_val.iat[i] = lower.iat[i] if trend.iat[i] == 1 else upper.iat[i]

    df['Supertrend'] = st_val
    df['ST_Trend'] = trend.astype(int)
    df['ATR'] = atr
    return df

# -------------------------
# Signals & Trade Simulation
# -------------------------

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ST_prev'] = df['ST_Trend'].shift(1).fillna(0).astype(int)
    df['Signal'] = 0
    df.loc[(df['ST_prev'] == -1) & (df['ST_Trend'] == 1), 'Signal'] = 1
    df.loc[(df['ST_prev'] == 1) & (df['ST_Trend'] == -1), 'Signal'] = -1
    df['Position'] = df['Signal'].replace(0, np.nan).ffill().fillna(0).astype(int)
    df.drop(columns=['ST_prev'], inplace=True)
    return df


def simulate_trades_sl_tp(df: pd.DataFrame, signal_col: str = 'FinalSignal', sl_pct: float = 0.05, tp_pct: float = 0.12) -> pd.DataFrame:
    trades = []
    open_trade = None
    for i in range(len(df)-1):
        row = df.iloc[i]
        nxt = df.iloc[i+1]
        sig = int(row.get(signal_col, 0))

        if sig in (1, -1) and open_trade is None:
            open_trade = {'side': 'LONG' if sig == 1 else 'SHORT', 'entry_time': nxt.name, 'entry': float(nxt['Open'])}
            if open_trade['side']=='LONG':
                open_trade['sl'] = open_trade['entry'] * (1 - sl_pct)
                open_trade['tp'] = open_trade['entry'] * (1 + tp_pct)
            else:
                open_trade['sl'] = open_trade['entry'] * (1 + sl_pct)
                open_trade['tp'] = open_trade['entry'] * (1 - tp_pct)
            continue

        if open_trade is not None:
            touched_sl = False
            touched_tp = False
            if open_trade['side']=='LONG':
                if row['Low'] <= open_trade['sl']:
                    touched_sl = True
                if row['High'] >= open_trade['tp']:
                    touched_tp = True
            else:
                if row['High'] >= open_trade['sl']:
                    touched_sl = True
                if row['Low'] <= open_trade['tp']:
                    touched_tp = True

            if touched_sl or touched_tp:
                exit_price = float(nxt['Open'])
                reason = 'TP' if touched_tp and not touched_sl else ('SL' if touched_sl and not touched_tp else 'TP+SL')
                pnl = (exit_price - open_trade['entry']) if open_trade['side']=='LONG' else (open_trade['entry'] - exit_price)
                trades.append({'entry_time': open_trade['entry_time'], 'exit_time': nxt.name, 'side': open_trade['side'], 'entry': open_trade['entry'], 'exit': exit_price, 'pnl': pnl, 'reason': reason})
                open_trade = None
                continue

            if sig != 0 and ((open_trade['side']=='LONG' and sig==-1) or (open_trade['side']=='SHORT' and sig==1)):
                exit_price = float(nxt['Open'])
                pnl = (exit_price - open_trade['entry']) if open_trade['side']=='LONG' else (open_trade['entry'] - exit_price)
                trades.append({'entry_time': open_trade['entry_time'], 'exit_time': nxt.name, 'side': open_trade['side'], 'entry': open_trade['entry'], 'exit': exit_price, 'pnl': pnl, 'reason': 'REVERSE'})
                open_trade = None
                continue

    if open_trade is not None:
        final_price = float(df['Close'].iat[-1])
        pnl = (final_price - open_trade['entry']) if open_trade['side']=='LONG' else (open_trade['entry'] - final_price)
        trades.append({'entry_time': open_trade['entry_time'], 'exit_time': df.index[-1], 'side': open_trade['side'], 'entry': open_trade['entry'], 'exit': final_price, 'pnl': pnl, 'reason': 'END'})
    return pd.DataFrame(trades)

# -------------------------
# MODULES (toggleable)
# -------------------------

def module_no_trade_zone(df: pd.DataFrame, atr_mult_threshold: float = 1.0, body_atr_ratio: float = 0.3) -> pd.DataFrame:
    # marks NTZ_blocked True if in quiet market / tiny body
    df = df.copy()
    df['body'] = (df['Close'] - df['Open']).abs()
    df['ATR_MA20'] = df['ATR'].rolling(20).mean()
    df['NTZ_blocked'] = (df['ATR'] < (df['ATR_MA20'] * atr_mult_threshold)) | (df['body'] < (body_atr_ratio * df['ATR']))
    return df


def module_breakout_confirm(df: pd.DataFrame, lookback_bars: int = 3) -> pd.DataFrame:
    df = df.copy()
    df['recent_high'] = df['High'].rolling(lookback_bars).max().shift(1)
    df['recent_low'] = df['Low'].rolling(lookback_bars).min().shift(1)
    df['Breakout_passed'] = False
    df.loc[(df['Signal']==1) & (df['Close'] > df['recent_high']), 'Breakout_passed'] = True
    df.loc[(df['Signal']==-1) & (df['Close'] < df['recent_low']), 'Breakout_passed'] = True
    return df


def module_regime(df: pd.DataFrame, atr_slope_bars: int = 5) -> pd.DataFrame:
    df = df.copy()
    # simple regime: ATR rising over last n bars and ST slope
    df['ATR_slope'] = df['ATR'] - df['ATR'].shift(atr_slope_bars)
    df['ST_slope'] = df['Supertrend'] - df['Supertrend'].shift(atr_slope_bars)
    df['Regime_passed'] = (df['ATR_slope'] > 0) & (df['ST_slope'].abs() > 0)
    return df


def module_candle_patterns(df: pd.DataFrame, doji_thresh: float = 0.1, long_wick_ratio: float = 0.5) -> pd.DataFrame:
    df = df.copy()
    df['body'] = (df['Close'] - df['Open']).abs()
    df['upper_wick'] = df['High'] - df[['Open','Close']].max(axis=1)
    df['lower_wick'] = df[['Open','Close']].min(axis=1) - df['Low']
    df['doji'] = df['body'] < (doji_thresh * df['ATR'])
    df['long_wick'] = ((df['upper_wick'] > long_wick_ratio * df['ATR']) | (df['lower_wick'] > long_wick_ratio * df['ATR']))
    df['Candle_blocked'] = df['doji'] | df['long_wick']
    return df


def module_confidence_score(df: pd.DataFrame, weight_ewo: float = 1.0, weight_aroon: float = 1.0, weight_atr: float = 0.5) -> pd.DataFrame:
    df = df.copy()
    # normalized components
    ewo_s = df['EWO'] / (df['ATR'] + 1e-9)
    aroon_s = df['AroonOsc'] / 100.0
    atr_s = (df['ATR'] - df['ATR'].rolling(20).mean()) / (df['ATR'].rolling(20).std().replace(0,1))
    score = weight_ewo * ewo_s + weight_aroon * aroon_s + weight_atr * atr_s
    df['ConfidenceScore'] = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return df


def module_ml_placeholder(df: pd.DataFrame, threshold: float = 0.6) -> pd.DataFrame:
    # Placeholder: simple logistic on EWO+Aroon for demo; in production train and save model
    df = df.copy()
    X = pd.DataFrame({'ewo': df['EWO'], 'aroon': df['AroonOsc']})
    # naive rule -> probability surrogate
    prob = 1/(1+np.exp(- (0.01*X['ewo'] + 0.01*X['aroon'])))
    df['ML_prob'] = prob
    df['ML_passed'] = df['ML_prob'] > threshold
    return df


def module_patterns(df: pd.DataFrame, lookback: int = 50, double_tol_pct: float = 0.01, hs_tolerance: float = 0.03, max_sep: int = 30) -> pd.DataFrame:
    """Lite pattern detector: Double Top/Bottom and Head & Shoulders (and inverse).
    Marks Pattern_blocked True when a reversal structure is present (before breakout confirmation).
    """
    df = df.copy()
    # pivots
    df['pivot_high'] = (df['High'] > df['High'].shift(1)) & (df['High'] > df['High'].shift(-1))
    df['pivot_low'] = (df['Low'] < df['Low'].shift(1)) & (df['Low'] < df['Low'].shift(-1))
    df['Pattern_blocked'] = False

    highs_idx = df.index[df['pivot_high']].tolist()
    lows_idx = df.index[df['pivot_low']].tolist()

    # helper to index positions
    idx_pos = {t:i for i,t in enumerate(df.index)}

    # Double Top detection
    for i in range(len(highs_idx)-1):
        p1 = highs_idx[i]
        p2 = highs_idx[i+1]
        pos1 = idx_pos[p1]
        pos2 = idx_pos[p2]
        sep = pos2 - pos1
        if sep < 2 or sep > max_sep:
            continue
        price1 = df.at[p1,'High']
        price2 = df.at[p2,'High']
        if abs(price1-price2)/max(price1,price2) <= double_tol_pct:
            # neckline is min low between peaks
            neckline = df['Low'].iloc[pos1:pos2+1].min()
            # mark from p2 onward as blocked until neckline break (close < neckline)
            for j in range(pos2, min(pos2+lookback, len(df))):
                if df['Close'].iat[j] < neckline:
                    break
                df.at[df.index[j],'Pattern_blocked'] = True

    # Double Bottom detection (inverse)
    for i in range(len(lows_idx)-1):
        p1 = lows_idx[i]
        p2 = lows_idx[i+1]
        pos1 = idx_pos[p1]
        pos2 = idx_pos[p2]
        sep = pos2 - pos1
        if sep < 2 or sep > max_sep:
            continue
        price1 = df.at[p1,'Low']
        price2 = df.at[p2,'Low']
        if abs(price1-price2)/max(price1,price2) <= double_tol_pct:
            neckline = df['High'].iloc[pos1:pos2+1].max()
            for j in range(pos2, min(pos2+lookback, len(df))):
                if df['Close'].iat[j] > neckline:
                    break
                df.at[df.index[j],'Pattern_blocked'] = True

    # Head and Shoulders detection (simple heuristic)
    # look for sequence: pivot_high (left shoulder), higher pivot_high (head), lower pivot_high (right shoulder)
    for i in range(len(highs_idx)-2):
        ls = highs_idx[i]
        hd = highs_idx[i+1]
        rs = highs_idx[i+2]
        pos_ls = idx_pos[ls]
        pos_hd = idx_pos[hd]
        pos_rs = idx_pos[rs]
        if not (2 <= pos_hd-pos_ls <= max_sep and 2 <= pos_rs-pos_hd <= max_sep):
            continue
        p_ls = df.at[ls,'High']
        p_hd = df.at[hd,'High']
        p_rs = df.at[rs,'High']
        if p_hd > p_ls and p_hd > p_rs and (abs(p_ls - p_rs)/max(p_ls,p_rs) <= hs_tolerance):
            neckline = df['Low'].iloc[pos_ls:pos_rs+1].max()
            # mark from rs onward until neckline break
            for j in range(pos_rs, min(pos_rs+lookback, len(df))):
                if df['Close'].iat[j] < neckline:
                    break
                df.at[df.index[j],'Pattern_blocked'] = True

    # Inverse H&S (mirror)
    for i in range(len(lows_idx)-2):
        ls = lows_idx[i]
        hd = lows_idx[i+1]
        rs = lows_idx[i+2]
        pos_ls = idx_pos[ls]
        pos_hd = idx_pos[hd]
        pos_rs = idx_pos[rs]
        if not (2 <= pos_hd-pos_ls <= max_sep and 2 <= pos_rs-pos_hd <= max_sep):
            continue
        p_ls = df.at[ls,'Low']
        p_hd = df.at[hd,'Low']
        p_rs = df.at[rs,'Low']
        if p_hd < p_ls and p_hd < p_rs and (abs(p_ls - p_rs)/max(p_ls,p_rs) <= hs_tolerance):
            neckline = df['High'].iloc[pos_ls:pos_rs+1].min()
            for j in range(pos_rs, min(pos_rs+lookback, len(df))):
                if df['Close'].iat[j] > neckline:
                    break
                df.at[df.index[j],'Pattern_blocked'] = True

        # --- Triangle Detection (lite) ---
    highs_lk = df['High'].rolling(max_sep).max()
    lows_lk = df['Low'].rolling(max_sep).min()
    df['triangle_detected'] = (highs_lk - lows_lk) < (df['ATR'] * 3)
    # flag confirmation logic
    if not confirm_only:
        df.loc[df['triangle_detected'], 'Pattern_blocked'] = True
    else:
        df['triangle_confirm'] = (df['Close'] > highs_lk.shift(1)) | (df['Close'] < lows_lk.shift(1))
        df.loc[df['triangle_detected'] & (~df['triangle_confirm']), 'Pattern_blocked'] = True

    # --- Flag/Pennant Detection (lite) ---
    impulse = (df['Close'] - df['Close'].shift(max_sep//2)).abs() > df['ATR'] * 4
    cons_zone = (df['High'].rolling(max_sep//3).max() - df['Low'].rolling(max_sep//3).min()) < df['ATR'] * 2
    df['flag_detected'] = impulse & cons_zone
    if not confirm_only:
        df.loc[df['flag_detected'], 'Pattern_blocked'] = True
    else:
        local_high = df['High'].rolling(max_sep//3).max().shift(1)
        local_low = df['Low'].rolling(max_sep//3).min().shift(1)
        df['flag_confirm'] = (df['Close'] > local_high) | (df['Close'] < local_low)
        df.loc[df['flag_detected'] & (~df['flag_confirm']), 'Pattern_blocked'] = True

    df['Pattern_passed'] = ~df['Pattern_blocked']
    return df

# -------------------------
# Metrics + Utilities
# -------------------------

def compute_perf(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {'total_pnl':0,'trades':0,'win_rate':0,'avg_win':0,'avg_loss':0,'profit_factor':0,'max_drawdown':0}
    total = trades['pnl'].sum()
    wins = trades[trades['pnl']>0]
    losses = trades[trades['pnl']<=0]
    win_rate = len(wins)/len(trades) if len(trades)>0 else 0
    avg_win = wins['pnl'].mean() if not wins.empty else 0
    avg_loss = losses['pnl'].mean() if not losses.empty else 0
    pf = (wins['pnl'].sum()/abs(losses['pnl'].sum())) if losses['pnl'].sum()!=0 else np.inf
    eq = trades['pnl'].cumsum()
    peak = eq.cummax()
    dd = (eq - peak)
    max_dd = dd.min() if not eq.empty else 0
    return {'total_pnl':total,'trades':len(trades),'win_rate':win_rate,'avg_win':avg_win,'avg_loss':avg_loss,'profit_factor':pf,'max_drawdown':max_dd}


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {'Open':'first','High':'max','Low':'min','Close':'last'}
    if 'Volume' in df.columns:
        agg['Volume']='sum'
    out = df.resample(rule).agg(agg).dropna()
    return out

# -------------------------
# Streamlit UI - Modules
# -------------------------

st.sidebar.header('Files & TF - MULTI SYMBOL')
main_files = st.sidebar.file_uploader('Main TF parquet(s) - upload multiple', type=['parquet'], accept_multiple_files=True)
htf_files = st.sidebar.file_uploader('Optional HTF parquet(s) - match order or leave empty', type=['parquet'], accept_multiple_files=True)
htf_choice = st.sidebar.selectbox('Choose HTF (used if HTF file not uploaded)', options=['15T','30T','60T','120T','D'], index=1, format_func=lambda x: {'15T':'15 min','30T':'30 min','60T':'1 hour','120T':'2 hour','D':'1 day'}[x])

st.sidebar.header('Supertrend settings')
atr_period = st.sidebar.slider('ATR Period',5,50,10)
mult = st.sidebar.slider('Multiplier',1.0,6.0,3.0)

st.sidebar.header('SL/TP & Optimizer')
sl_pct = st.sidebar.number_input('Stop Loss % (fixed)', value=5.0, min_value=0.5, max_value=20.0, step=0.5)/100.0
opt_tp_min = st.sidebar.number_input('TP min %', value=10, min_value=5, max_value=30, step=1)
opt_tp_max = st.sidebar.number_input('TP max %', value=14, min_value=5, max_value=60, step=1)

st.sidebar.header('Enable Modules (toggle)')
mod_ntz = st.sidebar.checkbox('No-Trade Zone (NTZ)', value=True)
mod_breakout = st.sidebar.checkbox('Breakout Confirmation', value=True)
mod_regime = st.sidebar.checkbox('Regime Filter (ATR rising + ST slope)', value=True)
mod_candle = st.sidebar.checkbox('Candle Pattern Filter (doji/long wick)', value=True)
mod_conf = st.sidebar.checkbox('Confidence Score', value=False)
mod_ml = st.sidebar.checkbox('ML Filter (placeholder)', value=False)
mod_patterns = st.sidebar.checkbox('Chart Pattern Module (lite)', value=True)

st.sidebar.markdown('---')
# conditional module params
if mod_ntz:
    ntz_atr_mult = st.sidebar.number_input('NTZ ATR mult threshold', value=1.0, step=0.1)
    ntz_body_atr = st.sidebar.number_input('NTZ body/ATR ratio', value=0.3, step=0.05)
if mod_breakout:
    breakout_bars = st.sidebar.number_input('Breakout lookback bars', value=3, min_value=1)
if mod_regime:
    regime_bars = st.sidebar.number_input('Regime ATR slope bars', value=5, min_value=1)
if mod_candle:
    doji_thresh = st.sidebar.number_input('Doji threshold (fraction of ATR)', value=0.1, step=0.01)
    long_wick_ratio = st.sidebar.number_input('Long wick ratio (of ATR)', value=0.5, step=0.05)
if mod_conf:
    conf_w_ewo = st.sidebar.number_input('Conf weight EWO', value=1.0, step=0.1)
    conf_w_aroon = st.sidebar.number_input('Conf weight Aroon', value=1.0, step=0.1)
    conf_w_atr = st.sidebar.number_input('Conf weight ATR', value=0.5, step=0.1)
    conf_thresh = st.sidebar.slider('Confidence threshold', 0.0, 1.0, 0.5, 0.05)
if mod_ml:
    ml_thresh = st.sidebar.slider('ML prob threshold', 0.0, 1.0, 0.6, 0.05)

run = st.sidebar.button('Run Batch Modular Optimizer')

if not main_files:
    st.info('Upload one or more parquet files (one per symbol/timeframe) to begin')
    st.stop()

if run:
    combined_results = []
    per_symbol_details = {}
    htf_files_map = {Path(f.name).stem: f for f in htf_files} if htf_files else {}

    for idx, mf in enumerate(main_files):
        symbol_name = Path(mf.name).stem
        st.write(f'Processing: {symbol_name}')
        try:
            df = pd.read_parquet(mf)
        except Exception as e:
            st.error(f'Failed reading {symbol_name}: {e}')
            continue

        for c in ['Open','High','Low','Close']:
            if c not in df.columns:
                st.error(f'Missing column {c} in {symbol_name}')
                continue

        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # core indicators
        df_st = compute_supertrend(df, atr_period=atr_period, mult=mult)
        df_st = elliott_oscillator(df_st)
        df_st = aroon_oscillator(df_st)

        # HTF
        htf_file = None
        if htf_files:
            for hf in htf_files:
                if Path(hf.name).stem == f"{symbol_name}_HTF" or Path(hf.name).stem == symbol_name:
                    htf_file = hf
                    break
        if htf_file is not None:
            htf_df = pd.read_parquet(htf_file)
            htf_df.index = pd.to_datetime(htf_df.index)
            htf_df = htf_df.sort_index()
        else:
            htf_df = resample_ohlc(df, htf_choice)

        htf_df = compute_supertrend(htf_df, atr_period=atr_period, mult=mult)
        htf_df.index = pd.to_datetime(htf_df.index)
        htf_df['ts'] = htf_df.index

        # prepare main with ts for merge_asof
        df_st = df_st.copy()
        df_st.index = pd.to_datetime(df_st.index)
        df_st['ts'] = df_st.index
        main = df_st.reset_index(drop=True)
        main = main.sort_values('ts')

        htf_map = htf_df[['ts','ST_Trend']].rename(columns={'ST_Trend':'HTF_ST'}).sort_values('ts')
        merged = pd.merge_asof(main, htf_map, on='ts', direction='backward')
        merged.set_index('ts', inplace=True)
        df_st['HTF_ST'] = merged['HTF_ST'].reindex(df_st.index, method='ffill').fillna(0).astype(int)

        # raw signals + default diagnostics
        df_sig = generate_signals(df_st)
        df_sig['ATR_MA20'] = df_sig['ATR'].rolling(20).mean()
        df_sig['ATR_ok'] = df_sig['ATR'] > df_sig['ATR_MA20']
        df_sig['EWO_pos'] = df_sig['EWO'] > 0
        df_sig['Aroon_pos'] = df_sig['AroonOsc'] > 0

        # apply modules (create diagnostic columns)
        if mod_ntz:
            df_sig = module_no_trade_zone(df_sig, atr_mult_threshold=ntz_atr_mult, body_atr_ratio=ntz_body_atr)
        else:
            df_sig['NTZ_blocked'] = False
        if mod_breakout:
            df_sig = module_breakout_confirm(df_sig, lookback_bars=breakout_bars)
        else:
            df_sig['Breakout_passed'] = True
        if mod_regime:
            df_sig = module_regime(df_sig, atr_slope_bars=regime_bars)
        else:
            df_sig['Regime_passed'] = True
        if mod_candle:
            df_sig = module_candle_patterns(df_sig, doji_thresh=doji_thresh, long_wick_ratio=long_wick_ratio)
        else:
            df_sig['Candle_blocked'] = False
        if mod_conf:
            df_sig = module_confidence_score(df_sig, weight_ewo=conf_w_ewo, weight_aroon=conf_w_aroon, weight_atr=conf_w_atr)
            df_sig['Conf_passed'] = df_sig['ConfidenceScore'] >= conf_thresh
        else:
            df_sig['ConfidenceScore'] = 1.0
            df_sig['Conf_passed'] = True
        if mod_ml:
            df_sig = module_ml_placeholder(df_sig, threshold=ml_thresh)
        else:
            df_sig['ML_passed'] = True

        # FinalSignal composition: require all selected filters to pass
        df_sig['FinalSignal'] = 0
        # ensure Pattern_passed exists
        if 'Pattern_passed' not in df_sig.columns:
            df_sig['Pattern_passed'] = True
        buy_mask = (
            (df_sig['Signal']==1) & (df_sig['EWO_pos']) & (df_sig['Aroon_pos']) & (df_sig['HTF_ST']==1) & (df_sig['ATR_ok'])
            & (~df_sig['NTZ_blocked']) & (df_sig['Breakout_passed']) & (df_sig['Regime_passed']) & (~df_sig['Candle_blocked']) & (df_sig['Conf_passed']) & (df_sig['ML_passed']) & (df_sig['Pattern_passed'])
        )
        sell_mask = (
            (df_sig['Signal']==-1) & (~df_sig['EWO_pos']) & (~df_sig['Aroon_pos']) & (df_sig['HTF_ST']==-1) & (df_sig['ATR_ok'])
            & (~df_sig['NTZ_blocked']) & (df_sig['Breakout_passed']) & (df_sig['Regime_passed']) & (~df_sig['Candle_blocked']) & (df_sig['Conf_passed']) & (df_sig['ML_passed']) & (df_sig['Pattern_passed'])
        )
        df_sig.loc[buy_mask,'FinalSignal'] = 1
        df_sig.loc[sell_mask,'FinalSignal'] = -1

        df_sig['Action'] = df_sig['Signal'].map({1:'BUY',-1:'SELL'}).fillna('')
        df_sig['FinalAction'] = df_sig['FinalSignal'].map({1:'BUY',-1:'SELL'}).fillna('')

        # optimize TP loop
        tp_results = []
        for tp in range(int(opt_tp_min), int(opt_tp_max)+1):
            trades = simulate_trades_sl_tp(df_sig, signal_col='FinalSignal', sl_pct=sl_pct, tp_pct=tp/100.0)
            perf = compute_perf(trades)
            perf['tp_pct'] = tp
            tp_results.append(perf)

        tp_df = pd.DataFrame(tp_results).set_index('tp_pct')
        per_symbol_details[symbol_name] = {'df_sig': df_sig, 'tp_df': tp_df}

        # record combined rows for all TP values
        for _, row in tp_df.reset_index().iterrows():
            combined_results.append({'symbol': symbol_name, 'tp_pct': int(row['tp_pct']), 'total_pnl': row['total_pnl'], 'trades': int(row['trades']), 'win_rate': row['win_rate'], 'profit_factor': row['profit_factor']})

    # present combined results
    combined_df = pd.DataFrame(combined_results)
    if combined_df.empty:
        st.info('No results generated')
        st.stop()

    st.subheader('Combined Optimization Results (all symbols & TP values)')
    st.dataframe(combined_df)

    # best TP per symbol
    best_rows = combined_df.loc[combined_df.groupby('symbol')['total_pnl'].idxmax()].copy()
    best_rows = best_rows.sort_values('total_pnl', ascending=False)
    st.subheader('Best TP per symbol (by Total PnL)')
    st.dataframe(best_rows)

    # drilldown per symbol + charts
    for symbol, details in per_symbol_details.items():
        with st.expander(f'{symbol} — details'):
            df_sig = details['df_sig']
            tp_df = details['tp_df']
            st.write('TP table:')
            st.dataframe(tp_df)

            if not tp_df.empty:
                best_tp = int(tp_df['total_pnl'].idxmax())
                st.write(f'Best TP: {best_tp}%')
                best_trades = simulate_trades_sl_tp(df_sig, signal_col='FinalSignal', sl_pct=sl_pct, tp_pct=best_tp/100.0)
                st.write('Trades:')
                st.dataframe(best_trades)

                # price chart with HTF shading and colored markers by reason
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df_sig.index, open=df_sig['Open'], high=df_sig['High'], low=df_sig['Low'], close=df_sig['Close'], name='price'))

                # add HTF shading — contiguous blocks
                htf_series = df_sig['HTF_ST']
                current = None
                start = None
                for t, val in htf_series.items():
                    if current is None:
                        current = val
                        start = t
                    elif val != current:
                        end = t
                        if current == 1:
                            fig.add_vrect(x0=start, x1=end, fillcolor='green', opacity=0.08, layer='below', line_width=0)
                        elif current == -1:
                            fig.add_vrect(x0=start, x1=end, fillcolor='red', opacity=0.06, layer='below', line_width=0)
                        current = val
                        start = t
                if current is not None:
                    end = df_sig.index[-1]
                    if current == 1:
                        fig.add_vrect(x0=start, x1=end, fillcolor='green', opacity=0.08, layer='below', line_width=0)
                    elif current == -1:
                        fig.add_vrect(x0=start, x1=end, fillcolor='red', opacity=0.06, layer='below', line_width=0)

                # --- Pattern boundary overlays (trendlines only) ---
                # Triangle boundaries
                if 'triangle_detected' in df_sig.columns:
                    tri_idx = df_sig.index[df_sig['triangle_detected']]
                    if len(tri_idx) > 5:
                        seg = df_sig.loc[tri_idx]
                        # upper and lower bounds
                        fig.add_trace(go.Scatter(x=seg.index, y=seg['High'], mode='lines', line=dict(color='blue', width=1), name='Triangle Upper'))
                        fig.add_trace(go.Scatter(x=seg.index, y=seg['Low'], mode='lines', line=dict(color='blue', width=1, dash='dot'), name='Triangle Lower'))

                # Flag boundaries
                if 'flag_detected' in df_sig.columns:
                    flag_idx = df_sig.index[df_sig['flag_detected']]
                    if len(flag_idx) > 5:
                        seg2 = df_sig.loc[flag_idx]
                        fig.add_trace(go.Scatter(x=seg2.index, y=seg2['High'], mode='lines', line=dict(color='green', width=1), name='Flag Upper'))
                        fig.add_trace(go.Scatter(x=seg2.index, y=seg2['Low'], mode='lines', line=dict(color='green', width=1, dash='dot'), name='Flag Lower'))

                # add markers for trade exits colored by reason
                color_map = {'TP':'green','SL':'red','TP+SL':'purple','REVERSE':'orange','END':'blue'}
                best_trades = best_trades.sort_values('exit_time') if not best_trades.empty else best_trades
                for _, tr in best_trades.iterrows():
                    xt = tr['exit_time']
                    reason = tr.get('reason','')
                    color = color_map.get(reason,'black')
                    fig.add_trace(go.Scatter(x=[xt], y=[tr['exit']], mode='markers', marker=dict(color=color, size=10), name=f"{reason}"))

                fig.update_layout(height=600, xaxis_rangeslider_visible=False, title=f'{symbol} price with HTF shading and trade exits')
                st.plotly_chart(fig, use_container_width=True)

                # --- Add breakout arrows for confirmed patterns ---
                if 'triangle_confirm' in df_sig.columns:
                    confirms = df_sig.index[df_sig['triangle_confirm'] == True]
                    for c in confirms:
                        fig.add_annotation(x=c, y=df_sig.loc[c,'High'], text='▲ TRI BK', showarrow=True, arrowhead=2, ax=0, ay=-30, font=dict(color='blue'))
                if 'flag_confirm' in df_sig.columns:
                    confirms = df_sig.index[df_sig['flag_confirm'] == True]
                    for c in confirms:
                        fig.add_annotation(x=c, y=df_sig.loc[c,'High'], text='▲ FLAG BK', showarrow=True, arrowhead=2, ax=0, ay=-30, font=dict(color='green'))

                # add H&S / Double Top/Bottom simple neckline (rolling proxy)
                if 'Pattern_blocked' in df_sig.columns and df_sig['Pattern_blocked'].any():
                    # draw a proxy neckline as rolling min/max over 30 bars around first blocked point
                    first_block = df_sig.index[df_sig['Pattern_blocked']].tolist()[0]
                    pos = df_sig.index.get_loc(first_block)
                    left = max(0, pos-15)
                    right = min(len(df_sig)-1, pos+15)
                    neckline_low = df_sig['Low'].iloc[left:right].min()
                    neckline_high = df_sig['High'].iloc[left:right].max()
                    fig.add_hline(y=neckline_low, line=dict(color='purple', width=1, dash='dash'), annotation_text='Neckline Low', annotation_position='bottom right')
                    fig.add_hline(y=neckline_high, line=dict(color='purple', width=1, dash='dash'), annotation_text='Neckline High', annotation_position='top right')

                # re-render chart with annotations
                st.plotly_chart(fig, use_container_width=True)

                # --- Pattern statistics ---
                st.write('Pattern detection stats:')
                num_tri = int(df_sig['triangle_detected'].sum()) if 'triangle_detected' in df_sig.columns else 0
                num_flag = int(df_sig['flag_detected'].sum()) if 'flag_detected' in df_sig.columns else 0
                num_pattern_blocked = int(df_sig['Pattern_blocked'].sum()) if 'Pattern_blocked' in df_sig.columns else 0
                st.write(f'Triangles detected: {num_tri} — Flags detected: {num_flag} — Pattern-blocked candles: {num_pattern_blocked}')

                # trades during patterns (approximate by matching entry_time)
                if not best_trades.empty:
                    tri_entries = best_trades[best_trades['entry_time'].isin(df_sig.index[df_sig.get('triangle_detected', False)])]
                    flag_entries = best_trades[best_trades['entry_time'].isin(df_sig.index[df_sig.get('flag_detected', False)])]
                    def winrate(tr):
                        return (tr['pnl']>0).mean() if len(tr)>0 else float('nan')
                    st.write(f'Trades entered during triangle zones: {len(tri_entries)} | Win rate: {winrate(tri_entries):.2f}')
                    st.write(f'Trades entered during flag zones: {len(flag_entries)} | Win rate: {winrate(flag_entries):.2f}')

    st.success('Batch run complete')
"""
pages/3_Simple_Supertrend_Signals.py

Modular Streamlit backtester
- Core indicators: Supertrend, EWO, Aroon
- HTF filter + ATR expansion
- SL/TP simulation (next-bar-open exit)
- TP optimizer
- Batch (multi-symbol) support
- NEW: Modular filters (toggleable) with conditional params and diagnostic columns
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Simple Supertrend — Modular Filters", layout="wide")
st.title("Simple Supertrend — Modular Filters (toggle modules on/off)")

# -------------------------
# Indicator helpers
# -------------------------

def elliott_oscillator(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['EWO'] = df['Close'].rolling(5).mean() - df['Close'].rolling(35).mean()
    return df


def aroon_oscillator(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    highs = df['High']
    lows = df['Low']

    def pos_last_high(x):
        return (len(x) - 1) - int(np.argmax(x))
    def pos_last_low(x):
        return (len(x) - 1) - int(np.argmin(x))

    aroon_up = 100 * (period - highs.rolling(period).apply(pos_last_high, raw=True)) / period
    aroon_down = 100 * (period - lows.rolling(period).apply(pos_last_low, raw=True)) / period

    df['AroonOsc'] = aroon_up - aroon_down
    return df

# -------------------------
# Supertrend
# -------------------------

# <<LOCAL compute_supertrend removed by migrate_pages.py - use core.indicators.compute_supertrenddf: pd.DataFrame, atr_period: int = 10, mult: float = 3.0) -> pd.DataFrame:
    df = df.copy()
    high = df['High']
    low = df['Low']
    close = df['Close']

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/atr_period, adjust=False).mean()

    hl2 = (high + low) / 2.0
    upper_basic = hl2 + mult * atr
    lower_basic = hl2 - mult * atr
    upper = upper_basic.copy()
    lower = lower_basic.copy()

    for i in range(1, len(df)):
        if upper_basic.iat[i] < upper.iat[i-1] or close.iat[i-1] > upper.iat[i-1]:
            upper.iat[i] = upper_basic.iat[i]
        else:
            upper.iat[i] = upper.iat[i-1]

        if lower_basic.iat[i] > lower.iat[i-1] or close.iat[i-1] < lower.iat[i-1]:
            lower.iat[i] = lower_basic.iat[i]
        else:
            lower.iat[i] = lower.iat[i-1]

    trend = pd.Series(1, index=df.index)
    st_val = pd.Series(np.nan, index=df.index)

    for i in range(1, len(df)):
        if close.iat[i] > upper.iat[i-1]:
            trend.iat[i] = 1
        elif close.iat[i] < lower.iat[i-1]:
            trend.iat[i] = -1
        else:
            trend.iat[i] = trend.iat[i-1]
        st_val.iat[i] = lower.iat[i] if trend.iat[i] == 1 else upper.iat[i]

    df['Supertrend'] = st_val
    df['ST_Trend'] = trend.astype(int)
    df['ATR'] = atr
    return df

# -------------------------
# Signals & Trade Simulation
# -------------------------

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ST_prev'] = df['ST_Trend'].shift(1).fillna(0).astype(int)
    df['Signal'] = 0
    df.loc[(df['ST_prev'] == -1) & (df['ST_Trend'] == 1), 'Signal'] = 1
    df.loc[(df['ST_prev'] == 1) & (df['ST_Trend'] == -1), 'Signal'] = -1
    df['Position'] = df['Signal'].replace(0, np.nan).ffill().fillna(0).astype(int)
    df.drop(columns=['ST_prev'], inplace=True)
    return df


def simulate_trades_sl_tp(df: pd.DataFrame, signal_col: str = 'FinalSignal', sl_pct: float = 0.05, tp_pct: float = 0.12) -> pd.DataFrame:
    trades = []
    open_trade = None
    for i in range(len(df)-1):
        row = df.iloc[i]
        nxt = df.iloc[i+1]
        sig = int(row.get(signal_col, 0))

        if sig in (1, -1) and open_trade is None:
            open_trade = {'side': 'LONG' if sig == 1 else 'SHORT', 'entry_time': nxt.name, 'entry': float(nxt['Open'])}
            if open_trade['side']=='LONG':
                open_trade['sl'] = open_trade['entry'] * (1 - sl_pct)
                open_trade['tp'] = open_trade['entry'] * (1 + tp_pct)
            else:
                open_trade['sl'] = open_trade['entry'] * (1 + sl_pct)
                open_trade['tp'] = open_trade['entry'] * (1 - tp_pct)
            continue

        if open_trade is not None:
            touched_sl = False
            touched_tp = False
            if open_trade['side']=='LONG':
                if row['Low'] <= open_trade['sl']:
                    touched_sl = True
                if row['High'] >= open_trade['tp']:
                    touched_tp = True
            else:
                if row['High'] >= open_trade['sl']:
                    touched_sl = True
                if row['Low'] <= open_trade['tp']:
                    touched_tp = True

            if touched_sl or touched_tp:
                exit_price = float(nxt['Open'])
                reason = 'TP' if touched_tp and not touched_sl else ('SL' if touched_sl and not touched_tp else 'TP+SL')
                pnl = (exit_price - open_trade['entry']) if open_trade['side']=='LONG' else (open_trade['entry'] - exit_price)
                trades.append({'entry_time': open_trade['entry_time'], 'exit_time': nxt.name, 'side': open_trade['side'], 'entry': open_trade['entry'], 'exit': exit_price, 'pnl': pnl, 'reason': reason})
                open_trade = None
                continue

            if sig != 0 and ((open_trade['side']=='LONG' and sig==-1) or (open_trade['side']=='SHORT' and sig==1)):
                exit_price = float(nxt['Open'])
                pnl = (exit_price - open_trade['entry']) if open_trade['side']=='LONG' else (open_trade['entry'] - exit_price)
                trades.append({'entry_time': open_trade['entry_time'], 'exit_time': nxt.name, 'side': open_trade['side'], 'entry': open_trade['entry'], 'exit': exit_price, 'pnl': pnl, 'reason': 'REVERSE'})
                open_trade = None
                continue

    if open_trade is not None:
        final_price = float(df['Close'].iat[-1])
        pnl = (final_price - open_trade['entry']) if open_trade['side']=='LONG' else (open_trade['entry'] - final_price)
        trades.append({'entry_time': open_trade['entry_time'], 'exit_time': df.index[-1], 'side': open_trade['side'], 'entry': open_trade['entry'], 'exit': final_price, 'pnl': pnl, 'reason': 'END'})
    return pd.DataFrame(trades)

# -------------------------
# MODULES (toggleable)
# -------------------------

def module_no_trade_zone(df: pd.DataFrame, atr_mult_threshold: float = 1.0, body_atr_ratio: float = 0.3) -> pd.DataFrame:
    # marks NTZ_blocked True if in quiet market / tiny body
    df = df.copy()
    df['body'] = (df['Close'] - df['Open']).abs()
    df['ATR_MA20'] = df['ATR'].rolling(20).mean()
    df['NTZ_blocked'] = (df['ATR'] < (df['ATR_MA20'] * atr_mult_threshold)) | (df['body'] < (body_atr_ratio * df['ATR']))
    return df


def module_breakout_confirm(df: pd.DataFrame, lookback_bars: int = 3) -> pd.DataFrame:
    df = df.copy()
    df['recent_high'] = df['High'].rolling(lookback_bars).max().shift(1)
    df['recent_low'] = df['Low'].rolling(lookback_bars).min().shift(1)
    df['Breakout_passed'] = False
    df.loc[(df['Signal']==1) & (df['Close'] > df['recent_high']), 'Breakout_passed'] = True
    df.loc[(df['Signal']==-1) & (df['Close'] < df['recent_low']), 'Breakout_passed'] = True
    return df


def module_regime(df: pd.DataFrame, atr_slope_bars: int = 5) -> pd.DataFrame:
    df = df.copy()
    # simple regime: ATR rising over last n bars and ST slope
    df['ATR_slope'] = df['ATR'] - df['ATR'].shift(atr_slope_bars)
    df['ST_slope'] = df['Supertrend'] - df['Supertrend'].shift(atr_slope_bars)
    df['Regime_passed'] = (df['ATR_slope'] > 0) & (df['ST_slope'].abs() > 0)
    return df


def module_candle_patterns(df: pd.DataFrame, doji_thresh: float = 0.1, long_wick_ratio: float = 0.5) -> pd.DataFrame:
    df = df.copy()
    df['body'] = (df['Close'] - df['Open']).abs()
    df['upper_wick'] = df['High'] - df[['Open','Close']].max(axis=1)
    df['lower_wick'] = df[['Open','Close']].min(axis=1) - df['Low']
    df['doji'] = df['body'] < (doji_thresh * df['ATR'])
    df['long_wick'] = ((df['upper_wick'] > long_wick_ratio * df['ATR']) | (df['lower_wick'] > long_wick_ratio * df['ATR']))
    df['Candle_blocked'] = df['doji'] | df['long_wick']
    return df


def module_confidence_score(df: pd.DataFrame, weight_ewo: float = 1.0, weight_aroon: float = 1.0, weight_atr: float = 0.5) -> pd.DataFrame:
    df = df.copy()
    # normalized components
    ewo_s = df['EWO'] / (df['ATR'] + 1e-9)
    aroon_s = df['AroonOsc'] / 100.0
    atr_s = (df['ATR'] - df['ATR'].rolling(20).mean()) / (df['ATR'].rolling(20).std().replace(0,1))
    score = weight_ewo * ewo_s + weight_aroon * aroon_s + weight_atr * atr_s
    df['ConfidenceScore'] = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return df


def module_ml_placeholder(df: pd.DataFrame, threshold: float = 0.6) -> pd.DataFrame:
    # Placeholder: simple logistic on EWO+Aroon for demo; in production train and save model
    df = df.copy()
    X = pd.DataFrame({'ewo': df['EWO'], 'aroon': df['AroonOsc']})
    # naive rule -> probability surrogate
    prob = 1/(1+np.exp(- (0.01*X['ewo'] + 0.01*X['aroon'])))
    df['ML_prob'] = prob
    df['ML_passed'] = df['ML_prob'] > threshold
    return df


def module_patterns(df: pd.DataFrame, lookback: int = 50, double_tol_pct: float = 0.01, hs_tolerance: float = 0.03, max_sep: int = 30) -> pd.DataFrame:
    """Lite pattern detector: Double Top/Bottom and Head & Shoulders (and inverse).
    Marks Pattern_blocked True when a reversal structure is present (before breakout confirmation).
    """
    df = df.copy()
    # pivots
    df['pivot_high'] = (df['High'] > df['High'].shift(1)) & (df['High'] > df['High'].shift(-1))
    df['pivot_low'] = (df['Low'] < df['Low'].shift(1)) & (df['Low'] < df['Low'].shift(-1))
    df['Pattern_blocked'] = False

    highs_idx = df.index[df['pivot_high']].tolist()
    lows_idx = df.index[df['pivot_low']].tolist()

    # helper to index positions
    idx_pos = {t:i for i,t in enumerate(df.index)}

    # Double Top detection
    for i in range(len(highs_idx)-1):
        p1 = highs_idx[i]
        p2 = highs_idx[i+1]
        pos1 = idx_pos[p1]
        pos2 = idx_pos[p2]
        sep = pos2 - pos1
        if sep < 2 or sep > max_sep:
            continue
        price1 = df.at[p1,'High']
        price2 = df.at[p2,'High']
        if abs(price1-price2)/max(price1,price2) <= double_tol_pct:
            # neckline is min low between peaks
            neckline = df['Low'].iloc[pos1:pos2+1].min()
            # mark from p2 onward as blocked until neckline break (close < neckline)
            for j in range(pos2, min(pos2+lookback, len(df))):
                if df['Close'].iat[j] < neckline:
                    break
                df.at[df.index[j],'Pattern_blocked'] = True

    # Double Bottom detection (inverse)
    for i in range(len(lows_idx)-1):
        p1 = lows_idx[i]
        p2 = lows_idx[i+1]
        pos1 = idx_pos[p1]
        pos2 = idx_pos[p2]
        sep = pos2 - pos1
        if sep < 2 or sep > max_sep:
            continue
        price1 = df.at[p1,'Low']
        price2 = df.at[p2,'Low']
        if abs(price1-price2)/max(price1,price2) <= double_tol_pct:
            neckline = df['High'].iloc[pos1:pos2+1].max()
            for j in range(pos2, min(pos2+lookback, len(df))):
                if df['Close'].iat[j] > neckline:
                    break
                df.at[df.index[j],'Pattern_blocked'] = True

    # Head and Shoulders detection (simple heuristic)
    # look for sequence: pivot_high (left shoulder), higher pivot_high (head), lower pivot_high (right shoulder)
    for i in range(len(highs_idx)-2):
        ls = highs_idx[i]
        hd = highs_idx[i+1]
        rs = highs_idx[i+2]
        pos_ls = idx_pos[ls]
        pos_hd = idx_pos[hd]
        pos_rs = idx_pos[rs]
        if not (2 <= pos_hd-pos_ls <= max_sep and 2 <= pos_rs-pos_hd <= max_sep):
            continue
        p_ls = df.at[ls,'High']
        p_hd = df.at[hd,'High']
        p_rs = df.at[rs,'High']
        if p_hd > p_ls and p_hd > p_rs and (abs(p_ls - p_rs)/max(p_ls,p_rs) <= hs_tolerance):
            neckline = df['Low'].iloc[pos_ls:pos_rs+1].max()
            # mark from rs onward until neckline break
            for j in range(pos_rs, min(pos_rs+lookback, len(df))):
                if df['Close'].iat[j] < neckline:
                    break
                df.at[df.index[j],'Pattern_blocked'] = True

    # Inverse H&S (mirror)
    for i in range(len(lows_idx)-2):
        ls = lows_idx[i]
        hd = lows_idx[i+1]
        rs = lows_idx[i+2]
        pos_ls = idx_pos[ls]
        pos_hd = idx_pos[hd]
        pos_rs = idx_pos[rs]
        if not (2 <= pos_hd-pos_ls <= max_sep and 2 <= pos_rs-pos_hd <= max_sep):
            continue
        p_ls = df.at[ls,'Low']
        p_hd = df.at[hd,'Low']
        p_rs = df.at[rs,'Low']
        if p_hd < p_ls and p_hd < p_rs and (abs(p_ls - p_rs)/max(p_ls,p_rs) <= hs_tolerance):
            neckline = df['High'].iloc[pos_ls:pos_rs+1].min()
            for j in range(pos_rs, min(pos_rs+lookback, len(df))):
                if df['Close'].iat[j] > neckline:
                    break
                df.at[df.index[j],'Pattern_blocked'] = True

        # --- Triangle Detection (lite) ---
    highs_lk = df['High'].rolling(max_sep).max()
    lows_lk = df['Low'].rolling(max_sep).min()
    df['triangle_detected'] = (highs_lk - lows_lk) < (df['ATR'] * 3)
    # flag confirmation logic
    if not confirm_only:
        df.loc[df['triangle_detected'], 'Pattern_blocked'] = True
    else:
        df['triangle_confirm'] = (df['Close'] > highs_lk.shift(1)) | (df['Close'] < lows_lk.shift(1))
        df.loc[df['triangle_detected'] & (~df['triangle_confirm']), 'Pattern_blocked'] = True

    # --- Flag/Pennant Detection (lite) ---
    impulse = (df['Close'] - df['Close'].shift(max_sep//2)).abs() > df['ATR'] * 4
    cons_zone = (df['High'].rolling(max_sep//3).max() - df['Low'].rolling(max_sep//3).min()) < df['ATR'] * 2
    df['flag_detected'] = impulse & cons_zone
    if not confirm_only:
        df.loc[df['flag_detected'], 'Pattern_blocked'] = True
    else:
        local_high = df['High'].rolling(max_sep//3).max().shift(1)
        local_low = df['Low'].rolling(max_sep//3).min().shift(1)
        df['flag_confirm'] = (df['Close'] > local_high) | (df['Close'] < local_low)
        df.loc[df['flag_detected'] & (~df['flag_confirm']), 'Pattern_blocked'] = True

    df['Pattern_passed'] = ~df['Pattern_blocked']
    return df

# -------------------------
# Metrics + Utilities
# -------------------------

def compute_perf(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {'total_pnl':0,'trades':0,'win_rate':0,'avg_win':0,'avg_loss':0,'profit_factor':0,'max_drawdown':0}
    total = trades['pnl'].sum()
    wins = trades[trades['pnl']>0]
    losses = trades[trades['pnl']<=0]
    win_rate = len(wins)/len(trades) if len(trades)>0 else 0
    avg_win = wins['pnl'].mean() if not wins.empty else 0
    avg_loss = losses['pnl'].mean() if not losses.empty else 0
    pf = (wins['pnl'].sum()/abs(losses['pnl'].sum())) if losses['pnl'].sum()!=0 else np.inf
    eq = trades['pnl'].cumsum()
    peak = eq.cummax()
    dd = (eq - peak)
    max_dd = dd.min() if not eq.empty else 0
    return {'total_pnl':total,'trades':len(trades),'win_rate':win_rate,'avg_win':avg_win,'avg_loss':avg_loss,'profit_factor':pf,'max_drawdown':max_dd}


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {'Open':'first','High':'max','Low':'min','Close':'last'}
    if 'Volume' in df.columns:
        agg['Volume']='sum'
    out = df.resample(rule).agg(agg).dropna()
    return out

# -------------------------
# Streamlit UI - Modules
# -------------------------

st.sidebar.header('Files & TF - MULTI SYMBOL')
main_files = st.sidebar.file_uploader('Main TF parquet(s) - upload multiple', type=['parquet'], accept_multiple_files=True)
htf_files = st.sidebar.file_uploader('Optional HTF parquet(s) - match order or leave empty', type=['parquet'], accept_multiple_files=True)
htf_choice = st.sidebar.selectbox('Choose HTF (used if HTF file not uploaded)', options=['15T','30T','60T','120T','D'], index=1, format_func=lambda x: {'15T':'15 min','30T':'30 min','60T':'1 hour','120T':'2 hour','D':'1 day'}[x])

st.sidebar.header('Supertrend settings')
atr_period = st.sidebar.slider('ATR Period',5,50,10)
mult = st.sidebar.slider('Multiplier',1.0,6.0,3.0)

st.sidebar.header('SL/TP & Optimizer')
sl_pct = st.sidebar.number_input('Stop Loss % (fixed)', value=5.0, min_value=0.5, max_value=20.0, step=0.5)/100.0
opt_tp_min = st.sidebar.number_input('TP min %', value=10, min_value=5, max_value=30, step=1)
opt_tp_max = st.sidebar.number_input('TP max %', value=14, min_value=5, max_value=60, step=1)

st.sidebar.header('Enable Modules (toggle)')
mod_ntz = st.sidebar.checkbox('No-Trade Zone (NTZ)', value=True)
mod_breakout = st.sidebar.checkbox('Breakout Confirmation', value=True)
mod_regime = st.sidebar.checkbox('Regime Filter (ATR rising + ST slope)', value=True)
mod_candle = st.sidebar.checkbox('Candle Pattern Filter (doji/long wick)', value=True)
mod_conf = st.sidebar.checkbox('Confidence Score', value=False)
mod_ml = st.sidebar.checkbox('ML Filter (placeholder)', value=False)
mod_patterns = st.sidebar.checkbox('Chart Pattern Module (lite)', value=True)

st.sidebar.markdown('---')
# conditional module params
if mod_ntz:
    ntz_atr_mult = st.sidebar.number_input('NTZ ATR mult threshold', value=1.0, step=0.1)
    ntz_body_atr = st.sidebar.number_input('NTZ body/ATR ratio', value=0.3, step=0.05)
if mod_breakout:
    breakout_bars = st.sidebar.number_input('Breakout lookback bars', value=3, min_value=1)
if mod_regime:
    regime_bars = st.sidebar.number_input('Regime ATR slope bars', value=5, min_value=1)
if mod_candle:
    doji_thresh = st.sidebar.number_input('Doji threshold (fraction of ATR)', value=0.1, step=0.01)
    long_wick_ratio = st.sidebar.number_input('Long wick ratio (of ATR)', value=0.5, step=0.05)
if mod_conf:
    conf_w_ewo = st.sidebar.number_input('Conf weight EWO', value=1.0, step=0.1)
    conf_w_aroon = st.sidebar.number_input('Conf weight Aroon', value=1.0, step=0.1)
    conf_w_atr = st.sidebar.number_input('Conf weight ATR', value=0.5, step=0.1)
    conf_thresh = st.sidebar.slider('Confidence threshold', 0.0, 1.0, 0.5, 0.05)
if mod_ml:
    ml_thresh = st.sidebar.slider('ML prob threshold', 0.0, 1.0, 0.6, 0.05)

run = st.sidebar.button('Run Batch Modular Optimizer')

if not main_files:
    st.info('Upload one or more parquet files (one per symbol/timeframe) to begin')
    st.stop()

if run:
    combined_results = []
    per_symbol_details = {}
    htf_files_map = {Path(f.name).stem: f for f in htf_files} if htf_files else {}

    for idx, mf in enumerate(main_files):
        symbol_name = Path(mf.name).stem
        st.write(f'Processing: {symbol_name}')
        try:
            df = pd.read_parquet(mf)
        except Exception as e:
            st.error(f'Failed reading {symbol_name}: {e}')
            continue

        for c in ['Open','High','Low','Close']:
            if c not in df.columns:
                st.error(f'Missing column {c} in {symbol_name}')
                continue

        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # core indicators
        df_st = compute_supertrend(df, atr_period=atr_period, mult=mult)
        df_st = elliott_oscillator(df_st)
        df_st = aroon_oscillator(df_st)

        # HTF
        htf_file = None
        if htf_files:
            for hf in htf_files:
                if Path(hf.name).stem == f"{symbol_name}_HTF" or Path(hf.name).stem == symbol_name:
                    htf_file = hf
                    break
        if htf_file is not None:
            htf_df = pd.read_parquet(htf_file)
            htf_df.index = pd.to_datetime(htf_df.index)
            htf_df = htf_df.sort_index()
        else:
            htf_df = resample_ohlc(df, htf_choice)

        htf_df = compute_supertrend(htf_df, atr_period=atr_period, mult=mult)
        htf_df.index = pd.to_datetime(htf_df.index)
        htf_df['ts'] = htf_df.index

        # prepare main with ts for merge_asof
        df_st = df_st.copy()
        df_st.index = pd.to_datetime(df_st.index)
        df_st['ts'] = df_st.index
        main = df_st.reset_index(drop=True)
        main = main.sort_values('ts')

        htf_map = htf_df[['ts','ST_Trend']].rename(columns={'ST_Trend':'HTF_ST'}).sort_values('ts')
        merged = pd.merge_asof(main, htf_map, on='ts', direction='backward')
        merged.set_index('ts', inplace=True)
        df_st['HTF_ST'] = merged['HTF_ST'].reindex(df_st.index, method='ffill').fillna(0).astype(int)

        # raw signals + default diagnostics
        df_sig = generate_signals(df_st)
        df_sig['ATR_MA20'] = df_sig['ATR'].rolling(20).mean()
        df_sig['ATR_ok'] = df_sig['ATR'] > df_sig['ATR_MA20']
        df_sig['EWO_pos'] = df_sig['EWO'] > 0
        df_sig['Aroon_pos'] = df_sig['AroonOsc'] > 0

        # apply modules (create diagnostic columns)
        if mod_ntz:
            df_sig = module_no_trade_zone(df_sig, atr_mult_threshold=ntz_atr_mult, body_atr_ratio=ntz_body_atr)
        else:
            df_sig['NTZ_blocked'] = False
        if mod_breakout:
            df_sig = module_breakout_confirm(df_sig, lookback_bars=breakout_bars)
        else:
            df_sig['Breakout_passed'] = True
        if mod_regime:
            df_sig = module_regime(df_sig, atr_slope_bars=regime_bars)
        else:
            df_sig['Regime_passed'] = True
        if mod_candle:
            df_sig = module_candle_patterns(df_sig, doji_thresh=doji_thresh, long_wick_ratio=long_wick_ratio)
        else:
            df_sig['Candle_blocked'] = False
        if mod_conf:
            df_sig = module_confidence_score(df_sig, weight_ewo=conf_w_ewo, weight_aroon=conf_w_aroon, weight_atr=conf_w_atr)
            df_sig['Conf_passed'] = df_sig['ConfidenceScore'] >= conf_thresh
        else:
            df_sig['ConfidenceScore'] = 1.0
            df_sig['Conf_passed'] = True
        if mod_ml:
            df_sig = module_ml_placeholder(df_sig, threshold=ml_thresh)
        else:
            df_sig['ML_passed'] = True

        # FinalSignal composition: require all selected filters to pass
        df_sig['FinalSignal'] = 0
        # ensure Pattern_passed exists
        if 'Pattern_passed' not in df_sig.columns:
            df_sig['Pattern_passed'] = True
        buy_mask = (
            (df_sig['Signal']==1) & (df_sig['EWO_pos']) & (df_sig['Aroon_pos']) & (df_sig['HTF_ST']==1) & (df_sig['ATR_ok'])
            & (~df_sig['NTZ_blocked']) & (df_sig['Breakout_passed']) & (df_sig['Regime_passed']) & (~df_sig['Candle_blocked']) & (df_sig['Conf_passed']) & (df_sig['ML_passed']) & (df_sig['Pattern_passed'])
        )
        sell_mask = (
            (df_sig['Signal']==-1) & (~df_sig['EWO_pos']) & (~df_sig['Aroon_pos']) & (df_sig['HTF_ST']==-1) & (df_sig['ATR_ok'])
            & (~df_sig['NTZ_blocked']) & (df_sig['Breakout_passed']) & (df_sig['Regime_passed']) & (~df_sig['Candle_blocked']) & (df_sig['Conf_passed']) & (df_sig['ML_passed']) & (df_sig['Pattern_passed'])
        )
        df_sig.loc[buy_mask,'FinalSignal'] = 1
        df_sig.loc[sell_mask,'FinalSignal'] = -1

        df_sig['Action'] = df_sig['Signal'].map({1:'BUY',-1:'SELL'}).fillna('')
        df_sig['FinalAction'] = df_sig['FinalSignal'].map({1:'BUY',-1:'SELL'}).fillna('')

        # optimize TP loop
        tp_results = []
        for tp in range(int(opt_tp_min), int(opt_tp_max)+1):
            trades = simulate_trades_sl_tp(df_sig, signal_col='FinalSignal', sl_pct=sl_pct, tp_pct=tp/100.0)
            perf = compute_perf(trades)
            perf['tp_pct'] = tp
            tp_results.append(perf)

        tp_df = pd.DataFrame(tp_results).set_index('tp_pct')
        per_symbol_details[symbol_name] = {'df_sig': df_sig, 'tp_df': tp_df}

        # record combined rows for all TP values
        for _, row in tp_df.reset_index().iterrows():
            combined_results.append({'symbol': symbol_name, 'tp_pct': int(row['tp_pct']), 'total_pnl': row['total_pnl'], 'trades': int(row['trades']), 'win_rate': row['win_rate'], 'profit_factor': row['profit_factor']})

    # present combined results
    combined_df = pd.DataFrame(combined_results)
    if combined_df.empty:
        st.info('No results generated')
        st.stop()

    st.subheader('Combined Optimization Results (all symbols & TP values)')
    st.dataframe(combined_df)

    # best TP per symbol
    best_rows = combined_df.loc[combined_df.groupby('symbol')['total_pnl'].idxmax()].copy()
    best_rows = best_rows.sort_values('total_pnl', ascending=False)
    st.subheader('Best TP per symbol (by Total PnL)')
    st.dataframe(best_rows)

    # drilldown per symbol + charts
    for symbol, details in per_symbol_details.items():
        with st.expander(f'{symbol} — details'):
            df_sig = details['df_sig']
            tp_df = details['tp_df']
            st.write('TP table:')
            st.dataframe(tp_df)

            if not tp_df.empty:
                best_tp = int(tp_df['total_pnl'].idxmax())
                st.write(f'Best TP: {best_tp}%')
                best_trades = simulate_trades_sl_tp(df_sig, signal_col='FinalSignal', sl_pct=sl_pct, tp_pct=best_tp/100.0)
                st.write('Trades:')
                st.dataframe(best_trades)

                # price chart with HTF shading and colored markers by reason
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df_sig.index, open=df_sig['Open'], high=df_sig['High'], low=df_sig['Low'], close=df_sig['Close'], name='price'))

                # add HTF shading — contiguous blocks
                htf_series = df_sig['HTF_ST']
                current = None
                start = None
                for t, val in htf_series.items():
                    if current is None:
                        current = val
                        start = t
                    elif val != current:
                        end = t
                        if current == 1:
                            fig.add_vrect(x0=start, x1=end, fillcolor='green', opacity=0.08, layer='below', line_width=0)
                        elif current == -1:
                            fig.add_vrect(x0=start, x1=end, fillcolor='red', opacity=0.06, layer='below', line_width=0)
                        current = val
                        start = t
                if current is not None:
                    end = df_sig.index[-1]
                    if current == 1:
                        fig.add_vrect(x0=start, x1=end, fillcolor='green', opacity=0.08, layer='below', line_width=0)
                    elif current == -1:
                        fig.add_vrect(x0=start, x1=end, fillcolor='red', opacity=0.06, layer='below', line_width=0)

                # --- Pattern boundary overlays (trendlines only) ---
                # Triangle boundaries
                if 'triangle_detected' in df_sig.columns:
                    tri_idx = df_sig.index[df_sig['triangle_detected']]
                    if len(tri_idx) > 5:
                        seg = df_sig.loc[tri_idx]
                        # upper and lower bounds
                        fig.add_trace(go.Scatter(x=seg.index, y=seg['High'], mode='lines', line=dict(color='blue', width=1), name='Triangle Upper'))
                        fig.add_trace(go.Scatter(x=seg.index, y=seg['Low'], mode='lines', line=dict(color='blue', width=1, dash='dot'), name='Triangle Lower'))

                # Flag boundaries
                if 'flag_detected' in df_sig.columns:
                    flag_idx = df_sig.index[df_sig['flag_detected']]
                    if len(flag_idx) > 5:
                        seg2 = df_sig.loc[flag_idx]
                        fig.add_trace(go.Scatter(x=seg2.index, y=seg2['High'], mode='lines', line=dict(color='green', width=1), name='Flag Upper'))
                        fig.add_trace(go.Scatter(x=seg2.index, y=seg2['Low'], mode='lines', line=dict(color='green', width=1, dash='dot'), name='Flag Lower'))

                # add markers for trade exits colored by reason
                color_map = {'TP':'green','SL':'red','TP+SL':'purple','REVERSE':'orange','END':'blue'}
                best_trades = best_trades.sort_values('exit_time') if not best_trades.empty else best_trades
                for _, tr in best_trades.iterrows():
                    xt = tr['exit_time']
                    reason = tr.get('reason','')
                    color = color_map.get(reason,'black')
                    fig.add_trace(go.Scatter(x=[xt], y=[tr['exit']], mode='markers', marker=dict(color=color, size=10), name=f"{reason}"))

                fig.update_layout(height=600, xaxis_rangeslider_visible=False, title=f'{symbol} price with HTF shading and trade exits')
                st.plotly_chart(fig, use_container_width=True)

                # --- Add breakout arrows for confirmed patterns ---
                if 'triangle_confirm' in df_sig.columns:
                    confirms = df_sig.index[df_sig['triangle_confirm'] == True]
                    for c in confirms:
                        fig.add_annotation(x=c, y=df_sig.loc[c,'High'], text='▲ TRI BK', showarrow=True, arrowhead=2, ax=0, ay=-30, font=dict(color='blue'))
                if 'flag_confirm' in df_sig.columns:
                    confirms = df_sig.index[df_sig['flag_confirm'] == True]
                    for c in confirms:
                        fig.add_annotation(x=c, y=df_sig.loc[c,'High'], text='▲ FLAG BK', showarrow=True, arrowhead=2, ax=0, ay=-30, font=dict(color='green'))

                # add H&S / Double Top/Bottom simple neckline (rolling proxy)
                if 'Pattern_blocked' in df_sig.columns and df_sig['Pattern_blocked'].any():
                    # draw a proxy neckline as rolling min/max over 30 bars around first blocked point
                    first_block = df_sig.index[df_sig['Pattern_blocked']].tolist()[0]
                    pos = df_sig.index.get_loc(first_block)
                    left = max(0, pos-15)
                    right = min(len(df_sig)-1, pos+15)
                    neckline_low = df_sig['Low'].iloc[left:right].min()
                    neckline_high = df_sig['High'].iloc[left:right].max()
                    fig.add_hline(y=neckline_low, line=dict(color='purple', width=1, dash='dash'), annotation_text='Neckline Low', annotation_position='bottom right')
                    fig.add_hline(y=neckline_high, line=dict(color='purple', width=1, dash='dash'), annotation_text='Neckline High', annotation_position='top right')

                # re-render chart with annotations
                st.plotly_chart(fig, use_container_width=True)

                # --- Pattern statistics ---
                st.write('Pattern detection stats:')
                num_tri = int(df_sig['triangle_detected'].sum()) if 'triangle_detected' in df_sig.columns else 0
                num_flag = int(df_sig['flag_detected'].sum()) if 'flag_detected' in df_sig.columns else 0
                num_pattern_blocked = int(df_sig['Pattern_blocked'].sum()) if 'Pattern_blocked' in df_sig.columns else 0
                st.write(f'Triangles detected: {num_tri} — Flags detected: {num_flag} — Pattern-blocked candles: {num_pattern_blocked}')

                # trades during patterns (approximate by matching entry_time)
                if not best_trades.empty:
                    tri_entries = best_trades[best_trades['entry_time'].isin(df_sig.index[df_sig.get('triangle_detected', False)])]
                    flag_entries = best_trades[best_trades['entry_time'].isin(df_sig.index[df_sig.get('flag_detected', False)])]
                    def winrate(tr):
                        return (tr['pnl']>0).mean() if len(tr)>0 else float('nan')
                    st.write(f'Trades entered during triangle zones: {len(tri_entries)} | Win rate: {winrate(tri_entries):.2f}')
                    st.write(f'Trades entered during flag zones: {len(flag_entries)} | Win rate: {winrate(flag_entries):.2f}')

    st.success('Batch run complete')
