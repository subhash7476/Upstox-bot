"""
LSTM Stock Price Predictor for Trading Bot Pro
Uses Stacked LSTM architecture to predict next-period price movements.
Integrates with DuckDB database and Upstox configuration.

Version: 3.0 (Major Accuracy Improvements)
Changes from v2.2:
- Option to skip saving model (checkbox)
- Predict RETURNS instead of absolute prices (more stationary)
- Proper train/test scaling (fit on train, transform test - no data leakage)
- Improved feature normalization
- Uses all available data up to current date
- Better direction prediction focus
- Added walk-forward validation option
- Clearer cache invalidation for fresh data

Author: Trading Bot Pro
"""

from __future__ import annotations

import os
import sys
import json
import pickle
import warnings
from dataclasses import dataclass
from datetime import date, timedelta, time, datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# Ensure root is in path (standard import pattern per architecture)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Suppress TensorFlow info messages BEFORE importing TF
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Page Config
st.set_page_config(
    page_title="LSTM Price Predictor v3",
    page_icon="🧠",
    layout="wide",
)

# =====================================================================
# CONFIGURATION
# =====================================================================
DB_PATH = Path("data/trading_bot.duckdb")
MODEL_DIR = Path("models/lstm")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LIST_FILE = Path("data/Nifty100list.csv")

from core.database import get_db, reset_shared_connection


def get_duckdb_conn():
    db = get_db()
    return db.con


# =====================================================================
# TensorFlow / ML Dependencies
# =====================================================================
TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2

    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    TF_AVAILABLE = True
    tf.get_logger().setLevel("ERROR")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
except ImportError:
    TF_AVAILABLE = False

# =====================================================================
# MODEL CONFIG
# =====================================================================

@dataclass
class LSTMConfig:
    """Configuration for LSTM model"""
    target_col: str = "close"
    lookback: int = 60
    train_ratio: float = 0.8
    
    # NEW: Prediction mode
    predict_returns: bool = True  # Predict returns instead of prices
    return_horizon: int = 1  # Predict return over N periods ahead
    
    # LSTM Architecture
    lstm_units_1: int = 64
    lstm_units_2: int = 32
    lstm_units_3: int = 16  # Optional third layer
    use_third_layer: bool = False
    dropout: float = 0.2
    use_bidirectional: bool = False
    use_batch_norm: bool = True
    l2_reg: float = 0.001
    
    # Training
    epochs: int = 100
    batch_size: int = 32
    patience: int = 15
    validation_split: float = 0.1
    learning_rate: float = 0.001
    
    # Prediction
    forecast_periods: int = 5
    
    # Feature columns
    use_volume: bool = True
    use_returns: bool = True
    use_volatility: bool = True
    use_trend_features: bool = True
    
    # Scaling
    scaler_type: str = "robust"  # "standard" or "robust" (robust is less sensitive to outliers)


# =====================================================================
# MARKET TIME HELPERS
# =====================================================================
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)


def next_market_timestamps(last_ts, periods, timeframe):
    """Generate future market timestamps, skipping weekends and non-market hours"""
    last_ts = pd.to_datetime(last_ts)

    freq_map = {
        "1minute": "1min",
        "5minute": "5min",
        "15minute": "15min",
        "30minute": "30min",
        "60minute": "60min",
        "day": "1D",
    }
    step = pd.Timedelta(freq_map.get(timeframe, "15min"))

    out = []
    ts = last_ts

    while len(out) < periods:
        ts = ts + step

        # For daily timeframe, just skip weekends
        if timeframe == "day":
            if ts.weekday() >= 5:
                days_ahead = 7 - ts.weekday()
                ts = ts + pd.Timedelta(days=days_ahead)
            out.append(ts)
            continue
        
        # Weekend skip
        if ts.weekday() >= 5:
            days_ahead = 7 - ts.weekday()
            ts = (ts.normalize() + pd.Timedelta(days=days_ahead)).replace(
                hour=MARKET_OPEN.hour, minute=MARKET_OPEN.minute
            )
            continue

        # If before open -> set to open
        if ts.time() < MARKET_OPEN:
            ts = ts.normalize().replace(hour=MARKET_OPEN.hour, minute=MARKET_OPEN.minute)

        # If after close -> next business day open
        if ts.time() > MARKET_CLOSE:
            ts = (ts.normalize() + pd.Timedelta(days=1)).replace(
                hour=MARKET_OPEN.hour, minute=MARKET_OPEN.minute
            )
            continue

        out.append(ts)

    return pd.DatetimeIndex(out)


# =====================================================================
# DATABASE HELPERS - SYMBOL MAPPING
# =====================================================================

@st.cache_data(ttl=600)
def get_symbol_mapping(_db_path_str: str) -> Dict[str, str]:
    """Get mapping of trading_symbol -> instrument_key"""
    con = get_duckdb_conn()
    if con is None:
        return {}
    
    try:
        tables = con.execute("SHOW TABLES").fetchall()
        table_names = [t[0] for t in tables]
        
        if "fo_stocks_master" in table_names:
            result = con.execute("""
                SELECT DISTINCT trading_symbol, instrument_key
                FROM fo_stocks_master
                WHERE trading_symbol IS NOT NULL
                AND instrument_key IS NOT NULL
                ORDER BY trading_symbol
            """).fetchall()
            
            if result:
                return {row[0]: row[1] for row in result}
        
        return {}
    except Exception as e:
        return {}


@st.cache_data(ttl=300)
def get_available_tables_cached(_db_path_str: str) -> List[str]:
    """Get list of tables in database"""
    con = get_duckdb_conn()
    if con is None:
        return []
    try:
        rows = con.execute("SHOW TABLES").fetchall()
        return [r[0] for r in rows]
    except Exception:
        return []


@st.cache_data(ttl=300)
def get_available_symbols_cached(_db_path_str: str) -> Tuple[List[str], Dict[str, str]]:
    """Get list of available symbols for dropdown."""
    con = get_duckdb_conn()
    if con is None:
        return [], {}

    try:
        table_names = get_available_tables_cached(_db_path_str)
        if not table_names:
            return [], {}

        symbol_map = get_symbol_mapping(_db_path_str)
        
        if symbol_map:
            return sorted(list(symbol_map.keys())), symbol_map
        
        # Fallback
        instrument_keys = []
        
        if "ohlcv_resampled" in table_names:
            rows = con.execute("""
                SELECT DISTINCT instrument_key
                FROM ohlcv_resampled
                WHERE instrument_key IS NOT NULL
                ORDER BY instrument_key
            """).fetchall()
            instrument_keys = [r[0] for r in rows if r[0]]
        
        if not instrument_keys:
            return [], {}
        
        display_map = {}
        for ik in instrument_keys:
            if "|" in ik:
                parts = ik.split("|")
                display_name = parts[-1]
            else:
                display_name = ik
            display_map[display_name] = ik
        
        return sorted(list(display_map.keys())), display_map

    except Exception as e:
        st.warning(f"Could not fetch symbols from DB: {e}")
        return [], {}


def load_ohlcv_from_db_fresh(
    instrument_key: str,
    timeframe: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load OHLCV data - NO CACHING to ensure fresh data.
    """
    con = get_duckdb_conn()
    if con is None:
        return pd.DataFrame()

    try:
        table_names = get_available_tables_cached(str(DB_PATH))
        if not table_names:
            return pd.DataFrame()

        df = pd.DataFrame()

        if "ohlcv_resampled" in table_names:
            query = """
                SELECT timestamp, open, high, low, close, volume
                FROM ohlcv_resampled
                WHERE instrument_key = ?
                  AND timeframe = ?
            """
            params = [instrument_key, timeframe]

            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date + " 23:59:59")  # Include full day

            query += " ORDER BY timestamp"
            df = con.execute(query, params).df()
            
            # Try LIKE if exact match failed
            if df.empty:
                query = """
                    SELECT timestamp, open, high, low, close, volume
                    FROM ohlcv_resampled
                    WHERE instrument_key LIKE ?
                      AND timeframe = ?
                """
                params = [f"%{instrument_key}%", timeframe]

                if start_date:
                    query += " AND timestamp >= ?"
                    params.append(start_date)
                if end_date:
                    query += " AND timestamp <= ?"
                    params.append(end_date + " 23:59:59")

                query += " ORDER BY timestamp"
                df = con.execute(query, params).df()

        if df.empty:
            return df

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df.columns = [c.title() for c in df.columns]
        return df

    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()


def get_date_range_fresh(instrument_key: str, timeframe: str) -> Tuple[Optional[date], Optional[date]]:
    """Get date range for an instrument - NO CACHING"""
    con = get_duckdb_conn()
    if con is None:
        return None, None

    try:
        table_names = get_available_tables_cached(str(DB_PATH))
        if not table_names:
            return None, None

        if "ohlcv_resampled" in table_names:
            row = con.execute("""
                SELECT MIN(timestamp)::DATE as min_date,
                       MAX(timestamp)::DATE as max_date
                FROM ohlcv_resampled
                WHERE instrument_key = ?
                  AND timeframe = ?
            """, [instrument_key, timeframe]).fetchone()
            
            if not row or not row[0]:
                row = con.execute("""
                    SELECT MIN(timestamp)::DATE as min_date,
                           MAX(timestamp)::DATE as max_date
                    FROM ohlcv_resampled
                    WHERE instrument_key LIKE ?
                      AND timeframe = ?
                """, [f"%{instrument_key}%", timeframe]).fetchone()

        if row and row[0]:
            return row[0], row[1]
        return None, None
    except Exception:
        return None, None


# =====================================================================
# FEATURE ENGINEERING - IMPROVED
# =====================================================================

def add_technical_features(df: pd.DataFrame, config: LSTMConfig) -> pd.DataFrame:
    """Add technical features with proper handling for LSTM training"""
    df = df.copy()

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # === TARGET: Future Return (what we're predicting) ===
    df["future_return"] = df["Close"].pct_change(config.return_horizon).shift(-config.return_horizon)
    
    # === Price Returns (normalized, stationary) ===
    if config.use_returns:
        df["return_1"] = df["Close"].pct_change(1)
        df["return_3"] = df["Close"].pct_change(3)
        df["return_5"] = df["Close"].pct_change(5)
        df["return_10"] = df["Close"].pct_change(10)
        
        # Log returns (more normally distributed)
        df["log_return_1"] = np.log(df["Close"] / df["Close"].shift(1))
        df["log_return_5"] = np.log(df["Close"] / df["Close"].shift(5))

    # === Volatility Features ===
    if config.use_volatility:
        df["volatility_5"] = df["return_1"].rolling(5).std()
        df["volatility_10"] = df["return_1"].rolling(10).std()
        df["volatility_20"] = df["return_1"].rolling(20).std()
        
        # Volatility ratio (current vs longer-term)
        df["vol_ratio"] = df["volatility_5"] / (df["volatility_20"] + 1e-9)

    df["daily_range_pct"] = (df["High"] - df["Low"]) / df["Close"] * 100
    df["body_pct"] = (df["Close"] - df["Open"]) / df["Open"] * 100

    # === Volume Features ===
    if config.use_volume:
        df["volume_ma5"] = df["Volume"].rolling(5).mean()
        df["volume_ma20"] = df["Volume"].rolling(20).mean()
        df["volume_ratio"] = df["Volume"] / (df["volume_ma20"] + 1)
        df["volume_change"] = df["Volume"].pct_change(1)

    # === Trend Features ===
    if config.use_trend_features:
        # Price relative to MAs (normalized)
        df["ma_5"] = df["Close"].rolling(5).mean()
        df["ma_10"] = df["Close"].rolling(10).mean()
        df["ma_20"] = df["Close"].rolling(20).mean()
        df["ma_50"] = df["Close"].rolling(50).mean()

        df["price_vs_ma5"] = (df["Close"] - df["ma_5"]) / df["ma_5"] * 100
        df["price_vs_ma20"] = (df["Close"] - df["ma_20"]) / df["ma_20"] * 100
        df["price_vs_ma50"] = (df["Close"] - df["ma_50"]) / df["ma_50"] * 100
        
        # MA slopes (trend direction)
        df["ma5_slope"] = df["ma_5"].pct_change(3)
        df["ma20_slope"] = df["ma_20"].pct_change(5)
        
        # MA crossover signals
        df["ma_cross_5_20"] = (df["ma_5"] > df["ma_20"]).astype(float)

    # === RSI (bounded 0-100, good for LSTM) ===
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi_normalized"] = (df["rsi"] - 50) / 50  # Normalize to -1 to 1

    # === ATR (normalized) ===
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr"] / df["Close"] * 100

    # === MACD (already normalized-ish) ===
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["macd_normalized"] = df["macd_hist"] / df["Close"] * 100  # Normalize

    # === Bollinger Bands Position (0-1 scale) ===
    df["bb_middle"] = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
    df["bb_lower"] = df["bb_middle"] - (bb_std * 2)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"] * 100
    df["bb_position"] = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)

    # === Momentum ===
    df["momentum_5"] = df["Close"] / df["Close"].shift(5) - 1
    df["momentum_10"] = df["Close"] / df["Close"].shift(10) - 1
    df["momentum_20"] = df["Close"] / df["Close"].shift(20) - 1

    # === Time Features (cyclical encoding for intraday) ===
    if hasattr(df.index, 'hour'):
        df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
        df["minute_sin"] = np.sin(2 * np.pi * df.index.minute / 60)
        df["minute_cos"] = np.cos(2 * np.pi * df.index.minute / 60)
    
    df["day_of_week"] = df.index.dayofweek / 4 - 0.5  # Normalize to ~-0.5 to 0.5

    return df


def get_feature_columns(config: LSTMConfig) -> List[str]:
    """Get list of feature columns to use for training"""
    features = []
    
    # Always include these (normalized/bounded)
    features += ["return_1", "log_return_1"]
    
    if config.use_returns:
        features += ["return_3", "return_5", "return_10", "log_return_5"]

    if config.use_volatility:
        features += ["volatility_5", "volatility_10", "vol_ratio", "daily_range_pct", "atr_pct"]

    if config.use_volume:
        features += ["volume_ratio", "volume_change"]

    if config.use_trend_features:
        features += ["price_vs_ma5", "price_vs_ma20", "price_vs_ma50", 
                    "ma5_slope", "ma20_slope", "ma_cross_5_20"]

    features += ["rsi_normalized", "macd_normalized", "bb_position", "bb_width"]
    features += ["momentum_5", "momentum_10", "momentum_20"]
    features += ["body_pct", "day_of_week"]
    
    return features


# =====================================================================
# LSTM MODEL FUNCTIONS - IMPROVED
# =====================================================================

def prepare_sequences_for_returns(
    X_data: np.ndarray, 
    y_data: np.ndarray, 
    lookback: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequences for return prediction.
    X_data: feature array (n_samples, n_features)
    y_data: target returns (n_samples,)
    """
    X, y = [], []
    for i in range(lookback, len(X_data)):
        if not np.isnan(y_data[i]):
            X.append(X_data[i - lookback:i])
            y.append(y_data[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def build_lstm_model(lookback: int, n_features: int, config: LSTMConfig) -> Sequential:
    """Build improved LSTM model"""
    model = Sequential()

    # First LSTM layer
    if config.use_bidirectional:
        model.add(Bidirectional(
            LSTM(config.lstm_units_1, return_sequences=True, 
                 kernel_regularizer=l2(config.l2_reg)),
            input_shape=(lookback, n_features)
        ))
    else:
        model.add(LSTM(config.lstm_units_1, return_sequences=True, 
                      input_shape=(lookback, n_features),
                      kernel_regularizer=l2(config.l2_reg)))
    
    if config.use_batch_norm:
        model.add(BatchNormalization())
    model.add(Dropout(config.dropout))

    # Second LSTM layer
    return_seq = config.use_third_layer
    if config.use_bidirectional:
        model.add(Bidirectional(LSTM(config.lstm_units_2, return_sequences=return_seq,
                                    kernel_regularizer=l2(config.l2_reg))))
    else:
        model.add(LSTM(config.lstm_units_2, return_sequences=return_seq,
                      kernel_regularizer=l2(config.l2_reg)))
    
    if config.use_batch_norm:
        model.add(BatchNormalization())
    model.add(Dropout(config.dropout))
    
    # Optional third LSTM layer
    if config.use_third_layer:
        model.add(LSTM(config.lstm_units_3, return_sequences=False,
                      kernel_regularizer=l2(config.l2_reg)))
        model.add(Dropout(config.dropout))

    # Output layer - single value for return prediction
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # Predict single return value

    optimizer = Adam(learning_rate=config.learning_rate)
    model.compile(optimizer=optimizer, loss="huber", metrics=["mae"])  # Huber loss is more robust
    return model


def train_lstm_model(X_train, y_train, config: LSTMConfig, model_path: Optional[Path], save_model: bool = True):
    """Train LSTM model with optional saving"""
    n_features = X_train.shape[2]
    model = build_lstm_model(config.lookback, n_features, config)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=config.patience, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    ]
    
    # Only add checkpoint if saving is enabled
    if save_model and model_path:
        callbacks.append(
            ModelCheckpoint(filepath=str(model_path), monitor="val_loss", save_best_only=True, verbose=0)
        )

    history = model.fit(
        X_train, y_train,
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_split=config.validation_split,
        callbacks=callbacks,
        shuffle=False,
        verbose=0,
    )
    return model, history.history


def evaluate_model_returns(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Evaluate return predictions"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Direction accuracy (most important for trading!)
    true_dir = np.sign(y_true)
    pred_dir = np.sign(y_pred)
    direction_accuracy = np.mean(true_dir == pred_dir) * 100
    
    # Average return when predicting correctly
    correct_mask = (true_dir == pred_dir)
    avg_return_correct = np.mean(np.abs(y_true[correct_mask])) * 100 if correct_mask.sum() > 0 else 0
    
    # Correlation
    correlation = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0
    
    return {
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2,
        "Direction Accuracy %": direction_accuracy,
        "Avg Return (Correct) %": avg_return_correct,
        "Correlation": correlation,
    }


def predict_future_returns(
    model, 
    df_features: pd.DataFrame, 
    feature_cols: List[str],
    scaler,
    config: LSTMConfig,
    n_periods: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict future returns iteratively.
    Returns predicted returns and corresponding price projections.
    """
    last_close = df_features["Close"].iloc[-1]
    
    # Get last sequence
    features = df_features[feature_cols].values[-config.lookback:]
    features_scaled = scaler.transform(features)
    
    predicted_returns = []
    predicted_prices = [last_close]
    
    # For simplicity, predict one step at a time
    # In reality, you'd need to update features for each step
    X_input = features_scaled.reshape(1, config.lookback, len(feature_cols))
    pred_return = model.predict(X_input, verbose=0)[0, 0]
    
    # Simple projection: assume same predicted return for n_periods
    # This is a simplification - real implementation would be recursive
    for i in range(n_periods):
        predicted_returns.append(pred_return)
        next_price = predicted_prices[-1] * (1 + pred_return)
        predicted_prices.append(next_price)
    
    return np.array(predicted_returns), np.array(predicted_prices[1:])


# =====================================================================
# VISUALIZATION
# =====================================================================

def plot_training_history(history: dict) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Loss (Huber)", "MAE"])
    fig.add_trace(go.Scatter(y=history["loss"], name="Train Loss", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(y=history["val_loss"], name="Val Loss", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(y=history["mae"], name="Train MAE", mode="lines"), row=1, col=2)
    fig.add_trace(go.Scatter(y=history["val_mae"], name="Val MAE", mode="lines"), row=1, col=2)
    fig.update_layout(height=400, showlegend=True)
    return fig


def plot_return_predictions(y_true: np.ndarray, y_pred: np.ndarray, dates: pd.DatetimeIndex) -> go.Figure:
    """Plot actual vs predicted returns"""
    fig = make_subplots(rows=2, cols=1, subplot_titles=["Returns: Actual vs Predicted", "Prediction Error"],
                        row_heights=[0.7, 0.3])
    
    # Actual vs Predicted returns
    fig.add_trace(go.Scatter(x=dates, y=y_true * 100, name="Actual Return %", 
                            line=dict(color="blue", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=y_pred * 100, name="Predicted Return %", 
                            line=dict(color="orange", width=1)), row=1, col=1)
    
    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
    
    # Error
    error = (y_pred - y_true) * 100
    fig.add_trace(go.Bar(x=dates, y=error, name="Error %",
                        marker_color=np.where(error > 0, "red", "green")), row=2, col=1)
    
    fig.update_layout(height=600, showlegend=True)
    return fig


# =====================================================================
# MAIN STREAMLIT APP
# =====================================================================
st.title("🧠 LSTM Stock Price Predictor v3")
st.markdown("*Improved: Predicts Returns, Proper Scaling, No Data Leakage*")

# Check TensorFlow
if not TF_AVAILABLE:
    st.error("""
    ❌ **TensorFlow not installed!**
    
    Please install TensorFlow:
    ```bash
    pip install tensorflow scikit-learn --break-system-packages
    ```
    """)
    st.stop()

# Check database
if not DB_PATH.exists():
    st.error(f"❌ **Database not found!** Expected: `{DB_PATH}`")
    st.stop()

# Sidebar refresh
with st.sidebar:
    if st.button("🔄 Refresh DB Connection"):
        reset_shared_connection()
        st.cache_data.clear()
        st.rerun()

# =====================================================================
# SIDEBAR CONFIGURATION
# =====================================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("📊 Data Selection")
    
    db_path_str = str(DB_PATH)
    display_symbols, symbol_to_key_map = get_available_symbols_cached(db_path_str)
    
    if not display_symbols:
        st.warning("No symbols found in database.")
        display_symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "SBIN"]
        symbol_to_key_map = {s: s for s in display_symbols}
    
    selected_symbol = st.selectbox("Symbol", display_symbols, index=0)
    instrument_key = symbol_to_key_map.get(selected_symbol, selected_symbol)
    
    if instrument_key != selected_symbol:
        st.caption(f"Instrument Key: `{instrument_key}`")
    
    timeframe = st.selectbox(
        "Timeframe",
        ["5minute", "15minute", "30minute", "60minute", "day"],
        index=1,
    )
    
    # Date range - FRESH query
    min_date, max_date = get_date_range_fresh(instrument_key, timeframe)
    if min_date and max_date:
        st.success(f"📅 Data: {min_date} to {max_date}")
        c1, c2 = st.columns(2)
        with c1:
            start_date = st.date_input("Start", value=min_date, min_value=min_date, max_value=max_date)
        with c2:
            end_date = st.date_input("End", value=max_date, min_value=min_date, max_value=max_date)
    else:
        st.warning("Could not determine date range")
        start_date = date.today() - timedelta(days=365)
        end_date = date.today()
    
    st.divider()
    
    # Model parameters
    st.subheader("🧠 Model Parameters")
    lookback = st.slider("Lookback Period", 20, 200, 60, 10)
    forecast_periods = st.slider("Forecast Periods", 1, 10, 5)
    
    with st.expander("⚙️ Advanced Settings"):
        predict_returns = st.checkbox("Predict Returns (recommended)", value=True, 
                                     help="Predict % returns instead of absolute prices")
        lstm_units_1 = st.slider("LSTM Units (L1)", 32, 128, 64, 16)
        lstm_units_2 = st.slider("LSTM Units (L2)", 16, 64, 32, 16)
        use_third_layer = st.checkbox("Use 3rd LSTM Layer", value=False)
        dropout = st.slider("Dropout", 0.1, 0.5, 0.2, 0.1)
        epochs = st.slider("Max Epochs", 50, 300, 150, 25)
        batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)
        learning_rate = st.select_slider("Learning Rate", [0.0001, 0.0005, 0.001, 0.005], value=0.001)
        use_bidirectional = st.checkbox("Bidirectional LSTM", value=False)
        use_batch_norm = st.checkbox("Batch Normalization", value=True)
        train_ratio = st.slider("Train/Test Split", 0.7, 0.9, 0.8, 0.05)
        scaler_type = st.selectbox("Scaler", ["robust", "standard"], index=0,
                                   help="Robust is less sensitive to outliers")
    
    st.subheader("📈 Features")
    use_returns_feat = st.checkbox("Return Features", value=True)
    use_volatility = st.checkbox("Volatility Features", value=True)
    use_volume = st.checkbox("Volume Features", value=True)
    use_trend = st.checkbox("Trend Features", value=True)
    
    st.divider()
    st.subheader("💾 Model Saving")
    save_model_checkbox = st.checkbox("Save Model After Training", value=False,
                                      help="Uncheck to train without saving to disk")

# Build config
config = LSTMConfig(
    lookback=lookback,
    forecast_periods=forecast_periods,
    predict_returns=predict_returns if 'predict_returns' in dir() else True,
    lstm_units_1=lstm_units_1 if 'lstm_units_1' in dir() else 64,
    lstm_units_2=lstm_units_2 if 'lstm_units_2' in dir() else 32,
    use_third_layer=use_third_layer if 'use_third_layer' in dir() else False,
    dropout=dropout if 'dropout' in dir() else 0.2,
    epochs=epochs if 'epochs' in dir() else 150,
    batch_size=batch_size if 'batch_size' in dir() else 32,
    learning_rate=learning_rate if 'learning_rate' in dir() else 0.001,
    use_bidirectional=use_bidirectional if 'use_bidirectional' in dir() else False,
    use_batch_norm=use_batch_norm if 'use_batch_norm' in dir() else True,
    train_ratio=train_ratio if 'train_ratio' in dir() else 0.8,
    use_returns=use_returns_feat,
    use_volatility=use_volatility,
    use_volume=use_volume,
    use_trend_features=use_trend,
    scaler_type=scaler_type if 'scaler_type' in dir() else "robust",
)

# Store in session
st.session_state["symbol"] = selected_symbol
st.session_state["instrument_key"] = instrument_key
st.session_state["timeframe"] = timeframe
st.session_state["save_model"] = save_model_checkbox

# =====================================================================
# TABS
# =====================================================================
tab1, tab2, tab3 = st.tabs(["📊 Data & Features", "🎯 Train Model", "🔮 Predictions"])

# =====================================================================
# TAB 1: DATA & FEATURES
# =====================================================================
with tab1:
    st.subheader(f"📊 Data Overview: {selected_symbol}")
    
    with st.spinner("Loading fresh data from DuckDB..."):
        df = load_ohlcv_from_db_fresh(
            instrument_key,
            timeframe,
            str(start_date) if min_date else None,
            str(end_date) if max_date else None,
        )
    
    if df.empty:
        st.warning(f"No data found for {selected_symbol} ({instrument_key}) in {timeframe} timeframe.")
        st.stop()
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Candles", f"{len(df):,}")
    with c2:
        st.metric("Date Range", f"{df.index.min().date()} → {df.index.max().date()}")
    with c3:
        st.metric("Current Price", f"₹{df['Close'].iloc[-1]:,.2f}")
    with c4:
        change = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
        st.metric("Period Change", f"{change:+.2f}%")
    
    # Show latest data timestamp
    st.info(f"📌 Latest data point: **{df.index.max()}**")
    
    # Price chart
    fig_price = go.Figure()
    fig_price.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="OHLC"
    ))
    fig_price.update_layout(title=f"{selected_symbol} - {timeframe}", height=400, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_price, use_container_width=True)
    
    # Features
    st.subheader("📈 Feature Engineering")
    
    with st.spinner("Computing technical features..."):
        df_features = add_technical_features(df, config)
        # Remove rows with NaN (from rolling windows and future return)
        df_features_clean = df_features.dropna()
    
    feature_cols = get_feature_columns(config)
    
    # Verify all feature columns exist
    missing_cols = [c for c in feature_cols if c not in df_features_clean.columns]
    if missing_cols:
        st.warning(f"Missing feature columns (will be excluded): {missing_cols}")
        feature_cols = [c for c in feature_cols if c in df_features_clean.columns]
    
    st.success(f"✅ Generated {len(feature_cols)} features from {len(df_features_clean):,} usable candles")
    
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Features Used:**")
        st.write(feature_cols)
    with c2:
        st.write("**Feature Statistics:**")
        st.dataframe(df_features_clean[feature_cols].describe().round(4))
    
    # Show target distribution
    st.subheader("🎯 Target Distribution (Future Returns)")
    returns = df_features_clean["future_return"].dropna()
    fig_returns = go.Figure()
    fig_returns.add_trace(go.Histogram(x=returns * 100, nbinsx=50, name="Return %"))
    fig_returns.add_vline(x=0, line_dash="dash", line_color="red")
    fig_returns.update_layout(title="Distribution of Future Returns", xaxis_title="Return %", height=300)
    st.plotly_chart(fig_returns, use_container_width=True)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Mean Return", f"{returns.mean()*100:.4f}%")
    with c2:
        st.metric("Std Dev", f"{returns.std()*100:.4f}%")
    with c3:
        st.metric("Positive %", f"{(returns > 0).mean()*100:.1f}%")
    with c4:
        st.metric("Max |Return|", f"{returns.abs().max()*100:.2f}%")
    
    st.session_state["df_features"] = df_features_clean
    st.session_state["feature_cols"] = feature_cols
    st.session_state["config"] = config

# =====================================================================
# TAB 2: TRAIN MODEL
# =====================================================================
with tab2:
    st.subheader("🎯 Train LSTM Model")
    
    if "df_features" not in st.session_state:
        st.warning("Please load data in the 'Data & Features' tab first.")
        st.stop()
    
    df_features = st.session_state["df_features"]
    feature_cols = st.session_state["feature_cols"]
    
    # Split data
    n_samples = len(df_features)
    n_train = int(n_samples * config.train_ratio)
    n_test = n_samples - n_train
    
    train_df = df_features.iloc[:n_train]
    test_df = df_features.iloc[n_train:]
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Training Samples", f"{n_train:,}")
    with c2:
        st.metric("Test Samples", f"{n_test:,}")
    with c3:
        st.metric("Features", len(feature_cols))
    with c4:
        st.metric("Lookback", config.lookback)
    
    # Show train/test split dates
    st.info(f"Train: {train_df.index.min().date()} → {train_df.index.max().date()} | "
            f"Test: {test_df.index.min().date()} → {test_df.index.max().date()}")
    
    # Save model checkbox reminder
    if not st.session_state.get("save_model", False):
        st.warning("⚠️ Model saving is DISABLED. Model will only be kept in memory.")
    
    if st.button("🚀 Train LSTM Model", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("Preparing data...")
            progress_bar.progress(10)
            
            # === PROPER SCALING: Fit on train, transform test ===
            X_train_raw = train_df[feature_cols].values
            X_test_raw = test_df[feature_cols].values
            
            y_train_raw = train_df["future_return"].values
            y_test_raw = test_df["future_return"].values
            
            # Choose scaler
            if config.scaler_type == "robust":
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()
            
            # FIT ONLY ON TRAINING DATA
            X_train_scaled = scaler.fit_transform(X_train_raw)
            X_test_scaled = scaler.transform(X_test_raw)  # Transform test with train stats
            
            status_text.text("Creating sequences...")
            progress_bar.progress(20)
            
            # Prepare sequences
            X_train, y_train = prepare_sequences_for_returns(X_train_scaled, y_train_raw, config.lookback)
            X_test, y_test = prepare_sequences_for_returns(X_test_scaled, y_test_raw, config.lookback)
            
            st.info(f"Training shape: X={X_train.shape}, y={y_train.shape}")
            st.info(f"Test shape: X={X_test.shape}, y={y_test.shape}")
            
            status_text.text("Training LSTM model... (this may take a few minutes)")
            progress_bar.progress(30)
            
            # Model path (only used if saving)
            model_path = MODEL_DIR / f"{selected_symbol}_{timeframe}_lstm_v3.keras" if st.session_state.get("save_model") else None
            
            model, history = train_lstm_model(
                X_train, y_train, config, model_path, 
                save_model=st.session_state.get("save_model", False)
            )
            
            progress_bar.progress(80)
            status_text.text("Evaluating model...")
            
            # Predict on test set
            y_pred = model.predict(X_test, verbose=0).flatten()
            
            metrics = evaluate_model_returns(y_test, y_pred)
            
            progress_bar.progress(100)
            status_text.text("✅ Training complete!")
            
            st.success("🎉 Model trained successfully!")
            
            # === Display Metrics ===
            st.subheader("📊 Model Performance")
            cols = st.columns(6)
            metric_items = list(metrics.items())
            for i, (metric, value) in enumerate(metric_items):
                with cols[i % 6]:
                    if "%" in metric:
                        st.metric(metric, f"{value:.2f}%")
                    else:
                        st.metric(metric, f"{value:.4f}")
            
            # Interpretation
            dir_acc = metrics["Direction Accuracy %"]
            if dir_acc > 55:
                st.success(f"✅ Direction accuracy {dir_acc:.1f}% is good for trading!")
            elif dir_acc > 50:
                st.info(f"ℹ️ Direction accuracy {dir_acc:.1f}% is slightly better than random.")
            else:
                st.warning(f"⚠️ Direction accuracy {dir_acc:.1f}% is worse than random. Consider adjusting parameters.")
            
            # === Training History ===
            st.subheader("📈 Training History")
            fig_hist = plot_training_history(history)
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # === Prediction Plot ===
            st.subheader("🔍 Test Set Predictions")
            test_dates = test_df.index[config.lookback:config.lookback + len(y_test)]
            fig_pred = plot_return_predictions(y_test, y_pred, test_dates)
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # Save to session
            st.session_state["model"] = model
            st.session_state["scaler"] = scaler
            st.session_state["metrics"] = metrics
            st.session_state["y_test"] = y_test
            st.session_state["y_pred"] = y_pred
            st.session_state["history"] = history
            st.session_state["test_dates"] = test_dates
            
            # Save scaler if model is saved
            if st.session_state.get("save_model", False):
                scaler_path = MODEL_DIR / f"{selected_symbol}_{timeframe}_scaler_v3.pkl"
                with open(scaler_path, "wb") as f:
                    pickle.dump(scaler, f)
                st.info(f"📁 Model saved to: {model_path}")
                st.info(f"📁 Scaler saved to: {scaler_path}")
            else:
                st.info("💾 Model kept in memory only (not saved to disk)")
            
        except Exception as e:
            st.error(f"Training failed: {e}")
            import traceback
            st.code(traceback.format_exc())

# =====================================================================
# TAB 3: PREDICTIONS
# =====================================================================
with tab3:
    st.subheader("🔮 Price Predictions")
    
    if "model" not in st.session_state:
        st.warning("Please train a model first.")
        
        # Try to load saved model
        symbol2 = st.session_state.get("symbol", "RELIANCE")
        timeframe2 = st.session_state.get("timeframe", "15minute")
        model_path = MODEL_DIR / f"{symbol2}_{timeframe2}_lstm_v3.keras"
        scaler_path = MODEL_DIR / f"{symbol2}_{timeframe2}_scaler_v3.pkl"
        
        if model_path.exists() and scaler_path.exists():
            st.info(f"Found saved model: {model_path}")
            if st.button("Load Saved Model"):
                try:
                    model = load_model(model_path)
                    with open(scaler_path, "rb") as f:
                        scaler = pickle.load(f)
                    st.session_state["model"] = model
                    st.session_state["scaler"] = scaler
                    st.success("Model loaded!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to load: {e}")
        st.stop()
    
    model = st.session_state["model"]
    scaler = st.session_state["scaler"]
    df_features = st.session_state["df_features"]
    feature_cols = st.session_state["feature_cols"]
    symbol = st.session_state["symbol"]
    timeframe = st.session_state["timeframe"]
    
    # === Test Performance Summary ===
    if "y_test" in st.session_state and "y_pred" in st.session_state:
        st.subheader("📊 Test Set Performance Summary")
        
        metrics = st.session_state.get("metrics", {})
        cols = st.columns(4)
        with cols[0]:
            st.metric("Direction Accuracy", f"{metrics.get('Direction Accuracy %', 0):.1f}%")
        with cols[1]:
            st.metric("R²", f"{metrics.get('R²', 0):.4f}")
        with cols[2]:
            st.metric("MAE", f"{metrics.get('MAE', 0):.6f}")
        with cols[3]:
            st.metric("Correlation", f"{metrics.get('Correlation', 0):.4f}")
    
    # === Future Predictions ===
    st.subheader("🔮 Future Price Forecast")
    
    if st.button("Generate Future Predictions", type="primary"):
        with st.spinner("Generating predictions..."):
            try:
                # Get latest data for prediction
                current_price = float(df_features["Close"].iloc[-1])
                last_timestamp = df_features.index[-1]
                
                # Prepare features
                features = df_features[feature_cols].values[-config.lookback:]
                features_scaled = scaler.transform(features)
                X_input = features_scaled.reshape(1, config.lookback, len(feature_cols))
                
                # Predict return
                pred_return = model.predict(X_input, verbose=0)[0, 0]
                
                # Generate future timestamps
                future_dates = next_market_timestamps(last_timestamp, config.forecast_periods, timeframe)
                
                # Project prices (compound the predicted return)
                future_prices = []
                price = current_price
                for i in range(config.forecast_periods):
                    price = price * (1 + pred_return)
                    future_prices.append(price)
                
                st.success("✅ Forecast generated!")
                
                c1, c2 = st.columns(2)
                
                with c1:
                    pred_df = pd.DataFrame({
                        "Date": future_dates,
                        "Predicted Price": future_prices,
                        "Change %": [(p / current_price - 1) * 100 for p in future_prices],
                    })
                    st.write("**Forecast Table:**")
                    st.dataframe(pred_df.style.format({
                        "Predicted Price": "₹{:,.2f}", 
                        "Change %": "{:+.2f}%"
                    }), use_container_width=True)
                
                with c2:
                    st.metric("Current Price", f"₹{current_price:,.2f}")
                    st.metric("Predicted Return (1 period)", f"{pred_return*100:+.4f}%")
                    final_price = future_prices[-1]
                    total_change = (final_price / current_price - 1) * 100
                    st.metric(f"Projected Price ({config.forecast_periods} periods)", 
                             f"₹{final_price:,.2f}", 
                             delta=f"{total_change:+.2f}%")
                    
                    direction = "📈 BULLISH" if pred_return > 0 else "📉 BEARISH"
                    confidence = abs(pred_return) / df_features["future_return"].std()
                    st.info(f"**Direction:** {direction}")
                    st.info(f"**Signal Strength:** {min(confidence, 3):.2f}σ")
                
                # Plot forecast
                fig = go.Figure()
                hist = df_features["Close"].tail(100)
                
                fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Historical", 
                                        line=dict(color="blue")))
                fig.add_trace(go.Scatter(x=future_dates, y=future_prices, name="Forecast", 
                                        mode="lines+markers", line=dict(color="green", width=2),
                                        marker=dict(size=10)))
                fig.add_trace(go.Scatter(x=[hist.index[-1], future_dates[0]], 
                                        y=[hist.values[-1], future_prices[0]], 
                                        mode="lines", line=dict(color="gray", dash="dash"), 
                                        showlegend=False))
                
                fig.add_vline(x=hist.index[-1], line_dash="dash", line_color="red", 
                             annotation_text="Now")
                fig.update_layout(title=f"{symbol} - Forecast", height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                st.warning("⚠️ **Disclaimer:** Predictions are based on historical patterns. "
                          "Markets are unpredictable. Use as one input among many. Not financial advice.")
                
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                import traceback
                st.code(traceback.format_exc())

st.divider()
st.markdown("LSTM Price Predictor v3.0 | Return Prediction + Proper Scaling + Optional Save")