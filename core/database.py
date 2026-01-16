"""
DuckDB Database Manager for Trading Bot
Handles all database operations with ACID transactions

Version: 2.2 - Thread-Safe Shared Connection Pattern
- Added shared connection management via st.session_state to prevent
  "Can't open a connection to same database file with a different configuration" errors
- Added thread-safe execute_safe() for WebSocket callbacks
- All existing TradingDB functionality preserved
"""

import threading
import duckdb
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, date
import json
import uuid

# ============================================================================
# SHARED CONNECTION MANAGEMENT
# ============================================================================

# Default database path
_DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "trading_bot.duckdb"

# Global connection storage (for non-Streamlit contexts)
_global_connection: Optional[duckdb.DuckDBPyConnection] = None

# Thread lock for safe concurrent access
_db_lock = threading.Lock()


def _get_streamlit_session_state():
    """
    Safely get Streamlit session_state if available.
    Returns None if not in Streamlit context.
    """
    try:
        import streamlit as st
        # Check if we're in a Streamlit context
        if hasattr(st, 'session_state'):
            return st.session_state
    except ImportError:
        pass
    except Exception:
        pass
    return None


def get_shared_connection(db_path: Path = None) -> Optional[duckdb.DuckDBPyConnection]:
    """
    Get or create a shared DuckDB connection.

    This function ensures only ONE connection exists per session (Streamlit)
    or globally (non-Streamlit), preventing connection conflicts.

    Args:
        db_path: Path to database file (defaults to data/trading_bot.duckdb)

    Returns:
        DuckDB connection or None if database doesn't exist
    """
    global _global_connection

    if db_path is None:
        db_path = _DEFAULT_DB_PATH

    db_path = Path(db_path)

    if not db_path.exists():
        return None

    session_state = _get_streamlit_session_state()

    # Streamlit context - use session_state
    if session_state is not None:
        conn_key = "shared_duckdb_conn"

        # Check for existing valid connection
        if conn_key in session_state and session_state[conn_key] is not None:
            con = session_state[conn_key]
            try:
                con.execute("SELECT 1").fetchone()
                return con
            except Exception:
                # Connection is stale
                try:
                    con.close()
                except:
                    pass
                session_state[conn_key] = None

        # Create new connection with optimized settings
        try:
            con = duckdb.connect(
                str(db_path),
                config={
                    'threads': 4,
                    'max_memory': '4GB',
                    'default_order': 'ASC'
                }
            )
            session_state[conn_key] = con
            return con
        except Exception as e:
            print(f"Failed to connect to database: {e}")
            return None

    # Non-Streamlit context - use global variable
    else:
        if _global_connection is not None:
            try:
                _global_connection.execute("SELECT 1").fetchone()
                return _global_connection
            except Exception:
                try:
                    _global_connection.close()
                except:
                    pass
                _global_connection = None

        # Create new connection
        try:
            _global_connection = duckdb.connect(
                str(db_path),
                config={
                    'threads': 4,
                    'max_memory': '4GB',
                    'default_order': 'ASC'
                }
            )
            return _global_connection
        except Exception as e:
            print(f"Failed to connect to database: {e}")
            return None


def close_shared_connection():
    """
    Close the shared database connection.
    Call this when you need to release the database lock.
    """
    global _global_connection

    session_state = _get_streamlit_session_state()

    # Close Streamlit session connection
    if session_state is not None:
        con = session_state.pop("shared_duckdb_conn", None)
        if con is not None:
            try:
                con.close()
            except:
                pass

    # Close global connection
    if _global_connection is not None:
        try:
            _global_connection.close()
        except:
            pass
        _global_connection = None


def reset_shared_connection():
    """
    Reset the database connection.
    Useful when switching between pages or after errors.
    """
    close_shared_connection()

    # Clear Streamlit cache if available
    try:
        import streamlit as st
        st.cache_data.clear()
    except:
        pass


class TradingDB:
    """
    Main database interface for the trading bot.
    Provides methods for OHLCV storage, instrument management, and backtesting.

    Now uses shared connection pattern to prevent connection conflicts
    between multiple Streamlit pages.

    Thread-safe: Use execute_safe() for WebSocket callbacks.
    """

    def __init__(self, db_path: Path = None):
        """
        Initialize database connection and create schema if needed.

        Args:
            db_path: Path to .duckdb file (default: data/trading_bot.duckdb)
        """
        if db_path is None:
            db_path = _DEFAULT_DB_PATH

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Use shared connection instead of creating new one
        self.con = get_shared_connection(self.db_path)

        if self.con is None:
            # Database doesn't exist yet, create it
            self.con = duckdb.connect(
                str(self.db_path),
                config={
                    'threads': 4,
                    'max_memory': '4GB',
                    'default_order': 'ASC'
                }
            )
            # Store in session state if available
            session_state = _get_streamlit_session_state()
            if session_state is not None:
                session_state["shared_duckdb_conn"] = self.con
            else:
                global _global_connection
                _global_connection = self.con

        self._create_schema()
        print(f"✅ Connected to DuckDB: {self.db_path}")

    def execute_safe(self, query, params=None, max_retries=3, retry_delay=0.1):
        """
        Thread-safe execute for WebSocket callbacks and concurrent access.

        Use this method when writing to the database from background threads
        (like WebSocket handlers) to prevent database corruption.

        Args:
            query: SQL query string
            params: Optional query parameters
            max_retries: Number of retries on lock contention
            retry_delay: Seconds to wait between retries

        Returns:
            Query result
        """
        import time

        for attempt in range(max_retries):
            with _db_lock:
                try:
                    if params:
                        return self.con.execute(query, params)
                    return self.con.execute(query)
                except Exception as e:
                    error_msg = str(e).lower()

                    # Retry on lock/constraint errors
                    if any(x in error_msg for x in ['lock', 'constraint', 'conflict', 'duplicate']):
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                            continue

                    # Don't retry on other errors
                    print(f"[DB ERROR] execute_safe failed: {e}")
                    if 'duplicate' in error_msg or 'constraint' in error_msg:
                        # Silently ignore duplicate key errors (INSERT OR IGNORE behavior)
                        return None
                    raise

        return None  # All retries exhausted

    def _create_schema(self):
        """Create all tables if they don't exist"""

        # Table 1: Instruments
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS instruments (
                instrument_key VARCHAR PRIMARY KEY,
                trading_symbol VARCHAR NOT NULL,
                name VARCHAR,
                instrument_type VARCHAR,
                exchange VARCHAR,
                segment VARCHAR NOT NULL,
                lot_size INTEGER,
                tick_size DOUBLE,
                expiry DATE,
                strike DOUBLE,
                option_type VARCHAR,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Table 2: OHLCV 1-minute (partitioned by date for fast queries)
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv_1m (
                instrument_key VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                open DOUBLE NOT NULL,
                high DOUBLE NOT NULL,
                low DOUBLE NOT NULL,
                close DOUBLE NOT NULL,
                volume BIGINT DEFAULT 0,
                oi BIGINT,
                PRIMARY KEY (instrument_key, timestamp)
            )
        """)

        # Table 3: Resampled OHLCV (5m, 15m, 30m, 60m, 1D)
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv_resampled (
                instrument_key VARCHAR NOT NULL,
                timeframe VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                open DOUBLE NOT NULL,
                high DOUBLE NOT NULL,
                low DOUBLE NOT NULL,
                close DOUBLE NOT NULL,
                volume BIGINT DEFAULT 0,
                PRIMARY KEY (instrument_key, timeframe, timestamp)
            )
        """)

        # Table 4: Trades (for backtests and live)
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id VARCHAR PRIMARY KEY,
                run_id VARCHAR,
                instrument_key VARCHAR NOT NULL,
                trade_type VARCHAR NOT NULL,
                direction VARCHAR NOT NULL,
                entry_time TIMESTAMP NOT NULL,
                entry_price DOUBLE NOT NULL,
                exit_time TIMESTAMP,
                exit_price DOUBLE,
                quantity INTEGER DEFAULT 1,
                exit_reason VARCHAR,
                pnl DOUBLE,
                pnl_pct DOUBLE,
                commission DOUBLE DEFAULT 0,
                notes VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Table 5: Backtest runs
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS backtest_runs (
                run_id VARCHAR PRIMARY KEY,
                strategy_name VARCHAR NOT NULL,
                instrument_key VARCHAR NOT NULL,
                timeframe VARCHAR NOT NULL,
                start_date DATE NOT NULL,
                end_date DATE NOT NULL,
                parameters VARCHAR,
                initial_capital DOUBLE,
                final_capital DOUBLE,
                total_trades INTEGER,
                win_rate DOUBLE,
                profit_factor DOUBLE,
                sharpe_ratio DOUBLE,
                max_drawdown_pct DOUBLE,
                metrics VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Table 6: Market status tracking
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS market_status (
                instrument_key VARCHAR PRIMARY KEY,
                last_1m_timestamp TIMESTAMP,
                last_5m_timestamp TIMESTAMP,
                last_1d_timestamp TIMESTAMP,
                last_checked TIMESTAMP,
                status VARCHAR
            )
        """)

        # Table 7: F&O Stocks Master
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS fo_stocks_master (
                instrument_key VARCHAR PRIMARY KEY,
                trading_symbol VARCHAR NOT NULL,
                name VARCHAR,
                lot_size INTEGER,
                is_active BOOLEAN DEFAULT TRUE,
                added_date DATE DEFAULT CURRENT_DATE,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Table 8: EHMA Universe (signal tracking)
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS ehma_universe (
                id INTEGER PRIMARY KEY,
                signal_date DATE NOT NULL,
                symbol VARCHAR NOT NULL,
                instrument_key VARCHAR,
                signal_type VARCHAR NOT NULL,
                signal_strength DOUBLE,
                bars_ago INTEGER,
                current_price DOUBLE,
                entry_price DOUBLE,
                stop_loss DOUBLE,
                target_price DOUBLE,
                rsi DOUBLE,
                trend VARCHAR,
                reasons VARCHAR,
                status VARCHAR DEFAULT 'ACTIVE',
                scan_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (signal_date, symbol, signal_type)
            )
        """)

        # Create indexes for common queries
        try:
            self.con.execute(
                "CREATE INDEX IF NOT EXISTS idx_ohlcv_1m_ts ON ohlcv_1m (timestamp)")
            self.con.execute(
                "CREATE INDEX IF NOT EXISTS idx_ohlcv_resampled_ts ON ohlcv_resampled (timestamp)")
            self.con.execute(
                "CREATE INDEX IF NOT EXISTS idx_trades_run ON trades (run_id)")
        except:
            pass

    # ========================================================================
    # INSTRUMENT MANAGEMENT
    # ========================================================================

    def upsert_instrument(self, instrument_key: str, trading_symbol: str,
                          segment: str, **kwargs) -> bool:
        """
        Insert or update an instrument.

        Args:
            instrument_key: Unique instrument identifier (e.g., "NSE_EQ|INE002A01018")
            trading_symbol: Trading symbol (e.g., "RELIANCE")
            segment: Market segment (e.g., "NSE_EQ")
            **kwargs: Additional fields (name, lot_size, tick_size, etc.)

        Returns:
            True if successful
        """
        # Build dynamic column list
        columns = ['instrument_key', 'trading_symbol', 'segment']
        values = [instrument_key, trading_symbol, segment]

        for key, val in kwargs.items():
            if val is not None:
                columns.append(key)
                values.append(val)

        placeholders = ', '.join(['?' for _ in values])
        col_str = ', '.join(columns)

        # Use INSERT OR REPLACE for upsert
        self.con.execute(f"""
            INSERT OR REPLACE INTO instruments ({col_str})
            VALUES ({placeholders})
        """, values)

        return True

    def get_instruments(self, segment: str = None,
                        active_only: bool = True) -> pd.DataFrame:
        """
        Get instruments filtered by segment and active status.

        Args:
            segment: Filter by segment (optional)
            active_only: Only return active instruments

        Returns:
            DataFrame with instrument details
        """
        query = "SELECT * FROM instruments WHERE 1=1"
        params = []

        if segment:
            query += " AND segment = ?"
            params.append(segment)

        if active_only:
            query += " AND is_active = TRUE"

        query += " ORDER BY trading_symbol"

        return self.con.execute(query, params).df()

    # ========================================================================
    # OHLCV DATA MANAGEMENT
    # ========================================================================

    def upsert_ohlcv_1m(self, instrument_key: str, df: pd.DataFrame) -> int:
        """
        Insert 1-minute OHLCV data.

        Args:
            instrument_key: Instrument identifier
            df: DataFrame with columns [timestamp, Open, High, Low, Close, Volume]
               (or lowercase versions)

        Returns:
            Number of rows inserted
        """
        if df.empty:
            return 0

        df = df.copy()

        # Standardize column names
        col_map = {
            'Timestamp': 'timestamp', 'Time': 'timestamp', 'Date': 'timestamp',
            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close',
            'Volume': 'volume', 'OI': 'oi', 'open_interest': 'oi'
        }
        df.rename(columns=col_map, inplace=True)

        # Ensure timestamp column exists
        if 'timestamp' not in df.columns and df.index.name in ['timestamp', 'Timestamp', None]:
            df = df.reset_index()
            if df.columns[0] != 'timestamp':
                df.rename(columns={df.columns[0]: 'timestamp'}, inplace=True)

        # Add instrument_key
        df['instrument_key'] = instrument_key

        # Select only needed columns
        cols = ['instrument_key', 'timestamp',
                'open', 'high', 'low', 'close', 'volume']
        if 'oi' in df.columns:
            cols.append('oi')
        df = df[[c for c in cols if c in df.columns]]

        # Insert with conflict handling
        initial_count = self.con.execute(
            "SELECT COUNT(*) FROM ohlcv_1m WHERE instrument_key = ?", [instrument_key]).fetchone()[0]

        self.con.execute("""
            INSERT OR IGNORE INTO ohlcv_1m 
            SELECT * FROM df
        """)

        final_count = self.con.execute(
            "SELECT COUNT(*) FROM ohlcv_1m WHERE instrument_key = ?", [instrument_key]).fetchone()[0]

        inserted = final_count - initial_count
        return inserted

    def get_ohlcv_1m(self, instrument_key: str,
                     start_date: datetime = None,
                     end_date: datetime = None) -> pd.DataFrame:
        """
        Get 1-minute OHLCV data for an instrument.

        Args:
            instrument_key: Instrument identifier
            start_date: Start datetime (optional)
            end_date: End datetime (optional)

        Returns:
            DataFrame with OHLCV data, indexed by timestamp
        """
        query = "SELECT * FROM ohlcv_1m WHERE instrument_key = ?"
        params = [instrument_key]

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)

        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)

        query += " ORDER BY timestamp"

        df = self.con.execute(query, params).df()

        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

        return df

    def insert_ohlcv_resampled(self, instrument_key: str, timeframe: str,
                               df: pd.DataFrame) -> int:
        """
        Insert resampled OHLCV data.

        Args:
            instrument_key: Instrument identifier
            timeframe: Timeframe string (e.g., '5minute', '15minute', '1day')
            df: DataFrame with OHLCV columns

        Returns:
            Number of rows inserted
        """
        if df.empty:
            return 0

        df = df.copy()

        # Standardize columns
        col_map = {
            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close',
            'Volume': 'volume'
        }
        df.rename(columns=col_map, inplace=True)

        # Handle index
        if df.index.name in ['timestamp', 'Timestamp', None]:
            df = df.reset_index()
            if df.columns[0] != 'timestamp':
                df.rename(columns={df.columns[0]: 'timestamp'}, inplace=True)

        df['instrument_key'] = instrument_key
        df['timeframe'] = timeframe

        cols = ['instrument_key', 'timeframe', 'timestamp',
                'open', 'high', 'low', 'close', 'volume']
        df = df[[c for c in cols if c in df.columns]]

        # Insert with conflict handling
        self.con.execute("""
            INSERT OR IGNORE INTO ohlcv_resampled
            SELECT * FROM df
        """)

        return len(df)

    def get_ohlcv_resampled(self, instrument_key: str, timeframe: str,
                            start_date: datetime = None,
                            end_date: datetime = None) -> pd.DataFrame:
        """
        Get resampled OHLCV data.

        Args:
            instrument_key: Instrument identifier
            timeframe: Timeframe string
            start_date: Start datetime (optional)
            end_date: End datetime (optional)

        Returns:
            DataFrame with OHLCV data, indexed by timestamp
        """
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv_resampled 
            WHERE instrument_key = ? AND timeframe = ?
        """
        params = [instrument_key, timeframe]

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)

        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)

        query += " ORDER BY timestamp"

        df = self.con.execute(query, params).df()

        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            # Rename to Title Case for compatibility
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

        return df

    # ========================================================================
    # RESAMPLING
    # ========================================================================

    def resample_and_store(self, instrument_key: str,
                           timeframes: List[str] = None) -> Dict[str, int]:
        """
        Resample 1m data to multiple timeframes and store.

        Args:
            instrument_key: Instrument to resample
            timeframes: List of target timeframes (default: ['5minute', '15minute', '60minute', '1day'])

        Returns:
            Dict with rows inserted per timeframe
        """
        if timeframes is None:
            timeframes = ['5minute', '15minute', '60minute', '1day']

        # Get all 1m data
        df_1m = self.get_ohlcv_1m(instrument_key)

        if df_1m.empty:
            return {tf: 0 for tf in timeframes}

        results = {}

        for tf in timeframes:
            rule = self._timeframe_to_rule(tf)
            if rule is None:
                continue

            # Resample
            df_resampled = df_1m.resample(rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

            if not df_resampled.empty:
                inserted = self.insert_ohlcv_resampled(
                    instrument_key, tf, df_resampled)
                results[tf] = inserted
            else:
                results[tf] = 0

        return results

    def _timeframe_to_rule(self, timeframe: str) -> str:
        """Convert timeframe string to pandas resample rule"""
        mapping = {
            '1minute': '1min',
            '5minute': '5min',
            '15minute': '15min',
            '30minute': '30min',
            '60minute': '60min',
            '1hour': '60min',
            '1day': '1D',
            'day': '1D'
        }
        return mapping.get(timeframe.lower())

    # ========================================================================
    # BACKTEST STORAGE
    # ========================================================================

    def save_backtest_results(self, strategy_name: str, instrument_key: str,
                              timeframe: str, start_date, end_date,
                              parameters: Dict, initial_capital: float,
                              trades_df: pd.DataFrame, metrics: Dict) -> str:
        """
        Save a complete backtest run with trades.

        Args:
            strategy_name: Name of strategy
            instrument_key: Instrument tested
            timeframe: Timeframe used
            start_date: Backtest start date
            end_date: Backtest end date
            parameters: Strategy parameters (dict)
            initial_capital: Starting capital
            trades_df: DataFrame with trade records
            metrics: Performance metrics (dict)

        Returns:
            run_id (UUID string)
        """
        run_id = str(uuid.uuid4())

        # Insert backtest metadata
        self.con.execute("""
            INSERT INTO backtest_runs (
                run_id, strategy_name, instrument_key, timeframe,
                start_date, end_date, parameters, initial_capital,
                final_capital, total_trades, win_rate, profit_factor,
                sharpe_ratio, max_drawdown_pct, metrics
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            run_id, strategy_name, instrument_key, timeframe,
            start_date, end_date, json.dumps(parameters), initial_capital,
            metrics.get('Final Capital'),
            metrics.get('Total Trades'),
            metrics.get('Win Rate %'),
            metrics.get('Profit Factor'),
            metrics.get('Sharpe Ratio'),
            metrics.get('Max Drawdown %'),
            json.dumps(metrics)
        ])

        # Insert trades
        if not trades_df.empty:
            trades_df = trades_df.copy()
            trades_df['trade_id'] = [str(uuid.uuid4())
                                     for _ in range(len(trades_df))]
            trades_df['run_id'] = run_id
            trades_df['instrument_key'] = instrument_key
            trades_df['trade_type'] = 'BACKTEST'

            # Rename columns to match schema
            col_map = {
                'Entry Time': 'entry_time',
                'Entry Price': 'entry_price',
                'Exit Time': 'exit_time',
                'Exit Price': 'exit_price',
                'Direction': 'direction',
                'Quantity': 'quantity',
                'PnL': 'pnl',
                'PnL %': 'pnl_pct',
                'Exit Reason': 'exit_reason',
                'Commission': 'commission'
            }
            trades_df.rename(columns=col_map, inplace=True)

            # Select only needed columns
            cols = ['trade_id', 'run_id', 'instrument_key', 'trade_type',
                    'direction', 'entry_time', 'entry_price', 'exit_time',
                    'exit_price', 'quantity', 'exit_reason', 'pnl', 'pnl_pct']
            trades_df = trades_df[[c for c in cols if c in trades_df.columns]]

            self.con.execute("""
                INSERT INTO trades 
                SELECT * FROM trades_df
            """)

        print(f"✅ Saved backtest {run_id[:8]}... with {len(trades_df)} trades")
        return run_id

    def get_backtest_history(self, instrument_key: Optional[str] = None,
                             strategy_name: Optional[str] = None,
                             limit: int = 50) -> pd.DataFrame:
        """
        Query backtest history.

        Args:
            instrument_key: Filter by instrument
            strategy_name: Filter by strategy
            limit: Max results

        Returns:
            DataFrame with backtest runs
        """
        query = "SELECT * FROM backtest_runs WHERE 1=1"
        params = []

        if instrument_key:
            query += " AND instrument_key = ?"
            params.append(instrument_key)

        if strategy_name:
            query += " AND strategy_name = ?"
            params.append(strategy_name)

        query += f" ORDER BY created_at DESC LIMIT {limit}"

        return self.con.execute(query, params).df()

    def get_trades(self, run_id: str) -> pd.DataFrame:
        """
        Get all trades for a specific backtest run.

        Args:
            run_id: Backtest run UUID

        Returns:
            DataFrame with trades
        """
        return self.con.execute("""
            SELECT * FROM trades
            WHERE run_id = ?
            ORDER BY entry_time
        """, [run_id]).df()

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def _update_market_status(self, instrument_key: str,
                              timeframe: str, last_timestamp):
        """Update market status tracking"""
        col_map = {
            '1m': 'last_1m_timestamp',
            '5minute': 'last_5m_timestamp',
            '1day': 'last_1d_timestamp'
        }

        if timeframe not in col_map:
            return

        col = col_map[timeframe]

        # Check if row exists
        existing = self.con.execute("""
            SELECT instrument_key FROM market_status
            WHERE instrument_key = ?
        """, [instrument_key]).fetchone()

        if existing:
            # Update existing row
            self.con.execute(f"""
                UPDATE market_status 
                SET {col} = ?, last_checked = CURRENT_TIMESTAMP
                WHERE instrument_key = ?
            """, [last_timestamp, instrument_key])
        else:
            # Insert new row
            self.con.execute(f"""
                INSERT INTO market_status (
                    instrument_key, {col}, last_checked
                ) VALUES (?, ?, CURRENT_TIMESTAMP)
            """, [instrument_key, last_timestamp])

    def get_data_status(self, instrument_key: str) -> Dict[str, Any]:
        """
        Get data availability status for an instrument.

        Returns:
            Dict with last timestamps for each timeframe
        """
        result = self.con.execute("""
            SELECT * FROM market_status
            WHERE instrument_key = ?
        """, [instrument_key]).fetchone()

        if result is None:
            return {'instrument_key': instrument_key, 'status': 'No data'}

        return dict(zip([desc[0] for desc in self.con.description], result))

    def vacuum(self):
        """Optimize database (run periodically)"""
        self.con.execute("VACUUM")
        print("✅ Database optimized")

    def close(self):
        """
        Close database connection.

        Note: With shared connection pattern, this only removes the reference.
        The actual connection may still be held by session_state.
        """
        # Don't actually close the shared connection - just remove reference
        self.con = None
        print("✅ Database reference released")

    def force_close(self):
        """
        Force close the shared database connection.
        Use this when you really need to release the database lock.
        """
        close_shared_connection()
        self.con = None
        print("✅ Database connection force closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Don't close shared connection on context exit
        pass


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_db() -> TradingDB:
    """
    Get a SINGLE TradingDB instance per Streamlit session.
    """
    session_state = _get_streamlit_session_state()

    if session_state is not None:
        if "trading_db_instance" not in session_state or session_state["trading_db_instance"] is None:
            session_state["trading_db_instance"] = TradingDB()
        return session_state["trading_db_instance"]

    # Non-Streamlit fallback (CLI, scripts)
    global _global_connection
    return TradingDB()


def db_exists() -> bool:
    """Check if the database file exists"""
    return _DEFAULT_DB_PATH.exists()


if __name__ == "__main__":
    # Test database creation
    db = TradingDB()
    print("✅ Database initialized successfully!")

    # Show tables
    tables = db.con.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'main'
    """).df()

    print(f"\n📊 Created tables:")
    for table in tables['table_name']:
        count = db.con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  - {table}: {count} rows")
