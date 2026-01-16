"""
DuckDB-Optimized Resampler (FIXED VERSION)
Replaces the old Parquet-based resampler with SQL-based resampling
10-50x faster due to columnar operations

FIXES:
1. SQL syntax error on line 191-192 (missing second ? in BETWEEN clause)
2. Better handling of 60minute timeframe (was using 1hour)
3. Improved skip logic diagnostics
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.database import get_db


class DuckDBResampler:
    """
    High-performance resampler using DuckDB's SQL engine.
    Much faster than Pandas due to columnar operations.
    """
    
    def __init__(self, db: get_db = None):
        """
        Initialize resampler.
        
        Args:
            db: TradingDB instance (creates new if None)
        """
        self.db = db if db else get_db()
    
    # ========================================================================
    # SINGLE SYMBOL RESAMPLING
    # ========================================================================
    
    def resample_symbol(self, instrument_key: str, timeframe: str,
                       incremental: bool = True, 
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> dict:
        """
        Resample 1m data to higher timeframe for a single symbol.
        
        Args:
            instrument_key: Instrument identifier
            timeframe: Target timeframe ('5minute', '15minute', '30minute', '60minute', '1hour', '1day')
            incremental: If True, only resample new data
            start_date: Force start from this date (overrides incremental)
            end_date: End date for resampling (default: far future)
        
        Returns:
            Dict with status info
        """
        start_time = datetime.now()
        
        # Map timeframe to SQL interval - FIXED: Added 60minute
        interval_map = {
            '5minute': '5 minutes',
            '15minute': '15 minutes',
            '30minute': '30 minutes',
            '60minute': '60 minutes',  # FIXED: was missing
            '1hour': '1 hour',
            '1day': '1 day'
        }
        
        # Normalize timeframe (60minute -> 1hour internally for some queries)
        normalized_tf = timeframe
        if timeframe == '60minute':
            normalized_tf = '60minute'  # Keep as-is for storage
        
        if timeframe not in interval_map:
            raise ValueError(f"Unsupported timeframe: {timeframe}. Supported: {list(interval_map.keys())}")
        
        interval = interval_map[timeframe]
        
        # Determine start point
        if start_date is None and incremental:
            # Get last timestamp in resampled table
            result = self.db.con.execute("""
                SELECT MAX(timestamp) as last_ts
                FROM ohlcv_resampled
                WHERE instrument_key = ? AND timeframe = ?
            """, [instrument_key, normalized_tf]).fetchone()
            
            if result[0] is not None:
                # Start from 1 period before last timestamp (for overlap correction)
                last_ts = pd.to_datetime(result[0])
                
                if timeframe == '1day':
                    start_date = (last_ts - timedelta(days=1)).strftime('%Y-%m-%d')
                elif timeframe in ('1hour', '60minute'):
                    start_date = (last_ts - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
                else:
                    minutes = int(timeframe.replace('minute', ''))
                    start_date = (last_ts - timedelta(minutes=minutes)).strftime('%Y-%m-%d %H:%M:%S')
                
                print(f"📅 Incremental resample from: {start_date}")
            else:
                # Full resample
                start_date = '1900-01-01'
                print(f"📅 Full resample (no existing data)")
        elif start_date is None:
            start_date = '1900-01-01'
            print(f"📅 Full resample")
        
        # Check if there's 1m data
        count_1m = self.db.con.execute("""
            SELECT COUNT(*) FROM ohlcv_1m
            WHERE instrument_key = ?
        """, [instrument_key]).fetchone()[0]
        
        if count_1m == 0:
            return {
                'status': 'no_data',
                'message': 'No 1-minute data available',
                'rows_processed': 0,
                'duration': 0
            }
        
        # --- Normalize date range to full timestamps (CRITICAL) ---
        if len(start_date) == 10:  # YYYY-MM-DD
            from_ts = f"{start_date} 00:00:00"
        else:
            from_ts = start_date

        # Handle end date
        if end_date:
            if len(end_date) == 10:
                to_ts = f"{end_date} 23:59:59"
            else:
                to_ts = end_date
        else:
            to_ts = "2100-01-01 23:59:59"

        # SQL-based resampling (SESSION-AWARE, SAFE)
        if timeframe == "1day":
            sql = """
            INSERT OR REPLACE INTO ohlcv_resampled
            SELECT
                instrument_key,
                '1day' AS timeframe,
                CAST(DATE(timestamp) AS TIMESTAMP) + INTERVAL '9 hours 15 minutes' AS timestamp,

                arg_min(open, timestamp)  AS open,
                MAX(high)                 AS high,
                MIN(low)                  AS low,
                arg_max(close, timestamp) AS close,
                SUM(volume)               AS volume,
                0 AS oi
            FROM ohlcv_1m
            WHERE instrument_key = ?
            AND timestamp BETWEEN ? AND ?
            GROUP BY instrument_key, DATE(timestamp)
            ORDER BY DATE(timestamp);
            """
        else:
            sql = f"""
            INSERT OR REPLACE INTO ohlcv_resampled
            SELECT
                instrument_key,
                '{normalized_tf}' AS timeframe,
                bucket_ts AS timestamp,

                arg_min(open, timestamp)  AS open,
                MAX(high)                 AS high,
                MIN(low)                  AS low,
                arg_max(close, timestamp) AS close,
                SUM(volume)               AS volume,
                0 AS oi
            FROM (
                SELECT
                    instrument_key,
                    time_bucket(
                        INTERVAL '{interval}',
                        timestamp,
                        TIMESTAMP '1970-01-01 09:15:00'
                    ) AS bucket_ts,
                    open, high, low, close, volume
                FROM ohlcv_1m
                WHERE instrument_key = ?
                AND timestamp BETWEEN ? AND ?
            ) t
            GROUP BY instrument_key, bucket_ts
            ORDER BY bucket_ts;
            """

        self.db.con.execute(sql, [instrument_key, from_ts, to_ts])
        
        # FIXED: Get row count - SQL syntax was broken (missing second ? in BETWEEN)
        rows_processed = self.db.con.execute("""
            SELECT COUNT(*) FROM ohlcv_resampled
            WHERE instrument_key = ? AND timeframe = ?
              AND timestamp BETWEEN ? AND ?
        """, [instrument_key, normalized_tf, from_ts, to_ts]).fetchone()[0]
        
        duration = (datetime.now() - start_time).total_seconds()
        
        return {
            'status': 'success',
            'message': f'Resampled to {timeframe}',
            'rows_processed': rows_processed,
            'duration': duration
        }
    
    # ========================================================================
    # BATCH RESAMPLING
    # ========================================================================
    
    def resample_all_symbols(self, timeframes: List[str] = None,
                            segment: Optional[str] = None,
                            incremental: bool = True) -> dict:
        """
        Resample all symbols to specified timeframes.
        
        Args:
            timeframes: List of timeframes (default: ['5minute', '15minute', '1day'])
            segment: Filter by segment (e.g., 'NSE_EQ')
            incremental: If True, only resample new data
        
        Returns:
            Dict with summary
        """
        if timeframes is None:
            timeframes = ['5minute', '15minute', '30minute', '60minute', '1day']
        
        print("\n" + "="*70)
        print("🔄 BATCH RESAMPLING")
        print("="*70)
        print(f"Timeframes: {', '.join(timeframes)}")
        print(f"Mode: {'Incremental' if incremental else 'Full'}")
        if segment:
            print(f"Segment: {segment}")
        print()
        
        start_time = datetime.now()
        
        # Get all instruments with 1m data
        query = """
            SELECT DISTINCT i.instrument_key, i.name, i.segment
            FROM instruments i
            JOIN ohlcv_1m o ON i.instrument_key = o.instrument_key
        """
        params = []
        
        if segment:
            query += " WHERE i.segment = ?"
            params.append(segment)
        
        query += " ORDER BY i.name"
        
        instruments = self.db.con.execute(query, params).df()
        
        print(f"📊 Found {len(instruments)} instruments with data\n")
        
        results = {
            'total_instruments': len(instruments),
            'total_timeframes': len(timeframes),
            'success': 0,
            'skipped': 0,
            'failed': 0,
            'total_rows': 0,
            'total_duration': 0,
            'details': []
        }
        
        for idx, row in instruments.iterrows():
            instrument_key = row['instrument_key']
            name = row['name']
            segment = row['segment']
            
            print(f"[{idx+1}/{len(instruments)}] {name} ({segment})")
            
            for timeframe in timeframes:
                try:
                    result = self.resample_symbol(
                        instrument_key=instrument_key,
                        timeframe=timeframe,
                        incremental=incremental
                    )
                    
                    if result['status'] == 'success':
                        results['success'] += 1
                        results['total_rows'] += result['rows_processed']
                        results['total_duration'] += result['duration']
                        
                        print(f"  ✅ {timeframe}: {result['rows_processed']} rows "
                              f"({result['duration']:.2f}s)")
                    elif result['status'] == 'no_data':
                        results['skipped'] += 1
                        print(f"  ⚠️  {timeframe}: {result['message']}")
                    else:
                        print(f"  ⚠️  {timeframe}: {result['message']}")
                    
                    results['details'].append({
                        'instrument_key': instrument_key,
                        'name': name,
                        'timeframe': timeframe,
                        'status': result['status'],
                        'rows': result['rows_processed'],
                        'duration': result['duration']
                    })
                    
                except Exception as e:
                    results['failed'] += 1
                    print(f"  ❌ {timeframe}: {str(e)}")
            
            print()
        
        total_duration = (datetime.now() - start_time).total_seconds()
        
        print("="*70)
        print("📊 RESAMPLING SUMMARY")
        print("="*70)
        print(f"✅ Successful: {results['success']}")
        print(f"⏭️  Skipped: {results['skipped']}")
        print(f"❌ Failed: {results['failed']}")
        print(f"📈 Total rows processed: {results['total_rows']:,}")
        print(f"⏱️  Total duration: {total_duration:.2f}s")
        if total_duration > 0:
            print(f"🚀 Average speed: {results['total_rows']/total_duration:.0f} rows/sec")
        
        results['total_duration'] = total_duration
        
        return results
    
    # ========================================================================
    # VALIDATION
    # ========================================================================
    
    def validate_resampling(self, instrument_key: str, 
                           timeframe: str, sample_date: str) -> dict:
        """
        Validate resampled data against 1m source data.
        
        Args:
            instrument_key: Instrument identifier
            timeframe: Timeframe to validate
            sample_date: Date to check (YYYY-MM-DD)
        
        Returns:
            Dict with validation results
        """
        # Get 1m data for the date
        df_1m = self.db.con.execute("""
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv_1m
            WHERE instrument_key = ?
              AND DATE(timestamp) = ?
            ORDER BY timestamp
        """, [instrument_key, sample_date]).df()
        
        if df_1m.empty:
            return {'status': 'no_data', 'message': 'No 1m data for this date'}
        
        # Get resampled data for the date
        df_resampled = self.db.con.execute("""
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv_resampled
            WHERE instrument_key = ?
              AND timeframe = ?
              AND DATE(timestamp) = ?
            ORDER BY timestamp
        """, [instrument_key, timeframe, sample_date]).df()
        
        if df_resampled.empty:
            return {'status': 'no_resampled_data', 'message': 'No resampled data for this date'}
        
        # Manual resample using Pandas for comparison
        df_1m['timestamp'] = pd.to_datetime(df_1m['timestamp'])
        df_1m.set_index('timestamp', inplace=True)
        
        freq_map = {
            '5minute': '5min',
            '15minute': '15min',
            '30minute': '30min',
            '60minute': '60min',
            '1hour': '1h',
            '1day': '1D'
        }
        
        freq = freq_map.get(timeframe, '5min')
        
        df_manual = df_1m.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        df_manual.reset_index(inplace=True)
        
        # Compare
        differences = []
        for idx, row_manual in df_manual.iterrows():
            ts = row_manual['timestamp']
            
            row_db = df_resampled[df_resampled['timestamp'] == ts]
            
            if row_db.empty:
                differences.append(f"Missing timestamp: {ts}")
                continue
            
            row_db = row_db.iloc[0]
            
            # Check each field
            for col in ['open', 'high', 'low', 'close', 'volume']:
                val_manual = row_manual[col]
                val_db = row_db[col]
                
                if abs(val_manual - val_db) > 0.01:  # Allow 0.01 difference for rounding
                    differences.append(f"{ts} - {col}: Manual={val_manual}, DB={val_db}")
        
        if differences:
            return {
                'status': 'mismatch',
                'message': f'Found {len(differences)} differences',
                'differences': differences[:10]  # First 10
            }
        else:
            return {
                'status': 'valid',
                'message': 'Resampled data matches manual calculation',
                'candles_checked': len(df_manual)
            }
    
    # ========================================================================
    # DIAGNOSTICS (NEW)
    # ========================================================================
    
    def diagnose_resampling_issue(self, trading_symbol: str = None) -> dict:
        """
        Diagnose why resampling might be failing or skipping.
        
        Args:
            trading_symbol: Optional symbol to focus on
            
        Returns:
            Dict with diagnostic info
        """
        results = {}
        
        # 1. Check 1m data availability
        if trading_symbol:
            count_1m = self.db.con.execute("""
                SELECT COUNT(*) FROM ohlcv_1m o
                JOIN instruments i ON o.instrument_key = i.instrument_key
                WHERE i.trading_symbol = ?
            """, [trading_symbol]).fetchone()[0]
            results['1m_candles_for_symbol'] = count_1m
            
            # Get date range
            date_range = self.db.con.execute("""
                SELECT MIN(timestamp) as first, MAX(timestamp) as last
                FROM ohlcv_1m o
                JOIN instruments i ON o.instrument_key = i.instrument_key
                WHERE i.trading_symbol = ?
            """, [trading_symbol]).fetchone()
            results['1m_date_range'] = {'first': str(date_range[0]), 'last': str(date_range[1])}
        else:
            count_1m = self.db.con.execute("SELECT COUNT(*) FROM ohlcv_1m").fetchone()[0]
            results['total_1m_candles'] = count_1m
        
        # 2. Check resampled data
        resampled_stats = self.db.con.execute("""
            SELECT timeframe, COUNT(*) as count
            FROM ohlcv_resampled
            GROUP BY timeframe
        """).df()
        results['resampled_stats'] = resampled_stats.to_dict('records') if not resampled_stats.empty else []
        
        # 3. Check instruments
        instruments_count = self.db.con.execute("""
            SELECT COUNT(DISTINCT instrument_key) FROM ohlcv_1m
        """).fetchone()[0]
        results['instruments_with_1m_data'] = instruments_count
        
        # 4. Check ohlcv_resampled table structure
        try:
            table_info = self.db.con.execute("DESCRIBE ohlcv_resampled").df()
            results['resampled_table_columns'] = table_info['column_name'].tolist()
        except:
            results['resampled_table_exists'] = False
        
        return results
    
    # ========================================================================
    # UTILITIES
    # ========================================================================
    
    def get_resampling_status(self) -> pd.DataFrame:
        """
        Get overview of resampling status for all instruments.
        
        Returns:
            DataFrame with status per instrument/timeframe
        """
        return self.db.con.execute("""
            SELECT 
                i.trading_symbol as symbol,
                i.segment,
                r.timeframe,
                COUNT(r.timestamp) as candle_count,
                MIN(DATE(r.timestamp)) as first_date,
                MAX(DATE(r.timestamp)) as last_date
            FROM ohlcv_resampled r
            JOIN instruments i ON r.instrument_key = i.instrument_key
            GROUP BY i.trading_symbol, i.segment, r.timeframe
            ORDER BY i.trading_symbol, r.timeframe
        """).df()
    
    def delete_resampled_data(self, instrument_key: str = None,
                             timeframe: str = None):
        """
        Delete resampled data (for re-processing).
        
        Args:
            instrument_key: Delete for specific instrument (None = all)
            timeframe: Delete specific timeframe (None = all)
        """
        where = []
        params = []
        
        if instrument_key:
            where.append("instrument_key = ?")
            params.append(instrument_key)
        
        if timeframe:
            where.append("timeframe = ?")
            params.append(timeframe)
        
        where_clause = " AND ".join(where) if where else "1=1"
        
        # First count
        count_query = f"SELECT COUNT(*) FROM ohlcv_resampled WHERE {where_clause}"
        rows_to_delete = self.db.con.execute(count_query, params).fetchone()[0]
        
        # Then delete
        query = f"DELETE FROM ohlcv_resampled WHERE {where_clause}"
        self.db.con.execute(query, params)
        
        print(f"✅ Deleted {rows_to_delete} rows")
        return rows_to_delete


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def resample_incremental(instrument_key: str, timeframe: str) -> dict:
    """
    Quick incremental resample for a single symbol.
    
    Args:
        instrument_key: Instrument identifier
        timeframe: Target timeframe
    
    Returns:
        Status dict
    """
    resampler = DuckDBResampler()
    return resampler.resample_symbol(instrument_key, timeframe, incremental=True)


def resample_all(timeframes: List[str] = None, segment: str = None) -> dict:
    """
    Resample all symbols to specified timeframes.
    
    Args:
        timeframes: List of timeframes
        segment: Filter by segment
    
    Returns:
        Summary dict
    """
    resampler = DuckDBResampler()
    return resampler.resample_all_symbols(timeframes, segment, incremental=True)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DuckDB Resampler")
    parser.add_argument("--timeframes", nargs="+", 
                       default=['5minute', '15minute', '1day'],
                       help="Timeframes to resample")
    parser.add_argument("--segment", help="Filter by segment")
    parser.add_argument("--full", action="store_true", 
                       help="Full resample (not incremental)")
    parser.add_argument("--status", action="store_true",
                       help="Show resampling status")
    parser.add_argument("--diagnose", action="store_true",
                       help="Run diagnostics")
    parser.add_argument("--symbol", help="Symbol for diagnostics")
    
    args = parser.parse_args()
    
    resampler = DuckDBResampler()
    
    if args.diagnose:
        print("\n🔍 RUNNING DIAGNOSTICS")
        print("="*50)
        diag = resampler.diagnose_resampling_issue(args.symbol)
        for key, value in diag.items():
            print(f"{key}: {value}")
    elif args.status:
        print("\n📊 Current Resampling Status:")
        print(resampler.get_resampling_status().to_string(index=False))
    else:
        results = resampler.resample_all_symbols(
            timeframes=args.timeframes,
            segment=args.segment,
            incremental=not args.full
        )