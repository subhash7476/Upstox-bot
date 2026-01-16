"""
Data Manager (DuckDB Version)
Fetches historical data from Upstox and stores in DuckDB
Replaces the old Parquet-based data_manager.py
"""

from core.api.upstox_client import UpstoxClient
from core.database import get_db
import sys
import os
from pathlib import Path
from datetime import datetime, date, timedelta
import pandas as pd
from typing import Optional, List
from dateutil.relativedelta import relativedelta

# Add project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# from core.api.historical import fetch_historical_data


class DataManager:
    """
    Manages historical data fetching and storage using DuckDB.
    Handles incremental updates and missing data detection.
    """

    def __init__(self, db: get_db = None):
        """
        Initialize data manager.

        Args:
            db: TradingDB instance (creates new if None)
        """
        self.db = db if db else get_db()
        self.client = None  # Lazy initialization

    def _get_client(self) -> UpstoxClient:
        """Get Upstox client (lazy initialization)"""
        if self.client is None:
            self.client = UpstoxClient()
        return self.client

    # ========================================================================
    # DATA FETCHING
    # ========================================================================

    def fetch_and_store(self, instrument_key: str, from_date: date,
                        to_date: date, interval: str = "1minute",
                        force: bool = False) -> dict:
        """
        Fetch historical data from Upstox and store in DuckDB.
        Only fetches missing data unless force=True.
        Automatically chunks requests based on Upstox API limits.

        Args:
            instrument_key: Instrument identifier (e.g., 'NSE_EQ|INE002A01018')
            from_date: Start date
            to_date: End date
            interval: Data interval ('1minute', '5minute', etc.)
            force: If True, fetch even if data exists

        Returns:
            Dict with status info
        """
        # Check what data already exists
        existing_data = self.get_date_coverage(instrument_key, interval)

        if not force and existing_data['has_data']:
            # Find gaps
            gaps = self._find_missing_dates(
                existing_data['first_date'],
                existing_data['last_date'],
                from_date,
                to_date
            )

            if not gaps:
                return {
                    'status': 'up_to_date',
                    'message': 'No new data to fetch',
                    'rows_added': 0
                }

            print(f"📅 Found {len(gaps)} date gaps to fill")
        else:
            # Fetch entire range
            gaps = [(from_date, to_date)]

        # Chunk gaps based on API limits
        all_chunks = []
        for gap_start, gap_end in gaps:
            chunks = self._chunk_date_range(gap_start, gap_end, interval)
            all_chunks.extend(chunks)

        print(
            f"📦 Split into {len(all_chunks)} API chunks (based on Upstox limits)")

        # Fetch data for each chunk
        total_rows = 0
        for chunk_idx, (chunk_start, chunk_end) in enumerate(all_chunks, 1):
            print(
                f"\n📥 Chunk {chunk_idx}/{len(all_chunks)}: {chunk_start} to {chunk_end}")

            try:
                print(f"   Fetching {instrument_key}...")

                # Get Upstox client
                client = self._get_client()

                # Map interval to Upstox API format
                interval_map = {
                    '1minute': ('minutes', 1),
                    '5minute': ('minutes', 5),
                    '15minute': ('minutes', 15),
                    '30minute': ('minutes', 30),
                    '1hour': ('hours', 1),
                    '1day': ('days', 1)
                }

                if interval not in interval_map:
                    print(f"⚠️  Unsupported interval: {interval}")
                    continue

                timeframe, interval_num = interval_map[interval]

                # Call Upstox API using fetch_ohlc method
                response = client.fetch_ohlc(
                    instrument_key=instrument_key,
                    timeframe=timeframe,
                    interval_num=interval_num,
                    from_date=chunk_start,
                    to_date=chunk_end
                )

                # Parse response
                if not response or 'data' not in response:
                    print(f"⚠️  No data in API response")

                    # Check if it's today and market might not be open
                    from datetime import datetime
                    if chunk_end == date.today():
                        current_time = datetime.now().time()
                        market_open = datetime.strptime(
                            "09:15", "%H:%M").time()
                        market_close = datetime.strptime(
                            "15:30", "%H:%M").time()

                        if current_time < market_open:
                            print(
                                f"   ℹ️  Market hasn't opened yet (opens at 9:15 AM)")
                        elif current_time < market_close:
                            print(
                                f"   ℹ️  Market is currently open, data will be available after market close")
                        else:
                            print(
                                f"   ℹ️  Market closed. Data should be available soon (might take 15-30 min after close)")

                    continue

                candles = response['data'].get('candles', [])

                if not candles:
                    print(f"⚠️  No candles returned")
                    continue

                # Convert to DataFrame
                # Upstox V3 format: [timestamp, open, high, low, close, volume, oi]
                df = pd.DataFrame(candles, columns=[
                                  'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI'])

                # Convert timestamp from string to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'])

                print(f"   Received {len(df)} candles from API")

                # Store in database
                if interval == "1minute":
                    self.db.upsert_ohlcv_1m(instrument_key, df)
                else:
                    # For other intervals, store in resampled table
                    df_copy = df.copy()
                    df_copy['instrument_key'] = instrument_key
                    df_copy['timeframe'] = interval
                    df_copy.columns = df_copy.columns.str.lower()

                    self.db.con.execute("""
                        INSERT OR REPLACE INTO ohlcv_resampled 
                        SELECT * FROM df_copy
                    """)

                total_rows += len(df)
                print(f"✅ Stored {len(df)} candles in DuckDB")

                # Small delay between API calls to avoid rate limiting
                import time
                time.sleep(0.3)

            except Exception as e:
                print(f"❌ Error fetching chunk: {e}")
                import traceback
                traceback.print_exc()

        return {
            'status': 'success',
            'message': f'Added {total_rows} new candles',
            'rows_added': total_rows
        }

    def _chunk_date_range(self, start_date: date, end_date: date, interval: str) -> list:
        """
        Split date range into chunks based on Upstox API limits.

        Limits:
        - 1-15 minute intervals: 1 month max
        - 30+ minute intervals (up to 15min): 1 quarter max
        - Hours: 1 quarter max
        - Days: 1 decade max

        Returns:
            List of (start_date, end_date) tuples
        """
        from datetime import timedelta
        from dateutil.relativedelta import relativedelta

        # Determine chunk size based on interval
        if interval in ['1minute', '5minute', '15minute']:
            # 1 month chunks
            chunk_delta = relativedelta(months=1)
        elif interval in ['30minute', '1hour']:
            # 3 month (1 quarter) chunks
            chunk_delta = relativedelta(months=3)
        elif interval == '1day':
            # 10 year (1 decade) chunks
            chunk_delta = relativedelta(years=10)
        else:
            # Default: 1 month
            chunk_delta = relativedelta(months=1)

        chunks = []
        current_start = start_date

        while current_start <= end_date:
            # Calculate chunk end (don't exceed final end_date)
            current_end = min(current_start + chunk_delta -
                              timedelta(days=1), end_date)
            chunks.append((current_start, current_end))

            # Move to next chunk
            current_start = current_end + timedelta(days=1)

        return chunks

    def fetch_multiple_symbols(self, symbols: List[str], segment: str,
                               from_date: date, to_date: date,
                               interval: str = "1minute") -> dict:
        """
        Fetch data for multiple symbols.

        Args:
            symbols: List of symbol names (e.g., ['RELIANCE', 'TCS'])
            segment: Segment (e.g., 'NSE_EQ')
            from_date: Start date
            to_date: End date
            interval: Data interval

        Returns:
            Dict with summary
        """
        results = {
            'success': [],
            'failed': [],
            'total_rows': 0
        }

        for symbol in symbols:
            # Get instrument key
            instruments = self.db.get_instruments(name=symbol, segment=segment)

            if instruments.empty:
                print(f"⚠️  Instrument not found: {symbol}")
                results['failed'].append(symbol)
                continue

            instrument_key = instruments.iloc[0]['instrument_key']

            print(f"\n{'='*60}")
            print(f"📊 Processing: {symbol}")
            print(f"{'='*60}")

            try:
                result = self.fetch_and_store(
                    instrument_key=instrument_key,
                    from_date=from_date,
                    to_date=to_date,
                    interval=interval
                )

                results['success'].append(symbol)
                results['total_rows'] += result['rows_added']

            except Exception as e:
                print(f"❌ Failed: {e}")
                results['failed'].append(symbol)

        return results

    # ========================================================================
    # DATA QUERIES
    # ========================================================================

    def get_date_coverage(self, instrument_key: str,
                          interval: str = "1minute") -> dict:
        """
        Check what dates are available for an instrument.

        Args:
            instrument_key: Instrument identifier
            interval: Data interval

        Returns:
            Dict with coverage info
        """
        if interval == "1minute":
            table = "ohlcv_1m"
            query = """
                SELECT 
                    MIN(DATE(timestamp)) as first_date,
                    MAX(DATE(timestamp)) as last_date,
                    COUNT(*) as total_rows
                FROM ohlcv_1m
                WHERE instrument_key = ?
            """
        else:
            table = "ohlcv_resampled"
            query = """
                SELECT 
                    MIN(DATE(timestamp)) as first_date,
                    MAX(DATE(timestamp)) as last_date,
                    COUNT(*) as total_rows
                FROM ohlcv_resampled
                WHERE instrument_key = ? AND timeframe = ?
            """

        params = [instrument_key] if interval == "1minute" else [
            instrument_key, interval]
        result = self.db.con.execute(query, params).fetchone()

        if result[0] is None:
            return {
                'has_data': False,
                'first_date': None,
                'last_date': None,
                'total_rows': 0
            }

        return {
            'has_data': True,
            'first_date': result[0],
            'last_date': result[1],
            'total_rows': result[2]
        }

    def get_missing_dates(self, instrument_key: str,
                          from_date: date, to_date: date,
                          interval: str = "1minute") -> List[date]:
        """
        Find dates with no data in a given range.

        Args:
            instrument_key: Instrument identifier
            from_date: Start date
            to_date: End date
            interval: Data interval

        Returns:
            List of missing dates
        """
        if interval == "1minute":
            table = "ohlcv_1m"
            where = "instrument_key = ?"
            params = [instrument_key]
        else:
            table = "ohlcv_resampled"
            where = "instrument_key = ? AND timeframe = ?"
            params = [instrument_key, interval]

        # Get all dates with data
        query = f"""
            SELECT DISTINCT DATE(timestamp) as date
            FROM {table}
            WHERE {where}
              AND DATE(timestamp) BETWEEN ? AND ?
            ORDER BY date
        """
        params.extend([from_date, to_date])

        result = self.db.con.execute(query, params).df()

        if result.empty:
            # All dates are missing
            return pd.date_range(from_date, to_date, freq='D').date.tolist()

        # Find gaps
        existing_dates = set(pd.to_datetime(result['date']).dt.date)
        all_dates = set(pd.date_range(from_date, to_date, freq='D').date)
        missing_dates = sorted(all_dates - existing_dates)

        return missing_dates

    def _find_missing_dates(self, existing_first: date, existing_last: date,
                            requested_first: date, requested_last: date) -> List[tuple]:
        """
        Find date gaps between existing and requested ranges.

        Returns:
            List of (start_date, end_date) tuples representing gaps
        """
        gaps = []

        # Gap before existing data
        if requested_first < existing_first:
            gaps.append((requested_first, existing_first - timedelta(days=1)))

        # Gap after existing data
        if requested_last > existing_last:
            gaps.append((existing_last + timedelta(days=1), requested_last))

        return gaps

    # ========================================================================
    # DATA OPERATIONS
    # ========================================================================

    def resample_to_timeframe(self, instrument_key: str, timeframe: str,
                              start_date: Optional[str] = None):
        """
        Resample 1m data to higher timeframe.
        Wrapper around database method.

        Args:
            instrument_key: Instrument identifier
            timeframe: Target timeframe
            start_date: Only resample from this date (incremental)
        """
        self.db.resample_to_timeframe(instrument_key, timeframe, start_date)

    def delete_data(self, instrument_key: str, interval: str = "1minute",
                    from_date: Optional[date] = None,
                    to_date: Optional[date] = None):
        """
        Delete data for an instrument (or date range).

        Args:
            instrument_key: Instrument identifier
            interval: Data interval
            from_date: Start date (None = all)
            to_date: End date (None = all)
        """
        if interval == "1minute":
            table = "ohlcv_1m"
            where = "instrument_key = ?"
            params = [instrument_key]
        else:
            table = "ohlcv_resampled"
            where = "instrument_key = ? AND timeframe = ?"
            params = [instrument_key, interval]

        if from_date and to_date:
            where += " AND timestamp BETWEEN ? AND ?"
            params.extend([from_date, to_date])

        query = f"DELETE FROM {table} WHERE {where}"

        rows_deleted = self.db.con.execute(query, params).fetchone()[0]
        print(f"✅ Deleted {rows_deleted} rows")

    def get_data_summary(self) -> pd.DataFrame:
        """
        Get summary of all stored data.

        Returns:
            DataFrame with instrument-level statistics
        """
        return self.db.con.execute("""
            SELECT 
                i.name as symbol,
                i.segment,
                COUNT(DISTINCT DATE(o.timestamp)) as days_available,
                MIN(DATE(o.timestamp)) as first_date,
                MAX(DATE(o.timestamp)) as last_date,
                COUNT(*) as total_candles
            FROM ohlcv_1m o
            JOIN instruments i ON o.instrument_key = i.instrument_key
            GROUP BY i.name, i.segment
            ORDER BY i.name
        """).df()


# ============================================================================
# CONVENIENCE FUNCTIONS (backward compatibility)
# ============================================================================

def fetch_historical_range(symbol: str, segment: str, interval: str,
                           from_date: date, to_date: date,
                           force: bool = False) -> dict:
    """
    Legacy function signature for backward compatibility.

    Args:
        symbol: Symbol name (e.g., 'RELIANCE')
        segment: Segment (e.g., 'NSE_EQ')
        interval: Interval (e.g., '1minute')
        from_date: Start date
        to_date: End date
        force: Force re-fetch

    Returns:
        Status dict
    """
    dm = DataManager()

    # Get instrument key
    instruments = dm.db.get_instruments(name=symbol, segment=segment)
    if instruments.empty:
        raise ValueError(f"Instrument not found: {symbol} ({segment})")

    instrument_key = instruments.iloc[0]['instrument_key']

    return dm.fetch_and_store(
        instrument_key=instrument_key,
        from_date=from_date,
        to_date=to_date,
        interval=interval,
        force=force
    )


if __name__ == "__main__":
    # Test data manager
    dm = DataManager()

    print("\n📊 Current data summary:")
    summary = dm.get_data_summary()
    if not summary.empty:
        print(summary.to_string(index=False))
    else:
        print("No data in database yet.")
