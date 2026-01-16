# Trading Bot Pro - SQL Query Collection
# DuckDB Queries for Data Management, Resampling, and Diagnostics
# ==============================================================================

# ==============================================================================
# 1. DIAGNOSTIC QUERIES
# ==============================================================================

# -----------------------------------------------------------------------------
# 1.1 Check 1-minute data availability
# -----------------------------------------------------------------------------

SELECT COUNT(*) as total_candles,
MIN(timestamp) as first_date,
MAX(timestamp) as last_date,
COUNT(DISTINCT instrument_key) as unique_instruments
FROM ohlcv_1m

# -----------------------------------------------------------------------------
# 1.2 Check 1-minute data for a specific symbol
# -----------------------------------------------------------------------------
SELECT
COUNT(*) as total,
MIN(timestamp) as first,
MAX(timestamp) as last,
COUNT(DISTINCT DATE(timestamp)) as trading_days
FROM ohlcv_1m o
JOIN instruments i ON o.instrument_key = i.instrument_key
WHERE i.trading_symbol = 'RELIANCE'

# -----------------------------------------------------------------------------
# 1.3 Check resampled data summary by timeframe
# -----------------------------------------------------------------------------
SELECT
timeframe,
COUNT(DISTINCT instrument_key) as symbols_count,
SUM(candle_count) as total_candles
FROM(
    SELECT instrument_key, timeframe, COUNT(*) as candle_count
    FROM ohlcv_resampled
    GROUP BY instrument_key, timeframe
)
GROUP BY timeframe
ORDER BY timeframe

# -----------------------------------------------------------------------------
# 1.4 Check resampled data for specific symbols
# -----------------------------------------------------------------------------
SELECT i.trading_symbol, r.timeframe, COUNT(*) as candles
FROM ohlcv_resampled r
JOIN instruments i ON r.instrument_key = i.instrument_key
WHERE i.trading_symbol IN('360ONE', 'ABB', 'ABCAPITAL', 'ADANIENT', 'ADANIENSOL', 'RELIANCE')
GROUP BY i.trading_symbol, r.timeframe
ORDER BY i.trading_symbol, r.timeframe

# -----------------------------------------------------------------------------
# 1.5 Find symbols with 1m data but NO resampled data
# -----------------------------------------------------------------------------
SELECT DISTINCT i.trading_symbol
FROM ohlcv_1m o
JOIN instruments i ON o.instrument_key = i.instrument_key
WHERE i.instrument_key NOT IN(
    SELECT DISTINCT instrument_key FROM ohlcv_resampled
)
ORDER BY i.trading_symbol

# -----------------------------------------------------------------------------
# 1.6 Find symbols missing a specific timeframe (e.g., 1day)
# -----------------------------------------------------------------------------
SELECT DISTINCT i.trading_symbol
FROM ohlcv_resampled r
JOIN instruments i ON r.instrument_key = i.instrument_key
WHERE i.instrument_key NOT IN(
    SELECT instrument_key FROM ohlcv_resampled WHERE timeframe='1day'
)
LIMIT 20

# -----------------------------------------------------------------------------
# 1.7 Check for bad/NULL data in 1m table
# -----------------------------------------------------------------------------
SELECT i.trading_symbol, COUNT(*) as bad_rows
FROM ohlcv_1m o
JOIN instruments i ON o.instrument_key = i.instrument_key
WHERE open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL
OR volume IS NULL OR volume < 0
OR high < low OR open > high OR open < low OR close > high OR close < low
GROUP BY i.trading_symbol
ORDER BY bad_rows DESC

# -----------------------------------------------------------------------------
# 1.8 Check database table sizes
# -----------------------------------------------------------------------------
SELECT
'instruments' as table_name, COUNT(*) as row_count FROM instruments
UNION ALL
SELECT 'ohlcv_1m', COUNT(*) FROM ohlcv_1m
UNION ALL
SELECT 'ohlcv_resampled', COUNT(*) FROM ohlcv_resampled
UNION ALL
SELECT 'backtest_runs', COUNT(*) FROM backtest_runs
UNION ALL
SELECT 'trades', COUNT(*) FROM trades

# -----------------------------------------------------------------------------
# 1.9 Check data coverage per symbol (days with data)
# -----------------------------------------------------------------------------
SELECT
i.trading_symbol,
i.name,
COUNT(DISTINCT DATE(o.timestamp)) as days_with_data,
MIN(DATE(o.timestamp)) as first_date,
MAX(DATE(o.timestamp)) as last_date
FROM ohlcv_1m o
JOIN instruments i ON o.instrument_key = i.instrument_key
WHERE i.segment = 'NSE_EQ'
GROUP BY i.trading_symbol, i.name
ORDER BY days_with_data DESC


# ==============================================================================
# 2. RESAMPLING QUERIES - SINGLE SYMBOL
# ==============================================================================

# -----------------------------------------------------------------------------
# 2.1 Resample single symbol to 5minute
# -----------------------------------------------------------------------------
INSERT OR REPLACE INTO ohlcv_resampled
SELECT
o.instrument_key,
'5minute' AS timeframe,
time_bucket(INTERVAL '5 minutes', timestamp, TIMESTAMP '1970-01-01 09:15:00') AS timestamp,
FIRST(open ORDER BY timestamp) AS open,
MAX(high) AS high,
MIN(low) AS low,
LAST(close ORDER BY timestamp) AS close,
SUM(volume) AS volume,
0 AS oi
FROM ohlcv_1m o
JOIN instruments i ON o.instrument_key = i.instrument_key
WHERE i.trading_symbol = 'RELIANCE'
GROUP BY o.instrument_key, time_bucket(INTERVAL '5 minutes', timestamp, TIMESTAMP '1970-01-01 09:15:00')

# -----------------------------------------------------------------------------
# 2.2 Resample single symbol to 15minute
# -----------------------------------------------------------------------------
INSERT OR REPLACE INTO ohlcv_resampled
SELECT
o.instrument_key,
'15minute' AS timeframe,
time_bucket(INTERVAL '15 minutes', timestamp, TIMESTAMP '1970-01-01 09:15:00') AS timestamp,
FIRST(open ORDER BY timestamp) AS open,
MAX(high) AS high,
MIN(low) AS low,
LAST(close ORDER BY timestamp) AS close,
SUM(volume) AS volume,
0 AS oi
FROM ohlcv_1m o
JOIN instruments i ON o.instrument_key = i.instrument_key
WHERE i.trading_symbol = 'RELIANCE'
GROUP BY o.instrument_key, time_bucket(INTERVAL '15 minutes', timestamp, TIMESTAMP '1970-01-01 09:15:00')

# -----------------------------------------------------------------------------
# 2.3 Resample single symbol to 30minute
# -----------------------------------------------------------------------------
INSERT OR REPLACE INTO ohlcv_resampled
SELECT
o.instrument_key,
'30minute' AS timeframe,
time_bucket(INTERVAL '30 minutes', timestamp, TIMESTAMP '1970-01-01 09:15:00') AS timestamp,
FIRST(open ORDER BY timestamp) AS open,
MAX(high) AS high,
MIN(low) AS low,
LAST(close ORDER BY timestamp) AS close,
SUM(volume) AS volume,
0 AS oi
FROM ohlcv_1m o
JOIN instruments i ON o.instrument_key = i.instrument_key
WHERE i.trading_symbol = 'RELIANCE'
GROUP BY o.instrument_key, time_bucket(INTERVAL '30 minutes', timestamp, TIMESTAMP '1970-01-01 09:15:00')

# -----------------------------------------------------------------------------
# 2.4 Resample single symbol to 60minute
# -----------------------------------------------------------------------------
INSERT OR REPLACE INTO ohlcv_resampled
SELECT
o.instrument_key,
'60minute' AS timeframe,
time_bucket(INTERVAL '60 minutes', timestamp, TIMESTAMP '1970-01-01 09:15:00') AS timestamp,
FIRST(open ORDER BY timestamp) AS open,
MAX(high) AS high,
MIN(low) AS low,
LAST(close ORDER BY timestamp) AS close,
SUM(volume) AS volume,
0 AS oi
FROM ohlcv_1m o
JOIN instruments i ON o.instrument_key = i.instrument_key
WHERE i.trading_symbol = 'RELIANCE'
GROUP BY o.instrument_key, time_bucket(INTERVAL '60 minutes', timestamp, TIMESTAMP '1970-01-01 09:15:00')

# -----------------------------------------------------------------------------
# 2.5 Resample single symbol to 1day
# -----------------------------------------------------------------------------
INSERT OR REPLACE INTO ohlcv_resampled
SELECT
o.instrument_key,
'1day' AS timeframe,
CAST(DATE(timestamp) AS TIMESTAMP) + INTERVAL '9 hours 15 minutes' AS timestamp,
FIRST(open ORDER BY timestamp) AS open,
MAX(high) AS high,
MIN(low) AS low,
LAST(close ORDER BY timestamp) AS close,
SUM(volume) AS volume,
0 AS oi
FROM ohlcv_1m o
JOIN instruments i ON o.instrument_key = i.instrument_key
WHERE i.trading_symbol = 'RELIANCE'
GROUP BY o.instrument_key, DATE(timestamp)
ORDER BY DATE(timestamp)


# ==============================================================================
# 3. RESAMPLING QUERIES - ALL SYMBOLS (BATCH)
# ==============================================================================

# -----------------------------------------------------------------------------
# 3.1 Resample ALL symbols to 5minute
# -----------------------------------------------------------------------------
INSERT OR REPLACE INTO ohlcv_resampled
SELECT
instrument_key,
'5minute' AS timeframe,
time_bucket(INTERVAL '5 minutes', timestamp, TIMESTAMP '1970-01-01 09:15:00') AS timestamp,
FIRST(open ORDER BY timestamp) AS open,
MAX(high) AS high,
MIN(low) AS low,
LAST(close ORDER BY timestamp) AS close,
SUM(volume) AS volume,
0 AS oi
FROM ohlcv_1m
GROUP BY instrument_key, time_bucket(INTERVAL '5 minutes', timestamp, TIMESTAMP '1970-01-01 09:15:00')

# -----------------------------------------------------------------------------
# 3.2 Resample ALL symbols to 15minute
# -----------------------------------------------------------------------------
INSERT OR REPLACE INTO ohlcv_resampled
SELECT
instrument_key,
'15minute' AS timeframe,
time_bucket(INTERVAL '15 minutes', timestamp, TIMESTAMP '1970-01-01 09:15:00') AS timestamp,
FIRST(open ORDER BY timestamp) AS open,
MAX(high) AS high,
MIN(low) AS low,
LAST(close ORDER BY timestamp) AS close,
SUM(volume) AS volume,
0 AS oi
FROM ohlcv_1m
GROUP BY instrument_key, time_bucket(INTERVAL '15 minutes', timestamp, TIMESTAMP '1970-01-01 09:15:00')

# -----------------------------------------------------------------------------
# 3.3 Resample ALL symbols to 30minute
# -----------------------------------------------------------------------------
INSERT OR REPLACE INTO ohlcv_resampled
SELECT
instrument_key,
'30minute' AS timeframe,
time_bucket(INTERVAL '30 minutes', timestamp, TIMESTAMP '1970-01-01 09:15:00') AS timestamp,
FIRST(open ORDER BY timestamp) AS open,
MAX(high) AS high,
MIN(low) AS low,
LAST(close ORDER BY timestamp) AS close,
SUM(volume) AS volume,
0 AS oi
FROM ohlcv_1m
GROUP BY instrument_key, time_bucket(INTERVAL '30 minutes', timestamp, TIMESTAMP '1970-01-01 09:15:00')

# -----------------------------------------------------------------------------
# 3.4 Resample ALL symbols to 60minute
# -----------------------------------------------------------------------------
INSERT OR REPLACE INTO ohlcv_resampled
SELECT
instrument_key,
'60minute' AS timeframe,
time_bucket(INTERVAL '60 minutes', timestamp, TIMESTAMP '1970-01-01 09:15:00') AS timestamp,
FIRST(open ORDER BY timestamp) AS open,
MAX(high) AS high,
MIN(low) AS low,
LAST(close ORDER BY timestamp) AS close,
SUM(volume) AS volume,
0 AS oi
FROM ohlcv_1m
GROUP BY instrument_key, time_bucket(INTERVAL '60 minutes', timestamp, TIMESTAMP '1970-01-01 09:15:00')

# -----------------------------------------------------------------------------
# 3.5 Resample ALL symbols to 1day
# -----------------------------------------------------------------------------
INSERT OR REPLACE INTO ohlcv_resampled
SELECT
instrument_key,
'1day' AS timeframe,
CAST(DATE(timestamp) AS TIMESTAMP) + INTERVAL '9 hours 15 minutes' AS timestamp,
FIRST(open ORDER BY timestamp) AS open,
MAX(high) AS high,
MIN(low) AS low,
LAST(close ORDER BY timestamp) AS close,
SUM(volume) AS volume,
0 AS oi
FROM ohlcv_1m
GROUP BY instrument_key, DATE(timestamp)


# ==============================================================================
# 4. INCREMENTAL RESAMPLING (DATE RANGE)
# ==============================================================================

# -----------------------------------------------------------------------------
# 4.1 Resample specific date range - 15minute (all symbols)
# -----------------------------------------------------------------------------
INSERT OR REPLACE INTO ohlcv_resampled
SELECT
instrument_key,
'15minute' AS timeframe,
time_bucket(INTERVAL '15 minutes', timestamp, TIMESTAMP '1970-01-01 09:15:00') AS timestamp,
FIRST(open ORDER BY timestamp) AS open,
MAX(high) AS high,
MIN(low) AS low,
LAST(close ORDER BY timestamp) AS close,
SUM(volume) AS volume,
0 AS oi
FROM ohlcv_1m
WHERE timestamp >= '2025-09-01 00:00:00': : TIMESTAMP
    AND timestamp <= '2026-01-09 23:59:59': : TIMESTAMP
GROUP BY instrument_key, time_bucket(INTERVAL '15 minutes', timestamp, TIMESTAMP '1970-01-01 09:15:00')

# -----------------------------------------------------------------------------
# 4.2 Resample specific date range - 1day (all symbols)
# -----------------------------------------------------------------------------
INSERT OR REPLACE INTO ohlcv_resampled
SELECT
instrument_key,
'1day' AS timeframe,
CAST(DATE(timestamp) AS TIMESTAMP) + INTERVAL '9 hours 15 minutes' AS timestamp,
FIRST(open ORDER BY timestamp) AS open,
MAX(high) AS high,
MIN(low) AS low,
LAST(close ORDER BY timestamp) AS close,
SUM(volume) AS volume,
0 AS oi
FROM ohlcv_1m
WHERE timestamp >= '2026-01-07 00:00:00': : TIMESTAMP
    AND timestamp <= '2026-01-08 23:59:59': : TIMESTAMP
GROUP BY instrument_key, DATE(timestamp)


# ==============================================================================
# 5. DATA RETRIEVAL QUERIES
# ==============================================================================

# -----------------------------------------------------------------------------
# 5.1 Get latest candles for a symbol (any timeframe)
# -----------------------------------------------------------------------------
SELECT
i.trading_symbol,
r.timeframe,
r.timestamp,
r.open, r.high, r.low, r.close,
r.volume
FROM ohlcv_resampled r
JOIN instruments i ON r.instrument_key = i.instrument_key
WHERE i.trading_symbol = 'RELIANCE'
AND r.timeframe = '15minute'
ORDER BY r.timestamp DESC
LIMIT 20

# -----------------------------------------------------------------------------
# 5.2 Get OHLCV data for specific date range
# -----------------------------------------------------------------------------
SELECT
i.trading_symbol,
r.timestamp,
r.open, r.high, r.low, r.close,
r.volume
FROM ohlcv_resampled r
JOIN instruments i ON r.instrument_key = i.instrument_key
WHERE i.trading_symbol = 'RELIANCE'
AND r.timeframe = '15minute'
AND DATE(r.timestamp) = '2026-01-07'
ORDER BY r.timestamp

# -----------------------------------------------------------------------------
# 5.3 Get daily OHLCV for multiple symbols
# -----------------------------------------------------------------------------
SELECT
i.trading_symbol,
r.timestamp,
r.open, r.high, r.low, r.close,
r.volume
FROM ohlcv_resampled r
JOIN instruments i ON r.instrument_key = i.instrument_key
WHERE i.trading_symbol IN('RELIANCE', 'TCS', 'INFY', 'HDFCBANK')
AND r.timeframe = '1day'
ORDER BY i.trading_symbol, r.timestamp DESC

# -----------------------------------------------------------------------------
# 5.4 Get intraday data for today
# -----------------------------------------------------------------------------
SELECT
i.trading_symbol,
r.timestamp,
r.open, r.high, r.low, r.close,
r.volume
FROM ohlcv_resampled r
JOIN instruments i ON r.instrument_key = i.instrument_key
WHERE i.trading_symbol = 'RELIANCE'
AND r.timeframe = '5minute'
AND DATE(r.timestamp) = CURRENT_DATE
ORDER BY r.timestamp


# ==============================================================================
# 6. DELETE / CLEANUP QUERIES
# ==============================================================================

# -----------------------------------------------------------------------------
# 6.1 Delete all resampled data (CAUTION!)
# -----------------------------------------------------------------------------
DELETE FROM ohlcv_resampled

# -----------------------------------------------------------------------------
# 6.2 Delete resampled data for specific timeframe
# -----------------------------------------------------------------------------
DELETE FROM ohlcv_resampled WHERE timeframe = '15minute'

# -----------------------------------------------------------------------------
# 6.3 Delete resampled data for specific symbol
# -----------------------------------------------------------------------------
DELETE FROM ohlcv_resampled
WHERE instrument_key = (
    SELECT instrument_key FROM instruments WHERE trading_symbol='RELIANCE' LIMIT 1
)

# -----------------------------------------------------------------------------
# 6.4 Delete resampled data for specific symbol and timeframe
# -----------------------------------------------------------------------------
DELETE FROM ohlcv_resampled
WHERE instrument_key = (
    SELECT instrument_key FROM instruments WHERE trading_symbol='RELIANCE' LIMIT 1
)
AND timeframe = '15minute'

# -----------------------------------------------------------------------------
# 6.5 Delete resampled data for date range
# -----------------------------------------------------------------------------
DELETE FROM ohlcv_resampled
WHERE timestamp >= '2026-01-01': : TIMESTAMP
    AND timestamp <= '2026-01-07':: TIMESTAMP

# -----------------------------------------------------------------------------
# 6.6 Delete 1m data for specific symbol (CAUTION!)
# -----------------------------------------------------------------------------
DELETE FROM ohlcv_1m
WHERE instrument_key = (
    SELECT instrument_key FROM instruments WHERE trading_symbol='RELIANCE' LIMIT 1
)


# ==============================================================================
# 7. INSTRUMENT QUERIES
# ==============================================================================

# -----------------------------------------------------------------------------
# 7.1 List all F&O stocks
# -----------------------------------------------------------------------------
SELECT trading_symbol, name, lot_size
FROM fo_stocks_master
WHERE is_active = TRUE
ORDER BY trading_symbol

# -----------------------------------------------------------------------------
# 7.2 Find instrument key for a symbol
# -----------------------------------------------------------------------------
SELECT instrument_key, trading_symbol, name, segment, lot_size
FROM instruments
WHERE trading_symbol = 'RELIANCE'
AND segment = 'NSE_EQ'

# -----------------------------------------------------------------------------
# 7.3 List all NSE_EQ instruments with data
# -----------------------------------------------------------------------------
SELECT DISTINCT
i.trading_symbol,
i.name,
COUNT(o.timestamp) as candles
FROM instruments i
JOIN ohlcv_1m o ON i.instrument_key = o.instrument_key
WHERE i.segment = 'NSE_EQ'
GROUP BY i.trading_symbol, i.name
ORDER BY i.trading_symbol

# -----------------------------------------------------------------------------
# 7.4 Search instruments by name
# -----------------------------------------------------------------------------
SELECT instrument_key, trading_symbol, name, segment
FROM instruments
WHERE name ILIKE '%RELIANCE%'
OR trading_symbol ILIKE '%RELIANCE%'
ORDER BY segment, trading_symbol


# ==============================================================================
# 8. VALIDATION QUERIES
# ==============================================================================

# -----------------------------------------------------------------------------
# 8.1 Compare 1m candle count vs resampled (should match ratio)
# -----------------------------------------------------------------------------
SELECT
i.trading_symbol,
(SELECT COUNT(*) FROM ohlcv_1m WHERE instrument_key=i.instrument_key) as candles_1m,
(SELECT COUNT(*) FROM ohlcv_resampled WHERE instrument_key=i.instrument_key AND timeframe='5minute') as candles_5m,
(SELECT COUNT(*) FROM ohlcv_resampled WHERE instrument_key=i.instrument_key AND timeframe='15minute') as candles_15m,
(SELECT COUNT(*) FROM ohlcv_resampled WHERE instrument_key=i.instrument_key AND timeframe='1day') as candles_1d
FROM instruments i
WHERE i.trading_symbol IN('RELIANCE', 'TCS', 'INFY')
AND i.segment = 'NSE_EQ'

# -----------------------------------------------------------------------------
# 8.2 Validate OHLC integrity (High >= Low, etc.)
# -----------------------------------------------------------------------------
SELECT
i.trading_symbol,
r.timeframe,
COUNT(*) as invalid_candles
FROM ohlcv_resampled r
JOIN instruments i ON r.instrument_key = i.instrument_key
WHERE r.high < r.low
OR r.open > r.high OR r.open < r.low
OR r.close > r.high OR r.close < r.low
GROUP BY i.trading_symbol, r.timeframe
HAVING COUNT(*) > 0

# -----------------------------------------------------------------------------
# 8.3 Check for gaps in daily data
# -----------------------------------------------------------------------------
WITH daily_dates AS(
    SELECT
    instrument_key,
    DATE(timestamp) as trade_date,
    LAG(DATE(timestamp)) OVER(PARTITION BY instrument_key ORDER BY timestamp) as prev_date
    FROM ohlcv_resampled
    WHERE timeframe='1day'
)
SELECT
i.trading_symbol,
d.trade_date,
d.prev_date,
d.trade_date - d.prev_date as gap_days
FROM daily_dates d
JOIN instruments i ON d.instrument_key = i.instrument_key
WHERE d.trade_date - d.prev_date > 4 - - More than 4 days gap(weekends + 1)
ORDER BY gap_days DESC
LIMIT 20


# ==============================================================================
# 9. PERFORMANCE / ANALYSIS QUERIES
# ==============================================================================

# -----------------------------------------------------------------------------
# 9.1 Daily returns for a symbol
# -----------------------------------------------------------------------------
SELECT
timestamp,
close,
LAG(close) OVER(ORDER BY timestamp) as prev_close,
ROUND((close - LAG(close) OVER(ORDER BY timestamp)) / LAG(close) OVER(ORDER BY timestamp) * 100, 2) as daily_return_pct
FROM ohlcv_resampled r
JOIN instruments i ON r.instrument_key = i.instrument_key
WHERE i.trading_symbol = 'RELIANCE'
AND r.timeframe = '1day'
ORDER BY timestamp DESC
LIMIT 30

# -----------------------------------------------------------------------------
# 9.2 Volatility (daily range as % of close)
# -----------------------------------------------------------------------------
SELECT
i.trading_symbol,
AVG((r.high - r.low) / r.close * 100) as avg_daily_range_pct,
MAX((r.high - r.low) / r.close * 100) as max_daily_range_pct,
MIN((r.high - r.low) / r.close * 100) as min_daily_range_pct
FROM ohlcv_resampled r
JOIN instruments i ON r.instrument_key = i.instrument_key
WHERE r.timeframe = '1day'
AND i.segment = 'NSE_EQ'
GROUP BY i.trading_symbol
ORDER BY avg_daily_range_pct DESC
LIMIT 20

# -----------------------------------------------------------------------------
# 9.3 Top gainers/losers today
# -----------------------------------------------------------------------------
WITH today_data AS(
    SELECT
    i.trading_symbol,
    FIRST(r.open ORDER BY r.timestamp) as day_open,
    LAST(r.close ORDER BY r.timestamp) as day_close
    FROM ohlcv_resampled r
    JOIN instruments i ON r.instrument_key=i.instrument_key
    WHERE r.timeframe='15minute'
    AND DATE(r.timestamp)=(SELECT MAX(DATE(timestamp)) FROM ohlcv_resampled WHERE timeframe='15minute')
    GROUP BY i.trading_symbol
)
SELECT
trading_symbol,
day_open,
day_close,
ROUND((day_close - day_open) / day_open * 100, 2) as change_pct
FROM today_data
ORDER BY change_pct DESC
LIMIT 10

# -----------------------------------------------------------------------------
# 9.4 Volume analysis - highest volume days
# -----------------------------------------------------------------------------
SELECT
i.trading_symbol,
DATE(r.timestamp) as trade_date,
SUM(r.volume) as total_volume,
r.close
FROM ohlcv_resampled r
JOIN instruments i ON r.instrument_key = i.instrument_key
WHERE i.trading_symbol = 'RELIANCE'
AND r.timeframe = '1day'
GROUP BY i.trading_symbol, DATE(r.timestamp), r.close
ORDER BY total_volume DESC
LIMIT 10


# ==============================================================================
# 10. MAINTENANCE QUERIES
# ==============================================================================

# -----------------------------------------------------------------------------
# 10.1 Vacuum database (reclaim space)
# -----------------------------------------------------------------------------
VACUUM

# -----------------------------------------------------------------------------
# 10.2 Analyze tables (update statistics)
# -----------------------------------------------------------------------------
ANALYZE

# -----------------------------------------------------------------------------
# 10.3 Check table schemas
# -----------------------------------------------------------------------------
DESCRIBE ohlcv_1m
DESCRIBE ohlcv_resampled
DESCRIBE instruments

# -----------------------------------------------------------------------------
# 10.4 Export resampled data to CSV (run from Python)
# -----------------------------------------------------------------------------
-- COPY(
    -- SELECT i.trading_symbol, r.*
    -- FROM ohlcv_resampled r
    - -     JOIN instruments i ON r.instrument_key=i.instrument_key
    - -     WHERE r.timeframe='1day'
    - -) TO 'daily_data.csv' (HEADER, DELIMITER ',')


# ==============================================================================
# NOTES
# ==============================================================================
#
# 1. Replace 'RELIANCE' with any trading symbol as needed
# 2. Replace dates like '2026-01-07' with your target dates
# 3. time_bucket origin '1970-01-01 09:15:00' aligns buckets to IST market open
# 4. Daily timestamp uses 09:15:00 to represent the trading day
# 5. Use LIMIT for large result sets to avoid memory issues
# 6. For batch operations on large datasets, consider running during off-hours
#
# ==============================================================================
