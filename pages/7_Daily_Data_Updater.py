"""
Daily Data Updater - Automated Incremental Updates
Fetches missing data for all stocks from their last available date to today
Perfect for running after market close (3:30 PM IST)
"""

import sys
import os
from pathlib import Path

# Ensure root is in path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta
import time

from data.data_manager import fetch_historical_range
from data.resampler import build_derived_parquet

# Page config
st.set_page_config(
    page_title="Daily Data Updater",
    page_icon="🔄",
    layout="wide"
)

st.title("🔄 Daily Data Updater")
st.markdown("**Automated incremental data updates for all stocks**")

# Constants
SEGMENT = "NSE_EQ"  # Default segment
INTERVAL = "1minute"
STOCKS_DIR = Path(ROOT) / "data" / "stocks"

# Helper Functions
@st.cache_data(ttl=60)
def get_all_stocks_with_data():
    """Get list of all stocks that have existing 1-minute data"""
    if not STOCKS_DIR.exists():
        return []
    
    stocks = []
    for symbol_dir in STOCKS_DIR.iterdir():
        if symbol_dir.is_dir():
            minute_dir = symbol_dir / "1minute"
            if minute_dir.exists() and list(minute_dir.rglob("*.parquet")):
                stocks.append(symbol_dir.name)
    
    return sorted(stocks)


def get_last_data_date(symbol):
    """
    Find the last date for which data exists for a symbol
    Returns: date object or None
    """
    minute_dir = STOCKS_DIR / symbol / "1minute"
    
    if not minute_dir.exists():
        return None
    
    # Find all year directories and get the latest
    year_dirs = sorted(minute_dir.glob("year=*"), reverse=True)
    
    if not year_dirs:
        return None
    
    for year_dir in year_dirs:
        month_dirs = sorted(year_dir.glob("month=*"), reverse=True)
        
        for month_dir in month_dirs:
            day_dirs = sorted(month_dir.glob("day=*"), reverse=True)
            
            for day_dir in day_dirs:
                # Extract date from path
                year = int(year_dir.name.split("=")[1])
                month = int(month_dir.name.split("=")[1])
                day = int(day_dir.name.split("=")[1])
                
                try:
                    return date(year, month, day)
                except ValueError:
                    continue
    
    return None


def get_data_date_range(symbol):
    """Get first and last date of data for a symbol"""
    minute_dir = STOCKS_DIR / symbol / "1minute"
    
    if not minute_dir.exists():
        return None, None
    
    all_dates = []
    
    for year_dir in minute_dir.glob("year=*"):
        year = int(year_dir.name.split("=")[1])
        
        for month_dir in year_dir.glob("month=*"):
            month = int(month_dir.name.split("=")[1])
            
            for day_dir in month_dir.glob("day=*"):
                day = int(day_dir.name.split("=")[1])
                
                try:
                    all_dates.append(date(year, month, day))
                except ValueError:
                    continue
    
    if not all_dates:
        return None, None
    
    return min(all_dates), max(all_dates)


def count_data_days(symbol):
    """Count number of days with data"""
    minute_dir = STOCKS_DIR / symbol / "1minute"
    
    if not minute_dir.exists():
        return 0
    
    return len(list(minute_dir.glob("year=*/month=*/day=*")))


def is_market_open_day(check_date):
    """
    Check if a date is a potential market day (weekday)
    Note: This doesn't account for holidays, but API will return empty for those
    """
    return check_date.weekday() < 5  # Monday=0, Friday=4


def get_missing_dates(symbol, end_date=None):
    """
    Get list of dates that need to be fetched
    Returns: (start_date, end_date, days_to_fetch)
    """
    last_date = get_last_data_date(symbol)
    
    if last_date is None:
        return None, None, 0
    
    if end_date is None:
        end_date = date.today()
    
    # Start from day after last data
    start_date = last_date + timedelta(days=1)
    
    # If last data is today or in future, nothing to fetch
    if start_date > end_date:
        return last_date, end_date, 0
    
    # Count weekdays (approximate market days)
    days_to_fetch = 0
    current = start_date
    while current <= end_date:
        if is_market_open_day(current):
            days_to_fetch += 1
        current += timedelta(days=1)
    
    return start_date, end_date, days_to_fetch


# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Update mode
    update_mode = st.radio(
        "Update Mode",
        ["Incremental (Smart)", "Custom Date Range"],
        help="Incremental: Fetches from last date to today\nCustom: Specify date range"
    )
    
    # Custom date range (if selected)
    if update_mode == "Custom Date Range":
        st.subheader("Custom Range")
        col1, col2 = st.columns(2)
        with col1:
            custom_start = st.date_input(
                "From Date",
                value=date(2025, 1, 1),
                max_value=date.today()
            )
        with col2:
            custom_end = st.date_input(
                "To Date",
                value=date.today(),
                max_value=date.today()
            )
    else:
        custom_start = None
        custom_end = date.today()
    
    st.divider()
    
    # Auto-resample option
    auto_resample = st.checkbox(
        "Auto-Resample After Fetch",
        value=True,
        help="Automatically update derived timeframes (15m, 30m, 1h, 4h, daily)"
    )
    
    if auto_resample:
        resample_timeframes = st.multiselect(
            "Timeframes to Update",
            ["15minute", "30minute", "60minute", "240minute", "1day"],
            default=["15minute", "30minute", "60minute"],
            help="Select which timeframes to regenerate"
        )
    
    st.divider()
    
    # Market timing info
    st.info("""
    **🕐 Market Hours (NSE)**
    - Open: 9:15 AM IST
    - Close: 3:30 PM IST
    
    **Best Update Time:**
    After 4:00 PM IST daily
    """)
    
    # Current time
    current_time = datetime.now()
    st.metric("Current Time", current_time.strftime("%I:%M %p"))

# Main Content
st.subheader("📊 Stock Status Overview")

# Get all stocks
all_stocks = get_all_stocks_with_data()

if not all_stocks:
    st.error("No stocks with existing data found in `data/stocks/`")
    st.info("Please fetch initial data first using Page 2 (Fetch & Manage Data)")
    st.stop()

# Analyze all stocks
with st.spinner("Analyzing stock data status..."):
    stock_status = []
    
    for symbol in all_stocks:
        first_date, last_date = get_data_date_range(symbol)
        days_count = count_data_days(symbol)
        
        # Calculate missing days
        if update_mode == "Incremental (Smart)":
            start_date, end_date, missing_days = get_missing_dates(symbol, custom_end)
        else:
            # For custom range, just show the range
            start_date = custom_start
            end_date = custom_end
            missing_days = (end_date - start_date).days if start_date and end_date else 0
        
        stock_status.append({
            'Symbol': symbol,
            'First Date': first_date,
            'Last Date': last_date,
            'Total Days': days_count,
            'Missing Days': missing_days,
            'Update From': start_date,
            'Update To': end_date,
            'Status': '✅ Up to date' if missing_days == 0 else f'⚠️ {missing_days} days behind'
        })

status_df = pd.DataFrame(stock_status)

# Summary metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Stocks", len(all_stocks))
with col2:
    up_to_date = len(status_df[status_df['Missing Days'] == 0])
    st.metric("Up to Date", up_to_date)
with col3:
    needs_update = len(status_df[status_df['Missing Days'] > 0])
    st.metric("Needs Update", needs_update)
with col4:
    total_missing = status_df['Missing Days'].sum()
    st.metric("Total Days to Fetch", int(total_missing))

st.divider()

# Display status table
st.subheader("📋 Stock Update Status")

# Filter options
col1, col2 = st.columns([2, 1])
with col1:
    show_filter = st.selectbox(
        "Filter",
        ["All Stocks", "Needs Update Only", "Up to Date Only"]
    )
with col2:
    sort_by = st.selectbox(
        "Sort By",
        ["Symbol", "Missing Days", "Last Date"]
    )

# Apply filters
if show_filter == "Needs Update Only":
    display_df = status_df[status_df['Missing Days'] > 0]
elif show_filter == "Up to Date Only":
    display_df = status_df[status_df['Missing Days'] == 0]
else:
    display_df = status_df

# Apply sorting
if sort_by == "Missing Days":
    display_df = display_df.sort_values('Missing Days', ascending=False)
elif sort_by == "Last Date":
    display_df = display_df.sort_values('Last Date', ascending=False)
else:
    display_df = display_df.sort_values('Symbol')

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    height=400
)

st.divider()

# Update Section
st.subheader("🚀 Run Update")

# Stock selection
col1, col2 = st.columns([3, 1])
with col1:
    stocks_to_update = st.multiselect(
        "Select Stocks to Update",
        options=all_stocks,
        default=all_stocks if needs_update > 0 else [],
        help="Select stocks to fetch data for"
    )
with col2:
    st.write("")  # Spacing
    st.write("")  # Spacing
    if st.button("Select All Pending", use_container_width=True):
        pending = status_df[status_df['Missing Days'] > 0]['Symbol'].tolist()
        stocks_to_update = pending
        st.rerun()

# Show what will be updated
if stocks_to_update:
    update_preview = status_df[status_df['Symbol'].isin(stocks_to_update)]
    total_days_to_fetch = update_preview['Missing Days'].sum()
    
    st.info(f"""
    **Update Preview:**
    - 📊 Stocks: {len(stocks_to_update)}
    - 📅 Total Days to Fetch: {int(total_days_to_fetch)}
    - ⏱️ Estimated Time: {int(len(stocks_to_update) * 2)} - {int(len(stocks_to_update) * 5)} minutes
    - 🔄 Auto-Resample: {'Yes (' + ', '.join(resample_timeframes) + ')' if auto_resample else 'No'}
    """)
    
    # Dry run option
    dry_run = st.checkbox("Dry Run (Preview only, don't fetch)", value=False)
    
    # Start Update Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        start_update = st.button(
            "🔄 Start Update" if not dry_run else "👁️ Preview Update",
            type="primary",
            use_container_width=True
        )
    
    # Run Update
    if start_update:
        st.subheader("⚙️ Update Progress")
        
        # Results tracking
        results = []
        success_count = 0
        failed_count = 0
        no_data_count = 0
        
        # Progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        start_time = time.time()
        
        for idx, symbol in enumerate(stocks_to_update):
            # Get date range for this symbol
            if update_mode == "Incremental (Smart)":
                start_date, end_date, days_expected = get_missing_dates(symbol, custom_end)
            else:
                start_date = custom_start
                end_date = custom_end
                days_expected = (end_date - start_date).days
            
            # Update progress
            progress = idx / len(stocks_to_update)
            progress_bar.progress(progress)
            status_text.text(f"Updating {symbol} ({idx + 1}/{len(stocks_to_update)})...")
            
            # Skip if no dates to fetch
            if days_expected == 0 or start_date is None:
                results.append({
                    'Symbol': symbol,
                    'Status': '⚠️ No Update Needed',
                    'Days Fetched': 0,
                    'From': '-',
                    'To': '-',
                    'Message': 'Already up to date'
                })
                no_data_count += 1
                continue
            
            if dry_run:
                # Dry run - just preview
                results.append({
                    'Symbol': symbol,
                    'Status': '👁️ Would Fetch',
                    'Days Fetched': days_expected,
                    'From': start_date.strftime('%Y-%m-%d'),
                    'To': end_date.strftime('%Y-%m-%d'),
                    'Message': f'Would fetch {days_expected} days'
                })
                success_count += 1
                continue
            
            # Actual fetch
            try:
                saved_files = fetch_historical_range(
                    symbol=symbol,
                    segment=SEGMENT,
                    interval=INTERVAL,
                    from_date=start_date,
                    to_date=end_date,
                    force=False
                )
                
                if saved_files:
                    results.append({
                        'Symbol': symbol,
                        'Status': '✅ Success',
                        'Days Fetched': len(saved_files),
                        'From': start_date.strftime('%Y-%m-%d'),
                        'To': end_date.strftime('%Y-%m-%d'),
                        'Message': f'Fetched {len(saved_files)} days'
                    })
                    success_count += 1
                    
                    # Auto-resample if enabled
                    if auto_resample and resample_timeframes:
                        for tf in resample_timeframes:
                            try:
                                build_derived_parquet(symbol, tf, overwrite=True)
                            except Exception as e:
                                pass  # Continue even if resample fails
                
                else:
                    results.append({
                        'Symbol': symbol,
                        'Status': '⚠️ No Data',
                        'Days Fetched': 0,
                        'From': start_date.strftime('%Y-%m-%d'),
                        'To': end_date.strftime('%Y-%m-%d'),
                        'Message': 'No data returned (holidays/weekends)'
                    })
                    no_data_count += 1
                
            except Exception as e:
                results.append({
                    'Symbol': symbol,
                    'Status': '❌ Failed',
                    'Days Fetched': 0,
                    'From': start_date.strftime('%Y-%m-%d') if start_date else '-',
                    'To': end_date.strftime('%Y-%m-%d') if end_date else '-',
                    'Message': str(e)[:100]
                })
                failed_count += 1
        
        # Complete
        progress_bar.progress(1.0)
        status_text.text("✅ Update Complete!")
        
        elapsed_time = time.time() - start_time
        
        # Summary
        if dry_run:
            st.info(f"""
            **Dry Run Complete!**
            - 📊 Would update: {success_count} stocks
            - ⏱️ Analysis Time: {elapsed_time:.2f}s
            
            No data was actually fetched. Uncheck "Dry Run" to proceed with actual update.
            """)
        else:
            st.success(f"""
            **Update Complete!**
            - ✅ Success: {success_count}
            - ⚠️ No Data: {no_data_count}
            - ❌ Failed: {failed_count}
            - ⏱️ Time: {elapsed_time:.2f}s ({elapsed_time/60:.1f} minutes)
            - 🔄 Resampled: {len(resample_timeframes) if auto_resample else 0} timeframes
            """)
        
        # Detailed results
        with st.expander("📊 Detailed Results", expanded=True):
            results_df = pd.DataFrame(results)
            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Download
            csv = results_df.to_csv(index=False)
            st.download_button(
                "📥 Download Results CSV",
                csv,
                f"daily_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )
        
        # Refresh data
        if not dry_run:
            st.info("💡 Data has been updated. Refresh the page to see updated statistics.")
            if st.button("🔄 Refresh Page"):
                st.rerun()

else:
    st.info("👆 Select stocks to update above")

# Footer
st.divider()

# Scheduling info
with st.expander("⏰ How to Schedule Automatic Daily Updates"):
    st.markdown("""
    ### Linux/Mac (Cron Job)
    
    Create a shell script `update_stocks.sh`:
    ```bash
    #!/bin/bash
    cd /path/to/trading-bot-pro
    source venv/bin/activate  # If using virtual env
    streamlit run pages/9_Daily_Data_Updater.py --server.headless=true
    ```
    
    Add to crontab (run at 4:30 PM daily):
    ```bash
    crontab -e
    # Add this line:
    30 16 * * 1-5 /path/to/update_stocks.sh >> /var/log/stock_update.log 2>&1
    ```
    
    ### Windows (Task Scheduler)
    
    1. Open Task Scheduler
    2. Create Basic Task: "Daily Stock Update"
    3. Trigger: Daily at 4:30 PM, Monday-Friday
    4. Action: Start Program
       - Program: `streamlit`
       - Arguments: `run pages/9_Daily_Data_Updater.py --server.headless=true`
       - Start in: `C:\\path\\to\\trading-bot-pro`
    
    ### Python Script (Alternative)
    
    Create `auto_update.py`:
    ```python
    from data.data_manager import fetch_historical_range
    from datetime import date, timedelta
    
    stocks = ["RELIANCE", "TCS", ...]  # Your 100 stocks
    
    for symbol in stocks:
        try:
            fetch_historical_range(
                symbol=symbol,
                segment="NSE_EQ",
                interval="1minute",
                from_date=date.today() - timedelta(days=5),
                to_date=date.today(),
                force=False
            )
            print(f"✅ {symbol} updated")
        except Exception as e:
            print(f"❌ {symbol} failed: {e}")
    ```
    
    Then schedule this script instead.
    """)

st.caption(f"""
**Daily Data Updater** | Smart incremental updates | 
Auto-resample enabled | Best run time: After 4:00 PM IST | 
Data Location: `{STOCKS_DIR}`
""")