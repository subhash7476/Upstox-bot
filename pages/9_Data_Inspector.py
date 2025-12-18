import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Data Inspector")

st.title("🔍 Data Inspector")

# Tabs for Raw vs Derived
mode = st.radio("Select Data Source", ["Raw Partitions (1min, etc)", "Derived/Resampled Data"], horizontal=True)

if mode == "Raw Partitions (1min, etc)":
    st.subheader("Raw Data Inspector (Parquet Partitions)")
    
    # Scan for available symbols
    root = Path("data/stocks")
    if not root.exists():
        st.error("No data/stocks folder found.")
        st.stop()
        
    symbols = [p.name for p in root.iterdir() if p.is_dir()]
    if not symbols:
        st.warning("No symbols found in data/stocks.")
        st.stop()
        
    c1, c2, c3 = st.columns(3)
    with c1:
        sel_sym = st.selectbox("Symbol", sorted(symbols))
    
    # Scan for intervals
    sym_path = root / sel_sym
    intervals = [p.name for p in sym_path.iterdir() if p.is_dir()]
    with c2:
        sel_int = st.selectbox("Interval", sorted(intervals))
        
    # Scan for dates (partitions)
    # We look for year=*/month=*/day=*
    part_path = sym_path / sel_int
    files = sorted(list(part_path.rglob("data.parquet")))
    
    if not files:
        st.info("No parquet files found for this selection.")
        st.stop()
        
    # Extract readable dates for dropdown
    file_map = {}
    for f in files:
        # structure: .../year=2025/month=12/day=10/data.parquet
        try:
            y = f.parent.parent.parent.name.replace("year=", "")
            m = f.parent.parent.name.replace("month=", "")
            d = f.parent.name.replace("day=", "")
            date_str = f"{y}-{m}-{d}"
            file_map[date_str] = f
        except:
            continue
            
    with c3:
        sel_date = st.selectbox("Select Date", sorted(file_map.keys(), reverse=True))
        
    if st.button("Load Data"):
        fpath = file_map[sel_date]
        df = pd.read_parquet(fpath)
        
        st.markdown(f"**Loaded:** `{fpath}`")
        st.write(f"**Rows:** {len(df)}")
        
        # Display Data
        st.dataframe(df, use_container_width=True)
        
        # Simple Chart
        if not df.empty and "Close" in df.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode='lines', name='Close'))
            st.plotly_chart(fig, use_container_width=True)

elif mode == "Derived/Resampled Data":
    st.subheader("Derived Data Inspector (Resampled)")
    
    root = Path("data/derived")
    if not root.exists():
        st.error("No data/derived folder found. Run the Resampler first.")
        st.stop()
        
    symbols = [p.name for p in root.iterdir() if p.is_dir()]
    if not symbols:
        st.warning("No derived data found.")
        st.stop()
        
    c1, c2 = st.columns(2)
    with c1:
        sel_sym = st.selectbox("Symbol", sorted(symbols))
        
    # Scan for derived files
    sym_path = root / sel_sym
    # usually folders like "15minute"
    tfs = [p.name for p in sym_path.iterdir() if p.is_dir()]
    
    with c2:
        sel_tf = st.selectbox("Timeframe", sorted(tfs))
        
    tf_path = sym_path / sel_tf
    files = sorted([f for f in tf_path.glob("*.parquet")])
    
    sel_file = st.selectbox("Select File", [f.name for f in files])
    
    if st.button("Load Derived File"):
        fpath = tf_path / sel_file
        df = pd.read_parquet(fpath)
        
        st.write(f"**Rows:** {len(df)}")
        st.dataframe(df.head(1000), use_container_width=True) # Show first 1000 to avoid lag
        
        if not df.empty and "Close" in df.columns:
            fig = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])
            fig.update_layout(height=600, title=f"{sel_sym} {sel_tf}")
            st.plotly_chart(fig, use_container_width=True)