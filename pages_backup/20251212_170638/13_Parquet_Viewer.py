import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Parquet File Browser", layout="wide")
st.title("📁 Parquet File Browser")

# ---------------------------------------
# CONFIGURATION
# ---------------------------------------
DEFAULT_DIR = "data/processed"

st.sidebar.header("Folder Selection")

folder = st.sidebar.text_input(
    "Folder path containing .parquet files",
    value=DEFAULT_DIR
)

# Validate folder
if not os.path.exists(folder):
    st.error(f"❌ Folder does not exist: {folder}")
    st.stop()

# List parquet files
files = [f for f in os.listdir(folder) if f.endswith(".parquet")]

if not files:
    st.warning("No .parquet files found in this folder.")
    st.stop()

files_sorted = sorted(files)

selected = st.sidebar.selectbox("Select a file", files_sorted)

file_path = os.path.join(folder, selected)

# ---------------------------------------
# FILE INFO
# ---------------------------------------
file_size = os.path.getsize(file_path) / (1024 * 1024)
st.info(f"📄 **File:** `{selected}` — {file_size:.2f} MB")

# ---------------------------------------
# LOAD DATA SAFELY
# ---------------------------------------
try:
    df = pd.read_parquet(file_path)
except Exception as e:
    st.error(f"Failed to load file: {e}")
    st.stop()

rows, cols = df.shape
st.success(f"Loaded successfully: **{rows:,} rows × {cols:,} columns**")

# ---------------------------------------
# PREVIEW TABLE (First 50 rows)
# ---------------------------------------
st.subheader("Data Preview (First 50 rows)")
preview = df.head(50)
st.dataframe(preview, use_container_width=True)

# ---------------------------------------
# OPTIONAL: COLUMN LIST
# ---------------------------------------
with st.expander("📌 Show Columns"):
    st.write(list(df.columns))

# ---------------------------------------
# OPTIONAL: SHOW FULL STATS
# ---------------------------------------
with st.expander("📊 Data Summary"):
    st.write(df.describe())
