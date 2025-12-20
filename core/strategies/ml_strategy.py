# In a notebook or script
import pandas as pd
from backtest.hmm_regime import supertrend_with_hmm, analyze_regime_distribution

df = pd.read_parquet("data/derived/HDFCBANK/15minute/merged_HDFCBANK_15minute.parquet")
df = supertrend_with_hmm(df, use_hmm=True)

print(analyze_regime_distribution(df))




BASE_DIR = Path(__file__).resolve().parents[1]
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)