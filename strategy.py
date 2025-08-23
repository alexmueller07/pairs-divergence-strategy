from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import yfinance as yf
from config import ROLLING_WINDOW, LIVE_PERIOD_MIN, LIVE_INTERVAL, DOWNLOAD_THREADING

def download_pair_prices(t1: str, t2: str,
                         period: str = LIVE_PERIOD_MIN,
                         interval: str = LIVE_INTERVAL) -> pd.DataFrame:

    data = yf.download(
        [t1, t2],
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=True,
        threads=DOWNLOAD_THREADING,
        progress=False,
    )
    if not hasattr(data.columns, "levels") or len(data.columns.levels) < 2:
        raise ValueError("Unexpected yfinance response for pair")

    df = pd.DataFrame({t1: data[t1]["Close"], t2: data[t2]["Close"]}).dropna().sort_index()
    if df.shape[0] < ROLLING_WINDOW + 5:
        raise ValueError("Not enough bars for signals")
    return df

def rolling_beta(a: pd.Series, b: pd.Series, window: int = ROLLING_WINDOW) -> pd.Series:
    """
    Rolling OLS: a ~ alpha + beta * b  => beta = cov(a,b)/var(b)
    """
    cov = a.rolling(window).cov(b)
    var = b.rolling(window).var()
    beta = cov / var.replace(0, np.nan)
    return beta

def zscore(series: pd.Series, window: int = ROLLING_WINDOW) -> pd.Series:
    m = series.rolling(window).mean()
    s = series.rolling(window).std(ddof=0)
    return (series - m) / s

# Optional: quick offline backtest on a pair
def quick_backtest(t1: str, t2: str, window: int = ROLLING_WINDOW) -> Dict[str, Any]:
    df = download_pair_prices(t1, t2, period="30d", interval="5m")
    la, lb = np.log(df[t1]), np.log(df[t2])
    beta = rolling_beta(la, lb, window).ffill()
    spread = la - beta * lb
    z = zscore(spread, window)
    return {
        "bars": len(df),
        "last_z": float(z.dropna().iloc[-1]) if z.dropna().size else float("nan"),
        "last_beta": float(beta.dropna().iloc[-1]) if beta.dropna().size else float("nan"),
    }
