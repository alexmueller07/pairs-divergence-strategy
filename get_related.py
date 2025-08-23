import pandas as pd
import yfinance as yf
from typing import List, Tuple
from get_tickers import get_tickers
from config import PERIOD, INTERVAL, CORRELATION_THRESHOLD, MIN_VALID_RATIO, DOWNLOAD_THREADING
import re

STOPWORDS = {
    "inc", "incorporated", "corporation", "corp", "ltd", "limited", "plc",
    "company", "co", "group", "holdings", "holding", "international", "global",
    "class", "cl", "sa", "ag", "nv", "oa", "oa", "adr", "ads", "trust",
    "fund", "etf", "lp", "partnership", "technologies", "technology",
    "systems", "solutions", "industries", "industry"
}

def normalize_company_name(name: str) -> set:
    if not name:
        return set()

    name = name.lower()
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    words = [w for w in name.split() if w and w not in STOPWORDS and len(w) >= 4]
    return set(words)

def fetch_company_names(tickers: list) -> dict:
    names = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            long_name = info.get("longName") or info.get("shortName") or ""
            names[t] = normalize_company_name(long_name)
        except Exception:
            names[t] = set()
    return names

def is_same_company(words1: set, words2: set) -> bool:
    if not words1 or not words2:
        return False
    overlap = words1.intersection(words2)

    min_size = min(len(words1), len(words2))
    if min_size == 0:
        return False
    return (len(overlap) / min_size) >= 0.7

# --- main function ---
def get_related_pairs(
    period: str = PERIOD,
    interval: str = INTERVAL,
    correlation_threshold: float = CORRELATION_THRESHOLD,
    min_valid_ratio: float = MIN_VALID_RATIO,
) -> List[Tuple[str, str, float]]:
    """
    Returns [(ticker1, ticker2, corr), ...] with |corr| >= threshold,
    sorted by |corr| desc, excluding pairs from the same company (GOOG/GOOGL etc.).
    """
    tickers = get_tickers()
    if not tickers:
        raise ValueError("No tickers from screener")

    print(f"Loaded {len(tickers)} tickers")
    print(f"Downloading {interval} close price data for period '{period}'...")

    data = yf.download(
        tickers,
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=True,
        threads=DOWNLOAD_THREADING,
        progress=False,
    )

    if not hasattr(data.columns, "levels") or len(data.columns.levels) < 2:
        raise ValueError("Unexpected yfinance shape for multi-ticker download")

    valid_tickers = [t for t in tickers if t in data.columns.levels[0]]
    print(f"{len(valid_tickers)} tickers successfully downloaded.")

    close = pd.DataFrame({t: data[t]["Close"] for t in valid_tickers}).sort_index()

    # Keep columns with sufficient data
    keep = close.columns[close.notna().mean() >= min_valid_ratio]
    close = close[keep]
    print(f"{len(keep)} tickers remaining after removing mostly-empty columns.")

    close = close.ffill().bfill()
    returns = close.pct_change().dropna(how="all")
    print(f"Returns shape: {returns.shape}")

    corr = returns.corr()
    print("Correlation matrix computed.\n")

    # --- fetch company names ---
    print("Fetching company names to filter duplicates...")
    company_names = fetch_company_names(list(close.columns))

    pairs = []
    cols = list(close.columns)
    for i, t1 in enumerate(cols):
        row = corr.loc[t1]
        for t2 in cols[i+1:]:
            c = float(row[t2])
            if abs(c) >= correlation_threshold:
                if is_same_company(company_names[t1], company_names[t2]):
                    print(f"Skipping pair {t1}-{t2} (same company match)")
                    continue
                pairs.append((t1, t2, c))

    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    print(f"Total highly correlated pairs found (corr >= {correlation_threshold}): {len(pairs)}\n")

    # Preview
    for t1, t2, c in pairs[:10]:
        print(f"{t1} <-> {t2} : correlation = {c:.4f}")

    return pairs
