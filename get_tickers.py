import requests

def get_tickers():
    """
    Scrape large/liquid US stocks from Nasdaq screener (cap >= $1B).
    Returns a sorted list of ticker symbols.
    """
    url = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=3000&offset=0&download=true"
    headers = {"User-Agent": "Mozilla/5.0"}

    print("Fetching Nasdaq screener data...")
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()

    rows = data["data"]["rows"]
    tickers = []

    for row in rows:
        mcap = row.get("marketCap")
        if not mcap:
            continue
        s = str(mcap).upper().replace(",", "").strip()
        try:
            if s.endswith("B"):
                cap = float(s[:-1]) * 1_000_000_000
            elif s.endswith("M"):
                cap = float(s[:-1]) * 1_000_000
            elif s.endswith("K"):
                cap = float(s[:-1]) * 1_000
            else:
                cap = float(s)
        except ValueError:
            continue

        if cap >= 1_000_000_000:
            t = row["symbol"].strip().replace("/", "-")
            tickers.append(t)

    tickers = sorted(set(tickers))
    print(f"\nTotal tickers found: {len(tickers)}")
    return tickers
