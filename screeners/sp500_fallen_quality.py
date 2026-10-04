"""
Screen the S&P 500 for stocks that are down more than a threshold year-to-date
while their fundamentals still look healthy.

Usage:
    pip install yfinance pandas
    python sp500_fallen_quality.py --drop 0.20 --out fallen_quality.csv

Notes:
    - YTD return uses split/dividend-adjusted closes (auto_adjust=True), so stock
      splits (e.g. CRWD's 4-for-1 in July 2026) don't show up as fake crashes.
    - Spin-offs can still distort returns; check flagged names by hand.
    - "Good fundamentals" here is trailing data. It cannot tell you whether the
      market is right that future growth is impaired.
"""

import argparse
import datetime as dt

import pandas as pd
import yfinance as yf

SP500_CSV = ("https://raw.githubusercontent.com/datasets/"
             "s-and-p-500-companies/main/data/constituents.csv")

# Minimum bar for "fundamentals still good". Tune to taste.
CRITERIA = {
    "revenueGrowth":    lambda v: v is not None and v >= 0.08,   # >= 8% YoY
    "earningsGrowth":   lambda v: v is not None and v > 0,       # EPS still growing
    "operatingMargins": lambda v: v is not None and v >= 0.15,   # >= 15%
    "freeCashflow":     lambda v: v is not None and v > 0,       # FCF positive
    "debtToEquity":     lambda v: v is None or v < 150,          # yfinance reports in %
}


def get_constituents():
    df = pd.read_csv(SP500_CSV)
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
    return df[["Symbol", "Security", "GICS Sector"]]


def ytd_returns(symbols):
    start = dt.date(dt.date.today().year - 1, 12, 24)
    px = yf.download(symbols, start=start, auto_adjust=True,
                     progress=False, threads=True)["Close"]
    year_start = pd.Timestamp(dt.date.today().year, 1, 1)
    base = px[px.index < year_start].ffill().iloc[-1]
    last = px.ffill().iloc[-1]
    return (last / base - 1).rename("ytd")


def fundamentals(symbol):
    info = yf.Ticker(symbol).info
    keys = list(CRITERIA) + ["forwardPE", "trailingPE", "returnOnEquity",
                             "marketCap"]
    return {k: info.get(k) for k in keys}


def passes(row):
    return all(test(row.get(k)) for k, test in CRITERIA.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop", type=float, default=0.20,
                    help="minimum YTD decline, e.g. 0.20 for -20%%")
    ap.add_argument("--out", default="fallen_quality.csv")
    args = ap.parse_args()

    names = get_constituents()
    ytd = ytd_returns(names["Symbol"].tolist())
    fallen = names.join(ytd, on="Symbol")
    fallen = fallen[fallen["ytd"] <= -args.drop].sort_values("ytd")
    print(f"{len(fallen)} S&P 500 stocks down >= {args.drop:.0%} YTD")

    rows = []
    for _, r in fallen.iterrows():
        try:
            f = fundamentals(r["Symbol"])
        except Exception as e:  # noqa: BLE001 - keep going on bad tickers
            print(f"  skip {r['Symbol']}: {e}")
            continue
        rows.append({**r.to_dict(), **f, "passes": passes(f)})

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    cols = ["Symbol", "Security", "ytd", "revenueGrowth", "earningsGrowth",
            "operatingMargins", "forwardPE"]
    print(out[out["passes"]][cols].to_string(index=False))
    print(f"\nFull table (passes and fails) written to {args.out}")


if __name__ == "__main__":
    main()
