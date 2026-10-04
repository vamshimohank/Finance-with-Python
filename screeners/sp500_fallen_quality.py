"""
Screen the S&P 500 for stocks that are down more than a threshold year-to-date
while their fundamentals still look healthy.

Data sources (no third-party aggregators):
    - Prices: Nasdaq's public historical-quote API (covers NYSE-listed stocks too)
    - Fundamentals: SEC EDGAR XBRL "companyfacts" (as filed in 10-K / 10-Q)
    - Constituents: github.com/datasets/s-and-p-500-companies (includes CIKs)

Usage:
    pip install pandas requests
    python sp500_fallen_quality.py --email you@example.com --drop 0.20

Notes:
    - The SEC requires a User-Agent with a contact email and allows max 10 req/s.
    - A one-day drop of more than 45% is flagged as a possible unadjusted split
      or spin-off; check those names by hand.
    - Fundamentals use the latest annual (CYxxxx) and quarterly (CYxxxxQn)
      frames from EDGAR. Banks and insurers report differently and will
      mostly show as missing data rather than failing.
    - "Good fundamentals" here is trailing data. It cannot tell you whether the
      market is right that future growth is impaired.
"""

import argparse
import datetime as dt
import re
import time

import pandas as pd
import requests

SP500_CSV = ("https://raw.githubusercontent.com/datasets/"
             "s-and-p-500-companies/main/data/constituents.csv")
NASDAQ_HIST = "https://api.nasdaq.com/api/quote/{sym}/historical"
SEC_FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

NASDAQ_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.nasdaq.com",
    "Referer": "https://www.nasdaq.com/",
}

# First tag with data wins.
TAGS = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax",
                "Revenues", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "op_income": ["OperatingIncomeLoss"],
    "eps": ["EarningsPerShareDiluted", "EarningsPerShareBasic"],
    "ocf": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment",
              "PaymentsToAcquireProductiveAssets"],
    "debt": ["LongTermDebt", "LongTermDebtNoncurrent"],
}

# Minimum bar for "fundamentals still good". Tune to taste.
CRITERIA = {
    "rev_growth_q":  lambda v: v is not None and v >= 0.08,  # latest qtr YoY
    "eps_growth_fy": lambda v: v is not None and v > 0,      # last fiscal year
    "op_margin_fy":  lambda v: v is not None and v >= 0.15,
    "fcf_fy":        lambda v: v is not None and v > 0,
    "debt_to_oi":    lambda v: v is None or v < 4,           # years of op income
}


def get_constituents():
    df = pd.read_csv(SP500_CSV)
    return df[["Symbol", "Security", "GICS Sector", "CIK"]]


# ---------------------------------------------------------------- prices

def parse_nasdaq_rows(rows):
    """Nasdaq rows -> Series of closes indexed by date, oldest first."""
    s = pd.Series(
        {pd.to_datetime(r["date"], format="%m/%d/%Y"):
         float(r["close"].replace("$", "").replace(",", "")) for r in rows})
    return s.sort_index()


def fetch_closes(session, sym, start, end):
    params = {"assetclass": "stocks", "fromdate": start.isoformat(),
              "todate": end.isoformat(), "limit": 9999}
    r = session.get(NASDAQ_HIST.format(sym=sym.replace(".", "^")),
                    params=params, headers=NASDAQ_HEADERS, timeout=20)
    r.raise_for_status()
    table = ((r.json().get("data") or {}).get("tradesTable") or {})
    return parse_nasdaq_rows(table.get("rows") or [])


def ytd_stats(closes, year):
    prior = closes[closes.index < pd.Timestamp(year, 1, 1)]
    if prior.empty or len(closes) < 2:
        return None, None, False
    base, last = prior.iloc[-1], closes.iloc[-1]
    split_flag = bool((closes.pct_change() < -0.45).any())
    return last / base - 1, last, split_flag


# ---------------------------------------------------------- fundamentals

def frames(facts, concept_tags, unit):
    """Return {frame: value} for the first tag that has data in `unit`."""
    gaap = facts.get("facts", {}).get("us-gaap", {})
    for tag in concept_tags:
        entries = gaap.get(tag, {}).get("units", {}).get(unit, [])
        out = {e["frame"]: e["val"] for e in entries if e.get("frame")}
        if out:
            return out
    return {}


def latest_annual(fr):
    years = sorted(k for k in fr if re.fullmatch(r"CY\d{4}", k))
    return years[-1] if years else None


def growth(cur, prev):
    if cur is None or prev in (None, 0):
        return None
    return (cur - prev) / abs(prev)


def latest_quarter_growth(fr):
    qs = sorted(k for k in fr if re.fullmatch(r"CY\d{4}Q[1-4]", k))
    for q in reversed(qs):
        prior = f"CY{int(q[2:6]) - 1}{q[6:]}"
        if prior in fr:
            return q, growth(fr[q], fr[prior])
    return None, None


def latest_instant(fr):
    ks = sorted(k for k in fr if re.fullmatch(r"CY\d{4}Q[1-4]I", k))
    return fr[ks[-1]] if ks else None


def fundamentals(facts):
    rev = frames(facts, TAGS["revenue"], "USD")
    oi = frames(facts, TAGS["op_income"], "USD")
    eps = frames(facts, TAGS["eps"], "USD/shares")
    ocf = frames(facts, TAGS["ocf"], "USD")
    capex = frames(facts, TAGS["capex"], "USD")
    debt = frames(facts, TAGS["debt"], "USD")

    fy = latest_annual(rev)
    prev = f"CY{int(fy[2:]) - 1}" if fy else None
    q, rev_growth_q = latest_quarter_growth(rev)

    out = {"fiscal_frame": fy, "quarter_frame": q,
           "rev_growth_q": rev_growth_q,
           "rev_growth_fy": growth(rev.get(fy), rev.get(prev)),
           "eps_fy": eps.get(fy),
           "eps_growth_fy": growth(eps.get(fy), eps.get(prev)),
           "op_margin_fy": None, "fcf_fy": None, "debt_to_oi": None}
    if fy in rev and fy in oi and rev[fy]:
        out["op_margin_fy"] = oi[fy] / rev[fy]
    if fy in ocf:
        out["fcf_fy"] = ocf[fy] - capex.get(fy, 0)
    d = latest_instant(debt)
    if d is not None and oi.get(fy, 0) > 0:
        out["debt_to_oi"] = d / oi[fy]
    return out


def passes(row):
    return all(test(row.get(k)) for k, test in CRITERIA.items())


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True,
                    help="contact email for the SEC User-Agent header")
    ap.add_argument("--drop", type=float, default=0.20,
                    help="minimum YTD decline, e.g. 0.20 for -20%%")
    ap.add_argument("--out", default="fallen_quality.csv")
    args = ap.parse_args()

    today = dt.date.today()
    start = dt.date(today.year - 1, 12, 20)
    names = get_constituents()
    session = requests.Session()

    price_rows = []
    for _, r in names.iterrows():
        try:
            closes = fetch_closes(session, r["Symbol"], start, today)
            ytd, last, split_flag = ytd_stats(closes, today.year)
        except Exception as e:  # noqa: BLE001 - keep going on bad tickers
            print(f"  price skip {r['Symbol']}: {e}")
            continue
        price_rows.append({**r.to_dict(), "ytd": ytd, "price": last,
                           "split_or_spinoff_flag": split_flag})
        time.sleep(0.2)

    prices = pd.DataFrame(price_rows).dropna(subset=["ytd"])
    fallen = prices[prices["ytd"] <= -args.drop].sort_values("ytd")
    print(f"{len(fallen)} of {len(prices)} priced S&P 500 stocks "
          f"down >= {args.drop:.0%} YTD")

    sec_headers = {"User-Agent": f"sp500-screener {args.email}"}
    rows = []
    for _, r in fallen.iterrows():
        try:
            resp = session.get(SEC_FACTS.format(cik=int(r["CIK"])),
                               headers=sec_headers, timeout=30)
            resp.raise_for_status()
            f = fundamentals(resp.json())
        except Exception as e:  # noqa: BLE001
            print(f"  SEC skip {r['Symbol']}: {e}")
            continue
        pe = (r["price"] / f["eps_fy"]
              if f["eps_fy"] and f["eps_fy"] > 0 else None)
        rows.append({**r.to_dict(), **f, "trailing_pe": pe,
                     "passes": passes(f)})
        time.sleep(0.15)

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    cols = ["Symbol", "Security", "ytd", "rev_growth_q", "eps_growth_fy",
            "op_margin_fy", "trailing_pe", "split_or_spinoff_flag"]
    print(out[out["passes"]][cols].to_string(index=False))
    print(f"\nFull table (passes and fails) written to {args.out}")


if __name__ == "__main__":
    main()
