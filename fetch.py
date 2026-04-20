"""
Data fetching: Barchart options chains + tvdatafeed/Barchart spot price.
"""

import requests
import pandas as pd
from urllib.parse import unquote

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

BC_FIELDS = (
    "symbol,baseSymbol,strikePrice,lastPrice,volatility,delta,gamma,"
    "theta,vega,rho,volume,openInterest,optionType,daysToExpiration,"
    "expirationDate,tradeTime,averageVolatility,historicVolatility30d,"
    "bidPrice,askPrice"
)

API_URL = "https://www.barchart.com/proxies/core-api/v1/options/get"
QUOTE_URL = "https://www.barchart.com/proxies/core-api/v1/quotes/get"


def safe_float(val, default=0.0):
    if val is None:
        return default
    try:
        return float(str(val).replace(",", ""))
    except (ValueError, TypeError):
        return default


# ═══════════════════════════════════════
# Spot Price
# ═══════════════════════════════════════

def fetch_spot_tv():
    """Primary: tvdatafeed TVC:SPX, no credentials."""
    try:
        from tvDatafeed import TvDatafeed, Interval
        tv = TvDatafeed()
        df = tv.get_hist(symbol="SPX", exchange="TVC",
                         interval=Interval.in_5_minute, n_bars=1)
        if df is not None and not df.empty:
            return float(df["close"].iloc[-1])
    except Exception:
        pass
    return None


def fetch_spot_barchart(sess, headers):
    """Fallback: Barchart quotes API."""
    try:
        r = sess.get(QUOTE_URL,
                     params={"symbols": "$SPX", "fields": "lastPrice,previousClose"},
                     headers=headers, timeout=10)
        r.raise_for_status()
        items = r.json().get("data", [])
        if items:
            raw = items[0].get("raw", items[0])
            return safe_float(raw.get("lastPrice"))
    except Exception:
        pass
    return None


def fetch_spot(sess=None, headers=None):
    """Get SPX spot: tvdatafeed first, Barchart fallback."""
    spot = fetch_spot_tv()
    if spot and spot > 1000:
        return spot
    if sess and headers:
        spot = fetch_spot_barchart(sess, headers)
        if spot and spot > 1000:
            return spot
    return None


# ═══════════════════════════════════════
# Barchart Session
# ═══════════════════════════════════════

def create_session():
    """Create Barchart session with XSRF token."""
    page_url = "https://www.barchart.com/indices/quotes/$SPX/volatility-greeks"
    sess = requests.Session()
    r = sess.get(page_url, params={"page": "all"}, headers={
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "accept-encoding": "gzip, deflate, br",
        "accept-language": "en-US,en;q=0.9",
        "cache-control": "max-age=0",
        "upgrade-insecure-requests": "1",
        "user-agent": _UA,
    }, timeout=15)
    r.raise_for_status()
    cookies = sess.cookies.get_dict()
    if "XSRF-TOKEN" not in cookies:
        raise ConnectionError("No XSRF-TOKEN in Barchart cookies")
    xsrf = unquote(cookies["XSRF-TOKEN"])
    headers = {
        "accept": "application/json",
        "accept-encoding": "gzip, deflate, br",
        "accept-language": "en-US,en;q=0.9",
        "referer": page_url,
        "user-agent": _UA,
        "x-xsrf-token": xsrf,
    }
    return sess, headers


# ═══════════════════════════════════════
# Expiry Discovery
# ═══════════════════════════════════════

def get_expiries(sess, headers):
    """Get weekly + monthly expiry lists from meta=expirations."""
    params = {
        "baseSymbol": "$SPX", "type": "Call",
        "fields": "strikePrice,expirationDate",
        "meta": "field.shortName,expirations,field.description",
        "raw": "1",
    }
    r = sess.get(API_URL, params=params, headers=headers, timeout=15)
    r.raise_for_status()
    meta = r.json().get("meta", {})
    exp_data = meta.get("expirations", {})
    return exp_data.get("weekly", []), exp_data.get("monthly", [])


# ═══════════════════════════════════════
# Chain Fetch (groupBy=optionType)
# ═══════════════════════════════════════

def _to_df(rows):
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    num_cols = ["strikePrice", "lastPrice", "volatility", "delta", "gamma",
                "theta", "vega", "rho", "volume", "openInterest",
                "bidPrice", "askPrice", "daysToExpiration",
                "averageVolatility", "historicVolatility30d"]
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: safe_float(x))
    return df


def _parse_grouped(data):
    """Parse groupBy=optionType response into separate call/put DataFrames."""
    calls_list, puts_list = [], []
    if isinstance(data, dict):
        for opt_type, rows in data.items():
            if not isinstance(rows, list):
                continue
            for item in rows:
                raw = item.get("raw", item)
                raw["optionType"] = opt_type
                if opt_type == "Call":
                    calls_list.append(raw)
                elif opt_type == "Put":
                    puts_list.append(raw)
    elif isinstance(data, list):
        for item in data:
            raw = item.get("raw", item)
            ot = raw.get("optionType", "")
            if ot == "Call":
                calls_list.append(raw)
            elif ot == "Put":
                puts_list.append(raw)
    return _to_df(calls_list), _to_df(puts_list)


def fetch_chain_grouped(sess, headers, expiry, order_dir="asc"):
    """Fetch chain using groupBy=optionType."""
    params = {
        "baseSymbol": "$SPX", "groupBy": "optionType",
        "expirationDate": expiry, "fields": BC_FIELDS,
        "orderBy": "strikePrice", "orderDir": order_dir,
        "raw": "1", "meta": "field.shortName,field.description",
    }
    r = sess.get(API_URL, params=params, headers=headers, timeout=15)
    if r.status_code != 200:
        return pd.DataFrame(), pd.DataFrame()
    return _parse_grouped(r.json().get("data", {}))


def fetch_full_chain(sess, headers, expiry, is_dense=False):
    """
    Fetch full chain. Dense expiries (monthlies): split asc+desc to beat 1000-row cap.
    Small expiries (0DTE): single fetch.
    """
    if is_dense:
        calls_asc, puts_asc = fetch_chain_grouped(sess, headers, expiry, "asc")
        calls_desc, puts_desc = fetch_chain_grouped(sess, headers, expiry, "desc")
        all_calls = pd.concat([calls_asc, calls_desc], ignore_index=True)
        all_puts = pd.concat([puts_asc, puts_desc], ignore_index=True)
        if not all_calls.empty:
            all_calls = all_calls.drop_duplicates(subset=["strikePrice"], keep="first")
            all_calls = all_calls.sort_values("strikePrice").reset_index(drop=True)
        if not all_puts.empty:
            all_puts = all_puts.drop_duplicates(subset=["strikePrice"], keep="first")
            all_puts = all_puts.sort_values("strikePrice").reset_index(drop=True)
    else:
        all_calls, all_puts = fetch_chain_grouped(sess, headers, expiry, "asc")
    return all_calls, all_puts


def pick_nearest_expiry(future_weekly, monthly_list):
    """
    Pick the right 0DTE expiry. On monthly OPEX day, today's chain is the
    expired AM monthly (wide strikes, no ATM) — skip it and use next weekly.
    """
    if not future_weekly:
        return None
    candidate = future_weekly[0]
    # If today is monthly OPEX, the AM chain is dead — use next weekly
    if candidate in monthly_list and len(future_weekly) > 1:
        return future_weekly[1]
    return candidate
