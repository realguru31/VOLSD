"""
All computations: GEX/VEX, BSM greeks, KDE gradient, crash risk,
risk surface heatmap, Breeden-Litzenberger forecast, charm decomposition.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


# ═══════════════════════════════════════
# Black-Scholes Greeks
# ═══════════════════════════════════════

def bsm_gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bsm_vega(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T) / 100


def bsm_charm(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    charm = -pdf_d1 * (2 * r * T - d2 * sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T))
    if option_type == "put":
        charm = charm + r * np.exp(-r * T)
    return charm


# ═══════════════════════════════════════
# GEX / VEX / GEX+ per Strike
# ═══════════════════════════════════════

def compute_gex_vex(calls, puts, spot, mult=100):
    """Compute GEX, VEX, GEX+ per strike from live chain data."""
    call_k = set(calls["strikePrice"].tolist()) if not calls.empty else set()
    put_k = set(puts["strikePrice"].tolist()) if not puts.empty else set()
    all_k = sorted(call_k | put_k)
    results = []
    for K in all_k:
        cr = calls[calls["strikePrice"] == K]
        pr = puts[puts["strikePrice"] == K]
        c_oi = float(cr["openInterest"].iloc[0]) if not cr.empty else 0
        c_g = float(cr["gamma"].iloc[0]) if not cr.empty else 0
        c_v = float(cr["vega"].iloc[0]) if not cr.empty else 0
        p_oi = float(pr["openInterest"].iloc[0]) if not pr.empty else 0
        p_g = float(pr["gamma"].iloc[0]) if not pr.empty else 0
        p_v = float(pr["vega"].iloc[0]) if not pr.empty else 0
        p_d = float(pr["delta"].iloc[0]) if not pr.empty else 0

        cg = c_oi * c_g * mult * spot**2 / 1e6
        pg = -p_oi * p_g * mult * spot**2 / 1e6
        cv = c_oi * c_v * mult / 1e6
        pv = -p_oi * p_v * mult / 1e6
        gex = cg + pg
        vex = cv + pv
        results.append({
            "strike": K, "gex": gex, "vex": vex, "gex_plus": gex + vex,
            "call_gex": cg, "put_gex": pg, "call_vex": cv, "put_vex": pv,
            "npd_contrib": p_oi * p_d * mult,
            "call_oi": c_oi, "put_oi": p_oi,
        })
    return pd.DataFrame(results)


def find_zero_gamma(gex_df, spot, pct_range=0.10):
    """Find zero-gamma crossing nearest to spot."""
    near = gex_df[(gex_df["strike"] > spot * (1 - pct_range)) &
                  (gex_df["strike"] < spot * (1 + pct_range))]
    if len(near) < 2:
        return None
    v = near["gex_plus"].values
    s = near["strike"].values
    for i in range(len(v) - 1):
        if v[i] * v[i + 1] < 0:
            return s[i] + (s[i + 1] - s[i]) * abs(v[i]) / (abs(v[i]) + abs(v[i + 1]))
    return None


# ═══════════════════════════════════════
# GEX+ at hypothetical spot (row-by-row, for crash risk)
# ═══════════════════════════════════════

def compute_gex_plus_at_spot(calls_df, puts_df, hypo_spot, r=0.05):
    """Recompute GEX+ at a hypothetical spot level using BSM."""
    total_gex, total_vex = 0.0, 0.0
    for _, row in calls_df.iterrows():
        K, oi, dte = row["strikePrice"], row["openInterest"], row["daysToExpiration"]
        iv = row["volatility"]
        if iv <= 0:
            iv = row.get("averageVolatility", 0)
        if oi <= 0 or iv <= 0 or dte < 0:
            continue
        T = max(dte / 365.0, 1 / (365 * 24))
        total_gex += oi * bsm_gamma(hypo_spot, K, T, r, iv) * 100 * hypo_spot**2 / 1e6
        total_vex += oi * bsm_vega(hypo_spot, K, T, r, iv) * 100 / 1e6
    for _, row in puts_df.iterrows():
        K, oi, dte = row["strikePrice"], row["openInterest"], row["daysToExpiration"]
        iv = row["volatility"]
        if iv <= 0:
            iv = row.get("averageVolatility", 0)
        if oi <= 0 or iv <= 0 or dte < 0:
            continue
        T = max(dte / 365.0, 1 / (365 * 24))
        total_gex -= oi * bsm_gamma(hypo_spot, K, T, r, iv) * 100 * hypo_spot**2 / 1e6
        total_vex -= oi * bsm_vega(hypo_spot, K, T, r, iv) * 100 / 1e6
    return total_gex, total_vex, total_gex + total_vex


# ═══════════════════════════════════════
# GEX+ Profile (sweep spot levels)
# ═══════════════════════════════════════

def build_gex_profile(calls, puts, spot, pct_range=0.15, n_points=50):
    spots = np.linspace(spot * (1 - pct_range), spot * (1 + pct_range), n_points)
    results = []
    for s in spots:
        gex, vex, gp = compute_gex_plus_at_spot(calls, puts, s)
        results.append({
            "spot_hypo": s, "pct_move": (s / spot - 1) * 100,
            "gex": gex, "vex": vex, "gex_plus": gp,
        })
    return pd.DataFrame(results)


# ═══════════════════════════════════════
# Risk Surface — VECTORIZED (fast)
# ═══════════════════════════════════════

def _prep_arrays(df, spot):
    """Pre-extract numpy arrays from DataFrame, filtered to relevant strikes."""
    d = df[(df["openInterest"] > 0) &
           (df["strikePrice"] > spot * 0.80) &
           (df["strikePrice"] < spot * 1.20)].copy()
    if d.empty:
        return np.array([]), np.array([]), np.array([]), np.array([])
    iv = d["volatility"].values.copy()
    if "averageVolatility" in d.columns:
        avg_iv = d["averageVolatility"].values
    else:
        avg_iv = np.full(len(d), 0.20)
    mask = iv <= 0
    iv[mask] = avg_iv[mask]
    iv[iv <= 0] = 0.20
    return (d["strikePrice"].values, d["openInterest"].values,
            iv, d["daysToExpiration"].values)


def build_risk_surface(calls, puts, spot, spot_pcts, iv_shocks, r=0.05):
    """Vectorized risk surface: rows=IV shocks, cols=spot moves."""
    c_K, c_OI, c_IV, c_DTE = _prep_arrays(calls, spot)
    p_K, p_OI, p_IV, p_DTE = _prep_arrays(puts, spot)

    surface = np.zeros((len(iv_shocks), len(spot_pcts)))

    for i, iv_sh in enumerate(iv_shocks):
        for j, sp_pct in enumerate(spot_pcts):
            S = spot * (1 + sp_pct)
            total = 0.0

            if len(c_K) > 0:
                T = np.maximum(c_DTE / 365.0, 1 / (365 * 24))
                sig = np.maximum(c_IV + iv_sh, 0.01)
                d1 = (np.log(S / c_K) + (r + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
                pdf_d1 = norm.pdf(d1)
                gamma = pdf_d1 / (S * sig * np.sqrt(T))
                vega = S * pdf_d1 * np.sqrt(T) / 100
                total += np.sum(c_OI * gamma * 100 * S**2 / 1e6)
                total += np.sum(c_OI * vega * 100 / 1e6)

            if len(p_K) > 0:
                T = np.maximum(p_DTE / 365.0, 1 / (365 * 24))
                sig = np.maximum(p_IV + iv_sh, 0.01)
                d1 = (np.log(S / p_K) + (r + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
                pdf_d1 = norm.pdf(d1)
                gamma = pdf_d1 / (S * sig * np.sqrt(T))
                vega = S * pdf_d1 * np.sqrt(T) / 100
                total -= np.sum(p_OI * gamma * 100 * S**2 / 1e6)
                total -= np.sum(p_OI * vega * 100 / 1e6)

            surface[i, j] = total

    return surface


# ═══════════════════════════════════════
# 0DTE Gradient (KDE Gaussian)
# ═══════════════════════════════════════

def compute_raw_exposures(calls, puts, spot, r=0.05):
    """Raw net GEX and charm at each strike."""
    call_k = set(calls["strikePrice"].tolist()) if not calls.empty else set()
    put_k = set(puts["strikePrice"].tolist()) if not puts.empty else set()
    all_k = sorted(call_k | put_k)
    results = []
    for K in all_k:
        c = calls[calls["strikePrice"] == K]
        p = puts[puts["strikePrice"] == K]
        c_oi = float(c["openInterest"].iloc[0]) if not c.empty else 0
        c_g = float(c["gamma"].iloc[0]) if not c.empty else 0
        c_iv = float(c["volatility"].iloc[0]) if not c.empty else 0
        c_dte = float(c["daysToExpiration"].iloc[0]) if not c.empty else 0
        p_oi = float(p["openInterest"].iloc[0]) if not p.empty else 0
        p_g = float(p["gamma"].iloc[0]) if not p.empty else 0
        p_iv = float(p["volatility"].iloc[0]) if not p.empty else 0
        p_dte = float(p["daysToExpiration"].iloc[0]) if not p.empty else 0

        net_gex = (c_oi * c_g - p_oi * p_g) * 100 * spot**2 / 1e9

        c_ch, p_ch = 0.0, 0.0
        if c_oi > 0 and c_iv > 0 and c_dte >= 0:
            T = max(c_dte / 365.0, 1 / (365 * 24))
            c_ch = c_oi * bsm_charm(spot, K, T, r, c_iv, "call") * 100
        if p_oi > 0 and p_iv > 0 and p_dte >= 0:
            T = max(p_dte / 365.0, 1 / (365 * 24))
            p_ch = p_oi * bsm_charm(spot, K, T, r, p_iv, "put") * 100

        results.append({"strike": K, "net_gex": net_gex, "net_charm": -(c_ch - p_ch)})
    return results


def kde_field(strikes, values, price_grid, sigma=20):
    """KDE: each strike's exposure as a Gaussian bell, all overlapping and summed."""
    field = np.zeros(len(price_grid))
    for K, val in zip(strikes, values):
        if val == 0:
            continue
        field += val * np.exp(-0.5 * ((price_grid - K) / sigma) ** 2)
    return field


# ═══════════════════════════════════════
# Charm Decomposition per Expiry
# ═══════════════════════════════════════

def compute_charm_for_expiry(calls, puts, spot, r=0.05):
    """Total net charm for one expiry."""
    total = 0.0
    for _, row in calls.iterrows():
        K, oi = row["strikePrice"], row["openInterest"]
        iv, dte = row["volatility"], row["daysToExpiration"]
        if iv <= 0:
            iv = row.get("averageVolatility", 0)
        if oi > 0 and iv > 0 and dte >= 0:
            T = max(dte / 365.0, 1 / (365 * 24))
            total += oi * bsm_charm(spot, K, T, r, iv, "call") * 100
    for _, row in puts.iterrows():
        K, oi = row["strikePrice"], row["openInterest"]
        iv, dte = row["volatility"], row["daysToExpiration"]
        if iv <= 0:
            iv = row.get("averageVolatility", 0)
        if oi > 0 and iv > 0 and dte >= 0:
            T = max(dte / 365.0, 1 / (365 * 24))
            total -= oi * bsm_charm(spot, K, T, r, iv, "put") * 100
    return total


# ═══════════════════════════════════════
# Breeden-Litzenberger Forecast
# ═══════════════════════════════════════

def breeden_litzenberger(calls, puts, spot, r=0.05):
    """Extract risk-neutral density from OTM option prices."""
    if calls.empty or puts.empty:
        return None

    dte = calls["daysToExpiration"].iloc[0] if not calls.empty else 30
    T = max(dte / 365.0, 1 / (365 * 24))

    otm = []

    otm_puts = puts[puts["strikePrice"] < spot].copy()
    for _, row in otm_puts.iterrows():
        bid = row.get("bidPrice", 0)
        ask = row.get("askPrice", 0)
        mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else row.get("lastPrice", 0)
        if mid > 0:
            otm.append({"strike": row["strikePrice"], "price": mid, "type": "put"})

    otm_calls = calls[calls["strikePrice"] > spot].copy()
    for _, row in otm_calls.iterrows():
        bid = row.get("bidPrice", 0)
        ask = row.get("askPrice", 0)
        mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else row.get("lastPrice", 0)
        if mid > 0:
            otm.append({"strike": row["strikePrice"], "price": mid, "type": "call"})

    if len(otm) < 10:
        return None

    otm_df = pd.DataFrame(otm).sort_values("strike").reset_index(drop=True)
    strikes = otm_df["strike"].values
    prices = otm_df["price"].values

    call_equiv = np.zeros(len(otm_df))
    for i, row in otm_df.iterrows():
        if row["type"] == "put":
            call_equiv[i] = row["price"] + spot - row["strike"] * np.exp(-r * T)
        else:
            call_equiv[i] = row["price"]

    n = len(strikes)
    if n < 5:
        return None

    density_strikes = []
    density_vals = []
    for i in range(1, n - 1):
        dk1 = strikes[i] - strikes[i - 1]
        dk2 = strikes[i + 1] - strikes[i]
        if dk1 <= 0 or dk2 <= 0:
            continue
        dk_avg = (dk1 + dk2) / 2
        d2c = (call_equiv[i + 1] - 2 * call_equiv[i] + call_equiv[i - 1]) / (dk_avg ** 2)
        q = np.exp(r * T) * d2c
        if q > 0:
            density_strikes.append(strikes[i])
            density_vals.append(q)

    if len(density_strikes) < 5:
        return None

    density_strikes = np.array(density_strikes)
    density_vals = np.array(density_vals)

    total_area = np.trapezoid(density_vals, density_strikes)
    if total_area > 0:
        density_vals = density_vals / total_area

    cdf_vals = np.cumsum(density_vals * np.gradient(density_strikes))
    cdf_vals = np.clip(cdf_vals / max(cdf_vals[-1], 1e-10), 0, 1)

    mean = np.trapezoid(density_strikes * density_vals, density_strikes)
    var = np.trapezoid((density_strikes - mean)**2 * density_vals, density_strikes)
    std = np.sqrt(max(var, 0))
    if std > 0:
        skew = np.trapezoid((density_strikes - mean)**3 * density_vals, density_strikes) / std**3
        kurt = np.trapezoid((density_strikes - mean)**4 * density_vals, density_strikes) / std**4
    else:
        skew, kurt = 0, 3

    atm_calls = calls[abs(calls["strikePrice"] - spot) < 20]
    if not atm_calls.empty:
        atm_iv = atm_calls["volatility"].mean()
        if atm_iv <= 0:
            atm_iv = atm_calls["averageVolatility"].mean()
    else:
        atm_iv = 0.20
    if atm_iv <= 0:
        atm_iv = 0.20

    sigma_1d = spot * atm_iv * np.sqrt(1 / 252)
    sigma_1w = spot * atm_iv * np.sqrt(5 / 252)
    sigma_exp = spot * atm_iv * np.sqrt(T)

    def cf_pct(z, s, k):
        return z + (z**2 - 1) * s / 6 + (z**3 - 3*z) * (k - 3) / 24 - (2*z**3 - 5*z) * s**2 / 36

    z_vals = {5: -1.645, 25: -0.674, 50: 0, 75: 0.674, 95: 1.645}
    pctiles_exp = {}
    for pct, z in z_vals.items():
        pctiles_exp[pct] = spot + cf_pct(z, skew, kurt) * sigma_exp

    prob_table = []
    for K_level in range(int(spot * 0.90), int(spot * 1.10) + 1, 50):
        p_below = float(np.interp(K_level, density_strikes, cdf_vals))
        prob_table.append({
            "level": K_level, "p_below": p_below * 100, "p_above": (1 - p_below) * 100,
        })

    return {
        "density_strikes": density_strikes, "density_vals": density_vals,
        "mean": mean, "std": std, "skew": skew, "kurt": kurt,
        "atm_iv": atm_iv, "sigma_exp": sigma_exp,
        "pctiles_exp": pctiles_exp, "prob_table": prob_table,
        "dte": dte, "T": T,
    }
