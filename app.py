"""
Diagnostic — Run on Streamlit Cloud to test data pipeline.
Temporarily replace app.py with this, push, check output, then revert.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
import traceback

st.set_page_config(page_title="Diagnostic", layout="wide")
st.markdown("## 🔧 Data Pipeline Diagnostic")

ET = pytz.timezone("US/Eastern")
et_now = datetime.now(ET)
st.write(f"**ET time:** {et_now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
st.write(f"**Weekday:** {et_now.strftime('%A')}")
st.write(f"**Python:** {__import__('sys').version}")
st.write(f"**NumPy:** {np.__version__}")

# ═══════════════════════════════════════
# Test 1: Barchart Session
# ═══════════════════════════════════════
st.markdown("---")
st.markdown("### 1. Barchart Session")
try:
    from fetch import create_session, fetch_spot, fetch_spot_tv, get_expiries, fetch_full_chain
    sess, headers = create_session()
    st.success(f"✅ Session created. XSRF token: {headers.get('x-xsrf-token', 'MISSING')[:20]}...")
except Exception as e:
    st.error(f"❌ Session failed: {e}")
    st.code(traceback.format_exc())
    st.stop()

# ═══════════════════════════════════════
# Test 2: Spot Price
# ═══════════════════════════════════════
st.markdown("---")
st.markdown("### 2. Spot Price")

spot_tv = fetch_spot_tv()
st.write(f"**tvdatafeed TVC:SPX:** {spot_tv}")

from fetch import fetch_spot_barchart
spot_bc = fetch_spot_barchart(sess, headers)
st.write(f"**Barchart $SPX:** {spot_bc}")

spot = spot_tv or spot_bc
if spot and spot > 1000:
    st.success(f"✅ Spot = {spot:,.2f}")
else:
    st.error(f"❌ No valid spot price")
    st.stop()

# ═══════════════════════════════════════
# Test 3: Expiry Discovery
# ═══════════════════════════════════════
st.markdown("---")
st.markdown("### 3. Expiry Discovery")
try:
    weekly, monthly = get_expiries(sess, headers)
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    st.write(f"**All weekly ({len(weekly)}):** {weekly[:10]}...")
    st.write(f"**All monthly ({len(monthly)}):** {monthly[:5]}")
    
    future_weekly = [e for e in weekly if e >= today_str]
    future_monthly = [e for e in monthly if e >= today_str]
    
    st.write(f"**Future weekly ({len(future_weekly)}):** {future_weekly[:8]}")
    st.write(f"**Future monthly ({len(future_monthly)}):** {future_monthly[:5]}")
    
    nearest = future_weekly[0] if future_weekly else None
    st.write(f"**Nearest (0DTE candidate):** {nearest}")
    st.write(f"**Today is monthly OPEX?** {today_str in monthly}")
    
    if nearest == today_str:
        st.warning(f"⚠️ Nearest expiry IS today ({today_str}). DTE will be 0.")
    
    st.success("✅ Expiries loaded")
except Exception as e:
    st.error(f"❌ Expiry discovery failed: {e}")
    st.code(traceback.format_exc())
    st.stop()

# ═══════════════════════════════════════
# Test 4: Chain Fetch — Nearest (0DTE)
# ═══════════════════════════════════════
st.markdown("---")
st.markdown(f"### 4. Chain Fetch — {nearest} (nearest)")
try:
    calls_0, puts_0 = fetch_full_chain(sess, headers, nearest, is_dense=False)
    st.write(f"**Calls:** {len(calls_0)} rows")
    st.write(f"**Puts:** {len(puts_0)} rows")
    
    if not calls_0.empty:
        st.write(f"**Call columns:** {list(calls_0.columns)}")
        st.write(f"**DTE values (calls):** {calls_0['daysToExpiration'].unique().tolist()}")
        
        # ATM slice
        atm_calls = calls_0[abs(calls_0["strikePrice"] - spot) < 30].head(5)
        st.write("**ATM calls (±30 pts):**")
        st.dataframe(atm_calls[["strikePrice", "openInterest", "volume", "gamma", 
                                 "volatility", "daysToExpiration"]].head(5))
        
        # Check for volume field
        if "volume" in calls_0.columns:
            total_vol = calls_0["volume"].sum()
            st.write(f"**Total call volume:** {total_vol:,.0f}")
        else:
            st.warning("⚠️ No 'volume' column in calls")
    
    if not puts_0.empty:
        st.write(f"**DTE values (puts):** {puts_0['daysToExpiration'].unique().tolist()}")
        atm_puts = puts_0[abs(puts_0["strikePrice"] - spot) < 30].head(5)
        st.write("**ATM puts (±30 pts):**")
        st.dataframe(atm_puts[["strikePrice", "openInterest", "volume", "gamma",
                                "volatility", "daysToExpiration", "delta"]].head(5))
        
        # Sanity: put delta should be negative
        if not atm_puts.empty:
            d = atm_puts.iloc[0]["delta"]
            if d < 0:
                st.success(f"✅ Put delta = {d:.4f} (negative, correct)")
            else:
                st.error(f"❌ Put delta = {d:.4f} (should be negative!)")
    
    st.success(f"✅ {nearest} chain loaded: {len(calls_0)}C + {len(puts_0)}P")
except Exception as e:
    st.error(f"❌ Chain fetch failed: {e}")
    st.code(traceback.format_exc())

# ═══════════════════════════════════════
# Test 5: Chain Fetch — First Monthly
# ═══════════════════════════════════════
if future_monthly:
    first_monthly = future_monthly[0]
    st.markdown("---")
    st.markdown(f"### 5. Chain Fetch — {first_monthly} (first monthly)")
    try:
        calls_m, puts_m = fetch_full_chain(sess, headers, first_monthly, is_dense=True)
        st.write(f"**Calls:** {len(calls_m)} rows, **Puts:** {len(puts_m)} rows")
        
        if not calls_m.empty:
            st.write(f"**DTE values:** {calls_m['daysToExpiration'].unique().tolist()}")
            st.write(f"**Strike range:** {calls_m['strikePrice'].min():.0f} - {calls_m['strikePrice'].max():.0f}")
            
            # OI check
            oi_nonzero = calls_m[calls_m["openInterest"] > 0]
            st.write(f"**Calls with OI > 0:** {len(oi_nonzero)} / {len(calls_m)}")
            
        st.success(f"✅ {first_monthly} chain loaded")
    except Exception as e:
        st.error(f"❌ Monthly chain failed: {e}")
        st.code(traceback.format_exc())

# ═══════════════════════════════════════
# Test 6: GEX Computation
# ═══════════════════════════════════════
st.markdown("---")
st.markdown("### 6. GEX Computation")

try:
    from compute import compute_gex_vex, compute_gex_plus_at_spot, bsm_gamma
    
    # Test BSM with DTE=0
    g_dte0 = bsm_gamma(spot, spot, 1/(365*24), 0.05, 0.20)
    g_dte1 = bsm_gamma(spot, spot, 1/365, 0.05, 0.20)
    st.write(f"**BSM gamma at DTE=0 (T=1hr):** {g_dte0:.6f}")
    st.write(f"**BSM gamma at DTE=1:** {g_dte1:.6f}")
    
    if not calls_0.empty and not puts_0.empty:
        gex_df = compute_gex_vex(calls_0, puts_0, spot)
        st.write(f"**GEX result:** {len(gex_df)} strikes")
        st.write(f"**Total GEX:** {gex_df['gex'].sum():.2f}M")
        st.write(f"**Total VEX:** {gex_df['vex'].sum():.2f}M")
        st.write(f"**Total GEX+:** {gex_df['gex_plus'].sum():.2f}M")
        
        # Check for zero results
        nonzero = gex_df[gex_df["gex_plus"] != 0]
        st.write(f"**Non-zero GEX+ strikes:** {len(nonzero)} / {len(gex_df)}")
        
        if len(nonzero) == 0:
            st.error("❌ ALL GEX values are zero! BSM is skipping every strike.")
            
            # Diagnose why
            st.markdown("**Diagnosing...**")
            sample = calls_0[calls_0["openInterest"] > 0].head(3)
            for _, row in sample.iterrows():
                K = row["strikePrice"]
                oi = row["openInterest"]
                iv = row["volatility"]
                dte = row["daysToExpiration"]
                st.write(f"  Strike {K}: OI={oi}, IV={iv}, DTE={dte}")
                st.write(f"    → oi>0: {oi>0}, iv>0: {iv>0}, dte>0: {dte>0}, dte<=0: {dte<=0}")
                if dte <= 0:
                    st.warning(f"    → DTE={dte} triggers 'dte <= 0' skip! This is the bug.")
                    st.info(f"    → FIX: Change 'dte <= 0' to 'dte < 0' in compute.py")
        else:
            # Top strikes
            top = nonzero.nlargest(5, "gex_plus")
            st.write("**Top 5 GEX+ strikes:**")
            st.dataframe(top[["strike", "gex", "vex", "gex_plus"]].reset_index(drop=True))
            st.success("✅ GEX computation working")
    
    # Monthly GEX+ at spot
    if future_monthly and not calls_m.empty and not puts_m.empty:
        gex_m, vex_m, gp_m = compute_gex_plus_at_spot(calls_m, puts_m, spot)
        st.write(f"**Monthly GEX+ at spot:** GEX={gex_m:.1f}M, VEX={vex_m:.1f}M, GEX+={gp_m:.1f}M")
        if gp_m == 0:
            st.warning("⚠️ Monthly GEX+ is zero — check DTE filter in compute_gex_plus_at_spot")

except Exception as e:
    st.error(f"❌ GEX computation failed: {e}")
    st.code(traceback.format_exc())

# ═══════════════════════════════════════
# Test 7: TV 1-min Bars
# ═══════════════════════════════════════
st.markdown("---")
st.markdown("### 7. tvdatafeed 1-min bars")
try:
    from tvDatafeed import TvDatafeed, Interval
    tv = TvDatafeed()
    df = tv.get_hist(symbol="SPX", exchange="TVC", interval=Interval.in_1_minute, n_bars=10)
    if df is not None and not df.empty:
        st.write(f"**Rows:** {len(df)}")
        st.write(f"**Latest bar:** {df.index[-1]} → close={df['close'].iloc[-1]:.2f}")
        st.success("✅ TV 1-min bars working")
    else:
        st.warning("⚠️ TV returned empty — will use snapshot spots as fallback")
except Exception as e:
    st.warning(f"⚠️ TV bars failed: {e}")

# ═══════════════════════════════════════
# Summary
# ═══════════════════════════════════════
st.markdown("---")
st.markdown("### Summary")
st.markdown(f"""
- Spot: **{spot:,.2f}**
- Nearest expiry: **{nearest}** (DTE = {calls_0['daysToExpiration'].iloc[0] if not calls_0.empty else 'N/A'})
- Monthly OPEX today: **{today_str in monthly}**
- 0DTE chain: **{len(calls_0)}C + {len(puts_0)}P**
- Volume field present: **{'volume' in calls_0.columns if not calls_0.empty else 'N/A'}**
""")
