"""
SPX Dealer Risk Monitor — Streamlit App
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import pytz

from fetch import (create_session, fetch_spot, get_expiries, fetch_full_chain)
from compute import (compute_gex_vex, find_zero_gamma, build_gex_profile,
                     compute_gex_plus_at_spot, build_risk_surface,
                     compute_raw_exposures, kde_field,
                     compute_charm_for_expiry, breeden_litzenberger)

st.set_page_config(page_title="SPX Dealer Risk", layout="wide", page_icon="📊")

# ═══════════════════════════════════════
# Password Auth
# ═══════════════════════════════════════
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("<h2 style='text-align:center;'>🔒 SPX Dealer Risk Monitor</h2>",
                unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        pwd = st.text_input("Enter password", type="password", key="pwd_input")
        if pwd:
            if pwd == st.secrets.get("APP_PASSWORD", ""):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password")
    st.stop()

# ═══════════════════════════════════════
# Theme
# ═══════════════════════════════════════
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

def get_theme():
    if st.session_state.dark_mode:
        return {
            "template": "plotly_dark",
            "bg": "#0a0a0a", "card_bg": "#141414", "text": "#e0e0e0",
            "accent": "#00FFAA", "red": "#CC3333", "green": "#33AA33",
            "blue": "#3366CC", "gold": "#CC8800", "muted": "#666666",
        }
    else:
        return {
            "template": "plotly_white",
            "bg": "#ffffff", "card_bg": "#f5f5f5", "text": "#1a1a1a",
            "accent": "#008866", "red": "#CC2222", "green": "#228822",
            "blue": "#2255BB", "gold": "#AA7700", "muted": "#999999",
        }

theme = get_theme()

# CSS
if st.session_state.dark_mode:
    st.markdown("""<style>
        .stApp { background-color: #0a0a0a; }
        .metric-card { background: #141414; border: 1px solid #222; border-radius: 8px;
                       padding: 12px 16px; text-align: center; }
        .metric-label { color: #888; font-size: 11px; text-transform: uppercase; font-family: monospace; }
        .metric-value { color: #e0e0e0; font-size: 22px; font-weight: bold; font-family: monospace; }
        .metric-sub { color: #555; font-size: 10px; font-family: monospace; }
        .regime-badge { padding: 4px 12px; border-radius: 4px; font-weight: bold; font-family: monospace; display: inline-block; }
        .regime-amp { background: #331111; color: #FF4444; border: 1px solid #FF4444; }
        .regime-damp { background: #113311; color: #44FF44; border: 1px solid #44FF44; }
        .crash-card { background: #141414; border: 1px solid #333; border-radius: 8px; padding: 16px; margin: 8px 0; }
        .crash-elevated { border-left: 4px solid #FF4444; }
        .crash-neutral { border-left: 4px solid #888; }
        .crash-contained { border-left: 4px solid #4444FF; }
    </style>""", unsafe_allow_html=True)
else:
    st.markdown("""<style>
        .metric-card { background: #f5f5f5; border: 1px solid #ddd; border-radius: 8px;
                       padding: 12px 16px; text-align: center; }
        .metric-label { color: #666; font-size: 11px; text-transform: uppercase; font-family: monospace; }
        .metric-value { color: #1a1a1a; font-size: 22px; font-weight: bold; font-family: monospace; }
        .metric-sub { color: #999; font-size: 10px; font-family: monospace; }
        .regime-badge { padding: 4px 12px; border-radius: 4px; font-weight: bold; font-family: monospace; display: inline-block; }
        .regime-amp { background: #ffeeee; color: #CC2222; border: 1px solid #CC2222; }
        .regime-damp { background: #eeffee; color: #228822; border: 1px solid #228822; }
        .crash-card { background: #f9f9f9; border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 8px 0; }
        .crash-elevated { border-left: 4px solid #CC2222; }
        .crash-neutral { border-left: 4px solid #888; }
        .crash-contained { border-left: 4px solid #2255BB; }
    </style>""", unsafe_allow_html=True)


# ═══════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════

@st.cache_data(ttl=300)
def load_all_data():
    sess, headers = create_session()
    spot = fetch_spot(sess, headers)
    if not spot:
        return None

    weekly, monthly = get_expiries(sess, headers)
    today_str = datetime.now().strftime("%Y-%m-%d")
    future_weekly = [e for e in weekly if e >= today_str]
    future_monthly = [e for e in monthly if e >= today_str]

    nearest = future_weekly[0] if future_weekly else (future_monthly[0] if future_monthly else None)
    monthlies = future_monthly[:3]

    chains = {}

    if nearest:
        c, p = fetch_full_chain(sess, headers, nearest, is_dense=False)
        if not c.empty and not p.empty:
            dte = (datetime.strptime(nearest, "%Y-%m-%d") - datetime.now()).days
            chains[nearest] = {"calls": c, "puts": p, "label": f"0DTE ({dte}d)"}

    for exp in monthlies:
        c, p = fetch_full_chain(sess, headers, exp, is_dense=True)
        if not c.empty and not p.empty:
            dte = (datetime.strptime(exp, "%Y-%m-%d") - datetime.now()).days
            chains[exp] = {"calls": c, "puts": p, "label": f"Monthly ({dte}d)"}

    return {
        "spot": spot, "nearest": nearest, "monthlies": monthlies,
        "chains": chains,
        "timestamp": datetime.now(pytz.timezone("US/Eastern")).strftime("%I:%M %p ET"),
    }


with st.spinner("Loading SPX options data..."):
    data = load_all_data()

if data is None:
    st.error("Failed to load data. Check network connection.")
    st.stop()

spot = data["spot"]
chains = data["chains"]
nearest_exp = data["nearest"]
target_monthlies = data["monthlies"]
ts = data["timestamp"]

# Pre-compute GEX
gex_data = {}
for exp, chain in chains.items():
    gex_data[exp] = compute_gex_vex(chain["calls"], chain["puts"], spot)

front_exp = target_monthlies[0] if target_monthlies else nearest_exp
if front_exp not in chains:
    front_exp = list(chains.keys())[0] if chains else None

if front_exp is None:
    st.error("No chain data available.")
    st.stop()

front_gex = gex_data.get(front_exp, pd.DataFrame())
front_calls = chains[front_exp]["calls"]
front_puts = chains[front_exp]["puts"]


# ═══════════════════════════════════════
# Header
# ═══════════════════════════════════════
hdr_left, hdr_right = st.columns([3, 1])
with hdr_left:
    st.markdown(f"### SPX Dealer Risk Monitor")
    st.markdown(f"**SPX {spot:,.2f}** · {front_exp} · {chains[front_exp]['label']} · {ts}")
with hdr_right:
    col_t, col_r = st.columns(2)
    with col_t:
        dark = st.toggle("🌙", value=st.session_state.dark_mode, key="theme_toggle")
        if dark != st.session_state.dark_mode:
            st.session_state.dark_mode = dark
            st.rerun()
    with col_r:
        if st.button("🔄 Refresh"):
            st.cache_data.clear()
            st.rerun()

# ═══════════════════════════════════════
# Top Metrics
# ═══════════════════════════════════════
if not front_gex.empty:
    total_gex = front_gex["gex"].sum()
    total_vex = front_gex["vex"].sum()
    total_gp = front_gex["gex_plus"].sum()
    total_npd = front_gex["npd_contrib"].sum()
    zg = find_zero_gamma(front_gex, spot)
    combined_gp = sum(gex_data[e]["gex_plus"].sum() for e in target_monthlies if e in gex_data)
    charm_total = compute_charm_for_expiry(front_calls, front_puts, spot) if front_exp in chains else 0
    regime_amp = combined_gp < 0

    cols = st.columns(7)
    metrics = [
        ("GEX", f"{total_gex:.1f}M", ""),
        ("VEX", f"{total_vex:.1f}M", ""),
        ("CHARM", f"{charm_total/1e6:.1f}M/d", ""),
        ("GEX+", f"{total_gp:.1f}M", ""),
        ("COMBINED", f"{combined_gp:.1f}M", "All monthlies"),
        ("NPD", f"{total_npd:,.0f}", ""),
        ("ZERO-Γ", f"{zg:.0f}" if zg else "N/A", f"{(zg/spot-1)*100:+.1f}%" if zg else ""),
    ]
    for col, (label, value, sub) in zip(cols, metrics):
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    badge_class = "regime-amp" if regime_amp else "regime-damp"
    badge_text = "⚠ AMPLIFYING" if regime_amp else "✓ DAMPENING"
    st.markdown(f'<div style="text-align:right;margin-top:-10px;">'
                f'<span class="regime-badge {badge_class}">{badge_text}</span></div>',
                unsafe_allow_html=True)

st.divider()

# ═══════════════════════════════════════
# Tabs
# ═══════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "HEATMAP", "RISK ANALYSIS", "CRASH RISK", "FORECAST", "0DTE GRADIENT"
])


# ── TAB 1: HEATMAP ──
with tab1:
    st.markdown("#### GEX+ Risk Surface — Spot Move × IV Shock")
    st.caption("Blue = dealer dampening. Red = dealer amplifying. ◉ = current position.")

    spot_pcts = np.linspace(-0.15, 0.10, 16)
    iv_shocks = np.linspace(-0.10, 0.40, 16)

    with st.spinner("Computing risk surface..."):
        surface = build_risk_surface(front_calls, front_puts, spot, spot_pcts, iv_shocks)

    # Numeric axes
    x_vals = (spot_pcts * 100).tolist()   # [-15, ..., 10]
    y_vals = (iv_shocks * 100).tolist()   # [-10, ..., 40]

    fig_hm = go.Figure()
    fig_hm.add_trace(go.Heatmap(
        z=surface,
        x=x_vals,
        y=y_vals,
        colorscale=[
            [0, "#8B0000"], [0.20, "#CC2222"], [0.40, "#441111"],
            [0.48, "#1a0505"], [0.5, "#0a0a0a"], [0.52, "#05051a"],
            [0.60, "#111144"], [0.80, "#2222CC"], [1, "#0033FF"],
        ],
        zmid=0, zsmooth="best",
        colorbar=dict(title="GEX+ ($M)"),
        hovertemplate="Spot: %{x:.0f}%<br>IV: %{y:.0f}pts<br>GEX+: %{z:.1f}M<extra></extra>",
    ))

    # Crosshairs at 0,0
    fig_hm.add_hline(y=0, line_dash="dash", line_color="white", line_width=1.5)
    fig_hm.add_vline(x=0, line_dash="dash", line_color="white", line_width=1.5)
    fig_hm.add_annotation(x=0, y=0, text="◉", showarrow=False,
                           font=dict(color="white", size=18))
    fig_hm.add_annotation(x=-12, y=35, text="DANGER ZONE", showarrow=False,
                           font=dict(color="#FF5555", size=12), bgcolor="rgba(80,0,0,0.6)")
    fig_hm.add_annotation(x=7, y=-7, text="SAFE ZONE", showarrow=False,
                           font=dict(color="#5555FF", size=12), bgcolor="rgba(0,0,80,0.6)")

    # Spot labels along top
    for pv in [-10, -5, 0, 5, 10]:
        s = spot * (1 + pv / 100)
        fig_hm.add_annotation(x=pv, y=max(y_vals) + 3, text=f"SPX {s:,.0f}",
            showarrow=False, font=dict(color="rgba(255,255,255,0.5)", size=9))

    fig_hm.update_layout(
        template=theme["template"], height=550,
        xaxis_title="Spot Move (%)", yaxis_title="IV Shock (vol pts)",
        xaxis=dict(dtick=5), yaxis=dict(dtick=10),
    )
    st.plotly_chart(fig_hm, width="stretch")

    c0 = surface[np.argmin(np.abs(iv_shocks)), np.argmin(np.abs(spot_pcts))]
    wi = np.unravel_index(surface.argmin(), surface.shape)
    bi = np.unravel_index(surface.argmax(), surface.shape)
    st.markdown(f"**Current:** GEX+ = {c0:.1f}M · "
                f"**Worst:** {surface.min():.1f}M (spot {spot_pcts[wi[1]]*100:+.0f}%, IV {iv_shocks[wi[0]]*100:+.0f}pts) · "
                f"**Best:** {surface.max():.1f}M (spot {spot_pcts[bi[1]]*100:+.0f}%, IV {iv_shocks[bi[0]]*100:+.0f}pts)")


# ── TAB 2: RISK ANALYSIS ──
with tab2:
    st.markdown("#### GEX+ by Strike — Decomposition")

    if not front_gex.empty:
        mask = (front_gex["strike"] > spot * 0.92) & (front_gex["strike"] < spot * 1.08)
        df_plot = front_gex[mask]

        fig_gex = make_subplots(rows=2, cols=1,
            subplot_titles=["GEX+ by Strike", "GEX vs VEX Decomposition"],
            vertical_spacing=0.12)
        colors = [theme["green"] if v >= 0 else theme["red"] for v in df_plot["gex_plus"]]
        fig_gex.add_trace(go.Bar(x=df_plot["strike"], y=df_plot["gex_plus"],
            marker_color=colors, showlegend=False), row=1, col=1)
        fig_gex.add_vline(x=spot, line_dash="dash", line_color="white",
            annotation_text=f"SPX {spot:.0f}", row=1, col=1)
        fig_gex.add_trace(go.Bar(x=df_plot["strike"], y=df_plot["gex"],
            name="GEX", marker_color=theme["blue"], opacity=0.7), row=2, col=1)
        fig_gex.add_trace(go.Bar(x=df_plot["strike"], y=df_plot["vex"],
            name="VEX", marker_color=theme["gold"], opacity=0.7), row=2, col=1)
        fig_gex.add_vline(x=spot, line_dash="dash", line_color="white", row=2, col=1)
        fig_gex.update_layout(template=theme["template"], height=700, barmode="group")
        fig_gex.update_yaxes(title_text="GEX+ ($M)", row=1, col=1)
        fig_gex.update_yaxes(title_text="$M", row=2, col=1)
        st.plotly_chart(fig_gex, width="stretch")

    st.markdown("#### Charm — Time Decay Hedging Pressure")
    charm_data = []
    for exp, chain in chains.items():
        ch = compute_charm_for_expiry(chain["calls"], chain["puts"], spot)
        dte = chain["calls"]["daysToExpiration"].iloc[0] if not chain["calls"].empty else 0
        charm_data.append({"Expiry": exp, "Label": chain["label"],
                           "DTE": f"{dte:.0f}", "Charm (delta/day)": f"{ch:,.0f}"})
    if charm_data:
        st.dataframe(pd.DataFrame(charm_data), width="stretch", hide_index=True)


# ── TAB 3: CRASH RISK ──
with tab3:
    st.markdown("#### GEX+ Profile — Dealer Hedging vs Spot")

    with st.spinner("Computing profile curve..."):
        profile = build_gex_profile(front_calls, front_puts, spot, pct_range=0.20, n_points=40)

    fig_prof = go.Figure()
    fig_prof.add_trace(go.Scatter(
        x=np.concatenate([profile["pct_move"].values, profile["pct_move"].values[::-1]]),
        y=np.concatenate([np.maximum(profile["gex_plus"].values, 0), np.zeros(len(profile))]),
        fill="toself", fillcolor="rgba(0,150,50,0.3)", line=dict(width=0), name="Dampening"))
    fig_prof.add_trace(go.Scatter(
        x=np.concatenate([profile["pct_move"].values, profile["pct_move"].values[::-1]]),
        y=np.concatenate([np.minimum(profile["gex_plus"].values, 0), np.zeros(len(profile))]),
        fill="toself", fillcolor="rgba(200,50,50,0.3)", line=dict(width=0), name="Amplifying"))
    fig_prof.add_trace(go.Scatter(x=profile["pct_move"], y=profile["gex_plus"],
        mode="lines", name="GEX+", line=dict(color=theme["accent"], width=2.5)))
    fig_prof.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_prof.add_vline(x=0, line_dash="dot", line_color="white")

    for cp in [-5, -10, -15, -20]:
        gp_val = float(np.interp(cp, profile["pct_move"], profile["gex_plus"]))
        color = theme["red"] if gp_val < 0 else theme["green"]
        fig_prof.add_trace(go.Scatter(x=[cp], y=[gp_val], mode="markers+text",
            marker=dict(size=10, color=color), text=[f"{cp}%"],
            textposition="top center", textfont=dict(color=color, size=10),
            showlegend=False))

    cur_gp = float(np.interp(0, profile["pct_move"], profile["gex_plus"]))
    fig_prof.add_annotation(x=0, y=cur_gp,
        text=f"SPX {spot:,.0f}<br>GEX+ {cur_gp:.1f}M", showarrow=True, arrowhead=2,
        font=dict(size=11, color="white"), bgcolor="rgba(0,0,0,0.8)", bordercolor="white")
    fig_prof.update_layout(template=theme["template"], height=400,
        xaxis_title="Spot Move (%)", yaxis_title="GEX+ ($M)", hovermode="x unified")
    st.plotly_chart(fig_prof, width="stretch")

    # Crash cards
    st.markdown("#### Crash Risk — GEX+ at Drawdown Levels")
    crash_cols = st.columns(4)
    for i, cp in enumerate([-5, -10, -15, -20]):
        cs = spot * (1 + cp / 100)
        gex, vex, gp = compute_gex_plus_at_spot(front_calls, front_puts, cs)
        if gp < -50:
            sev, sev_class = "ELEVATED", "crash-elevated"
        elif gp < 0:
            sev, sev_class = "ELEVATED", "crash-elevated"
        elif gp < 50:
            sev, sev_class = "NEUTRAL", "crash-neutral"
        else:
            sev, sev_class = "CONTAINED", "crash-contained"

        with crash_cols[i]:
            st.markdown(f"""<div class="crash-card {sev_class}">
                <div style="font-size:20px;font-weight:bold;font-family:monospace;">{cp}%</div>
                <div style="font-size:11px;color:{theme['muted']};margin-top:4px;">
                    CRASH SPOT: {cs:,.0f}<br>MARKET IV: ~{20+abs(cp)*0.4:.0f}%
                </div>
                <div style="font-size:28px;font-weight:bold;font-family:monospace;margin:8px 0;
                            color:{'#FF4444' if gp < 0 else '#44FF44'};">
                    {gp:.0f}M
                </div>
                <div style="font-size:10px;color:{theme['muted']};">
                    GEX: {gex:.1f}M · VEX: {vex:.1f}M
                </div>
                <div style="font-size:11px;font-weight:bold;margin-top:6px;
                            color:{'#FF4444' if sev=='ELEVATED' else '#888'};">
                    {sev}
                </div>
            </div>""", unsafe_allow_html=True)


# ── TAB 4: FORECAST ──
with tab4:
    st.markdown("#### SPX Probability Forecast")
    st.caption("Breeden-Litzenberger risk-neutral density · Cornish-Fisher adjustment")

    bl = breeden_litzenberger(front_calls, front_puts, spot)
    if bl:
        st.markdown(f"**{front_exp} — {bl['dte']:.0f} DTE**")
        pcols = st.columns(5)
        for col, (pct, label) in zip(pcols, [(5, "5th"), (25, "25th"), (50, "MEDIAN"), (75, "75th"), (95, "95th")]):
            val = bl["pctiles_exp"][pct]
            with col:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">{label} PCTILE</div>
                    <div class="metric-value">{val:,.1f}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown(f"**1σ Move:** ±{bl['sigma_exp']:.0f} pts (±{bl['sigma_exp']/spot*100:.1f}%) · "
                    f"**ATM IV:** {bl['atm_iv']*100:.1f}% · "
                    f"**Skew:** {bl['skew']:.3f} · **Kurtosis:** {bl['kurt']:.2f}")

        if bl["prob_table"]:
            prob_df = pd.DataFrame(bl["prob_table"])
            prob_df.columns = ["SPX Level", "P(below)", "P(above)"]
            prob_df["P(below)"] = prob_df["P(below)"].apply(lambda x: f"{x:.1f}%")
            prob_df["P(above)"] = prob_df["P(above)"].apply(lambda x: f"{x:.1f}%")
            st.dataframe(prob_df, width="stretch", hide_index=True)

        if len(bl["density_strikes"]) > 5:
            fig_den = go.Figure()
            fig_den.add_trace(go.Scatter(
                x=bl["density_strikes"], y=bl["density_vals"],
                fill="tozeroy", fillcolor="rgba(0,150,200,0.3)",
                line=dict(color=theme["blue"], width=2), name="RN Density"))
            fig_den.add_vline(x=spot, line_dash="dash", line_color="white",
                annotation_text=f"SPX {spot:.0f}")
            fig_den.add_vline(x=bl["mean"], line_dash="dot", line_color=theme["accent"],
                annotation_text=f"BL Mean {bl['mean']:.0f}")
            fig_den.update_layout(template=theme["template"], height=350,
                xaxis_title="SPX at Expiry", yaxis_title="Probability Density")
            st.plotly_chart(fig_den, width="stretch")
    else:
        st.warning("Insufficient OTM option data for Breeden-Litzenberger density extraction.")


# ── TAB 5: 0DTE GRADIENT ──
with tab5:
    st.markdown("#### 0DTE Gamma & Charm Gradient")

    if nearest_exp and nearest_exp in chains:
        dte_calls = chains[nearest_exp]["calls"]
        dte_puts = chains[nearest_exp]["puts"]

        raw_exp = compute_raw_exposures(dte_calls, dte_puts, spot)

        # Wider view: ±3% of spot
        lo, hi = spot * 0.97, spot * 1.03
        margin = 80
        raw_wide = [r for r in raw_exp if (lo - margin) <= r["strike"] <= (hi + margin)]
        strikes_raw = [r["strike"] for r in raw_wide]
        gex_raw = [r["net_gex"] for r in raw_wide]
        charm_raw = [r["net_charm"] for r in raw_wide]

        # Fine grid: 0.5pt resolution
        price_grid = np.arange(lo, hi + 0.5, 0.5)

        # TIGHT KDE (sigma=4) — preserves strike-level peaks and valleys
        gex_field = kde_field(strikes_raw, gex_raw, price_grid, sigma=4)
        charm_field = kde_field(strikes_raw, charm_raw, price_grid, sigma=5)

        # Power scale to compress extremes and show mid-range variation
        def power_scale(arr, power=0.4):
            return np.sign(arr) * np.abs(arr) ** power

        gex_display = power_scale(gex_field, 0.4)
        charm_display = power_scale(charm_field, 0.4)

        n_cols = 50
        gex_matrix = np.tile(gex_display.reshape(-1, 1), (1, n_cols))
        charm_matrix = np.tile(charm_display.reshape(-1, 1), (1, n_cols))
        x_labels = list(range(n_cols))

        gamma_colorscale = [
            [0, "#CC1111"], [0.10, "#AA2222"], [0.20, "#883333"],
            [0.30, "#663333"], [0.38, "#442222"], [0.44, "#2a1111"],
            [0.48, "#150808"], [0.50, "#080808"], [0.52, "#081508"],
            [0.56, "#112a11"], [0.62, "#224422"], [0.70, "#336633"],
            [0.80, "#448844"], [0.90, "#55AA55"], [1, "#66CC66"],
        ]

        charm_colorscale = [
            [0, "#CC8800"], [0.10, "#AA7722"], [0.20, "#886633"],
            [0.30, "#665522"], [0.38, "#443311"], [0.44, "#2a1a08"],
            [0.48, "#150f05"], [0.50, "#080808"], [0.52, "#050815"],
            [0.56, "#081a2a"], [0.62, "#113344"], [0.70, "#225566"],
            [0.80, "#337788"], [0.90, "#4499AA"], [1, "#55BBCC"],
        ]

        gcol, ccol = st.columns(2)

        with gcol:
            st.markdown(f"**Gamma** — {ts}")
            fig_g = go.Figure()
            fig_g.add_trace(go.Heatmap(
                z=gex_matrix, y=price_grid, x=x_labels,
                colorscale=gamma_colorscale,
                zmid=0, zsmooth="best",
                colorbar=dict(title="GEX"),
                hovertemplate="Strike: %{y:.0f}<br>GEX: %{z:.3f}<extra></extra>",
            ))
            fig_g.add_hline(y=spot, line_color="white", line_width=2,
                annotation_text=f"SPX {spot:.0f}", annotation_position="top left",
                annotation_font=dict(color="white", size=11))
            fig_g.update_layout(template=theme["template"], height=650,
                yaxis_title="SPX Level", yaxis=dict(dtick=25),
                xaxis=dict(showticklabels=False, showgrid=False),
                margin=dict(l=50, r=60, t=30, b=30))
            st.plotly_chart(fig_g, width="stretch")

        with ccol:
            st.markdown(f"**Charm** — {ts}")
            fig_c = go.Figure()
            fig_c.add_trace(go.Heatmap(
                z=charm_matrix, y=price_grid, x=x_labels,
                colorscale=charm_colorscale,
                zmid=0, zsmooth="best",
                colorbar=dict(title="Charm"),
                hovertemplate="Strike: %{y:.0f}<br>Charm: %{z:.0f}<extra></extra>",
            ))
            fig_c.add_hline(y=spot, line_color="white", line_width=2,
                annotation_text=f"SPX {spot:.0f}", annotation_position="top left",
                annotation_font=dict(color="white", size=11))
            fig_c.update_layout(template=theme["template"], height=650,
                yaxis_title="SPX Level", yaxis=dict(dtick=25),
                xaxis=dict(showticklabels=False, showgrid=False),
                margin=dict(l=50, r=60, t=30, b=30))
            st.plotly_chart(fig_c, width="stretch")

        spot_idx = int((spot - price_grid[0]) / 0.5)
        if 0 <= spot_idx < len(gex_field):
            spot_gex = gex_field[spot_idx]
            st.markdown(f"**At spot:** GEX={spot_gex:.3f}B → "
                        f"{'DAMPENING' if spot_gex > 0 else 'AMPLIFYING'} · "
                        f"**Max dampening:** {price_grid[np.argmax(gex_field)]:.0f} · "
                        f"**Max amplifying:** {price_grid[np.argmin(gex_field)]:.0f}")
    else:
        st.warning("No 0DTE chain available.")

# Footer
st.divider()
st.caption(f"Data: Barchart · Spot: tvdatafeed TVC:SPX · Last refresh: {ts}")
