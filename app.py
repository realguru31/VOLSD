"""
SPX Dealer Risk Monitor — Streamlit App
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pytz
import os
import pickle
import time

from fetch import (create_session, fetch_spot, fetch_spot_tv, get_expiries,
                   fetch_full_chain)
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
    st.markdown("<h2 style='text-align:center;color:#c0d0e8;'>🔒 SPX Dealer Risk Monitor</h2>",
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

NAVY_BG = "#0a1628"
NAVY_CARD = "#13213a"
NAVY_BORDER = "#1e3050"
NAVY_TEXT = "#c0d0e8"
NAVY_TEXT_DIM = "#7a93b8"


def get_theme():
    if st.session_state.dark_mode:
        return {
            "template": "plotly_dark", "bg": NAVY_BG, "card_bg": NAVY_CARD,
            "text": NAVY_TEXT, "accent": "#44CCFF", "red": "#FF4466",
            "green": "#44CC77", "blue": "#4488DD", "gold": "#DDAA44",
            "muted": "#5a7090", "grid": NAVY_BORDER,
        }
    else:
        return {
            "template": "plotly_white", "bg": "#ffffff", "card_bg": "#f5f5f5",
            "text": "#1a1a1a", "accent": "#008866", "red": "#CC2222",
            "green": "#228822", "blue": "#2255BB", "gold": "#AA7700",
            "muted": "#999999", "grid": "#dddddd",
        }

theme = get_theme()


def apply_plotly_theme(fig):
    if st.session_state.dark_mode:
        fig.update_layout(paper_bgcolor=NAVY_BG, plot_bgcolor=NAVY_BG,
                          font=dict(color=NAVY_TEXT))
        fig.update_xaxes(gridcolor=NAVY_BORDER, zerolinecolor="#2a3f5f")
        fig.update_yaxes(gridcolor=NAVY_BORDER, zerolinecolor="#2a3f5f")
    return fig


# CSS
if st.session_state.dark_mode:
    st.markdown("""<style>
        #MainMenu {visibility: hidden;}
        header {visibility: hidden; height: 0;}
        footer {visibility: hidden;}
        .stDeployButton {display: none;}
        [data-testid="stToolbar"] {visibility: hidden; height: 0; position: fixed;}
        [data-testid="stDecoration"] {visibility: hidden; height: 0; position: fixed;}
        [data-testid="stStatusWidget"] {visibility: hidden; height: 0; position: fixed;}
        [data-testid="stHeader"] {display: none;}

        .stApp { background-color: #0a1628 !important; }
        [data-testid="stAppViewContainer"] { background-color: #0a1628 !important; }
        [data-testid="stMain"] { background-color: #0a1628 !important; }
        .main { background-color: #0a1628 !important; }
        .block-container { padding-top: 1rem !important; background-color: #0a1628 !important; }

        .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
        .stMarkdown h4, .stMarkdown h5, .stMarkdown li, p, h1, h2, h3, h4, h5 {
            color: #c0d0e8 !important;
        }

        .metric-card { background: #13213a; border: 1px solid #1e3050; border-radius: 8px;
                       padding: 12px 16px; text-align: center; }
        .metric-label { color: #7a93b8; font-size: 11px; text-transform: uppercase; font-family: monospace; }
        .metric-value { color: #e0e8f5; font-size: 22px; font-weight: bold; font-family: monospace; }
        .metric-sub { color: #4a5d7e; font-size: 10px; font-family: monospace; }

        .regime-badge { padding: 4px 12px; border-radius: 4px; font-weight: bold; font-family: monospace; display: inline-block; }
        .regime-amp { background: #2a1520; color: #FF6688; border: 1px solid #FF4466; }
        .regime-damp { background: #152a1f; color: #66FF99; border: 1px solid #44CC77; }

        .crash-card { background: #13213a; border: 1px solid #1e3050; border-radius: 8px; padding: 16px; margin: 8px 0; }
        .crash-elevated { border-left: 4px solid #FF4466; }
        .crash-neutral { border-left: 4px solid #5a7090; }
        .crash-contained { border-left: 4px solid #4488DD; }

        .stButton > button {
            background-color: #13213a !important; color: #c0d0e8 !important;
            border: 1px solid #2a3f5f !important; font-family: monospace;
        }
        .stButton > button:hover {
            background-color: #1c2f50 !important; border-color: #4a6090 !important;
            color: #ffffff !important;
        }
        div[data-baseweb="select"] > div {
            background-color: #13213a !important; border-color: #2a3f5f !important;
            color: #c0d0e8 !important;
        }
        div[data-baseweb="popover"] { background-color: #13213a !important; }
        li[role="option"] { background-color: #13213a !important; color: #c0d0e8 !important; }
        li[role="option"]:hover { background-color: #1c2f50 !important; }
        input[type="text"], input[type="password"] {
            background-color: #13213a !important; color: #c0d0e8 !important;
            border-color: #2a3f5f !important;
        }
        div[data-testid="stAlert"] {
            background-color: rgba(30,60,90,0.3) !important;
            border: 1px solid #2a4060 !important; color: #88BBDD !important;
        }
        [data-testid="stTabs"] button { color: #7a93b8 !important; }
        [data-testid="stTabs"] button[aria-selected="true"] { color: #88BBFF !important; }
        hr { border-color: #1e3050 !important; }
        label, [data-testid="stWidgetLabel"] { color: #c0d0e8 !important; }
        [data-testid="stPlotlyChart"] { background-color: #0a1628 !important; }
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
# Snapshot Storage
# ═══════════════════════════════════════
SNAPSHOT_DIR = "snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

ET = pytz.timezone("US/Eastern")


def snapshot_file_for_today():
    return os.path.join(SNAPSHOT_DIR, f"intervals_{datetime.now(ET).strftime('%Y%m%d')}.pkl")


def load_snapshots():
    path = snapshot_file_for_today()
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return []
    return []


def save_snapshot(snapshot):
    snapshots = load_snapshots()
    minute_key = datetime.now(ET).strftime("%H%M")
    snapshots = [s for s in snapshots if s.get("minute_key") != minute_key]
    snapshot["minute_key"] = minute_key
    snapshots.append(snapshot)
    with open(snapshot_file_for_today(), "wb") as f:
        pickle.dump(snapshots, f)


def take_interval_snapshot():
    """Fetch 0DTE chain, compute GEX + volume per strike, save snapshot."""
    sess, headers = create_session()
    spot_now = fetch_spot_tv()
    if not spot_now:
        spot_now = fetch_spot(sess, headers)
    if not spot_now:
        return False

    weekly, monthly = get_expiries(sess, headers)
    today_str = datetime.now().strftime("%Y-%m-%d")
    future_weekly = [e for e in weekly if e >= today_str]
    if not future_weekly:
        return False
    nearest = future_weekly[0]

    calls, puts = fetch_full_chain(sess, headers, nearest, is_dense=False)
    if calls.empty or puts.empty:
        return False

    gex_df = compute_gex_vex(calls, puts, spot_now)

    # Extract volume per strike (for Volume Flow chart)
    call_vols = {}
    put_vols = {}
    for _, row in calls.iterrows():
        K = row["strikePrice"]
        call_vols[K] = row.get("volume", 0) or 0
    for _, row in puts.iterrows():
        K = row["strikePrice"]
        put_vols[K] = row.get("volume", 0) or 0

    et_now = datetime.now(ET)
    # Minutes from 9:30 AM for X-axis positioning
    market_open = et_now.replace(hour=9, minute=30, second=0, microsecond=0)
    minutes_from_open = max(0, (et_now - market_open).total_seconds() / 60)

    snapshot = {
        "timestamp": et_now.isoformat(),
        "time_label": et_now.strftime("%I:%M %p"),
        "minutes_from_open": minutes_from_open,
        "spot": spot_now,
        "expiry": nearest,
        "strikes": gex_df["strike"].tolist(),
        "gex_plus": gex_df["gex_plus"].tolist(),
        "gex": gex_df["gex"].tolist(),
        "call_volumes": call_vols,
        "put_volumes": put_vols,
    }
    save_snapshot(snapshot)
    return True


# ═══════════════════════════════════════
# Main Data Loading (cached)
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
        "timestamp": datetime.now(ET).strftime("%I:%M %p ET"),
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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "HEATMAP", "RISK ANALYSIS", "CRASH RISK", "FORECAST",
    "0DTE PROFILE", "INTERVAL MAP"
])


# ── TAB 1: HEATMAP ──
with tab1:
    st.markdown("#### GEX+ Risk Surface — Spot Move × IV Shock")
    st.caption("Blue = dealer dampening. Red = dealer amplifying. ◉ = current position.")

    spot_pcts = np.linspace(-0.15, 0.10, 16)
    iv_shocks = np.linspace(-0.10, 0.40, 16)

    with st.spinner("Computing risk surface..."):
        surface = build_risk_surface(front_calls, front_puts, spot, spot_pcts, iv_shocks)

    fig_hm = go.Figure()
    fig_hm.add_trace(go.Heatmap(
        z=surface, x=(spot_pcts * 100).tolist(), y=(iv_shocks * 100).tolist(),
        colorscale=[
            [0, "#8B0000"], [0.20, "#CC2222"], [0.40, "#441111"],
            [0.48, "#1a0505"], [0.5, NAVY_BG], [0.52, "#0a1020"],
            [0.60, "#1a2850"], [0.80, "#3366CC"], [1, "#4488FF"],
        ],
        zmid=0, zsmooth="best", colorbar=dict(title="GEX+ ($M)"),
        hovertemplate="Spot: %{x:.0f}%<br>IV: %{y:.0f}pts<br>GEX+: %{z:.1f}M<extra></extra>",
    ))
    fig_hm.add_hline(y=0, line_dash="dash", line_color="white", line_width=1.5)
    fig_hm.add_vline(x=0, line_dash="dash", line_color="white", line_width=1.5)
    fig_hm.add_annotation(x=0, y=0, text="◉", showarrow=False, font=dict(color="white", size=18))
    fig_hm.add_annotation(x=-12, y=35, text="DANGER ZONE", showarrow=False,
                           font=dict(color="#FF6688", size=12), bgcolor="rgba(80,0,0,0.6)")
    fig_hm.add_annotation(x=7, y=-7, text="SAFE ZONE", showarrow=False,
                           font=dict(color="#66AAFF", size=12), bgcolor="rgba(0,20,60,0.6)")
    fig_hm.update_layout(template=theme["template"], height=550,
        xaxis_title="Spot Move (%)", yaxis_title="IV Shock (vol pts)",
        xaxis=dict(dtick=5), yaxis=dict(dtick=10))
    fig_hm = apply_plotly_theme(fig_hm)
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
            subplot_titles=["GEX+ by Strike", "GEX vs VEX Decomposition"], vertical_spacing=0.12)
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
        fig_gex = apply_plotly_theme(fig_gex)
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
            textposition="top center", textfont=dict(color=color, size=10), showlegend=False))
    cur_gp = float(np.interp(0, profile["pct_move"], profile["gex_plus"]))
    fig_prof.add_annotation(x=0, y=cur_gp, text=f"SPX {spot:,.0f}<br>GEX+ {cur_gp:.1f}M",
        showarrow=True, arrowhead=2, font=dict(size=11, color="white"),
        bgcolor="rgba(10,22,40,0.9)", bordercolor="white")
    fig_prof.update_layout(template=theme["template"], height=400,
        xaxis_title="Spot Move (%)", yaxis_title="GEX+ ($M)", hovermode="x unified")
    fig_prof = apply_plotly_theme(fig_prof)
    st.plotly_chart(fig_prof, width="stretch")

    st.markdown("#### Crash Risk — GEX+ at Drawdown Levels")
    crash_cols = st.columns(4)
    for i, cp in enumerate([-5, -10, -15, -20]):
        cs = spot * (1 + cp / 100)
        gex, vex, gp = compute_gex_plus_at_spot(front_calls, front_puts, cs)
        sev = "ELEVATED" if gp < 0 else ("NEUTRAL" if gp < 50 else "CONTAINED")
        sev_class = "crash-elevated" if gp < 0 else ("crash-neutral" if gp < 50 else "crash-contained")
        with crash_cols[i]:
            st.markdown(f"""<div class="crash-card {sev_class}">
                <div style="font-size:20px;font-weight:bold;font-family:monospace;color:#e0e8f5;">{cp}%</div>
                <div style="font-size:11px;color:#7a93b8;margin-top:4px;">CRASH SPOT: {cs:,.0f}<br>MARKET IV: ~{20+abs(cp)*0.4:.0f}%</div>
                <div style="font-size:28px;font-weight:bold;font-family:monospace;margin:8px 0;color:{'#FF6688' if gp < 0 else '#66FF99'};">{gp:.0f}M</div>
                <div style="font-size:10px;color:#7a93b8;">GEX: {gex:.1f}M · VEX: {vex:.1f}M</div>
                <div style="font-size:11px;font-weight:bold;margin-top:6px;color:{'#FF6688' if sev=='ELEVATED' else '#7a93b8'};">{sev}</div>
            </div>""", unsafe_allow_html=True)


# ── TAB 4: FORECAST ──
with tab4:
    st.markdown("#### SPX Probability Forecast")
    st.caption("Breeden-Litzenberger risk-neutral density · Cornish-Fisher adjustment")
    bl = breeden_litzenberger(front_calls, front_puts, spot)
    if bl:
        st.markdown(f"**{front_exp} — {bl['dte']:.0f} DTE**")
        pcols = st.columns(5)
        for col, (pct, label) in zip(pcols, [(5,"5th"),(25,"25th"),(50,"MEDIAN"),(75,"75th"),(95,"95th")]):
            val = bl["pctiles_exp"][pct]
            with col:
                st.markdown(f"""<div class="metric-card"><div class="metric-label">{label} PCTILE</div>
                    <div class="metric-value">{val:,.1f}</div></div>""", unsafe_allow_html=True)
        st.markdown(f"**1σ Move:** ±{bl['sigma_exp']:.0f} pts · **ATM IV:** {bl['atm_iv']*100:.1f}% · "
                    f"**Skew:** {bl['skew']:.3f} · **Kurtosis:** {bl['kurt']:.2f}")
        if bl["prob_table"]:
            prob_df = pd.DataFrame(bl["prob_table"])
            prob_df.columns = ["SPX Level", "P(below)", "P(above)"]
            prob_df["P(below)"] = prob_df["P(below)"].apply(lambda x: f"{x:.1f}%")
            prob_df["P(above)"] = prob_df["P(above)"].apply(lambda x: f"{x:.1f}%")
            st.dataframe(prob_df, width="stretch", hide_index=True)
        if len(bl["density_strikes"]) > 5:
            fig_den = go.Figure()
            fig_den.add_trace(go.Scatter(x=bl["density_strikes"], y=bl["density_vals"],
                fill="tozeroy", fillcolor="rgba(68,136,221,0.3)",
                line=dict(color=theme["blue"], width=2), name="RN Density"))
            fig_den.add_vline(x=spot, line_dash="dash", line_color="white", annotation_text=f"SPX {spot:.0f}")
            fig_den.add_vline(x=bl["mean"], line_dash="dot", line_color=theme["accent"],
                annotation_text=f"BL Mean {bl['mean']:.0f}")
            fig_den.update_layout(template=theme["template"], height=350,
                xaxis_title="SPX at Expiry", yaxis_title="Probability Density")
            fig_den = apply_plotly_theme(fig_den)
            st.plotly_chart(fig_den, width="stretch")
    else:
        st.warning("Insufficient OTM option data for BL density extraction.")


# ── TAB 5: 0DTE PROFILE ──
with tab5:
    st.markdown("#### 0DTE Exposure Profile — Current Snapshot")
    if nearest_exp and nearest_exp in chains:
        dte_calls = chains[nearest_exp]["calls"]
        dte_puts = chains[nearest_exp]["puts"]
        raw_exp = compute_raw_exposures(dte_calls, dte_puts, spot)
        lo, hi = spot * 0.98, spot * 1.02
        filtered = [r for r in raw_exp if lo <= r["strike"] <= hi]
        strikes = np.array([r["strike"] for r in filtered])
        gex_vals = np.array([r["net_gex"] for r in filtered])
        charm_vals = np.array([r["net_charm"] for r in filtered])
        if len(strikes) > 5:
            gcol, ccol = st.columns(2)
            with gcol:
                st.markdown(f"**Gamma** — {ts}")
                fig_g = go.Figure()
                if (gex_vals >= 0).any():
                    fig_g.add_trace(go.Scatter(x=np.where(gex_vals >= 0, gex_vals, 0), y=strikes,
                        fill="tozerox", fillcolor="rgba(68,204,119,0.5)",
                        line=dict(color="#66DD88", width=2), name="Dampening", mode="lines"))
                if (gex_vals < 0).any():
                    fig_g.add_trace(go.Scatter(x=np.where(gex_vals < 0, gex_vals, 0), y=strikes,
                        fill="tozerox", fillcolor="rgba(255,68,102,0.5)",
                        line=dict(color="#FF6688", width=2), name="Amplifying", mode="lines"))
                fig_g.add_vline(x=0, line_color="gray", line_dash="dash")
                fig_g.add_hline(y=spot, line_color="white", line_width=2, annotation_text=f"SPX {spot:.0f}")
                fig_g.update_layout(template=theme["template"], height=600,
                    xaxis_title="Net GEX ($B)", yaxis_title="Strike", hovermode="y unified")
                fig_g = apply_plotly_theme(fig_g)
                st.plotly_chart(fig_g, width="stretch")
            with ccol:
                st.markdown(f"**Charm** — {ts}")
                fig_c = go.Figure()
                if (charm_vals >= 0).any():
                    fig_c.add_trace(go.Scatter(x=np.where(charm_vals >= 0, charm_vals, 0), y=strikes,
                        fill="tozerox", fillcolor="rgba(68,136,221,0.5)",
                        line=dict(color="#4488DD", width=2), name="Buying support", mode="lines"))
                if (charm_vals < 0).any():
                    fig_c.add_trace(go.Scatter(x=np.where(charm_vals < 0, charm_vals, 0), y=strikes,
                        fill="tozerox", fillcolor="rgba(221,170,68,0.5)",
                        line=dict(color="#DDAA44", width=2), name="Selling pressure", mode="lines"))
                fig_c.add_vline(x=0, line_color="gray", line_dash="dash")
                fig_c.add_hline(y=spot, line_color="white", line_width=2, annotation_text=f"SPX {spot:.0f}")
                fig_c.update_layout(template=theme["template"], height=600,
                    xaxis_title="Net Charm", yaxis_title="Strike", hovermode="y unified")
                fig_c = apply_plotly_theme(fig_c)
                st.plotly_chart(fig_c, width="stretch")
    else:
        st.warning("No 0DTE chain available.")


# ── TAB 6: INTERVAL MAP (fragment — only this refreshes) ──
with tab6:

    @st.fragment(run_every=60)
    def interval_map_fragment():
        st.markdown("#### Interval Map — 0DTE GEX + Volume Flow")
        st.caption("**Top:** GEX landscape (green=dampening, red=amplifying). "
                   "**Bottom:** Net volume flow (green=call volume, red=put volume). "
                   "Blue line = SPX spot.")

        ctrl_cols = st.columns([1.4, 1.2, 0.9, 0.9, 1.0])
        with ctrl_cols[0]:
            pct_range = st.selectbox("Y-axis range",
                options=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
                format_func=lambda x: f"±{x}%", index=1, key="pct_range_select")
        with ctrl_cols[1]:
            refresh_minutes = st.selectbox("Refresh every",
                options=[1, 2, 3], format_func=lambda x: f"{x} min",
                index=1, key="refresh_min_select")
        with ctrl_cols[2]:
            auto_refresh = st.toggle("Auto", value=True, key="auto_refresh_toggle")
        with ctrl_cols[3]:
            if st.button("🔄 Fetch", key="manual_fetch"):
                with st.spinner("Fetching..."):
                    if take_interval_snapshot():
                        st.success("Added")
                    else:
                        st.error("Failed")
        with ctrl_cols[4]:
            if st.button("🗑 Clear", key="clear_today"):
                path = snapshot_file_for_today()
                if os.path.exists(path):
                    os.remove(path)
                    st.success("Cleared")

        # Auto-fetch logic (runs inside fragment every 60s)
        if auto_refresh:
            et_now = datetime.now(ET)
            is_weekday = et_now.weekday() < 5
            market_open = et_now.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = et_now.replace(hour=16, minute=0, second=0, microsecond=0)
            in_market = is_weekday and market_open <= et_now <= market_close

            if in_market:
                if "last_interval_fetch" not in st.session_state:
                    st.session_state.last_interval_fetch = 0
                refresh_seconds = refresh_minutes * 60
                if time.time() - st.session_state.last_interval_fetch >= refresh_seconds:
                    take_interval_snapshot()
                    st.session_state.last_interval_fetch = time.time()

        snapshots = load_snapshots()

        if not snapshots:
            st.info("No snapshots yet. Click 'Fetch' or wait for auto-refresh (9:30-4:00 ET).")
            return

        refresh_sec = refresh_minutes * 60
        remaining = max(0, refresh_sec - int(time.time() - st.session_state.get("last_interval_fetch", 0)))
        st.caption(f"Snapshots: **{len(snapshots)}** · Latest: {snapshots[-1]['time_label']} · Next: ~{remaining}s")

        # ── Build strike grid ──
        latest_spot = snapshots[-1]["spot"]
        range_pts = latest_spot * pct_range / 100
        lo_strike = int(round((latest_spot - range_pts) / 5.0)) * 5
        hi_strike = int(round((latest_spot + range_pts) / 5.0)) * 5
        strike_levels = np.arange(lo_strike, hi_strike + 5, 5)
        n_strikes = len(strike_levels)
        n_times = len(snapshots)

        # ── Full RTH X-axis: 0 = 9:30, 390 = 4:00 ──
        RTH_MINUTES = 390  # 6.5 hours
        tick_positions = list(range(0, RTH_MINUTES + 1, 30))
        tick_labels = []
        for m in tick_positions:
            h = 9 + (m + 30) // 60
            mn = (m + 30) % 60
            ampm = "AM" if h < 12 else "PM"
            if h > 12: h -= 12
            tick_labels.append(f"{h}:{mn:02d} {ampm}")

        # Get each snapshot's X position (minutes from 9:30)
        snap_x_positions = []
        for snap in snapshots:
            mfo = snap.get("minutes_from_open", 0)
            snap_x_positions.append(min(mfo, RTH_MINUTES))

        # ── Chart 1: GEX Landscape ──
        gex_grid = np.zeros((n_strikes, n_times))
        for t_idx, snap in enumerate(snapshots):
            strikes_arr = np.array(snap["strikes"])
            gex_arr = np.array(snap["gex_plus"])
            for s_idx, K in enumerate(strike_levels):
                matches = np.where(strikes_arr == K)[0]
                if len(matches) > 0:
                    gex_grid[s_idx, t_idx] = gex_arr[matches[0]]
                elif len(strikes_arr) >= 2 and strikes_arr.min() <= K <= strikes_arr.max():
                    gex_grid[s_idx, t_idx] = np.interp(K, strikes_arr, gex_arr)

        # Flatten for scatter
        flat_y_gex, flat_x_gex, flat_z_gex = [], [], []
        for s_idx in range(n_strikes):
            for t_idx in range(n_times):
                flat_y_gex.append(strike_levels[s_idx])
                flat_x_gex.append(snap_x_positions[t_idx])
                flat_z_gex.append(gex_grid[s_idx, t_idx])

        flat_y_gex = np.array(flat_y_gex)
        flat_x_gex = np.array(flat_x_gex)
        flat_z_gex = np.array(flat_z_gex)

        max_abs_gex = max(np.abs(flat_z_gex).max(), 0.001)
        sizes_gex = 4 + 28 * (np.abs(flat_z_gex) / max_abs_gex)

        colors_gex = []
        for v in flat_z_gex:
            if v > max_abs_gex * 0.03:
                intensity = min(1.0, v / max_abs_gex)
                colors_gex.append(f"rgba(50,{int(130+125*intensity)},60,{0.4+0.6*intensity})")
            elif v < -max_abs_gex * 0.03:
                intensity = min(1.0, -v / max_abs_gex)
                colors_gex.append(f"rgba({int(130+125*intensity)},50,80,{0.4+0.6*intensity})")
            else:
                colors_gex.append("rgba(100,120,150,0.15)")

        fig_gex_map = go.Figure()
        fig_gex_map.add_trace(go.Scatter(
            x=flat_x_gex, y=flat_y_gex, mode="markers",
            marker=dict(size=sizes_gex, color=colors_gex, line=dict(width=0)),
            hovertemplate="Strike: %{y}<br>GEX+: %{text}<extra></extra>",
            text=[f"{v:+.2f}M" for v in flat_z_gex],
            showlegend=False, name="GEX+",
        ))
        # Spot line
        fig_gex_map.add_trace(go.Scatter(
            x=snap_x_positions, y=[s["spot"] for s in snapshots],
            mode="lines", line=dict(color="#44CCFF", width=2.5),
            name="SPX", hovertemplate="SPX: %{y:.2f}<extra></extra>",
        ))
        fig_gex_map.update_layout(
            template=theme["template"], height=400,
            title_text="GEX Landscape",
            xaxis=dict(title="", range=[0, RTH_MINUTES],
                       tickmode="array", tickvals=tick_positions, ticktext=tick_labels,
                       tickangle=-45),
            yaxis=dict(title="Strike", dtick=5, range=[lo_strike-2.5, hi_strike+2.5]),
            showlegend=True, legend=dict(x=0.01, y=0.99, bgcolor="rgba(10,22,40,0.7)"),
            margin=dict(l=60, r=40, t=40, b=60),
        )
        fig_gex_map = apply_plotly_theme(fig_gex_map)
        st.plotly_chart(fig_gex_map, width="stretch")

        # ── Chart 2: Volume Flow (net call vol - put vol, incremental) ──
        if n_times >= 2:
            flat_y_vol, flat_x_vol, flat_z_vol = [], [], []

            for t_idx in range(1, n_times):
                prev = snapshots[t_idx - 1]
                curr = snapshots[t_idx]
                prev_cv = prev.get("call_volumes", {})
                prev_pv = prev.get("put_volumes", {})
                curr_cv = curr.get("call_volumes", {})
                curr_pv = curr.get("put_volumes", {})

                for K in strike_levels:
                    Kf = float(K)
                    delta_call = curr_cv.get(Kf, 0) - prev_cv.get(Kf, 0)
                    delta_put = curr_pv.get(Kf, 0) - prev_pv.get(Kf, 0)
                    net_vol = delta_call - delta_put  # positive = more calls, negative = more puts

                    flat_y_vol.append(K)
                    flat_x_vol.append(snap_x_positions[t_idx])
                    flat_z_vol.append(net_vol)

            if flat_z_vol:
                flat_y_vol = np.array(flat_y_vol)
                flat_x_vol = np.array(flat_x_vol)
                flat_z_vol = np.array(flat_z_vol)

                max_abs_vol = max(np.abs(flat_z_vol).max(), 1)
                sizes_vol = 3 + 25 * (np.abs(flat_z_vol) / max_abs_vol)

                colors_vol = []
                for v in flat_z_vol:
                    if v > max_abs_vol * 0.05:
                        intensity = min(1.0, v / max_abs_vol)
                        colors_vol.append(f"rgba(50,{int(130+125*intensity)},60,{0.4+0.6*intensity})")
                    elif v < -max_abs_vol * 0.05:
                        intensity = min(1.0, -v / max_abs_vol)
                        colors_vol.append(f"rgba({int(130+125*intensity)},50,80,{0.4+0.6*intensity})")
                    else:
                        colors_vol.append("rgba(100,120,150,0.15)")

                fig_vol = go.Figure()
                fig_vol.add_trace(go.Scatter(
                    x=flat_x_vol, y=flat_y_vol, mode="markers",
                    marker=dict(size=sizes_vol, color=colors_vol, line=dict(width=0)),
                    hovertemplate="Strike: %{y}<br>Net Vol: %{text}<extra></extra>",
                    text=[f"{v:+,.0f}" for v in flat_z_vol],
                    showlegend=False, name="Volume Flow",
                ))
                # Spot line
                fig_vol.add_trace(go.Scatter(
                    x=snap_x_positions, y=[s["spot"] for s in snapshots],
                    mode="lines", line=dict(color="#44CCFF", width=2.5),
                    name="SPX", hovertemplate="SPX: %{y:.2f}<extra></extra>",
                ))
                fig_vol.update_layout(
                    template=theme["template"], height=400,
                    title_text="Volume Flow — Net (Call Vol − Put Vol) per Interval",
                    xaxis=dict(title="Time (ET)", range=[0, RTH_MINUTES],
                               tickmode="array", tickvals=tick_positions, ticktext=tick_labels,
                               tickangle=-45),
                    yaxis=dict(title="Strike", dtick=5, range=[lo_strike-2.5, hi_strike+2.5]),
                    showlegend=True, legend=dict(x=0.01, y=0.99, bgcolor="rgba(10,22,40,0.7)"),
                    margin=dict(l=60, r=40, t=40, b=60),
                )
                fig_vol = apply_plotly_theme(fig_vol)
                st.plotly_chart(fig_vol, width="stretch")
        else:
            st.caption("Volume flow requires 2+ snapshots — waiting for next fetch.")

        latest = snapshots[-1]
        st.markdown(f"**Current:** SPX {latest['spot']:,.2f} · Expiry: {latest['expiry']} · "
                    f"Day range: {min(s['spot'] for s in snapshots):,.2f} - "
                    f"{max(s['spot'] for s in snapshots):,.2f}")

    interval_map_fragment()


st.divider()
st.caption(f"Data: Barchart · Spot: tvdatafeed TVC:SPX · Last refresh: {ts}")
