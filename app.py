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


# Navy color palette
NAVY_BG = "#0a1628"
NAVY_CARD = "#13213a"
NAVY_BORDER = "#1e3050"
NAVY_TEXT = "#c0d0e8"
NAVY_TEXT_DIM = "#7a93b8"
NAVY_TEXT_MUTED = "#5a7090"


def get_theme():
    if st.session_state.dark_mode:
        return {
            "template": "plotly_dark",
            "bg": NAVY_BG, "card_bg": NAVY_CARD, "text": NAVY_TEXT,
            "accent": "#44CCFF", "red": "#FF4466", "green": "#44CC77",
            "blue": "#4488DD", "gold": "#DDAA44", "muted": NAVY_TEXT_MUTED,
            "grid": "#1e3050",
        }
    else:
        return {
            "template": "plotly_white",
            "bg": "#ffffff", "card_bg": "#f5f5f5", "text": "#1a1a1a",
            "accent": "#008866", "red": "#CC2222", "green": "#228822",
            "blue": "#2255BB", "gold": "#AA7700", "muted": "#999999",
            "grid": "#dddddd",
        }

theme = get_theme()


def apply_plotly_theme(fig):
    """Apply navy background to any plotly figure in dark mode."""
    if st.session_state.dark_mode:
        fig.update_layout(
            paper_bgcolor=NAVY_BG,
            plot_bgcolor=NAVY_BG,
            font=dict(color=NAVY_TEXT),
        )
        fig.update_xaxes(gridcolor="#1e3050", zerolinecolor="#2a3f5f")
        fig.update_yaxes(gridcolor="#1e3050", zerolinecolor="#2a3f5f")
    return fig


# CSS
if st.session_state.dark_mode:
    st.markdown("""<style>
        /* Hide Streamlit default chrome */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden; height: 0;}
        footer {visibility: hidden;}
        .stDeployButton {display: none;}
        [data-testid="stToolbar"] {visibility: hidden; height: 0; position: fixed;}
        [data-testid="stDecoration"] {visibility: hidden; height: 0; position: fixed;}
        [data-testid="stStatusWidget"] {visibility: hidden; height: 0; position: fixed;}
        [data-testid="stHeader"] {display: none;}

        /* Navy background everywhere */
        .stApp { background-color: #0a1628 !important; }
        [data-testid="stAppViewContainer"] { background-color: #0a1628 !important; }
        [data-testid="stMain"] { background-color: #0a1628 !important; }
        .main { background-color: #0a1628 !important; }
        .block-container { padding-top: 1rem !important; background-color: #0a1628 !important; }

        /* Body text */
        .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
        .stMarkdown h4, .stMarkdown h5, .stMarkdown li, p, h1, h2, h3, h4, h5 {
            color: #c0d0e8 !important;
        }

        /* Metric cards */
        .metric-card { background: #13213a; border: 1px solid #1e3050; border-radius: 8px;
                       padding: 12px 16px; text-align: center; }
        .metric-label { color: #7a93b8; font-size: 11px; text-transform: uppercase; font-family: monospace; }
        .metric-value { color: #e0e8f5; font-size: 22px; font-weight: bold; font-family: monospace; }
        .metric-sub { color: #4a5d7e; font-size: 10px; font-family: monospace; }

        /* Regime badges */
        .regime-badge { padding: 4px 12px; border-radius: 4px; font-weight: bold; font-family: monospace; display: inline-block; }
        .regime-amp { background: #2a1520; color: #FF6688; border: 1px solid #FF4466; }
        .regime-damp { background: #152a1f; color: #66FF99; border: 1px solid #44CC77; }

        /* Crash cards */
        .crash-card { background: #13213a; border: 1px solid #1e3050; border-radius: 8px; padding: 16px; margin: 8px 0; }
        .crash-elevated { border-left: 4px solid #FF4466; }
        .crash-neutral { border-left: 4px solid #5a7090; }
        .crash-contained { border-left: 4px solid #4488DD; }

        /* Buttons */
        .stButton > button {
            background-color: #13213a !important;
            color: #c0d0e8 !important;
            border: 1px solid #2a3f5f !important;
            font-family: monospace;
        }
        .stButton > button:hover {
            background-color: #1c2f50 !important;
            border-color: #4a6090 !important;
            color: #ffffff !important;
        }

        /* Dropdowns */
        div[data-baseweb="select"] > div {
            background-color: #13213a !important;
            border-color: #2a3f5f !important;
            color: #c0d0e8 !important;
        }
        div[data-baseweb="popover"] {
            background-color: #13213a !important;
        }
        div[data-baseweb="select"] input {
            color: #c0d0e8 !important;
        }
        li[role="option"] {
            background-color: #13213a !important;
            color: #c0d0e8 !important;
        }
        li[role="option"]:hover {
            background-color: #1c2f50 !important;
        }

        /* Text inputs */
        input[type="text"], input[type="password"] {
            background-color: #13213a !important;
            color: #c0d0e8 !important;
            border-color: #2a3f5f !important;
        }

        /* Alerts */
        div[data-testid="stAlert"] {
            background-color: rgba(30,60,90,0.3) !important;
            border: 1px solid #2a4060 !important;
            color: #88BBDD !important;
        }

        /* Tabs */
        [data-testid="stTabs"] button {
            color: #7a93b8 !important;
        }
        [data-testid="stTabs"] button[aria-selected="true"] {
            color: #88BBFF !important;
        }

        /* Dividers */
        hr { border-color: #1e3050 !important; }

        /* Dataframe */
        [data-testid="stDataFrame"] {
            background-color: #13213a !important;
        }

        /* Labels above widgets */
        label, [data-testid="stWidgetLabel"] {
            color: #c0d0e8 !important;
        }

        /* Captions */
        .stCaption, [data-testid="stCaptionContainer"] {
            color: #7a93b8 !important;
        }

        /* Plotly chart container background */
        [data-testid="stPlotlyChart"] {
            background-color: #0a1628 !important;
        }
        .js-plotly-plot {
            background-color: #0a1628 !important;
        }
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


def snapshot_file_for_today():
    et_date = datetime.now(pytz.timezone("US/Eastern")).strftime("%Y%m%d")
    return os.path.join(SNAPSHOT_DIR, f"intervals_{et_date}.pkl")


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
    et_now = datetime.now(pytz.timezone("US/Eastern"))
    minute_key = et_now.strftime("%H%M")
    snapshots = [s for s in snapshots if s.get("minute_key") != minute_key]
    snapshot["minute_key"] = minute_key
    snapshots.append(snapshot)
    path = snapshot_file_for_today()
    with open(path, "wb") as f:
        pickle.dump(snapshots, f)


def take_interval_snapshot():
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

    et_now = datetime.now(pytz.timezone("US/Eastern"))
    snapshot = {
        "timestamp": et_now.isoformat(),
        "time_label": et_now.strftime("%I:%M %p"),
        "unix_ts": et_now.timestamp(),
        "spot": spot_now,
        "expiry": nearest,
        "strikes": gex_df["strike"].tolist(),
        "gex_plus": gex_df["gex_plus"].tolist(),
        "gex": gex_df["gex"].tolist(),
    }
    save_snapshot(snapshot)
    return True


# ═══════════════════════════════════════
# Main Data Loading
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

    x_vals = (spot_pcts * 100).tolist()
    y_vals = (iv_shocks * 100).tolist()

    fig_hm = go.Figure()
    fig_hm.add_trace(go.Heatmap(
        z=surface, x=x_vals, y=y_vals,
        colorscale=[
            [0, "#8B0000"], [0.20, "#CC2222"], [0.40, "#441111"],
            [0.48, "#1a0505"], [0.5, "#0a1628"], [0.52, "#0a1020"],
            [0.60, "#1a2850"], [0.80, "#3366CC"], [1, "#4488FF"],
        ],
        zmid=0, zsmooth="best",
        colorbar=dict(title="GEX+ ($M)"),
        hovertemplate="Spot: %{x:.0f}%<br>IV: %{y:.0f}pts<br>GEX+: %{z:.1f}M<extra></extra>",
    ))
    fig_hm.add_hline(y=0, line_dash="dash", line_color="white", line_width=1.5)
    fig_hm.add_vline(x=0, line_dash="dash", line_color="white", line_width=1.5)
    fig_hm.add_annotation(x=0, y=0, text="◉", showarrow=False,
                           font=dict(color="white", size=18))
    fig_hm.add_annotation(x=-12, y=35, text="DANGER ZONE", showarrow=False,
                           font=dict(color="#FF6688", size=12), bgcolor="rgba(80,0,0,0.6)")
    fig_hm.add_annotation(x=7, y=-7, text="SAFE ZONE", showarrow=False,
                           font=dict(color="#66AAFF", size=12), bgcolor="rgba(0,20,60,0.6)")

    for pv in [-10, -5, 0, 5, 10]:
        s = spot * (1 + pv / 100)
        fig_hm.add_annotation(x=pv, y=max(y_vals) + 3, text=f"SPX {s:,.0f}",
            showarrow=False, font=dict(color="rgba(192,208,232,0.6)", size=9))

    fig_hm.update_layout(
        template=theme["template"], height=550,
        xaxis_title="Spot Move (%)", yaxis_title="IV Shock (vol pts)",
        xaxis=dict(dtick=5), yaxis=dict(dtick=10),
    )
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
            textposition="top center", textfont=dict(color=color, size=10),
            showlegend=False))

    cur_gp = float(np.interp(0, profile["pct_move"], profile["gex_plus"]))
    fig_prof.add_annotation(x=0, y=cur_gp,
        text=f"SPX {spot:,.0f}<br>GEX+ {cur_gp:.1f}M", showarrow=True, arrowhead=2,
        font=dict(size=11, color="white"), bgcolor="rgba(10,22,40,0.9)", bordercolor="white")
    fig_prof.update_layout(template=theme["template"], height=400,
        xaxis_title="Spot Move (%)", yaxis_title="GEX+ ($M)", hovermode="x unified")
    fig_prof = apply_plotly_theme(fig_prof)
    st.plotly_chart(fig_prof, width="stretch")

    st.markdown("#### Crash Risk — GEX+ at Drawdown Levels")
    crash_cols = st.columns(4)
    for i, cp in enumerate([-5, -10, -15, -20]):
        cs = spot * (1 + cp / 100)
        gex, vex, gp = compute_gex_plus_at_spot(front_calls, front_puts, cs)
        if gp < 0:
            sev, sev_class = "ELEVATED", "crash-elevated"
        elif gp < 50:
            sev, sev_class = "NEUTRAL", "crash-neutral"
        else:
            sev, sev_class = "CONTAINED", "crash-contained"

        with crash_cols[i]:
            st.markdown(f"""<div class="crash-card {sev_class}">
                <div style="font-size:20px;font-weight:bold;font-family:monospace;color:#e0e8f5;">{cp}%</div>
                <div style="font-size:11px;color:#7a93b8;margin-top:4px;">
                    CRASH SPOT: {cs:,.0f}<br>MARKET IV: ~{20+abs(cp)*0.4:.0f}%
                </div>
                <div style="font-size:28px;font-weight:bold;font-family:monospace;margin:8px 0;
                            color:{'#FF6688' if gp < 0 else '#66FF99'};">
                    {gp:.0f}M
                </div>
                <div style="font-size:10px;color:#7a93b8;">
                    GEX: {gex:.1f}M · VEX: {vex:.1f}M
                </div>
                <div style="font-size:11px;font-weight:bold;margin-top:6px;
                            color:{'#FF6688' if sev=='ELEVATED' else '#7a93b8'};">
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
                fill="tozeroy", fillcolor="rgba(68,136,221,0.3)",
                line=dict(color=theme["blue"], width=2), name="RN Density"))
            fig_den.add_vline(x=spot, line_dash="dash", line_color="white",
                annotation_text=f"SPX {spot:.0f}")
            fig_den.add_vline(x=bl["mean"], line_dash="dot", line_color=theme["accent"],
                annotation_text=f"BL Mean {bl['mean']:.0f}")
            fig_den.update_layout(template=theme["template"], height=350,
                xaxis_title="SPX at Expiry", yaxis_title="Probability Density")
            fig_den = apply_plotly_theme(fig_den)
            st.plotly_chart(fig_den, width="stretch")
    else:
        st.warning("Insufficient OTM option data for Breeden-Litzenberger density extraction.")


# ── TAB 5: 0DTE PROFILE ──
with tab5:
    st.markdown("#### 0DTE Exposure Profile — Current Snapshot")
    st.caption("Vertical slice of current exposure curve.")

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
                pos_mask = gex_vals >= 0
                if pos_mask.any():
                    fig_g.add_trace(go.Scatter(
                        x=np.where(pos_mask, gex_vals, 0),
                        y=strikes, fill="tozerox",
                        fillcolor="rgba(68,204,119,0.5)",
                        line=dict(color="#66DD88", width=2),
                        name="Dampening (+)", mode="lines",
                    ))
                neg_mask = gex_vals < 0
                if neg_mask.any():
                    fig_g.add_trace(go.Scatter(
                        x=np.where(neg_mask, gex_vals, 0),
                        y=strikes, fill="tozerox",
                        fillcolor="rgba(255,68,102,0.5)",
                        line=dict(color="#FF6688", width=2),
                        name="Amplifying (-)", mode="lines",
                    ))
                fig_g.add_vline(x=0, line_color="gray", line_dash="dash")
                fig_g.add_hline(y=spot, line_color="white", line_width=2,
                    annotation_text=f"SPX {spot:.0f}")
                fig_g.update_layout(template=theme["template"], height=600,
                    xaxis_title="Net GEX ($B)", yaxis_title="Strike",
                    hovermode="y unified")
                fig_g = apply_plotly_theme(fig_g)
                st.plotly_chart(fig_g, width="stretch")

            with ccol:
                st.markdown(f"**Charm** — {ts}")
                fig_c = go.Figure()
                pos_mask = charm_vals >= 0
                if pos_mask.any():
                    fig_c.add_trace(go.Scatter(
                        x=np.where(pos_mask, charm_vals, 0),
                        y=strikes, fill="tozerox",
                        fillcolor="rgba(68,136,221,0.5)",
                        line=dict(color="#4488DD", width=2),
                        name="Buying support (+)", mode="lines",
                    ))
                neg_mask = charm_vals < 0
                if neg_mask.any():
                    fig_c.add_trace(go.Scatter(
                        x=np.where(neg_mask, charm_vals, 0),
                        y=strikes, fill="tozerox",
                        fillcolor="rgba(221,170,68,0.5)",
                        line=dict(color="#DDAA44", width=2),
                        name="Selling pressure (-)", mode="lines",
                    ))
                fig_c.add_vline(x=0, line_color="gray", line_dash="dash")
                fig_c.add_hline(y=spot, line_color="white", line_width=2,
                    annotation_text=f"SPX {spot:.0f}")
                fig_c.update_layout(template=theme["template"], height=600,
                    xaxis_title="Net Charm", yaxis_title="Strike",
                    hovermode="y unified")
                fig_c = apply_plotly_theme(fig_c)
                st.plotly_chart(fig_c, width="stretch")

            max_gex_idx = np.argmax(np.abs(gex_vals))
            max_charm_idx = np.argmax(np.abs(charm_vals))
            st.markdown(
                f"**Peak GEX:** {gex_vals[max_gex_idx]:+.3f}B at strike {strikes[max_gex_idx]:.0f} · "
                f"**Peak Charm:** {charm_vals[max_charm_idx]:+,.0f} at strike {strikes[max_charm_idx]:.0f}"
            )
        else:
            st.warning("Not enough strikes in view range.")
    else:
        st.warning("No 0DTE chain available.")


# ── TAB 6: INTERVAL MAP ──
with tab6:
    st.markdown("#### Interval Map — 0DTE GEX Over Time")
    st.caption("Y-axis: actual strikes ±range from spot. Blue line: SPX from TVC:SPX.")

    ctrl_cols = st.columns([1.4, 1.2, 0.9, 0.9, 1.0])

    with ctrl_cols[0]:
        pct_range = st.selectbox(
            "Y-axis range",
            options=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            format_func=lambda x: f"±{x}%",
            index=1,
            key="pct_range_select",
        )

    with ctrl_cols[1]:
        refresh_minutes = st.selectbox(
            "Refresh every",
            options=[1, 2, 3],
            format_func=lambda x: f"{x} min",
            index=1,
            key="refresh_min_select",
        )

    with ctrl_cols[2]:
        auto_refresh = st.toggle("Auto", value=True, key="auto_refresh_toggle")

    with ctrl_cols[3]:
        if st.button("🔄 Fetch", key="manual_fetch"):
            with st.spinner("Fetching..."):
                success = take_interval_snapshot()
                if success:
                    st.success("Added")
                else:
                    st.error("Failed")

    with ctrl_cols[4]:
        if st.button("🗑 Clear", key="clear_today"):
            path = snapshot_file_for_today()
            if os.path.exists(path):
                os.remove(path)
                st.success("Cleared")

    if auto_refresh:
        et_now = datetime.now(pytz.timezone("US/Eastern"))
        is_weekday = et_now.weekday() < 5
        market_open = et_now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = et_now.replace(hour=16, minute=0, second=0, microsecond=0)
        in_market = is_weekday and market_open <= et_now <= market_close

        if in_market:
            if "last_interval_fetch" not in st.session_state:
                st.session_state.last_interval_fetch = 0

            now_ts = time.time()
            refresh_seconds = refresh_minutes * 60
            if now_ts - st.session_state.last_interval_fetch >= refresh_seconds:
                take_interval_snapshot()
                st.session_state.last_interval_fetch = now_ts

            st.markdown(
                f'<meta http-equiv="refresh" content="{refresh_seconds}">',
                unsafe_allow_html=True,
            )

    snapshots = load_snapshots()

    if not snapshots:
        st.info("No snapshots yet. Click 'Fetch' or wait for auto-refresh.")
    else:
        refresh_sec = refresh_minutes * 60
        st.caption(f"Snapshots today: **{len(snapshots)}** · "
                   f"Latest: {snapshots[-1]['time_label']} · "
                   f"Next auto-fetch in ~{max(0, refresh_sec - int(time.time() - st.session_state.get('last_interval_fetch', 0)))}s")

        latest_spot = snapshots[-1]["spot"]
        range_pts = latest_spot * pct_range / 100
        lo_strike = int(round((latest_spot - range_pts) / 5.0)) * 5
        hi_strike = int(round((latest_spot + range_pts) / 5.0)) * 5
        strike_levels = np.arange(lo_strike, hi_strike + 5, 5)

        n_strikes = len(strike_levels)
        n_times = len(snapshots)
        gex_grid = np.zeros((n_strikes, n_times))

        for t_idx, snap in enumerate(snapshots):
            strikes_arr = np.array(snap["strikes"])
            gex_plus_arr = np.array(snap["gex_plus"])
            for s_idx, K in enumerate(strike_levels):
                matches = np.where(strikes_arr == K)[0]
                if len(matches) > 0:
                    gex_grid[s_idx, t_idx] = gex_plus_arr[matches[0]]
                elif len(strikes_arr) >= 2 and strikes_arr.min() <= K <= strikes_arr.max():
                    gex_grid[s_idx, t_idx] = np.interp(K, strikes_arr, gex_plus_arr)

        y_strikes, x_times = np.meshgrid(strike_levels, range(n_times), indexing="ij")
        flat_y = y_strikes.flatten()
        flat_x = x_times.flatten()
        flat_z = gex_grid.flatten()

        max_abs = max(np.abs(flat_z).max(), 0.001)
        sizes = 5 + 30 * (np.abs(flat_z) / max_abs)

        colors = []
        for v in flat_z:
            if v > max_abs * 0.05:
                intensity = min(1.0, v / max_abs)
                g = int(255 * intensity)
                colors.append(f"rgb(50,{100+g//2},50)")
            elif v < -max_abs * 0.05:
                intensity = min(1.0, -v / max_abs)
                r = int(255 * intensity)
                colors.append(f"rgb({100+r//2},50,80)")
            else:
                colors.append("rgba(100,120,150,0.3)")

        time_labels = [s["time_label"] for s in snapshots]

        fig_int = go.Figure()
        fig_int.add_trace(go.Scatter(
            x=flat_x, y=flat_y,
            mode="markers",
            marker=dict(size=sizes, color=colors, line=dict(width=0)),
            hovertemplate="Time: %{customdata}<br>Strike: %{y}<br>GEX+: %{text}<extra></extra>",
            text=[f"{v:+.2f}M" for v in flat_z],
            customdata=[time_labels[int(x)] for x in flat_x],
            showlegend=False,
            name="GEX+",
        ))

        spot_line_y = [s["spot"] for s in snapshots]
        spot_line_x = list(range(n_times))

        fig_int.add_trace(go.Scatter(
            x=spot_line_x, y=spot_line_y,
            mode="lines+markers",
            line=dict(color="#44CCFF", width=2),
            marker=dict(size=5, color="#44CCFF"),
            name="SPX (TVC)",
            hovertemplate="Time: %{text}<br>SPX: %{y:.2f}<extra></extra>",
            text=time_labels,
        ))

        tick_vals = list(range(0, n_times, max(1, n_times // 12)))
        tick_text = [time_labels[i] for i in tick_vals]

        fig_int.update_layout(
            template=theme["template"], height=650,
            xaxis=dict(title="Time (ET)", tickmode="array",
                       tickvals=tick_vals, ticktext=tick_text),
            yaxis=dict(title="Strike", dtick=5,
                       range=[lo_strike - 2.5, hi_strike + 2.5]),
            showlegend=True,
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(10,22,40,0.7)"),
            margin=dict(l=60, r=40, t=30, b=60),
        )
        fig_int = apply_plotly_theme(fig_int)
        st.plotly_chart(fig_int, width="stretch")

        latest = snapshots[-1]
        st.markdown(
            f"**Current:** SPX {latest['spot']:,.2f} · "
            f"Expiry: {latest['expiry']} · "
            f"Day range: {min(s['spot'] for s in snapshots):,.2f} - "
            f"{max(s['spot'] for s in snapshots):,.2f}"
        )


st.divider()
st.caption(f"Data: Barchart · Spot: tvdatafeed TVC:SPX · Last refresh: {ts}")
