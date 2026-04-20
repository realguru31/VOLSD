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
                   fetch_full_chain, pick_nearest_expiry)
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
            color: #c0d0e8 !important; }
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
            border: 1px solid #2a3f5f !important; font-family: monospace; }
        .stButton > button:hover {
            background-color: #1c2f50 !important; border-color: #4a6090 !important;
            color: #ffffff !important; }
        div[data-baseweb="select"] > div {
            background-color: #13213a !important; border-color: #2a3f5f !important;
            color: #c0d0e8 !important; }
        div[data-baseweb="popover"] { background-color: #13213a !important; }
        li[role="option"] { background-color: #13213a !important; color: #c0d0e8 !important; }
        li[role="option"]:hover { background-color: #1c2f50 !important; }
        input[type="text"], input[type="password"] {
            background-color: #13213a !important; color: #c0d0e8 !important;
            border-color: #2a3f5f !important; }
        div[data-testid="stAlert"] {
            background-color: rgba(30,60,90,0.3) !important;
            border: 1px solid #2a4060 !important; color: #88BBDD !important; }
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

    # OPEX fix: skip AM monthly if today is monthly expiry
    nearest = pick_nearest_expiry(future_weekly, monthly)
    if not nearest:
        return False

    calls, puts = fetch_full_chain(sess, headers, nearest, is_dense=False)
    if calls.empty or puts.empty:
        return False

    gex_df = compute_gex_vex(calls, puts, spot_now)

    call_vols, put_vols = {}, {}
    for _, row in calls.iterrows():
        call_vols[row["strikePrice"]] = row.get("volume", 0) or 0
    for _, row in puts.iterrows():
        put_vols[row["strikePrice"]] = row.get("volume", 0) or 0

    et_now = datetime.now(ET)
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


def fetch_tv_1min_bars():
    """Fetch full-day 1-min SPX bars from TVC:SPX for continuous spot line."""
    try:
        from tvDatafeed import TvDatafeed, Interval
        tv = TvDatafeed()
        df = tv.get_hist(symbol="SPX", exchange="TVC",
                         interval=Interval.in_1_minute, n_bars=390)
        if df is not None and not df.empty:
            df = df.copy()
            if df.index.tz is None:
                try:
                    df.index = df.index.tz_localize("UTC")
                except Exception:
                    pass
            try:
                df.index = df.index.tz_convert(ET)
            except Exception:
                pass
            return df
    except Exception:
        pass
    return None


# ═══════════════════════════════════════
# Interval Map Helpers
# ═══════════════════════════════════════
RTH_MINUTES = 390
RTH_TICKS = list(range(0, RTH_MINUTES + 1, 30))
RTH_LABELS = []
for _m in RTH_TICKS:
    _h = 9 + (_m + 30) // 60
    _mn = (_m + 30) % 60
    _ap = "AM" if _h < 12 else "PM"
    if _h > 12: _h -= 12
    RTH_LABELS.append(f"{_h}:{_mn:02d} {_ap}")

DODGER_BLUE = "#1E90FF"
CRIMSON = "#DC143C"


def build_dot_scatter(flat_x, flat_y, flat_z, name=""):
    """Sized + colored dot scatter: sqrt scaling, p95 cap, dodger blue / crimson."""
    flat_z = np.array(flat_z, dtype=float)
    abs_z = np.abs(flat_z)
    nonzero = abs_z[abs_z > 0]
    p95 = float(np.percentile(nonzero, 95)) if len(nonzero) > 0 else 1.0
    p95 = max(p95, 0.001)
    capped = np.minimum(abs_z / p95, 1.0)
    sizes = 4 + 22 * np.sqrt(capped)

    colors = []
    threshold = p95 * 0.02
    for v in flat_z:
        if v > threshold:
            colors.append("rgba(30,144,255,0.45)")
        elif v < -threshold:
            colors.append("rgba(220,20,60,0.45)")
        else:
            colors.append("rgba(100,120,150,0.10)")

    return go.Scatter(
        x=list(flat_x), y=list(flat_y), mode="markers",
        marker=dict(size=sizes.tolist(), color=colors, line=dict(width=0)),
        hovertemplate="Strike: %{y}<br>Value: %{text}<extra></extra>",
        text=[f"{v:+.2f}" for v in flat_z],
        showlegend=False, name=name,
    )


def build_spot_line(tv_bars_df, snap_spots, snap_x_positions):
    """White spot line. Uses 1-min TV bars filtered to today RTH only."""
    traces = []
    today_et = datetime.now(ET).strftime("%Y-%m-%d")

    if tv_bars_df is not None and not tv_bars_df.empty:
        spot_x, spot_y = [], []
        for ts_idx, row in tv_bars_df.iterrows():
            bar_time = ts_idx
            if not hasattr(bar_time, 'hour'):
                continue
            # Filter: today's date only, RTH hours only
            bar_date = bar_time.strftime("%Y-%m-%d") if hasattr(bar_time, 'strftime') else ""
            if bar_date != today_et:
                continue
            mfo = (bar_time.hour - 9) * 60 + bar_time.minute - 30
            if 0 <= mfo <= RTH_MINUTES:
                spot_x.append(mfo)
                spot_y.append(row["close"])
        if spot_x:
            traces.append(go.Scatter(
                x=spot_x, y=spot_y, mode="lines",
                line=dict(color="#FFFFFF", width=3),
                name="SPX",
                hovertemplate="SPX: %{y:.2f}<extra></extra>",
            ))
            return traces

    # Fallback: snapshot spots
    if snap_spots and snap_x_positions:
        traces.append(go.Scatter(
            x=list(snap_x_positions), y=list(snap_spots), mode="lines",
            line=dict(color="#FFFFFF", width=3),
            name="SPX",
            hovertemplate="SPX: %{y:.2f}<extra></extra>",
        ))

    return traces

    # Fallback: snapshot spots
    if snap_spots and snap_x_positions:
        traces.append(go.Scatter(
            x=list(snap_x_positions), y=list(snap_spots), mode="lines",
            line=dict(color="#FFFFFF", width=3),
            name="SPX",
            hovertemplate="SPX: %{y:.2f}<extra></extra>",
        ))

    return traces


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

    # OPEX fix: pick correct 0DTE expiry
    nearest = pick_nearest_expiry(future_weekly, monthly)
    monthlies = future_monthly[:3]

    chains = {}
    if nearest:
        c, p = fetch_full_chain(sess, headers, nearest, is_dense=False)
        if not c.empty and not p.empty:
            dte = c["daysToExpiration"].iloc[0] if not c.empty else 0
            chains[nearest] = {"calls": c, "puts": p, "label": f"0DTE ({dte:.0f}d)"}

    for exp in monthlies:
        if exp == nearest:
            continue  # don't double-fetch
        c, p = fetch_full_chain(sess, headers, exp, is_dense=True)
        if not c.empty and not p.empty:
            dte = c["daysToExpiration"].iloc[0] if not c.empty else 0
            chains[exp] = {"calls": c, "puts": p, "label": f"Monthly ({dte:.0f}d)"}

    return {
        "spot": spot, "nearest": nearest, "monthlies": monthlies,
        "monthly_list": monthly,
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

# Front monthly for BSM tabs — skip DTE=0 (expired AM monthly)
front_exp = None
for m_exp in target_monthlies:
    if m_exp in chains:
        c_df = chains[m_exp]["calls"]
        if not c_df.empty and c_df["daysToExpiration"].iloc[0] > 0:
            front_exp = m_exp
            break
if front_exp is None:
    for exp, chain in chains.items():
        if not chain["calls"].empty and chain["calls"]["daysToExpiration"].iloc[0] > 0:
            front_exp = exp
            break
if front_exp is None:
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
    st.markdown("### SPX Dealer Risk Monitor")
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
    charm_total = compute_charm_for_expiry(front_calls, front_puts, spot)
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
        z=surface, x=(spot_pcts*100).tolist(), y=(iv_shocks*100).tolist(),
        colorscale=[[0,"#8B0000"],[0.2,"#CC2222"],[0.4,"#441111"],
                     [0.48,"#1a0505"],[0.5,NAVY_BG],[0.52,"#0a1020"],
                     [0.6,"#1a2850"],[0.8,"#3366CC"],[1,"#4488FF"]],
        zmid=0, zsmooth="best", colorbar=dict(title="GEX+ ($M)"),
        hovertemplate="Spot: %{x:.0f}%<br>IV: %{y:.0f}pts<br>GEX+: %{z:.1f}M<extra></extra>"))
    fig_hm.add_hline(y=0, line_dash="dash", line_color="white", line_width=1.5)
    fig_hm.add_vline(x=0, line_dash="dash", line_color="white", line_width=1.5)
    fig_hm.add_annotation(x=0, y=0, text="◉", showarrow=False, font=dict(color="white", size=18))
    fig_hm.add_annotation(x=-12, y=35, text="DANGER ZONE", showarrow=False,
                           font=dict(color="#FF6688",size=12), bgcolor="rgba(80,0,0,0.6)")
    fig_hm.add_annotation(x=7, y=-7, text="SAFE ZONE", showarrow=False,
                           font=dict(color="#66AAFF",size=12), bgcolor="rgba(0,20,60,0.6)")
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
        mask = (front_gex["strike"] > spot*0.92) & (front_gex["strike"] < spot*1.08)
        df_plot = front_gex[mask]
        fig_gex = make_subplots(rows=2, cols=1,
            subplot_titles=["GEX+ by Strike","GEX vs VEX Decomposition"], vertical_spacing=0.12)
        colors = [theme["green"] if v>=0 else theme["red"] for v in df_plot["gex_plus"]]
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
    for cp in [-5,-10,-15,-20]:
        gp_val = float(np.interp(cp, profile["pct_move"], profile["gex_plus"]))
        color = theme["red"] if gp_val<0 else theme["green"]
        fig_prof.add_trace(go.Scatter(x=[cp], y=[gp_val], mode="markers+text",
            marker=dict(size=10,color=color), text=[f"{cp}%"],
            textposition="top center", textfont=dict(color=color,size=10), showlegend=False))
    cur_gp = float(np.interp(0, profile["pct_move"], profile["gex_plus"]))
    fig_prof.add_annotation(x=0, y=cur_gp, text=f"SPX {spot:,.0f}<br>GEX+ {cur_gp:.1f}M",
        showarrow=True, arrowhead=2, font=dict(size=11,color="white"),
        bgcolor="rgba(10,22,40,0.9)", bordercolor="white")
    fig_prof.update_layout(template=theme["template"], height=400,
        xaxis_title="Spot Move (%)", yaxis_title="GEX+ ($M)", hovermode="x unified")
    fig_prof = apply_plotly_theme(fig_prof)
    st.plotly_chart(fig_prof, width="stretch")

    st.markdown("#### Crash Risk — GEX+ at Drawdown Levels")
    crash_cols = st.columns(4)
    for i, cp in enumerate([-5,-10,-15,-20]):
        cs = spot*(1+cp/100)
        gex, vex, gp = compute_gex_plus_at_spot(front_calls, front_puts, cs)
        sev = "ELEVATED" if gp<0 else ("NEUTRAL" if gp<50 else "CONTAINED")
        sc = "crash-elevated" if gp<0 else ("crash-neutral" if gp<50 else "crash-contained")
        with crash_cols[i]:
            st.markdown(f"""<div class="crash-card {sc}">
                <div style="font-size:20px;font-weight:bold;font-family:monospace;color:#e0e8f5;">{cp}%</div>
                <div style="font-size:11px;color:#7a93b8;margin-top:4px;">CRASH SPOT: {cs:,.0f}<br>MARKET IV: ~{20+abs(cp)*0.4:.0f}%</div>
                <div style="font-size:28px;font-weight:bold;font-family:monospace;margin:8px 0;color:{'#FF6688' if gp<0 else '#66FF99'};">{gp:.0f}M</div>
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
        for col, (pct,label) in zip(pcols, [(5,"5th"),(25,"25th"),(50,"MEDIAN"),(75,"75th"),(95,"95th")]):
            val = bl["pctiles_exp"][pct]
            with col:
                st.markdown(f"""<div class="metric-card"><div class="metric-label">{label} PCTILE</div>
                    <div class="metric-value">{val:,.1f}</div></div>""", unsafe_allow_html=True)
        st.markdown(f"**1σ Move:** ±{bl['sigma_exp']:.0f} pts · **ATM IV:** {bl['atm_iv']*100:.1f}% · "
                    f"**Skew:** {bl['skew']:.3f} · **Kurtosis:** {bl['kurt']:.2f}")
        if bl["prob_table"]:
            prob_df = pd.DataFrame(bl["prob_table"])
            prob_df.columns = ["SPX Level","P(below)","P(above)"]
            prob_df["P(below)"] = prob_df["P(below)"].apply(lambda x: f"{x:.1f}%")
            prob_df["P(above)"] = prob_df["P(above)"].apply(lambda x: f"{x:.1f}%")
            st.dataframe(prob_df, width="stretch", hide_index=True)
        if len(bl["density_strikes"]) > 5:
            fig_den = go.Figure()
            fig_den.add_trace(go.Scatter(x=bl["density_strikes"], y=bl["density_vals"],
                fill="tozeroy", fillcolor="rgba(68,136,221,0.3)",
                line=dict(color=theme["blue"],width=2), name="RN Density"))
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
        lo, hi = spot*0.98, spot*1.02
        filtered = [r for r in raw_exp if lo <= r["strike"] <= hi]
        strikes = np.array([r["strike"] for r in filtered])
        gex_vals = np.array([r["net_gex"] for r in filtered])
        charm_vals = np.array([r["net_charm"] for r in filtered])
        if len(strikes) > 5:
            gcol, ccol = st.columns(2)
            with gcol:
                st.markdown(f"**Gamma** — {ts}")
                fig_g = go.Figure()
                if (gex_vals>=0).any():
                    fig_g.add_trace(go.Scatter(x=np.where(gex_vals>=0,gex_vals,0), y=strikes,
                        fill="tozerox", fillcolor="rgba(68,204,119,0.5)",
                        line=dict(color="#66DD88",width=2), name="Dampening", mode="lines"))
                if (gex_vals<0).any():
                    fig_g.add_trace(go.Scatter(x=np.where(gex_vals<0,gex_vals,0), y=strikes,
                        fill="tozerox", fillcolor="rgba(255,68,102,0.5)",
                        line=dict(color="#FF6688",width=2), name="Amplifying", mode="lines"))
                fig_g.add_vline(x=0, line_color="gray", line_dash="dash")
                fig_g.add_hline(y=spot, line_color="white", line_width=2, annotation_text=f"SPX {spot:.0f}")
                fig_g.update_layout(template=theme["template"], height=600,
                    xaxis_title="Net GEX ($B)", yaxis_title="Strike", hovermode="y unified")
                fig_g = apply_plotly_theme(fig_g)
                st.plotly_chart(fig_g, width="stretch")
            with ccol:
                st.markdown(f"**Charm** — {ts}")
                fig_c = go.Figure()
                if (charm_vals>=0).any():
                    fig_c.add_trace(go.Scatter(x=np.where(charm_vals>=0,charm_vals,0), y=strikes,
                        fill="tozerox", fillcolor="rgba(68,136,221,0.5)",
                        line=dict(color="#4488DD",width=2), name="Buying support", mode="lines"))
                if (charm_vals<0).any():
                    fig_c.add_trace(go.Scatter(x=np.where(charm_vals<0,charm_vals,0), y=strikes,
                        fill="tozerox", fillcolor="rgba(221,170,68,0.5)",
                        line=dict(color="#DDAA44",width=2), name="Selling pressure", mode="lines"))
                fig_c.add_vline(x=0, line_color="gray", line_dash="dash")
                fig_c.add_hline(y=spot, line_color="white", line_width=2, annotation_text=f"SPX {spot:.0f}")
                fig_c.update_layout(template=theme["template"], height=600,
                    xaxis_title="Net Charm", yaxis_title="Strike", hovermode="y unified")
                fig_c = apply_plotly_theme(fig_c)
                st.plotly_chart(fig_c, width="stretch")
    else:
        st.warning("No 0DTE chain available.")

# ═══════════════════════════════════════
# TAB 6: INTERVAL MAP — Fragment
# ═══════════════════════════════════════
with tab6:

    @st.fragment(run_every=60)
    def interval_map_fragment():
        st.markdown("#### Interval Map — 0DTE GEX + Volume Flow")
        st.caption("**GEX:** 5-min intervals. **Volume:** configurable. "
                   "Blue = positive (call/dampening). Crimson = negative (put/amplifying). "
                   "White line = SPX (1-min TVC:SPX).")

        ctrl_cols = st.columns([1.4, 1.2, 0.9, 0.9, 1.0])
        with ctrl_cols[0]:
            pct_range = st.selectbox("Y-axis range",
                options=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
                format_func=lambda x: f"±{x}%", index=1, key="pct_range_select")
        with ctrl_cols[1]:
            vol_refresh_min = st.selectbox("Vol refresh",
                options=[1, 2, 3], format_func=lambda x: f"{x} min",
                index=1, key="vol_refresh_select")
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

        st.caption("GEX refresh: fixed 5 min")

        # Auto-fetch
        if auto_refresh:
            et_now = datetime.now(ET)
            is_weekday = et_now.weekday() < 5
            mo = et_now.replace(hour=9, minute=30, second=0, microsecond=0)
            mc = et_now.replace(hour=16, minute=0, second=0, microsecond=0)
            in_market = is_weekday and mo <= et_now <= mc
            if in_market:
                if "last_interval_fetch" not in st.session_state:
                    st.session_state.last_interval_fetch = 0
                refresh_seconds = vol_refresh_min * 60
                if time.time() - st.session_state.last_interval_fetch >= refresh_seconds:
                    take_interval_snapshot()
                    st.session_state.last_interval_fetch = time.time()

        snapshots = load_snapshots()

        if not snapshots:
            st.info("No snapshots yet. Click 'Fetch' or wait for auto-refresh (9:30-4:00 ET weekdays).")
            return

        refresh_sec = vol_refresh_min * 60
        remaining = max(0, refresh_sec - int(time.time() - st.session_state.get("last_interval_fetch", 0)))
        st.caption(f"Snapshots: **{len(snapshots)}** · Latest: {snapshots[-1]['time_label']} · Next: ~{remaining}s")

        # Strike grid
        latest_spot = snapshots[-1]["spot"]
        range_pts = latest_spot * pct_range / 100
        lo_strike = int(round((latest_spot - range_pts) / 5.0)) * 5
        hi_strike = int(round((latest_spot + range_pts) / 5.0)) * 5
        strike_levels = np.arange(lo_strike, hi_strike + 5, 5)

        # Snap X positions
        snap_x = [min(s.get("minutes_from_open", 0), RTH_MINUTES) for s in snapshots]
        snap_spots = [s["spot"] for s in snapshots]

        # TV 1-min bars for spot line
        tv_bars = fetch_tv_1min_bars()

        # Filter to 5-min for GEX chart
        gex_snaps, gex_snap_x = [], []
        last_bucket = -999
        for idx, snap in enumerate(snapshots):
            mfo = snap.get("minutes_from_open", 0)
            bucket = int(mfo // 5) * 5
            if bucket > last_bucket:
                gex_snaps.append(snap)
                gex_snap_x.append(snap_x[idx])
                last_bucket = bucket

        # ── CHART 1: GEX Landscape ──
        flat_x_g, flat_y_g, flat_z_g = [], [], []
        for t_idx, snap in enumerate(gex_snaps):
            strikes_arr = np.array(snap["strikes"])
            gex_arr = np.array(snap["gex_plus"])
            for K in strike_levels:
                matches = np.where(strikes_arr == K)[0]
                if len(matches) > 0:
                    val = gex_arr[matches[0]]
                elif len(strikes_arr) >= 2 and strikes_arr.min() <= K <= strikes_arr.max():
                    val = float(np.interp(K, strikes_arr, gex_arr))
                else:
                    val = 0
                flat_x_g.append(gex_snap_x[t_idx])
                flat_y_g.append(K)
                flat_z_g.append(val)

        fig_gex_map = go.Figure()
        fig_gex_map.add_trace(build_dot_scatter(flat_x_g, flat_y_g, flat_z_g, "GEX+"))
        for tr in build_spot_line(tv_bars, snap_spots, snap_x):
            fig_gex_map.add_trace(tr)
        fig_gex_map.update_layout(
            template=theme["template"], height=380,
            title_text="GEX Landscape — 5 min intervals",
            xaxis=dict(range=[0, RTH_MINUTES], tickmode="array",
                       tickvals=RTH_TICKS, ticktext=RTH_LABELS, tickangle=-45),
            yaxis=dict(title="Strike", dtick=5, range=[lo_strike-2.5, hi_strike+2.5]),
            showlegend=True, legend=dict(x=0.01, y=0.99, bgcolor="rgba(10,22,40,0.7)"),
            margin=dict(l=60, r=40, t=40, b=50))
        fig_gex_map = apply_plotly_theme(fig_gex_map)
        st.plotly_chart(fig_gex_map, width="stretch")

        # ── CHART 2: Volume Incremental ──
        if len(snapshots) >= 2:
            flat_x_vi, flat_y_vi, flat_z_vi = [], [], []
            for t_idx in range(1, len(snapshots)):
                prev = snapshots[t_idx - 1]
                curr = snapshots[t_idx]
                pcv, ppv = prev.get("call_volumes", {}), prev.get("put_volumes", {})
                ccv, cpv = curr.get("call_volumes", {}), curr.get("put_volumes", {})
                for K in strike_levels:
                    Kf = float(K)
                    net = (ccv.get(Kf, 0) - pcv.get(Kf, 0)) - (cpv.get(Kf, 0) - ppv.get(Kf, 0))
                    flat_x_vi.append(snap_x[t_idx])
                    flat_y_vi.append(K)
                    flat_z_vi.append(net)

            fig_vol_inc = go.Figure()
            fig_vol_inc.add_trace(build_dot_scatter(flat_x_vi, flat_y_vi, flat_z_vi, "Incremental"))
            for tr in build_spot_line(tv_bars, snap_spots, snap_x):
                fig_vol_inc.add_trace(tr)
            fig_vol_inc.update_layout(
                template=theme["template"], height=380,
                title_text="Volume Flow — Incremental (delta from previous)",
                xaxis=dict(range=[0, RTH_MINUTES], tickmode="array",
                           tickvals=RTH_TICKS, ticktext=RTH_LABELS, tickangle=-45),
                yaxis=dict(title="Strike", dtick=5, range=[lo_strike-2.5, hi_strike+2.5]),
                showlegend=True, legend=dict(x=0.01, y=0.99, bgcolor="rgba(10,22,40,0.7)"),
                margin=dict(l=60, r=40, t=40, b=50))
            fig_vol_inc = apply_plotly_theme(fig_vol_inc)
            st.plotly_chart(fig_vol_inc, width="stretch")
        else:
            st.caption("Volume Incremental — needs 2+ snapshots.")

        # ── CHART 3: Volume Cumulative ──
        if len(snapshots) >= 2:
            first = snapshots[0]
            first_cv, first_pv = first.get("call_volumes", {}), first.get("put_volumes", {})
            flat_x_vc, flat_y_vc, flat_z_vc = [], [], []
            for t_idx in range(1, len(snapshots)):
                curr = snapshots[t_idx]
                ccv, cpv = curr.get("call_volumes", {}), curr.get("put_volumes", {})
                for K in strike_levels:
                    Kf = float(K)
                    net = (ccv.get(Kf, 0) - first_cv.get(Kf, 0)) - (cpv.get(Kf, 0) - first_pv.get(Kf, 0))
                    flat_x_vc.append(snap_x[t_idx])
                    flat_y_vc.append(K)
                    flat_z_vc.append(net)

            fig_vol_cum = go.Figure()
            fig_vol_cum.add_trace(build_dot_scatter(flat_x_vc, flat_y_vc, flat_z_vc, "Cumulative"))
            for tr in build_spot_line(tv_bars, snap_spots, snap_x):
                fig_vol_cum.add_trace(tr)
            fig_vol_cum.update_layout(
                template=theme["template"], height=380,
                title_text="Volume Flow — Cumulative (delta from first snapshot)",
                xaxis=dict(title="Time (ET)", range=[0, RTH_MINUTES], tickmode="array",
                           tickvals=RTH_TICKS, ticktext=RTH_LABELS, tickangle=-45),
                yaxis=dict(title="Strike", dtick=5, range=[lo_strike-2.5, hi_strike+2.5]),
                showlegend=True, legend=dict(x=0.01, y=0.99, bgcolor="rgba(10,22,40,0.7)"),
                margin=dict(l=60, r=40, t=40, b=60))
            fig_vol_cum = apply_plotly_theme(fig_vol_cum)
            st.plotly_chart(fig_vol_cum, width="stretch")
        else:
            st.caption("Volume Cumulative — needs 2+ snapshots.")

        latest = snapshots[-1]
        st.markdown(f"**Current:** SPX {latest['spot']:,.2f} · Expiry: {latest['expiry']} · "
                    f"Day range: {min(s['spot'] for s in snapshots):,.2f} - "
                    f"{max(s['spot'] for s in snapshots):,.2f}")

    interval_map_fragment()

st.divider()
st.caption(f"Data: Barchart · Spot: tvdatafeed TVC:SPX · Last refresh: {ts}")
