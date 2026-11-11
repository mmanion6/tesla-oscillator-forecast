import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import yfinance as yf
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# 1. Market Cap Ceiling Calculation
# $20 Trillion / 3.752B Shares Outstanding = ~$5,330 per share limit.
# ---------------------------------------------------------------------------
CEILING_PRICE = 5330
CEILING_LOG = np.log10(CEILING_PRICE)
P0 = [-0.1, 1.8, 0.45, 7.0, 30.0, 0.2, 0.4]

# Palette
C_ACTUAL = "crimson"
C_MODEL = "#111111"
C_CHANNEL = "#9aa3ad"
C_CONSOL = "#eef2f8"
C_EXPAND = "#e6f4ea"
C_KICKOFF = "#00aa00"
C_CEILING = "#6b4fbb"
C_TEXT = "#333333"
C_MUTED = "#8a8f98"

st.set_page_config(page_title="TSLA Punctuated Growth", layout="wide")
st.markdown(
    """
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    div[data-testid="stMetric"] {
        background: #fafbfd;
        border: 1px solid #e6e9ef;
        border-radius: 10px;
        padding: 12px 16px;
    }
    div[data-testid="stMetricLabel"] p {color: #6b7280; font-size: 0.8rem; letter-spacing: 0.02em;}
    div[data-testid="stMetricValue"] {font-size: 1.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Tesla Underdamped Power-Law Staircase")
st.caption("Recalibrated Phase Analysis: Long Consolidations vs. Violent Breakouts ($20T Cap)")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def load_data():
    df = yf.download("TSLA", period="max", interval="1d", progress=False)
    if df is None or df.empty:
        return None, None

    # Flatten MultiIndex columns introduced in recent yfinance editions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df.columns = [str(col).strip().lower() for col in df.columns]

    if "date" in df.columns and "close" in df.columns:
        df = df[["date", "close"]].copy()
    elif "timestamp" in df.columns and "close" in df.columns:
        df = df[["timestamp", "close"]].copy()
    else:
        df = df.iloc[:, [0, 4]].copy()

    df.columns = ["timestamp", "price"]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna()
    df = df[df["price"] > 0].sort_values("timestamp")

    genesis = pd.to_datetime("2010-06-29")
    df["t"] = ((df["timestamp"] - genesis).dt.total_seconds() / (24 * 3600 * 365.25)) + 1e-6
    return df, genesis


try:
    with st.spinner("Loading TSLA history..."):
        df, genesis = load_data()
except Exception as exc:  # noqa: BLE001
    df, genesis = None, None
    st.error(f"Data load failed: {exc}")

if df is None or df.empty:
    st.error("Failed to map yfinance infrastructure. Check web node rules or try again.")
    st.stop()


# ---------------------------------------------------------------------------
# 2. Refined Model
# ---------------------------------------------------------------------------
def power_law_staircase(t, a, b, step_h, step_f, sharp, amp, decay):
    trend = a + b * np.log10(t)
    steps = np.floor(t / step_f)
    breakout = 1 / (1 + np.exp(-sharp * (t % step_f - step_f / 2)))
    phase = (t % step_f) / step_f
    wobble = amp * np.exp(-decay * (t % step_f)) * np.sin(2 * np.pi * phase)
    val = trend + (step_h * (steps + breakout)) + wobble
    return np.minimum(val, CEILING_LOG)


log_price = np.log10(df["price"].values)
try:
    popt, _ = curve_fit(power_law_staircase, df["t"].values, log_price, p0=P0, maxfev=15000)
    if not np.all(np.isfinite(popt)):
        raise RuntimeError("non-finite parameters")
except Exception as exc:  # noqa: BLE001
    st.warning(f"Curve fit did not converge ({exc}); using initial parameters.")
    popt = np.array(P0)

fitted_log = power_law_staircase(df["t"].values, *popt)
ss_res = np.sum((log_price - fitted_log) ** 2)
ss_tot = np.sum((log_price - log_price.mean()) ** 2)
r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def model_price_at(date_str):
    t = (pd.to_datetime(date_str) - genesis).days / 365.25
    return float(10 ** power_law_staircase(max(t, 1e-6), *popt))


# ---------------------------------------------------------------------------
# Headline numbers
# ---------------------------------------------------------------------------
last_date = df["timestamp"].iloc[-1]
last_price = float(df["price"].iloc[-1])
model_now = model_price_at(last_date)
model_2030 = model_price_at("2030-01-01")
model_2040 = model_price_at("2040-12-31")
gap_pct = (last_price / model_now - 1) * 100

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Last close", f"${last_price:,.2f}", f"{last_date:%b %d, %Y}", delta_color="off")
c2.metric("Model today", f"${model_now:,.2f}", f"{gap_pct:+.1f}% actual vs model")
c3.metric("Model Jan 2030", f"${model_2030:,.0f}")
c4.metric("Model Dec 2040", f"${model_2040:,.0f}", f"${CEILING_PRICE:,} ceiling", delta_color="off")
c5.metric("Fit R² (log10)", f"{r2:.4f}" if np.isfinite(r2) else "n/a", f"step {popt[3]:.2f}y", delta_color="off")

st.write("")

# ---------------------------------------------------------------------------
# 3. Visualization
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#d0d4db",
    "axes.labelcolor": C_TEXT,
    "xtick.color": C_MUTED,
    "ytick.color": C_MUTED,
    "text.color": C_TEXT,
})

fig, ax = plt.subplots(figsize=(22, 11))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

all_dates = pd.date_range("2012-01-01", "2040-12-31", freq="W")
all_t = np.maximum(((all_dates - genesis).total_seconds() / (24 * 3600)) / 365.25, 1e-6)
model_prices = 10 ** power_law_staircase(all_t, *popt)

# Grey Support Channel (30% discount floor)
ax.fill_between(all_dates, model_prices * 0.70, model_prices, color=C_CHANNEL, alpha=0.18, lw=0, zorder=2)

# 4. AUDITED IRREGULAR PHASE DATA
milestones = [
    ("Consolidation", "2010-06-29", "2013-01-01"),
    ("EXPANSION", "2013-01-02", "2013-12-31"),
    ("Consolidation", "2014-01-01", "2019-10-21"),
    ("EXPANSION", "2019-10-22", "2021-11-04"),
    ("Consolidation", "2021-11-05", "2027-03-15"),
    ("EXPANSION", "2027-03-16", "2028-06-30"),
    ("Consolidation", "2028-07-01", "2033-12-31"),
    ("EXPANSION", "2034-01-01", "2035-05-15"),
    ("Consolidation", "2035-05-16", "2039-12-31"),
    ("EXPANSION", "2040-01-01", "2040-12-31"),
]

table_rows = []
for phase, start, end in milestones:
    s_dt, e_dt = pd.to_datetime(start), pd.to_datetime(end)
    is_exp = phase == "EXPANSION"
    ax.axvspan(s_dt, e_dt, color=C_EXPAND if is_exp else C_CONSOL, alpha=0.9, lw=0, zorder=0)

    t1, t2 = (s_dt - genesis).days / 365.25, (e_dt - genesis).days / 365.25
    p1 = float(10 ** power_law_staircase(max(t1, 1e-6), *popt))
    p2 = float(10 ** power_law_staircase(max(t2, 1e-6), *popt))
    cagr = ((p2 / p1) ** (1 / max(0.1, (t2 - t1))) - 1) * 100

    table_rows.append({
        "Phase": phase.title(),
        "Start Date": start,
        "End Date": end,
        "Start Price": f"${p1:,.2f}",
        "End Price": f"${p2:,.2f}",
        "CAGR": f"{cagr:.1f}%",
    })

    if is_exp:
        # Green Hollow Breakout Arrow
        ax.plot(s_dt, p1, marker="^", linestyle="none", markeredgecolor=C_KICKOFF,
                markerfacecolor="none", markersize=14, markeredgewidth=2, zorder=15)
        ax.vlines(s_dt, p1 * 0.22, p1 * 0.85, color=C_KICKOFF, lw=0.8, alpha=0.5, zorder=14)
        ax.text(s_dt, p1 * 0.2, f"{s_dt:%b %Y}   ${p1:,.0f}", ha="center", va="top",
                fontsize=10, color=C_KICKOFF, fontweight="bold", rotation=90, zorder=15)

# Ceiling
ax.axhline(CEILING_PRICE, color=C_CEILING, ls=":", lw=1.3, alpha=0.9, zorder=3)
ax.text(all_dates[-1], CEILING_PRICE * 1.08, f"$20T market cap  (${CEILING_PRICE:,})",
        ha="right", va="bottom", fontsize=10, color=C_CEILING)

# Price and model
ax.plot(df["timestamp"], df["price"], color=C_ACTUAL, lw=1.5, zorder=10)
ax.plot(all_dates, model_prices, color=C_MODEL, ls="--", lw=1.3, alpha=0.8, zorder=5)

# Last close marker
ax.plot(last_date, last_price, marker="o", markersize=7, color=C_ACTUAL,
        markeredgecolor="white", markeredgewidth=1.5, zorder=12)
ax.annotate(f"${last_price:,.0f}", xy=(last_date, last_price), xytext=(10, 0),
            textcoords="offset points", va="center", fontsize=10, color=C_ACTUAL, fontweight="bold")

# Format Controls
ax.set_yscale("log")
ax.set_ylim(0.5, 1e4)
ax.set_xlim(pd.Timestamp("2010-01-01"), pd.Timestamp("2041-06-30"))
ax.yaxis.set_major_locator(LogLocator(base=10.0))
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"${x:,.0f}" if x >= 1 else f"${x:.2f}"))
ax.yaxis.set_minor_formatter(NullFormatter())
ax.grid(True, which="major", axis="y", linestyle="-", lw=0.6, color="#dfe3ea", alpha=0.9, zorder=1)
ax.grid(True, which="minor", axis="y", linestyle="-", lw=0.4, color="#eceff4", alpha=0.9, zorder=1)
ax.grid(False, axis="x")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.tick_params(axis="both", which="both", length=0, labelsize=10)
ax.set_ylabel("Price per share (log scale)", fontsize=11, color=C_MUTED)

ax.text(0.005, 0.985,
        f"Fit: a={popt[0]:.3f}  b={popt[1]:.3f}  step_h={popt[2]:.3f}  step_f={popt[3]:.2f}y  "
        f"sharp={popt[4]:.1f}  amp={popt[5]:.3f}  decay={popt[6]:.3f}   |   R² = {r2:.4f}",
        transform=ax.transAxes, fontsize=9, color=C_MUTED, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#e6e9ef"))

legend_elements = [
    Line2D([0], [0], color=C_ACTUAL, lw=2, label="Actual price (split-adjusted)"),
    Line2D([0], [0], color=C_MODEL, ls="--", lw=1.5, label="Model forecast ($20T ceiling)"),
    Line2D([0], [0], marker="^", color="w", markeredgecolor=C_KICKOFF, markeredgewidth=2,
           label="Expansion kickoff", markerfacecolor="none", markersize=12),
    Patch(facecolor=C_CHANNEL, alpha=0.3, label="Support channel (30% below model)"),
    Patch(facecolor=C_EXPAND, label="Expansion epoch"),
    Patch(facecolor=C_CONSOL, label="Consolidation epoch"),
]
ax.legend(handles=legend_elements, loc="lower right", ncol=3, fontsize=10,
          frameon=True, framealpha=0.95, edgecolor="#e6e9ef")

fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# ---------------------------------------------------------------------------
# Phase table
# ---------------------------------------------------------------------------
st.subheader("TSLA Punctuated Phase Analysis (Audited)")
phase_df = pd.DataFrame(table_rows)


def _row_style(row):
    bg = C_EXPAND if row["Phase"] == "Expansion" else C_CONSOL
    return [f"background-color: {bg}; color: #111111"] * len(row)


st.dataframe(phase_df.style.apply(_row_style, axis=1), use_container_width=True, hide_index=True)
