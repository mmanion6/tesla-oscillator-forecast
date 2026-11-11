"""
Tesla (TSLA) Underdamped Power-Law Staircase & Macro Oscillator Model
=====================================================================

Streamlit application that fits a market-cap-constrained, step-wise power-law
model to historical TSLA prices and projects it through 2040.

Model (in log10 price space):

    log10(P(t)) = min( Trend(t) + Staircase(t) + Wobble(t), log10(CEILING) )

    Trend(t)     = a + b * log10(t)
    Staircase(t) = step_h * ( floor(t / step_f) + sigmoid(sharp * (t mod step_f - step_f / 2)) )
    Wobble(t)    = amp * exp(-decay * (t mod step_f)) * sin(2 * pi * (t mod step_f) / step_f)

where t is fractional years since the TSLA IPO (2010-06-29).

Run with:  streamlit run app.py
"""

from __future__ import annotations

import io
import traceback

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IPO_DATE = pd.Timestamp("2010-06-29")
EPS = 1e-6  # offset so log10(t) is defined on the IPO day

SHARES_OUTSTANDING = 3.75e9
MCAP_CEILING_USD = 20e12
PRICE_CEILING = MCAP_CEILING_USD / SHARES_OUTSTANDING  # ~ $5,333 / share
LOG_CEILING = np.log10(PRICE_CEILING)

PARAM_NAMES = ["a", "b", "step_h", "step_f", "sharp", "amp", "decay"]
P0 = [-0.1, 1.8, 0.45, 7.0, 30.0, 0.2, 0.4]
BOUNDS = ([-5, 0.1, 0.1, 5, 20, -2, 0], [5, 5, 2, 10, 60, 2, 5])

PROJECTION_START = pd.Timestamp("2012-01-01")
PROJECTION_END = pd.Timestamp("2040-12-31")

EXPANSION_FRACTION = 0.20  # share of each step period spent in the breakout

COLOR_CONSOLIDATION = "#dfe7f3"
COLOR_EXPANSION = "#fde9c9"
COLOR_KICKOFF = "#00aa00"


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def dates_to_t(index) -> np.ndarray:
    """Convert datetimes to fractional years since IPO, with epsilon offset."""
    idx = pd.DatetimeIndex(index)
    days = (idx - IPO_DATE).days.values.astype(float)
    return days / 365.25 + EPS


def t_to_dates(t) -> pd.DatetimeIndex:
    """Inverse of dates_to_t."""
    t = np.asarray(t, dtype=float)
    return IPO_DATE + pd.to_timedelta((t - EPS) * 365.25, unit="D")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def model_log10(t, a, b, step_h, step_f, sharp, amp, decay):
    """log10 price under the ceiling-constrained power-law staircase model."""
    t = np.maximum(np.asarray(t, dtype=float), EPS)
    phase = np.mod(t, step_f)

    trend = a + b * np.log10(t)

    z = np.clip(sharp * (phase - step_f / 2.0), -500.0, 500.0)
    sigmoid = 1.0 / (1.0 + np.exp(-z))
    staircase = step_h * (np.floor(t / step_f) + sigmoid)

    wobble = amp * np.exp(-decay * phase) * np.sin(2.0 * np.pi * phase / step_f)

    return np.minimum(trend + staircase + wobble, LOG_CEILING)


def model_price(t, params) -> np.ndarray:
    return 10.0 ** model_log10(t, *params)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def load_tsla() -> pd.DataFrame:
    """Download and clean TSLA daily closes. Returns a DataFrame indexed by Date
    with a single 'Close' column."""
    import yfinance as yf

    raw = yf.download("TSLA", period="max", interval="1d", progress=False, auto_adjust=True)
    if raw is None or len(raw) == 0:
        raise ValueError("yfinance returned no data for TSLA.")

    # Flatten multi-index columns (newer yfinance returns (field, ticker) pairs)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw.reset_index()

    # Case-insensitive column mapping
    col_map: dict = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in ("date", "timestamp", "datetime", "index") and "Date" not in col_map.values():
            col_map[col] = "Date"
        elif key == "close" and "Close" not in col_map.values():
            col_map[col] = "Close"
    # Fall back to adjusted close if a plain close column is missing
    if "Close" not in col_map.values():
        for col in df.columns:
            if str(col).strip().lower() in ("adj close", "adj_close", "adjclose"):
                col_map[col] = "Close"
                break
    df = df.rename(columns=col_map)

    missing = [c for c in ("Date", "Close") if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected column(s) after cleaning: {missing}. Got {list(df.columns)}")

    df = df[["Date", "Close"]].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if getattr(df["Date"].dt, "tz", None) is not None:
        df["Date"] = df["Date"].dt.tz_localize(None)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    df = df.dropna(subset=["Date", "Close"])
    df = df[np.isfinite(df["Close"]) & (df["Close"] > 0)]
    df = df.sort_values("Date").drop_duplicates(subset="Date").set_index("Date")

    if len(df) < 100:
        raise ValueError(f"Only {len(df)} usable rows after cleaning; refusing to fit.")
    return df


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_model(t: np.ndarray, log_p: np.ndarray):
    """Fit model parameters with bounded least squares.
    Returns (params, converged, message)."""
    mask = np.isfinite(t) & np.isfinite(log_p)
    t, log_p = t[mask], log_p[mask]
    if len(t) < 50:
        return np.array(P0, dtype=float), False, "Too few finite samples to fit; using default parameters."
    try:
        popt, _ = curve_fit(
            model_log10,
            t,
            log_p,
            p0=P0,
            bounds=BOUNDS,
            method="trf",
            maxfev=40000,
        )
        if not np.all(np.isfinite(popt)):
            return np.array(P0, dtype=float), False, "Optimizer returned non-finite parameters; using defaults."
        return popt, True, "Converged."
    except (RuntimeError, ValueError, TypeError) as exc:
        return np.array(P0, dtype=float), False, f"curve_fit failed ({exc}); using default parameters."


def r_squared(y: np.ndarray, y_hat: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(y_hat)
    y, y_hat = y[mask], y_hat[mask]
    if len(y) == 0:
        return float("nan")
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


# ---------------------------------------------------------------------------
# Phase segmentation
# ---------------------------------------------------------------------------

def build_phases(params, t_start: float, t_end: float) -> pd.DataFrame:
    """Segment [t_start, t_end] into alternating Expansion / Consolidation epochs.

    Each step period (step_f) contains a breakout centred on the sigmoid midpoint
    (t mod step_f = step_f / 2). The expansion epoch is a window of width
    EXPANSION_FRACTION * step_f centred there; the remainder of the period is
    consolidation. A macro cycle therefore runs kickoff -> breakout -> long
    consolidation -> next kickoff.
    """
    step_f = float(params[3])
    half_w = EXPANSION_FRACTION * step_f / 2.0

    k_lo = int(np.floor(t_start / step_f)) - 1
    k_hi = int(np.ceil(t_end / step_f)) + 1

    rows = []
    for k in range(k_lo, k_hi + 1):
        kickoff = k * step_f + step_f / 2.0 - half_w
        exp_end = kickoff + 2.0 * half_w
        next_kickoff = kickoff + step_f
        rows.append(("Expansion", kickoff, exp_end))
        rows.append(("Consolidation", exp_end, next_kickoff))

    out = []
    for phase, s, e in rows:
        s_c, e_c = max(s, t_start), min(e, t_end)
        if e_c - s_c <= 1e-6:
            continue
        p_s = float(model_price(s_c, params))
        p_e = float(model_price(e_c, params))
        dur = e_c - s_c
        cagr = (p_e / p_s) ** (1.0 / dur) - 1.0 if dur > 0 and p_s > 0 else float("nan")
        out.append(
            {
                "Phase": phase,
                "t_start": s_c,
                "t_end": e_c,
                "Start Date": t_to_dates([s_c])[0].normalize(),
                "End Date": t_to_dates([e_c])[0].normalize(),
                "Start Price": p_s,
                "End Price": p_e,
                "Years": dur,
                "CAGR": cagr,
                "is_kickoff": phase == "Expansion" and abs(s_c - s) < 1e-6,
            }
        )
    return pd.DataFrame(out).sort_values("t_start").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def currency_formatter(y, _pos=None) -> str:
    if y >= 1:
        return f"${y:,.0f}"
    return f"${y:,.2f}"


def build_figure(hist: pd.DataFrame, params, phases: pd.DataFrame, discount: float,
                 show_labels: bool, fit_start: pd.Timestamp):
    proj_t = np.linspace(dates_to_t([PROJECTION_START])[0], dates_to_t([PROJECTION_END])[0], 3000)
    proj_dates = t_to_dates(proj_t)
    proj_price = model_price(proj_t, params)

    fig, ax = plt.subplots(figsize=(16, 8.5))

    # Phase bands
    for _, row in phases.iterrows():
        color = COLOR_EXPANSION if row["Phase"] == "Expansion" else COLOR_CONSOLIDATION
        ax.axvspan(row["Start Date"], row["End Date"], color=color, alpha=0.6, lw=0, zorder=0)

    # Support channel
    ax.fill_between(
        proj_dates,
        proj_price * (1.0 - discount),
        proj_price,
        color="grey",
        alpha=0.15,
        lw=0,
        zorder=2,
        label=f"Support channel ({int(discount * 100)}% below model)",
    )

    # Model projection
    ax.plot(proj_dates, proj_price, "--", color="black", alpha=0.8, lw=1.4, zorder=5, label="Power-law staircase model")

    # Ceiling
    ax.axhline(PRICE_CEILING, color="#6a0dad", ls=":", lw=1.2, zorder=4,
               label=f"$20T market-cap ceiling (${PRICE_CEILING:,.0f})")

    # Actual price
    ax.plot(hist.index, hist["Close"].values, color="crimson", lw=1.5, zorder=10, label="TSLA actual (split-adjusted)")

    # Fit window marker
    ax.axvline(fit_start, color="#555555", ls="-.", lw=0.8, alpha=0.6, zorder=3)
    ax.text(fit_start, PRICE_CEILING * 1.15, " fit window start", fontsize=8, color="#555555", va="bottom")

    # Kickoff arrows
    kickoffs = phases[phases["is_kickoff"]]
    first = True
    for _, row in kickoffs.iterrows():
        ax.plot(
            row["Start Date"], row["Start Price"],
            marker="^", markersize=13, markerfacecolor="none", markeredgecolor=COLOR_KICKOFF,
            markeredgewidth=2.0, linestyle="none", zorder=11,
            label="Expansion kickoff" if first else None,
        )
        first = False
        if show_labels:
            ax.annotate(
                f"{row['Start Date']:%b %Y}  ${row['Start Price']:,.0f}",
                xy=(row["Start Date"], row["Start Price"]),
                xytext=(0, 14), textcoords="offset points",
                rotation=90, ha="center", va="bottom",
                fontsize=8, color=COLOR_KICKOFF, zorder=12,
            )

    # Axes
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
    ax.yaxis.set_major_formatter(FuncFormatter(currency_formatter))
    ax.yaxis.set_minor_formatter(NullFormatter())

    y_lo = max(0.5, float(hist["Close"].min()) * 0.5)
    y_hi = PRICE_CEILING * 3.0
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(PROJECTION_START - pd.Timedelta(days=180), PROJECTION_END)

    ax.grid(True, which="major", axis="y", ls="-", lw=0.5, alpha=0.4)
    ax.grid(True, which="minor", axis="y", ls="-", lw=0.3, alpha=0.2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price per share (log scale)")
    ax.set_title("TSLA  |  Underdamped Power-Law Staircase & Macro Oscillator  |  projection to 2040", fontsize=14)

    step_f = params[3]
    ax.text(
        0.01, 0.02,
        f"Shaded bands: blue = consolidation (~{int((1 - EXPANSION_FRACTION) * 100)}% of {step_f:.2f}y step), "
        f"orange = expansion (~{int(EXPANSION_FRACTION * 100)}%)",
        transform=ax.transAxes, fontsize=8, color="#444444", va="bottom",
    )

    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="TSLA Power-Law Staircase Model",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        "<style>#MainMenu, footer, header {visibility: hidden;}</style>",
        unsafe_allow_html=True,
    )

    st.title("Tesla (TSLA) Underdamped Power-Law Staircase & Macro Oscillator")
    st.caption(
        "Log-log power-law trend with sigmoid step breakouts, an exponentially decaying "
        "oscillator, and a hard $20T market-cap ceiling. Historical data via yfinance. "
        "This is a curve-fitting exercise, not investment advice."
    )

    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        fit_start_year = st.slider(
            "Fit window start year", min_value=2010, max_value=2020, value=2011,
            help="Earliest data used in the fit. Days immediately after IPO have t close to zero and dominate log10(t).",
        )
        discount = st.slider("Support channel discount", 0.10, 0.50, 0.30, 0.05, format="%.2f")
        show_labels = st.checkbox("Annotate kickoff arrows", value=True)
        st.divider()
        st.markdown(
            f"**Ceiling:** $20T / {SHARES_OUTSTANDING / 1e9:.2f}B shares = **${PRICE_CEILING:,.0f}**"
        )
        st.markdown("**Initial guess p0:** `" + ", ".join(f"{v}" for v in P0) + "`")
        if st.button("Clear data cache"):
            load_tsla.clear()
            st.rerun()

    # Data
    try:
        with st.spinner("Downloading TSLA history..."):
            hist = load_tsla()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load TSLA data: {exc}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())
        st.stop()

    fit_start = pd.Timestamp(year=fit_start_year, month=1, day=1)
    fit_df = hist[hist.index >= fit_start]
    if len(fit_df) < 100:
        st.warning("Fit window too short; falling back to the full history.")
        fit_df = hist
        fit_start = hist.index[0]

    # Fit
    try:
        t_fit = dates_to_t(fit_df.index)
        log_p_fit = np.log10(fit_df["Close"].values.astype(float))
        with st.spinner("Fitting model..."):
            params, converged, msg = fit_model(t_fit, log_p_fit)
        r2 = r_squared(log_p_fit, model_log10(t_fit, *params))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Model fitting failed unexpectedly: {exc}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())
        st.stop()

    if not converged:
        st.warning(msg)

    # Phases
    try:
        phases = build_phases(
            params,
            dates_to_t([PROJECTION_START])[0],
            dates_to_t([PROJECTION_END])[0],
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Phase segmentation failed: {exc}")
        st.stop()

    # Headline metrics
    last_date = hist.index[-1]
    last_price = float(hist["Close"].iloc[-1])
    model_now = float(model_price(dates_to_t([last_date])[0], params))
    model_2030 = float(model_price(dates_to_t([pd.Timestamp("2030-01-01")])[0], params))
    model_2040 = float(model_price(dates_to_t([PROJECTION_END])[0], params))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Last close", f"${last_price:,.2f}", f"{last_date:%Y-%m-%d}")
    c2.metric("Model today", f"${model_now:,.2f}", f"{(last_price / model_now - 1) * 100:+.1f}% actual vs model")
    c3.metric("Model Jan 2030", f"${model_2030:,.0f}")
    c4.metric("Model Dec 2040", f"${model_2040:,.0f}")
    c5.metric("Fit R² (log10)", f"{r2:.4f}" if np.isfinite(r2) else "n/a")

    # Chart
    try:
        fig = build_figure(hist, params, phases, discount, show_labels, fit_start)
        st.pyplot(fig, use_container_width=True)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
        st.download_button("Download chart (PNG)", buf.getvalue(), "tsla_staircase_model.png", "image/png")
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Chart rendering failed: {exc}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())

    # Parameters
    st.subheader("Fitted parameters")
    param_df = pd.DataFrame(
        {
            "Parameter": PARAM_NAMES,
            "Value": [float(v) for v in params],
            "Lower bound": BOUNDS[0],
            "Upper bound": BOUNDS[1],
            "Initial guess": P0,
        }
    )
    st.dataframe(
        param_df.style.format({"Value": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )
    step_f = float(params[3])
    st.caption(
        f"Step period {step_f:.2f} years, step height {params[2]:.3f} decades "
        f"({(10 ** params[2] - 1) * 100:.0f}% per step), wobble amplitude {params[5]:.3f} decades, "
        f"decay {params[6]:.3f} per year. Fit window: {fit_start:%Y-%m-%d} to {last_date:%Y-%m-%d} "
        f"({len(fit_df):,} daily closes)."
    )

    # Phase table
    st.subheader("Macro cycle data terminal (2012 to 2040)")
    table = phases[["Phase", "Start Date", "End Date", "Start Price", "End Price", "Years", "CAGR"]].copy()
    display = table.copy()
    display["Start Date"] = display["Start Date"].dt.strftime("%Y-%m-%d")
    display["End Date"] = display["End Date"].dt.strftime("%Y-%m-%d")
    display["Start Price"] = display["Start Price"].map(lambda v: f"${v:,.2f}")
    display["End Price"] = display["End Price"].map(lambda v: f"${v:,.2f}")
    display["Years"] = display["Years"].map(lambda v: f"{v:.2f}")
    display["CAGR"] = display["CAGR"].map(lambda v: f"{v * 100:+.1f}%" if np.isfinite(v) else "n/a")

    def _row_style(row):
        bg = COLOR_EXPANSION if row["Phase"] == "Expansion" else COLOR_CONSOLIDATION
        return [f"background-color: {bg}; color: #111111"] * len(row)

    st.dataframe(display.style.apply(_row_style, axis=1), use_container_width=True, hide_index=True)

    csv_bytes = table.to_csv(index=False).encode("utf-8")
    st.download_button("Download phase table (CSV)", csv_bytes, "tsla_macro_phases.csv", "text/csv")

    with st.expander("Model definition"):
        st.latex(r"\log_{10} P(t) = \min\big(\text{Trend}(t) + \text{Staircase}(t) + \text{Wobble}(t),\ \log_{10}\text{CEILING}\big)")
        st.latex(r"\text{Trend}(t) = a + b\,\log_{10}(t)")
        st.latex(
            r"\text{Staircase}(t) = h \left(\left\lfloor \tfrac{t}{f} \right\rfloor + "
            r"\frac{1}{1 + e^{-s\,(t \bmod f - f/2)}}\right)"
        )
        st.latex(r"\text{Wobble}(t) = A\, e^{-\lambda\,(t \bmod f)} \sin\!\left(2\pi \tfrac{t \bmod f}{f}\right)")
        st.markdown(
            "t is fractional years since the IPO on 2010-06-29 (plus 1e-6). The expansion epoch in each "
            "step period is the window of width 20% of the period centred on the sigmoid midpoint, which is "
            "where the breakout happens; the remaining 80% is consolidation."
        )


if __name__ == "__main__":
    main()
