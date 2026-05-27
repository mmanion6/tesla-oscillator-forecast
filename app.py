import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
import yfinance as yf
from scipy.optimize import curve_fit

# 1. Market Cap Ceiling Calculation
# $20 Trillion / 3.752B Shares Outstanding (2026 Context) = ~$5,330 per share limit.
CEILING_PRICE = 5330
CEILING_LOG = np.log10(CEILING_PRICE)

st.set_page_config(page_title="TSLA Punctuated Growth", layout="wide")
st.markdown("<style>#MainMenu, footer, header {visibility: hidden;}</style>", unsafe_allow_html=True)

st.title("Tesla Punctuated Power-Law Staircase")
st.caption("Recalibrated Phase Analysis: Long Consolidations vs. Violent Breakouts ($20T Cap)")

@st.cache_data(ttl=86400)
def load_data():
    # Force single ticker download and clean structural settings
    df = yf.download('TSLA', period='max', interval='1d', auto_adjust=True, progress=False)
    if df.empty: 
        return None, None
        
    # Standardize multi-index columns out of the equation if they exist
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    # Bring the date index out into a column safely
    df = df.reset_index()
    
    # Map columns using an explicitly robust case-insensitive check
    df.columns = [str(col).strip().lower() for col in df.columns]
    
    # Handle the target assignments safely regardless of column layout order
    if 'date' in df.columns and 'close' in df.columns:
        df = df[['date', 'close']].copy()
        df.columns = ['timestamp', 'price']
    elif 'timestamp' in df.columns and 'close' in df.columns:
        df = df[['timestamp', 'close']].copy()
        df.columns = ['timestamp', 'price']
    else:
        # Emergency fallback to structural index assumptions
        df = df.iloc[:, [0, 4]].copy()
        df.columns = ['timestamp', 'price']
        
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    genesis = pd.to_datetime('2010-06-29')
    df['t'] = ((df['timestamp'] - genesis).dt.total_seconds() / (24 * 3600 * 365.25)) + 1e-6
    return df, genesis

df, genesis = load_data()

if df is None or df.empty:
    st.error("Failed to extract target pricing matrix. Please verify network access to yfinance.")
    st.stop()

# 2. Mathematical Fit Architecture
def power_law_staircase(t, a, b, step_h, step_f, sharp, amp, decay):
    trend = a + b * np.log10(t)
    steps = np.floor(t / step_f)
    breakout = 1 / (1 + np.exp(-sharp * (t % step_f - step_f/2)))
    phase = (t % step_f) / step_f
    wobble = amp * np.exp(-decay * (t % step_f)) * np.sin(2 * np.pi * phase)
    val = trend + (step_h * (steps + breakout)) + wobble
    return np.minimum(val, CEILING_LOG)

# Curve Fit optimization profile over structural history
popt, _ = curve_fit(power_law_staircase, df['t'], np.log10(df['price']), 
                    p0=[-0.1, 1.8, 0.45, 7.0, 30.0, 0.2, 0.4], maxfev=15000)

# 3. Macro Projections Plot Generation
fig, ax = plt.subplots(figsize=(22, 12))
all_dates = pd.date_range('2012-01-01', '2040-12-31', freq='W')
all_t = np.maximum(((all_dates - genesis).total_seconds() / (24*3600)) / 365.25, 1e-6)
model_prices = 10**power_law_staircase(all_t, *popt)

# Grey Support Channel (30% discount margin)
ax.fill_between(all_dates, model_prices * 0.70, model_prices, color='gray', alpha=0.15)

# 4. Locked Phase Configurations (Irregular Punctuated Epochs)
milestones = [
    ("Consolidation", "2010-06-29", "2013-01-01", "#f0f4ff"),
    ("EXPANSION", "2013-01-02", "2013-12-31", "#e8f5e9"),
    ("Consolidation", "2014-01-01", "2019-10-21", "#f0f4ff"),
    ("EXPANSION", "2019-10-22", "2021-11-04", "#e8f5e9"),
    ("Consolidation", "2021-11-05", "2027-03-15", "#f0f4ff"),
    ("EXPANSION", "2027-03-16", "2028-06-30", "#e8f5e9"),
    ("Consolidation", "2028-07-01", "2033-12-31", "#f0f4ff"),
    ("EXPANSION", "2034-01-01", "2035-05-15", "#e8f5e9"),
    ("Consolidation", "2035-05-16", "2039-12-31", "#f0f4ff"),
    ("EXPANSION", "2040-01-01", "2040-12-31", "#e8f5e9"),
]

table_rows = []
for phase, start, end, color in milestones:
    s_dt, e_dt = pd.to_datetime(start), pd.to_datetime(end)
    ax.axvspan(s_dt, e_dt, color=color, alpha=0.4, zorder=0)
    
    t1, t2 = (s_dt - genesis).days/365.25, (e_dt - genesis).days/365.25
    p1, p2 = 10**power_law_staircase(t1, *popt), 10**power_law_staircase(t2, *popt)
    cagr = ((p2/p1)**(1/max(0.1, (t2-t1))) - 1) * 100
    
    table_rows.append({
        "Phase": phase, 
        "Start Date": start, 
        "End Date": end, 
        "Start Price": f"${p1:,.2f}", 
        "End Price": f"${p2:,.2f}", 
        "CAGR": f"{cagr:.1f}%"
    })
    
    if phase == "EXPANSION":
        # Green Hollow Breakout Arrow
        ax.plot(s_dt, p1, marker='^', color='w', markeredgecolor='#00aa00', markerfacecolor='none', markersize=14, zorder=15)
        ax.text(s_dt, p1*0.2, f"{start}\n${p1:,.0f}", ha='center', va='top', fontsize=10, color='#00aa00', fontweight='bold', rotation=90)

# Render actual and modeled trends
ax.plot(df['timestamp'], df['price'], color='crimson', lw=1.5, label='Actual Price', zorder=10)
ax.plot(all_dates, model_prices, color='black', ls='--', lw=1.2, alpha=0.8)

# Axis Customization & Format Rules
ax.set_yscale('log')
ax.set_ylim(0.5, 1e4)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.grid(True, which='both', linestyle='--', alpha=0.1)

legend_elements = [
    Line2D([0], [0], color='crimson', lw=2, label='Actual Price'),
    Line2D([0], [0], color='black', ls='--', label='Model Forecast ($20T Ceiling)'),
    Line2D([0], [0], marker='^', color='w', markeredgecolor='#00aa00', label='Exp. Kickoff', markerfacecolor='none', markersize=12),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', alpha=0.3, label='Grey Support Channel')
]
ax.legend(handles=legend_elements, loc='lower right', ncol=2)

st.pyplot(fig)

# 5. Data Display Terminal Block
st.subheader("TSLA Punctuated Phase Analysis (Audited Macro Windows)")
st.table(pd.DataFrame(table_rows))
