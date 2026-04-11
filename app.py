import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
import yfinance as yf
from scipy.optimize import curve_fit

# Page Config
st.set_page_config(page_title="TSLA Power-Law Cycles", layout="wide")
st.markdown("<style>#MainMenu, footer, header {visibility: hidden;}</style>", unsafe_allow_html=True)

st.title("Tesla Underdamped Power-Law Staircase")
st.caption("Market Cap Constrained ($20T) • CAGR Phase Mapping • Exponential Kickoff Dates")

# ====================== DATA LOADING ======================
@st.cache_data(ttl=86400)
def load_data():
    df = yf.download('TSLA', period='max', interval='1d', progress=False, auto_adjust=True)
    if df.empty: return None, None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.columns = [str(c).lower() for c in df.columns]
    
    if 'close' in df.columns:
        df = df[['date', 'close']].copy()
        df.columns = ['timestamp', 'price']
    else:
        df = df.iloc[:, [0, 1]]
        df.columns = ['timestamp', 'price']

    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    genesis = pd.to_datetime('2010-06-29')
    df['t'] = ((df['timestamp'] - genesis).dt.total_seconds() / (24 * 3600 * 365.25)) + 1e-6
    return df, genesis

df, genesis = load_data()

# ====================== CONSTRAINED POWER-LAW MODEL ======================
# $20T Cap / 3.75B Shares ≈ $5,333/share. log10(5333) ≈ 3.72
CEILING_LOG = np.log10(5333)

def power_law_staircase(t, a, b, step_h, step_f, sharp, amp, decay):
    # Log-Log Trend (Power Law)
    trend = a + b * np.log10(t)
    # Stepped Logic
    steps = np.floor(t / step_f)
    breakout = 1 / (1 + np.exp(-sharp * (t % step_f - step_f/2)))
    # Underdamped Overshoot
    phase = (t % step_f) / step_f
    wobble = amp * np.exp(-decay * (t % step_f)) * np.sin(2 * np.pi * phase)
    
    val = trend + (step_h * (steps + breakout)) + wobble
    return np.minimum(val, CEILING_LOG)

# Optimization [a, b, step_h, step_f, sharp, amp, decay]
p0 = [-0.2, 1.8, 0.45, 6.8, 15.0, 0.2, 0.4]
bounds = ([-5, 0.1, 0.1, 4, 5, -2, 0], [5, 5, 2, 10, 50, 2, 5])
popt, _ = curve_fit(power_law_staircase, df['t'], np.log10(df['price']), p0=p0, bounds=bounds)

# ====================== PROJECTION ======================
all_dates = pd.date_range('2012-01-01', '2040-12-31', freq='W')
all_t = ((all_dates - genesis).total_seconds() / (24*3600)) / 365.25
all_t = np.maximum(all_t, 1e-6)

pred_log = power_law_staircase(all_t, *popt)
all_pred_price = 10**pred_log

# ====================== VISUALIZATION ======================
fig, ax = plt.subplots(figsize=(22, 12))

# 1. Background Shading by CAGR Period
milestones = [2012, 2016, 2020, 2024, 2028, 2032, 2036, 2040]
phase_colors = ['#f0f4ff', '#e8f5e9', '#fff3e0'] # Alternating subtle hues
table_rows = []

for i in range(len(milestones)-1):
    s_yr, e_yr = milestones[i], milestones[i+1]
    s_date, e_date = pd.to_datetime(f'{s_yr}-01-01'), pd.to_datetime(f'{e_yr}-01-01')
    
    # Shading
    ax.axvspan(s_date, e_date, color=phase_colors[i % len(phase_colors)], alpha=0.4, zorder=0)
    
    # CAGR Calc
    t1, t2 = (s_date - genesis).days/365.25, (e_date - genesis).days/365.25
    p1, p2 = 10**power_law_staircase(t1, *popt), 10**power_law_staircase(t2, *popt)
    cagr = ((p2/p1)**(1/(t2-t1)) - 1) * 100
    
    # Purple Label Placement
    mid_date = s_date + (e_date - s_date)/2
    ax.text(mid_date, 1e5, f"{cagr:.1f}% CAGR", color='purple', fontsize=12, fontweight='bold', ha='center')
    
    table_rows.append({"Period": f"{s_yr}-{e_yr}", "Start Price": f"${p1:,.0f}", "End Price": f"${p2:,.0f}", "CAGR": f"{cagr:.1f}%"})

# 2. Kickoff Dates (Breakout Points)
# Identifying when the 'breakout' term is rising fastest
kickoff_years = np.arange(2013, 2041, popt[3])
for yr_offset in kickoff_years:
    kickoff_t = yr_offset
    kickoff_date = genesis + pd.Timedelta(days=kickoff_t * 365.25)
    if pd.to_datetime('2012-01-01') < kickoff_date < pd.to_datetime('2040-12-31'):
        ax.axvline(kickoff_date, color='orange', linestyle=':', alpha=0.6, lw=2)
        ax.text(kickoff_date, 0.7, 'KICKOFF', color='orange', rotation=90, fontsize=9, va='bottom', transform=ax.get_xaxis_transform())

# 3. Main Lines
ax.plot(df['timestamp'], df['price'], color='crimson', lw=2, label='Actual TSLA Price', zorder=10)
ax.plot(all_dates, all_pred_price, color='black', ls='--', lw=1.5, alpha=0.8, label='Power-Law Staircase Model')

# Formatting
ax.set_yscale('log')
ax.set_ylim(0.5, 2e5)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.set_xlim(pd.to_datetime('2012-01-01'), pd.to_datetime('2040-12-31'))
ax.grid(True, which='both', linestyle='--', alpha=0.1)

# Custom Legend
legend_elements = [
    Line2D([0], [0], color='crimson', lw=2, label='Actual Price'),
    Line2D([0], [0], color='black', lw=1.5, ls='--', label='Model Forecast'),
    Line2D([0], [0], color='orange', lw=2, ls=':', label='Exponential Kickoff Date'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#e8f5e9', label='CAGR Accumulation Phase', markersize=10)
]
ax.legend(handles=legend_elements, loc='lower right', ncol=2)

st.pyplot(fig)

# ====================== PERFORMANCE TABLE ======================
st.subheader("Cycle Performance & Projections")
st.table(pd.DataFrame(table_rows))
