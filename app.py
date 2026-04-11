import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
import yfinance as yf
from scipy.optimize import curve_fit

# 1. Setup & Market Cap Ceiling Logic
# $20T Cap / ~3.75B Shares = $5,333 per share. log10(5333) ≈ 3.727
CEILING_LOG = 3.727

st.set_page_config(page_title="TSLA $20T Power-Law Staircase", layout="wide")
st.markdown("<style>#MainMenu, footer, header {visibility: hidden;}</style>", unsafe_allow_html=True)

st.title("Tesla Underdamped Power-Law Staircase")
st.caption("Market Cap Limited ($20T) • Phase-Based Shading • Support Channels & Kickoff Arrows")

@st.cache_data(ttl=86400)
def load_data():
    try:
        df = yf.download('TSLA', period='max', interval='1d', progress=False, auto_adjust=True)
        if df.empty: return None, None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df.columns = [str(c).lower() for c in df.columns]
        
        if 'close' in df.columns:
            df = df[['date', 'close']].copy(); df.columns = ['timestamp', 'price']
        else:
            df = df.iloc[:, [0, 1]]; df.columns = ['timestamp', 'price']

        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        genesis = pd.to_datetime('2010-06-29')
        df['t'] = ((df['timestamp'] - genesis).dt.total_seconds() / (24 * 3600 * 365.25)) + 1e-6
        return df, genesis
    except Exception as e:
        st.error(f"Data Loading Error: {e}"); return None, None

df, genesis = load_data()
if df is None: st.stop()

# 2. The Constrained Model
def power_law_staircase(t, a, b, step_h, step_f, sharp, amp, decay):
    trend = a + b * np.log10(t)
    steps = np.floor(t / step_f)
    breakout = 1 / (1 + np.exp(-sharp * (t % step_f - step_f/2)))
    phase = (t % step_f) / step_f
    wobble = amp * np.exp(-decay * (t % step_f)) * np.sin(2 * np.pi * phase)
    val = trend + (step_h * (steps + breakout)) + wobble
    return np.minimum(val, CEILING_LOG)

p0 = [-0.1, 1.8, 0.45, 6.8, 15.0, 0.2, 0.4]
bounds = ([-5, 0.1, 0.1, 4, 5, -2, 0], [5, 5, 2, 10, 50, 2, 5])
popt, _ = curve_fit(power_law_staircase, df['t'], np.log10(df['price']), p0=p0, bounds=bounds, maxfev=15000)

# 3. Visualization
fig, ax = plt.subplots(figsize=(22, 12))

# Milestones & Shading
milestones = [2012, 2016, 2020, 2024, 2028, 2032, 2036, 2040]
phase_colors = ['#f0f4ff', '#e8f5e9', '#fff3e0'] 
table_data = []

# Generate Main Projection
all_dates = pd.date_range('2012-01-01', '2040-12-31', freq='W')
all_t = np.maximum(((all_dates - genesis).total_seconds() / (24*3600)) / 365.25, 1e-6)
model_prices = 10**power_law_staircase(all_t, *popt)

# Consolidation Channel Shading (Price Lower Bound)
support_channel = model_prices * 0.70 # 30% discount as support
ax.fill_between(all_dates, support_channel, model_prices, color='gray', alpha=0.15, label='Support Channel')

for i in range(len(milestones)-1):
    s_yr, e_yr = milestones[i], milestones[i+1]
    s_dt, e_dt = pd.to_datetime(f'{s_yr}-01-01'), pd.to_datetime(f'{e_yr}-01-01')
    ax.axvspan(s_dt, e_dt, color=phase_colors[i % 3], alpha=0.4, zorder=0)
    
    t1, t2 = (s_dt - genesis).days/365.25, (e_dt - genesis).days/365.25
    p1, p2 = 10**power_law_staircase(t1, *popt), 10**power_law_staircase(t2, *popt)
    cagr = ((p2/p1)**(1/(t2-t1)) - 1) * 100
    ax.text(s_dt + pd.Timedelta(days=730), 2e5, f"{cagr:.1f}%", color='purple', fontsize=12, fontweight='bold', ha='center')
    table_data.append({"Period": f"{s_yr}-{e_yr}", "Start": f"${p1:,.0f}", "End": f"${p2:,.0f}", "CAGR": f"{cagr:.1f}%"})

# GREEN HOLLOW KICKOFF ARROWS
step_freq = popt[3]
kickoff_ts = np.arange(step_freq/2, 40, step_freq)
for kt in kickoff_ts:
    k_dt = genesis + pd.to_timedelta(kt * 365.25, unit='D')
    if pd.to_datetime('2012-01-01') < k_dt < pd.to_datetime('2040-12-31'):
        model_p = 10**power_law_staircase(kt, *popt)
        # Transparent (hollow) arrow style using facecolor=none
        ax.annotate('', xy=(k_dt, model_p * 1.5), xytext=(k_dt, model_p * 0.5),
                    arrowprops=dict(edgecolor='#00aa00', facecolor='none', width=8, headwidth=18, alpha=0.9),
                    zorder=15)
        ax.text(k_dt, model_p * 0.3, f"{k_dt.date()}\n${model_p:,.0f}", 
                ha='center', va='top', fontsize=10, color='#00aa00', fontweight='bold', rotation=90)

# Main Price Lines
ax.plot(df['timestamp'], df['price'], color='crimson', lw=2, label='Actual Price', zorder=10)
ax.plot(all_dates, model_prices, color='black', ls='--', lw=1.5, alpha=0.8)

# Formatting
ax.set_yscale('log')
ax.set_ylim(0.5, 4e5)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.set_xlim(pd.to_datetime('2012-01-01'), pd.to_datetime('2040-12-31'))
ax.grid(True, which='both', linestyle='--', alpha=0.1)

# Custom Legend
legend_elements = [
    Line2D([0], [0], color='crimson', lw=2, label='Actual Price'),
    Line2D([0], [0], color='black', ls='--', label='Model Forecast ($20T Cap)'),
    Line2D([0], [0], marker='^', color='w', markeredgecolor='#00aa00', label='Exp. Kickoff', markerfacecolor='none', markersize=12),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', alpha=0.3, label='Support Channel', markersize=10)
]
ax.legend(handles=legend_elements, loc='lower right', ncol=2)

st.pyplot(fig)
st.subheader("Cycle Performance Metrics")
st.table(pd.DataFrame(table_data))
