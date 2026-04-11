import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
import yfinance as yf
from scipy.optimize import curve_fit

# 1. Market Cap Ceiling Constants
# $20T Cap / ~3.75B Shares = $5,333 per share. log10(5333) ≈ 3.727
CEILING_LOG = 3.727

st.set_page_config(page_title="TSLA Punctuated Equilibrium", layout="wide")
st.markdown("<style>#MainMenu, footer, header {visibility: hidden;}</style>", unsafe_allow_html=True)

st.title("Tesla Underdamped Power-Law Staircase")
st.caption("Irregular Growth Cycles: Long Consolidation vs. Violent Breakouts ($20T Cap)")

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

# 2. The Model Function (Underdamped Staircase)
def power_law_staircase(t, a, b, step_h, step_f, sharp, amp, decay):
    trend = a + b * np.log10(t)
    steps = np.floor(t / step_f)
    breakout = 1 / (1 + np.exp(-sharp * (t % step_f - step_f/2)))
    phase = (t % step_f) / step_f
    wobble = amp * np.exp(-decay * (t % step_f)) * np.sin(2 * np.pi * phase)
    val = trend + (step_h * (steps + breakout)) + wobble
    return np.minimum(val, CEILING_LOG)

# Optimization [a, b, step_h, step_f, sharp, amp, decay]
p0 = [-0.1, 1.8, 0.45, 6.8, 30.0, 0.2, 0.4]
bounds = ([-5, 0.1, 0.1, 5, 20, -2, 0], [5, 5, 2, 10, 60, 2, 5])
popt, _ = curve_fit(power_law_staircase, df['t'], np.log10(df['price']), p0=p0, bounds=bounds, maxfev=15000)

# 3. Visualization Setup
fig, ax = plt.subplots(figsize=(22, 12))
all_dates = pd.date_range('2012-01-01', '2040-12-31', freq='W')
all_t = np.maximum(((all_dates - genesis).total_seconds() / (24*3600)) / 365.25, 1e-6)
model_prices = 10**power_law_staircase(all_t, *popt)

# Grey Support Channel (30% discount from model trend)
ax.fill_between(all_dates, model_prices * 0.70, model_prices, color='gray', alpha=0.15, label='Support Channel')

# 4. Irregular Epoch Logic (Exact 80/20 Split)
step_freq = popt[3]
growth_window = 0.20 
table_rows = []
current_t = 1.5 # Start year ~2012

while current_t < 31:
    # Phase 1: Consolidation (80% of time)
    c_start, c_end = current_t, current_t + (step_freq * (1 - growth_window))
    # Phase 2: Expansion (20% of time)
    g_start, g_end = c_end, current_t + step_freq
    
    for start, end, label, color in [(c_start, c_end, 'Consolidation', '#f0f4ff'), (g_start, g_end, 'Expansion', '#e8f5e9')]:
        s_dt = genesis + pd.to_timedelta(start * 365.25, unit='D')
        e_dt = genesis + pd.to_timedelta(end * 365.25, unit='D')
        
        if s_dt < pd.to_datetime('2040-12-31'):
            e_dt_lim = min(e_dt, pd.to_datetime('2040-12-31'))
            ax.axvspan(s_dt, e_dt_lim, color=color, alpha=0.4, zorder=0)
            
            p1, p2 = 10**power_law_staircase(start, *popt), 10**power_law_staircase(end, *popt)
            cagr = ((p2/p1)**(1/(end-start)) - 1) * 100
            
            if s_dt > pd.to_datetime('2012-01-01'):
                table_rows.append({
                    "Phase": label,
                    "Start Date": s_dt.date(),
                    "End Date": e_dt_lim.date(),
                    "Start Price": f"${p1:,.0f}",
                    "End Price": f"${p2:,.0f}",
                    "CAGR": f"{cagr:.1f}%"
                })
                # Purple CAGR labels on chart
                mid_date = s_dt + (e_dt_lim - s_dt)/2
                ax.text(mid_date, 1.8e5, f"{cagr:.1f}%", color='purple', fontsize=11, fontweight='bold', ha='center')

                # Kickoff Arrow on Expansion start
                if label == 'Expansion':
                    ax.plot(s_dt, p1, marker='^', color='w', markeredgecolor='#00aa00', markerfacecolor='none', markersize=14, zorder=15)
                    ax.text(s_dt, p1 * 0.15, f"{s_dt.date()}\nKickoff: ${p1:,.0f}", ha='center', va='top', fontsize=10, color='#00aa00', fontweight='bold', rotation=90)
    
    current_t += step_freq

# 5. Final Plot Polish
ax.plot(df['timestamp'], df['price'], color='crimson', lw=1.8, label='Actual Price', zorder=10)
ax.plot(all_dates, model_prices, color='black', ls='--', lw=1.2, alpha=0.8)

ax.set_yscale('log')
ax.set_ylim(0.5, 5e5)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.set_xlim(pd.to_datetime('2012-01-01'), pd.to_datetime('2040-12-31'))
ax.grid(True, which='both', linestyle='--', alpha=0.1)

legend_elements = [
    Line2D([0], [0], color='crimson', lw=2, label='Actual Price'),
    Line2D([0], [0], color='black', ls='--', label='Model Forecast'),
    Line2D([0], [0], marker='^', color='w', markeredgecolor='#00aa00', label='Exp. Kickoff', markerfacecolor='none', markersize=12),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', alpha=0.3, label='Grey Support Channel')
]
ax.legend(handles=legend_elements, loc='lower right', ncol=2)

st.pyplot(fig)

# 6. Detailed Table Output
st.subheader("TSLA Punctuated Phase Analysis (Exact Boundaries)")
st.table(pd.DataFrame(table_rows))
