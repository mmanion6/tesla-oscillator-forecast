import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import yfinance as yf
from scipy.optimize import curve_fit

# Page Config
st.set_page_config(page_title="TSLA Underdamped Staircase", layout="wide")
st.markdown("<style>#MainMenu, footer, header {visibility: hidden;}</style>", unsafe_allow_html=True)

st.title("Tesla Underdamped Staircase Model")
st.caption("Exponential Growth with Cyclical Overshoots & Consolidation Plateaus (2012–2040)")

# ====================== DATA LOADING ======================
@st.cache_data(ttl=86400)
def load_data():
    df = yf.download('TSLA', period='max', interval='1d', progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df = df[['Close']].reset_index()
    df.columns = ['timestamp', 'price']
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    
    genesis = pd.to_datetime('2010-06-29')
    # Time in years with epsilon to prevent log(0)
    df['t'] = ((df['timestamp'] - genesis).dt.total_seconds() / (24 * 3600 * 365.25)) + 1e-6
    df['log_price'] = np.log10(df['price'])
    return df, genesis

df, genesis = load_data()

# ====================== THE UNDERDAMPED MODEL ======================
def underdamped_staircase(t, a, b, step_h, step_f, sharp, amp, decay):
    # 1. Base Trend
    trend = a + b * np.log(t)
    # 2. Stepping Logic
    steps = np.floor(t / step_f)
    breakout = 1 / (1 + np.exp(-sharp * (t % step_f - step_f/2)))
    # 3. Overshoot/Undershoot Logic (Decaying Sine triggered by step)
    phase = (t % step_f) / step_f
    wobble = amp * np.exp(-decay * (t % step_f)) * np.sin(2 * np.pi * phase)
    
    return trend + (step_h * (steps + breakout)) + wobble

# Initial Guesses [a, b, step_h, step_f, sharp, amp, decay]
p0 = [-1.0, 1.5, 0.6, 6.5, 12.0, 0.3, 0.4]
bounds = ([-10, 0, 0.1, 3, 5, -2, 0], [5, 5, 2, 10, 50, 2, 5])

popt, _ = curve_fit(underdamped_staircase, df['t'], df['log_price'], p0=p0, bounds=bounds, maxfev=20000)

# ====================== PROJECTION ======================
all_dates = pd.date_range('2012-01-01', '2040-12-31', freq='W')
all_t = ((all_dates - genesis).total_seconds() / (24*3600)) / 365.25
all_t = np.maximum(all_t, 1e-6)

pred_log = underdamped_staircase(all_t, *popt)
all_pred_price = 10**pred_log
all_df = pd.DataFrame({'timestamp': all_dates, 'price': all_pred_price, 't': all_t})
all_df['is_hist'] = all_df['timestamp'] <= df['timestamp'].max()

# ====================== PLOTTING ======================
fig, ax = plt.subplots(figsize=(22, 11))

# 1. Price Lines
ax.plot(df['timestamp'], df['price'], color='crimson', linewidth=2, label='Actual TSLA Price', zorder=10)
ax.plot(all_df['timestamp'], all_df['price'], color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Underdamped Staircase Model')

# 2. Historical vs Forecast Distinction
fut = all_df[~all_df['is_hist']]
ax.plot(fut['timestamp'], fut['price'], color='black', linewidth=3, alpha=0.9, zorder=5)

# 3. CAGR Milestones (4-Year Windows)
milestones = [2012, 2016, 2020, 2024, 2028, 2032, 2036, 2040]
for i in range(len(milestones)-1):
    s_date, e_date = pd.to_datetime(f'{milestones[i]}-01-01'), pd.to_datetime(f'{milestones[i+1]}-01-01')
    t1, t2 = (s_date - genesis).days / 365.25, (e_date - genesis).days / 365.25
    p1, p2 = 10**underdamped_staircase(t1, *popt), 10**underdamped_staircase(t2, *popt)
    cagr = (p2/p1)**(1/(t2-t1)) - 1
    
    mid = s_date + (e_date - s_date)/2
    ax.text(mid, ax.get_ylim()[0]*1.5, f"{milestones[i]}-{milestones[i+1]}\n{cagr*100:.1f}% CAGR", 
            ha='center', va='bottom', fontsize=10, fontweight='bold', color='purple',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# 4. Equation Box
eq = r"$\log_{10}(P) = \text{Trend} + \text{Step}(t) + \text{Overshoot}(t)$"
ax.text(0.02, 0.95, eq, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

# Formatting
ax.set_yscale('log')
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.set_xlim(pd.to_datetime('2012-01-01'), pd.to_datetime('2040-12-31'))
ax.grid(True, which='both', linestyle='--', alpha=0.3)
ax.legend(loc='upper left', fontsize=12)

st.pyplot(fig)
