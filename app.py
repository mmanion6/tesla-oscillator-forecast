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
    # Fetching with auto_adjust to ensure column names are consistent
    df = yf.download('TSLA', period='max', interval='1d', progress=False, auto_adjust=True)
    
    if df.empty:
        return None, None
    
    # FIX: Handle potential multi-index columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Reset index to get Date column
    df = df.reset_index()
    
    # Ensure column names are clean
    df.columns = [str(c).lower() for c in df.columns]
    
    # Identify price column (usually 'close')
    if 'close' in df.columns:
        df = df[['date', 'close']].copy()
        df.columns = ['timestamp', 'price']
    else:
        # Fallback if names are unexpected
        df = df.iloc[:, [0, 1]]
        df.columns = ['timestamp', 'price']

    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    
    genesis = pd.to_datetime('2010-06-29')
    # Time in years with epsilon to prevent log(0)
    df['t'] = ((df['timestamp'] - genesis).dt.total_seconds() / (24 * 3600 * 365.25)) + 1e-6
    df['log_price'] = np.log10(df['price'])
    return df, genesis

df, genesis = load_data()

# SAFETY CHECK: Stop the app if data is empty
if df is None or len(df) == 0:
    st.error("🚨 Error: No data found for TSLA. Yahoo Finance might be down or blocking the request. Please refresh.")
    st.stop()

# ====================== THE UNDERDAMPED MODEL ======================
def underdamped_staircase(t, a, b, step_h, step_f, sharp, amp, decay):
    trend = a + b * np.log(t)
    steps = np.floor(t / step_f)
    breakout = 1 / (1 + np.exp(-sharp * (t % step_f - step_f/2)))
    phase = (t % step_f) / step_f
    wobble = amp * np.exp(-decay * (t % step_f)) * np.sin(2 * np.pi * phase)
    return trend + (step_h * (steps + breakout)) + wobble

# Initial Guesses
p0 = [-1.0, 1.5, 0.6, 6.5, 12.0, 0.3, 0.4]
bounds = ([-10, 0, 0.1, 3, 5, -2, 0], [5, 5, 2, 10, 50, 2, 5])

try:
    popt, _ = curve_fit(underdamped_staircase, df['t'], df['log_price'], p0=p0, bounds=bounds, maxfev=20000)
except Exception as e:
    st.error(f"📈 Model fitting failed: {e}")
    st.stop()

# ====================== PROJECTION & PLOTTING ======================
all_dates = pd.date_range('2012-01-01', '2040-12-31', freq='W')
all_t = ((all_dates - genesis).total_seconds() / (24*3600)) / 365.25
all_t = np.maximum(all_t, 1e-6)

pred_log = underdamped_staircase(all_t, *popt)
all_pred_price = 10**pred_log

fig, ax = plt.subplots(figsize=(22, 11))
ax.plot(df['timestamp'], df['price'], color='crimson', linewidth=2, label='Actual TSLA Price', zorder=10)
ax.plot(all_dates, all_pred_price, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Model Trend')

# CAGR Blocks
milestones = [2012, 2016, 2020, 2024, 2028, 2032, 2036, 2040]
for i in range(len(milestones)-1):
    s_date, e_date = pd.to_datetime(f'{milestones[i]}-01-01'), pd.to_datetime(f'{milestones[i+1]}-01-01')
    t1, t2 = (s_date - genesis).days / 365.25, (e_date - genesis).days / 365.25
    p1, p2 = 10**underdamped_staircase(t1, *popt), 10**underdamped_staircase(t2, *popt)
    cagr = (p2/p1)**(1/(t2-t1)) - 1
    mid = s_date + (e_date - s_date)/2
    ax.text(mid, df['price'].min(), f"{milestones[i]}-{milestones[i+1]}\n{cagr*100:.1f}% CAGR", 
            ha='center', va='bottom', fontsize=10, fontweight='bold', color='purple',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

ax.set_yscale('log')
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.set_xlim(pd.to_datetime('2012-01-01'), pd.to_datetime('2040-12-31'))
ax.grid(True, which='both', linestyle='--', alpha=0.3)
ax.legend(loc='upper left', fontsize=12)

st.pyplot(fig)
