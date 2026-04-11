import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import curve_fit

st.set_page_config(page_title="TSLA Exponential Staircase", layout="wide")

# ====================== DATA ======================
@st.cache_data(ttl=86400)
def load_data():
    df = yf.download('TSLA', period='max', interval='1d', progress=False)
    df = df[['Close']].reset_index()
    df.columns = ['timestamp', 'price']
    genesis = pd.to_datetime('2010-06-29')
    # Time in years since IPO
    df['t'] = (df['timestamp'] - genesis).dt.total_seconds() / (24 * 3600 * 365.25)
    return df, genesis

df, genesis = load_data()

# ====================== STAIRCASE MODEL ======================
def staircase_model(t, a, b, step_height, step_freq, sharpness):
    """
    a, b: Exponential trend params
    step_height: How big each 'jump' is
    step_freq: How many years between jumps
    sharpness: How vertical the breakout is
    """
    trend = a + b * t
    # The 'Staircase' component
    steps = np.floor(t / step_freq)
    # The 'Breakout' transition (Sigmoid)
    breakout = 1 / (1 + np.exp(-sharpness * (t % step_freq - step_freq/2)))
    
    return trend + (step_height * (steps + breakout))

# Fit to Log Price
log_p = np.log10(df['price'])
# Initial guesses: trend starts low, grows, jumps every ~6 years
p0 = [-1.0, 0.1, 0.5, 6.0, 10.0] 
popt, _ = curve_fit(staircase_model, df['t'], log_p, p0=p0, maxfev=10000)

# ====================== PROJECTION ======================
future_dates = pd.date_range('2012-01-01', '2040-12-31', freq='W')
future_t = (future_dates - genesis).total_seconds() / (24 * 3600 * 365.25)
pred_log_price = staircase_model(future_t, *popt)
pred_price = 10**pred_log_price

# ====================== PLOT ======================
fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(df['timestamp'], df['price'], color='crimson', label='Actual TSLA Price', alpha=0.9)
ax.plot(future_dates, pred_price, color='black', linestyle='--', label='Exponential Staircase Model', linewidth=2)

# Shading the consolidation zones
for i in range(len(future_dates)-1):
    # If the growth rate is low, it's a consolidation zone
    if (pred_log_price[i+1] - pred_log_price[i]) < 0.001:
        ax.axvspan(future_dates[i], future_dates[i+1], color='gray', alpha=0.05)

ax.set_yscale('log')
ax.set_title("Tesla 'Staircase' Growth Model (2012-2040)", fontsize=16)
ax.set_ylabel("Price (USD, Log Scale)")
ax.grid(True, which='both', alpha=0.2)
ax.legend()

st.pyplot(fig)

st.info(f"Model detects a major step breakout every **{popt[3]:.1f} years**.")
