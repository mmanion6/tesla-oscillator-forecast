import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import yfinance as yf
from scipy.optimize import curve_fit

# 1. Setup
st.set_page_config(page_title="TSLA Oscillator Forecast", layout="wide")

# 2. Robust Data Loading
@st.cache_data(ttl=86400)
def load_data():
    try:
        ticker = yf.Ticker("TSLA")
        df = ticker.history(period="max")
        
        if df.empty:
            return None, None
            
        df = df[['Close']].reset_index()
        df.columns = ['timestamp', 'price']
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        
        genesis = pd.to_datetime('2010-06-29')
        # + 1e-6 prevents ln(0) errors in the math model
        df['t'] = ((df['timestamp'] - genesis).dt.total_seconds() / (24 * 3600 * 365.25)) + 1e-6
        df['log_price'] = np.log10(df['price'])
        return df, genesis
    except Exception as e:
        st.error(f"Data Fetch Error: {e}")
        return None, None

df, genesis = load_data()

if df is None or df.empty:
    st.error("⚠️ Could not fetch data from Yahoo Finance. Please check your internet or try again later.")
    st.stop()

# 3. The Model
def damped_osc(t, a, b, A, gamma, omega, phi):
    return a + b * np.log(t) + A * np.exp(-gamma * t) * np.sin(omega * t + phi)

# Optimization
p0 = [-2.0, 2.5, 0.3, 0.01, 1.57, 0.0]
bounds = ([-10, 0.1, -2, 0.0001, 0.1, -np.pi], [5, 10, 2, 0.5, 10, np.pi])

try:
    popt, _ = curve_fit(damped_osc, df['t'], df['log_price'], p0=p0, bounds=bounds, maxfev=10000)
except ValueError as e:
    st.error(f"Model fitting failed: {e}. Try adjusting the IPO date or initial parameters.")
    st.stop()

# 4. Forecast Calculations
all_dates = pd.date_range('2012-01-01', '2040-12-31', freq='D')
all_t = ((all_dates - genesis).total_seconds() / (24*3600)) / 365.25
all_t = np.maximum(all_t, 1e-6) # Safety for log calculations

pred_price = 10 ** damped_osc(all_t, *popt)
osc_component = popt[2] * np.exp(-popt[3] * all_t) * np.sin(popt[4] * all_t + popt[5])

# 5. UI & Plotting
st.title("Tesla Damped-Oscillator Forecast")

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['timestamp'], df['price'], color='crimson', label='Actual Price', alpha=0.8)
ax.plot(all_dates, pred_price, color='royalblue', linestyle='--', label='Model Forecast', alpha=0.6)

ax.set_yscale('log')
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.legend()
ax.grid(True, which="both", alpha=0.2)

st.pyplot(fig)
