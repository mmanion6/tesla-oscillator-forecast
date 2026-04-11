import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import yfinance as yf
from scipy.optimize import curve_fit
from datetime import datetime

# Page Config
st.set_page_config(page_title="TSLA Oscillator Forecast", layout="wide")

# Hide Streamlit Header/Footer
hide_style = """<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}</style>"""
st.markdown(hide_style, unsafe_allow_html=True)

st.title("Tesla Damped-Oscillator Model")
st.caption("Historical Cycles & Forecast (2012–2040) • Logarithmic Regression")

# ====================== DATA LOADING ======================
@st.cache_data(ttl=86400)
def load_data():
    df = yf.download('TSLA', period='max', interval='1d', progress=False)
    df = df[['Close']].reset_index()
    df.columns = ['timestamp', 'price']
    df = df.dropna(subset=['price']).reset_index(drop=True)
    
    genesis = pd.to_datetime('2010-06-29')
    df['t'] = (df['timestamp'] - genesis).dt.total_seconds() / (24 * 3600 * 365.25)
    df['log_price'] = np.log10(df['price'])
    return df, genesis

df, genesis = load_data()

# ====================== MODELING ======================
def damped_osc(t, a, b, A, gamma, omega, phi):
    # a + b*ln(t) provides the growth curve; the rest is the sine wave decay
    return a + b * np.log(t) + A * np.exp(-gamma * t) * np.sin(omega * t + phi)

# Fit the model
p0 = [-2.0, 2.5, 0.3, 0.01, 1.57, 0.0]
bounds = ([-10, 0.1, -2, 0.0001, 0.1, -np.pi], [5, 10, 2, 0.5, 10, np.pi])

popt, _ = curve_fit(damped_osc, df['t'], df['log_price'], p0=p0, bounds=bounds, maxfev=10000)
a, b, A, gamma, omega, phi = popt

# ====================== FORECAST GENERATION ======================
chart_start = pd.to_datetime('2012-01-01')
chart_end = pd.to_datetime('2040-12-31')
all_dates = pd.date_range(chart_start, chart_end, freq='D')
all_t = ((all_dates - genesis).total_seconds() / (24*3600)) / 365.25

pred_log_price = damped_osc(all_t, *popt)
all_pred_price = 10 ** pred_log_price
all_osc = A * np.exp(-gamma * all_t) * np.sin(omega * all_t + phi)
all_trend = 10 ** (a + b * np.log(all_t))

# Threshold for Buy/ATH zones
thresh = 0.20 * np.abs(A * np.exp(-gamma * all_t))

all_df = pd.DataFrame({
    'timestamp': all_dates,
    'pred_price': all_pred_price,
    'osc': all_osc,
    'regime': np.where(all_osc < -thresh, 'buy', 
              np.where(all_osc > thresh, 'ath', 'consolidation')),
    'is_hist': all_dates <= df['timestamp'].max()
})

# ====================== VISUALIZATION ======================
def create_plot():
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Background Shading
    colors = {'buy': '#d4f4d4', 'consolidation': '#e0f0ff', 'ath': '#ffe6cc'}
    for regime, color in colors.items():
        mask = all_df['regime'] == regime
        ax.fill_between(all_df['timestamp'], 0.1, all_df['pred_price']*10, 
                        where=mask, color=color, alpha=0.5, zorder=1)

    # Actual Price
    ax.plot(df['timestamp'], df['price'], color='crimson', label='Actual Price', linewidth=1.5, zorder=10)
    
    # Model Lines
    hist = all_df[all_df['is_hist']]
    fut = all_df[~all_df['is_hist']]
    
    ax.plot(hist['timestamp'], hist['pred_price'], color='black', linestyle='--', alpha=0.3, label='Model Fit')
    ax.plot(fut['timestamp'], fut['pred_price'], color='blue', linewidth=2, label='Model Forecast')
    
    # Formatting
    ax.set_yscale('log')
    ax.set_ylim(df['price'].min() * 0.8, all_pred_price.max() * 1.2)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax.set_xlim(chart_start, chart_end)
    
    ax.grid(True, which='both', linestyle='--', alpha=0.2)
    ax.legend(loc='lower right', frameon=True)
    ax.set_title(f"TSLA Price vs Damped Oscillator (Projected Cycle: {2*np.pi/omega:.1f} yrs)")
    
    plt.tight_layout()
    return fig

st.pyplot(create_plot())

# Equation Display
st.divider()
st.latex(r"P(t) = 10^{a + b \ln(t) + A e^{-\gamma t} \sin(\omega t + \phi)}")
col1, col2, col3 = st.columns(3)
col1.metric("Damping (γ)", f"{gamma:.4f}")
col2.metric("Cycle Length", f"{2*np.pi/omega:.2f} Years")
col3.metric("Current Regime", all_df.iloc[len(hist)]['regime'].upper())
