import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import yfinance as yf
from scipy.optimize import curve_fit

# Page Config
st.set_page_config(page_title="TSLA Oscillator Forecast", layout="wide")

# CSS to mimic the clean original look
st.markdown("""<style>#MainMenu, footer, header {visibility: hidden;}</style>""", unsafe_allow_html=True)

st.title("Tesla Damped-Oscillator Model")
st.caption("Live Update • Buy / Consolidation / ATH Zones")

# ====================== DATA LOADING ======================
@st.cache_data(ttl=86400)
def load_data():
    df = yf.download('TSLA', period='max', interval='1d', progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[['Close']].reset_index()
    df.columns = ['timestamp', 'price']
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    
    genesis = pd.to_datetime('2010-06-29')
    # Use 1e-6 offset to avoid ln(0) errors
    df['t'] = ((df['timestamp'] - genesis).dt.total_seconds() / (24 * 3600 * 365.25)) + 1e-6
    df['log_price'] = np.log10(df['price'])
    return df, genesis

df, genesis = load_data()

# ====================== MODEL & FIT ======================
def damped_osc(t, a, b, A, gamma, omega, phi):
    return a + b * np.log(t) + A * np.exp(-gamma * t) * np.sin(omega * t + phi)

p0 = [-2.0, 2.5, 0.3, 0.01, 1.57, 0.0]
bounds = ([-10, 0.1, -2, 0.0001, 0.1, -np.pi], [5, 10, 2, 0.5, 10, np.pi])

popt, _ = curve_fit(damped_osc, df['t'], df['log_price'], p0=p0, bounds=bounds, maxfev=10000)
a, b, A, gamma, omega, phi = popt

# ====================== FORECAST DATA ======================
chart_dates = pd.date_range('2012-01-01', '2040-12-31', freq='D')
all_t = ((chart_dates - genesis).total_seconds() / (24*3600)) / 365.25
all_t = np.maximum(all_t, 1e-6)

pred_log = damped_osc(all_t, *popt)
all_pred_price = 10 ** pred_log
all_osc = A * np.exp(-gamma * all_t) * np.sin(omega * all_t + phi)
all_trend = 10 ** (a + b * np.log(all_t))
thresh = 0.20 * np.abs(A * np.exp(-gamma * all_t))

all_df = pd.DataFrame({
    'timestamp': chart_dates,
    'pred_price': all_pred_price,
    'osc': all_osc,
    'trend': all_trend,
    'regime': np.where(all_osc < -thresh, 'buy', np.where(all_osc > thresh, 'ath', 'consolidation')),
    'is_hist': chart_dates <= df['timestamp'].max()
})

# ====================== PLOTTING ======================
fig, ax = plt.subplots(figsize=(20, 10))

# Background Regime Shading
colors = {'buy': '#d4f4d4', 'consolidation': '#e0f0ff', 'ath': '#ffe6cc'}
for regime, color in colors.items():
    mask = all_df['regime'] == regime
    ax.fill_between(all_df['timestamp'], 0.1, all_df['pred_price']*10, where=mask, color=color, alpha=0.7, zorder=1)

# Price and Trend Lines
ax.plot(df['timestamp'], df['price'], color='crimson', linewidth=2, label='Actual TSLA Price', zorder=10)
ax.plot(all_df['timestamp'], all_df['trend'], color='navy', linestyle=':', alpha=0.5, label='Log Trend Line')

# Split Historical vs Forecast lines
hist = all_df[all_df['is_hist']]
fut = all_df[~all_df['is_hist']]

ax.plot(hist['timestamp'], hist['pred_price'], color='black', linestyle='--', alpha=0.4, label='Model Fit (Historical)')
ax.plot(fut['timestamp'], fut['pred_price'], color='blue', linewidth=2.5, label='Model Forecast (2025-2040)')

# Formatting
ax.set_yscale('log')
ax.set_ylim(df['price'].min()*0.5, all_pred_price.max()*2)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.set_xlim(pd.to_datetime('2012-01-01'), pd.to_datetime('2040-12-31'))
ax.grid(True, which='both', linestyle='--', alpha=0.2)
ax.legend(loc='lower right', ncol=2)

# Equation Box
eq_txt = f"log10(P) = {a:.2f} + {b:.2f}ln(t) + {A:.2f}e^{{-{gamma:.3f}t}}sin({omega:.2f}t + {phi:.2f})"
ax.text(0.02, 0.95, eq_txt, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8), fontsize=12)

st.pyplot(fig)

# Metrics footer
st.divider()
c1, c2, c3 = st.columns(3)
c1.metric("Current Regime", all_df[all_df['is_hist']].iloc[-1]['regime'].upper())
c2.metric("Est. Cycle Length", f"{2*np.pi/omega:.1f} Years")
c3.metric("Damping Factor", f"{gamma:.4f}")
