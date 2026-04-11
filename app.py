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
st.set_page_config(page_title="TSLA Underdamped Staircase", layout="wide")
st.markdown("<style>#MainMenu, footer, header {visibility: hidden;}</style>", unsafe_allow_html=True)

st.title("Tesla Underdamped Staircase Model")
st.caption("Live Analysis • Stepped Exponential Growth with Cyclical Overshoots")

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
    df['log_price'] = np.log10(df['price'])
    return df, genesis

df, genesis = load_data()

if df is None:
    st.error("🚨 Failed to fetch data. Please check your connection and requirements.txt.")
    st.stop()

# ====================== MODELING ======================
def underdamped_staircase(t, a, b, step_h, step_f, sharp, amp, decay):
    trend = a + b * np.log(t)
    steps = np.floor(t / step_f)
    breakout = 1 / (1 + np.exp(-sharp * (t % step_f - step_f/2)))
    phase = (t % step_f) / step_f
    wobble = amp * np.exp(-decay * (t % step_f)) * np.sin(2 * np.pi * phase)
    return trend + (step_h * (steps + breakout)) + wobble

p0 = [-1.0, 1.5, 0.6, 6.5, 12.0, 0.3, 0.4]
bounds = ([-10, 0, 0.1, 3, 5, -2, 0], [5, 5, 2, 10, 50, 2, 5])

popt, _ = curve_fit(underdamped_staircase, df['t'], df['log_price'], p0=p0, bounds=bounds, maxfev=20000)

# ====================== PROJECTION & ZONES ======================
all_dates = pd.date_range('2012-01-01', '2040-12-31', freq='W')
all_t = ((all_dates - genesis).total_seconds() / (24*3600)) / 365.25
all_t = np.maximum(all_t, 1e-6)

pred_log = underdamped_staircase(all_t, *popt)
all_pred_price = 10**pred_log

all_df = pd.DataFrame({'timestamp': all_dates, 'price': all_pred_price, 't': all_t, 'log_p': pred_log})
# Zone Logic based on distance from the pure trend (a + b*ln(t))
trend_only = popt[0] + popt[1] * np.log(all_df['t'])
all_df['deviation'] = all_df['log_p'] - trend_only
zone_thresh = 0.18
all_df['regime'] = np.where(all_df['deviation'] < -zone_thresh, 'buy',
                   np.where(all_df['deviation'] > zone_thresh, 'ath', 'consolidation'))

# ====================== PLOTTING ======================
fig, ax = plt.subplots(figsize=(20, 10))

# Shading Regimes
regime_colors = {'buy': '#d4f4d4', 'consolidation': '#e0f0ff', 'ath': '#ffe6cc'}
for regime, color in regime_colors.items():
    mask = all_df['regime'] == regime
    ax.fill_between(all_df['timestamp'], 0.1, all_df['price']*20, where=mask, color=color, alpha=0.6, zorder=1)

# Price Lines
ax.plot(df['timestamp'], df['price'], color='crimson', linewidth=1.5, label='Actual TSLA Price', zorder=10)
ax.plot(all_dates, all_pred_price, color='black', linestyle='--', linewidth=1.2, alpha=0.8, label='Model Trend')

# CAGR Calcs for the Table
milestones = [2012, 2016, 2020, 2024, 2028, 2032, 2036, 2040]
cagr_data = []

for i in range(len(milestones)-1):
    s_yr, e_yr = milestones[i], milestones[i+1]
    t1, t2 = (pd.to_datetime(f'{s_yr}-01-01') - genesis).days/365.25, (pd.to_datetime(f'{e_yr}-01-01') - genesis).days/365.25
    p1, p2 = 10**underdamped_staircase(t1, *popt), 10**underdamped_staircase(t2, *popt)
    cagr = ((p2/p1)**(1/(t2-t1)) - 1) * 100
    cagr_data.append({"Period": f"{s_yr}-{e_yr}", "Start Price": f"${p1:,.2f}", "End Price": f"${p2:,.2f}", "CAGR": f"{cagr:.2f}%"})
    
    # CAGR Text on Chart
    mid = pd.to_datetime(f'{s_yr}-01-01') + pd.Timedelta(days=730)
    ax.text(mid, 0.5, f"{cagr:.1f}%", ha='center', fontweight='bold', color='purple', fontsize=9, transform=ax.get_xaxis_transform())

# Formatting
ax.set_yscale('log')
ax.set_ylim(1, all_pred_price.max()*5)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.set_xlim(pd.to_datetime('2012-01-01'), pd.to_datetime('2040-12-31'))
ax.grid(True, which='both', linestyle='--', alpha=0.2)

# Custom Legend
legend_elements = [
    Line2D([0], [0], color='crimson', lw=2, label='Actual Price'),
    Line2D([0], [0], color='black', lw=1.5, ls='--', label='Underdamped Staircase'),
    Line2D([0], [0], marker='s', color='w', label='Buy Zone', markerfacecolor='#d4f4d4', markersize=10),
    Line2D([0], [0], marker='s', color='w', label='Consolidation', markerfacecolor='#e0f0ff', markersize=10),
    Line2D([0], [0], marker='s', color='w', label='ATH Zone', markerfacecolor='#ffe6cc', markersize=10)
]
ax.legend(handles=legend_elements, loc='lower right', ncol=2)

st.pyplot(fig)

# ====================== DATA TABLE ======================
st.subheader("Model Projection & Performance Table")
st.table(pd.DataFrame(cagr_data))
