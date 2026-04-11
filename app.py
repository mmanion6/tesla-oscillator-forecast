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
st.set_page_config(page_title="TSLA $20T Ceiling Model", layout="wide")
st.markdown("<style>#MainMenu, footer, header {visibility: hidden;}</style>", unsafe_allow_html=True)

st.title("Tesla Underdamped Power-Law Staircase")
st.caption("Market Cap Constrained ($20T Max) • Periodic CAGR Milestones • Dynamic Regime Shading")

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

# ====================== THE CONSTRAINED MODEL ======================
# Ceiling Logic: $20T Cap / ~3.18B shares ≈ $6,289 per share. log10(6289) ≈ 3.8
PRICE_CEILING_LOG = np.log10(6289)

def constrained_staircase(t, a, b, step_h, step_f, sharp, amp, decay):
    # 1. Power Law Trend (Diminishing returns)
    trend = a + b * np.log10(t)
    
    # 2. Stepping Logic
    steps = np.floor(t / step_f)
    breakout = 1 / (1 + np.exp(-sharp * (t % step_f - step_f/2)))
    
    # 3. Underdamped Wobble (Overshoot/Undershoot)
    phase = (t % step_f) / step_f
    wobble = amp * np.exp(-decay * (t % step_f)) * np.sin(2 * np.pi * phase)
    
    raw_log_p = trend + (step_h * (steps + breakout)) + wobble
    
    # Apply soft-cap approach to mimic market cap ceiling
    return np.minimum(raw_log_p, PRICE_CEILING_LOG)

p0 = [-0.5, 2.0, 0.4, 6.5, 15.0, 0.25, 0.4]
bounds = ([-5, 0.1, 0.1, 4, 5, -2, 0], [5, 10, 2, 10, 50, 2, 5])

popt, _ = curve_fit(constrained_staircase, df['t'], df['log_price'], p0=p0, bounds=bounds, maxfev=20000)

# ====================== PROJECTION & SHADING LOGIC ======================
all_dates = pd.date_range('2012-01-01', '2040-12-31', freq='W')
all_t = ((all_dates - genesis).total_seconds() / (24*3600)) / 365.25
all_t = np.maximum(all_t, 1e-6)

pred_log = constrained_staircase(all_t, *popt)
all_pred_price = 10**pred_log
all_df = pd.DataFrame({'timestamp': all_dates, 'price': all_pred_price, 't': all_t, 'log_p': pred_log})

# Fixed Shading: Based on the instantaneous slope and deviation
# We use the underdamped wobble as the 'risk' indicator
trend_base = popt[0] + popt[1] * np.log10(all_df['t'])
all_df['residual'] = all_df['log_p'] - trend_base
zone_limit = 0.18 # Sensitivity for green/orange

all_df['regime'] = 'consolidation'
all_df.loc[all_df['residual'] < -zone_limit, 'regime'] = 'buy'
all_df.loc[all_df['residual'] > zone_limit, 'regime'] = 'ath'

# ====================== VISUALIZATION ======================
fig, ax = plt.subplots(figsize=(22, 11))

# 1. Corrected Regime Shading
colors = {'buy': '#d4f4d4', 'consolidation': '#e0f0ff', 'ath': '#ffe6cc'}
for regime, color in colors.items():
    mask = all_df['regime'] == regime
    ax.fill_between(all_df['timestamp'], 0.1, 1e6, where=mask, color=color, alpha=0.5, step='mid', zorder=1)

# 2. Lines
ax.plot(df['timestamp'], df['price'], color='crimson', linewidth=1.8, label='Actual TSLA Price', zorder=10)
ax.plot(all_dates, all_pred_price, color='black', linestyle='--', linewidth=1.5, alpha=0.8, label='Market-Cap Limited Fit')

# 3. Purple CAGR Labels (Fixed Placement)
milestones = [2012, 2016, 2020, 2024, 2028, 2032, 2036, 2040]
table_rows = []

for i in range(len(milestones)-1):
    s_yr, e_yr = milestones[i], milestones[i+1]
    t1, t2 = (pd.to_datetime(f'{s_yr}-01-01') - genesis).days/365.25, (pd.to_datetime(f'{e_yr}-01-01') - genesis).days/365.25
    p1, p2 = 10**constrained_staircase(t1, *popt), 10**constrained_staircase(t2, *popt)
    cagr = ((p2/p1)**(1/(t2-t1)) - 1) * 100
    
    # Plot Purple Label
    mid_date = pd.to_datetime(f'{s_yr}-01-01') + pd.Timedelta(days=730)
    # Positioning labels high on the log chart
    ax.text(mid_date, 1.2e5, f"{cagr:.1f}%", color='purple', fontsize=11, fontweight='bold', ha='center', zorder=15)
    
    table_rows.append({"Period": f"{s_yr}-{e_yr}", "Start": f"${p1:,.0f}", "End": f"${p2:,.0f}", "CAGR": f"{cagr:.1f}%"})

# 4. Formatting
ax.set_yscale('log')
ax.set_ylim(0.5, 2e5) # Extra headroom for CAGR labels
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.set_xlim(pd.to_datetime('2012-01-01'), pd.to_datetime('2040-12-31'))
ax.grid(True, which='both', linestyle='--', alpha=0.15)

# Custom Legend
legend_elements = [
    Line2D([0], [0], color='crimson', lw=2, label='Actual Price'),
    Line2D([0], [0], color='black', lw=1.5, ls='--', label='Model Fit ($20T Cap)'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#d4f4d4', label='Buy Zone'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#e0f0ff', label='Consolidation'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#ffe6cc', label='ATH Zone')
]
ax.legend(handles=legend_elements, loc='lower right', ncol=2, frameon=True, shadow=True)

st.pyplot(fig)

# ====================== TABLE ======================
st.subheader("Projected Cycle Performance")
st.table(pd.DataFrame(table_rows))
