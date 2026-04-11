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
st.set_page_config(page_title="TSLA Power-Law Staircase", layout="wide")
st.markdown("<style>#MainMenu, footer, header {visibility: hidden;}</style>", unsafe_allow_html=True)

st.title("Tesla Power-Law Staircase Model")
st.caption("Growth with Diminishing Returns, Stepped Breakouts, and Underdamped Volatility")

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
    # Time offset +1e-6 to ensure no log(0)
    df['t'] = ((df['timestamp'] - genesis).dt.total_seconds() / (24 * 3600 * 365.25)) + 1e-6
    df['log_price'] = np.log10(df['price'])
    return df, genesis

df, genesis = load_data()

# ====================== THE POWER-LAW STAIRCASE MODEL ======================
def power_law_staircase(t, a, b, step_h, step_f, sharp, amp, decay):
    # TREND: Now uses a power-law (log-log) growth limiter
    # a + b * log10(t) instead of linear t
    trend = a + b * np.log10(t)
    
    # STEPS: Breakouts triggered every 'step_f' years
    steps = np.floor(t / step_f)
    breakout = 1 / (1 + np.exp(-sharp * (t % step_f - step_f/2)))
    
    # OSCILLATION: Underdamped wobble (momentum + correction)
    phase = (t % step_f) / step_f
    wobble = amp * np.exp(-decay * (t % step_f)) * np.sin(2 * np.pi * phase)
    
    return trend + (step_h * (steps + breakout)) + wobble

# Guesses: a (intercept), b (growth exponent), step_h, step_f, sharp, amp, decay
p0 = [-0.5, 2.0, 0.4, 6.0, 15.0, 0.25, 0.4]
bounds = ([-5, 0.1, 0.1, 3, 5, -2, 0], [5, 10, 2, 10, 50, 2, 5])

popt, _ = curve_fit(power_law_staircase, df['t'], df['log_price'], p0=p0, bounds=bounds, maxfev=20000)

# ====================== PROJECTION & SHADING ======================
all_dates = pd.date_range('2012-01-01', '2040-12-31', freq='W')
all_t = ((all_dates - genesis).total_seconds() / (24*3600)) / 365.25
all_t = np.maximum(all_t, 1e-6)

pred_log = power_law_staircase(all_t, *popt)
all_pred_price = 10**pred_log

all_df = pd.DataFrame({'timestamp': all_dates, 'price': all_pred_price, 't': all_t, 'log_p': pred_log})

# Fixed Shading Logic: Define zones based on the underlying Power-Law trend
trend_base = popt[0] + popt[1] * np.log10(all_df['t'])
all_df['residual'] = all_df['log_p'] - trend_base
zone_lim = 0.20 # 20% log deviation

all_df['regime'] = 'consolidation'
all_df.loc[all_df['residual'] < -zone_lim, 'regime'] = 'buy'
all_df.loc[all_df['residual'] > zone_lim, 'regime'] = 'ath'

# ====================== PLOTTING ======================
fig, ax = plt.subplots(figsize=(20, 10))

# Fixed Regime Shading
colors = {'buy': '#d4f4d4', 'consolidation': '#e0f0ff', 'ath': '#ffe6cc'}
for regime, color in colors.items():
    mask = all_df['regime'] == regime
    # Create distinct blocks to prevent "smearing" at boundaries
    ax.fill_between(all_df['timestamp'], 0.1, 1e6, where=mask, color=color, alpha=0.6, step='mid', zorder=1)

# Price lines
ax.plot(df['timestamp'], df['price'], color='crimson', linewidth=1.5, label='Actual Price', zorder=10)
ax.plot(all_dates, all_pred_price, color='black', linestyle='--', linewidth=1.2, alpha=0.8, label='Power-Law Model')

# Formatting
ax.set_yscale('log')
ax.set_ylim(df['price'].min()*0.5, all_pred_price.max()*5)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.set_xlim(pd.to_datetime('2012-01-01'), pd.to_datetime('2040-12-31'))
ax.grid(True, which='both', linestyle='--', alpha=0.2)

# Labels & Legend
legend_elements = [
    Line2D([0], [0], color='crimson', lw=2, label='Actual Price'),
    Line2D([0], [0], color='black', lw=1.5, ls='--', label='Model Fit'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#d4f4d4', label='Buy Zone'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#ffe6cc', label='ATH Zone')
]
ax.legend(handles=legend_elements, loc='lower right', ncol=2)

st.pyplot(fig)

# CAGR Calculation Table
st.subheader("Growth Statistics (4-Year Windows)")
milestones = [2012, 2016, 2020, 2024, 2028, 2032, 2036, 2040]
table_rows = []
for i in range(len(milestones)-1):
    s, e = milestones[i], milestones[i+1]
    t1, t2 = (pd.to_datetime(f'{s}-01-01') - genesis).days/365.25, (pd.to_datetime(f'{e}-01-01') - genesis).days/365.25
    p1, p2 = 10**power_law_staircase(t1, *popt), 10**power_law_staircase(t2, *popt)
    cagr = ((p2/p1)**(1/(t2-t1)) - 1) * 100
    table_rows.append({"Period": f"{s}-{e}", "Start Price": f"${p1:,.2f}", "End Price": f"${p2:,.2f}", "CAGR": f"{cagr:.1f}%"})

st.table(pd.DataFrame(table_rows))
