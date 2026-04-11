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
st.set_page_config(page_title="TSLA Underdamped Power-Law", layout="wide")
st.markdown("<style>#MainMenu, footer, header {visibility: hidden;}</style>", unsafe_allow_html=True)

st.title("Tesla Underdamped Power-Law Staircase")
st.caption("Market Cap Limited ($20T) • CAGR Epoch Shading • Trough Entry Markers")

# ====================== DATA LOADING ======================
@st.cache_data(ttl=86400)
def load_data():
    df = yf.download('TSLA', period='max', interval='1d', progress=False, auto_adjust=True)
    if df.empty: return None, None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.columns = [str(c).lower() for c in df.columns]
    
    # Extract close price and date
    if 'close' in df.columns:
        df = df[['date', 'close']].copy(); df.columns = ['timestamp', 'price']
    else:
        df = df.iloc[:, [0, 1]]; df.columns = ['timestamp', 'price']

    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    genesis = pd.to_datetime('2010-06-29')
    df['t'] = ((df['timestamp'] - genesis).dt.total_seconds() / (24 * 3600 * 365.25)) + 1e-6
    return df, genesis

df, genesis = load_data()

# ====================== CONSTRAINED POWER-LAW MODEL ======================
# Ceiling Logic: $20T Cap / ~3.75B Shares = $5,333 per share. log10(5333) ≈ 3.727
CEILING_LOG = np.log10(5333)

def power_law_staircase(t, a, b, step_h, step_f, sharp, amp, decay):
    trend = a + b * np.log10(t)
    steps = np.floor(t / step_f)
    breakout = 1 / (1 + np.exp(-sharp * (t % step_f - step_f/2)))
    phase = (t % step_f) / step_f
    # Underdamped Wobble (Volatility)
    wobble = amp * np.exp(-decay * (t % step_f)) * np.sin(2 * np.pi * phase)
    val = trend + (step_h * (steps + breakout)) + wobble
    return np.minimum(val, CEILING_LOG)

p0 = [-0.1, 1.8, 0.45, 6.8, 15.0, 0.2, 0.4]
bounds = ([-5, 0.1, 0.1, 4, 5, -2, 0], [5, 5, 2, 10, 50, 2, 5])
popt, _ = curve_fit(power_law_staircase, df['t'], np.log10(df['price']), p0=p0, bounds=bounds, maxfev=15000)

# ====================== VISUALIZATION ======================
fig, ax = plt.subplots(figsize=(22, 12))

# Period Shading & CAGR Labels
milestones = [2012, 2016, 2020, 2024, 2028, 2032, 2036, 2040]
phase_colors = ['#f0f4ff', '#e8f5e9', '#fff3e0'] # Blue, Green, Orange subtle hues
table_rows = []

for i in range(len(milestones)-1):
    s_yr, e_yr = milestones[i], milestones[i+1]
    s_dt, e_dt = pd.to_datetime(f'{s_yr}-01-01'), pd.to_datetime(f'{e_yr}-01-01')
    
    # Shading the entire period
    ax.axvspan(s_dt, e_dt, color=phase_colors[i % 3], alpha=0.5, zorder=0)
    
    # CAGR Calc
    t1, t2 = (s_dt - genesis).days/365.25, (e_dt - genesis).days/365.25
    p1, p2 = 10**power_law_staircase(t1, *popt), 10**power_law_staircase(t2, *popt)
    cagr = ((p2/p1)**(1/(t2-t1)) - 1) * 100
    
    # Purple CAGR Labels
    ax.text(s_dt + pd.Timedelta(days=730), 1.8e5, f"{cagr:.1f}% CAGR", color='purple', fontsize=12, fontweight='bold', ha='center')
    table_rows.append({"Epoch": f"{s_yr}-{e_yr}", "Start Price": f"${p1:,.0f}", "End Price": f"${p2:,.0f}", "CAGR": f"{cagr:.1f}%"})

# GREEN "STUBBY" TROUGH ARROWS (Lows before next period highs)
step_freq = popt[3]
# Identifying the undershoot troughs (3/4 through the period frequency)
trough_points = np.arange(step_freq * 0.75, 40, step_freq)
for kt in trough_points:
    k_dt = genesis + pd.to_timedelta(kt * 365.25, unit='D')
    if pd.to_datetime('2012-01-01') < k_dt < pd.to_datetime('2040-12-31'):
        # Get model price at that trough point (the undershoot)
        model_p = 10**power_law_staircase(kt, *popt)
        
        # Plot Stubby Green Arrow *above* the trough, pointing *down* to it
        ax.annotate('', xy=(k_dt, model_p), xytext=(k_dt, model_p * 2),
                    arrowprops=dict(facecolor='#00aa00', shrink=0.05, width=10, headwidth=20, alpha=0.8))
        
        # Label Date/Price above the arrow
        ax.text(k_dt, model_p * 2.2, f"{k_dt.date()}\nSupport: ${model_p:,.0f}", 
                ha='center', va='bottom', fontsize=10, color='#00aa00', fontweight='bold', rotation=90)

# Main Price Lines
all_dates = pd.date_range('2012-01-01', '2040-12-31', freq='W')
all_t = np.maximum(((all_dates - genesis).total_seconds() / (24*3600)) / 365.25, 1e-6)
ax.plot(df['timestamp'], df['price'], color='crimson', lw=2, label='Actual TSLA Price', zorder=10)
ax.plot(all_dates, 10**power_law_staircase(all_t, *popt), color='black', ls='--', lw=1.5, alpha=0.8, label='Power-Law Staircase')

# Formatting
ax.set_yscale('log')
ax.set_ylim(0.5, 4e5)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.set_xlim(pd.to_datetime('2012-01-01'), pd.to_datetime('2040-12-31'))
ax.grid(True, which='both', linestyle='--', alpha=0.15)

# Custom Legend
legend_elements = [
    Line2D([0], [0], color='crimson', lw=2, label='Actual Price'),
    Line2D([0], [0], color='black', ls='--', label='Model Forecast ($20T Ceiling)'),
    Line2D([0], [0], marker='v', color='w', markerfacecolor='#00aa00', label='Trough Support', markersize=14),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#e8f5e9', label='Epoch Period Phase', markersize=10)
]
ax.legend(handles=legend_elements, loc='lower right', ncol=2)

st.pyplot(fig)

# ====================== DATA TABLE ======================
st.subheader("Projected Epoch Performance (4-Year Windows)")
st.table(pd.DataFrame(table_rows))
