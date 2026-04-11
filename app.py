# ====================== CHART ======================
def generate_plot():
    fig, ax = plt.subplots(figsize=(16, 9))  # Slightly adjusted for web screens

    # Background regime shading
    regime_colors = {'buy': '#d4f4d4', 'consolidation': '#e0f0ff', 'ath': '#ffe6cc'}
    for regime, color in regime_colors.items():
        mask = all_df['regime'] == regime
        ax.fill_between(all_df['timestamp'], 0.1, all_df['pred_price']*10, 
                        where=mask, color=color, alpha=0.6, zorder=1)

    # Actual price
    ax.plot(df['timestamp'], df['price'], color='crimson', linewidth=1.5, label='Actual TSLA Price', zorder=6)

    # Plotting helper
    def plot_regime(df_in, regime, color, ls, lw, label):
        mask = df_in['regime'] == regime
        if mask.any():
            ax.plot(df_in['timestamp'], df_in['pred_price'].where(mask), 
                    color=color, linestyle=ls, linewidth=lw, label=label, zorder=5)

    hist = all_df[all_df['is_hist']]
    fut = all_df[~all_df['is_hist']]

    plot_regime(hist, 'buy', '#00aa00', '--', 1.5, 'Buy Zone (Hist)')
    plot_regime(hist, 'consolidation', '#1e90ff', ':', 1.5, 'Consolidation (Hist)')
    plot_regime(hist, 'ath', '#e07000', '-', 1.5, 'ATH/Sell (Hist)')

    plot_regime(fut, 'buy', 'limegreen', '--', 2.5, 'Buy Zone (Forecast)')
    plot_regime(fut, 'consolidation', 'dodgerblue', ':', 2.0, 'Consolidation (Forecast)')
    plot_regime(fut, 'ath', 'darkorange', '-', 2.5, 'ATH/Sell (Forecast)')

    ax.plot(all_dates, all_trend, color='navy', linestyle=':', linewidth=1.2, alpha=0.5, label='Log Trend')
    ax.axvline(df['timestamp'].max(), color='black', linestyle='-', linewidth=1, alpha=0.5)

    # Milestones (Simplified for clarity)
    # ... [Keep your milestone logic here] ...

    # Formatting
    ax.set_yscale('log')
    ax.set_ylim(df['price'].min() * 0.5, all_pred_price.max() * 1.5)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    ax.set_xlim(chart_start, chart_end)
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.legend(loc='lower right', fontsize=8, ncol=2)
    
    fig.tight_layout()
    return fig

st.pyplot(generate_plot(), use_container_width=True)
