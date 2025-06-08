import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import scipy.optimize as opt
from datetime import datetime
import warnings

# ---- Page Config & Styling ----
warnings.filterwarnings('ignore', category=FutureWarning)
sns.set_theme(style='darkgrid')
plt.style.use('seaborn-v0_8-darkgrid')

st.set_page_config(
    page_title='SHI Zertifikate im Vergleich',
    page_icon='📊',
    layout='wide'
)

RISK_FREE_RATE = 0.02  # 2% p.a.
BUSINESS_DAYS = 'B'

# ---- Helper Functions ----
def fetch_data(tickers: list[str], start: datetime, end: datetime) -> tuple[dict, dict, dict]:
    """
    Downloads adjusted close prices, computes daily returns and cumulative returns,
    reindexed to a continuous business-day calendar (forward-filled).
    """
    bidx = pd.date_range(start, end, freq=BUSINESS_DAYS)
    prices, returns, cumulatives = {}, {}, {}
    for t in tickers:
        df = yf.download(t, start=start, end=end, progress=False)['Close']
        s = df.reindex(bidx).ffill().dropna()
        r = s.pct_change().dropna()
        c = (1 + r).cumprod()
        prices[t], returns[t], cumulatives[t] = s, r, c
    return prices, returns, cumulatives


def compute_metrics(returns: pd.Series, cumulative: pd.Series) -> dict:
    """
    Calculates a suite of risk metrics for a single return series.
    """
    stats = {}
    # Ensure numeric types
    r = to_1d_series(returns)
    c = to_1d_series(cumulative)
    if len(r) < 2 or len(c) < 2:
        return stats
    # Time delta
    days = (c.index[-1] - c.index[0]).days
    total_ret = float(c.iloc[-1] / c.iloc[0] - 1)
    stats['Total Return'] = total_ret
    stats['Annual Return'] = float((1 + total_ret)**(365 / days) - 1) if days > 0 else np.nan
    annual_vol = float(r.std() * np.sqrt(252))
    stats['Annual Volatility'] = annual_vol
    # Sharpe Ratio
    stats['Sharpe Ratio'] = ((stats['Annual Return'] - RISK_FREE_RATE) / annual_vol) if annual_vol > 0 else np.nan
    # Sortino Ratio
    downside = r[r < 0]
    std_down = float(downside.std(ddof=0)) if len(downside) > 0 else np.nan
    stats['Sortino Ratio'] = ((r.mean() - RISK_FREE_RATE) / std_down * np.sqrt(252)) if std_down > 0 else np.nan
    # Drawdowns
    drawdown = c / c.cummax() - 1
    stats['Max Drawdown'] = float(drawdown.min())
    stats['Calmar Ratio'] = float(stats['Annual Return'] / abs(stats['Max Drawdown'])) if stats['Max Drawdown'] < 0 else np.nan
    # Value at Risk
    var95 = float(r.quantile(0.05))
    stats['VaR (95%)'] = var95
    stats['CVaR (95%)'] = float(r[r <= var95].mean())
    # Omega & Tail
    stats['Omega Ratio'] = float(omega_ratio(r))
    stats['Tail Ratio'] = float(tail_ratio(r))
    # Basic Win/Loss Stats
    stats['Win Rate'] = float((r > 0).mean())
    stats['Avg Win'] = float(r[r > 0].mean())
    stats['Avg Loss'] = float(r[r < 0].mean())
    stats['Profit Factor'] = float(-stats['Avg Win'] / stats['Avg Loss']) if stats['Avg Loss'] < 0 else np.nan
    # Monthly
    monthly = r.resample('M').apply(lambda x: (1 + x).prod() - 1)
    stats['Positive Months'] = float((monthly > 0).mean())
    return stats

# ---- Plotting ----
def plot_line(data: dict[str, pd.Series], title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, series in data.items():
        ax.plot(series.index, series.values, label=name, linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Datum')
    ax.legend(fontsize=8)
    st.pyplot(fig)

# ---- Main App ----
def main():
    st.title('📊 SHI Zertifikate im Vergleich')
    st.caption('Performance‑, Risiko‑ und Benchmarkanalyse')
    # Sidebar
    with st.sidebar:
        start = st.date_input('Startdatum', datetime(2023,1,1))
        end = st.date_input('Enddatum', datetime.today())
        tickers = st.text_area('Tickers (kommagetrennt)', 'AAPL, MSFT').split(',')
        tickers = [t.strip().upper() for t in tickers if t.strip()]
    if len(tickers) < 1:
        st.error('Bitte mindestens einen Ticker angeben.')
        return
    with st.spinner('Lade Daten...'):
        prices, returns, cumulatives = fetch_data(tickers, start, end)
    # Synchronisiere Index (bereits Business-Days reindexed)

    # Tabs
    tabs = st.tabs(['Metriken', 'Performance', 'Sharpe/Korrelation', 'Monatsrenditen', 'Composite'])
    # Metriken
    with tabs[0]:
        df_stats = pd.DataFrame([compute_metrics(returns[t], cumulatives[t]) for t in tickers], index=tickers)
        st.dataframe(df_stats.style.format('{:.2%}'))
    # Performance
    with tabs[1]:
        plot_line(prices, 'Kursverlauf', 'Preis')
        norm = {t: cumulatives[t] / cumulatives[t].iloc[0] for t in tickers}
        plot_line(norm, 'Normierte Performance (Start=1)', 'Index')
    # Sharpe/Korrelation
    with tabs[2]:
        rolling_sh = pd.DataFrame({t: returns[t].rolling(126).apply(lambda x: (x.mean()*252 - RISK_FREE_RATE) / (x.std()*np.sqrt(252)) if x.std()>0 else np.nan) for t in tickers})
        plot_line(rolling_sh, 'Rollierender Sharpe (126 Tage)', 'Sharpe')
        corr = pd.DataFrame(returns).corr()
        fig, ax = plt.subplots(figsize=(6,4))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        ax.set_title('Korrelationsmatrix')
        st.pyplot(fig)
    # Monatsrenditen
    with tabs[3]:
        mret = pd.DataFrame({t: returns[t].resample('M').apply(lambda x: (1+x).prod()-1) for t in tickers})
        fig, ax = plt.subplots(figsize=(7, max(2, len(tickers)*0.4)))
        sns.heatmap(mret.T, annot=True, fmt='-.1%', cmap='RdYlGn', center=0, ax=ax)
        ax.set_title('Monatliche Renditen')
        st.pyplot(fig)
    # Composite Index
    with tabs[4]:
        if len(tickers) < 2:
            st.info('Mindestens zwei Assets benötigt für Composite Index.')
        else:
            dfR = pd.DataFrame(returns)
            cons = ({'type':'eq','fun':lambda w: w.sum()-1},)
            bnds = [(0,1)]*len(tickers)
            x0 = np.ones(len(tickers)) / len(tickers)
            sol = opt.minimize(lambda w: -((dfR.mean()*w).sum()*252 - RISK_FREE_RATE)/np.sqrt(w@dfR.cov()*252@w), x0, bounds=bnds, constraints=cons)
            w = sol.x if sol.success else x0
            st.write('Gewichte:', dict(zip(tickers, (w*100).round(1).astype(str) + '%')))
            comp_ret = returns.dot(w)
            comp_cum = (1+comp_ret).cumprod()
            plot_line({**prices, 'Composite': comp_cum * prices[tickers[0]].iloc[0]}, 'Kurs Composite', 'Preis')
            plot_line({**cumulatives, 'Composite': comp_cum}, 'Normierte Composite', 'Index')
            comp_stats = compute_metrics(comp_ret, comp_cum)
            st.json(comp_stats)

if __name__ == '__main__':
    main()
