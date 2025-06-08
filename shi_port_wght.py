from st_aggrid import AgGrid, GridOptionsBuilder
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import warnings
import seaborn as sns
import scipy.optimize as opt
from datetime import datetime, timedelta

# ---- Styling & Optionen ----
warnings.simplefilter(action='ignore', category=FutureWarning)
sns.set_theme(style="darkgrid")
plt.style.use('seaborn-v0_8-darkgrid')
pd.set_option('display.float_format', '{:.2%}'.format)

st.set_page_config(
    page_title="SHI Zertifikate im Vergleich",
    page_icon="📊",
    layout="wide"
)

RISK_FREE_RATE = 0.02  # 2% p.a.

# --- Hilfsfunktionen ---
def to_1d_series(series):
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    return pd.to_numeric(series, errors='coerce').dropna()


def load_data_from_yahoo(ticker, start, end):
    prices = yf.download(ticker, start=start, end=end + timedelta(days=1), progress=False)["Close"].dropna()
    returns = prices.pct_change().dropna()
    cumulative = (1 + returns).cumprod()
    return prices, returns, cumulative

# --- Risiko-Kennzahlen ---
def sortino_ratio(returns, risk_free=0.0, annualization=252):
    downside = returns[returns < risk_free]
    std_down = downside.std(ddof=0)
    mean_ret = returns.mean()
    if std_down == 0 or np.isnan(std_down):
        return np.nan
    return (mean_ret - risk_free) / std_down * np.sqrt(annualization)

def omega_ratio(returns, risk_free=0.0):
    gains = (returns > risk_free).sum()
    losses = (returns <= risk_free).sum()
    return gains / losses if losses > 0 else np.nan

def tail_ratio(returns):
    try:
        return np.percentile(returns, 95) / abs(np.percentile(returns, 5))
    except Exception:
        return np.nan


def calculate_metrics(returns_dict, cumulative_dict):
    cols = [
        'Total Return','Annual Return','Annual Volatility','Sharpe Ratio','Sortino Ratio',
        'Max Drawdown','Calmar Ratio','VaR (95%)','CVaR (95%)','Omega Ratio','Tail Ratio',
        'Win Rate','Avg Win','Avg Loss','Profit Factor','Positive Months'
    ]
    metrics = pd.DataFrame(columns=cols)
    for name, ret in returns_dict.items():
        try:
            r = to_1d_series(ret)
            c = cumulative_dict[name]
            if isinstance(c, pd.DataFrame):
                c = c.iloc[:, 0]
            c = to_1d_series(c)
            if len(r) < 2 or len(c) < 2:
                continue
            days = (c.index[-1] - c.index[0]).days
            total = c.iloc[-1] / c.iloc[0] - 1
            annual_ret = (1 + total)**(365 / days) - 1 if days > 0 else np.nan
            annual_vol = r.std() * np.sqrt(252)
            sharpe = (annual_ret - RISK_FREE_RATE) / annual_vol if annual_vol > 0 else np.nan
            sortino = sortino_ratio(r)
            drawdowns = c / c.cummax() - 1
            mdd = drawdowns.min()
            calmar = annual_ret / abs(mdd) if mdd < 0 else np.nan
            var95 = r.quantile(0.05)
            cvar95 = r[r <= var95].mean()
            omega = omega_ratio(r)
            tail = tail_ratio(r)
            win_rate = (r > 0).mean()
            avg_win = r[r > 0].mean()
            avg_loss = r[r < 0].mean()
            profit_factor = -avg_win / avg_loss if avg_loss < 0 else np.nan
            monthly = r.resample('M').apply(lambda x: (1 + x).prod() - 1)
            pos_months = (monthly > 0).mean()
            metrics.loc[name] = [
                total, annual_ret, annual_vol, sharpe, sortino,
                mdd, calmar, var95, cvar95, omega, tail,
                win_rate, avg_win, avg_loss, profit_factor, pos_months
            ]
        except Exception:
            continue
    return metrics

# --- Plotfunktionen ---
def plot_overview_prices(prices_dict):
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, price in prices_dict.items():
        ax.plot(price.index, price, label=name, linewidth=0.8)
    ax.set_title("Preis-Übersicht aller Assets")
    ax.set_xlabel("Datum")
    ax.set_ylabel("Preis")
    ax.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)


def plot_normalized_performance(cum_dict):
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, cum in cum_dict.items():
        ax.plot(cum.index, cum / cum.iloc[0], label=name, linewidth=0.8)
    ax.set_title("Normierte Performance (Start=1)")
    ax.set_xlabel("Datum")
    ax.set_ylabel("Index")
    ax.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)


def plot_individual_charts(prices_dict, cum_dict):
    for name in prices_dict:
        price = to_1d_series(prices_dict[name])
        cum = cum_dict[name]
        if isinstance(cum, pd.DataFrame):
            cum = cum.iloc[:, 0]
        cum = to_1d_series(cum)
        if len(price) < 2 or len(cum) < 2:
            continue
        drawdown = cum / cum.cummax() - 1
        idx = price.index.intersection(drawdown.index)
        p = price.loc[idx]
        d = drawdown.loc[idx]
        if len(p) < 2 or len(d) < 2:
            continue
        x = p.index
        y_price = p.values
        y_dd = d.values.flatten()
        zero = np.zeros_like(y_dd)
        fig, axes = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
        axes[0].plot(x, y_price)
        axes[0].set_title(f"{name} Preisverlauf")
        axes[0].set_ylabel("Preis")
        axes[1].fill_between(x, y_dd, zero, alpha=0.3)
        axes[1].plot(x, y_dd, linewidth=0.8)
        axes[1].set_title(f"{name} Drawdown")
        axes[1].set_ylabel("Drawdown")
        axes[1].set_xlabel("Datum")
        plt.tight_layout()
        st.pyplot(fig)

# --- Korrelations- & Rolling-Sharpe ---
def analyze_correlations(returns_dict):
    df = pd.DataFrame({name: to_1d_series(ret) for name, ret in returns_dict.items()})
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, linewidths=0.5, annot_kws={'size':5}, ax=ax)
    ax.set_title("Korrelationsmatrix")
    plt.tight_layout()
    st.pyplot(fig)


def analyze_rolling_performance(returns_dict, window=126):
    df = pd.DataFrame({name: to_1d_series(ret) for name, ret in returns_dict.items()})
    rolling = (df.rolling(window).mean() * 252 - RISK_FREE_RATE) / (df.rolling(window).std() * np.sqrt(252))
    fig, ax = plt.subplots(figsize=(6, 3))
    for col in rolling:
        ax.plot(rolling.index, rolling[col], linewidth=0.8)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_title(f"Rollierender Sharpe Ratio ({window}-Tage)")
    plt.tight_layout()
    st.pyplot(fig)
    return rolling

# --- Streamlit App ---
def main():
    st.markdown('<h3 style="font-weight:400;">📊 SHI Zertifikate im Vergleich</h3>', unsafe_allow_html=True)
    st.caption("Performance-, Risiko- und Benchmarkanalyse")

    with st.sidebar:
        st.header("Einstellungen")
        start = st.date_input("Startdatum", datetime(2023,1,1))
        end = st.date_input("Enddatum", datetime.today())
        tickers_input = st.text_area("Ticker (kommagetrennt)", placeholder="AAPL, MSFT")
        tickers = [t.strip() for t in tickers_input.replace(';',',').split(',') if t.strip()]

    if not tickers:
        st.warning("Bitte mindestens einen Ticker eingeben.")
        return

    data = {}
    for t in tickers:
        try:
            data[t] = load_data_from_yahoo(t, start, end)
        except Exception:
            st.warning(f"{t}: Daten konnten nicht geladen werden.")

    if not data:
        st.warning("Keine gültigen Daten vorhanden.")
        return

    prices = {t: vals[0] for t, vals in data.items()}
    returns = {t: vals[1] for t, vals in data.items()}
    cums = {t: vals[2] for t, vals in data.items()}

    idx_common = sorted(set.intersection(*[set(ret.index) for ret in returns.values()]))
    returns = {t: returns[t].loc[idx_common] for t in returns}
    cums = {t: cums[t].loc[idx_common] for t in cums}

    tabs = st.tabs(["Metriken","Performance","Sharpe/Korrelation","Monatsrenditen","Composite"])

    # Tab 0: Metriken
    with tabs[0]:
        st.subheader("Erweiterte Risikokennzahlen")
        df_metrics = calculate_metrics(returns, cums)
        st.dataframe(df_metrics, use_container_width=True)

    # Tab 1: Performance & Drawdown
    with tabs[1]:
        st.subheader("Kumulative Performance & Drawdown")
        plot_overview_prices(prices)
        plot_normalized_performance(cums)
        plot_individual_charts(prices, cums)

    # Tab 2: Rollierender Sharpe & Korrelation
    with tabs[2]:
        st.subheader("Rollierender Sharpe Ratio (126 Tage)")
        analyze_rolling_performance(returns)
        st.subheader("Korrelationsmatrix der Tagesrenditen")
        analyze_correlations(returns)

    # Tab 3: Monatsrenditen
    with tabs[3]:
        st.subheader("Monatliche Renditen")
        mth = pd.DataFrame({
            t: to_1d_series(ret).resample('M').apply(lambda x: (1+x).prod()-1)
            for t, ret in returns.items()
        })
        if mth.empty:
            st.warning("Keine Monatsrenditen verfügbar.")
        else:
            fig, ax = plt.subplots(figsize=(max(7, len(mth.columns)*0.4), 4))
            sns.heatmap(
                mth.T,
                annot=True,
                fmt='-.1%',
                cmap='RdYlGn',
                center=0,
                linewidths=0.5,
                annot_kws={'size':4},
                ax=ax
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=4)
            plt.tight_layout()
            st.pyplot(fig)

    # Tab 4: Composite Index
    with tabs[4]:
        st.subheader("🔀 Composite Index aus gewählten Assets")
        # Filter für gültige Returns
        valid_returns = {t: r for t, r in returns.items() if isinstance(r, pd.Series) and len(r) > 1}
        if len(valid_returns) < 2:
            st.info("Bitte mindestens zwei Assets für den Composite Index laden.")
        else:
            dfR = pd.DataFrame(valid_returns)
            def neg_sh(w):
                ret = (dfR.mean() * w).sum() * 252
                vol = np.sqrt(w.T @ (dfR.cov()*252) @ w)
                return -(ret - RISK_FREE_RATE) / vol if vol>0 else np.nan

            cons = ({'type':'eq','fun':lambda w: w.sum()-1},)
            bnds = [(0,1)]*dfR.shape[1]
            x0 = np.repeat(1/dfR.shape[1], dfR.shape[1])
            sol = opt.minimize(neg_sh, x0, method='SLSQP', bounds=bnds, constraints=cons)
            weights = sol.x if sol.success else x0

            cols = st.columns(dfR.shape[1])
            rem = 100
            ws = []
            for i, asset in enumerate(dfR.columns):
                if i < dfR.shape[1] - 1:
                    val = cols[i].slider(asset, 0, rem, int(weights[i]*100), 1, key=f'w{i}')
                    ws.append(val)
                    rem -= val
                else:
                    cols[i].number_input(asset, min_value=0, max_value=100, value=rem, disabled=True)
                    ws.append(rem)

            w_arr = np.array(ws)/100
            st.markdown(f"**Summe der Gewichte:** {sum(ws)}%")
            cret = (dfR * w_arr).sum(axis=1)
            ccum = (1+cret).cumprod()

            plot_overview_prices({**prices, 'Composite': ccum * prices[list(prices.keys())[0]].iloc[0]})
            plot_normalized_performance({**cums, 'Composite': ccum})
            df_comp = calculate_metrics({**returns, 'Composite': cret}, {**cums, 'Composite': ccum})
            st.dataframe(df_comp, use_container_width=True)

if __name__ == "__main__":
    main()
