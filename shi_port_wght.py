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
def to_1d_series(ret):
    if isinstance(ret, pd.DataFrame):
        ret = ret.iloc[:, 0]
    return pd.to_numeric(ret, errors='coerce').dropna()


def load_data_from_yahoo(ticker, start, end):
    price = yf.download(ticker, start=start, end=end + timedelta(days=1), progress=False)["Close"].dropna()
    returns = price.pct_change().dropna()
    cumulative = (1 + returns).cumprod()
    return price, returns, cumulative

# --- Risiko-Kennzahlen ---
def sortino_ratio(returns, risk_free=0.0, annualization=252):
    downside = returns[returns < risk_free]
    downside_std = downside.std(ddof=0)
    mean_ret = returns.mean()
    if downside_std == 0 or np.isnan(downside_std):
        return np.nan
    daily_sortino = (mean_ret - risk_free) / downside_std
    return daily_sortino * np.sqrt(annualization)

def omega_ratio(returns, risk_free=0.0):
    gain = (returns > risk_free).sum()
    loss = (returns <= risk_free).sum()
    if loss == 0:
        return np.nan
    return gain / loss

def tail_ratio(returns):
    try:
        return np.percentile(returns, 95) / abs(np.percentile(returns, 5))
    except Exception:
        return np.nan

def calculate_metrics(returns_dict, cumulative_dict):
    metrics = pd.DataFrame()
    for name in returns_dict:
        # Returns als Serie
        ret = to_1d_series(returns_dict[name])
        # Cumulative als Serie
        cum = cumulative_dict[name]
        if isinstance(cum, pd.DataFrame):
            cum = cum.iloc[:, 0]
        cum = pd.to_numeric(cum, errors='coerce').dropna()
        if ret.empty or cum.empty:
            continue
        days = (cum.index[-1] - cum.index[0]).days
        total_ret = cum.iloc[-1] / cum.iloc[0] - 1
        annual_ret = (1 + total_ret)**(365/days) - 1 if days > 0 else np.nan
        annual_vol = ret.std() * np.sqrt(252)
        sharpe = (annual_ret - RISK_FREE_RATE) / annual_vol if annual_vol > 0 else np.nan
        sortino = sortino_ratio(ret)
        # Drawdowns
        drawdowns = cum / cum.cummax() - 1
        mdd = drawdowns.min() if not drawdowns.empty else np.nan
        calmar = (annual_ret / abs(mdd)) if (pd.api.types.is_scalar(mdd) and mdd < 0) else np.nan
        var_95 = ret.quantile(0.05)
        cvar_95 = ret[ret <= var_95].mean()
        omega = omega_ratio(ret)
        tail = tail_ratio(ret)
        win_rate = (ret > 0).mean()
        avg_win = ret[ret > 0].mean()
        avg_loss = ret[ret < 0].mean()
        profit_factor = -avg_win / avg_loss if avg_loss < 0 else np.nan
        monthly_ret = ret.resample('M').apply(lambda x: (1 + x).prod() - 1)
        positive_months = (monthly_ret > 0).mean()
        metrics.loc[name] = [
            total_ret, annual_ret, annual_vol, sharpe, sortino,
            mdd, calmar, var_95, cvar_95, omega, tail,
            win_rate, avg_win, avg_loss, profit_factor,
            positive_months
        ]
    metrics.columns = [
        'Total Return','Annual Return','Annual Volatility','Sharpe Ratio','Sortino Ratio',
        'Max Drawdown','Calmar Ratio','VaR (95%)','CVaR (95%)','Omega Ratio','Tail Ratio',
        'Win Rate','Avg Win','Avg Loss','Profit Factor','Positive Months'
    ]
    return metrics

# --- Plotfunktionen für Performance & Drawdowns ---
def plot_overview_prices(price_dict):
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, price in price_dict.items():
        ax.plot(price.index, price, label=name, linewidth=0.8)
    ax.set_title("Schlusskurs-Übersicht aller Assets")
    ax.set_xlabel("Datum")
    ax.set_ylabel("Preis")
    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)

def plot_normalized_performance(cum_dict):
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, cum in cum_dict.items():
        ax.plot(cum.index, cum/cum.iloc[0], label=name, linewidth=0.8)
    ax.set_title("Normierte kumulative Performance (Start = 1)")
    ax.set_xlabel("Datum")
    ax.set_ylabel("Indexierte Entwicklung")
    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)

def plot_individual_charts(price_dict, cum_dict):
    for name in price_dict:
        price = price_dict[name]
        cum = cum_dict[name]
        drawdown = cum/cum.cummax() - 1
        fig, axes = plt.subplots(2,1, figsize=(6,4), sharex=True)
        axes[0].plot(price.index, price)
        axes[0].set_title(f"{name} Preisverlauf")
        axes[0].set_ylabel("Preis")
        axes[1].fill_between(drawdown.index, drawdown, 0, alpha=0.3)
        axes[1].set_title(f"{name} Drawdown")
        axes[1].set_ylabel("Drawdown")
        axes[1].set_xlabel("Datum")
        plt.tight_layout()
        st.pyplot(fig)

# --- Zusätzliche Analysefunktionen ---
def analyze_correlations(returns_dict):
    returns_clean = {name: to_1d_series(ret) for name, ret in returns_dict.items()}
    df = pd.DataFrame(returns_clean)
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(6,3))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0,
                linewidths=0.5, annot_kws={"size":5}, ax=ax)
    ax.set_title("Korrelationsmatrix der täglichen Renditen", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    return corr


def analyze_rolling_performance(returns_dict, window=126):
    df = pd.DataFrame({name: to_1d_series(ret) for name, ret in returns_dict.items()})
    rolling_sharpe = (df.rolling(window).mean()*252 - RISK_FREE_RATE) / (df.rolling(window).std()*np.sqrt(252))
    fig, ax = plt.subplots(figsize=(6,3))
    for col in rolling_sharpe:
        ax.plot(rolling_sharpe.index, rolling_sharpe[col], linewidth=0.8)
    ax.set_title(f"Rollierender Sharpe Ratio ({window}-Tage)", fontsize=8)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    st.pyplot(fig)
    return rolling_sharpe

# --- Streamlit App ---
def main():
    st.markdown('<h3 style="font-weight:400; margin-bottom:0.2rem;">📊 SHI Zertifikate im Vergleich</h3>', unsafe_allow_html=True)
    st.caption("Performance-, Risiko- und Benchmarkanalyse auf Monatsbasis")

    with st.sidebar:
        st.header("Datenquellen auswählen")
        start = st.date_input("Startdatum", value=datetime(2023,1,1))
        end   = st.date_input("Enddatum", value=datetime.today())
        tickers_input = st.text_area(
            "Yahoo Finance Ticker (kommagetrennt)",
            placeholder="z.B. AAPL, MSFT, GOOG"
        )
        tickers = [t.strip() for t in tickers_input.replace(';',',').split(',') if t.strip()]
        st.write("Ausgewählte Ticker:", tickers)

    price_dict, returns_dict, cum_dict = {}, {}, {}
    for ticker in tickers:
        try:
            price, ret, cum = load_data_from_yahoo(ticker, start, end)
            price_dict[ticker] = price
            returns_dict[ticker] = ret
            cum_dict[ticker] = cum
        except Exception as e:
            st.warning(f"Fehler beim Laden von {ticker}: {e}")

    if not price_dict:
        st.warning("Bitte mindestens einen Ticker eingeben.")
        return

    # Zeitachsen synchronisieren
    common_idx = sorted(set.intersection(*[set(r.index) for r in returns_dict.values()]))
    for k in returns_dict:
        returns_dict[k] = returns_dict[k].loc[common_idx]
        cum_dict[k]     = cum_dict[k].loc[common_idx]

    tabs = st.tabs([
        "🚦 Metriken", "📈 Performance & Drawdown",
        "📉 Sharpe & Korrelation", "📊 Monatsrenditen", "🔀 Composite Index"
    ])

    with tabs[0]:
        st.subheader("Erweiterte Risikokennzahlen")
        metrics = calculate_metrics(returns_dict, cum_dict)
        st.dataframe(metrics, use_container_width=True)

    with tabs[1]:
        st.subheader("Kumulative Performance & Drawdown")
        plot_overview_prices(price_dict)
        plot_normalized_performance(cum_dict)
        plot_individual_charts(price_dict, cum_dict)

    with tabs[2]:
        st.subheader("Rollierender Sharpe Ratio")
        analyze_rolling_performance(returns_dict)
        st.subheader("Korrelation der Tagesrenditen")
        analyze_correlations(returns_dict)

    with tabs[3]:
        st.subheader("Monatliche Renditen")
        monthly = pd.DataFrame({
            name: to_1d_series(ret).resample('M').apply(lambda x: (1+x).prod()-1)
            for name, ret in returns_dict.items()
        })
        if not monthly.empty:
            fig, ax = plt.subplots(figsize=(7, max(2.2, len(monthly.columns)*0.33)))
            sns.heatmap(
                monthly.T, annot=True, fmt='-.1%', cmap='RdYlGn', center=0,
                linewidths=0.5, ax=ax, annot_kws={"size":4}
            )
            ax.set_title("Monatliche Renditen", fontsize=8)
            ax.set_xticklabels([d.strftime('%Y-%m') for d in monthly.index], rotation=90, fontsize=4)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("Keine Monatsrenditen verfügbar.")

    with tabs[4]:
        st.subheader("🔀 Composite Index aus gewählten Assets")
        assets = list(returns_dict.keys())
        if len(assets)<2:
            st.info("Bitte mindestens zwei Assets laden.")
        else:
            dfR = pd.DataFrame(returns_dict)
            # Optimale Gewichtung
            def neg_sharpe(w):
                ret = (dfR.mean()*w).sum()*252
                vol = np.sqrt(w.T @ (dfR.cov()*252) @ w)
                return -((ret-RISK_FREE_RATE)/vol if vol>0 else 0)
            cons = ({'type':'eq','fun':lambda x: x.sum()-1})
            bnds=[(0,1)]*len(assets)
            x0=np.ones(len(assets))/len(assets)
            res=opt.minimize(neg_sharpe, x0, method='SLSQP', bounds=bnds, constraints=cons)
            weights = res.x if res.success else x0
            sliders=[]
            cols=st.columns(len(assets))
            rem=100
            for i,a in enumerate(assets[:-1]):
                val=st.session_state.get(f'w_{a}', int(weights[i]*100))
                s=cols[i].slider(a,0,rem,val,1,key=f'w_{a}')
                sliders.append(s)
                rem-=s
            sliders.append(rem)
            cols[-1].number_input(assets[-1],min_value=0,max_value=100,value=rem,disabled=True)
            w_arr=np.array(sliders)/100
            st.markdown(f"**Summe:** {sum(sliders)}%")
            # Composite
            comp_ret=(dfR*w_arr).sum(axis=1)
            comp_cum=(1+comp_ret).cumprod()
            plot_overview_prices({**price_dict, 'Composite':comp_cum * price_dict[assets[0]].iloc[0]})
            plot_normalized_performance({**cum_dict, 'Composite':comp_cum})
            st.subheader("Risikokennzahlen Composite")
            m_comp=calculate_metrics({**returns_dict, 'Composite':comp_ret}, {**cum_dict, 'Composite':comp_cum})
            st.dataframe(m_comp, use_container_width=True)

if __name__ == "__main__":
    main()
