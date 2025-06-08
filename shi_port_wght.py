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

# Daten-Loader

def load_returns_from_csv(file):
    df = pd.read_csv(file, index_col=0, parse_dates=True)
    close = pd.to_numeric(df['Close'], errors='coerce').ffill().dropna()
    returns = close.pct_change().dropna()
    cumulative = (1 + returns).cumprod()
    return returns, cumulative


def load_returns_from_yahoo(ticker, start, end):
    df = yf.download(ticker, start=start, end=end + timedelta(days=1), progress=False)['Close'].dropna()
    returns = df.pct_change().dropna()
    cumulative = (1 + returns).cumprod()
    return returns, cumulative

# Risikokennzahlen

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
    for name, ret in returns_dict.items():
        cum = cumulative_dict.get(name)
        if isinstance(ret, pd.DataFrame):
            ret = ret.iloc[:, 0]
        ret = pd.to_numeric(ret, errors='coerce').dropna()
        if ret.empty or cum is None or cum.empty:
            continue
        days = (cum.index[-1] - cum.index[0]).days
        total_ret = cum.iloc[-1] / cum.iloc[0] - 1
        annual_ret = (1 + total_ret) ** (365 / days) - 1 if days > 0 else np.nan
        annual_vol = ret.std() * np.sqrt(252)
        sharpe = (annual_ret - RISK_FREE_RATE) / annual_vol if annual_vol > 0 else np.nan
        sortino = sortino_ratio(ret)
        drawdowns = cum / cum.cummax() - 1
        mdd = drawdowns.min() if not drawdowns.empty else np.nan
        calmar = annual_ret / abs(mdd) if mdd < 0 else np.nan
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

        metrics.loc[name, 'Total Return'] = total_ret
        metrics.loc[name, 'Annual Return'] = annual_ret
        metrics.loc[name, 'Annual Volatility'] = annual_vol
        metrics.loc[name, 'Sharpe Ratio'] = sharpe
        metrics.loc[name, 'Sortino Ratio'] = sortino
        metrics.loc[name, 'Max Drawdown'] = mdd
        metrics.loc[name, 'Calmar Ratio'] = calmar
        metrics.loc[name, 'VaR (95%)'] = var_95
        metrics.loc[name, 'CVaR (95%)'] = cvar_95
        metrics.loc[name, 'Omega Ratio'] = omega
        metrics.loc[name, 'Tail Ratio'] = tail
        metrics.loc[name, 'Win Rate'] = win_rate
        metrics.loc[name, 'Avg Win'] = avg_win
        metrics.loc[name, 'Avg Loss'] = avg_loss
        metrics.loc[name, 'Profit Factor'] = profit_factor
        metrics.loc[name, 'Positive Months'] = positive_months
    return metrics

# Plot- und Analysefunktionen

def plot_performance(cumulative_dict):
    fig, ax = plt.subplots(figsize=(6, 3))
    for name, cum in cumulative_dict.items():
        ax.plot(cum.index, cum / cum.iloc[0], label=name, linewidth=0.3)
    ax.set_title("Kumulative Performance (Start = 1.0)", fontsize=8)
    ax.set_xlabel("Datum", fontsize=5)
    ax.set_ylabel("Indexierte Entwicklung", fontsize=5)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, fontsize=5)
    ax.tick_params(labelsize=5)
    plt.tight_layout()
    st.pyplot(fig)

    # Drawdown
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    for name, cum in cumulative_dict.items():
        drawdown = cum / cum.cummax() - 1
        ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3)
        ax2.plot(drawdown.index, drawdown.values, linewidth=0.25)
    ax2.set_title("Drawdown-Verlauf", fontsize=8)
    ax2.set_ylabel("Drawdown", fontsize=5)
    ax2.legend(cumulative_dict.keys(), loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, fontsize=5)
    ax2.tick_params(labelsize=5)
    plt.tight_layout()
    st.pyplot(fig2)


def analyze_correlations(returns_dict):
    df = pd.DataFrame({name: to_1d_series(ret) for name, ret in returns_dict.items()})
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                linewidths=0.5, annot_kws={'size':5}, ax=ax)
    ax.set_title("Korrelationsmatrix der täglichen Renditen", fontsize=6)
    ax.tick_params(labelsize=5)
    plt.tight_layout()
    st.pyplot(fig)
    return corr


def analyze_rolling_performance(returns_dict, window=126):
    df = pd.DataFrame({name: to_1d_series(ret) for name, ret in returns_dict.items()})
    rolling_sharpe = (df.rolling(window).mean() * 252 - RISK_FREE_RATE) /
                     (df.rolling(window).std() * np.sqrt(252))
    fig, ax = plt.subplots(figsize=(6, 2.5))
    for col in rolling_sharpe:
        ax.plot(rolling_sharpe.index, rolling_sharpe[col], linewidth=0.3)
    ax.set_title(f"Rollierender Sharpe Ratio ({window}-Tage)", fontsize=8)
    ax.axhline(0, linestyle='--', linewidth=0.25)
    ax.tick_params(labelsize=5)
    plt.tight_layout()
    st.pyplot(fig)
    return rolling_sharpe

if __name__ == '__main__':
    def main():
        st.markdown('<h3 style="font-weight:400; margin-bottom:0.2rem;">📊 SHI Zertifikate im Vergleich</h3>', unsafe_allow_html=True)
        st.caption("Performance-, Risiko- und Benchmarkanalyse auf Monatsbasis")

        # Sidebar
        with st.sidebar:
            st.header("Datenquellen auswählen")
            start = st.date_input("Startdatum", value=datetime(2023,1,1))
            end = st.date_input("Enddatum", value=datetime.today())
            default_tickers = "QBTS, VOW3.DE, INTC, BIDU, EL, TCEHY, LUMN, PNGAY, PDD, BABA"
            tickers = [t.strip() for line in default_tickers.splitlines()
                       for t in line.replace(";", ",").split(",") if t.strip()]
            tickers_input = st.text_area("Ticker", value=default_tickers)
            tickers = [t.strip() for line in tickers_input.splitlines()
                       for t in line.replace(";", ",").split(",") if t.strip()]
            st.write("Verarbeitete Ticker:", tickers)

        # Daten laden
        returns_dict, cumulative_dict = {}, {}
        for ticker in tickers:
            try:
                info = yf.Ticker(ticker).info
                name = info.get("shortName") or info.get("longName") or ticker
            except:
                name = ticker
            try:
                ret, cum = load_returns_from_yahoo(ticker, start, end)
                returns_dict[name] = ret
                cumulative_dict[name] = cum
            except Exception as e:
                st.warning(f"Fehler beim Laden von {ticker}: {e}")

        # Zeitachsen synchronisieren
        if returns_dict:
            common = sorted(set.intersection(*(set(r.index) for r in returns_dict.values())))
            for name in returns_dict:
                returns_dict[name] = returns_dict[name].loc[common]
                cumulative_dict[name] = cumulative_dict[name].loc[common]

        # Tabs
        tabs = st.tabs(["🚦 Metriken", "📈 Performance & Drawdown",
                       "📉 Sharpe & Korrelation", "📊 Monatsrenditen",
                       "🔀 Composite Index"])

        # Tab 0: Metriken
        with tabs[0]:
            st.subheader("Erweiterte Risikokennzahlen")
            if not returns_dict:
                st.warning("Bitte Datenquelle(n) hochladen oder Ticker eingeben.")
            else:
                metrics = calculate_metrics(returns_dict, cumulative_dict)
                # Formatierung...
                st.dataframe(metrics, use_container_width=True)

        # Tab 1: Performance & Drawdown
        with tabs[1]:
            st.subheader("Kumulative Performance & Drawdown")
            if returns_dict:
                plot_performance(cumulative_dict)
            else:
                st.warning("Bitte Datenquelle(n) hochladen oder Ticker eingeben.")

        # Tab 2: Rolling Sharpe & Korrelation
        with tabs[2]:
            st.subheader("Rollierender Sharpe Ratio")
            if returns_dict:
                analyze_rolling_performance(returns_dict)
            st.subheader("Korrelation der Tagesrenditen")
            if returns_dict:
                analyze_correlations(returns_dict)

        # Tab 3: Monatsrenditen
        with tabs[3]:
            st.subheader("Monatliche Renditen")
            if returns_dict:
                monthly_df = pd.DataFrame({n: to_1d_series(r).resample('M').apply(lambda x: (1+x).prod()-1)
                                           for n, r in returns_dict.items()})
                if not monthly_df.empty:
                    fig, ax = plt.subplots(figsize=(7, len(monthly_df.columns)*0.3))
                    sns.heatmap(monthly_df.T, annot=True, fmt='-.1%', cmap='RdYlGn', center=0,
                                annot_kws={'size':4}, cbar_kws={'shrink':0.8}, ax=ax)
                    ax.set_xticklabels([d.strftime('%Y-%m') for d in monthly_df.index],
                                       rotation=90, fontsize=4)
                    ax.set_yticklabels(ax.get_yticklabels(), fontsize=4)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.warning("Keine Monatsrenditen für diesen Zeitraum.")
            else:
                st.warning("Keine Daten vorhanden.")

        # Tab 4: Composite Index
        with tabs[4]:
            st.subheader("🔀 Composite Index aus gewählten Assets")
            assets = list(returns_dict.keys())
            n = len(assets)
            if n < 2:
                st.info("Bitte mindestens zwei Assets laden, um einen eigenen Index zu bauen.")
            else:
                returns_df = pd.DataFrame({k: to_1d_series(v) for k, v in returns_dict.items()}).dropna()
                # Optimale Sharpe-Gewichte
                def neg_sharpe(w):
                    pr = np.dot(returns_df.mean(), w) * 252
                    pv = np.sqrt(w.T @ (returns_df.cov()*252) @ w)
                    return -((pr - RISK_FREE_RATE)/pv) if pv>0 else np.inf
                cons = ({'type':'eq','fun': lambda x: x.sum()-1})
                bnds = tuple((0,1) for _ in range(n))
                x0 = np.ones(n)/n
                res = opt.minimize(neg_sharpe, x0, method='SLSQP', bounds=bnds, constraints=cons)
                opt_w = (res.x if res.success else x0) * 100

                # Slider-Logik
                cols = st.columns(n)
                rest = 100
                chosen = []
                for i in range(n-1):
                    key = f"w_{i}"
                    min_v, max_v = 0, rest
                    default = int(min(max(opt_w[i], min_v), max_v))
                    step = 1 if max_v-min_v>=1 else None
                    val = cols[i].slider(assets[i], min_value=min_v, max_value=max_v,
                                         value=default, step=step, key=key)
                    chosen.append(val)
                    rest -= val
                chosen.append(rest)
                cols[-1].markdown(f"**{assets[-1]}:** {rest}% (automatisch)")

                total = sum(chosen)
                if total != 100:
                    st.error(f"⚠️ Die Summe der Gewichte muss 100% sein – aktuell: {total}%!")
                    st.stop()

                # Index berechnen
                w_arr = np.array(chosen)/100
                custom_ret = (returns_df * w_arr).sum(axis=1)
                custom_cum = (1+custom_ret).cumprod()
                compare_cum = {**cumulative_dict, "Composite Index": custom_cum}
                compare_ret = {**returns_dict, "Composite Index": custom_ret}

                st.markdown("**Kumulative Performance (Composite Index vs. Assets):**")
                plot_performance(compare_cum)
                st.markdown("**Risikokennzahlen (Composite Index vs. Assets):**")
                metrics = calculate_metrics(compare_ret, compare_cum)
                st.dataframe(metrics, use_container_width=True)

    main()
