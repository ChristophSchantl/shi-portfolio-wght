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
    df = yf.download(ticker, start=start, end=end + timedelta(days=1), progress=False)["Close"].dropna()
    prices = df.copy()
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
    return np.nan if losses == 0 else gains / losses

def tail_ratio(returns):
    try:
        return np.percentile(returns, 95) / abs(np.percentile(returns, 5))
    except Exception:
        return np.nan


def calculate_metrics(returns_dict, cum_dict):
    cols = [
        'Total Return','Annual Return','Annual Volatility','Sharpe Ratio','Sortino Ratio',
        'Max Drawdown','Calmar Ratio','VaR (95%)','CVaR (95%)','Omega Ratio','Tail Ratio',
        'Win Rate','Avg Win','Avg Loss','Profit Factor','Positive Months'
    ]
    metrics = pd.DataFrame(columns=cols)
    for name, ret in returns_dict.items():
        try:
            r = to_1d_series(ret)
            cum = cum_dict[name]
            if isinstance(cum, pd.DataFrame): cum = cum.iloc[:,0]
            cum = to_1d_series(cum)
            if r.empty or cum.empty:
                st.warning(f"{name}: keine Daten für Kennzahlen.")
                continue
            days = (cum.index[-1] - cum.index[0]).days
            total = cum.iloc[-1] / cum.iloc[0] - 1
            ann_ret = (1+total)**(365/days) -1 if days>0 else np.nan
            ann_vol = r.std()*np.sqrt(252)
            sharpe = (ann_ret - RISK_FREE_RATE)/ann_vol if ann_vol>0 else np.nan
            sortino = sortino_ratio(r)
            dd = cum/cum.cummax() -1
            mdd = dd.min()
            calmar = ann_ret/abs(mdd) if mdd<0 else np.nan
            var95 = r.quantile(0.05)
            cvar95 = r[r<=var95].mean()
            omega = omega_ratio(r)
            tail = tail_ratio(r)
            win = (r>0).mean()
            avg_win = r[r>0].mean()
            avg_loss = r[r<0].mean()
            pf = -avg_win/avg_loss if avg_loss<0 else np.nan
            m_ret = r.resample('M').apply(lambda x: (1+x).prod()-1)
            pos_months = (m_ret>0).mean()
            metrics.loc[name] = [
                total, ann_ret, ann_vol, sharpe, sortino,
                mdd, calmar, var95, cvar95, omega, tail,
                win, avg_win, avg_loss, pf, pos_months
            ]
        except Exception as e:
            st.warning(f"Kennzahlen {name} fehlgeschlagen: {e}")
            continue
    return metrics

# --- Plotfunktionen ---
def plot_overview_prices(prices):
    fig, ax = plt.subplots(figsize=(8,4))
    for name, price in prices.items():
        ax.plot(price.index, price, label=name, linewidth=0.8)
    ax.set(title="Preis-Übersicht aller Assets", xlabel="Datum", ylabel="Preis")
    ax.legend(fontsize=8)
    plt.tight_layout(); st.pyplot(fig)


def plot_normalized_performance(cums):
    fig, ax = plt.subplots(figsize=(8,4))
    for name, cum in cums.items():
        ax.plot(cum.index, cum/cum.iloc[0], label=name, linewidth=0.8)
    ax.set(title="Normierte Performance (Start=1)", xlabel="Datum", ylabel="Index")
    ax.legend(fontsize=8)
    plt.tight_layout(); st.pyplot(fig)


def plot_individual_charts(prices, cums):
    for name in prices:
        try:
            price = to_1d_series(prices[name])
            cum = cums[name]
            if isinstance(cum, pd.DataFrame): cum = cum.iloc[:,0]
            cum = to_1d_series(cum)
            dd = cum/cum.cummax() -1
            # align
            idx = price.index.intersection(dd.index)
            p = price.loc[idx]
            d = dd.loc[idx]
            if len(p)<2 or len(d)<2:
                st.warning(f"Einzelchart {name}: nicht genügend Daten.")
                continue
            x = idx
            y1 = p.values
            y2 = d.values.flatten()
            zero = np.zeros_like(y2)
            fig, ax = plt.subplots(2,1,figsize=(6,4),sharex=True)
            ax[0].plot(x,y1); ax[0].set(title=f"{name} Preis", ylabel="Preis")
            ax[1].fill_between(x,y2,zero,alpha=0.3); ax[1].plot(x,y2,linewidth=0.8)
            ax[1].set(title=f"{name} Drawdown", ylabel="Drawdown", xlabel="Datum")
            plt.tight_layout(); st.pyplot(fig)
        except Exception as e:
            st.warning(f"Einzelchart Fehler {name}: {e}")
            continue

# --- Korrelations- & Rolling-Sharpe ---
def analyze_correlations(returns):
    df = pd.DataFrame({n: to_1d_series(r) for n,r in returns.items()})
    corr = df.corr(); fig, ax = plt.subplots(figsize=(6,3))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, linewidths=0.5, annot_kws={'size':5}, ax=ax)
    ax.set_title("Korrelationsmatrix" ,fontsize=8); plt.tight_layout(); st.pyplot(fig)
    return corr


def analyze_rolling_performance(returns, window=126):
    df = pd.DataFrame({n: to_1d_series(r) for n,r in returns.items()})
    roll_mean = df.rolling(window).mean()*252
    roll_std = df.rolling(window).std()*np.sqrt(252)
    roll_sh = (roll_mean - RISK_FREE_RATE)/roll_std
    fig, ax = plt.subplots(figsize=(6,3))
    for col in roll_sh: ax.plot(roll_sh.index, roll_sh[col], linewidth=0.8)
    ax.set(title=f"Rollierender Sharpe ({window}-Tage)"); ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.tight_layout(); st.pyplot(fig)
    return roll_sh

# --- Streamlit App ---
def main():
    st.markdown('<h3 style="font-weight:400;">📊 SHI Zertifikate im Vergleich</h3>', unsafe_allow_html=True)
    st.caption("Performance-, Risiko- und Benchmarkanalyse")
    with st.sidebar:
        st.header("Einstellungen")
        start = st.date_input("Startdatum", datetime(2023,1,1))
        end = st.date_input("Enddatum", datetime.today())
        tickers = [t.strip() for t in st.text_area("Ticker (kommagetrennt)", placeholder="AAPL, MSFT").replace(';',',').split(',') if t.strip()]
    data = {}
    for t in tickers:
        try: data[t] = load_data_from_yahoo(t, start, end)
        except Exception as e: st.warning(f"{t} Ladefehler: {e}")
    if not data: st.warning("Keine Ticker"); return
    prices = {t: v[0] for t,v in data.items()}
    returns = {t: v[1] for t,v in data.items()}
    cums = {t: v[2] for t,v in data.items()}
    # syncronize indices
    common = sorted(set.intersection(*[set(r.index) for r in returns.values()]))
    for t in returns: returns[t] = returns[t].loc[common]; cums[t] = cums[t].loc[common]
    tabs = st.tabs(["Metriken","Performance","Sharpe/Korrelation","Monatsrenditen","Composite"])
    with tabs[0]: st.dataframe(calculate_metrics(returns, cums))
    with tabs[1]: plot_overview_prices(prices); plot_normalized_performance(cums); plot_individual_charts(prices,cums)
    with tabs[2]: analyze_rolling_performance(returns); analyze_correlations(returns)
    with tabs[3]:    
        mth = pd.DataFrame({t: to_1d_series(r).resample('M').apply(lambda x:(1+x).prod()-1) for t,r in returns.items()})
        if mth.empty: st.warning("keine Daten");
        else:
            fig, ax = plt.subplots(figsize=(7,4)); sns.heatmap(mth.T, annot=True, fmt='-.1%', cmap='RdYlGn', center=0, linewidths=0.5, annot_kws={'size':4}, ax=ax); ax.set_xticklabels([d.strftime('%Y-%m') for d in mth.index], rotation=90, fontsize=4); plt.tight_layout(); st.pyplot(fig)
    with tabs[4]:
        dfR = pd.DataFrame(returns)
        def neg_sh(w): ret=(dfR.mean()*w).sum()*252; vol=np.sqrt(w.T@(dfR.cov()*252)@w); return -(ret-RISK_FREE_RATE)/vol if vol>0 else np.nan
        cons=({'type':'eq','fun':lambda w:w.sum()-1},); bnds=[(0,1)]*len(dfR.columns); x0=np.repeat(1/len(dfR.columns),len(dfR.columns))
        sol=opt.minimize(neg_sh,x0,method='SLSQP',bounds=bnds,constraints=cons)
        w=sol.x if sol.success else x0
        cols=st.columns(len(w))
        rem=100; ws=[]
        for i,(asset,wi) in enumerate(zip(dfR.columns,w)):
            if i<len(w)-1: val=cols[i].slider(asset,0,rem,int(wi*100),1,key=f'w{i}'); ws.append(val); rem-=val
            else: cols[i].number_input(asset,min_value=0,max_value=100,value=rem,disabled=True); ws.append(rem)
        w_arr=np.array(ws)/100; st.write(f"Summe: {sum(ws)}%")
        cret=(dfR*w_arr).sum(axis=1); ccum=(1+cret).cumprod()
        plot_overview_prices({**prices,'Composite':ccum*prices[dfR.columns[0]].iloc[0]}); plot_normalized_performance({**cums,'Composite':ccum})
        st.dataframe(calculate_metrics({**returns,'Composite':cret},{**cums,'Composite':ccum}))

if __name__=="__main__": main()
