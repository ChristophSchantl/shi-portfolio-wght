"""
Streamlit App zur Analyse und Vergleich von SHI Zertifikaten und anderen Finanzinstrumenten.

Enthält:
- Performance-Metriken
- Risikoanalyse
- Portfolio-Optimierung
- Vergleichsvisualisierungen
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import scipy.optimize as opt
from datetime import datetime, date
from typing import Dict, Tuple, List, Optional
import warnings
from functools import lru_cache

# ---- Konfiguration und Styling ----
warnings.filterwarnings('ignore', category=FutureWarning)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_theme(style='darkgrid', palette='muted')

st.set_page_config(
    page_title='SHI Zertifikate im Vergleich',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Konstanten
RISK_FREE_RATE = 0.02  # 2% risikofreier Zinssatz p.a.
BUSINESS_DAYS = 'B'
ANNUALIZATION_FACTOR = 252  # Handelstage pro Jahr
MIN_DATA_POINTS = 5  # Mindestanzahl Datenpunkte für Berechnungen

# ---- Hilfsfunktionen ----
def to_1d_series(series: pd.Series | pd.DataFrame) -> pd.Series:
    """Konvertiert DataFrame-Spalte oder Series in eine bereinigte 1D-Serie.
    
    Args:
        series: Input-Daten (DataFrame oder Series)
        
    Returns:
        Bereinigte pandas Series mit numerischen Werten
    """
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    return pd.to_numeric(series, errors='coerce').dropna()

def calculate_annualized_return(cumulative_returns: pd.Series) -> float:
    """Berechnet die annualisierte Rendite aus kumulierten Returns.
    
    Args:
        cumulative_returns: Kumulierte Returns-Serie
        
    Returns:
        Annualisierte Rendite als float (oder np.nan bei Fehler)
    """
    if len(cumulative_returns) < 2:
        return np.nan
    
    days = (cumulative_returns.index[-1] - cumulative_returns.index[0]).days
    if days <= 0:
        return np.nan
    
    total_return = cumulative_returns.iloc[-1] / cumulative_returns.iloc[0] - 1
    return (1 + total_return) ** (365 / days) - 1

def omega_ratio(returns: pd.Series, risk_free: float = RISK_FREE_RATE) -> float:
    """Berechnet das Omega-Verhältnis für eine gegebene Renditeserie.
    
    Args:
        returns: Tägliche Renditen
        risk_free: Risikofreier Zinssatz
        
    Returns:
        Omega Ratio als float
    """
    excess_returns = returns - risk_free / ANNUALIZATION_FACTOR
    gains = excess_returns[excess_returns > 0].sum()
    losses = -excess_returns[excess_returns <= 0].sum()
    
    return float(gains / losses) if losses > 0 else np.nan

def tail_ratio(returns: pd.Series) -> float:
    """Berechnet das Tail Ratio (95. Perzentil / abs(5. Perzentil)).
    
    Args:
        returns: Tägliche Renditen
        
    Returns:
        Tail Ratio als float
    """
    if len(returns) < MIN_DATA_POINTS:
        return np.nan
    
    try:
        top = np.percentile(returns, 95)
        bottom = abs(np.percentile(returns, 5))
        return float(top / bottom)
    except Exception:
        return np.nan

@st.cache_data(ttl=3600, show_spinner="Lade Marktdaten...")
def fetch_data(
    tickers: List[str], 
    start: date, 
    end: date
) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series], Dict[str, pd.Series]]:
    """Lädt historische Preisdaten von Yahoo Finance und berechnet Kennzahlen.
    
    Args:
        tickers: Liste von Ticker-Symbolen
        start: Startdatum
        end: Enddatum
        
    Returns:
        Tuple mit:
        - Preise (dict mit pd.Series)
        - Tägliche Renditen (dict mit pd.Series)
        - Kumulierte Performance (dict mit pd.Series)
    """
    bidx = pd.date_range(start, end, freq=BUSINESS_DAYS)
    prices, returns, cumulatives = {}, {}, {}
    
    for t in tickers:
        try:
            raw = yf.download(t, start=start, end=end, progress=False)['Close']
            if raw.empty:
                st.warning(f"Keine Daten für {t} verfügbar")
                continue
                
            # Daten bereinigen und ausrichten
            s = raw.reindex(bidx).ffill().dropna()
            if len(s) < MIN_DATA_POINTS:
                st.warning(f"Zu wenige Datenpunkte für {t}")
                continue
                
            r = s.pct_change().dropna()
            c = (1 + r).cumprod()
            
            prices[t], returns[t], cumulatives[t] = s, r, c
        except Exception as e:
            st.error(f"Fehler beim Laden von {t}: {str(e)}")
    
    return prices, returns, cumulatives

def compute_metrics(returns: pd.Series, cumulative: pd.Series) -> Dict[str, float]:
    """Berechnet umfassende Performance- und Risikokennzahlen.
    
    Args:
        returns: Tägliche Renditen
        cumulative: Kumulierte Performance
        
    Returns:
        Dictionary mit berechneten Metriken
    """
    stats: Dict[str, float] = {}
    r = to_1d_series(returns)
    c = to_1d_series(cumulative)
    
    if len(r) < MIN_DATA_POINTS or len(c) < MIN_DATA_POINTS:
        return stats
    
    # Grundlegende Performance-Metriken
    total_return = float(c.iloc[-1] / c.iloc[0] - 1)
    annual_return = calculate_annualized_return(c)
    annual_vol = float(r.std() * np.sqrt(ANNUALIZATION_FACTOR))
    
    stats.update({
        'Total Return': total_return,
        'Annual Return': annual_return,
        'Annual Volatility': annual_vol,
        'Sharpe Ratio': ((annual_return - RISK_FREE_RATE) / annual_vol 
                        if annual_vol > 0 else np.nan)
    })
    
    # Downside-Risiko-Metriken
    downside = r[r < 0]
    std_down = float(downside.std(ddof=0)) if len(downside) > 0 else np.nan
    stats['Sortino Ratio'] = ((r.mean() - RISK_FREE_RATE/ANNUALIZATION_FACTOR) / std_down * 
                            np.sqrt(ANNUALIZATION_FACTOR)) if std_down > 0 else np.nan
    
    # Drawdown-Analyse
    dd = c / c.cummax() - 1
    max_dd = float(dd.min())
    stats.update({
        'Max Drawdown': max_dd,
        'Calmar Ratio': (float(annual_return / abs(max_dd)) 
                        if max_dd < 0 and annual_return > 0 else np.nan)
    })
    
    # Risikomaße
    var95 = float(r.quantile(0.05))
    stats.update({
        'VaR (95%)': var95,
        'CVaR (95%)': float(r[r <= var95].mean()),
        'Omega Ratio': omega_ratio(r),
        'Tail Ratio': tail_ratio(r)
    })
    
    # Gewinn/Verlust-Statistiken
    win_rate = float((r > 0).mean())
    avg_win = float(r[r > 0].mean())
    avg_loss = float(r[r < 0].mean())
    
    stats.update({
        'Win Rate': win_rate,
        'Avg Win': avg_win,
        'Avg Loss': avg_loss,
        'Profit Factor': (-avg_win / avg_loss if avg_loss < 0 else np.nan),
        'Expectancy': win_rate * avg_win - (1-win_rate) * abs(avg_loss)
    })
    
    # Monatliche Performance
    monthly = r.resample('M').apply(lambda x: (1 + x).prod() - 1)
    stats['Positive Months'] = float((monthly > 0).mean())
    
    return stats

def create_line_plot(
    data: Dict[str, pd.Series], 
    title: str, 
    ylabel: str,
    log_scale: bool = False
) -> plt.Figure:
    """Erstellt einen Liniengraphen für mehrere Zeitreihen.
    
    Args:
        data: Dictionary mit pd.Series (key = Name)
        title: Plot-Titel
        ylabel: Y-Achsen-Beschriftung
        log_scale: Log-Skala aktivieren
        
    Returns:
        Matplotlib Figure Objekt
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for name, series in data.items():
        ax.plot(series.index, series.values, label=name, linewidth=1.2)
    
    ax.set_title(title, pad=15)
    ax.set_xlabel('Datum')
    ax.set_ylabel(ylabel)
    
    if log_scale:
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))
    
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    return fig

def optimize_portfolio(returns: pd.DataFrame) -> Optional[np.ndarray]:
    """Optimiert Portfolio-Gewichte für maximale Sharpe Ratio.
    
    Args:
        returns: DataFrame mit täglichen Renditen
        
    Returns:
        Optimierte Gewichte oder None bei Fehler
    """
    if len(returns.columns) < 2:
        return None
    
    dfR = returns.dropna()
    if len(dfR) < MIN_DATA_POINTS:
        return None
    
    # Constraints: Summe der Gewichte = 1, keine Leerverkäufe
    cons = ({'type': 'eq', 'fun': lambda w: w.sum() - 1},)
    bnds = [(0, 1) for _ in range(len(dfR.columns))]
    x0 = np.ones(len(dfR.columns)) / len(dfR.columns)
    
    def neg_sharpe(w):
        port_return = np.dot(dfR.mean(), w) * ANNUALIZATION_FACTOR
        port_vol = np.sqrt(w @ (dfR.cov() * ANNUALIZATION_FACTOR) @ w)
        return -(port_return - RISK_FREE_RATE) / port_vol if port_vol > 0 else 0
    
    try:
        sol = opt.minimize(
            neg_sharpe,
            x0,
            method='SLSQP',
            bounds=bnds,
            constraints=cons,
            options={'maxiter': 1000}
        )
        return sol.x if sol.success else None
    except Exception:
        return None

# ---- Streamlit App ----
def main() -> None:
    """Hauptfunktion der Streamlit App."""
    st.title('📊 SHI Zertifikate im Vergleich')
    st.caption('Performance-, Risiko- und Benchmarkanalyse')
    
    with st.sidebar:
        st.header('Konfiguration')
        start = st.date_input(
            'Startdatum', 
            datetime(2023, 1, 1),
            max_value=datetime.today()
        )
        end = st.date_input(
            'Enddatum', 
            datetime.today(),
            min_value=start,
            max_value=datetime.today()
        )
        
        default_tickers = 'AAPL, MSFT, ^GDAXI, BTC-USD'
        tickers_input = st.text_area(
            'Ticker-Symbole (kommagetrennt)', 
            default_tickers,
            help='Beispiele: Aktien (AAPL), Indizes (^GDAXI), Krypto (BTC-USD)'
        )
        tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
        
        st.markdown("""
        **Ticker-Beispiele:**
        - Aktien: AAPL, MSFT
        - Indizes: ^GDAXI, ^IXIC
        - ETFs: VOO, EUNK.DE
        - Krypto: BTC-USD, ETH-USD
        """)
    
    if not tickers:
        st.error('Bitte mindestens einen Ticker eingeben.')
        return
    
    # Daten laden
    with st.spinner('Lade Marktdaten...'):
        prices, returns, cumulatives = fetch_data(tickers, start, end)
    
    if not prices:
        st.error('Keine Daten für die ausgewählten Ticker verfügbar.')
        return
    
    # Tabs erstellen
    tab_titles = ['Metriken', 'Performance', 'Risikoanalyse', 'Monatsrenditen', 'Portfolio']
    tabs = st.tabs(tab_titles)
    
    # 1. Metriken-Tab
    with tabs[0]:
        st.subheader('Performance- und Risikokennzahlen')
        
        stats = []
        for t in prices.keys():
            try:
                stats.append(compute_metrics(returns[t], cumulatives[t]))
            except Exception as e:
                st.error(f"Fehler bei Berechnung für {t}: {str(e)}")
                continue
        
        if not stats:
            st.warning('Keine Metriken berechenbar')
        else:
            df_stats = pd.DataFrame(stats, index=prices.keys())
            
            # Formatierung für bessere Lesbarkeit
            format_dict = {col: '{:.2%}' for col in df_stats.columns 
                          if col not in ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 
                                       'Omega Ratio', 'Tail Ratio', 'Expectancy']}
            format_dict.update({
                col: '{:.2f}' for col in ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio',
                                         'Omega Ratio', 'Tail Ratio', 'Expectancy']
            })
            
            st.dataframe(
                df_stats.style.format(format_dict)
                .background_gradient(cmap='RdYlGn', axis=0, subset=[
                    'Annual Return', 'Sharpe Ratio', 'Sortino Ratio',
                    'Positive Months', 'Win Rate'
                ])
                .background_gradient(cmap='RdYlGn_r', axis=0, subset=[
                    'Annual Volatility', 'Max Drawdown', 'VaR (95%)'
                ]),
                use_container_width=True
            )
            
            # Download-Button für Daten
            csv = df_stats.to_csv().encode('utf-8')
            st.download_button(
                "Metriken als CSV exportieren",
                csv,
                "portfolio_metrics.csv",
                "text/csv",
                key='download-metrics'
            )
    
    # 2. Performance-Tab
    with tabs[1]:
        st.subheader('Preisentwicklung')
        fig = create_line_plot(prices, 'Kursverlauf', 'Preis in USD')
        st.pyplot(fig)
        
        st.subheader('Normierte Performance (Start = 1)')
        norm = {t: cumulatives[t] / cumulatives[t].iloc[0] for t in prices.keys()}
        fig = create_line_plot(norm, 'Normierte Performance', 'Multiplikator')
        st.pyplot(fig)
    
    # 3. Risikoanalyse-Tab
    with tabs[2]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader('Rollierende Sharpe Ratio (6 Monate)')
            rolling_window = 126  # ~6 Monate
            rolling_sharpe = pd.DataFrame({
                t: returns[t].rolling(rolling_window).apply(
                    lambda x: ((x.mean() * ANNUALIZATION_FACTOR - RISK_FREE_RATE) / 
                            (x.std() * np.sqrt(ANNUALIZATION_FACTOR)) 
                            if x.std() > 0 else np.nan
                ) for t in prices.keys()
            })
            fig = create_line_plot(
                {t: rolling_sharpe[t] for t in prices.keys()},
                'Rollierende Sharpe Ratio',
                'Sharpe Ratio'
            )
            st.pyplot(fig)
        
        with col2:
            st.subheader('Korrelation der Renditen')
            corr = pd.DataFrame({t: returns[t] for t in prices.keys()}).corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                corr, 
                annot=True, 
                cmap='coolwarm', 
                fmt='.2f', 
                vmin=-1, 
                vmax=1,
                ax=ax,
                mask=np.triu(np.ones_like(corr, dtype=bool))
            )
            ax.set_title('Korrelationsmatrix der täglichen Renditen')
            st.pyplot(fig)
        
        st.subheader('Drawdown-Analyse')
        fig, ax = plt.subplots(figsize=(10, 4))
        for t in prices.keys():
            dd = cumulatives[t] / cumulatives[t].cummax() - 1
            ax.plot(dd.index, dd.values, label=t)
        ax.set_title('Kumulative Drawdowns')
        ax.set_ylabel('Drawdown')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # 4. Monatsrenditen-Tab
    with tabs[3]:
        st.subheader('Monatliche Renditen')
        mret = pd.DataFrame({
            t: returns[t].resample('M').apply(lambda x: (1+x).prod()-1) 
            for t in prices.keys()
        })
        
        fig, ax = plt.subplots(figsize=(10, max(3, len(prices)*0.5)))
        sns.heatmap(
            mret.T * 100, 
            annot=True, 
            fmt='.1f', 
            cmap='RdYlGn', 
            center=0,
            ax=ax,
            cbar_kws={'label': 'Rendite in %'}
        )
        ax.set_title('Monatliche Renditen in %')
        ax.set_xlabel('Monat/Jahr')
        ax.set_ylabel('Ticker')
        st.pyplot(fig)
        
        st.subheader('Jahresperformance')
        yret = pd.DataFrame({
            t: returns[t].resample('Y').apply(lambda x: (1+x).prod()-1) 
            for t in prices.keys()
        })
        st.dataframe(
            yret.style.format('{:.1%}')
            .background_gradient(cmap='RdYlGn', axis=1)
        )
    
    # 5. Portfolio-Tab
    with tabs[4]:
        if len(prices) < 2:
            st.info('Mindestens zwei Assets benötigt für Portfolio-Analyse.')
        else:
            st.subheader('Portfolio-Optimierung')
            
            # Portfolio optimieren
            df_returns = pd.DataFrame({t: returns[t] for t in prices.keys()}).dropna()
            weights = optimize_portfolio(df_returns)
            
            if weights is None:
                st.warning('Optimierung fehlgeschlagen - Gleichgewichtung verwendet')
                weights = np.ones(len(prices)) / len(prices)
            
            # Gewichte anzeigen
            weight_df = pd.DataFrame({
                'Ticker': list(prices.keys()),
                'Gewichtung': weights
            }).sort_values('Gewichtung', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write('**Optimale Portfolio-Gewichtung:**')
                st.dataframe(
                    weight_df.style.format({'Gewichtung': '{:.1%}'}),
                    hide_index=True
                )
            
            with col2:
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.pie(
                    weight_df['Gewichtung'],
                    labels=weight_df['Ticker'],
                    autopct='%1.1f%%',
                    startangle=90
                )
                ax.set_title('Portfolio-Zusammensetzung')
                st.pyplot(fig)
            
            # Portfolio-Performance berechnen
            port_ret = df_returns.dot(weights)
            port_cum = (1 + port_ret).cumprod()
            
            st.subheader('Portfolio-Performance')
            fig = create_line_plot(
                {**{t: cumulatives[t] for t in prices.keys()}, 
                 'Optimiertes Portfolio': port_cum},
                'Vergleich mit Einzelassets',
                'Normierte Performance'
            )
            st.pyplot(fig)
            
            # Portfolio-Metriken anzeigen
            port_stats = compute_metrics(port_ret, port_cum)
            st.json({
                k: f'{v:.2%}' if isinstance(v, float) and k not in [
                    'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio',
                    'Omega Ratio', 'Tail Ratio', 'Expectancy'
                ] else v
                for k, v in port_stats.items()
            })

if __name__ == '__main__':
    main()
