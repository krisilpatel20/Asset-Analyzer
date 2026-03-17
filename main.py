import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

app = FastAPI(
    title="OpenBB Single Asset Lite",
    description="Stable single-ticker custom backend for OpenBB Workspace",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE = Path(__file__).parent.resolve()


def _safe_ticker(ticker: str) -> str:
    return (ticker or "MSFT").strip().upper()


def _safe_benchmark(ticker: str) -> str:
    return (ticker or "SPY").strip().upper()


def _error_markdown(title: str, detail: str) -> str:
    return f"## {title}\n\n{detail}\n\nTry a large-cap U.S. ticker like `MSFT`, `AAPL`, `NVDA`, or `SPY`."


def _empty_fig(title: str, detail: str) -> Dict:
    fig = go.Figure()
    fig.add_annotation(
        text=detail,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 16},
    )
    fig.update_layout(title=title, xaxis={"visible": False}, yaxis={"visible": False})
    return json.loads(fig.to_json())


def _empty_table(message: str) -> List[Dict]:
    return [{"status": "unavailable", "message": message}]


@lru_cache(maxsize=128)
def get_history(ticker: str, start_date: str) -> pd.DataFrame:
    t = _safe_ticker(ticker)
    hist = yf.Ticker(t).history(start=start_date, auto_adjust=True)
    if hist.empty:
        raise ValueError(f"No price history returned for {t}.")
    hist = hist.reset_index()
    if "Date" not in hist.columns:
        raise ValueError(f"Date column missing for {t}.")
    hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)
    return hist


@lru_cache(maxsize=128)
def get_info(ticker: str) -> Dict:
    t = _safe_ticker(ticker)
    info = yf.Ticker(t).info or {}
    return info


def get_pair_history(ticker: str, benchmark: str, start_date: str) -> pd.DataFrame:
    left = get_history(ticker, start_date)[["Date", "Close"]].rename(columns={"Close": _safe_ticker(ticker)})
    right = get_history(benchmark, start_date)[["Date", "Close"]].rename(columns={"Close": _safe_benchmark(benchmark)})
    df = pd.merge(left, right, on="Date", how="inner").sort_values("Date").dropna()
    if df.empty:
        raise ValueError("No overlapping price history.")
    return df


def compute_returns(df: pd.DataFrame, ticker: str, benchmark: str) -> pd.DataFrame:
    t = _safe_ticker(ticker)
    b = _safe_benchmark(benchmark)
    out = df.copy()
    out[f"{t}_ret"] = out[t].pct_change()
    out[f"{b}_ret"] = out[b].pct_change()
    return out.dropna().reset_index(drop=True)


def annualized_return(series: pd.Series) -> float:
    if series.empty:
        return np.nan
    total = (1 + series).prod()
    years = len(series) / 252
    if years <= 0 or total <= 0:
        return np.nan
    return total ** (1 / years) - 1


def annualized_vol(series: pd.Series) -> float:
    return float(series.std() * np.sqrt(252)) if not series.empty else np.nan


def max_drawdown(price: pd.Series) -> float:
    roll_max = price.cummax()
    dd = price / roll_max - 1
    return float(dd.min()) if not dd.empty else np.nan


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series) -> pd.DataFrame:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    line = ema12 - ema26
    signal = line.ewm(span=9, adjust=False).mean()
    hist = line - signal
    return pd.DataFrame({"macd": line, "signal": signal, "hist": hist})


@app.get("/")
def root() -> Dict:
    return {"name": "OpenBB Single Asset Lite", "status": "ok"}


@app.get("/widgets.json")
def widgets() -> JSONResponse:
    return JSONResponse(content=json.load((BASE / "widgets.json").open()))


@app.get("/apps.json")
def apps() -> JSONResponse:
    return JSONResponse(content=json.load((BASE / "apps.json").open()))


@app.get("/summary")
def summary(ticker: str = "MSFT", benchmark: str = "SPY", start_date: str = "2023-01-01"):
    try:
        info = get_info(ticker)
        pair = get_pair_history(ticker, benchmark, start_date)
        returns = compute_returns(pair, ticker, benchmark)
        t = _safe_ticker(ticker)
        tr = returns[f"{t}_ret"]
        price_now = pair[t].iloc[-1]
        first_price = pair[t].iloc[0]
        total_return = price_now / first_price - 1
        md = [
            f"## {_safe_ticker(ticker)} summary",
            "",
            f"- **Company:** {info.get('shortName', _safe_ticker(ticker))}",
            f"- **Sector:** {info.get('sector', 'N/A')}",
            f"- **Industry:** {info.get('industry', 'N/A')}",
            f"- **Current price:** {price_now:,.2f}",
            f"- **Total return since {start_date}:** {total_return:.2%}",
            f"- **Annualized return:** {annualized_return(tr):.2%}",
            f"- **Annualized volatility:** {annualized_vol(tr):.2%}",
            f"- **Max drawdown:** {max_drawdown(pair[t]):.2%}",
            f"- **Benchmark:** {_safe_benchmark(benchmark)}",
        ]
        return PlainTextResponse("\n".join(md))
    except Exception as e:
        return PlainTextResponse(_error_markdown("Summary unavailable", str(e)))


@app.get("/price_chart")
def price_chart(ticker: str = "MSFT", start_date: str = "2023-01-01"):
    try:
        hist = get_history(ticker, start_date)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["Date"], y=hist["Close"], mode="lines", name=_safe_ticker(ticker)))
        fig.update_layout(title=f"{_safe_ticker(ticker)} price", xaxis_title="Date", yaxis_title="Price")
        return json.loads(fig.to_json())
    except Exception as e:
        return _empty_fig("Price chart unavailable", str(e))


@app.get("/growth_chart")
def growth_chart(ticker: str = "MSFT", benchmark: str = "SPY", start_date: str = "2023-01-01"):
    try:
        pair = get_pair_history(ticker, benchmark, start_date)
        t, b = _safe_ticker(ticker), _safe_benchmark(benchmark)
        pair[f"{t}_growth"] = pair[t] / pair[t].iloc[0]
        pair[f"{b}_growth"] = pair[b] / pair[b].iloc[0]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pair["Date"], y=pair[f"{t}_growth"], mode="lines", name=t))
        fig.add_trace(go.Scatter(x=pair["Date"], y=pair[f"{b}_growth"], mode="lines", name=b))
        fig.update_layout(title="Growth of $1", xaxis_title="Date", yaxis_title="Growth")
        return json.loads(fig.to_json())
    except Exception as e:
        return _empty_fig("Growth chart unavailable", str(e))


@app.get("/drawdown_chart")
def drawdown_chart(ticker: str = "MSFT", start_date: str = "2023-01-01"):
    try:
        hist = get_history(ticker, start_date)
        dd = hist["Close"] / hist["Close"].cummax() - 1
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["Date"], y=dd, fill="tozeroy", mode="lines", name="Drawdown"))
        fig.update_layout(title=f"{_safe_ticker(ticker)} drawdown", xaxis_title="Date", yaxis_title="Drawdown")
        return json.loads(fig.to_json())
    except Exception as e:
        return _empty_fig("Drawdown unavailable", str(e))


@app.get("/rolling_vol_chart")
def rolling_vol_chart(ticker: str = "MSFT", start_date: str = "2023-01-01", window: int = 63):
    try:
        hist = get_history(ticker, start_date)
        ret = hist["Close"].pct_change()
        vol = ret.rolling(window).std() * np.sqrt(252)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["Date"], y=vol, mode="lines", name="Rolling vol"))
        fig.update_layout(title=f"{window}d rolling volatility", xaxis_title="Date", yaxis_title="Volatility")
        return json.loads(fig.to_json())
    except Exception as e:
        return _empty_fig("Rolling volatility unavailable", str(e))


@app.get("/rolling_beta_chart")
def rolling_beta_chart(ticker: str = "MSFT", benchmark: str = "SPY", start_date: str = "2023-01-01", window: int = 63):
    try:
        pair = compute_returns(get_pair_history(ticker, benchmark, start_date), ticker, benchmark)
        t, b = _safe_ticker(ticker), _safe_benchmark(benchmark)
        cov = pair[f"{t}_ret"].rolling(window).cov(pair[f"{b}_ret"])
        var = pair[f"{b}_ret"].rolling(window).var()
        beta = cov / var.replace(0, np.nan)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pair["Date"], y=beta, mode="lines", name="Rolling beta"))
        fig.update_layout(title=f"{window}d rolling beta vs {b}", xaxis_title="Date", yaxis_title="Beta")
        return json.loads(fig.to_json())
    except Exception as e:
        return _empty_fig("Rolling beta unavailable", str(e))


@app.get("/rsi_chart")
def rsi_chart(ticker: str = "MSFT", start_date: str = "2023-01-01"):
    try:
        hist = get_history(ticker, start_date)
        rs = rsi(hist["Close"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["Date"], y=rs, mode="lines", name="RSI"))
        fig.add_hline(y=70)
        fig.add_hline(y=30)
        fig.update_layout(title=f"{_safe_ticker(ticker)} RSI (14)", xaxis_title="Date", yaxis_title="RSI")
        return json.loads(fig.to_json())
    except Exception as e:
        return _empty_fig("RSI unavailable", str(e))


@app.get("/macd_chart")
def macd_chart(ticker: str = "MSFT", start_date: str = "2023-01-01"):
    try:
        hist = get_history(ticker, start_date)
        m = macd(hist["Close"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["Date"], y=m["macd"], mode="lines", name="MACD"))
        fig.add_trace(go.Scatter(x=hist["Date"], y=m["signal"], mode="lines", name="Signal"))
        fig.add_trace(go.Bar(x=hist["Date"], y=m["hist"], name="Histogram"))
        fig.update_layout(title=f"{_safe_ticker(ticker)} MACD", xaxis_title="Date", yaxis_title="Value", barmode="relative")
        return json.loads(fig.to_json())
    except Exception as e:
        return _empty_fig("MACD unavailable", str(e))


@app.get("/support_resistance_chart")
def support_resistance_chart(ticker: str = "MSFT", start_date: str = "2023-01-01"):
    try:
        hist = get_history(ticker, start_date)
        close = hist["Close"]
        support = close.rolling(20).min()
        resistance = close.rolling(20).max()
        trend = close.rolling(50).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["Date"], y=close, mode="lines", name="Close"))
        fig.add_trace(go.Scatter(x=hist["Date"], y=support, mode="lines", name="20d support"))
        fig.add_trace(go.Scatter(x=hist["Date"], y=resistance, mode="lines", name="20d resistance"))
        fig.add_trace(go.Scatter(x=hist["Date"], y=trend, mode="lines", name="50d trend"))
        fig.update_layout(title=f"{_safe_ticker(ticker)} support / resistance", xaxis_title="Date", yaxis_title="Price")
        return json.loads(fig.to_json())
    except Exception as e:
        return _empty_fig("Support / resistance unavailable", str(e))


@app.get("/return_stats")
def return_stats(ticker: str = "MSFT", benchmark: str = "SPY", start_date: str = "2023-01-01"):
    try:
        pair = get_pair_history(ticker, benchmark, start_date)
        returns = compute_returns(pair, ticker, benchmark)
        t, b = _safe_ticker(ticker), _safe_benchmark(benchmark)
        t_ret = returns[f"{t}_ret"]
        b_ret = returns[f"{b}_ret"]
        beta = np.cov(t_ret, b_ret)[0, 1] / np.var(b_ret) if len(returns) > 2 and np.var(b_ret) != 0 else np.nan
        downside = t_ret[t_ret < 0].std() * np.sqrt(252)
        sharpe = annualized_return(t_ret) / annualized_vol(t_ret) if annualized_vol(t_ret) not in (0, np.nan) else np.nan
        sortino = annualized_return(t_ret) / downside if downside not in (0, np.nan) else np.nan
        rows = [
            {"metric": "Total return", "value": f"{(pair[t].iloc[-1] / pair[t].iloc[0] - 1):.2%}"},
            {"metric": "Annualized return", "value": f"{annualized_return(t_ret):.2%}"},
            {"metric": "Annualized volatility", "value": f"{annualized_vol(t_ret):.2%}"},
            {"metric": "Max drawdown", "value": f"{max_drawdown(pair[t]):.2%}"},
            {"metric": f"Beta vs {b}", "value": f"{beta:.2f}" if pd.notna(beta) else "N/A"},
            {"metric": "Sharpe (rf=0)", "value": f"{sharpe:.2f}" if pd.notna(sharpe) else "N/A"},
            {"metric": "Sortino (rf=0)", "value": f"{sortino:.2f}" if pd.notna(sortino) else "N/A"},
            {"metric": "Hit rate", "value": f"{(t_ret.gt(0).mean()):.2%}"},
        ]
        return rows
    except Exception as e:
        return _empty_table(str(e))


@app.get("/monthly_heatmap")
def monthly_heatmap(ticker: str = "MSFT", start_date: str = "2023-01-01"):
    try:
        hist = get_history(ticker, start_date).set_index("Date")
        monthly = hist["Close"].resample("M").last().pct_change().dropna()
        hm = monthly.to_frame("ret")
        hm["year"] = hm.index.year
        hm["month"] = hm.index.strftime("%b")
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        pivot = hm.pivot(index="year", columns="month", values="ret").reindex(columns=months)
        fig = go.Figure(data=go.Heatmap(z=pivot.values, x=list(pivot.columns), y=[str(y) for y in pivot.index], text=np.round(pivot.values * 100, 1), texttemplate="%{text}", hovertemplate="%{y} %{x}: %{z:.2%}<extra></extra>"))
        fig.update_layout(title=f"{_safe_ticker(ticker)} monthly return heatmap")
        return json.loads(fig.to_json())
    except Exception as e:
        return _empty_fig("Monthly heatmap unavailable", str(e))


@app.get("/risk_scorecard")
def risk_scorecard(ticker: str = "MSFT", benchmark: str = "SPY", start_date: str = "2023-01-01", window: int = 63):
    try:
        pair = get_pair_history(ticker, benchmark, start_date)
        returns = compute_returns(pair, ticker, benchmark)
        t, b = _safe_ticker(ticker), _safe_benchmark(benchmark)
        t_ret = returns[f"{t}_ret"]
        b_ret = returns[f"{b}_ret"]
        current_vol = float(t_ret.tail(window).std() * np.sqrt(252)) if len(t_ret) >= window else annualized_vol(t_ret)
        current_beta = float(np.cov(t_ret.tail(window), b_ret.tail(window))[0, 1] / np.var(b_ret.tail(window))) if len(t_ret) >= window and np.var(b_ret.tail(window)) != 0 else np.nan
        current_dd = float((pair[t] / pair[t].cummax() - 1).iloc[-1])
        rows = [
            {"check": "Volatility level", "status": "High" if current_vol > 0.4 else "Moderate" if current_vol > 0.25 else "Low", "value": f"{current_vol:.2%}"},
            {"check": f"Beta vs {b}", "status": "Aggressive" if pd.notna(current_beta) and current_beta > 1.2 else "Defensive" if pd.notna(current_beta) and current_beta < 0.8 else "Neutral", "value": f"{current_beta:.2f}" if pd.notna(current_beta) else "N/A"},
            {"check": "Current drawdown", "status": "Deep" if current_dd < -0.15 else "Normal", "value": f"{current_dd:.2%}"},
            {"check": "Trend regime", "status": "Uptrend" if pair[t].iloc[-1] > pair[t].rolling(50).mean().iloc[-1] else "Downtrend", "value": f"50d MA {pair[t].rolling(50).mean().iloc[-1]:,.2f}"},
            {"check": "RSI regime", "status": "Overbought" if rsi(pair[t]).iloc[-1] > 70 else "Oversold" if rsi(pair[t]).iloc[-1] < 30 else "Balanced", "value": f"{rsi(pair[t]).iloc[-1]:.1f}"},
        ]
        return rows
    except Exception as e:
        return _empty_table(str(e))


@app.get("/company_snapshot")
def company_snapshot(ticker: str = "MSFT"):
    try:
        info = get_info(ticker)
        rows = [
            {"field": "Ticker", "value": _safe_ticker(ticker)},
            {"field": "Name", "value": info.get("shortName", "N/A")},
            {"field": "Market cap", "value": info.get("marketCap", "N/A")},
            {"field": "Forward PE", "value": info.get("forwardPE", "N/A")},
            {"field": "Trailing PE", "value": info.get("trailingPE", "N/A")},
            {"field": "Price to book", "value": info.get("priceToBook", "N/A")},
            {"field": "Dividend yield", "value": info.get("dividendYield", "N/A")},
            {"field": "52w high", "value": info.get("fiftyTwoWeekHigh", "N/A")},
            {"field": "52w low", "value": info.get("fiftyTwoWeekLow", "N/A")},
            {"field": "Average volume", "value": info.get("averageVolume", "N/A")},
        ]
        return rows
    except Exception as e:
        return _empty_table(str(e))



def rolling_sharpe(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean() * 252
    vol = series.rolling(window).std() * np.sqrt(252)
    return mean / vol.replace(0, np.nan)


def rolling_sortino(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean() * 252
    downside = series.where(series < 0).rolling(window).std() * np.sqrt(252)
    return mean / downside.replace(0, np.nan)


def rolling_alpha_beta(t_ret: pd.Series, b_ret: pd.Series, window: int):
    cov = t_ret.rolling(window).cov(b_ret)
    var = b_ret.rolling(window).var().replace(0, np.nan)
    beta = cov / var
    alpha = (t_ret.rolling(window).mean() - beta * b_ret.rolling(window).mean()) * 252
    return alpha, beta


def momentum_scores(close: pd.Series) -> Dict[str, float]:
    out = {}
    for d in (21, 63, 126, 252):
        if len(close) > d:
            out[f"{d}d"] = float(close.iloc[-1] / close.iloc[-d-1] - 1)
        else:
            out[f"{d}d"] = np.nan
    return out


@app.get('/rolling_sharpe_chart')
def rolling_sharpe_chart(ticker: str = 'MSFT', start_date: str = '2023-01-01', window: int = 63):
    try:
        hist = get_history(ticker, start_date)
        ret = hist['Close'].pct_change().dropna()
        s = rolling_sharpe(ret, window)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist['Date'].iloc[1:], y=s, mode='lines', name='Rolling Sharpe'))
        fig.add_hline(y=0)
        fig.update_layout(title=f'{window}d rolling Sharpe (rf=0)', xaxis_title='Date', yaxis_title='Sharpe')
        return json.loads(fig.to_json())
    except Exception as e:
        return _empty_fig('Rolling Sharpe unavailable', str(e))


@app.get('/rolling_sortino_chart')
def rolling_sortino_chart(ticker: str = 'MSFT', start_date: str = '2023-01-01', window: int = 63):
    try:
        hist = get_history(ticker, start_date)
        ret = hist['Close'].pct_change().dropna()
        s = rolling_sortino(ret, window)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist['Date'].iloc[1:], y=s, mode='lines', name='Rolling Sortino'))
        fig.add_hline(y=0)
        fig.update_layout(title=f'{window}d rolling Sortino (rf=0)', xaxis_title='Date', yaxis_title='Sortino')
        return json.loads(fig.to_json())
    except Exception as e:
        return _empty_fig('Rolling Sortino unavailable', str(e))


@app.get('/rolling_alpha_chart')
def rolling_alpha_chart(ticker: str = 'MSFT', benchmark: str = 'SPY', start_date: str = '2023-01-01', window: int = 63):
    try:
        pair = compute_returns(get_pair_history(ticker, benchmark, start_date), ticker, benchmark)
        t, b = _safe_ticker(ticker), _safe_benchmark(benchmark)
        alpha, _ = rolling_alpha_beta(pair[f'{t}_ret'], pair[f'{b}_ret'], window)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pair['Date'], y=alpha, mode='lines', name='Rolling alpha'))
        fig.add_hline(y=0)
        fig.update_layout(title=f'{window}d rolling alpha vs {b}', xaxis_title='Date', yaxis_title='Alpha (annualized)')
        return json.loads(fig.to_json())
    except Exception as e:
        return _empty_fig('Rolling alpha unavailable', str(e))


@app.get('/relative_strength_chart')
def relative_strength_chart(ticker: str = 'MSFT', benchmark: str = 'SPY', start_date: str = '2023-01-01'):
    try:
        pair = get_pair_history(ticker, benchmark, start_date)
        t, b = _safe_ticker(ticker), _safe_benchmark(benchmark)
        rs = pair[t] / pair[b]
        rs_ma = rs.rolling(20).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pair['Date'], y=rs, mode='lines', name=f'{t}/{b}'))
        fig.add_trace(go.Scatter(x=pair['Date'], y=rs_ma, mode='lines', name='20d mean'))
        fig.update_layout(title=f'Relative strength: {t} vs {b}', xaxis_title='Date', yaxis_title='Price ratio')
        return json.loads(fig.to_json())
    except Exception as e:
        return _empty_fig('Relative strength unavailable', str(e))


@app.get('/momentum_scorecard')
def momentum_scorecard(ticker: str = 'MSFT', benchmark: str = 'SPY', start_date: str = '2023-01-01'):
    try:
        pair = get_pair_history(ticker, benchmark, start_date)
        t, b = _safe_ticker(ticker), _safe_benchmark(benchmark)
        t_scores = momentum_scores(pair[t])
        b_scores = momentum_scores(pair[b])
        rows = []
        for k in ('21d','63d','126d','252d'):
            tv = t_scores[k]
            bv = b_scores[k]
            rel = tv - bv if pd.notna(tv) and pd.notna(bv) else np.nan
            rows.append({
                'horizon': k,
                t: f'{tv:.2%}' if pd.notna(tv) else 'N/A',
                b: f'{bv:.2%}' if pd.notna(bv) else 'N/A',
                'relative': f'{rel:.2%}' if pd.notna(rel) else 'N/A',
                'status': 'Leading' if pd.notna(rel) and rel > 0 else 'Lagging' if pd.notna(rel) and rel < 0 else 'Neutral'
            })
        return rows
    except Exception as e:
        return _empty_table(str(e))


@app.get('/peer_correlation')
def peer_correlation(ticker: str = 'MSFT', peers: str = 'AAPL,NVDA,GOOGL,AMZN', start_date: str = '2023-01-01'):
    try:
        tickers = [_safe_ticker(ticker)] + [p.strip().upper() for p in (peers or '').split(',') if p.strip()]
        seen = []
        for x in tickers:
            if x not in seen:
                seen.append(x)
        prices = []
        for tk in seen[:8]:
            hist = get_history(tk, start_date)[['Date', 'Close']].rename(columns={'Close': tk})
            prices.append(hist)
        df = prices[0]
        for p in prices[1:]:
            df = df.merge(p, on='Date', how='inner')
        corr = df.drop(columns=['Date']).pct_change().dropna().corr().round(2)
        rows = []
        for idx, row in corr.iterrows():
            item = {'ticker': idx}
            item.update({col: (None if pd.isna(val) else float(val)) for col, val in row.items()})
            rows.append(item)
        return rows
    except Exception as e:
        return _empty_table(str(e))


# ---- Added models: volatility regime, mean reversion, Monte Carlo ----

def _safe_sims(value) -> int:
    try:
        v = int(value)
        return max(100, min(v, 5000))
    except Exception:
        return 500


def _safe_days(value) -> int:
    try:
        v = int(value)
        return max(5, min(v, 252))
    except Exception:
        return 63


def _rolling_vol(close: pd.Series, window: int) -> pd.Series:
    return close.pct_change().rolling(window).std() * np.sqrt(TRADING_DAYS)


def _vol_regime_frame(ticker: str, start_date: str, window: int) -> pd.DataFrame:
    hist = get_history(ticker, start_date).copy()
    hist['rv'] = _rolling_vol(hist['Close'], window)
    valid = hist['rv'].dropna()
    if valid.empty:
        raise ValueError('Not enough data for volatility regime model.')
    low = float(valid.quantile(0.33))
    high = float(valid.quantile(0.67))
    hist['low_thr'] = low
    hist['high_thr'] = high
    hist['regime'] = np.where(hist['rv'] <= low, 'Low', np.where(hist['rv'] >= high, 'High', 'Medium'))
    return hist


def _mean_reversion_frame(ticker: str, start_date: str, window: int) -> pd.DataFrame:
    hist = get_history(ticker, start_date).copy()
    hist['mean'] = hist['Close'].rolling(window).mean()
    hist['std'] = hist['Close'].rolling(window).std()
    hist['z'] = (hist['Close'] - hist['mean']) / hist['std'].replace(0, np.nan)
    hist['upper'] = hist['mean'] + 2 * hist['std']
    hist['lower'] = hist['mean'] - 2 * hist['std']
    return hist


def _half_life(series: pd.Series) -> float:
    s = pd.Series(series).dropna()
    if len(s) < 30:
        return np.nan
    lag = s.shift(1).dropna()
    delta = s.diff().dropna()
    lag = lag.loc[delta.index]
    try:
        beta = np.polyfit(lag.values, delta.values, 1)[0]
        if beta >= 0:
            return np.nan
        return float(-np.log(2) / beta)
    except Exception:
        return np.nan


def _mc_paths(ticker: str, start_date: str, days: int, sims: int) -> tuple[pd.DataFrame, float]:
    hist = get_history(ticker, start_date)
    ret = hist['Close'].pct_change().dropna()
    if ret.empty:
        raise ValueError('Not enough return history for Monte Carlo simulation.')
    mu = float(ret.mean())
    sigma = float(ret.std())
    spot = float(hist['Close'].iloc[-1])
    dt = 1.0
    rng = np.random.default_rng(42)
    shocks = rng.normal(size=(days, sims))
    increments = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks)
    paths = np.vstack([np.ones((1, sims)), increments]).cumprod(axis=0) * spot
    idx = np.arange(paths.shape[0])
    return pd.DataFrame(paths, index=idx), spot


@app.get('/vol_regime_chart')
def vol_regime_chart(ticker: str = 'MSFT', start_date: str = '2023-01-01', window: int = 63):
    try:
        window = _safe_window(window)
        df = _vol_regime_frame(ticker, start_date, window)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['rv'], mode='lines', name='Realized vol'))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['low_thr'], mode='lines', name='Low threshold', line=dict(dash='dot')))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['high_thr'], mode='lines', name='High threshold', line=dict(dash='dot')))
        fig.update_layout(title=f'Volatility regime model ({window}d realized vol)', xaxis_title='Date', yaxis_title='Annualized vol')
        return json.loads(fig.to_json())
    except Exception as e:
        return _empty_fig('Volatility regime unavailable', str(e))


@app.get('/vol_regime_scorecard')
def vol_regime_scorecard(ticker: str = 'MSFT', start_date: str = '2023-01-01', window: int = 63):
    try:
        window = _safe_window(window)
        df = _vol_regime_frame(ticker, start_date, window).dropna(subset=['rv']).copy()
        last = df.iloc[-1]
        counts = df['regime'].value_counts()
        current_pctile = float(df['rv'].rank(pct=True).iloc[-1])
        rows = [
            {'metric': 'Current regime', 'value': str(last['regime'])},
            {'metric': f'{window}d realized vol', 'value': _fmt_pct(last['rv'])},
            {'metric': 'Vol percentile', 'value': _fmt_pct(current_pctile)},
            {'metric': 'Low threshold', 'value': _fmt_pct(last['low_thr'])},
            {'metric': 'High threshold', 'value': _fmt_pct(last['high_thr'])},
            {'metric': 'Days in low regime', 'value': _fmt_int(counts.get('Low', 0))},
            {'metric': 'Days in medium regime', 'value': _fmt_int(counts.get('Medium', 0))},
            {'metric': 'Days in high regime', 'value': _fmt_int(counts.get('High', 0))},
        ]
        return rows
    except Exception as e:
        return _empty_table(str(e))


@app.get('/mean_reversion_chart')
def mean_reversion_chart(ticker: str = 'MSFT', start_date: str = '2023-01-01', window: int = 63):
    try:
        window = _safe_window(window)
        df = _mean_reversion_frame(ticker, start_date, window)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['z'], mode='lines', name='Z-score'))
        fig.add_hline(y=2)
        fig.add_hline(y=-2)
        fig.add_hline(y=0)
        fig.update_layout(title=f'Mean reversion z-score ({window}d)', xaxis_title='Date', yaxis_title='Z-score')
        return json.loads(fig.to_json())
    except Exception as e:
        return _empty_fig('Mean reversion unavailable', str(e))


@app.get('/mean_reversion_scorecard')
def mean_reversion_scorecard(ticker: str = 'MSFT', start_date: str = '2023-01-01', window: int = 63):
    try:
        window = _safe_window(window)
        df = _mean_reversion_frame(ticker, start_date, window).dropna(subset=['z']).copy()
        if df.empty:
            raise ValueError('Not enough data for mean reversion scorecard.')
        last = df.iloc[-1]
        hl = _half_life(df['Close'] - df['mean'])
        signal = 'Neutral'
        if last['z'] >= 2:
            signal = 'Overextended above mean'
        elif last['z'] <= -2:
            signal = 'Overextended below mean'
        rows = [
            {'metric': 'Current z-score', 'value': _fmt_num(last['z'])},
            {'metric': 'Price', 'value': _fmt_num(last['Close'])},
            {'metric': f'{window}d mean', 'value': _fmt_num(last['mean'])},
            {'metric': 'Distance from mean', 'value': _fmt_pct(last['Close'] / last['mean'] - 1 if last['mean'] else np.nan)},
            {'metric': 'Estimated half-life', 'value': f"{hl:.1f} days" if pd.notna(hl) else 'N/A'},
            {'metric': 'Signal', 'value': signal},
        ]
        return rows
    except Exception as e:
        return _empty_table(str(e))


@app.get('/monte_carlo_paths')
def monte_carlo_paths(ticker: str = 'MSFT', start_date: str = '2023-01-01', sim_days: int = 63, n_sims: int = 500):
    try:
        days = _safe_days(sim_days)
        sims = _safe_sims(n_sims)
        paths, spot = _mc_paths(ticker, start_date, days, sims)
        fig = go.Figure()
        for col in paths.columns[: min(60, sims)]:
            fig.add_trace(go.Scatter(x=paths.index, y=paths[col], mode='lines', line=dict(width=1), opacity=0.18, showlegend=False))
        q = paths.quantile([0.05, 0.5, 0.95], axis=1).T
        fig.add_trace(go.Scatter(x=q.index, y=q[0.5], mode='lines', name='Median path', line=dict(width=3)))
        fig.add_trace(go.Scatter(x=q.index, y=q[0.95], mode='lines', name='95th pct', line=dict(dash='dot')))
        fig.add_trace(go.Scatter(x=q.index, y=q[0.05], mode='lines', name='5th pct', line=dict(dash='dot')))
        fig.update_layout(title=f'Monte Carlo simulation: {days} trading days, {sims} sims (spot {spot:.2f})', xaxis_title='Trading days forward', yaxis_title='Simulated price')
        return json.loads(fig.to_json())
    except Exception as e:
        return _empty_fig('Monte Carlo unavailable', str(e))


@app.get('/monte_carlo_scorecard')
def monte_carlo_scorecard(ticker: str = 'MSFT', start_date: str = '2023-01-01', sim_days: int = 63, n_sims: int = 500):
    try:
        days = _safe_days(sim_days)
        sims = _safe_sims(n_sims)
        paths, spot = _mc_paths(ticker, start_date, days, sims)
        end = paths.iloc[-1]
        rows = [
            {'metric': 'Spot', 'value': _fmt_num(spot)},
            {'metric': f'Expected price in {days}d', 'value': _fmt_num(end.mean())},
            {'metric': 'Median terminal price', 'value': _fmt_num(end.median())},
            {'metric': '5th percentile', 'value': _fmt_num(end.quantile(0.05))},
            {'metric': '95th percentile', 'value': _fmt_num(end.quantile(0.95))},
            {'metric': 'Prob. finish above spot', 'value': _fmt_pct((end > spot).mean())},
            {'metric': 'Prob. finish > +10%', 'value': _fmt_pct((end > spot * 1.10).mean())},
            {'metric': 'Prob. finish < -10%', 'value': _fmt_pct((end < spot * 0.90).mean())},
        ]
        return rows
    except Exception as e:
        return _empty_table(str(e))


@app.get("/monthly_heatmap_table")
def monthly_heatmap_table(ticker: str = "MSFT", start_date: str = "2023-01-01"):
    try:
        hist = get_history(ticker, start_date).set_index("Date")
        monthly = hist["Close"].resample("M").last().pct_change().dropna()
        hm = monthly.to_frame("ret")
        hm["year"] = hm.index.year.astype(str)
        hm["month"] = hm.index.strftime("%b")
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        pivot = hm.pivot(index="year", columns="month", values="ret").reindex(columns=months)
        rows = []
        for year, row in pivot.iterrows():
            out = {"year": str(year)}
            for m in months:
                val = row.get(m)
                out[m] = None if pd.isna(val) else round(float(val) * 100, 2)
            rows.append(out)
        if not rows:
            raise ValueError("Not enough monthly data for heatmap table.")
        return rows
    except Exception as e:
        return _empty_table(str(e))
