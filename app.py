import os
import io
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timezone

st.set_page_config(page_title="AI Trading Analyst PRO", page_icon="📈", layout="wide")

SPOT_BASE = "https://data-api.binance.vision"
FUTURES_BASE = "https://fapi.binance.com"
INTERVALS = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"]

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
.metric-card {background: rgba(120,120,120,.08); padding: 14px; border-radius: 12px;}
.small-note {font-size: .85rem; opacity: .8;}
</style>
""", unsafe_allow_html=True)

st.title("📈 AI Trading Analyst PRO")
st.caption("Binance Spot + USDⓈ-M Futures | Multi-indicator analysis | Signal-only starter build")

with st.sidebar:
    st.header("Market Settings")
    market = st.selectbox("Market", ["Spot", "Futures"])
    symbol = st.text_input("Symbol", "BTCUSDT").upper().replace("/", "").strip()
    interval = st.selectbox("Main timeframe", INTERVALS, index=2)
    limit = st.slider("Candles", 200, 1000, 500, 100)
    st.divider()
    st.subheader("Risk Management")
    balance = st.number_input("Account balance (USDT)", min_value=0.0, value=1000.0, step=100.0)
    risk_pct = st.number_input("Risk per trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    st.caption("Position sizing is an estimate. Fees, slippage, funding and leverage are not included.")
    st.divider()
    min_score = st.slider("Minimum signal score", 50, 95, 75, 1)
    refresh = st.button("🔄 Refresh market data", use_container_width=True)



@st.cache_data(ttl=15)
def get_market_data(symbol, market, interval, limit=500):

    symbol = symbol.upper().replace("/", "").replace("-", "")

    if market == "Spot":

        url = f"{SPOT_BASE}/api/v3/klines"

        r = requests.get(
            url,
            params={
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            },
            timeout=20
        )

        r.raise_for_status()
        raw = r.json()

        cols = [
            "open_time", "open", "high", "low", "close",
            "volume", "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ]

        df = pd.DataFrame(raw, columns=cols)

        price_url = f"{SPOT_BASE}/api/v3/ticker/price"

    else:

        url = f"{FUTURES_BASE}/fapi/v1/klines"

        r = requests.get(
            url,
            params={
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            },
            timeout=20
        )

        r.raise_for_status()
        raw = r.json()

        cols = [
            "open_time", "open", "high", "low", "close",
            "volume", "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ]

        df = pd.DataFrame(raw, columns=cols)

        price_url = f"{FUTURES_BASE}/fapi/v1/ticker/price"

    pr = requests.get(
        price_url,
        params={"symbol": symbol},
        timeout=20
    )

    pr.raise_for_status()

    current_price = float(pr.json()["price"])

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        unit="ms",
        utc=True
    )

    df["time"] = df["open_time"]

    return df, current_price


       else:
        # Binance USD-M Futures public API
        url = f"{FUTURES_BASE}/fapi/v1/klines"

        r = requests.get(
            url,
            params={
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            },
            timeout=20
        )
        r.raise_for_status()

        raw = r.json()

        cols = [
            "open_time", "open", "high", "low", "close",
            "volume", "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ]

        df = pd.DataFrame(raw, columns=cols)

        quote_url = f"{FUTURES_BASE}/fapi/v1/ticker/price"

        qr = requests.get(
            quote_url,
            params={"symbol": symbol},
            timeout=20
        )
        qr.raise_for_status()

        current_price = float(qr.json()["price"])

        if not bars:
            raise ValueError(
                f"No Futures candle data found for {symbol}"
            )

        df = pd.DataFrame(bars)

        df = df.rename(columns={
            "openTime": "open_time",
            "closeTime": "close_time",
            "quoteVolume": "quote_volume",
            "takerBuyVolume": "taker_buy_base",
            "takerBuyQuoteVolume": "taker_buy_quote"
        })

        df["trades"] = 0
        df["ignore"] = 0

        # Get live Futures quote
        quote_url = (
            "https://api.coinbeacon.io/public/market/quotes/batch"
        )

        qr = requests.get(
            quote_url,
            headers=headers,
            params={"m": f"binance-futures:{symbol}"},
            timeout=20
        )
        qr.raise_for_status()

        quotes = qr.json().get("quotes", [])

        if not quotes or quotes[0].get("price") is None:
            raise ValueError("Futures live price unavailable")

        current_price = float(quotes[0]["price"])

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if market == "Spot":
        df["open_time"] = pd.to_datetime(
            df["open_time"], unit="ms", utc=True
        )
    else:
        df["open_time"] = pd.to_datetime(
            df["open_time"], unit="ms", utc=True
        )
        df["time"] = df["open_time"]

    return df, current_price


def add_indicators(df):
    d = df.copy()
    close, high, low, vol = d.close, d.high, d.low, d.volume
    d["sma20"] = close.rolling(20).mean()
    d["sma50"] = close.rolling(50).mean()
    for n in [9, 21, 50, 200]:
        d[f"ema{n}"] = close.ewm(span=n, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    d["rsi"] = 100 - (100 / (1 + rs))
    prev_close = close.shift(1)
    tr = pd.concat([(high-low), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()
    tp = (high + low + close) / 3
    d["vwap"] = (tp * vol).cumsum() / vol.cumsum().replace(0, np.nan)
    # TSI: double-smoothed momentum oscillator
    mom = close.diff()
    abs_mom = mom.abs()
    sm1 = mom.ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
    sa1 = abs_mom.ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
    d["tsi"] = 100 * sm1 / sa1.replace(0, np.nan)
    d["vol_sma20"] = vol.rolling(20).mean()
    # Pivot-based S/R zones
    d["pivot_high"] = high.where((high == high.rolling(11, center=True).max()))
    d["pivot_low"] = low.where((low == low.rolling(11, center=True).min()))
    return d

add(
    "TSI Divergence",
    bull_div,
    bear_div,
    15,
    "Bullish divergence" if bull_div
    else "Bearish divergence" if bear_div
    else "No divergence"
)
    # Heuristic swing comparison; not a full pivot-confirmed divergence engine.
    if len(d) < 40:
        return False, False
    recent = d.tail(30).reset_index(drop=True)
    p1 = recent.iloc[:15]
    p2 = recent.iloc[15:]
    low1, low2 = p1.low.min(), p2.low.min()
    rsi_low1, rsi_low2 = p1.loc[p1.low.idxmin(), "tsi"], p2.loc[p2.low.idxmin(), "tsi"]
    high1, high2 = p1.high.max(), p2.high.max()
    rsi_high1, rsi_high2 = p1.loc[p1.high.idxmax(), "tsi"], p2.loc[p2.high.idxmax(), "tsi"]
    bull = low2 < low1 and rsi_low2 > rsi_low1
    bear = high2 > high1 and rsi_high2 < rsi_high1
    return bool(bull), bool(bear)
add("TSI Divergence"...
def divergence_flags(d):

    if len(d) < 40:
        return False, False

    recent = d.tail(30).reset_index(drop=True)

    p1 = recent.iloc[:15]
    p2 = recent.iloc[15:]

    low1, low2 = p1.low.min(), p2.low.min()

    tsi_low1 = p1.loc[p1.low.idxmin(), "tsi"]
    tsi_low2 = p2.loc[p2.low.idxmin(), "tsi"]

    high1, high2 = p1.high.max(), p2.high.max()

    tsi_high1 = p1.loc[p1.high.idxmax(), "tsi"]
    tsi_high2 = p2.loc[p2.high.idxmax(), "tsi"]

    bull = low2 < low1 and tsi_low2 > tsi_low1

    bear = high2 > high1 and tsi_high2 < tsi_high1

    return bool(bull), bool(bear)

def analyze(d):
    last = d.iloc[-1]
    prev = d.iloc[-2]
    score_long, score_short = 0, 0
    reasons = []
    checks = []

    def add(name, long_ok, short_ok, weight, detail):
        nonlocal score_long, score_short
        if long_ok: score_long += weight
        if short_ok: score_short += weight
        checks.append((name, "Bullish" if long_ok and not short_ok else "Bearish" if short_ok and not long_ok else "Mixed" if long_ok and short_ok else "Neutral", weight))
        reasons.append(f"{name}: {detail}")

    add("MA / EMA trend", last.ema50 > last.ema200 and last.close > last.sma50,
        last.ema50 < last.ema200 and last.close < last.sma50, 15,
        f"EMA50 {last.ema50:.4f}, EMA200 {last.ema200:.4f}")
    add("RSI", 50 < last.rsi < 70, 30 < last.rsi < 50, 10, f"RSI {last.rsi:.1f}")
    # Approximate local structure break
    prior_high = d.high.iloc[-21:-1].max()
    prior_low = d.low.iloc[-21:-1].min()
    bull_bos, bear_bos = last.close > prior_high, last.close < prior_low
    add("SNR / Structure", bull_bos, bear_bos, 15, "Break above 20-candle high" if bull_bos else "Break below 20-candle low" if bear_bos else "No confirmed 20-candle breakout")
    # Lightweight SMC proxies: BOS + three-candle FVG
    bull_fvg = d.low.iloc[-1] > d.high.iloc[-3]
    bear_fvg = d.high.iloc[-1] < d.low.iloc[-3]
    add("SMC proxy", bull_bos or bull_fvg, bear_bos or bear_fvg, 15,
        "BOS/FVG proxy detected" if (bull_bos or bear_bos or bull_fvg or bear_fvg) else "No BOS/FVG proxy")
    # Elliott wave is intentionally treated as weak heuristic, not automated wave labeling.
    momentum_up = last.close > last.ema21 and last.ema21 > prev.ema21
    momentum_dn = last.close < last.ema21 and last.ema21 < prev.ema21
    add("Elliott momentum proxy", momentum_up, momentum_dn, 10, "EMA21 slope/momentum proxy; not formal wave count")
    vol_ok = last.volume > 1.5 * last.vol_sma20 if pd.notna(last.vol_sma20) else False
    add("Volume", vol_ok and last.close > last.open, vol_ok and last.close < last.open, 10,
        f"Volume / 20-SMA = {last.volume / last.vol_sma20:.2f}" if last.vol_sma20 else "Volume baseline unavailable")
    add("VWAP", last.close > last.vwap, last.close < last.vwap, 5, f"Close {last.close:.4f} vs VWAP {last.vwap:.4f}")
    # ATR is used for sizing/stops, not directional edge; award only if ATR is finite and nonzero.
    atr_ok = pd.notna(last.atr) and last.atr > 0
    add("ATR volatility", atr_ok, atr_ok, 5, f"ATR {last.atr:.6f}" if atr_ok else "ATR unavailable")
    bull_div, bear_div = divergence_flags(d)
    add("TSI / RSI divergence", bull_div, bear_div, 15,
        "Bullish divergence" if bull_div else "Bearish divergence" if bear_div else "No divergence")

    # normalize because neutral checks can contribute to both; raw max remains 100
    return score_long, score_short, checks, reasons, bull_div, bear_div

def send_telegram(message):
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip() or str(st.secrets.get("TELEGRAM_BOT_TOKEN", "")).strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip() or str(st.secrets.get("TELEGRAM_CHAT_ID", "")).strip()
    if not token or not chat_id:
        return False, "Telegram credentials are not configured."
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          data={"chat_id": chat_id, "text": message}, timeout=12)
        r.raise_for_status()
        return True, "Telegram alert sent."
    except Exception as e:
        return False, f"Telegram error: {e}"

if refresh:
    st.cache_data.clear()

try:
    with st.spinner("Fetching market data..."):

        raw, current_price = get_market_data(
            symbol, market, interval, limit
        )

        df = add_indicators(raw)

    if len(df) < 210:
        st.warning(
            "For EMA200 and stable analysis, load at least 210 candles."
        )

    long_score, short_score, checks, reasons, bull_div, bear_div = analyze(df)

except Exception as e:

    st.error(f"Could not load market data: {e}")

    st.info(
        "Check the symbol, internet connection, and Binance API availability."
    )

    st.stop()


last = df.iloc[-1]
risk_amount = balance * (risk_pct / 100)
atr_stop_distance = 1.5 * last.atr if pd.notna(last.atr) else np.nan
qty = risk_amount / atr_stop_distance if atr_stop_distance and np.isfinite(atr_stop_distance) else np.nan
long_ready = long_score >= min_score and long_score > short_score
short_ready = short_score >= min_score and short_score > long_score

if long_ready:
    signal = "BUY" if market == "Spot" else "LONG"
    signal_color = "green"
elif short_ready and market == "Futures":
    signal, signal_color = "SHORT", "red"
elif short_ready and market == "Spot":
    signal, signal_color = "NO BUY"
else:
    signal, signal_color = "WAIT", "orange"

c1,c2,c3,c4 = st.columns(4)
c1.metric("Live Price", f"{current_price:,.4f} USDT")
c2.metric("Long Score", f"{long_score}/100")
c3.metric("Short Score", f"{short_score}/100")
c4.metric("Signal", signal)

st.caption(f"Last candle time: {last.time.strftime('%Y-%m-%d %H:%M UTC')} | Data: Binance public market endpoint")

# Higher-timeframe context
st.subheader("Multi-Timeframe Trend Confirmation")

mtf_cols = st.columns(3)

for col, tf in zip(mtf_cols, ["15m", "1h", "4h"]):

    try:

        tf_raw, _ = get_market_data(
            symbol, market, tf, 250
        )

        tf_df = add_indicators(tf_raw)

        tf_last = tf_df.iloc[-1]

        if tf_last.close > tf_last.ema50 > tf_last.ema200:

            mtf_state = "Bullish"

        elif tf_last.close < tf_last.ema50 < tf_last.ema200:

            mtf_state = "Bearish"

        else:

            mtf_state = "Mixed"

        col.metric(
            tf,
            mtf_state,
            f"Close {tf_last.close:,.4f}"
        )

    except Exception as e:

        col.metric(
            tf,
            "Unavailable",
            "Could not load"
        )

if signal in ["BUY", "LONG", "SHORT"]:
    st.success(f"Potential {signal} setup meets the selected score threshold. Confirm candle close and higher-timeframe context before trading.")
elif signal == "NO BUY":
    st.warning("Bearish conditions detected, but Spot mode does not issue short-selling signals.")
else:
    st.warning("No trade: confluence threshold not met. Waiting is a valid trading decision.")

left, right = st.columns([1.7,1])
with left:
    st.subheader("Price Chart")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.time, open=df.open, high=df.high, low=df.low, close=df.close, name="Candles"))
    for col, label in [("ema21","EMA 21"),("ema50","EMA 50"),("ema200","EMA 200"),("vwap","VWAP")]:
        fig.add_trace(go.Scatter(x=df.time, y=df[col], mode="lines", name=label))
    fig.update_layout(height=560, xaxis_rangeslider_visible=False, legend_orientation="h")
    st.plotly_chart(fig, use_container_width=True)
with right:
    st.subheader("Strategy Confirmation")
    st.dataframe(pd.DataFrame(checks, columns=["Strategy","State","Weight"]), hide_index=True, use_container_width=True)
    st.markdown("**Analysis notes**")
    for item in reasons:
        st.write("• " + item)

st.subheader("Trade Planning – Risk First")
p1,p2,p3 = st.columns(3)
p1.metric("Risk budget", f"{risk_amount:,.2f} USDT")
p2.metric("ATR stop distance", f"{atr_stop_distance:,.6f}" if np.isfinite(atr_stop_distance) else "N/A")
p3.metric("Estimated quantity", f"{qty:.6f}" if np.isfinite(qty) else "N/A")
if signal in ["BUY","LONG","SHORT"] and np.isfinite(atr_stop_distance):
    if signal in ["BUY","LONG"]:
        sl = last.close - atr_stop_distance
        tp1, tp2 = last.close + 2*atr_stop_distance, last.close + 3*atr_stop_distance
    else:
        sl = last.close + atr_stop_distance
        tp1, tp2 = last.close - 2*atr_stop_distance, last.close - 3*atr_stop_distance
    a,b,c = st.columns(3)
    a.metric("Illustrative Stop Loss", f"{sl:,.4f}")
    b.metric("TP1 (2R)", f"{tp1:,.4f}")
    c.metric("TP2 (3R)", f"{tp2:,.4f}")
    st.caption("Illustrative levels only. Adjust for nearby structure, fees, slippage, funding and exchange precision.")

st.subheader("Screenshot Analysis Input")
uploaded = st.file_uploader("Upload a chart screenshot (PNG/JPG)", type=["png","jpg","jpeg"])
if uploaded:
    st.image(uploaded, caption="Uploaded reference screenshot", use_container_width=True)
    st.info("This starter build displays the screenshot alongside live Binance analysis. Full visual AI chart interpretation requires connecting a vision-model API; the screenshot is not yet automatically interpreted.")

st.subheader("Telegram Alerts")
alert_text = (f"AI Trading Analyst PRO\\n{symbol} | {market} | {interval}\\nSignal: {signal}\\nLong score: {long_score}/100 | Short score: {short_score}/100\\nPrice: {last.close}\\nRisk budget: {risk_amount:.2f} USDT\\nEducational signal only; verify before trading.")
if st.button("Send current analysis to Telegram"):
    ok, msg = send_telegram(alert_text)
    st.success(msg) if ok else st.warning(msg)

st.subheader("Data Export")
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download candles + indicators (CSV)", csv, file_name=f"{symbol}_{interval}_analysis.csv", mime="text/csv")

with st.expander("Methodology & limitations"):
    st.markdown("""
    - Scores are rule-based confluence scores, **not probabilities or predicted win rates**.
    - SMC, support/resistance, divergence and Elliott-wave components are simplified proxies in this starter build.
    - Elliott Wave requires subjective swing labeling; this app does not claim a formal wave count.
    - ATR-based sizing is an estimate and excludes fees, slippage, funding, liquidation risk and contract specifications.
    - No order execution is implemented. No Binance API key is required for public market data.
    - Test with historical out-of-sample data and paper trading before risking capital.
    """)
st.caption("AI Trading Analyst PRO v1.0 | Signal-only prototype | Not financial advice")
