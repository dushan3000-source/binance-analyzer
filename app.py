import os
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ============================================================
# AI TRADING ANALYST PRO — Binance Spot + USD-M Futures
# Rule-based analysis only. No order execution.
# ============================================================

st.set_page_config(page_title="AI Trading Analyst PRO", page_icon="📈", layout="wide")

SPOT_BASE = "https://data-api.binance.vision"
FUTURES_BASE = "https://fapi.binance.com"
INTERVALS = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"]

st.markdown("""
<style>
.block-container {padding-top: 1.1rem; padding-bottom: 2rem;}
div[data-testid="stMetric"] {background: rgba(120,120,120,.07); padding: 12px; border-radius: 10px;}
.small-note {font-size: .85rem; opacity: .8;}
</style>
""", unsafe_allow_html=True)

st.title("📈 AI Trading Analyst PRO")
st.caption("Binance Spot + USDⓈ-M Futures • Structure • Order-book liquidity • Risk-first trade planning")

# --------------------------- Sidebar ---------------------------

with st.sidebar:
    st.header("Market Settings")
    market = st.selectbox("Market", ["Spot", "Futures"])
    symbol = st.text_input("Symbol", "BTCUSDT").upper().replace("/", "").replace("-", "").strip()
    interval = st.selectbox("Main timeframe", INTERVALS, index=2)
    limit = st.slider("Candles", 210, 1000, 500, 50)
    st.divider()
    st.subheader("Risk Management")
    balance = st.number_input("Account balance (USDT)", min_value=0.0, value=1000.0, step=100.0)
    risk_pct = st.number_input("Risk per trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    rr1 = st.number_input("TP1 risk/reward (R)", min_value=0.5, max_value=5.0, value=1.5, step=0.5)
    rr2 = st.number_input("TP2 risk/reward (R)", min_value=1.0, max_value=10.0, value=2.5, step=0.5)
    rr3 = st.number_input("TP3 risk/reward (R)", min_value=1.5, max_value=15.0, value=3.5, step=0.5)
    min_score = st.slider("Minimum confluence score", 50, 95, 70, 1)
    st.caption("Position sizing is an estimate. Fees, slippage, funding, leverage and contract rules are excluded.")
    refresh = st.button("🔄 Refresh market data", use_container_width=True)

# --------------------------- Data ---------------------------

@st.cache_data(ttl=15)
def get_market_data(symbol, market, interval, limit=500):
    symbol = symbol.upper().replace("/", "").replace("-", "")
    if not symbol.isalnum():
        raise ValueError("Enter a valid Binance symbol, e.g. BTCUSDT.")

    if market == "Spot":
        base, path = SPOT_BASE, "/api/v3/klines"
        ticker_path = "/api/v3/ticker/price"
    else:
        base, path = FUTURES_BASE, "/fapi/v1/klines"
        ticker_path = "/fapi/v1/ticker/price"

    response = requests.get(
        base + path,
        params={"symbol": symbol, "interval": interval, "limit": int(limit)},
        timeout=20,
    )
    response.raise_for_status()
    raw = response.json()
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"No candle data returned for {symbol}.")

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(raw, columns=cols)

    ticker = requests.get(base + ticker_path, params={"symbol": symbol}, timeout=15)
    ticker.raise_for_status()
    current_price = float(ticker.json()["price"])

    numeric = ["open", "high", "low", "close", "volume", "taker_buy_base", "quote_volume"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["time"] = df["open_time"]
    df = df.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)
    if len(df) < 50:
        raise ValueError("Not enough valid candles returned.")
    return df, current_price


@st.cache_data(ttl=8)
def get_order_book(symbol, market, depth_limit=100):
    """Fetch public DOM snapshot. This is a snapshot, not historical order flow."""
    base = SPOT_BASE if market == "Spot" else FUTURES_BASE
    path = "/api/v3/depth" if market == "Spot" else "/fapi/v1/depth"
    response = requests.get(
        base + path,
        params={"symbol": symbol, "limit": depth_limit},
        timeout=15,
    )
    response.raise_for_status()
    book = response.json()
    bids = [(float(p), float(q)) for p, q in book.get("bids", [])]
    asks = [(float(p), float(q)) for p, q in book.get("asks", [])]
    if not bids or not asks:
        raise ValueError("Order book is empty.")
    bid_qty = sum(q for _, q in bids[:20])
    ask_qty = sum(q for _, q in asks[:20])
    total = bid_qty + ask_qty
    imbalance = (bid_qty - ask_qty) / total if total else 0.0
    best_bid, best_ask = bids[0][0], asks[0][0]
    mid = (best_bid + best_ask) / 2
    spread_pct = ((best_ask - best_bid) / mid * 100) if mid else np.nan
    bid_wall = max(bids[:20], key=lambda x: x[1])
    ask_wall = max(asks[:20], key=lambda x: x[1])
    return {
        "bid_qty": bid_qty, "ask_qty": ask_qty, "imbalance": imbalance,
        "spread_pct": spread_pct, "best_bid": best_bid, "best_ask": best_ask,
        "bid_wall": bid_wall, "ask_wall": ask_wall,
    }


# --------------------------- Indicators ---------------------------

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
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    d["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()

    typical = (high + low + close) / 3
    d["vwap"] = (typical * vol).cumsum() / vol.cumsum().replace(0, np.nan)

    mom = close.diff()
    smoothed_mom = mom.ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
    smoothed_abs = mom.abs().ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
    d["tsi"] = 100 * smoothed_mom / smoothed_abs.replace(0, np.nan)
    d["vol_sma20"] = vol.rolling(20).mean()

    # Approximate aggressive-buy volume ratio from Binance kline taker-buy field.
    d["buy_ratio"] = d["taker_buy_base"] / d["volume"].replace(0, np.nan)
    d["buy_ratio"] = d["buy_ratio"].clip(0, 1)
    d["sell_ratio"] = 1 - d["buy_ratio"]
    return d


def divergence_flags(d):
    if len(d) < 40:
        return False, False
    recent = d.tail(30).reset_index(drop=True)
    first, second = recent.iloc[:15], recent.iloc[15:]
    low1, low2 = first.low.min(), second.low.min()
    tsi_low1 = first.loc[first.low.idxmin(), "tsi"]
    tsi_low2 = second.loc[second.low.idxmin(), "tsi"]
    high1, high2 = first.high.max(), second.high.max()
    tsi_high1 = first.loc[first.high.idxmax(), "tsi"]
    tsi_high2 = second.loc[second.high.idxmax(), "tsi"]
    bullish = low2 < low1 and tsi_low2 > tsi_low1
    bearish = high2 > high1 and tsi_high2 < tsi_high1
    return bool(bullish), bool(bearish)


# --------------------------- Market structure ---------------------------

def find_support_resistance(d, price):
    """Recent confirmed-ish swing levels; clustered into practical nearby levels."""
    window = d.tail(180).copy()
    ph = window.high[(window.high.shift(1) < window.high) & (window.high.shift(-1) < window.high)]
    pl = window.low[(window.low.shift(1) > window.low) & (window.low.shift(-1) > window.low)]
    highs = sorted([float(x) for x in ph.dropna() if x > price])
    lows = sorted([float(x) for x in pl.dropna() if x < price], reverse=True)
    resistance = highs[0] if highs else float(window.high.max())
    support = lows[0] if lows else float(window.low.min())
    return support, resistance, lows[:5], highs[:5]


def detect_reversal(d, support, resistance, book):
    last = d.iloc[-1]
    candle_range = max(float(last.high - last.low), 1e-12)
    lower_wick = min(last.open, last.close) - last.low
    upper_wick = last.high - max(last.open, last.close)
    lower_ratio = lower_wick / candle_range
    upper_ratio = upper_wick / candle_range
    vol_ratio = (last.volume / last.vol_sma20) if pd.notna(last.vol_sma20) and last.vol_sma20 > 0 else 0
    swept_support = last.low < support and last.close > support
    swept_resistance = last.high > resistance and last.close < resistance
    bullish_reversal = (swept_support or (lower_ratio > 0.45 and last.close > last.open)) and last.rsi < 55
    bearish_reversal = (swept_resistance or (upper_ratio > 0.45 and last.close < last.open)) and last.rsi > 45
    # Absorption is a candle/volume/taker-flow heuristic, not proof of hidden orders.
    bullish_absorption = lower_ratio > 0.40 and vol_ratio > 1.2 and last.buy_ratio < 0.58
    bearish_absorption = upper_ratio > 0.40 and vol_ratio > 1.2 and last.buy_ratio > 0.42
    # DOM imbalance is a short-lived order-book snapshot.
    bullish_dom = book is not None and book["imbalance"] > 0.12
    bearish_dom = book is not None and book["imbalance"] < -0.12
    return {
        "lower_wick_ratio": lower_ratio, "upper_wick_ratio": upper_ratio,
        "volume_ratio": vol_ratio, "swept_support": bool(swept_support),
        "swept_resistance": bool(swept_resistance),
        "bullish_reversal": bool(bullish_reversal),
        "bearish_reversal": bool(bearish_reversal),
        "bullish_absorption": bool(bullish_absorption),
        "bearish_absorption": bool(bearish_absorption),
        "bullish_dom": bool(bullish_dom), "bearish_dom": bool(bearish_dom),
    }


# --------------------------- Scoring ---------------------------

def analyze(d, book, support, resistance):
    last, prev = d.iloc[-1], d.iloc[-2]
    long_score = 0
    short_score = 0
    checks = []
    reasons = []

    def add(name, long_ok, short_ok, weight, detail):
        nonlocal long_score, short_score
        if long_ok:
            long_score += weight
        if short_ok:
            short_score += weight
        state = "Bullish" if long_ok and not short_ok else "Bearish" if short_ok and not long_ok else "Mixed" if long_ok and short_ok else "Neutral"
        checks.append((name, state, weight))
        reasons.append(f"{name}: {detail}")

    trend_up = last.close > last.ema50 > last.ema200
    trend_down = last.close < last.ema50 < last.ema200
    add("EMA trend", trend_up, trend_down, 15, f"EMA50 {last.ema50:.4f} / EMA200 {last.ema200:.4f}")

    add("RSI momentum", 50 < last.rsi < 70, 30 < last.rsi < 50, 8, f"RSI {last.rsi:.1f}")

    prior_high = d.high.iloc[-21:-1].max()
    prior_low = d.low.iloc[-21:-1].min()
    bull_bos = last.close > prior_high
    bear_bos = last.close < prior_low
    add("Break of structure", bull_bos, bear_bos, 12, "Bullish BOS" if bull_bos else "Bearish BOS" if bear_bos else "No 20-candle close break")

    bull_fvg = d.low.iloc[-1] > d.high.iloc[-3]
    bear_fvg = d.high.iloc[-1] < d.low.iloc[-3]
    add("Fair value gap proxy", bull_fvg, bear_fvg, 7, "3-candle imbalance proxy")

    add("VWAP", last.close > last.vwap, last.close < last.vwap, 6, f"Close {last.close:.4f} / VWAP {last.vwap:.4f}")
    add("Volume confirmation",
        last.volume > 1.3 * last.vol_sma20 and last.close > last.open if pd.notna(last.vol_sma20) else False,
        last.volume > 1.3 * last.vol_sma20 and last.close < last.open if pd.notna(last.vol_sma20) else False,
        8, f"Volume ratio {last.volume / last.vol_sma20:.2f}" if pd.notna(last.vol_sma20) and last.vol_sma20 else "Unavailable")

    bull_div, bear_div = divergence_flags(d)
    add("TSI divergence", bull_div, bear_div, 8, "Bullish divergence" if bull_div else "Bearish divergence" if bear_div else "No divergence")

    reversal = detect_reversal(d, support, resistance, book)
    add("Market reversal / liquidity sweep",
        reversal["bullish_reversal"], reversal["bearish_reversal"], 10,
        "Bullish rejection/sweep" if reversal["bullish_reversal"] else "Bearish rejection/sweep" if reversal["bearish_reversal"] else "No clear reversal")

    add("Absorption estimate",
        reversal["bullish_absorption"], reversal["bearish_absorption"], 8,
        "Possible buy-side absorption" if reversal["bullish_absorption"] else "Possible sell-side absorption" if reversal["bearish_absorption"] else "No strong absorption pattern")

    add("DOM liquidity imbalance",
        reversal["bullish_dom"], reversal["bearish_dom"], 8,
        f"Bid/ask imbalance {book['imbalance']:+.1%}" if book else "DOM unavailable")

    near_support = abs(last.close - support) <= 1.0 * last.atr
    near_resistance = abs(resistance - last.close) <= 1.0 * last.atr
    add("Support / resistance location", near_support, near_resistance, 10,
        f"Support {support:.4f} / Resistance {resistance:.4f}")

    return min(long_score, 100), min(short_score, 100), checks, reasons, reversal


# --------------------------- Telegram ---------------------------

def send_telegram(message):
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    try:
        if not token:
            token = str(st.secrets.get("TELEGRAM_BOT_TOKEN", "")).strip()
        if not chat_id:
            chat_id = str(st.secrets.get("TELEGRAM_CHAT_ID", "")).strip()
    except Exception:
        pass
    if not token or not chat_id:
        return False, "Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to environment variables or Streamlit secrets."
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": message},
            timeout=12,
        )
        response.raise_for_status()
        return True, "Telegram alert sent."
    except Exception as exc:
        return False, f"Telegram error: {exc}"


# --------------------------- Main analysis ---------------------------

if refresh:
    st.cache_data.clear()

try:
    with st.spinner("Loading Binance candles and live order book..."):
        candles, current_price = get_market_data(symbol, market, interval, limit)
        df = add_indicators(candles)
        try:
            order_book = get_order_book(symbol, market)
            book_error = None
        except Exception as exc:
            order_book = None
            book_error = str(exc)

    last = df.iloc[-1]
    support, resistance, support_levels, resistance_levels = find_support_resistance(df, current_price)
    long_score, short_score, checks, reasons, reversal = analyze(df, order_book, support, resistance)
except Exception as exc:
    st.error(f"Could not load/analyze market data: {exc}")
    st.info("Check the symbol, connection and Binance API availability.")
    st.stop()

long_ready = long_score >= min_score and long_score > short_score
short_ready = short_score >= min_score and short_score > long_score
if long_ready:
    signal = "BUY" if market == "Spot" else "LONG"
elif short_ready:
    signal = "NO BUY" if market == "Spot" else "SHORT"
else:
    signal = "WAIT"

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Live Price", f"{current_price:,.4f}")
m2.metric("Long Score", f"{long_score}/100")
m3.metric("Short Score", f"{short_score}/100")
m4.metric("Signal", signal)
m5.metric("RSI", f"{last.rsi:.1f}")
st.caption(f"Last candle: {last.time.strftime('%Y-%m-%d %H:%M UTC')} • Public Binance market data • Scores are confluence points, not win probabilities.")

# --------------------------- Structure + DOM ---------------------------

st.subheader("📍 Support & Resistance")
s1, s2, s3 = st.columns(3)
s1.metric("Nearest Support", f"{support:,.4f}", f"{(current_price-support)/current_price*100:.2f}% below" if current_price else None)
s2.metric("Current Price", f"{current_price:,.4f}")
s3.metric("Nearest Resistance", f"{resistance:,.4f}", f"{(resistance-current_price)/current_price*100:.2f}% above" if current_price else None)

if support_levels:
    st.write("Nearby support levels:", " • ".join(f"{x:,.4f}" for x in support_levels[:5]))
if resistance_levels:
    st.write("Nearby resistance levels:", " • ".join(f"{x:,.4f}" for x in resistance_levels[:5]))

st.subheader("📚 DOM / Order-book Liquidity")
if order_book:
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Bid liquidity (top 20)", f"{order_book['bid_qty']:,.3f}")
    d2.metric("Ask liquidity (top 20)", f"{order_book['ask_qty']:,.3f}")
    d3.metric("Bid/Ask imbalance", f"{order_book['imbalance']:+.1%}")
    d4.metric("Spread", f"{order_book['spread_pct']:.4f}%")
    st.write(
        f"Nearest visible bid wall: **{order_book['bid_wall'][0]:,.4f}** "
        f"(qty {order_book['bid_wall'][1]:,.3f})  |  "
        f"Nearest visible ask wall: **{order_book['ask_wall'][0]:,.4f}** "
        f"(qty {order_book['ask_wall'][1]:,.3f})"
    )
    st.caption("DOM is a live snapshot of displayed limit orders. Orders can be cancelled or moved; visible walls are not guaranteed support/resistance.")
else:
    st.warning(f"Order book unavailable: {book_error}")

st.subheader("🧲 Absorption & Market Reversal")
a1, a2, a3, a4 = st.columns(4)
a1.metric("Bullish absorption", "Possible" if reversal["bullish_absorption"] else "Not detected")
a2.metric("Bearish absorption", "Possible" if reversal["bearish_absorption"] else "Not detected")
a3.metric("Bullish reversal", "Detected" if reversal["bullish_reversal"] else "Not detected")
a4.metric("Bearish reversal", "Detected" if reversal["bearish_reversal"] else "Not detected")
st.caption(
    f"Volume / 20-candle average: {reversal['volume_ratio']:.2f}x • "
    f"Lower wick: {reversal['lower_wick_ratio']:.0%} • Upper wick: {reversal['upper_wick_ratio']:.0%} • "
    f"Buy taker ratio: {last.buy_ratio:.1%}"
)
st.info("Absorption is estimated from candle wicks, relative volume and Binance taker-buy volume. It cannot identify hidden orders or prove institutional activity.")

# --------------------------- Multi-timeframe ---------------------------

st.subheader("🕒 Multi-Timeframe Trend")
mtf_cols = st.columns(3)
for col, tf in zip(mtf_cols, ["15m", "1h", "4h"]):
    try:
        tf_raw, _ = get_market_data(symbol, market, tf, 250)
        tf_df = add_indicators(tf_raw)
        tf_last = tf_df.iloc[-1]
        if tf_last.close > tf_last.ema50 > tf_last.ema200:
            state = "Bullish"
        elif tf_last.close < tf_last.ema50 < tf_last.ema200:
            state = "Bearish"
        else:
            state = "Mixed"
        col.metric(tf, state, f"Close {tf_last.close:,.4f}")
    except Exception:
        col.metric(tf, "Unavailable", "Could not load")

# --------------------------- Trade plan ---------------------------

st.subheader("🎯 Entry Zone • Stop Loss • Take Profit Plan")
atr = float(last.atr) if pd.notna(last.atr) else np.nan
entry_low = entry_high = entry_mid = stop = tp1 = tp2 = tp3 = qty = np.nan
risk_amount = balance * risk_pct / 100
setup_note = "No trade plan: wait for a valid signal."

if signal in ["BUY", "LONG", "SHORT"] and np.isfinite(atr) and atr > 0:
    if signal in ["BUY", "LONG"]:
        # Pullback zone around current price / nearest support; invalidation below structure.
        entry_low = max(support, current_price - 0.35 * atr)
        entry_high = current_price + 0.10 * atr
        if entry_low > entry_high:
            entry_low = current_price - 0.20 * atr
        entry_mid = (entry_low + entry_high) / 2
        stop_anchor = min(support, float(df.low.tail(12).min()))
        stop = stop_anchor - 0.25 * atr
        if stop >= entry_mid:
            stop = entry_mid - 1.5 * atr
        risk_per_unit = entry_mid - stop
        tp1, tp2, tp3 = entry_mid + rr1*risk_per_unit, entry_mid + rr2*risk_per_unit, entry_mid + rr3*risk_per_unit
        setup_note = "Long setup: prefer a pullback into the entry zone, then wait for bullish candle/structure confirmation."
    else:
        entry_low = current_price - 0.10 * atr
        entry_high = min(resistance, current_price + 0.35 * atr)
        if entry_high < entry_low:
            entry_high = current_price + 0.20 * atr
        entry_mid = (entry_low + entry_high) / 2
        stop_anchor = max(resistance, float(df.high.tail(12).max()))
        stop = stop_anchor + 0.25 * atr
        if stop <= entry_mid:
            stop = entry_mid + 1.5 * atr
        risk_per_unit = stop - entry_mid
        tp1, tp2, tp3 = entry_mid - rr1*risk_per_unit, entry_mid - rr2*risk_per_unit, entry_mid - rr3*risk_per_unit
        setup_note = "Short setup: prefer a pullback into the entry zone, then wait for bearish candle/structure confirmation."

    if risk_per_unit > 0:
        qty = risk_amount / risk_per_unit if risk_amount > 0 else 0.0

if np.isfinite(entry_mid):
    e1, e2, e3 = st.columns(3)
    e1.metric("Entry zone — low", f"{entry_low:,.4f}")
    e2.metric("Preferred entry (mid)", f"{entry_mid:,.4f}")
    e3.metric("Entry zone — high", f"{entry_high:,.4f}")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Stop Loss", f"{stop:,.4f}")
    r2.metric(f"TP1 ({rr1:g}R)", f"{tp1:,.4f}")
    r3.metric(f"TP2 ({rr2:g}R)", f"{tp2:,.4f}")
    r4.metric(f"TP3 ({rr3:g}R)", f"{tp3:,.4f}")

    risk_pct_from_entry = abs(entry_mid - stop) / entry_mid * 100
    p1, p2, p3 = st.columns(3)
    p1.metric("Risk budget", f"{risk_amount:,.2f} USDT")
    p2.metric("Stop distance", f"{risk_pct_from_entry:.2f}%")
    p3.metric("Estimated quantity", f"{qty:.6f}")
    st.success(setup_note)
    st.caption("Levels are algorithmic estimates from ATR and recent swing structure. They are not guaranteed execution prices. Recalculate if price moves materially before entry.")
else:
    st.warning(setup_note)

# --------------------------- Charts + scoring ---------------------------

left, right = st.columns([1.7, 1])
with left:
    st.subheader("Price Chart with Levels")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.time, open=df.open, high=df.high, low=df.low, close=df.close, name="Candles"
    ))
    for col_name, label in [("ema21", "EMA 21"), ("ema50", "EMA 50"), ("ema200", "EMA 200"), ("vwap", "VWAP")]:
        fig.add_trace(go.Scatter(x=df.time, y=df[col_name], mode="lines", name=label))
    for level, label in [(support, "Support"), (resistance, "Resistance")]:
        fig.add_hline(y=level, line_dash="dot", annotation_text=label)
    if np.isfinite(stop):
        for level, label in [(entry_low, "Entry low"), (entry_high, "Entry high"), (stop, "Stop"), (tp1, "TP1"), (tp2, "TP2"), (tp3, "TP3")]:
            fig.add_hline(y=level, line_dash="dash", annotation_text=label)
    fig.update_layout(height=620, xaxis_rangeslider_visible=False, legend_orientation="h")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Strategy Confirmation")
    st.dataframe(
        pd.DataFrame(checks, columns=["Condition", "State", "Weight"]),
        hide_index=True, use_container_width=True,
    )
    st.markdown("**Analysis notes**")
    for item in reasons:
        st.write("• " + item)

# --------------------------- Screenshot input ---------------------------

st.subheader("🖼️ Screenshot Analysis Input")
uploaded_chart = st.file_uploader(
    "Upload a chart screenshot (PNG/JPG)",
    type=["png", "jpg", "jpeg"],
)
if uploaded_chart is not None:
    st.image(
        uploaded_chart,
        caption="Uploaded chart reference",
        use_container_width=True,
    )
    st.info(
        "The screenshot is displayed as a reference only. "
        "This version does not automatically interpret chart images."
    )

# --------------------------- Alerts + export ---------------------------

st.subheader("🔔 Telegram Alert")
alert_text = (
    f"AI Trading Analyst PRO\n{symbol} | {market} | {interval}\n"
    f"Signal: {signal} | Long {long_score}/100 | Short {short_score}/100\n"
    f"Price: {current_price}\nSupport: {support} | Resistance: {resistance}\n"
)
if np.isfinite(entry_mid):
    alert_text += (
        f"Entry zone: {entry_low} - {entry_high}\nStop: {stop}\n"
        f"TP1: {tp1} | TP2: {tp2} | TP3: {tp3}\n"
    )
alert_text += "Educational analysis only; verify before trading."
if st.button("Send current analysis to Telegram"):
    ok, message = send_telegram(alert_text)
    st.success(message) if ok else st.warning(message)

st.subheader("📥 Data Export")
csv_data = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download candles + indicators (CSV)", csv_data,
    file_name=f"{symbol}_{interval}_analysis.csv", mime="text/csv",
)

with st.expander("Methodology, assumptions & limitations"):
    st.markdown("""
    - **Entry zone:** an estimated pullback area around the current price and nearest structure.
    - **Stop loss:** placed beyond a recent swing/support/resistance with an ATR buffer.
    - **TP targets:** calculated from configurable risk/reward multiples (R).
    - **Support/resistance:** recent local swing highs/lows; levels are estimates, not guaranteed barriers.
    - **DOM liquidity:** current public order-book snapshot, not a record of executed trades.
    - **Absorption:** heuristic using wick size, relative volume and taker-buy ratio; hidden liquidity cannot be verified from this feed.
    - **Market reversal:** rejection candle and/or sweep-and-reclaim heuristic, not a prediction.
    - Scores are rule-based confluence points, not probabilities or guaranteed win rates.
    - Quantity estimate excludes fees, slippage, funding, leverage, liquidation risk, lot size and contract specifications.
    - No orders are placed and no API trading key is requested.
    - Backtest and paper-trade before risking capital. This tool is educational, not financial advice.
    """)
st.caption("AI Trading Analyst PRO v2.0 • Signal-only tool • Not financial advice")
