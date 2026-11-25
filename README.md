# AI Trading Analyst PRO – v1.0

A Streamlit starter app for Binance Spot and USDⓈ-M Futures market analysis.

## Included
- Binance public candles (no API key needed)
- Spot and Futures market selection
- RSI, SMA, EMA 9/21/50/200, ATR, VWAP, TSI
- Basic pivot support/resistance and BOS/FVG-style SMC proxies
- Heuristic divergence and momentum checks
- Weighted confluence scoring (scores are not win probabilities)
- ATR-based illustrative risk sizing and 2R/3R levels
- Screenshot upload/display (not automatic vision interpretation yet)
- Optional Telegram alert via environment variables
- CSV export

## Run locally
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Deploy with GitHub + Streamlit Community Cloud
1. Create a GitHub repository named `binance-ai-trading-analyzer`.
2. Upload `app.py`, `requirements.txt`, and `README.md` to the repository root.
3. In Streamlit Community Cloud, choose **New app**, select the repository, branch `main`, and main file `app.py`.
4. Deploy. The app is designed for browser use on Android and Windows.

## Optional Telegram alerts
Create a Telegram bot using BotFather and obtain your chat ID. In Streamlit app settings, add these secrets:
```toml
TELEGRAM_BOT_TOKEN = "your_bot_token"
TELEGRAM_CHAT_ID = "your_chat_id"
```
The app reads these from Streamlit Secrets (or environment variables). Do not commit secrets to GitHub.

## Safety
- This is a signal-only prototype; it does not place trades.
- Never enable withdrawal permissions on exchange API keys. This version does not need an exchange key.
- 1% is the configured risk budget, not a guarantee that realized loss will be limited to exactly 1%.
- Scores are confluence points, not probabilities. No 90% win rate is claimed.
- Validate with realistic fees, slippage, funding, and out-of-sample backtests before live trading.
