# Stock Signal Analyzer

Web-based stock trading signal app using **MACD**, **RSI(6)**, and **Elliott Wave** analysis.

## Features

- **Live data** from Yahoo Finance (stocks, ETFs, crypto tickers)
- **MACD (12, 26, 9)** crossover detection
- **RSI with period 6** — oversold/overbought zones
- **Simplified Elliott Wave** swing detection and wave labelling
- **Composite scoring** — combines all three indicators into BUY / SELL / HOLD
- Interactive candlestick, MACD, and RSI charts with buy/sell markers
- Auto-refresh mode (60-second interval)

## Signal Logic

| Indicator | +1 (Bullish) | -1 (Bearish) |
|-----------|-------------|--------------|
| MACD | Bullish crossover | Bearish crossover |
| RSI(6) | Below 30 (oversold) | Above 70 (overbought) |
| Elliott Wave | Bullish bias | Bearish bias |

- **Score >= 2** → BUY
- **Score <= -2** → SELL
- **Otherwise** → HOLD

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000 in your browser.

## Disclaimer

Signals are for **informational/educational purposes only** — not financial advice.
