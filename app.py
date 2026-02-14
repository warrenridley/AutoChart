"""
Stock Trading Signal App
Uses MACD, RSI(6), and Elliott Wave analysis to generate buy/sell signals.
"""

from flask import Flask, render_template, jsonify, request
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import traceback

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Technical Indicators
# ---------------------------------------------------------------------------

def compute_macd(close: pd.Series, fast=12, slow=26, signal=9):
    """Compute MACD line, signal line, and histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_rsi(close: pd.Series, period=6):
    """Compute RSI with the given period (default 6)."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ---------------------------------------------------------------------------
# Elliott Wave Detection (simplified swing-based approach)
# ---------------------------------------------------------------------------

def detect_swings(close: pd.Series, order=5):
    """
    Detect local swing highs and lows using a rolling window comparison.
    `order` is half the window size used to compare neighbours.
    Returns lists of dicts with index, price, and type ('H' or 'L').
    """
    highs = []
    lows = []
    arr = close.values
    for i in range(order, len(arr) - order):
        if arr[i] == max(arr[i - order: i + order + 1]):
            highs.append({"idx": i, "price": float(arr[i]), "type": "H"})
        if arr[i] == min(arr[i - order: i + order + 1]):
            lows.append({"idx": i, "price": float(arr[i]), "type": "L"})
    return highs, lows


def classify_elliott_wave(close: pd.Series, order=5):
    """
    Simplified Elliott Wave classifier.

    Looks at the most recent sequence of swing highs/lows and attempts to
    map them onto the classic 5-3 Elliott pattern:
        Impulse:  1(up) 2(down) 3(up) 4(down) 5(up)
        Corrective: A(down) B(up) C(down)

    Returns:
        wave_label  – e.g. "Wave 3 (Impulse)" or "Wave B (Corrective)"
        wave_bias   – "bullish" / "bearish" / "neutral"
        swings      – list of swing points used for chart overlay
    """
    highs, lows = detect_swings(close, order)

    # Merge and sort swings chronologically
    swings = sorted(highs + lows, key=lambda s: s["idx"])

    # De-duplicate consecutive same-type swings (keep most extreme)
    filtered = []
    for s in swings:
        if filtered and filtered[-1]["type"] == s["type"]:
            if s["type"] == "H" and s["price"] > filtered[-1]["price"]:
                filtered[-1] = s
            elif s["type"] == "L" and s["price"] < filtered[-1]["price"]:
                filtered[-1] = s
        else:
            filtered.append(s)

    swings = filtered

    # We need at least 6 alternating swings to attempt labelling
    if len(swings) < 4:
        return "Indeterminate", "neutral", swings

    # Take the last 8 swings (enough for 5-wave impulse + start of correction)
    recent = swings[-8:] if len(swings) >= 8 else swings

    # Determine overall trend of recent swings
    first_price = recent[0]["price"]
    last_price = recent[-1]["price"]
    trend_up = last_price > first_price

    # Count alternating moves
    moves = []
    for i in range(1, len(recent)):
        moves.append(recent[i]["price"] - recent[i - 1]["price"])

    # Try to map to impulse wave position
    up_moves = [m for m in moves if m > 0]
    down_moves = [m for m in moves if m < 0]

    wave_label = "Indeterminate"
    wave_bias = "neutral"

    n = len(moves)

    if trend_up:
        # Bullish impulse mapping
        if n >= 7:
            # Could be in corrective phase after 5-wave impulse
            wave_label = "Wave C (Corrective)"
            wave_bias = "bearish"  # correction is bearish within bull trend
        elif n >= 5:
            # Likely completing wave 5 or starting correction
            if moves[-1] < 0:
                wave_label = "Wave A (Corrective)"
                wave_bias = "bearish"
            else:
                wave_label = "Wave 5 (Impulse)"
                wave_bias = "bullish"
        elif n >= 3:
            if moves[-1] > 0:
                wave_label = "Wave 3 (Impulse)"
                wave_bias = "bullish"
            else:
                wave_label = "Wave 4 (Impulse)"
                wave_bias = "neutral"
        elif n >= 1:
            if moves[-1] > 0:
                wave_label = "Wave 1 (Impulse)"
                wave_bias = "bullish"
            else:
                wave_label = "Wave 2 (Impulse)"
                wave_bias = "neutral"
    else:
        # Bearish / downtrend mapping (mirror)
        if n >= 7:
            wave_label = "Wave C (Corrective)"
            wave_bias = "bullish"  # correction within bear trend can bounce
        elif n >= 5:
            if moves[-1] > 0:
                wave_label = "Wave A (Corrective)"
                wave_bias = "bullish"
            else:
                wave_label = "Wave 5 (Impulse Down)"
                wave_bias = "bearish"
        elif n >= 3:
            if moves[-1] < 0:
                wave_label = "Wave 3 (Impulse Down)"
                wave_bias = "bearish"
            else:
                wave_label = "Wave 4 (Impulse Down)"
                wave_bias = "neutral"
        elif n >= 1:
            if moves[-1] < 0:
                wave_label = "Wave 1 (Impulse Down)"
                wave_bias = "bearish"
            else:
                wave_label = "Wave 2 (Impulse Down)"
                wave_bias = "neutral"

    return wave_label, wave_bias, swings


# ---------------------------------------------------------------------------
# Combined Signal Engine
# ---------------------------------------------------------------------------

def generate_signals(df: pd.DataFrame):
    """
    Generate BUY / SELL / HOLD for each bar based on:
        - MACD crossover
        - RSI(6) overbought / oversold
        - Elliott Wave bias

    Scoring system (per bar):
        +1  MACD bullish crossover (MACD crosses above signal)
        -1  MACD bearish crossover
        +1  RSI < 30  (oversold → buy pressure)
        -1  RSI > 70  (overbought → sell pressure)
        +1  Elliott wave bias == bullish
        -1  Elliott wave bias == bearish

    Score >=  2  →  BUY
    Score <= -2  →  SELL
    Else         →  HOLD
    """
    close = df["Close"]

    macd_line, signal_line, histogram = compute_macd(close)
    rsi = compute_rsi(close, period=6)
    wave_label, wave_bias, swings = classify_elliott_wave(close)

    # Elliott bias score (same for all bars – it's a regime label)
    ew_score = {"bullish": 1, "bearish": -1, "neutral": 0}.get(wave_bias, 0)

    signals = []
    scores_list = []
    for i in range(1, len(df)):
        score = 0

        # MACD crossover
        if macd_line.iloc[i] > signal_line.iloc[i] and macd_line.iloc[i - 1] <= signal_line.iloc[i - 1]:
            score += 1  # bullish crossover
        elif macd_line.iloc[i] < signal_line.iloc[i] and macd_line.iloc[i - 1] >= signal_line.iloc[i - 1]:
            score -= 1  # bearish crossover

        # RSI
        r = rsi.iloc[i]
        if not np.isnan(r):
            if r < 30:
                score += 1
            elif r > 70:
                score -= 1

        # Elliott Wave
        score += ew_score

        if score >= 2:
            signals.append("BUY")
        elif score <= -2:
            signals.append("SELL")
        else:
            signals.append("HOLD")
        scores_list.append(score)

    # Pad first row
    signals = ["HOLD"] + signals
    scores_list = [0] + scores_list

    df = df.copy()
    df["MACD"] = macd_line
    df["MACD_Signal"] = signal_line
    df["MACD_Hist"] = histogram
    df["RSI"] = rsi
    df["Signal"] = signals
    df["Score"] = scores_list

    return df, wave_label, wave_bias, swings


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_stock_data(symbol: str, period: str = "6mo", interval: str = "1d"):
    """Fetch historical data from Yahoo Finance."""
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    if df.empty:
        return None, None
    info = {}
    try:
        fast_info = ticker.fast_info
        info = {
            "name": getattr(fast_info, "short_name", symbol) if hasattr(fast_info, "short_name") else symbol,
            "currency": getattr(fast_info, "currency", "USD"),
            "exchange": getattr(fast_info, "exchange", "N/A"),
            "price": float(df["Close"].iloc[-1]),
        }
    except Exception:
        info = {
            "name": symbol,
            "currency": "USD",
            "exchange": "N/A",
            "price": float(df["Close"].iloc[-1]),
        }
    return df, info


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["GET"])
def analyze():
    symbol = request.args.get("symbol", "AAPL").upper().strip()
    period = request.args.get("period", "6mo")
    interval = request.args.get("interval", "1d")

    try:
        df, info = fetch_stock_data(symbol, period, interval)
        if df is None:
            return jsonify({"error": f"No data found for {symbol}"}), 404

        df, wave_label, wave_bias, swings = generate_signals(df)

        # Build JSON-serialisable response
        dates = [d.strftime("%Y-%m-%d %H:%M") for d in df.index]
        ohlc = {
            "dates": dates,
            "open": [round(float(v), 2) for v in df["Open"]],
            "high": [round(float(v), 2) for v in df["High"]],
            "low": [round(float(v), 2) for v in df["Low"]],
            "close": [round(float(v), 2) for v in df["Close"]],
            "volume": [int(v) for v in df["Volume"]],
        }

        indicators = {
            "macd": [round(float(v), 4) if not np.isnan(v) else None for v in df["MACD"]],
            "macd_signal": [round(float(v), 4) if not np.isnan(v) else None for v in df["MACD_Signal"]],
            "macd_hist": [round(float(v), 4) if not np.isnan(v) else None for v in df["MACD_Hist"]],
            "rsi": [round(float(v), 2) if not np.isnan(v) else None for v in df["RSI"]],
        }

        # Latest signal info
        latest_signal = df["Signal"].iloc[-1]
        latest_score = int(df["Score"].iloc[-1])
        latest_rsi = round(float(df["RSI"].iloc[-1]), 2) if not np.isnan(df["RSI"].iloc[-1]) else None
        latest_macd = round(float(df["MACD"].iloc[-1]), 4)
        latest_macd_sig = round(float(df["MACD_Signal"].iloc[-1]), 4)

        # Recent signals (last 30 bars)
        recent_signals = []
        for i in range(max(0, len(df) - 30), len(df)):
            if df["Signal"].iloc[i] != "HOLD":
                recent_signals.append({
                    "date": dates[i],
                    "signal": df["Signal"].iloc[i],
                    "price": round(float(df["Close"].iloc[i]), 2),
                    "score": int(df["Score"].iloc[i]),
                })

        # Signal markers for chart
        buy_markers = {"dates": [], "prices": []}
        sell_markers = {"dates": [], "prices": []}
        for i in range(len(df)):
            if df["Signal"].iloc[i] == "BUY":
                buy_markers["dates"].append(dates[i])
                buy_markers["prices"].append(round(float(df["Close"].iloc[i]), 2))
            elif df["Signal"].iloc[i] == "SELL":
                sell_markers["dates"].append(dates[i])
                sell_markers["prices"].append(round(float(df["Close"].iloc[i]), 2))

        # Elliott wave swing points for chart overlay
        swing_points = []
        for s in swings:
            idx = s["idx"]
            if idx < len(dates):
                swing_points.append({
                    "date": dates[idx],
                    "price": round(s["price"], 2),
                    "type": s["type"],
                })

        return jsonify({
            "symbol": symbol,
            "info": info,
            "ohlc": ohlc,
            "indicators": indicators,
            "latest": {
                "signal": latest_signal,
                "score": latest_score,
                "rsi": latest_rsi,
                "macd": latest_macd,
                "macd_signal": latest_macd_sig,
                "price": round(float(df["Close"].iloc[-1]), 2),
            },
            "elliott_wave": {
                "label": wave_label,
                "bias": wave_bias,
                "swings": swing_points,
            },
            "buy_markers": buy_markers,
            "sell_markers": sell_markers,
            "recent_signals": recent_signals,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
