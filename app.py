import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time

st.set_page_config(page_title="NSE 500 Scanner", layout="wide")

st.title("📈 NSE 500 Broad Bullish Scanner")

# ==============================
# NSE 500 List
# ==============================
@st.cache_data
def get_nse500_stocks():
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    df = pd.read_csv(url)
    symbols = df['Symbol'].tolist()
    return [symbol + ".NS" for symbol in symbols]

# ==============================
# RSI
# ==============================
def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# ==============================
# MACD
# ==============================
def calculate_macd(data):
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

# ==============================
# Scanner
# ==============================
def scan_market():

    stocks = get_nse500_stocks()
    results = []

    progress = st.progress(0)
    total = len(stocks)

    for i, stock in enumerate(stocks):
        try:
            data = yf.download(
                stock,
                period="6mo",
                interval="1d",
                auto_adjust=True,
                progress=False
            )

            if data.empty or len(data) < 60:
                continue

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            data = data.dropna()

            data['20_DMA'] = data['Close'].rolling(20).mean()
            data['50_DMA'] = data['Close'].rolling(50).mean()
            data['RSI'] = calculate_rsi(data)
            data['MACD'], data['Signal'] = calculate_macd(data)
            data['Avg_Volume'] = data['Volume'].rolling(20).mean()

            latest = data.iloc[-1]

            close = float(latest['Close'])
            dma20 = float(latest['20_DMA'])
            dma50 = float(latest['50_DMA'])
            rsi = float(latest['RSI'])
            macd = float(latest['MACD'])
            signal = float(latest['Signal'])
            volume = float(latest['Volume'])
            avg_volume = float(latest['Avg_Volume'])

            if (
                close > dma20 and
                rsi > 48 and
                macd > signal and
                volume > 0.9 * avg_volume
            ):

                trend_strength = (dma20 - dma50) / dma50 if dma50 != 0 else 0
                score = (rsi / 100) + (volume / avg_volume) + trend_strength

                results.append({
                    "Stock": stock,
                    "Price": round(close, 2),
                    "RSI": round(rsi, 2),
                    "Score": round(score, 3)
                })

            progress.progress((i + 1) / total)

        except:
            continue

    return pd.DataFrame(results).sort_values(by="Score", ascending=False)


# ==============================
# BUTTON
# ==============================

if st.button("🚀 Run Scanner"):

    with st.spinner("Scanning NSE 500 stocks..."):
        df = scan_market()

    st.success(f"Found {len(df)} bullish stocks")

    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name="nse500_scan.csv",
        mime="text/csv",
    )
