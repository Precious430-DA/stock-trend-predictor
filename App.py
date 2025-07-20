import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import datetime

# --- APP TITLE ---
st.set_page_config(page_title="PredictiTrade", layout="wide")
st.title("📈 PredictiTrade - Stock Trend Predictor")

# --- SIDEBAR ---
st.sidebar.header("📌 Configuration")

ticker = st.sidebar.text_input("Enter Stock Ticker (e.g. AAPL, TSLA, MSFT)", value="AAPL").upper()
start_date = st.sidebar.date_input("Start Date", datetime.date(2022, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())
show_candlestick = st.sidebar.checkbox("Show Candlestick Chart", value=False)
show_sma50 = st.sidebar.checkbox("Show SMA50")
show_sma100 = st.sidebar.checkbox("Show SMA100")
show_rsi = st.sidebar.checkbox("Show RSI")

# --- DOWNLOAD DATA ---
@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
    df.reset_index(inplace=True)
    return df

df = load_data(ticker, start_date, end_date)

if df.empty:
    st.error("No data found for the selected ticker and date range.")
    st.stop()

# --- CALCULATE INDICATORS ---
df['SMA50'] = df['Close'].rolling(window=50).mean()
df['SMA100'] = df['Close'].rolling(window=100).mean()

# RSI function
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['RSI'] = calculate_rsi(df)

# --- DISPLAY PRICE CHART ---
st.subheader(f"📊 {ticker} Stock Price Chart")
fig = go.Figure()

if show_candlestick:
    fig.add_trace(go.Candlestick(x=df['Date'],
                                 open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'],
                                 name='Candlestick'))
else:
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Close Price", line=dict(color='blue')))

if show_sma50:
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA50'], name="SMA50", line=dict(color='orange')))

if show_sma100:
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA100'], name="SMA100", line=dict(color='green')))

fig.update_layout(xaxis_title="Date", yaxis_title="Price", xaxis_rangeslider_visible=False, height=500)
st.plotly_chart(fig, use_container_width=True)

# --- RSI CHART BELOW ---
if show_rsi:
    st.subheader("📐 RSI Indicator")
    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name="RSI", line=dict(color='purple')))
    rsi_fig.add_hline(y=70, line_dash="dot", line_color="red")
    rsi_fig.add_hline(y=30, line_dash="dot_
