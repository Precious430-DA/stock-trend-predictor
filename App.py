import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
from io import StringIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="PredictiTrade NGX", layout="centered")
st.title("📈 AI Stock Trend Predictor (US & NGX)")

# --- Source toggle
source = st.radio("📊 Choose Market", ["US Stocks", "NGX Stocks"])

# --- Date inputs
start_date = st.date_input("📅 Start date", pd.to_datetime("2024-01-01"))
end_date = st.date_input("📅 End date", pd.to_datetime("today"))
show_candle = st.checkbox("📊 Show Candlestick Chart", value=True)

# --- Get US data (via yfinance)
@st.cache_data
def get_us_data(ticker, start, end):
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end)
    df.dropna(inplace=True)
    df['Target'] = df['Close'].shift(-1) > df['Close']
    df.dropna(inplace=True)
    return df

# --- Get NGX data (scraped from GTI)
@st.cache_data
def get_ngx_data(ticker):
    url = "https://research.gti.com.ng/ngx-daily-price-list/"
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table")
    df = pd.read_html(str(table))[0]
    df.columns = df.columns.str.strip()
    df = df[df["Security"].str.upper() == ticker.upper()]
    df = df.rename(columns={
        "Previous Close": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Volume": "Volume"
    })
    df['Date'] = pd.to_datetime("today")  # Only today's data available
    df.set_index('Date', inplace=True)
    df['Target'] = df['Close'].shift(-1) > df['Close']
    df.dropna(inplace=True)
    return df

# --- Main logic
if source == "US Stocks":
    ticker = st.text_input("Enter US Ticker (e.g. AAPL, MSFT)", "AAPL")
    if ticker:
        try:
            df = get_us_data(ticker, start_date, end_date)
        except Exception as e:
            st.error(f"❌ Error fetching US data: {e}")
            st.stop()
else:
    ticker = st.text_input("Enter NGX Ticker (e.g. FCMB, GTCO)", "FCMB")
    if ticker:
        try:
            df = get_ngx_data(ticker)
        except Exception as e:
            st.error(f"❌ Error fetching NGX data: {e}")
            st.stop()

# --- Show chart
if not df.empty:
    st.subheader(f"📊 Data for {ticker.upper()}")

    if show_candle:
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
        )])
        fig.update_layout(title=f"Candlestick chart for {ticker.upper()}", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(df['Close'])

    st.dataframe(df.tail())

    # --- ML Prediction
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    X = df[features]
    y = df['Target']

    if len(df) < 5:
        st.warning("📉 Not enough data to train ML model.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.success(f"✅ Model trained with {acc*100:.2f}% accuracy")

        next_day = model.predict(X.tail(1))[0]
        st.markdown("### 🔮 Prediction for Tomorrow:")
        if next_day:
            st.markdown("📈 The stock might go UP tomorrow.")
        else:
            st.markdown("📉 The stock might go DOWN tomorrow.")

# --- Simple Chat/Feedback
st.markdown("---")
st.markdown("🗨️ Ask or Leave a Comment")
user_message = st.text_input("💬 Type your message here:")
if user_message:
    st.info("Thanks! We'll use your feedback to improve.")

# --- Footer
st.markdown("---")
st.markdown("🧠 Powered by [GTI NGX](https://gtiportal.com), [scikit-learn](https://scikit-learn.org/), and [Streamlit](https://streamlit.io/)")
st.markdown("💻 Made by Precious Ofoyekpene")
