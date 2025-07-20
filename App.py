import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
from datetime import date

st.set_page_config(page_title="PredictiTrade 📈", layout="wide")
st.title("📊 PredictiTrade - AI Stock Trend Predictor (US Stocks)")

# Sidebar for user inputs
st.sidebar.header("🔍 Select Stock and Date Range")
ticker = st.sidebar.text_input("Enter US stock ticker (e.g., AAPL, MSFT, TSLA)", "AAPL")

start_date = st.sidebar.date_input("Start Date", value=date(2024, 1, 1))
end_date = st.sidebar.date_input("End Date", value=date.today())

def get_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df.dropna(inplace=True)
    df['Target'] = df['Close'].shift(-1) > df['Close']
    df.dropna(inplace=True)
    return df

if ticker:
    try:
        df = get_data(ticker, start_date, end_date)

        st.subheader(f"📅 Historical Data: {ticker.upper()} ({start_date} to {end_date})")
        st.dataframe(df.tail(), use_container_width=True)

        # Candlestick Chart
        st.subheader("📈 Candlestick Chart")
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Price"
        )])
        fig.update_layout(xaxis_rangeslider_visible=False, height=500)
        st.plotly_chart(fig, use_container_width=True)

        # ML Prediction
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        X = df[features]
        y = df['Target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        st.success(f"✅ Model trained with {acc*100:.2f}% accuracy")

        next_day = model.predict(X.tail(1))[0]
        if next_day:
            st.markdown("### 🔮 Prediction: The stock might go **📈 UP** tomorrow.")
        else:
            st.markdown("### 🔮 Prediction: The stock might go **📉 DOWN** tomorrow.")

    except Exception as e:
        st.error(f"An error occurred: {e}")



