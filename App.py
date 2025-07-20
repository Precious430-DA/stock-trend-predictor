import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
import numpy as np

# Streamlit page config
st.set_page_config(page_title="PredictiTrade", layout="wide")
st.title("📈 PredictiTrade - AI Stock Trend Predictor")

# User input
ticker = st.text_input("Enter US stock ticker (e.g., AAPL, MSFT, TSLA)", "AAPL")
start_date = st.date_input("Start Date", pd.to_datetime("2024-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("today"))

# Data fetch
def get_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df.dropna(inplace=True)
    df['Target'] = df['Close'].shift(-1) > df['Close']
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df.dropna(inplace=True)
    return df

# Main execution
if ticker:
    try:
        df = get_data(ticker, start_date, end_date)

        # Chart type selection
        chart_type = st.selectbox("Select Chart Type", ["Line", "Candlestick"])
        st.subheader(f"📊 Price Chart for {ticker.upper()}")

        if chart_type == "Line":
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close'))
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], name='SMA20'))
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name='SMA50'))
            fig.update_layout(title=f"{ticker.upper()} Closing Price with SMA",
                              xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Candlestick":
            fig = go.Figure(data=[go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close']
            )])
            fig.update_layout(title=f"{ticker.upper()} Candlestick Chart", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)

        # RSI Chart
        st.subheader("📉 RSI (Relative Strength Index)")
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI"))
        rsi_fig.update_layout(yaxis_range=[0, 100])
        st.plotly_chart(rsi_fig, use_container_width=True)

        # Raw data table
        st.dataframe(df.tail())

        # ML Prediction
        st.subheader("🧠 Model Prediction")
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
        st.markdown("### 🔮 Prediction for Tomorrow:")
        st.markdown("📈 The stock might go **UP** tomorrow." if next_day else "📉 The stock might go **DOWN** tomorrow.")

        # Chat simulation
        st.markdown("### 💬 Chat/Comment Box")
        user_msg = st.text_area("Type your thoughts or questions...")
        if st.button("Submit"):
            st.info("🧠 Thanks for your input! This chat will become smarter soon.")

        # Footer
        st.markdown("---")
        st.markdown("📚 Powered by [yfinance](https://pypi.org/project/yfinance/), [scikit-learn](https://scikit-learn.org/), and [Streamlit](https://streamlit.io/)")
        st.markdown("💻 Created by Precious Ofoyekpene")

    except Exception as e:
        st.error(f"An error occurred: {e}")
