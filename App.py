import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Page setup
st.set_page_config(page_title="📈 AI Stock Trend Predictor (US Stocks)")
st.title("📈 AI Stock Trend Predictor (US Stocks)")
st.markdown("Enter US stock ticker (e.g., AAPL, MSFT, TSLA)")

# Input
ticker = st.text_input("Stock Ticker", value="AAPL")

if ticker:
    st.write(f"Latest data for **{ticker.upper()}**")

    # Load data
    df = yf.download(ticker.upper(), period="6mo")

    if df.empty:
        st.error("Failed to fetch data. Please try another ticker.")
    else:
        # Feature Engineering
        df['Tomorrow Close'] = df['Close'].shift(-1)
        df['Target'] = (df['Tomorrow Close'] > df['Close']).astype(int)
        df.dropna(inplace=True)

        df['Open-Close'] = df['Open'] - df['Close']
        df['High-Low'] = df['High'] - df['Low']

        features = ['Open-Close', 'High-Low', 'Volume']
        X = df[features]
        y = df['Target']

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

        # Model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate
        accuracy = accuracy_score(y_test, model.predict(X_test))
        st.success(f"✅ Model trained with {round(accuracy * 100, 2)}% accuracy")

        # Predict tomorrow
        latest_data = X.tail(1)
        prediction = model.predict(latest_data)[0]

        if prediction == 1:
            st.markdown("📈 **Prediction: The stock might go UP tomorrow.**")
        else:
            st.markdown("📉 **Prediction: The stock might go DOWN tomorrow.**")
