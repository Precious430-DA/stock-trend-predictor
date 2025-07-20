import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI

# Page config
st.set_page_config(page_title="PredictiTrade", layout="wide")
st.title("📈 AI Stock Trend Predictor")

# Sidebar - User Inputs
ticker = st.sidebar.text_input("Enter US Stock Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-01-01"))
add_chatbot = st.sidebar.checkbox("💬 Enable AI Assistant")

# Fetch stock data
df = yf.download(ticker, start=start_date, end=end_date)
st.subheader(f"Raw Data for {ticker}")
st.dataframe(df.tail())

# Feature Engineering
df['SMA_5'] = df['Close'].rolling(window=5).mean()
df['SMA_10'] = df['Close'].rolling(window=10).mean()
df['Return'] = df['Close'].pct_change()
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df.dropna(inplace=True)

# Train/Test Split
X = df[['Close', 'SMA_5', 'SMA_10', 'Return']]
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

# Model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)
st.subheader("📊 Prediction Accuracy")
st.write(f"{acc * 100:.2f}%")

# Plot
st.subheader("📉 Stock Closing Price")
plt.figure(figsize=(10, 4))
plt.plot(df['Close'])
st.pyplot(plt)

# Prediction
latest_data = X.tail(1)
trend = model.predict(latest_data)[0]
st.subheader("📍 Prediction for Next Day:")
st.write("⬆️ Price likely to go **UP**" if trend else "⬇️ Price likely to go **DOWN**")

# Optional GPT Assistant
if add_chatbot:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧠 GPT Assistant")

    prompt = st.sidebar.text_area("Ask me anything about markets:")
    if st.sidebar.button("Ask GPT"):
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        st.sidebar.markdown("**GPT Response:**")
        st.sidebar.write(response.choices[0].message.content)
