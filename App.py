import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI

# Setup OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Streamlit page config
st.set_page_config(page_title="PredictiTrade", layout="wide")
st.title("📈 AI Stock Trend Predictor (US Stocks)")

# Sidebar
st.sidebar.header("📊 Configuration")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g. AAPL)", value="AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))

# Load data
@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, start=start_date, end=end_date)
    df["Return"] = df["Close"].pct_change()
    df["Target"] = np.where(df["Return"].shift(-1) > 0, 1, 0)
    return df.dropna()

df = load_data(ticker)

# Feature engineering
features = df[["Open", "High", "Low", "Close", "Volume"]]
target = df["Target"]
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)

# Show accuracy
st.subheader("📌 Model Accuracy")
st.metric("Accuracy", f"{acc:.2%}")

# Plot close price
st.subheader(f"📉 {ticker} Closing Price")
fig, ax = plt.subplots()
df["Close"].plot(ax=ax, label="Close Price")
ax.set_ylabel("Price")
ax.set_title(f"{ticker} Price Chart")
st.pyplot(fig)

# Predict next day
st.subheader("🔮 Prediction for Next Day")
latest_data = features.iloc[-1:].values.reshape(1, -1)
prediction = model.predict(latest_data)[0]
trend = "📈 Up" if prediction == 1 else "📉 Down"
st.success(f"Predicted Trend: {trend}")

# ChatGPT integration
st.subheader("💬 Ask ChatGPT About Markets")

user_prompt = st.text_area("Ask anything about trading or stocks:")
if user_prompt:
    with st.spinner("ChatGPT is thinking..."):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful trading assistant."},
                {"role": "user", "content": user_prompt}
            ]
        )
        st.write(response.choices[0].message.content)
