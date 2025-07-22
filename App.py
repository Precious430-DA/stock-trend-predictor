import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta

st.set_page_config(page_title="PredictiTrade", layout="wide")

st.title("📈 PredictiTrade: AI-Powered Stock Trend Predictor")

# Sidebar Inputs
ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", value=datetime.now())

# Download Data
try:
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if data.empty or len(data) < 50:
        st.error("❌ Unable to load sufficient data. Check the ticker and date range.")
        st.stop()
except Exception as e:
    st.error(f"Error downloading data: {e}")
    st.stop()

# Feature Engineering
data['Return'] = data['Close'].pct_change()
data['Target'] = (data['Return'].shift(-1) > 0).astype(int)

# Drop rows with NaN
data.dropna(inplace=True)

# Features
features = ['Open', 'High', 'Low', 'Close', 'Volume']
X = data[features]
y = data['Target']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

# Display
st.subheader(f"Model Accuracy: {accuracy * 100:.2f}%")

# Add predictions to DataFrame
data = data.iloc[-len(y_test):]
data['Prediction'] = predictions

# Candlestick Chart
fig = go.Figure(data=[go.Candlestick(x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'])])
fig.update_layout(title=f"{ticker} Candlestick Chart", xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# Prediction Result Table
st.subheader("📊 Prediction Results")
st.dataframe(data[['Open', 'High', 'Low', 'Close', 'Prediction']].tail(10))

# Footer
st.markdown("""
<style>
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)
