import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Technical indicators
def calculate_technical_indicators(df):
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

# Page Config
st.set_page_config(page_title="PredictiTrade", layout="centered")
st.title("📈 AI Stock Trend Predictor (US Stocks)")

# Sidebar: User Inputs
ticker = st.text_input("Enter US stock ticker (e.g., AAPL, MSFT, TSLA)", "AAPL")
start_date = st.date_input("📅 Start date", pd.to_datetime("2024-01-01"))
end_date = st.date_input("📅 End date", pd.to_datetime("today"))

show_tech = st.checkbox("📐 Show Technical Indicators", value=True)

# Get Data
@st.cache_data
def get_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df.dropna(inplace=True)
    df['Target'] = df['Close'].shift(-1) > df['Close']
    df = calculate_technical_indicators(df)
    df.dropna(inplace=True)
    return df

if ticker:
    try:
        df = get_data(ticker, start_date, end_date)

        st.subheader(f"📊 Data for {ticker.upper()} from {start_date} to {end_date}")

        # Show technical chart
        if show_tech:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], mode='lines', name='SMA50'))
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], mode='lines', name='SMA200'))
            fig.update_layout(title=f"{ticker.upper()} - Close Price with SMA50 and SMA200",
                              xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)

            # RSI chart
            rsi_fig = go.Figure()
            rsi_fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI', line=dict(color='orange')))
            rsi_fig.update_layout(title="RSI (Relative Strength Index)", yaxis_title="RSI", xaxis_title="Date")
            st.plotly_chart(rsi_fig, use_container_width=True)
        else:
            st.line_chart(df['Close'])

        st.dataframe(df.tail())

        # ML Part
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
        if next_day:
            st.markdown("📈 The stock might go **UP** tomorrow.")
        else:
            st.markdown("📉 The stock might go **DOWN** tomorrow.")

        # Simple Chat Box
        st.markdown("---")
        st.markdown("🗨️ **Ask or Leave a Comment**")
        user_message = st.text_input("💬 Type your message here:")
        if user_message:
            st.write("🤖 Bot Response:")
            st.info("Thanks for your message! We will improve your experience soon. 🙌")

        # Footer
        st.markdown("---")
