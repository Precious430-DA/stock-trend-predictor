import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Page Config
st.set_page_config(page_title="PredictiTrade", layout="centered")
st.title("📈 AI Stock Trend Predictor (US Stocks)")

# Sidebar Inputs
ticker = st.text_input("Enter US stock ticker (e.g., AAPL, MSFT, TSLA)", "AAPL")
start_date = st.date_input("📅 Start date", pd.to_datetime("2024-01-01"))
end_date = st.date_input("📅 End date", pd.to_datetime("today"))
show_tech_indicators = st.checkbox("🧠 Show Technical Indicators", value=True)

@st.cache_data
def get_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df.dropna(inplace=True)
    df['Target'] = df['Close'].shift(-1) > df['Close']
    df.dropna(inplace=True)
    return df

# Technical Indicator Functions
def add_technical_indicators(df):
    df['SMA50'] = df['Close'].rolling(window=50).mean()

    # RSI Calculation
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

if ticker:
    try:
        df = get_data(ticker, start_date, end_date)
        df = add_technical_indicators(df)

        st.subheader(f"📊 {ticker.upper()} Price Data from {start_date} to {end_date}")
        st.line_chart(df['Close'])  # <-- Fixed here by passing a 1D Series

        if show_tech_indicators:
            st.markdown("### 📐 Technical Indicators")
            st.line_chart(df[['Close', 'SMA50']])
            st.line_chart(df['RSI'])

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
        st.info("📈 UP" if next_day else "📉 DOWN")

        # Chat box
        st.markdown("---")
        st.markdown("🗨️ **Ask or Leave a Comment**")
        user_message = st.text_input("💬 Type your message here:")
        if user_message:
            st.write("🤖 Bot Response:")
            st.info("Thanks for your message! We’ll get better every day. 🙌")

        # Footer
        st.markdown("---")
        st.markdown("🧠 Powered by [yfinance](https://pypi.org/project/yfinance/), [scikit-learn](https://scikit-learn.org/), and [Streamlit](https://streamlit.io/)")
        st.markdown("💻 Made by Precious Ofoyekpene")

    except Exception as e:
        st.error(f"An error occurred: {e}")
