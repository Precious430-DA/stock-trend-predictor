import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="PredictiTrade", layout="centered")
st.title("📈 AI Stock Trend Predictor (US Stocks)")

ticker = st.text_input("Enter US stock ticker (e.g., AAPL, MSFT, TSLA)", "AAPL")

def get_data(ticker):
    df = yf.download(ticker, period="6mo")
    df.dropna(inplace=True)
    df['Target'] = df['Close'].shift(-1) > df['Close']
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA100'] = df['Close'].rolling(window=100).mean()

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df.dropna(inplace=True)
    return df

if ticker:
    try:
        df = get_data(ticker)

        st.subheader(f"📊 Closing Price Chart for {ticker.upper()}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], mode='lines', name='SMA50'))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA100'], mode='lines', name='SMA100'))
        fig.update_layout(title=f"{ticker.upper()} Price with SMA", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig)

        st.subheader("📐 Technical Indicators")
        rsi_fig = px.line(df, x=df.index, y='RSI', title='Relative Strength Index (RSI)')
        rsi_fig.add_hline(y=70, line_dash="dot", line_color="red")
        rsi_fig.add_hline(y=30, line_dash="dot", line_color="green")
        st.plotly_chart(rsi_fig)

        st.subheader("🧾 Raw Data")
        st.dataframe(df.tail())

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

        st.markdown("---")
        st.markdown("🧠 Powered by [yfinance](https://pypi.org/project/yfinance/), [scikit-learn](https://scikit-learn.org/), and [Streamlit](https://streamlit.io/)")
        st.markdown("💻 Made by Precious Ofoyekpene")

    except Exception as e:
        st.error(f"An error occurred: {e}")
