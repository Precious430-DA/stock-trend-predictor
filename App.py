import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go

st.set_page_config(page_title="PredictiTrade", layout="centered")
st.title("📈 AI Stock Trend Predictor (US Stocks)")

ticker = st.text_input("Enter US stock ticker (e.g., AAPL, MSFT, TSLA)", "AAPL")

def get_data(ticker):
    df = yf.download(ticker, period="6mo")
    df.dropna(inplace=True)
    df['Target'] = df['Close'].shift(-1) > df['Close']
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df.dropna(inplace=True)
    return df

if ticker:
    try:
        df = get_data(ticker)

        st.subheader(f"📊 Price Trend for {ticker.upper()}")

        # Plotting with Plotly (Line chart)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], mode='lines', name='SMA50'))
        fig.update_layout(title=f"{ticker.upper()} Price & SMA50", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)

        # RSI chart
        st.subheader("📐 Technical Indicators")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'))
        fig_rsi.update_layout(title="RSI (Relative Strength Index)", xaxis_title="Date", yaxis_title="RSI Value")
        st.plotly_chart(fig_rsi, use_container_width=True)

        st.dataframe(df[['Close', 'SMA50', 'RSI']].tail())

        # Model training
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

    except Exception as e:
        st.error(f"An error occurred: {e}")

    st.markdown("---")
    st.markdown("🧠 Powered by [yfinance](https://pypi.org/project/yfinance/), [scikit-learn](https://scikit-learn.org/), and [Streamlit](https://streamlit.io/)")
    st.markdown("💻 Made by Precious Ofoyekpene")
