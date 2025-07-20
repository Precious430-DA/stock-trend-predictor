import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="PredictiTrade", layout="centered")
st.title("📈 AI Stock Trend Predictor (US Stocks)")

ticker = st.text_input("Enter US stock ticker (e.g., AAPL, MSFT, TSLA)", "AAPL")

def get_data(ticker):
    df = yf.download(ticker, period="6mo")
    df.dropna(inplace=True)
    df['Target'] = df['Close'].shift(-1) > df['Close']
    df.dropna(inplace=True)
    return df

if ticker:
    try:
        df = get_data(ticker)

        st.subheader(f"📊 Latest Data for {ticker.upper()}")
        st.line_chart(df['Close'])

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
        st.markdown(" MADE BY Precious Ofoyekpene")

    except Exception as e:
        st.error(f"An error occurred: {e}")


