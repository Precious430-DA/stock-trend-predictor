import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import openai

# Page Config
st.set_page_config(page_title="PredictiTrade", layout="centered")
st.title("📈 AI Stock Trend Predictor (US Stocks)")

# Sidebar: User Inputs
ticker = st.text_input("Enter US stock ticker (e.g., AAPL, MSFT, TSLA)", "AAPL")
start_date = st.date_input("📅 Start date", pd.to_datetime("2024-01-01"))
end_date = st.date_input("📅 End date", pd.to_datetime("today"))
show_candle = st.checkbox("📊 Show Candlestick Chart", value=True)

# Get Data
@st.cache_data
def get_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df.dropna(inplace=True)
    df['Target'] = df['Close'].shift(-1) > df['Close']
    df.dropna(inplace=True)
    return df

if ticker:
    try:
        df = get_data(ticker, start_date, end_date)

        st.subheader(f"📊 Data for {ticker.upper()} from {start_date} to {end_date}")

        if show_candle:
            # Candlestick Chart
            fig = go.Figure(data=[go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name="Candles"
            )])
            fig.update_layout(title=f"Candlestick chart for {ticker.upper()}",
                              xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)
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

    except Exception as e:
        st.error(f"An error occurred: {e}")

# ChatGPT Integration
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.markdown("---")
st.markdown("🧠 **Ask ChatGPT about stocks, trading, or investing**")
user_question = st.text_input("💬 Ask ChatGPT:")

if user_question:
    with st.spinner("Thinking..."):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",  # Or use "gpt-3.5-turbo"
                messages=[{"role": "user", "content": user_question}],
                temperature=0.7,
                max_tokens=300,
            )
            st.success("🤖 ChatGPT says:")
            st.write(response.choices[0].message["content"])
        except Exception as e:
            st.error(f"❌ Failed to get response: {e}")

# Footer
st.markdown("---")
st.markdown("🧠 Powered by [yfinance](https://pypi.org/project/yfinance/), [scikit-learn](https://scikit-learn.org/), and [Streamlit](https://streamlit.io/)")
st.markdown("💻 Made by Precious Ofoyekpene")
