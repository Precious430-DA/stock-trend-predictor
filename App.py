import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(layout="wide")
st.title("📊 PredictiTrade: AI-Powered Stock Trend Predictor")

# Sidebar Inputs
st.sidebar.header("Input Options")
ticker = st.sidebar.text_input("Enter stock ticker symbol (e.g. AAPL, TSLA, MSFT)", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
st.sidebar.caption("ℹ️ For non-US stocks, use correct suffix (e.g. '.NS', '.L', or '.SA').")

# --- Data Download Function ---
@st.cache_data
def get_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df is None or df.empty or df.shape[0] < 50:
            return None

        df = df.loc[:, ~df.columns.duplicated()]  # remove duplicated columns

        if 'Volume' in df.columns:
            if isinstance(df['Volume'], pd.DataFrame):
                df['Volume'] = df['Volume'].iloc[:, 0]  # take first column if Volume is DataFrame

        df = create_features(df)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return None

# --- Feature Engineering Function ---
def create_features(df):
    df['Daily_Return'] = df['Close'].pct_change()
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()

    if 'Volume' in df.columns and df['Volume'].ndim == 1:
        df['Volume_MA'] = df['Volume'].rolling(window=10).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    else:
        df['Volume_MA'] = np.nan
        df['Volume_Ratio'] = np.nan

    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    return df

# --- Load Data ---
df = get_data(ticker, start_date, end_date)

if df is None or len(df) < 50:
    st.error("📉 Unable to load sufficient data. Please check the ticker symbol and date range.")
    st.stop()

# --- Visualization: Candlestick ---
st.subheader(f"📈 Candlestick Chart: {ticker.upper()}")
fig = go.Figure(data=[go.Candlestick(
    x=df.index,
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close']
)])
fig.update_layout(xaxis_title='Date', yaxis_title='Price (USD)', xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# --- Machine Learning Section ---
st.subheader("🤖 AI Prediction Model")

features = ['Daily_Return', 'MA_5', 'MA_10', 'MA_20', 'Volume_Ratio']
df.dropna(inplace=True)
X = df[features]
y = df['Target']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"✅ **Model Accuracy**: {accuracy:.2%}")

# --- Prediction Results ---
df_pred = df.iloc[-len(y_test):].copy()
df_pred['Prediction'] = y_pred
df_pred['Prediction_Label'] = df_pred['Prediction'].map({1: '📈 Up', 0: '📉 Down'})

st.subheader("🔮 Prediction Results")
st.dataframe(df_pred[['Close', 'Prediction_Label']].tail(10))

# --- Chat Box ---
st.subheader("💬 Ask a Question About This Stock")
question = st.text_input("Enter your question:")
if question:
    st.write("💡 *AI Response Placeholder: This feature will answer your questions about the stock using AI.*")
