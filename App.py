import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page Config
st.set_page_config(page_title="PredictiTrade", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-up {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .prediction-down {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 AI Stock Trend Predictor (Enhanced)")
st.markdown("*Predict stock movements using advanced machine learning*")

# Sidebar: User Inputs
with st.sidebar:
    st.header("📊 Configuration")
    ticker = st.text_input("Stock Ticker", "AAPL", help="Enter a valid US stock ticker (e.g., AAPL, MSFT, TSLA)")
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", pd.to_datetime("2023-01-01"))
    with col2:
        end_date = st.date_input("End Date", pd.to_datetime("today"))
    
    # Model Parameters
    st.subheader("🤖 Model Settings")
    test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
    n_estimators = st.slider("Random Forest Trees", 50, 200, 100)
    
    # Chart Options
    st.subheader("📈 Chart Options")
    chart_type = st.selectbox("Chart Type", ["Candlestick", "Line Chart", "Both"])
    show_volume = st.checkbox("Show Volume", value=True)

# Feature Engineering Function
def create_features(df):
    df = df.copy()

    # Drop duplicate columns just in case
    df = df.loc[:, ~df.columns.duplicated()]

    # Price-based features
    df['Price_Change'] = df['Close'].pct_change()
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Open_Close_Ratio'] = df['Open'] / df['Close']

    # Moving averages
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()

    # Volatility
    df['Volatility'] = df['Close'].rolling(window=10).std()

    # Handle Volume safely
    if 'Volume' in df.columns and df['Volume'].ndim == 1:
        df['Volume_MA'] = df['Volume'].rolling(window=10).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    else:
        df['Volume_MA'] = np.nan
        df['Volume_Ratio'] = np.nan

    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Target variable (1 if price will go up next day, else 0)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    # Drop NaNs introduced by rolling
    df = df.dropna()

    return df
    
# Get and process data
@st.cache_data
def get_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            return None
        df = create_features(df)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return None

# Main logic
if ticker:
    if start_date >= end_date:
        st.error("Start date must be before end date!")
        st.stop()
    
    with st.spinner(f"Loading data for {ticker.upper()}..."):
        df = get_data(ticker, start_date, end_date)
    
    if df is None or len(df) < 50:
        st.error("Unable to load sufficient data. Please check the ticker symbol and date range.")
        st.stop()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Days", len(df))
    with col2:
        st.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
    with col3:
        price_change = df['Close'].iloc[-1] - df['Close'].iloc[-2]
        st.metric("Daily Change", f"${price_change:.2f}", delta=f"{(price_change/df['Close'].iloc[-2]*100):.2f}%")
    with col4:
        st.metric("Volatility", f"{df['Volatility'].iloc[-1]:.2f}")
    
    st.subheader(f"📊 {ticker.upper()} Price Analysis")
    
    if chart_type in ["Candlestick", "Both"]:
        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Price",
            yaxis="y1"
        ))

        fig.add_trace(go.Scatter(
            x=df.index, y=df['MA_20'],
            mode='lines',
            name='MA 20',
            line=dict(color='blue', width=1),
            yaxis="y1"
        ))

        if show_volume:
            fig.add_trace(go.Bar(
                x=df.index,
                y=df['Volume'],
                name="Volume",
                yaxis="y2",
                opacity=0.3
            ))

        fig.update_layout(
            title=f"{ticker.upper()} Stock Analysis",
            xaxis_title="Date",
            yaxis=dict(title="Price ($)", side="left"),
            yaxis2=dict(title="Volume", side="right", overlaying="y") if show_volume else None,
            height=600,
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)
    
    if chart_type == "Line Chart":
        st.line_chart(df['Close'])

    st.subheader("🤖 AI Prediction Model")

    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Price_Change',
                      'High_Low_Ratio', 'MA_5', 'MA_10', 'MA_20', 'Volatility',
                      'Volume_Ratio', 'RSI']

    available_features = [col for col in feature_columns if col in df.columns and df[col].notna().sum() > len(df) * 0.7]

    X = df[available_features]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False, random_state=42
    )

    with st.spinner("Training AI model..."):
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)

    col1, col2 = st.columns(2)
    with col1:
        st.success(f"✅ Model Accuracy: **{acc*100:.2f}%**")

        feature_importance = pd.DataFrame({
            'Feature': available_features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)

        st.write("**Top Features:**")
        for i, row in feature_importance.head(5).iterrows():
            st.write(f"• {row['Feature']}: {row['Importance']:.3f}")
    
    with col2:
        if len(X) > 0:
            last_features = X.tail(1)
            next_day_pred = model.predict(last_features)[0]
            next_day_proba = model.predict_proba(last_features)[0]

            st.markdown("### 🔮 Tomorrow's Prediction")

            if next_day_pred == 1:
                confidence = next_day_proba[1] * 100
                st.markdown(f"""
                <div class="prediction-up">
                    <h4>📈 BULLISH</h4>
                    <p>Stock likely to go <strong>UP</strong></p>
                    <p>Confidence: <strong>{confidence:.1f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                confidence = next_day_proba[0] * 100
                st.markdown(f"""
                <div class="prediction-down">
                    <h4>📉 BEARISH</h4>
                    <p>Stock likely to go <strong>DOWN</strong></p>
                    <p>Confidence: <strong>{confidence:.1f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)

    st.subheader("📋 Recent Data")
    display_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_20', 'RSI']
    available_display_cols = [col for col in display_columns if col in df.columns]
    st.dataframe(df[available_display_cols].tail(10), use_container_width=True)

    st.markdown("---")
    st.subheader("🤖 AI Assistant")
    user_question = st.text_area("Ask me about this stock analysis:", 
                                 placeholder="e.g., 'What factors are most important for this prediction?'")

    if user_question:
        question_lower = user_question.lower()

        if any(word in question_lower for word in ['accuracy', 'performance', 'reliable']):
            st.info(f"📊 The current model has an accuracy of {acc*100:.1f}%.")
        
        elif any(word in question_lower for word in ['features', 'important', 'factors']):
            st.info(f"🔍 The top factors for {ticker.upper()} predictions are:\n" + 
                   "\n".join([f"• **{row['Feature']}**: {row['Importance']:.3f}" 
                             for _, row in feature_importance.head(3).iterrows()]))
        
        elif any(word in question_lower for word in ['risk', 'warning', 'disclaimer']):
            st.warning("⚠️ This tool is for educational purposes only. Please do your own research.")
        
        elif any(word in question_lower for word in ['volume', 'trading']):
            recent_volume = df['Volume'].tail(5).mean()
            st.info(f"📈 Recent average volume: {recent_volume:,.0f} shares.")
        
        else:
            st.info("🤖 This analysis uses machine learning to predict stock movements. Ask me about features, accuracy, or risks.")

    st.markdown("---")
    st.markdown("**⚠️ Disclaimer**: This tool is for educational purposes only. Investments carry risk.")
    st.markdown("🔧 **Powered by**: yfinance, scikit-learn, Streamlit, Plotly")

else:
    st.info("👆 Please enter a stock ticker symbol to begin analysis!")

    st.markdown("### 💡 Popular Tickers to Try:")
    sample_tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA", "META", "NFLX"]
    cols = st.columns(4)
    for i, ticker_example in enumerate(sample_tickers):
        with cols[i % 4]:
            if st.button(f"📊 {ticker_example}", key=ticker_example):
                st.rerun()
