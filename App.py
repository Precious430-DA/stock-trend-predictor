import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page Config
st.set_page_config(page_title="PredictiTrade", layout="wide")

# Custom CSS for better styling
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
    """Create additional technical indicators as features"""
    df = df.copy()
    
    try:
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
        
        # Volume indicators (with error handling)
        if 'Volume' in df.columns:
            volume_ma = df['Volume'].rolling(window=10).mean()
            df['Volume_MA'] = volume_ma
            # Fix: Ensure we're doing element-wise division
            df['Volume_Ratio'] = df['Volume'].div(volume_ma).fillna(1.0)
        else:
            df['Volume_MA'] = 1.0
            df['Volume_Ratio'] = 1.0
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        # Avoid division by zero
        rs = gain.div(loss).fillna(0)
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['RSI'].fillna(50)  # Fill NaN with neutral RSI value
        
        # Target variable
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        # Fill any remaining NaN values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(method='forward').fillna(method='backward')
        
    except Exception as e:
        st.error(f"Error in feature engineering: {e}")
        # Return minimal dataframe if feature engineering fails
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    return df

# Get and process data
@st.cache_data
def get_data(ticker, start, end):
    try:
        # Download data with progress disabled
        df = yf.download(ticker, start=start, end=end, progress=False)
        
        if df.empty:
            st.error(f"No data found for ticker '{ticker}'. Please check if it's a valid stock symbol.")
            return None
            
        if len(df) < 30:
            st.warning(f"Limited data available ({len(df)} days). Consider extending the date range.")
            
        # Reset index to ensure proper datetime handling
        df = df.reset_index()
        df.set_index('Date', inplace=True)
        
        # Apply feature engineering
        df = create_features(df)
        
        # Drop rows with NaN values
        initial_length = len(df)
        df.dropna(inplace=True)
        
        if len(df) == 0:
            st.error("No valid data after processing. Try a different date range.")
            return None
            
        if len(df) < initial_length * 0.7:
            st.warning(f"Significant data loss during processing. Using {len(df)} out of {initial_length} rows.")
            
        return df
        
    except Exception as e:
        st.error(f"Error downloading data: {str(e)}")
        return None

# Main Application Logic
if ticker:
    # Validate date range
    if start_date >= end_date:
        st.error("Start date must be before end date!")
        st.stop()
    
    # Load data
    with st.spinner(f"Loading data for {ticker.upper()}..."):
        df = get_data(ticker, start_date, end_date)
    
    if df is None or len(df) < 50:
        st.error("Unable to load sufficient data. Please check the ticker symbol and date range.")
        st.stop()
    
    # Display basic info
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
    
    # Charts
    st.subheader(f"📊 {ticker.upper()} Price Analysis")
    
    if chart_type in ["Candlestick", "Both"]:
        fig = go.Figure()
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Price",
            yaxis="y1"
        ))
        
        # Moving averages
        fig.add_trace(go.Scatter(
            x=df.index, y=df['MA_20'],
            mode='lines',
            name='MA 20',
            line=dict(color='blue', width=1),
            yaxis="y1"
        ))
        
        if show_volume:
            # Volume
            fig.add_trace(go.Bar(
                x=df.index,
                y=df['Volume'],
                name="Volume",
                yaxis="y2",
                opacity=0.3
            ))
        
        # Layout
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
    
    # Machine Learning Section
    st.subheader("🤖 AI Prediction Model")
    
    # Feature selection with fallback
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Price_Change', 
                      'High_Low_Ratio', 'MA_5', 'MA_10', 'MA_20', 'Volatility', 
                      'Volume_Ratio', 'RSI']
    
    # Only use features that exist and have sufficient data
    available_features = []
    for col in feature_columns:
        if col in df.columns and df[col].notna().sum() > len(df) * 0.5:
            available_features.append(col)
    
    # Fallback to basic features if advanced features failed
    if len(available_features) < 5:
        basic_features = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_features = [col for col in basic_features if col in df.columns]
        st.info("Using basic features due to data processing issues.")
    
    if len(available_features) == 0:
        st.error("No valid features available for modeling.")
        st.stop()
    
    X = df[available_features]
    y = df['Target']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False, random_state=42
    )
    
    # Train model
    with st.spinner("Training AI model..."):
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Model performance
    acc = accuracy_score(y_test, y_pred)
    
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"✅ Model Accuracy: **{acc*100:.2f}%**")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': available_features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        st.write("**Top Features:**")
        for i, row in feature_importance.head(5).iterrows():
            st.write(f"• {row['Feature']}: {row['Importance']:.3f}")
    
    with col2:
        # Prediction for tomorrow
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
    
    # Recent data preview
    st.subheader("📋 Recent Data")
    display_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_20', 'RSI']
    available_display_cols = [col for col in display_columns if col in df.columns]
    st.dataframe(df[available_display_cols].tail(10), use_container_width=True)
    
    # Interactive Chat Section
    st.markdown("---")
    st.subheader("🤖 AI Assistant")
    
    user_question = st.text_area("Ask me about this stock analysis:", 
                                placeholder="e.g., 'What factors are most important for this prediction?'")
    
    if user_question:
        # Simple rule-based responses
        question_lower = user_question.lower()
        
        if any(word in question_lower for word in ['accuracy', 'performance', 'reliable']):
            st.info(f"📊 The current model has an accuracy of {acc*100:.1f}%. "
                   f"This means it correctly predicts the direction {acc*100:.1f}% of the time on test data. "
                   f"Remember that past performance doesn't guarantee future results!")
        
        elif any(word in question_lower for word in ['features', 'important', 'factors']):
            st.info(f"🔍 The most important factors for {ticker.upper()} predictions are:\n" + 
                   "\n".join([f"• **{row['Feature']}**: {row['Importance']:.3f}" 
                             for _, row in feature_importance.head(3).iterrows()]))
        
        elif any(word in question_lower for word in ['risk', 'warning', 'disclaimer']):
            st.warning("⚠️ **Important Disclaimer**: This is for educational purposes only. "
                      "Stock predictions are inherently uncertain. Never invest money you can't afford to lose. "
                      "Always do your own research and consider consulting a financial advisor.")
        
        elif any(word in question_lower for word in ['volume', 'trading']):
            recent_volume = df['Volume'].tail(5).mean()
            st.info(f"📈 Recent average trading volume: {recent_volume:,.0f} shares. "
                   f"Volume is important as it indicates investor interest and liquidity.")
        
        else:
            st.info("🤖 Thank you for your question! This analysis uses machine learning to predict stock movements "
                   "based on technical indicators. Feel free to ask about accuracy, important features, risks, or volume.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **⚠️ Disclaimer**: This tool is for educational purposes only. Stock market investments carry risk. 
    Past performance does not guarantee future results. Please consult with a qualified financial advisor before making investment decisions.
    """)
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("🔧 **Powered by**: yfinance, scikit-learn, Streamlit, Plotly")
    with col2:
        st.markdown("👨‍💻 **Enhanced by**: AI Assistant | **Original by**: Precious Ofoyekpene")

else:
    st.info("👆 Please enter a stock ticker symbol to begin analysis!")
    
    # Sample tickers
    st.markdown("### 💡 Popular Tickers to Try:")
    sample_tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA", "META", "NFLX"]
    cols = st.columns(4)
    for i, ticker_example in enumerate(sample_tickers):
        with cols[i % 4]:
            if st.button(f"📊 {ticker_example}", key=ticker_example):
                st.rerun()
