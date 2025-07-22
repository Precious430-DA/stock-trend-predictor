import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import talib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page Config
st.set_page_config(
    page_title="PredictiTrade Pro", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .bullish { color: #00ff00; font-weight: bold; }
    .bearish { color: #ff0000; font-weight: bold; }
    .neutral { color: #ffaa00; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("📈 PredictiTrade Pro - Advanced Stock Analysis & Prediction")
st.markdown("*Professional-grade stock analysis with real-time data and trading insights*")

# Sidebar Configuration
with st.sidebar:
    st.header("📊 Trading Configuration")
    
    # Stock Selection
    ticker = st.text_input("Stock Ticker", "AAPL", help="Enter any valid stock ticker (e.g., AAPL, MSFT, TSLA)")
    
    # Time Range
    time_range = st.selectbox("Time Range", [
        "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"
    ], index=3)
    
    # Or custom date range
    use_custom_dates = st.checkbox("Use Custom Date Range")
    if use_custom_dates:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
        end_date = st.date_input("End Date", datetime.now())
    
    # Analysis Options
    st.header("🔧 Analysis Options")
    include_premarket = st.checkbox("Include Pre/Post Market", value=False)
    include_dividends = st.checkbox("Include Dividend Data", value=True)
    include_splits = st.checkbox("Include Stock Splits", value=True)
    
    # Technical Indicators
    st.header("📈 Technical Indicators")
    show_ma = st.multiselect("Moving Averages", [5, 10, 20, 50, 100, 200], default=[20, 50])
    show_bollinger = st.checkbox("Bollinger Bands", value=True)
    show_rsi = st.checkbox("RSI", value=True)
    show_macd = st.checkbox("MACD", value=True)
    show_volume = st.checkbox("Volume Analysis", value=True)
    
    # ML Model Selection
    st.header("🤖 ML Model")
    model_type = st.selectbox("Model Type", [
        "Random Forest", "Gradient Boosting", "Logistic Regression", "Ensemble"
    ])
    
    prediction_days = st.slider("Prediction Horizon (days)", 1, 30, 5)

@st.cache_data
def get_stock_info(ticker):
    """Get comprehensive stock information"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info
    except:
        return {}

@st.cache_data
def download_stock_data(ticker, period=None, start=None, end=None, premarket=False):
    """Download comprehensive stock data"""
    try:
        stock = yf.Ticker(ticker)
        
        # Download main data
        if period:
            data = stock.history(period=period, prepost=premarket)
        else:
            data = stock.history(start=start, end=end, prepost=premarket)
        
        if data.empty:
            return None, None, None, None
            
        # Get additional data
        dividends = stock.dividends if include_dividends else None
        splits = stock.splits if include_splits else None
        
        # Get options data (if available)
        try:
            options_dates = stock.options
            options_data = None
            if options_dates:
                # Get options for next expiry
                options_data = stock.option_chain(options_dates[0])
        except:
            options_data = None
            
        return data, dividends, splits, options_data
        
    except Exception as e:
        st.error(f"Error downloading data: {str(e)}")
        return None, None, None, None

def calculate_technical_indicators(df):
    """Calculate comprehensive technical indicators"""
    df = df.copy()
    
    # Price-based indicators
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Moving averages
    for period in [5, 10, 20, 50, 100, 200]:
        df[f'MA_{period}'] = df['Close'].rolling(window=period).mean()
        df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
    
    # Bollinger Bands
    df['BB_Upper'] = df['Close'].rolling(window=20).mean() + (df['Close'].rolling(window=20).std() * 2)
    df['BB_Lower'] = df['Close'].rolling(window=20).mean() - (df['Close'].rolling(window=20).std() * 2)
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # RSI
    try:
        df['RSI'] = talib.RSI(df['Close'].values, timeperiod=14)
    except:
        # Manual RSI calculation if talib fails
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    try:
        df['MACD'], df['MACD_Signal'], df['MACD_Histogram'] = talib.MACD(df['Close'].values)
    except:
        # Manual MACD calculation
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Stochastic Oscillator
    try:
        df['Stoch_K'], df['Stoch_D'] = talib.STOCH(df['High'].values, df['Low'].values, df['Close'].values)
    except:
        # Manual calculation
        lowest_low = df['Low'].rolling(window=14).min()
        highest_high = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * (df['Close'] - lowest_low) / (highest_high - lowest_low)
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
    
    # Volume indicators
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    
    # Volatility indicators
    df['ATR'] = df[['High', 'Low', 'Close']].apply(lambda x: 
        max(x['High'] - x['Low'], abs(x['High'] - x['Close']), abs(x['Low'] - x['Close'])), axis=1).rolling(window=14).mean()
    df['Volatility'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
    
    # Price patterns
    df['Higher_High'] = (df['High'] > df['High'].shift(1)).astype(int)
    df['Lower_Low'] = (df['Low'] < df['Low'].shift(1)).astype(int)
    df['Doji'] = (abs(df['Close'] - df['Open']) <= (df['High'] - df['Low']) * 0.1).astype(int)
    
    # Support and Resistance levels
    df['Resistance'] = df['High'].rolling(window=20).max()
    df['Support'] = df['Low'].rolling(window=20).min()
    
    return df

def create_prediction_targets(df, days_ahead=5):
    """Create multiple prediction targets"""
    df = df.copy()
    
    # Binary targets
    df['Target_Direction'] = (df['Close'].shift(-days_ahead) > df['Close']).astype(int)
    df['Target_Strong_Up'] = (df['Close'].shift(-days_ahead) / df['Close'] > 1.05).astype(int)
    df['Target_Strong_Down'] = (df['Close'].shift(-days_ahead) / df['Close'] < 0.95).astype(int)
    
    # Regression targets
    df['Target_Price'] = df['Close'].shift(-days_ahead)
    df['Target_Return'] = (df['Close'].shift(-days_ahead) / df['Close']) - 1
    
    return df

def build_ml_models(df, target_col='Target_Direction'):
    """Build and evaluate multiple ML models"""
    # Select features
    feature_cols = [col for col in df.columns if not col.startswith('Target_') and 
                   col not in ['Open', 'High', 'Low', 'Close', 'Volume'] and
                   not pd.isna(df[col]).all()]
    
    # Remove highly correlated features
    correlation_matrix = df[feature_cols].corr().abs()
    upper_triangle = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
    feature_cols = [col for col in feature_cols if col not in to_drop]
    
    # Prepare data
    X = df[feature_cols].fillna(df[feature_cols].mean())
    y = df[target_col]
    
    # Remove rows with NaN targets
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    if len(X) < 50:
        return None, None, None, None
    
    # Train-test split (time series split)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Models
    models = {}
    
    if model_type == "Random Forest" or model_type == "Ensemble":
        models['Random Forest'] = RandomForestClassifier(n_estimators=100, random_state=42)
    
    if model_type == "Gradient Boosting" or model_type == "Ensemble":
        models['Gradient Boosting'] = GradientBoostingClassifier(n_estimators=100, random_state=42)
        
    if model_type == "Logistic Regression" or model_type == "Ensemble":
        models['Logistic Regression'] = LogisticRegression(random_state=42)
    
    # Train and evaluate models
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred,
            'actual': y_test
        }
    
    return results, X_train, X_test, feature_cols

def generate_trading_signals(df, predictions=None):
    """Generate trading signals based on technical analysis and ML predictions"""
    signals = pd.DataFrame(index=df.index)
    
    # Technical signals
    signals['MA_Signal'] = np.where(df['Close'] > df['MA_20'], 1, -1)
    signals['RSI_Signal'] = np.where(df['RSI'] < 30, 1, np.where(df['RSI'] > 70, -1, 0))
    signals['MACD_Signal'] = np.where(df['MACD'] > df['MACD_Signal'], 1, -1)
    signals['BB_Signal'] = np.where(df['Close'] < df['BB_Lower'], 1, 
                                   np.where(df['Close'] > df['BB_Upper'], -1, 0))
    
    # Volume confirmation
    signals['Volume_Confirm'] = np.where(df['Volume'] > df['Volume_MA'], 1, 0)
    
    # Combine signals
    signals['Technical_Score'] = (
        signals['MA_Signal'] + signals['RSI_Signal'] + 
        signals['MACD_Signal'] + signals['BB_Signal']
    ) / 4
    
    # Add ML predictions if available
    if predictions is not None:
        signals['ML_Signal'] = predictions
        signals['Combined_Signal'] = (signals['Technical_Score'] + signals['ML_Signal']) / 2
    else:
        signals['Combined_Signal'] = signals['Technical_Score']
    
    return signals

# Main Application
if ticker:
    # Get stock information
    with st.spinner("Loading stock information..."):
        stock_info = get_stock_info(ticker)
    
    # Display stock info
    if stock_info:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Company", stock_info.get('shortName', ticker))
            st.metric("Sector", stock_info.get('sector', 'N/A'))
        
        with col2:
            current_price = stock_info.get('currentPrice', stock_info.get('regularMarketPrice', 'N/A'))
            previous_close = stock_info.get('previousClose', 'N/A')
            if current_price != 'N/A' and previous_close != 'N/A':
                change = current_price - previous_close
                change_pct = (change / previous_close) * 100
                st.metric("Current Price", f"${current_price:.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
            else:
                st.metric("Current Price", current_price)
        
        with col3:
            st.metric("Market Cap", 
                     f"${stock_info.get('marketCap', 0)/1e9:.2f}B" if stock_info.get('marketCap') else 'N/A')
            st.metric("P/E Ratio", f"{stock_info.get('trailingPE', 'N/A'):.2f}" if stock_info.get('trailingPE') else 'N/A')
        
        with col4:
            st.metric("52W High", f"${stock_info.get('fiftyTwoWeekHigh', 'N/A')}")
            st.metric("52W Low", f"${stock_info.get('fiftyTwoWeekLow', 'N/A')}")
    
    # Download data
    with st.spinner("Downloading market data..."):
        if use_custom_dates:
            data, dividends, splits, options = download_stock_data(
                ticker, start=start_date, end=end_date, premarket=include_premarket
            )
        else:
            data, dividends, splits, options = download_stock_data(
                ticker, period=time_range, premarket=include_premarket
            )
    
    if data is not None and not data.empty:
        st.success(f"✅ Downloaded {len(data)} trading days of data")
        
        # Calculate technical indicators
        with st.spinner("Calculating technical indicators..."):
            data_with_indicators = calculate_technical_indicators(data)
            data_with_targets = create_prediction_targets(data_with_indicators, prediction_days)
        
        # Create main price chart
        st.subheader("📊 Price Chart with Technical Analysis")
        
        # Determine subplot configuration
        subplot_count = 1
        if show_volume: subplot_count += 1
        if show_rsi: subplot_count += 1
        if show_macd: subplot_count += 1
        
        subplot_titles = ['Price']
        if show_volume: subplot_titles.append('Volume')
        if show_rsi: subplot_titles.append('RSI')
        if show_macd: subplot_titles.append('MACD')
        
        fig = make_subplots(
            rows=subplot_count, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=subplot_titles,
            row_heights=[0.6] + [0.4/(subplot_count-1)]*(subplot_count-1) if subplot_count > 1 else [1]
        )
        
        # Main price chart
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ), row=1, col=1)
        
        # Moving averages
        for ma in show_ma:
            if f'MA_{ma}' in data_with_indicators.columns:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data_with_indicators[f'MA_{ma}'],
                    name=f'MA {ma}',
                    line=dict(width=1)
                ), row=1, col=1)
        
        # Bollinger Bands
        if show_bollinger:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data_with_indicators['BB_Upper'],
                line=dict(color='rgba(0,100,80,0.2)'),
                showlegend=False
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data_with_indicators['BB_Lower'],
                fill='tonexty',
                fillcolor='rgba(0,100,80,0.1)',
                line=dict(color='rgba(0,100,80,0.2)'),
                name='Bollinger Bands'
            ), row=1, col=1)
        
        current_row = 2
        
        # Volume
        if show_volume:
            fig.add_trace(go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color='lightblue'
            ), row=current_row, col=1)
            current_row += 1
        
        # RSI
        if show_rsi:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data_with_indicators['RSI'],
                name='RSI',
                line=dict(color='purple')
            ), row=current_row, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1)
            current_row += 1
        
        # MACD
        if show_macd:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data_with_indicators['MACD'],
                name='MACD',
                line=dict(color='blue')
            ), row=current_row, col=1)
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data_with_indicators['MACD_Signal'],
                name='MACD Signal',
                line=dict(color='red')
            ), row=current_row, col=1)
            fig.add_trace(go.Bar(
                x=data.index,
                y=data_with_indicators['MACD_Histogram'],
                name='MACD Histogram',
                marker_color='gray'
            ), row=current_row, col=1)
        
        fig.update_layout(
            title=f"{ticker} - Comprehensive Technical Analysis",
            height=800,
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Machine Learning Predictions
        st.subheader("🤖 Machine Learning Predictions")
        
        with st.spinner("Training ML models..."):
            ml_results, X_train, X_test, features = build_ml_models(
                data_with_targets, f'Target_Direction'
            )
        
        if ml_results:
            # Display model performance
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Model Performance:**")
                performance_df = pd.DataFrame({
                    'Model': list(ml_results.keys()),
                    'Accuracy': [r['accuracy'] for r in ml_results.values()],
                    'CV Mean': [r['cv_mean'] for r in ml_results.values()],
                    'CV Std': [r['cv_std'] for r in ml_results.values()]
                })
                st.dataframe(performance_df, use_container_width=True)
            
            with col2:
                # Best model prediction
                best_model_name = max(ml_results.keys(), key=lambda k: ml_results[k]['accuracy'])
                best_model = ml_results[best_model_name]['model']
                
                # Make prediction for next period
                if len(data_with_targets) > 0:
                    last_features = data_with_targets[features].fillna(method='ffill').iloc[-1:].fillna(0)
                    prediction = best_model.predict(last_features)[0]
                    prediction_proba = best_model.predict_proba(last_features)[0]
                    
                    st.write(f"**{prediction_days}-Day Prediction ({best_model_name}):**")
                    if prediction == 1:
                        st.markdown('<p class="bullish">📈 BULLISH</p>', unsafe_allow_html=True)
                        confidence = prediction_proba[1] * 100
                    else:
                        st.markdown('<p class="bearish">📉 BEARISH</p>', unsafe_allow_html=True)
                        confidence = prediction_proba[0] * 100
                    
                    st.write(f"**Confidence:** {confidence:.1f}%")
        
        # Trading Signals
        st.subheader("🎯 Trading Signals")
        
        # Generate signals
        latest_predictions = None
        if ml_results:
            best_model = ml_results[best_model_name]['model']
            recent_features = data_with_targets[features].fillna(method='ffill').tail(30).fillna(0)
            latest_predictions = best_model.predict(recent_features)
        
        signals = generate_trading_signals(data_with_indicators.tail(30), latest_predictions)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            latest_signal = signals['Combined_Signal'].iloc[-1]
            if latest_signal > 0.3:
                st.success("🟢 BUY Signal")
            elif latest_signal < -0.3:
                st.error("🔴 SELL Signal")
            else:
                st.warning("🟡 HOLD Signal")
        
        with col2:
            st.metric("Signal Strength", f"{abs(latest_signal):.2f}")
        
        with col3:
            rsi_latest = data_with_indicators['RSI'].iloc[-1]
            if rsi_latest < 30:
                st.success("Oversold (RSI)")
            elif rsi_latest > 70:
                st.error("Overbought (RSI)")
            else:
                st.info("Neutral (RSI)")
        
        # Additional Data
        if include_dividends and dividends is not None and not dividends.empty:
            st.subheader("💰 Dividend History")
            st.dataframe(dividends.tail(10), use_container_width=True)
        
        if include_splits and splits is not None and not splits.empty:
            st.subheader("📊 Stock Splits")
            st.dataframe(splits.tail(5), use_container_width=True)
        
        # Risk Metrics
        st.subheader("⚠️ Risk Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        returns = data_with_indicators['Returns'].dropna()
        
        with col1:
            volatility = returns.std() * np.sqrt(252) * 100
            st.metric("Annualized Volatility", f"{volatility:.2f}%")
        
        with col2:
            sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        
        with col3:
            max_drawdown = ((data['Close'] / data['Close'].cummax()) - 1).min() * 100
            st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
        
        with col4:
            beta = np.corrcoef(returns.values[1:], returns.values[:-1])[0,1] if len(returns) > 1 else 0
            st.metric("Beta (auto-correlation)", f"{beta:.2f}")
        
        # Export functionality
        st.subheader("📥 Export Data")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Historical Data"):
                csv = data_with_indicators.to_csv()
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{ticker}_historical_data.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("Export Predictions"):
                if ml_results:
                    pred_df = pd.DataFrame({
                        'Date': X_test.index if hasattr(X_test, 'index') else range(len(X_test)),
                        'Actual': ml_results[best_model_name]['actual'],
                        'Predicted': ml_results[best_model_name]['predictions']
                    })
                    csv = pred_df.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions CSV",
                        data=csv,
                        file_name=f"{ticker}_predictions.csv",
                        mime="text/csv"
                    )
    
    else:
        st.error("❌ Unable to download data. Please check the ticker symbol and try again.")

else:
    st.info("👆 Enter a ticker symbol to start analyzing!")
    
    # Market overview
    st.subheader("📈 Popular Stocks")
    popular_stocks = {
        "Tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"],
        "Finance": ["JPM", "BAC", "WFC", "GS", "MS", "C"],
        "Healthcare": ["JNJ", "PFE", "UNH", "MRNA", "ABBV", "TMO"],
        "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "PXD"]
    }
    
    for sector, stocks in popular_stocks.items():
        st.write(f"**{sector}:** {', '.join(stocks)}")

# Footer
st.markdown("---")
st.markdown("""
**Disclaimer:** This app is for educational purposes only. Not financial advice. 
Always do your own research and consult with financial professionals before making investment decisions.
""")
