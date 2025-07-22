import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Page Config
st.set_page_config(page_title="PredictiTrade - Simple", layout="wide")

st.title("📈 Stock Predictor - Debug Version")
st.write("*Simplified version to troubleshoot data issues*")

# Sidebar
with st.sidebar:
    st.header("📊 Settings")
    ticker = st.text_input("Stock Ticker", "AAPL")
    start_date = st.date_input("Start Date", pd.to_datetime("2023-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("today"))

def download_and_process_data(ticker, start_date, end_date):
    """Simple data download and processing with debugging"""
    try:
        st.info(f"🔄 Downloading data for {ticker}...")
        
        # Download data
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            st.error(f"❌ No data found for {ticker}")
            return None
            
        st.success(f"✅ Downloaded {len(data)} days of data")
        
        # Show raw data info
        st.write("**Raw Data Info:**")
        st.write(f"- Shape: {data.shape}")
        st.write(f"- Columns: {list(data.columns)}")
        st.write(f"- Date range: {data.index[0]} to {data.index[-1]}")
        
        # Check for missing values
        missing_info = data.isnull().sum()
        if missing_info.sum() > 0:
            st.warning("⚠️ Missing values found:")
            st.write(missing_info[missing_info > 0])
        else:
            st.success("✅ No missing values in raw data")
        
        return data
        
    except Exception as e:
        st.error(f"❌ Error downloading data: {str(e)}")
        return None

def create_simple_features(df):
    """Create simple features with extensive debugging"""
    try:
        st.info("🔄 Creating features...")
        df = df.copy()
        
        # Basic features
        df['Price_Change'] = df['Close'].pct_change()
        st.write("✅ Created Price_Change")
        
        # Simple moving averages
        df['MA_5'] = df['Close'].rolling(window=5, min_periods=1).mean()
        df['MA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
        st.write("✅ Created Moving Averages")
        
        # Volatility
        df['Volatility'] = df['Close'].rolling(window=10, min_periods=1).std()
        st.write("✅ Created Volatility")
        
        # Target variable
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        st.write("✅ Created Target variable")
        
        # Check for issues
        st.write("**Feature Info:**")
        for col in ['Price_Change', 'MA_5', 'MA_20', 'Volatility', 'Target']:
            if col in df.columns:
                nan_count = df[col].isnull().sum()
                st.write(f"- {col}: {nan_count} NaN values")
            else:
                st.error(f"- {col}: Column missing!")
        
        # Drop NaN values
        initial_rows = len(df)
        df = df.dropna()
        final_rows = len(df)
        
        st.info(f"🔄 Dropped NaN: {initial_rows} → {final_rows} rows ({final_rows/initial_rows*100:.1f}% retained)")
        
        if len(df) == 0:
            st.error("❌ No data left after dropping NaN values")
            return None
            
        return df
        
    except Exception as e:
        st.error(f"❌ Error creating features: {str(e)}")
        st.write("**Error details:**")
        import traceback
        st.code(traceback.format_exc())
        return None

# Main app logic
if ticker:
    if start_date >= end_date:
        st.error("❌ Start date must be before end date")
        st.stop()
    
    # Download data
    raw_data = download_and_process_data(ticker, start_date, end_date)
    
    if raw_data is not None:
        # Show sample of raw data
        st.subheader("📋 Raw Data Sample")
        st.dataframe(raw_data.head(), use_container_width=True)
        
        # Process features
        processed_data = create_simple_features(raw_data)
        
        if processed_data is not None:
            st.success(f"✅ Data processing complete! Final shape: {processed_data.shape}")
            
            # Show processed data
            st.subheader("🔧 Processed Data Sample")
            st.dataframe(processed_data.head(), use_container_width=True)
            
            # Basic chart
            st.subheader("📊 Price Chart")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=processed_data.index, y=processed_data['Close'], 
                                   mode='lines', name='Close Price'))
            if 'MA_20' in processed_data.columns:
                fig.add_trace(go.Scatter(x=processed_data.index, y=processed_data['MA_20'], 
                                       mode='lines', name='MA 20'))
            fig.update_layout(title=f"{ticker} Price Chart", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)
            
            # Machine Learning
            st.subheader("🤖 Machine Learning Prediction")
            
            try:
                # Select features
                feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Price_Change', 'MA_5', 'MA_20', 'Volatility']
                available_features = [col for col in feature_cols if col in processed_data.columns]
                
                st.write(f"**Available features:** {available_features}")
                
                if len(available_features) < 3:
                    st.error("❌ Not enough features for modeling")
                else:
                    X = processed_data[available_features]
                    y = processed_data['Target']
                    
                    st.write(f"**Training data shape:** X={X.shape}, y={y.shape}")
                    
                    # Train/test split
                    test_size = 0.2
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, shuffle=False, random_state=42
                    )
                    
                    st.write(f"**Train/Test split:** Train={len(X_train)}, Test={len(X_test)}")
                    
                    # Train model
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    
                    # Predict
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    st.success(f"✅ Model trained! Accuracy: {accuracy*100:.2f}%")
                    
                    # Feature importance
                    importance_df = pd.DataFrame({
                        'Feature': available_features,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    st.write("**Feature Importance:**")
                    st.dataframe(importance_df, use_container_width=True)
                    
                    # Tomorrow's prediction
                    if len(X) > 0:
                        last_row = X.tail(1)
                        tomorrow_pred = model.predict(last_row)[0]
                        tomorrow_prob = model.predict_proba(last_row)[0]
                        
                        st.subheader("🔮 Tomorrow's Prediction")
                        if tomorrow_pred == 1:
                            st.success(f"📈 **UP** - Confidence: {tomorrow_prob[1]*100:.1f}%")
                        else:
                            st.error(f"📉 **DOWN** - Confidence: {tomorrow_prob[0]*100:.1f}%")
            
            except Exception as e:
                st.error(f"❌ Error in machine learning: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
        
        else:
            st.error("❌ Feature processing failed")
    else:
        st.error("❌ Data download failed")

else:
    st.info("👆 Enter a ticker symbol to start!")
    
    # Popular tickers
    st.write("**Try these popular stocks:**")
    cols = st.columns(4)
    popular = ["AAPL", "MSFT", "GOOGL", "TSLA"]
    for i, stock in enumerate(popular):
        with cols[i]:
            st.code(stock)

st.markdown("---")
st.info("🐛 This is a debug version to identify data processing issues. Check the logs above for detailed information.")
