import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import time

# Comprehensive Stock Data Access Examples

st.title("📈 Comprehensive Stock Data Access Guide")
st.markdown("*Learn how to access maximum stock data with yfinance and other sources*")

# 1. BASIC YFINANCE ACCESS
st.header("1. 📊 Basic yfinance Access")
st.code("""
import yfinance as yf

# Basic download
data = yf.download('AAPL', start='2020-01-01', end='2024-01-01')

# Multiple stocks at once
data = yf.download(['AAPL', 'MSFT', 'GOOGL'], start='2020-01-01', end='2024-01-01')

# All available data for a stock
data = yf.download('AAPL', period='max')
""")

# 2. ENHANCED DATA ACCESS
st.header("2. 🚀 Enhanced Data Access")

def get_comprehensive_stock_data(ticker, period='2y'):
    """Get all available data for a stock"""
    try:
        # Create ticker object for more detailed info
        stock = yf.Ticker(ticker)
        
        # Get historical data
        hist_data = stock.history(period=period)
        
        # Get additional info
        info = stock.info
        
        # Get financials
        try:
            financials = stock.financials
            balance_sheet = stock.balance_sheet
            cashflow = stock.cashflow
        except:
            financials = balance_sheet = cashflow = None
        
        # Get dividends and splits
        dividends = stock.dividends
        splits = stock.splits
        
        # Get recommendations
        try:
            recommendations = stock.recommendations
        except:
            recommendations = None
        
        # Get earnings
        try:
            earnings = stock.earnings
            quarterly_earnings = stock.quarterly_earnings
        except:
            earnings = quarterly_earnings = None
        
        return {
            'historical': hist_data,
            'info': info,
            'financials': financials,
            'balance_sheet': balance_sheet,
            'cashflow': cashflow,
            'dividends': dividends,
            'splits': splits,
            'recommendations': recommendations,
            'earnings': earnings,
            'quarterly_earnings': quarterly_earnings
        }
    
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

# Demo with user input
st.subheader("📝 Try Comprehensive Data Access")
demo_ticker = st.text_input("Enter a stock ticker:", "AAPL")

if demo_ticker and st.button("Get Comprehensive Data"):
    with st.spinner(f"Fetching all available data for {demo_ticker}..."):
        comprehensive_data = get_comprehensive_stock_data(demo_ticker)
    
    if comprehensive_data:
        st.success("✅ Data retrieved successfully!")
        
        # Historical data
        if comprehensive_data['historical'] is not None and not comprehensive_data['historical'].empty:
            st.subheader("📈 Historical Price Data")
            st.write(f"Shape: {comprehensive_data['historical'].shape}")
            st.dataframe(comprehensive_data['historical'].tail())
            
            # Show available columns
            st.write("**Available columns:**", list(comprehensive_data['historical'].columns))
        
        # Company info
        if comprehensive_data['info']:
            st.subheader("🏢 Company Information")
            info_df = pd.DataFrame(list(comprehensive_data['info'].items()), 
                                 columns=['Metric', 'Value'])
            st.dataframe(info_df.head(20))
        
        # Dividends
        if comprehensive_data['dividends'] is not None and not comprehensive_data['dividends'].empty:
            st.subheader("💰 Dividends")
            st.dataframe(comprehensive_data['dividends'].tail())
        
        # Recommendations
        if comprehensive_data['recommendations'] is not None and not comprehensive_data['recommendations'].empty:
            st.subheader("📊 Analyst Recommendations")
            st.dataframe(comprehensive_data['recommendations'].tail())

# 3. ACCESSING MULTIPLE STOCKS
st.header("3. 🌍 Accessing Multiple Stocks")

# Popular stock lists
popular_stocks = {
    "S&P 500 Sample": ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V'],
    "Tech Giants": ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'NFLX', 'ADBE', 'CRM', 'ORCL'],
    "Dow Jones Sample": ['AAPL', 'MSFT', 'JPM', 'JNJ', 'PG', 'UNH', 'HD', 'DIS', 'BA', 'MCD'],
    "Crypto-Related": ['COIN', 'MSTR', 'RIOT', 'MARA', 'SQ', 'PYPL'],
    "Banking": ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'TFC', 'COF']
}

selected_list = st.selectbox("Select a stock list:", list(popular_stocks.keys()))

if st.button("Download Multiple Stocks"):
    tickers = popular_stocks[selected_list]
    
    with st.spinner(f"Downloading {len(tickers)} stocks..."):
        try:
            # Download multiple stocks at once
            multi_data = yf.download(tickers, period='1y', group_by='ticker')
            
            st.success(f"✅ Downloaded data for {len(tickers)} stocks!")
            st.write(f"**Shape:** {multi_data.shape}")
            st.write(f"**Stocks:** {', '.join(tickers)}")
            
            # Show sample data for first stock
            if len(tickers) > 1:
                first_stock = tickers[0]
                st.subheader(f"📊 Sample Data - {first_stock}")
                st.dataframe(multi_data[first_stock].tail())
            else:
                st.dataframe(multi_data.tail())
                
        except Exception as e:
            st.error(f"Error downloading multiple stocks: {e}")

# 4. DIFFERENT TIME PERIODS AND INTERVALS
st.header("4. ⏰ Different Time Periods & Intervals")

st.code("""
# Different periods
periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']

# Different intervals
intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']

# Examples:
data_1min = yf.download('AAPL', period='1d', interval='1m')  # 1-minute data for 1 day
data_weekly = yf.download('AAPL', period='2y', interval='1wk')  # Weekly data for 2 years
data_monthly = yf.download('AAPL', period='10y', interval='1mo')  # Monthly data for 10 years
""")

# 5. INTERNATIONAL STOCKS
st.header("5. 🌍 International Stocks")

international_examples = {
    "European": ["ASML", "SAP", "NESN.SW", "MC.PA", "AZN.L"],
    "Asian": ["TSM", "BABA", "7203.T", "005930.KS", "2330.TW"],
    "Canadian": ["SHOP.TO", "CNQ.TO", "RY.TO", "BNS.TO"],
    "Australian": ["CBA.AX", "BHP.AX", "CSL.AX", "WBC.AX"]
}

st.write("**International Stock Examples:**")
for region, stocks in international_examples.items():
    st.write(f"**{region}:** {', '.join(stocks)}")

# Test international stock
test_international = st.selectbox("Test an international stock:", 
                                ["ASML", "TSM", "NESN.SW", "SHOP.TO", "CBA.AX"])

if st.button("Test International Stock"):
    try:
        intl_data = yf.download(test_international, period='3mo')
        if not intl_data.empty:
            st.success(f"✅ Successfully downloaded {test_international}")
            st.dataframe(intl_data.tail())
        else:
            st.error(f"No data found for {test_international}")
    except Exception as e:
        st.error(f"Error: {e}")

# 6. LIMITATIONS AND ALTERNATIVES
st.header("6. ⚠️ Limitations & Professional Alternatives")

st.markdown("""
### 🚫 yfinance Limitations:
- **Free but not guaranteed**: Yahoo Finance can change their API
- **Rate limits**: Too many requests can get blocked
- **Data quality**: Occasional missing or incorrect data
- **Real-time limits**: 15-20 minute delay on free data
- **No tick-by-tick data**: Limited to minute-level at best

### 💰 Professional Data Sources:
- **Alpha Vantage**: 500 free API calls/day
- **IEX Cloud**: Reliable, reasonably priced
- **Quandl/NASDAQ**: High-quality financial data
- **Bloomberg API**: Professional grade (expensive)
- **Refinitiv (Reuters)**: Institutional level
- **Polygon.io**: Real-time and historical data
- **TD Ameritrade API**: Free with account
- **Interactive Brokers API**: Professional trading

### 🔧 Code for Alpha Vantage Alternative:
```python
import requests
import pandas as pd

def get_alpha_vantage_data(symbol, api_key):
    url = f'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'apikey': api_key,
        'outputsize': 'full'
    }
    response = requests.get(url, params=params)
    data = response.json()
    return data
```
""")

# 7. TIPS FOR MAXIMUM ACCESS
st.header("7. 💡 Tips for Maximum Data Access")

tips = [
    "**Use ticker objects** for detailed info: `stock = yf.Ticker('AAPL')`",
    "**Batch downloads** are more efficient: `yf.download(['AAPL', 'MSFT'])`",
    "**Handle errors gracefully** - some stocks may not have all data types",
    "**Use appropriate periods** - don't request 1-minute data for 10 years",
    "**Cache data locally** to avoid repeated API calls",
    "**Respect rate limits** - add delays between large requests",
    "**Validate data** - check for missing values and outliers",
    "**Keep backups** - save important datasets locally"
]

for tip in tips:
    st.write(f"• {tip}")

# 8. EXAMPLE: BUILDING A COMPREHENSIVE SCREENER
st.header("8. 🔍 Example: Stock Screener")

def create_stock_screener(tickers, min_volume=1000000, min_price=5):
    """Create a simple stock screener"""
    results = []
    
    progress_bar = st.progress(0)
    for i, ticker in enumerate(tickers):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='5d')
            info = stock.info
            
            if not hist.empty and info:
                latest = hist.iloc[-1]
                avg_volume = hist['Volume'].mean()
                price_change = ((latest['Close'] - hist.iloc[-5]['Close']) / hist.iloc[-5]['Close']) * 100
                
                if avg_volume >= min_volume and latest['Close'] >= min_price:
                    results.append({
                        'Ticker': ticker,
                        'Price': latest['Close'],
                        'Volume': avg_volume,
                        '5D_Change_%': price_change,
                        'Market_Cap': info.get('marketCap', 'N/A'),
                        'Sector': info.get('sector', 'N/A')
                    })
            
            progress_bar.progress((i + 1) / len(tickers))
            time.sleep(0.1)  # Rate limiting
            
        except:
            continue
    
    return pd.DataFrame(results)

if st.button("Run Sample Stock Screener"):
    sample_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'ADBE', 'CRM']
    
    with st.spinner("Screening stocks..."):
        screener_results = create_stock_screener(sample_tickers)
    
    if not screener_results.empty:
        st.success("✅ Screening complete!")
        st.dataframe(screener_results)
    else:
        st.warning("No stocks met the screening criteria")

st.markdown("---")
st.info("💡 **Pro Tip**: For production applications, consider paid APIs for reliability and real-time data!")
