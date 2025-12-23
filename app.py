import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import StockBarsRequest, NewsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from datetime import datetime, timedelta, date
import math
import pandas as pd
import numpy as np
import yfinance as yf

# --- 1. APP CONFIGURATION ---
st.set_page_config(page_title="Iron Condor Master", layout="wide", page_icon="🦅")

st.markdown("""
<style>
    .metric-container { background-color: #1e1e1e; border: 1px solid #333; border-radius: 10px; padding: 15px; text-align: center; }
    .trade-card { background-color: #262730; padding: 20px; border-radius: 10px; margin-bottom: 15px; }
    .call-side { border-left: 5px solid #FF5252; }
    .put-side { border-left: 5px solid #00E676; }
    .profit { color: #00E676; font-weight: bold; }
    .loss { color: #FF5252; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- 2. AUTO-LOAD KEYS ---
try:
    API_KEY = st.secrets["ALPACA_API_KEY"]
    SECRET_KEY = st.secrets["ALPACA_SECRET_KEY"]
except FileNotFoundError:
    st.error("❌ Critical Error: Could not find .streamlit/secrets.toml")
    st.stop()

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("🦅 Strategy Control")
    symbol = st.text_input("Ticker Symbol", value="SPY").upper()
    days_back = st.slider("Analysis Window (Days)", 100, 500, 365)
    
    st.divider()
    st.subheader("⚙️ Global Settings")
    wing_width = st.slider("Wing Width ($)", 1, 10, 5, help="Width of the protection wings.")
    target_iv = st.number_input("Implied Volatility (Est.)", 0.1, 1.0, 0.15, step=0.01)

# --- 4. MATH FUNCTIONS ---
def get_next_trading_day(days_offset):
    target = date.today()
    added = 0
    while added < days_offset:
        target += timedelta(days=1)
        if target.weekday() < 5:
            added += 1
    return target

def round_to_strike(price, interval=1.0):
    return round(price / interval) * interval

def normal_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def calculate_probability(price, upper, lower, iv, days):
    t = days / 365.0
    vol = iv * math.sqrt(t)
    if vol == 0: return 0
    z_upper = math.log(upper / price) / vol
    z_lower = math.log(lower / price) / vol
    prob = normal_cdf(z_upper) - normal_cdf(z_lower)
    return prob * 100

def get_stock_data(sym):
    stock_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
    req = StockBarsRequest(
        symbol_or_symbols=[sym],
        timeframe=TimeFrame.Day,
        start=datetime.now() - timedelta(days=days_back),
        end=datetime.now(),
        feed=DataFeed.IEX 
    )
    return stock_client.get_stock_bars(req).df.reset_index()

def get_news_data(sym):
    news_client = NewsClient(API_KEY, SECRET_KEY)
    try:
        news_req = NewsRequest(symbols=sym, limit=5)
        return news_client.get_news(news_req).news
    except:
        return []

def plot_payoff(current_price, short_call, long_call, short_put, long_put, net_credit):
    min_price = long_put - (wing_width * 1.5)
    max_price = long_call + (wing_width * 1.5)
    prices = np.linspace(min_price, max_price, 100)
    profits = []
    for price in prices:
        p_long_put = max(long_put - price, 0)
        p_short_put = -max(short_put - price, 0)
        p_short_call = -max(price - short_call, 0)
        p_long_call = max(price - long_call, 0)
        total = (p_long_put + p_short_put + p_short_call + p_long_call) * 100 + (net_credit * 100)
        profits.append(total)
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices, y=profits, mode='lines', name='P/L Line', line=dict(color='#00E676', width=3)))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=current_price, line_dash="dot", line_color="yellow", annotation_text="Price")
    fig.update_layout(title="Profit/Loss Diagram", template="plotly_dark", height=300, margin=dict(l=10, r=10, t=30, b=10))
    return fig

# --- 5. YAHOO FINANCE LOGIC ---
def get_option_price_yf(ticker_obj, expiry_date, strike, option_type):
    try:
        opt = ticker_obj.option_chain(expiry_date)
        data = opt.calls if option_type == "call" else opt.puts
        contract_row = data.iloc[(data['strike'] - strike).abs().argsort()[:1]]
        if not contract_row.empty:
            price = contract_row['lastPrice'].values[0]
            actual_strike = contract_row['strike'].values[0]
            return price, actual_strike
        return 0.0, strike
    except Exception as e:
        return 0.0, strike

def get_closest_expiry_yf(ticker_obj, target_date):
    avail_dates = ticker_obj.options
    valid_dates = [datetime.strptime(d, "%Y-%m-%d").date() for d in avail_dates]
    closest = min(valid_dates, key=lambda x: abs(x - target_date))
    return closest.strftime("%Y-%m-%d")

def generate_scenario_card(symbol, current_price, days_offset, scenario_name):
    # 1. Setup Data
    yf_ticker = yf.Ticker(symbol)
    
    # 2. Find Correct Expiry
    target_date = get_next_trading_day(days_offset)
    try:
        expiry_str = get_closest_expiry_yf(yf_ticker, target_date)
    except:
        st.error("No options data found in Yahoo Finance.")
        return

    days_to_expiry = (datetime.strptime(expiry_str, "%Y-%m-%d").date() - date.today()).days
    if days_to_expiry < 1: days_to_expiry = 1
    
    # 3. Calculate Strikes
    move = current_price * target_iv * math.sqrt(days_to_expiry/365.0)
    s_call = round_to_strike(current_price + move, 1.0)
    l_call = s_call + wing_width
    s_put = round_to_strike(current_price - move, 1.0)
    l_put = s_put - wing_width
    
    st.markdown(f"### {scenario_name} (Exp: {expiry_str})")
    
    # 4. Fetch Prices from Yahoo
    legs = [
        (s_call, "call", "SELL"),
        (l_call, "call", "BUY"),
        (s_put, "put", "SELL"),
        (l_put, "put", "BUY")
    ]
    
    total_credit = 0.0
    prices = []
    
    for strike, otype, side in legs:
        price, actual_strike = get_option_price_yf(yf_ticker, expiry_str, strike, otype)
        prices.append(price)
        if side == "SELL": total_credit += price
        else: total_credit -= price
    
    # 5. Display
    max_profit = total_credit * 100
    max_loss = (wing_width * 100) - max_profit
    pop = calculate_probability(current_price, s_call, s_put, target_iv, days_to_expiry)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Max Profit", f"${max_profit:.0f}")
    c2.metric("Max Loss", f"${max_loss:.0f}")
    c3.metric("Win Prob", f"{pop:.1f}%")
    
    col_call, col_put = st.columns(2)
    with col_call:
        st.markdown(f"""<div class='trade-card call-side'>
        <strong style='color:#FF5252'>Bear Call Side</strong><br>
        Sell ${s_call} Call (~${prices[0]:.2f})<br>Buy ${l_call} Call (~${prices[1]:.2f})
        </div>""", unsafe_allow_html=True)
    with col_put:
        st.markdown(f"""<div class='trade-card put-side'>
        <strong style='color:#00E676'>Bull Put Side</strong><br>
        Sell ${s_put} Put (~${prices[2]:.2f})<br>Buy ${l_put} Put (~${prices[3]:.2f})
        </div>""", unsafe_allow_html=True)
        
    with st.expander("See Graph"):
         # FIX: Added unique key using scenario_name to prevent duplicate ID error
         st.plotly_chart(plot_payoff(current_price, s_call, l_call, s_put, l_put, total_credit), use_container_width=True, key=f"chart_{scenario_name}")


# --- 6. MAIN DASHBOARD ---
st.title(f"🦅 {symbol} Multi-Timeframe Analyzer")

if symbol:
    try:
        # Load Stock Data
        df = get_stock_data(symbol)
        current_price = df.iloc[-1]['close']
        
        # Technicals for Chart
        df['SMA_50'] = df['close'].rolling(50).mean()
        df['Upper_Band'] = df['SMA_50'] + (df['close'].rolling(20).std() * 2)
        df['Lower_Band'] = df['SMA_50'] - (df['close'].rolling(20).std() * 2)
        df['RSI'] = 100 - (100 / (1 + (df['close'].diff().where(lambda x: x > 0, 0).rolling(14).mean() / df['close'].diff().where(lambda x: x < 0, 0).abs().rolling(14).mean())))

        # --- TABS ---
        tab1, tab2, tab3 = st.tabs(["📊 Trade Builder", "📉 Chart", "📰 News"])

        with tab1:
            st.write(f"Current Price: **${current_price:.2f}** | Data Source: Yahoo Finance (Last Closing Price)")
            
            if st.button("🔴 GENERATE ALL SCENARIOS"):
                
                t1, t2, t3 = st.tabs(["1 Day (Gamma Scalp)", "2 Days (Swing)", "7 Days (Standard)"])
                
                with t1:
                    with st.spinner("Fetching Yahoo Finance Data..."):
                        generate_scenario_card(symbol, current_price, 1, "1-Day Expiry")
                
                with t2:
                    with st.spinner("Fetching Yahoo Finance Data..."):
                        generate_scenario_card(symbol, current_price, 2, "2-Day Expiry")
                        
                with t3:
                    with st.spinner("Fetching Yahoo Finance Data..."):
                        generate_scenario_card(symbol, current_price, 7, "7-Day Expiry")

            else:
                st.info("Click the button to fetch last known market prices.")

        with tab2:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['Upper_Band'], line=dict(color='orange', width=1, dash='dot'), name='Upper Band'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['Lower_Band'], line=dict(color='orange', width=1, dash='dot'), name='Lower Band'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['RSI'], line=dict(color='#A020F0', width=2), name='RSI'), row=2, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
            fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            news = get_news_data(symbol)
            if news:
                for article in news:
                    if isinstance(article, tuple): article = article[1]
                    try:
                        st.markdown(f"**[{article.headline}]({article.url})**")
                        st.caption(f"Source: {article.source} | {article.created_at.strftime('%Y-%m-%d')}")
                        st.divider()
                    except: pass
            else:
                st.write("No news found.")

    except Exception as e:
        st.error(f"Error: {e}")
