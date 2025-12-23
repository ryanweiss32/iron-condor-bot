import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.historical.news import NewsClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, NewsRequest, OptionSnapshotRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOptionContractsRequest
from alpaca.trading.enums import AssetStatus, ContractType
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from datetime import datetime, timedelta, date
import math
import pandas as pd
import numpy as np

# --- 1. APP CONFIGURATION ---
st.set_page_config(page_title="Iron Condor Master", layout="wide", page_icon="🦅")

st.markdown("""
<style>
    .metric-container { background-color: #1e1e1e; border: 1px solid #333; border-radius: 10px; padding: 15px; text-align: center; }
    .trade-box { background-color: #262730; padding: 20px; border-radius: 10px; border-left: 5px solid #2196F3; }
    .live-data { color: #00E676; font-weight: bold; font-family: monospace; }
    .est-data { color: #FFB74D; font-weight: bold; font-family: monospace; }
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
    st.subheader("⚙️ Trade Settings")
    wing_width = st.slider("Wing Width ($)", 1, 10, 5)
    target_iv = st.number_input("Implied Volatility (Est.)", 0.1, 1.0, 0.15, step=0.01)

# --- 4. HELPER FUNCTIONS ---
def round_to_strike(price, interval=1.0):
    return round(price / interval) * interval

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

def find_contract(client, underlying, strike, type, expiry):
    try:
        req = GetOptionContractsRequest(
            underlying_symbol=[underlying],
            status=AssetStatus.ACTIVE,
            expiration_date=expiry,
            type=type,
            strike_price_gte=strike - 0.05, 
            strike_price_lte=strike + 0.05,
            limit=1
        )
        res = client.get_option_contracts(req)
        if res.option_contracts:
            return res.option_contracts[0]
        return None
    except:
        return None

def get_live_quotes(option_symbols):
    if not option_symbols: return {}
    client = OptionHistoricalDataClient(API_KEY, SECRET_KEY)
    try:
        req = OptionSnapshotRequest(symbol_or_symbols=option_symbols)
        return client.get_option_snapshot(req)
    except:
        return {}

def plot_payoff(current_price, short_call, long_call, short_put, long_put, net_credit):
    # Determine plot range
    min_price = long_put - (wing_width * 1.5)
    max_price = long_call + (wing_width * 1.5)
    prices = np.linspace(min_price, max_price, 100)
    
    profits = []
    for price in prices:
        # 1. Put Wing Payoff
        p_long_put = max(long_put - price, 0)
        p_short_put = -max(short_put - price, 0)
        
        # 2. Call Wing Payoff
        p_short_call = -max(price - short_call, 0)
        p_long_call = max(price - long_call, 0)
        
        # Total Payoff (plus credit received)
        total = (p_long_put + p_short_put + p_short_call + p_long_call) * 100 + (net_credit * 100)
        profits.append(total)
        
    fig = go.Figure()
    
    # The Payoff Line
    fig.add_trace(go.Scatter(x=prices, y=profits, mode='lines', name='P/L at Expiry', 
                             line=dict(color='#00E676', width=3)))
    
    # Zero Line (Breakeven)
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    # Current Price Marker
    fig.add_vline(x=current_price, line_dash="dot", line_color="yellow", annotation_text="Current Price")
    
    # Styling
    fig.update_layout(
        title="Strategy Payoff Diagram (At Expiration)",
        xaxis_title="Stock Price",
        yaxis_title="Profit / Loss ($)",
        template="plotly_dark",
        height=400
    )
    
    # Add Shading
    fig.add_hrect(y0=0, y1=max(profits), fillcolor="green", opacity=0.1, line_width=0)
    fig.add_hrect(y0=min(profits), y1=0, fillcolor="red", opacity=0.1, line_width=0)
    
    return fig

# --- 5. MAIN DASHBOARD ---
st.title(f"🦅 {symbol} Strategy Center")

if symbol:
    try:
        df = get_stock_data(symbol)
        current_price = df.iloc[-1]['close']
        
        # Technicals
        df['SMA_50'] = df['close'].rolling(50).mean()
        df['Upper_Band'] = df['SMA_50'] + (df['close'].rolling(20).std() * 2)
        df['Lower_Band'] = df['SMA_50'] - (df['close'].rolling(20).std() * 2)
        df['RSI'] = 100 - (100 / (1 + (df['close'].diff().where(lambda x: x > 0, 0).rolling(14).mean() / df['close'].diff().where(lambda x: x < 0, 0).abs().rolling(14).mean())))

        # --- SCENARIO GENERATION ---
        days_to_expiry = 7
        base_move = current_price * target_iv * math.sqrt(days_to_expiry/365)
        
        short_call_strike = round_to_strike(current_price + base_move, 1.0)
        long_call_strike = short_call_strike + wing_width
        short_put_strike = round_to_strike(current_price - base_move, 1.0)
        long_put_strike = short_put_strike - wing_width

        # --- TABS ---
        tab1, tab2, tab3 = st.tabs(["📊 Trade Builder", "📉 Analysis", "📰 News"])

        with tab1:
            st.subheader(f"Strategy: Iron Condor ({days_to_expiry} DTE)")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Current Price", f"${current_price:.2f}")
            c2.metric("Call Spread", f"${short_call_strike} / ${long_call_strike}")
            c3.metric("Put Spread", f"${short_put_strike} / ${long_put_strike}")
            
            st.divider()
            
            if st.button("🔴 Build Trade & Visualize"):
                trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
                
                # Dynamic Expiry
                today = date.today()
                days_ahead = 4 - today.weekday()
                if days_ahead <= 0: days_ahead += 7
                expiry = today + timedelta(days=days_ahead)
                
                with st.spinner(f"Scanning chain for {expiry}..."):
                    legs = []
                    # Exact 4-Leg Order Definition:
                    # 1. Sell OTM Put
                    # 2. Buy Further OTM Put
                    # 3. Sell OTM Call
                    # 4. Buy Further OTM Call
                    leg_configs = [
                        (short_call_strike, ContractType.CALL, "SELL"),
                        (long_call_strike, ContractType.CALL, "BUY"),
                        (short_put_strike, ContractType.PUT, "SELL"),
                        (long_put_strike, ContractType.PUT, "BUY")
                    ]
                    
                    found_all = True
                    trade_rows = []
                    total_credit = 0.0
                    is_live = True
                    symbols = []
                    
                    # 1. Find Contracts
                    for strike, ctype, side in leg_configs:
                        c = find_contract(trading_client, symbol, strike, ctype, expiry)
                        if c:
                            legs.append(c)
                            symbols.append(c.symbol)
                        else:
                            found_all = False
                    
                    # 2. Get Data
                    if found_all:
                        quotes = get_live_quotes(symbols)
                        if not quotes: is_live = False
                        
                        for i, contract in enumerate(legs):
                            side = leg_configs[i][2]
                            strike = leg_configs[i][0]
                            
                            if is_live and contract.symbol in quotes:
                                q = quotes[contract.symbol].latest_quote
                                price = (q.ask_price + q.bid_price) / 2
                                src = "LIVE"
                            else:
                                # Estimate for fallback
                                dist = abs(current_price - strike)
                                price = max(0.1, 5.0 - (dist * 0.15)) if dist < 15 else 0.05
                                src = "EST"
                            
                            if side == "SELL": total_credit += price
                            else: total_credit -= price
                            
                            trade_rows.append({
                                "Leg": f"{side} {contract.type.value}",
                                "Strike": f"${strike}",
                                "Price": f"${price:.2f}",
                                "Source": src
                            })
                            
                        # 3. Visuals
                        c_left, c_right = st.columns([1, 1])
                        
                        with c_left:
                            st.table(pd.DataFrame(trade_rows))
                            st.markdown(f"""
                            <div class='trade-box'>
                            <p><b>Net Credit:</b> ${total_credit*100:.0f}</p>
                            <p><b>Max Risk:</b> ${(wing_width - total_credit)*100:.0f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        with c_right:
                            fig_payoff = plot_payoff(current_price, short_call_strike, long_call_strike, short_put_strike, long_put_strike, total_credit)
                            st.plotly_chart(fig_payoff, use_container_width=True)
                            
                    else:
                        st.error("Could not find all 4 legs. Market might be closed.")
            else:
                st.info("Click the button to scan the option chain.")

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
