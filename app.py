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

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Iron Condor Strategy Builder", layout="wide", page_icon="🦅")

st.markdown("""
<style>
    /* Main Container Styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 30px;
        color: white;
        text-align: center;
    }
    
    /* Educational Cards */
    .info-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 25px;
        border-radius: 12px;
        margin: 15px 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    .info-card h3 {
        color: #ffd700;
        margin-top: 0;
        font-size: 1.3em;
    }
    
    .info-card ul {
        margin: 10px 0;
        padding-left: 20px;
    }
    
    /* Strategy Cards */
    .strategy-card {
        background: #1a1a2e;
        border: 2px solid #16213e;
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.4);
    }
    
    .strategy-header {
        font-size: 1.5em;
        font-weight: bold;
        margin-bottom: 15px;
        color: #00d4ff;
        text-align: center;
    }
    
    /* Leg Cards */
    .leg-card {
        background: #262730;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid;
    }
    
    .call-sell { border-color: #ff4757; background: rgba(255, 71, 87, 0.1); }
    .call-buy { border-color: #ff6348; background: rgba(255, 99, 72, 0.1); }
    .put-sell { border-color: #2ed573; background: rgba(46, 213, 115, 0.1); }
    .put-buy { border-color: #1e90ff; background: rgba(30, 144, 255, 0.1); }
    
    .leg-header {
        font-size: 1.2em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    .leg-details {
        font-size: 0.95em;
        line-height: 1.6;
    }
    
    /* Metrics */
    .metric-row {
        display: flex;
        gap: 15px;
        margin: 20px 0;
        flex-wrap: wrap;
    }
    
    .metric-box {
        flex: 1;
        min-width: 150px;
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    
    .metric-label {
        font-size: 0.85em;
        color: #a0a0a0;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-size: 1.8em;
        font-weight: bold;
        color: #00ff88;
    }
    
    .metric-value.negative {
        color: #ff4757;
    }
    
    /* Alert Boxes */
    .alert-box {
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
        font-weight: 500;
    }
    
    .alert-info {
        background: rgba(52, 152, 219, 0.2);
        border-left: 4px solid #3498db;
        color: #3498db;
    }
    
    .alert-success {
        background: rgba(46, 213, 115, 0.2);
        border-left: 4px solid #2ed573;
        color: #2ed573;
    }
    
    .alert-warning {
        background: rgba(255, 193, 7, 0.2);
        border-left: 4px solid #ffc107;
        color: #ffc107;
    }
    
    /* Visual Strategy Diagram */
    .strategy-visual {
        background: #1a1a2e;
        padding: 30px;
        border-radius: 15px;
        margin: 20px 0;
        text-align: center;
    }
    
    .price-ladder {
        display: flex;
        flex-direction: column;
        gap: 15px;
        margin: 20px auto;
        max-width: 400px;
    }
    
    .strike-level {
        padding: 12px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.1em;
    }
    
    .strike-long-call { background: #ff6348; color: white; }
    .strike-short-call { background: #ff4757; color: white; }
    .strike-current { background: #ffd700; color: #1a1a2e; border: 3px solid white; }
    .strike-short-put { background: #2ed573; color: white; }
    .strike-long-put { background: #1e90ff; color: white; }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 15px 30px;
        font-size: 1.1em;
        border-radius: 8px;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# --- AUTO-LOAD KEYS ---
try:
    API_KEY = st.secrets["ALPACA_API_KEY"]
    SECRET_KEY = st.secrets["ALPACA_SECRET_KEY"]
except:
    st.error("❌ Critical Error: Could not find .streamlit/secrets.toml")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 🦅 Strategy Control Panel")
    symbol = st.text_input("📈 Ticker Symbol", value="SPY").upper()
    days_back = st.slider("📊 Analysis Window (Days)", 100, 500, 365)
    
    st.divider()
    st.markdown("### ⚙️ Strategy Parameters")
    wing_width = st.slider("🔧 Wing Width ($)", 1, 10, 5, 
                          help="Distance between short and long strikes. Wider wings = lower risk but lower profit.")
    target_iv = st.number_input("📉 Implied Volatility (Est.)", 0.1, 1.0, 0.15, step=0.01,
                               help="Expected volatility. Higher IV = wider spreads needed.")
    
    st.divider()
    st.markdown("### 📚 Quick Guide")
    with st.expander("What is an Iron Condor?"):
        st.write("""
        An Iron Condor is a neutral options strategy that profits when the underlying stock stays within a range.
        
        **Components:**
        - Bear Call Spread (above current price)
        - Bull Put Spread (below current price)
        
        **Best Used When:**
        - Expecting low volatility
        - Stock trading sideways
        - Want defined risk/reward
        """)

# --- HELPER FUNCTIONS ---
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
    except:
        return 0.0, strike

def get_closest_expiry_yf(ticker_obj, target_date):
    avail_dates = ticker_obj.options
    valid_dates = [datetime.strptime(d, "%Y-%m-%d").date() for d in avail_dates]
    closest = min(valid_dates, key=lambda x: abs(x - target_date))
    return closest.strftime("%Y-%m-%d")

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
    
    # Profit/Loss line
    fig.add_trace(go.Scatter(
        x=prices, y=profits, 
        mode='lines', 
        name='P/L',
        line=dict(color='#00d4ff', width=4),
        fill='tozeroy',
        fillcolor='rgba(0, 212, 255, 0.1)'
    ))
    
    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="white", line_width=2, opacity=0.5)
    
    # Current price
    fig.add_vline(x=current_price, line_dash="dot", line_color="#ffd700", 
                  line_width=3, annotation_text="Current Price", annotation_position="top")
    
    # Strike levels
    for strike, name, color in [
        (long_call, "Long Call", "#ff6348"),
        (short_call, "Short Call", "#ff4757"),
        (short_put, "Short Put", "#2ed573"),
        (long_put, "Long Put", "#1e90ff")
    ]:
        fig.add_vline(x=strike, line_dash="dot", line_color=color, 
                     line_width=2, opacity=0.6)
    
    fig.update_layout(
        title="Profit/Loss at Expiration",
        template="plotly_dark",
        height=400,
        xaxis_title="Stock Price at Expiration",
        yaxis_title="Profit/Loss ($)",
        hovermode='x unified',
        plot_bgcolor='#1a1a2e',
        paper_bgcolor='#0f0f1e'
    )
    
    return fig

def create_strategy_visual(current_price, short_call, long_call, short_put, long_put):
    return f"""
    <div class='strategy-visual'>
        <h3 style='color: #00d4ff;'>📊 Strike Price Ladder</h3>
        <div class='price-ladder'>
            <div class='strike-level strike-long-call'>
                🔒 BUY ${long_call:.2f} Call (Protection)
            </div>
            <div class='strike-level strike-short-call'>
                💰 SELL ${short_call:.2f} Call (Income)
            </div>
            <div class='strike-level strike-current'>
                ⭐ Current: ${current_price:.2f}
            </div>
            <div class='strike-level strike-short-put'>
                💰 SELL ${short_put:.2f} Put (Income)
            </div>
            <div class='strike-level strike-long-put'>
                🔒 BUY ${long_put:.2f} Put (Protection)
            </div>
        </div>
        <p style='color: #a0a0a0; margin-top: 20px;'>
            ✅ Profit Zone: Between ${short_put:.2f} and ${short_call:.2f}<br>
            ⚠️ Maximum profit if stock stays between these strikes at expiration
        </p>
    </div>
    """

def generate_scenario_card(symbol, current_price, days_offset, scenario_name):
    yf_ticker = yf.Ticker(symbol)
    
    target_date = get_next_trading_day(days_offset)
    try:
        expiry_str = get_closest_expiry_yf(yf_ticker, target_date)
    except:
        st.error("❌ No options data found in Yahoo Finance.")
        return

    days_to_expiry = (datetime.strptime(expiry_str, "%Y-%m-%d").date() - date.today()).days
    if days_to_expiry < 1: days_to_expiry = 1
    
    # Calculate strikes
    move = current_price * target_iv * math.sqrt(days_to_expiry/365.0)
    s_call = round_to_strike(current_price + move, 1.0)
    l_call = s_call + wing_width
    s_put = round_to_strike(current_price - move, 1.0)
    l_put = s_put - wing_width
    
    st.markdown(f"<div class='strategy-header'>🎯 {scenario_name}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='alert-box alert-info'>📅 Expiration: {expiry_str} ({days_to_expiry} days away)</div>", unsafe_allow_html=True)
    
    # Fetch prices
    legs = [
        (s_call, "call", "SELL", "Short Call"),
        (l_call, "call", "BUY", "Long Call"),
        (s_put, "put", "SELL", "Short Put"),
        (l_put, "put", "BUY", "Long Put")
    ]
    
    total_credit = 0.0
    prices = []
    
    for strike, otype, side, _ in legs:
        price, actual_strike = get_option_price_yf(yf_ticker, expiry_str, strike, otype)
        prices.append(price)
        if side == "SELL": total_credit += price
        else: total_credit -= price
    
    # Calculate metrics
    max_profit = total_credit * 100
    max_loss = (wing_width * 100) - max_profit
    pop = calculate_probability(current_price, s_call, s_put, target_iv, days_to_expiry)
    risk_reward = abs(max_profit / max_loss) if max_loss != 0 else 0
    
    # Display strategy visual
    st.markdown(create_strategy_visual(current_price, s_call, l_call, s_put, l_put), unsafe_allow_html=True)
    
    # Metrics row
    st.markdown("<div class='metric-row'>", unsafe_allow_html=True)
    cols = st.columns(4)
    
    with cols[0]:
        st.markdown(f"""
        <div class='metric-box'>
            <div class='metric-label'>💵 Max Profit</div>
            <div class='metric-value'>${max_profit:.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        st.markdown(f"""
        <div class='metric-box'>
            <div class='metric-label'>⚠️ Max Loss</div>
            <div class='metric-value negative'>${max_loss:.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[2]:
        st.markdown(f"""
        <div class='metric-box'>
            <div class='metric-label'>🎲 Win Probability</div>
            <div class='metric-value'>{pop:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[3]:
        st.markdown(f"""
        <div class='metric-box'>
            <div class='metric-label'>⚖️ Risk/Reward</div>
            <div class='metric-value'>1:{risk_reward:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Detailed legs breakdown
    st.markdown("### 📋 Trade Legs Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class='leg-card call-sell'>
            <div class='leg-header'>🔴 SELL Call (Bear Call Spread - Top)</div>
            <div class='leg-details'>
                <strong>Strike:</strong> ${s_call:.2f}<br>
                <strong>Premium Received:</strong> ${prices[0]:.2f}<br>
                <strong>Purpose:</strong> Collect income if stock stays below this level<br>
                <strong>Risk:</strong> Unlimited if unprotected (but we have protection!)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class='leg-card call-buy'>
            <div class='leg-header'>🟠 BUY Call (Protection)</div>
            <div class='leg-details'>
                <strong>Strike:</strong> ${l_call:.2f}<br>
                <strong>Premium Paid:</strong> ${prices[1]:.2f}<br>
                <strong>Purpose:</strong> Caps maximum loss from short call<br>
                <strong>Protects:</strong> Against unlimited upside risk
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='leg-card put-sell'>
            <div class='leg-header'>🟢 SELL Put (Bull Put Spread - Bottom)</div>
            <div class='leg-details'>
                <strong>Strike:</strong> ${s_put:.2f}<br>
                <strong>Premium Received:</strong> ${prices[2]:.2f}<br>
                <strong>Purpose:</strong> Collect income if stock stays above this level<br>
                <strong>Risk:</strong> Large if unprotected (but we have protection!)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class='leg-card put-buy'>
            <div class='leg-header'>🔵 BUY Put (Protection)</div>
            <div class='leg-details'>
                <strong>Strike:</strong> ${l_put:.2f}<br>
                <strong>Premium Paid:</strong> ${prices[3]:.2f}<br>
                <strong>Purpose:</strong> Caps maximum loss from short put<br>
                <strong>Protects:</strong> Against large downside moves
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Strategy explanation
    st.markdown("""
    <div class='alert-box alert-success'>
        <strong>✅ How This Strategy Works:</strong><br>
        • You receive a NET CREDIT upfront (the premium collected from selling exceeds what you pay for protection)<br>
        • Maximum profit occurs if the stock stays between your short strikes at expiration<br>
        • Your risk is limited to the wing width minus the credit received<br>
        • This is a NEUTRAL strategy - you want the stock to stay calm and not move much
    </div>
    """, unsafe_allow_html=True)
    
    # Profit/Loss diagram
    with st.expander("📈 View Profit/Loss Diagram", expanded=False):
        st.plotly_chart(
            plot_payoff(current_price, s_call, l_call, s_put, l_put, total_credit), 
            use_container_width=True, 
            key=f"chart_{scenario_name}"
        )

# --- MAIN DASHBOARD ---
st.markdown("""
<div class='main-header'>
    <h1 style='margin: 0; font-size: 2.5em;'>🦅 Iron Condor Strategy Builder</h1>
    <p style='margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.9;'>
        Generate defined-risk, neutral income strategies with real market data
    </p>
</div>
""", unsafe_allow_html=True)

# Educational Section
with st.expander("📚 Learn: What is an Iron Condor?", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='info-card'>
            <h3>🎯 Core Concept</h3>
            <p>An Iron Condor combines two credit spreads:</p>
            <ul>
                <li><strong>Bear Call Spread:</strong> Sell a call + Buy a higher call</li>
                <li><strong>Bull Put Spread:</strong> Sell a put + Buy a lower put</li>
            </ul>
            <p>You profit if the stock stays between your short strikes until expiration.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-card'>
            <h3>💰 Profit & Loss</h3>
            <ul>
                <li><strong>Max Profit:</strong> Net credit received (happens if stock stays in range)</li>
                <li><strong>Max Loss:</strong> Wing width - Net credit (if stock moves outside range)</li>
                <li><strong>Breakeven:</strong> Two points - at short strikes ± net credit</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='info-card'>
            <h3>✅ Best Used When</h3>
            <ul>
                <li>Expecting <strong>low volatility</strong></li>
                <li>Stock trading <strong>sideways</strong></li>
                <li>Want <strong>defined risk</strong> (no surprises)</li>
                <li>Looking for <strong>income generation</strong></li>
                <li>Short time frames (7-45 days)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-card'>
            <h3>⚠️ Key Risks</h3>
            <ul>
                <li><strong>Large moves:</strong> Big price swings can hit max loss</li>
                <li><strong>Early assignment:</strong> Short options can be assigned early</li>
                <li><strong>Volatility expansion:</strong> Spreads widen when IV increases</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if symbol:
    try:
        df = get_stock_data(symbol)
        current_price = df.iloc[-1]['close']
        
        # Calculate technical indicators
        df['SMA_50'] = df['close'].rolling(50).mean()
        df['Upper_Band'] = df['SMA_50'] + (df['close'].rolling(20).std() * 2)
        df['Lower_Band'] = df['SMA_50'] - (df['close'].rolling(20).std() * 2)
        df['RSI'] = 100 - (100 / (1 + (df['close'].diff().where(lambda x: x > 0, 0).rolling(14).mean() / 
                                       df['close'].diff().where(lambda x: x < 0, 0).abs().rolling(14).mean())))

        # Display current price
        st.markdown(f"""
        <div class='alert-box alert-info'>
            <strong>📊 {symbol} Current Price:</strong> ${current_price:.2f} | 
            <strong>Data Source:</strong> Alpaca Markets (Last Closing Price) | 
            <strong>Options Data:</strong> Yahoo Finance
        </div>
        """, unsafe_allow_html=True)

        # Main tabs
        tab1, tab2, tab3 = st.tabs(["🎯 Strategy Builder", "📈 Technical Chart", "📰 Market News"])

        with tab1:
            st.markdown("### Generate Iron Condor Strategies")
            
            if st.button("🚀 GENERATE ALL TIMEFRAMES", use_container_width=True):
                t1, t2, t3 = st.tabs(["⚡ 1-Day (Gamma Scalp)", "📊 2-Day (Swing Trade)", "📅 7-Day (Standard)"])
                
                with t1:
                    st.markdown("<div class='alert-box alert-warning'>⚡ Ultra-short timeframe - High risk, high theta decay</div>", unsafe_allow_html=True)
                    with st.spinner("Fetching market data..."):
                        generate_scenario_card(symbol, current_price, 1, "1-Day Expiry")
                
                with t2:
                    st.markdown("<div class='alert-box alert-warning'>📊 Short-term trade - Quick profits or losses</div>", unsafe_allow_html=True)
                    with st.spinner("Fetching market data..."):
                        generate_scenario_card(symbol, current_price, 2, "2-Day Expiry")
                
                with t3:
                    st.markdown("<div class='alert-box alert-success'>📅 Ideal timeframe - Balance of premium and time</div>", unsafe_allow_html=True)
                    with st.spinner("Fetching market data..."):
                        generate_scenario_card(symbol, current_price, 7, "7-Day Expiry")
            else:
                st.info("👆 Click the button above to generate iron condor strategies with live market prices")

        with tab2:
            st.markdown("### Technical Analysis")
            fig = make_subplots(
                rows=2, cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.05,
                row_heights=[0.7, 0.3],
                subplot_titles=('Price Action with Bollinger Bands', 'RSI Indicator')
            )
            
            # Candlestick chart
            fig.add_trace(go.Candlestick(
                x=df['timestamp'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price'
            ), row=1, col=1)
            
            # Bollinger Bands
            fig.add_trace(go.Scatter(
                x=df['timestamp'], 
                y=df['Upper_Band'],
                line=dict(color='rgba(255, 152, 0, 0.5)', width=1, dash='dot'),
                name='Upper Band'
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['Lower_Band'],
                line=dict(color='rgba(255, 152, 0, 0.5)', width=1, dash='dot'),
                name='Lower Band',
                fill='tonexty',
                fillcolor='rgba(255, 152, 0, 0.1)'
            ), row=1, col=1)
            
            # RSI
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
