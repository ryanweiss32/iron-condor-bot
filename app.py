import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date
import math
import pandas as pd
import numpy as np
import requests
import time
from functools import wraps

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Iron Condor Strategy Builder", layout="wide", page_icon="🦅")

# --- LOAD SECRETS (POLYGON) ---
try:
    MASSIVE_API_KEY = st.secrets["MASSIVE_API_KEY"]
except:
    st.error("❌ Critical Error: Could not find MASSIVE_API_KEY in secrets.")
    st.stop()

MASSIVE_BASE_URL = "https://api.polygon.io"

# --- RATE LIMITING & API WRAPPER ---
def make_api_request(url, params, max_retries=3):
    """
    Wrapper to handle API requests with rate limiting (429) backoff.
    """
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=10)
            
            # If successful, return JSON
            if resp.status_code == 200:
                return resp.json()
            
            # If Rate Limited (429), wait and retry
            elif resp.status_code == 429:
                wait_time = (2 ** attempt) + 1  # Exponential backoff: 2s, 3s, 5s...
                time.sleep(wait_time)
                continue
            
            # If other error, print and return None
            else:
                return None
                
        except requests.exceptions.RequestException:
            time.sleep(1)
            continue
            
    return None

# --- CACHED DATA FUNCTIONS ---
@st.cache_data(ttl=300)  # Cache for 5 minutes to save API calls
def get_stock_data_polygon(symbol, limit=365):
    """Fetch historical stock data with caching."""
    end_date = date.today().strftime("%Y-%m-%d")
    start_date = (date.today() - timedelta(days=limit*2)).strftime("%Y-%m-%d") # Buffer for weekends
    
    url = f"{MASSIVE_BASE_URL}/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
    params = {
        "apiKey": MASSIVE_API_KEY,
        "adjusted": "true",
        "sort": "asc",
        "limit": 5000
    }
    
    data = make_api_request(url, params)
    
    if data and "results" in data and len(data["results"]) > 0:
        df = pd.DataFrame(data["results"])
        df["t"] = pd.to_datetime(df["t"], unit="ms")
        df = df.rename(columns={"t": "date", "c": "close", "h": "high", "l": "low", "o": "open", "v": "volume"})
        return df
    return pd.DataFrame()

@st.cache_data(ttl=60)  # Cache current price for 1 minute
def get_current_price_polygon(symbol):
    """Fetch real-time/delayed current price."""
    url = f"{MASSIVE_BASE_URL}/v2/aggs/ticker/{symbol}/prev"
    params = {"apiKey": MASSIVE_API_KEY}
    
    data = make_api_request(url, params)
    
    if data and "results" in data and len(data["results"]) > 0:
        return data["results"][0]["c"]
    return 0.0

@st.cache_data(ttl=60)
def get_option_price_polygon(symbol, expiration, strike, option_type):
    """
    Attempt to fetch live option price. 
    Returns None if fails (so we can fallback to estimation).
    """
    # Format ticker for Polygon Options API: O:SPY231222C00450000
    try:
        parsed_date = datetime.strptime(expiration, "%Y-%m-%d")
        date_str = parsed_date.strftime("%y%m%d")
        type_str = "C" if option_type == "call" else "P"
        price_str = f"{int(strike*1000):08d}"
        
        ticker = f"O:{symbol}{date_str}{type_str}{price_str}"
        
        url = f"{MASSIVE_BASE_URL}/v2/aggs/ticker/{ticker}/prev"
        params = {"apiKey": MASSIVE_API_KEY}
        
        data = make_api_request(url, params)
        
        if data and "results" in data and len(data["results"]) > 0:
            return data["results"][0]["c"]
            
    except Exception:
        pass
        
    return None

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

def calculate_historical_volatility(df, window=30):
    """Calculate annualized historical volatility from price data"""
    if len(df) < window:
        window = len(df)
    
    if window == 0: return 0.25

    # Create a copy to avoid modifying original
    df_copy = df.copy()
    
    # Calculate daily returns
    df_copy['returns'] = df_copy['close'].pct_change()
    
    # Calculate standard deviation of returns
    volatility = df_copy['returns'].tail(window).std()
    
    # Annualize (252 trading days per year)
    annual_volatility = volatility * np.sqrt(252)
    
    return annual_volatility if not np.isnan(annual_volatility) else 0.25

def estimate_option_price(sym, expiry, strike, option_type, current_price_cache=None, iv_cache=None):
    """Estimate option price using Black-Scholes when real data unavailable"""
    try:
        # Use cached values if passed to avoid re-fetching
        current_price = current_price_cache if current_price_cache else get_current_price_polygon(sym)
        
        if current_price == 0:
            return 0.0
        
        # Use cached IV or default
        iv = iv_cache if iv_cache else 0.25
        
        # Calculate time to expiry
        exp_date = datetime.strptime(expiry, "%Y-%m-%d").date()
        days_to_expiry = (exp_date - date.today()).days
        if days_to_expiry < 1:
            days_to_expiry = 1
        t = days_to_expiry / 365.0
        
        # Risk-free rate (approximate)
        r = 0.05
        
        # Black-Scholes calculation
        d1 = (math.log(current_price / strike) + (r + 0.5 * iv**2) * t) / (iv * math.sqrt(t))
        d2 = d1 - iv * math.sqrt(t)
        
        if option_type == "call":
            price = current_price * normal_cdf(d1) - strike * math.exp(-r * t) * normal_cdf(d2)
        else:  # put
            price = strike * math.exp(-r * t) * normal_cdf(-d2) - current_price * normal_cdf(-d1)
        
        # Add a tiny bit of "slippage" or minimum value for OTM options
        return max(0.01, price)
        
    except Exception as e:
        # Emergency fallback: Intrinsic value
        if current_price_cache:
            if option_type == "call":
                return max(0, current_price_cache - strike)
            else:
                return max(0, strike - current_price_cache)
        return 0.0

def get_option_price_safe(symbol, expiration, strike, option_type, current_price, iv):
    """Smart wrapper: Tries API first, falls back to Black-Scholes estimate"""
    # 1. Try Live/Prev Data (if user has subscription)
    price = get_option_price_polygon(symbol, expiration, strike, option_type)
    
    # 2. If API returns valid price, use it
    if price is not None:
        return price, False  # False = Not Estimated
        
    # 3. Fallback to Black-Scholes
    estimated = estimate_option_price(symbol, expiration, strike, option_type, current_price, iv)
    return estimated, True # True = Estimated

# Test API connection
def test_api_connection():
    """Test if API key is valid"""
    url = f"{MASSIVE_BASE_URL}/v2/aggs/ticker/SPY/prev"
    params = {"apiKey": MASSIVE_API_KEY}
    data = make_api_request(url, params)
    
    if data:
        st.sidebar.success("✅ API Connection Working!")
        return True
    else:
        st.sidebar.error("❌ Connection Failed or Rate Limited")
        return False

# CSS STYLING
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
    
    /* Stock Suggestion Cards */
    .suggestion-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        border-left: 5px solid;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    
    .suggestion-hot { border-color: #ff4757; }
    .suggestion-moderate { border-color: #ffa502; }
    .suggestion-cool { border-color: #1e90ff; }
    
    .stock-header {
        font-size: 1.4em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    .stock-metrics {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 10px;
        margin: 15px 0;
    }
    
    .stock-metric-item {
        background: rgba(255, 255, 255, 0.1);
        padding: 10px;
        border-radius: 8px;
        text-align: center;
    }
    
    .recommendation-badge {
        display: inline-block;
        padding: 8px 15px;
        border-radius: 20px;
        font-weight: bold;
        margin: 5px;
        font-size: 0.9em;
    }
    
    .badge-buy { background: #2ed573; color: white; }
    .badge-watch { background: #ffa502; color: white; }
    .badge-caution { background: #ff4757; color: white; }
    
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

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 🦅 Strategy Control Panel")
    
    # Test API Connection
    with st.expander("🔌 Test API Connection", expanded=True):
        if st.button("Test Polygon API"):
            test_api_connection()
    
    symbol = st.text_input("📈 Ticker Symbol", value="SPY").upper()
    days_back = st.slider("📊 Analysis Window (Days)", 100, 500, 365)
    
    st.divider()
    
    # API Status Check
    with st.expander("🔌 API Connection Status", expanded=False):
        st.markdown("""
        **Options Data Requirements:**
        - Polygon.io subscription with **Options tier**
        - If you see pricing errors, check your plan includes options data
        - Free tier only includes stocks data
        
        The app will automatically use estimated prices if live data is unavailable.
        """)
    
    st.divider()
    st.markdown("### ⚙️ Strategy Parameters")
    
    strategy_profile = st.radio("🎯 Strategy Profile", 
                                ["Show All 3", "Aggressive Only", "Balanced Only", "Conservative Only"],
                                help="Choose which risk profiles to display")
    
    iv_mode = st.radio("IV Calculation Method", 
                       ["Auto (Historical)", "Manual Override"],
                       help="Auto calculates IV from recent price action. Manual lets you set your own.")
    
    target_iv = 0.20
    if iv_mode == "Manual Override":
        target_iv = st.number_input("📉 Implied Volatility", 0.1, 1.0, 0.20, step=0.01,
                                   help="Expected volatility. Higher IV = wider spreads needed.")
    
    st.divider()
    st.markdown("### 🎯 Strike Selection")
    strike_stdev = st.slider("Standard Deviations", 0.5, 2.0, 1.0, step=0.1,
                             help="How far from current price to place short strikes. Higher = safer but less premium.")
    
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

# --- MAIN LOGIC ---
st.markdown('<div class="main-header"><h1>🦅 Advanced Iron Condor Builder</h1><p>Analyze, Build, and Optimize Neutral Income Strategies</p></div>', unsafe_allow_html=True)

# Fetch Data
with st.spinner(f"📊 Analyzing {symbol} market data..."):
    df = get_stock_data_polygon(symbol, days_back)
    current_price = get_current_price_polygon(symbol)

if df.empty or current_price == 0:
    st.error(f"❌ Could not fetch data for {symbol}. Check ticker symbol or API Key.")
    st.stop()

# Calculations
if iv_mode == "Auto (Historical)":
    calculated_iv = calculate_historical_volatility(df)
    used_iv = calculated_iv
else:
    used_iv = target_iv
    
# Determine Expiration (Targeting ~30-45 DTE usually, but for this simplified tool we'll pick next Friday + 30 days)
target_date = get_next_trading_day(30)
expiration_str = target_date.strftime("%Y-%m-%d")
days_to_expiry = (target_date - date.today()).days

# Calculate Expected Move
expected_move = current_price * used_iv * math.sqrt(days_to_expiry/365.0)

# Display Market Data
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Current Price", f"${current_price:.2f}")
with col2:
    st.metric("Implied Volatility (Ann.)", f"{used_iv:.1%}")
with col3:
    st.metric("Days to Expiration", f"{days_to_expiry} Days", f"{expiration_str}")
with col4:
    st.metric("Expected Move (±1σ)", f"±${expected_move:.2f}")

st.divider()

# --- STRATEGY GENERATION ---

def generate_condor(name, risk_factor):
    """
    Generates an Iron Condor setup based on risk factor.
    Risk Factor: 1.0 = Standard (1 SD), 0.5 = Aggressive, 1.5 = Conservative
    """
    adjusted_move = expected_move * risk_factor * strike_stdev
    
    # Strikes
    short_call = round_to_strike(current_price + adjusted_move)
    long_call = round_to_strike(short_call + (short_call * 0.05)) # 5% width
    
    short_put = round_to_strike(current_price - adjusted_move)
    long_put = round_to_strike(short_put - (short_put * 0.05)) # 5% width
    
    # Price Estimation with Retry Logic
    sc_price, sc_est = get_option_price_safe(symbol, expiration_str, short_call, "call", current_price, used_iv)
    lc_price, lc_est = get_option_price_safe(symbol, expiration_str, long_call, "call", current_price, used_iv)
    sp_price, sp_est = get_option_price_safe(symbol, expiration_str, short_put, "put", current_price, used_iv)
    lp_price, lp_est = get_option_price_safe(symbol, expiration_str, long_put, "put", current_price, used_iv)
    
    # Logic fix for credit calculation
    # Credit = (Short Put + Short Call) - (Long Put + Long Call)
    net_credit = (sc_price + sp_price) - (lc_price + lp_price)
    
    # Safety: If prices are estimated/invalid and result in debit, force min credit
    if net_credit <= 0:
        net_credit = 0.01

    max_risk = (short_call - long_call) - net_credit # Width - Credit
    # Since width is negative (short < long), fix calc:
    width = long_call - short_call
    max_risk = width - net_credit
    
    # Win Probability
    prob_profit = calculate_probability(current_price, short_call, short_put, used_iv, days_to_expiry)
    
    is_estimated = any([sc_est, lc_est, sp_est, lp_est])
    
    return {
        "name": name,
        "short_call": short_call, "long_call": long_call,
        "short_put": short_put, "long_put": long_put,
        "net_credit": net_credit,
        "max_risk": max_risk,
        "roi": (net_credit / max_risk) * 100 if max_risk > 0 else 0,
        "prob_profit": prob_profit,
        "is_estimated": is_estimated
    }

# Generate Strategies
strategies = []
if strategy_profile in ["Show All 3", "Aggressive Only"]:
    strategies.append(generate_condor("🔥 Aggressive", 0.6))
if strategy_profile in ["Show All 3", "Balanced Only"]:
    strategies.append(generate_condor("⚖️ Balanced", 1.0))
if strategy_profile in ["Show All 3", "Conservative Only"]:
    strategies.append(generate_condor("🛡️ Conservative", 1.4))

# --- DISPLAY STRATEGIES ---
for strat in strategies:
    with st.container():
        st.markdown(f'<div class="strategy-card"><div class="strategy-header">{strat["name"]} Iron Condor</div>', unsafe_allow_html=True)
        
        if strat["is_estimated"]:
            st.warning("⚠️ Using Estimated Prices: Live options data unavailable or rate limited. Prices derived via Black-Scholes.")

        # Strikes Visualization
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="leg-card leg-details put-buy"><b>Long Put</b><br>${strat["long_put"]}</div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="leg-card leg-details put-sell"><b>Short Put</b><br>${strat["short_put"]}</div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="leg-card leg-details call-sell"><b>Short Call</b><br>${strat["short_call"]}</div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="leg-card leg-details call-buy"><b>Long Call</b><br>${strat["long_call"]}</div>', unsafe_allow_html=True)
        
        # Metrics
        st.markdown('<div class="metric-row">', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Net Credit (Profit)", f"${strat['net_credit']*100:.0f}")
        m2.metric("Max Risk (Loss)", f"${strat['max_risk']*100:.0f}")
        m3.metric("Return on Risk", f"{strat['roi']:.1f}%")
        m4.metric("Prob. of Profit", f"{strat['prob_profit']:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Profit Diagram
st.markdown("### 📊 Profit/Loss Diagram (Balanced Strategy)")

def plot_payoff(strategy):
    spot_prices = np.linspace(strategy['long_put'] * 0.9, strategy['long_call'] * 1.1, 100)
    
    # Payoff Calculation
    def get_payoff(price, strat):
        # Short Put
        p_sp = np.where(price < strat['short_put'], price - strat['short_put'], 0) + strat['net_credit']/4 # Rough attribution
        # Long Put
        p_lp = np.where(price < strat['long_put'], strat['long_put'] - price, 0) - strat['net_credit']/8
        # Short Call
        p_sc = np.where(price > strat['short_call'], strat['short_call'] - price, 0) + strat['net_credit']/4
        # Long Call
        p_lc = np.where(price > strat['long_call'], price - strat['long_call'], 0) - strat['net_credit']/8
        
        # Total Payoff logic (simplified for visualization)
        payoff = np.zeros_like(price)
        payoff += strat['net_credit'] # Initial Credit
        
        # Losses start beyond short strikes
        payoff += np.where(price < strat['short_put'], price - strat['short_put'], 0)
        payoff += np.where(price > strat['short_call'], strat['short_call'] - price, 0)
        
        # Hedged by long strikes
        payoff += np.where(price < strat['long_put'], strat['long_put'] - price, 0)
        payoff += np.where(price > strat['long_call'], price - strat['long_call'], 0)
        
        return payoff * 100 # Multiplier

    # Use Balanced strategy for chart if available, else first
    strat_to_plot = strategies[1] if len(strategies) > 1 else strategies[0]
    
    payoff_vals = get_payoff(spot_prices, strat_to_plot)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spot_prices, y=payoff_vals, mode='lines', name='P/L at Expiry', 
                             line=dict(color='#00ff88', width=3)))
    
    # Zero Line
    fig.add_shape(type="line", x0=min(spot_prices), y0=0, x1=max(spot_prices), y1=0,
                  line=dict(color="white", dash="dash"))
    
    # Current Price
    fig.add_vline(x=current_price, line_dash="dot", line_color="yellow", annotation_text="Current Price")
    
    fig.update_layout(
        title=f"P/L Diagram: {strat_to_plot['name']}",
        xaxis_title="Stock Price at Expiration",
        yaxis_title="Profit / Loss ($)",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)

if len(strategies) > 0:
    plot_payoff(strategies[0])
