import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date
import math
import pandas as pd
import numpy as np
import requests

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Iron Condor Strategy Builder", layout="wide", page_icon="🦅")

# --- LOAD SECRETS (MASSIVE / POLYGON) ---
try:
    MASSIVE_API_KEY = st.secrets["MASSIVE_API_KEY"]
    st.sidebar.success(f"✅ API Key Loaded: {MASSIVE_API_KEY[:8]}...")
except:
    st.error("❌ Critical Error: Could not find MASSIVE_API_KEY in secrets.")
    st.stop()

MASSIVE_BASE_URL = "https://api.polygon.io"

# Test API connection
def test_api_connection():
    """Test if API key is valid"""
    url = f"{MASSIVE_BASE_URL}/v2/aggs/ticker/SPY/prev"
    params = {"apiKey": MASSIVE_API_KEY}
    try:
        resp = requests.get(url, params=params, timeout=10)
        st.sidebar.info(f"API Status Code: {resp.status_code}")
        if resp.status_code == 200:
            st.sidebar.success("✅ API Connection Working!")
            return True
        elif resp.status_code == 401:
            st.sidebar.error("❌ Invalid API Key")
            return False
        elif resp.status_code == 403:
            st.sidebar.error("❌ API Key doesn't have required permissions")
            return False
        else:
            st.sidebar.warning(f"⚠️ Unexpected response: {resp.status_code}")
            return False
    except Exception as e:
        st.sidebar.error(f"❌ Connection Error: {str(e)}")
        return False

# Popular stocks by price range for screening
STOCK_UNIVERSE = {
    "under_50": ["F", "SOFI", "NIO", "PLTR", "SNAP", "AAL", "RIVN", "LCID", "PLUG", "VALE"],
    "under_100": ["AMD", "UBER", "SNOW", "ABNB", "DKNG", "RBLX", "COIN", "NET", "HOOD", "ZM"],
    "under_500": ["NVDA", "TSLA", "NFLX", "META", "GOOGL", "AMZN", "AVGO", "SHOP", "AAPL", "MSFT"],
    "premium": ["SPY", "QQQ", "IWM", "DIA"]
}

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
    
    # Create a copy to avoid modifying original
    df_copy = df.copy()
    
    # Calculate daily returns
    df_copy['returns'] = df_copy['close'].pct_change()
    
    # Calculate standard deviation of returns
    volatility = df_copy['returns'].tail(window).std()
    
    # Annualize (252 trading days per year)
    annual_volatility = volatility * np.sqrt(252)
    
    return annual_volatility if not np.isnan(annual_volatility) else 0.25

def estimate_option_price(sym, expiry, strike, option_type):
    """Estimate option price using Black-Scholes when real data unavailable"""
    try:
        current_price = get_current_price_polygon(sym)
        if current_price == 0:
            return 0.0
        
        # Get historical data for IV estimation
        hist = get_stock_data_polygon(sym, 60)
        if len(hist) < 30:
            iv = 0.25  # Default IV
        else:
            hist_copy = hist.copy()
            iv = calculate_historical_volatility(hist_copy, 30)
        
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
        
        return max(0.01, price)  # Minimum price of $0.01
        
    except Exception as e:
        # If all else fails, use simple intrinsic + time value estimate
        current_price = get_current_price_polygon(sym)
        days_to_expiry = (datetime.strptime(expiry, "%Y-%m-%d").date() - date.today()).days
        
        if option_type == "call":
            intrinsic = max(0, current_price - strike)
        else:
            intrinsic = max(0, strike - current_price)
        
        time_value = 0.02 * abs(current_price - strike) * math.sqrt(days_to_expiry / 30)
        return max(0.01, intrinsic + time_value)

# --- MASSIVE / POLYGON API FUNCTIONS ---
def get_stock_data_polygon(sym, lookback_days):
    """Fetch Historical Data from Polygon"""
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    
    url = f"{MASSIVE_BASE_URL}/v2/aggs/ticker/{sym}/range/1/day/{start_date}/{end_date}"
    params = {"apiKey": MASSIVE_API_KEY, "limit": 50000}
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        
        # Debug info
        st.sidebar.info(f"📊 Stock Data Status: {resp.status_code}")
        
        if resp.status_code == 401:
            st.error("❌ Invalid API Key - Check your Polygon/Massive API key")
            return pd.DataFrame()
        elif resp.status_code == 403:
            st.error("❌ Access Denied - Your API plan may not include this data")
            return pd.DataFrame()
        elif resp.status_code == 404:
            st.error(f"❌ Ticker '{sym}' not found")
            return pd.DataFrame()
        elif resp.status_code != 200:
            st.error(f"❌ API Error: Status {resp.status_code}")
            st.code(resp.text)
            return pd.DataFrame()
        
        data = resp.json()
        
        if data.get("status") == "ERROR":
            st.error(f"❌ API Error: {data.get('error', 'Unknown error')}")
            st.code(str(data))
            return pd.DataFrame()
        
        if data.get("results"):
            df = pd.DataFrame(data["results"])
            df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
            # Polygon returns: o, h, l, c, v. Rename to standard.
            df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
            st.sidebar.success(f"✅ Loaded {len(df)} bars")
            return df
        else:
            st.error(f"❌ No data returned for {sym}")
            st.code(str(data))
            return pd.DataFrame()
    except requests.exceptions.Timeout:
        st.error("❌ Request timed out - Try again")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        return pd.DataFrame()

def get_current_price_polygon(sym):
    """Fetch Real-Time Price from Polygon"""
    url = f"{MASSIVE_BASE_URL}/v2/aggs/ticker/{sym}/prev"
    params = {"apiKey": MASSIVE_API_KEY}
    try:
        resp = requests.get(url, params=params).json()
        if resp.get("results"):
            return resp["results"][0]["c"]
        return 0.0
    except:
        return 0.0

def get_closest_expiry_polygon(sym, target_date):
    """Get expirations from Polygon and find closest"""
    url = f"{MASSIVE_BASE_URL}/v3/reference/options/contracts"
    params = {
        "apiKey": MASSIVE_API_KEY,
        "underlying_ticker": sym,
        "expired": "false",
        "limit": 1000
    }
    
    try:
        resp = requests.get(url, params=params).json()
        dates = set()
        if resp.get("results"):
            for c in resp["results"]:
                if "expiration_date" in c:
                    dates.add(c["expiration_date"])
        
        valid_dates = [datetime.strptime(d, "%Y-%m-%d").date() for d in dates]
        valid_dates.sort()
        
        # Find closest date >= target
        future_dates = [d for d in valid_dates if d >= target_date]
        if not future_dates: return None
        
        closest = min(future_dates, key=lambda x: abs(x - target_date))
        return closest.strftime("%Y-%m-%d")
    except:
        return None

def get_option_price_polygon(sym, expiry, strike, option_type):
    """Construct Polygon Symbol and Fetch Snapshot with Better Error Handling"""
    try:
        # Convert date to YYMMDD
        exp_date = datetime.strptime(expiry, "%Y-%m-%d")
        exp_str = exp_date.strftime("%y%m%d")
        
        # Format strike (e.g. 450.5 -> 00450500)
        strike_int = int(strike * 1000)
        strike_str = f"{strike_int:08d}"
        
        # Type "C" or "P"
        type_str = "C" if option_type == "call" else "P"
        
        # Construct Ticker: O:SPY240101C00450000
        ticker = f"O:{sym}{exp_str}{type_str}{strike_str}"
        
        url = f"{MASSIVE_BASE_URL}/v3/snapshot/options/{sym}/{ticker}"
        params = {"apiKey": MASSIVE_API_KEY}
        
        resp = requests.get(url, params=params, timeout=10)
        
        # Check for API errors
        if resp.status_code == 403:
            return estimate_option_price(sym, expiry, strike, option_type), strike
        elif resp.status_code != 200:
            return estimate_option_price(sym, expiry, strike, option_type), strike
        
        data = resp.json()
        
        if data.get("status") == "ERROR":
            return estimate_option_price(sym, expiry, strike, option_type), strike
        
        if data.get("results"):
            res = data["results"]
            
            # Try multiple price sources in order of preference
            price = 0.0
            
            # Try last trade price
            if res.get("last_trade"):
                price = res["last_trade"].get("price", 0.0)
            
            # Try midpoint of bid/ask
            if price == 0.0 and res.get("last_quote"):
                bid = res["last_quote"].get("bid", 0.0)
                ask = res["last_quote"].get("ask", 0.0)
                if bid > 0 and ask > 0:
                    price = (bid + ask) / 2
            
            # Try day close
            if price == 0.0 and res.get("day"):
                price = res["day"].get("close", 0.0)
            
            if price > 0:
                return price, strike
        
        # If no data found, estimate
        return estimate_option_price(sym, expiry, strike, option_type), strike
        
    except requests.exceptions.Timeout:
        return estimate_option_price(sym, expiry, strike, option_type), strike
    except Exception as e:
        return estimate_option_price(sym, expiry, strike, option_type), strike

def get_news_data_polygon(sym):
    url = f"{MASSIVE_BASE_URL}/v2/reference/news"
    params = {"apiKey": MASSIVE_API_KEY, "ticker": sym, "limit": 5}
    try:
        resp = requests.get(url, params=params).json()
        if resp.get("results"):
            return resp["results"]
        return []
    except:
        return []

def plot_payoff(current_price, short_call, long_call, short_put, long_put, net_credit):
    # Use a default wing_width for plotting if not available globally
    plot_wing_width = abs(long_call - short_call)
    
    min_price = long_put - (plot_wing_width * 1.5)
    max_price = long_call + (plot_wing_width * 1.5)
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

def calculate_optimal_wing_widths(current_price, calculated_iv, days_to_expiry):
    expected_move = current_price * calculated_iv * math.sqrt(days_to_expiry/365.0)
    aggressive = max(2, round(expected_move * 0.25))
    balanced = max(3, round(expected_move * 0.45))
    conservative = max(5, round(expected_move * 0.70))
    if aggressive >= balanced: balanced = aggressive + 2
    if balanced >= conservative: conservative = balanced + 3
    if current_price > 500:
        aggressive = round(aggressive / 5) * 5
        balanced = round(balanced / 5) * 5
        conservative = round(conservative / 5) * 5
    return aggressive, balanced, conservative

def generate_scenario_card(symbol, current_price, days_offset, scenario_name, calculated_iv):
    
    target_date = get_next_trading_day(days_offset)
    expiry_str = get_closest_expiry_polygon(symbol, target_date)
    
    if not expiry_str:
        st.error("❌ No options data found in Polygon/Massive.")
        return

    days_to_expiry = (datetime.strptime(expiry_str, "%Y-%m-%d").date() - date.today()).days
    if days_to_expiry < 1: days_to_expiry = 1
    
    aggressive_width, balanced_width, conservative_width = calculate_optimal_wing_widths(
        current_price, calculated_iv, days_to_expiry
    )
    
    st.markdown(f"<div class='strategy-header'>🎯 {scenario_name}</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='alert-box alert-info'>
        📅 Expiration: {expiry_str} ({days_to_expiry} days away) | 
        📊 Using IV: {calculated_iv*100:.1f}%
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class='strategy-visual'>
        <h4 style='color: #00d4ff; margin-bottom: 15px;'>📏 Calculated Optimal Wing Widths</h4>
        <div style='display: flex; gap: 15px; justify-content: center; flex-wrap: wrap;'>
            <div style='background: rgba(255, 71, 87, 0.2); padding: 15px; border-radius: 8px; border: 2px solid #ff4757;'>
                <strong style='color: #ff4757;'>🔥 AGGRESSIVE</strong><br>
                <span style='font-size: 1.5em; color: white;'>${aggressive_width}</span><br>
                <span style='font-size: 0.85em; color: #a0a0a0;'>Max Profit Focus</span>
            </div>
            <div style='background: rgba(0, 212, 255, 0.2); padding: 15px; border-radius: 8px; border: 2px solid #00d4ff;'>
                <strong style='color: #00d4ff;'>⚖️ BALANCED</strong><br>
                <span style='font-size: 1.5em; color: white;'>${balanced_width}</span><br>
                <span style='font-size: 0.85em; color: #a0a0a0;'>Optimal Risk/Reward</span>
            </div>
            <div style='background: rgba(46, 213, 115, 0.2); padding: 15px; border-radius: 8px; border: 2px solid #2ed573;'>
                <strong style='color: #2ed573;'>🛡️ CONSERVATIVE</strong><br>
                <span style='font-size: 1.5em; color: white;'>${conservative_width}</span><br>
                <span style='font-size: 0.85em; color: #a0a0a0;'>Safety First</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    profiles_to_show = []
    if strategy_profile in ["Show All 3", "Aggressive Only"]:
        profiles_to_show.append(("🔥 Aggressive", aggressive_width, "aggressive"))
    if strategy_profile in ["Show All 3", "Balanced Only"]:
        profiles_to_show.append(("⚖️ Balanced", balanced_width, "balanced"))
    if strategy_profile in ["Show All 3", "Conservative Only"]:
        profiles_to_show.append(("🛡️ Conservative", conservative_width, "conservative"))
    
    for profile_name, wing_width, profile_key in profiles_to_show:
        generate_single_strategy(symbol, current_price, days_offset, expiry_str, days_to_expiry, 
                                calculated_iv, wing_width, profile_name, profile_key)

def generate_single_strategy(symbol, current_price, days_offset, expiry_str, days_to_expiry, 
                            calculated_iv, wing_width, profile_name, profile_key):
    
    actual_iv = calculated_iv
    
    if days_offset <= 3:
        stdev_mult = strike_stdev * 0.7
    elif days_offset <= 7:
        stdev_mult = strike_stdev * 0.85
    elif days_offset <= 15:
        stdev_mult = strike_stdev * 1.0
    else:
        stdev_mult = strike_stdev * 1.15
    
    move = current_price * actual_iv * math.sqrt(days_to_expiry/365.0) * stdev_mult
    strike_interval = 1.0 if current_price > 100 else 0.50
    
    s_call = round_to_strike(current_price + move, strike_interval)
    l_call = s_call + wing_width
    s_put = round_to_strike(current_price - move, strike_interval)
    l_put = s_put - wing_width
    
    st.markdown(f"""
    <div class='strategy-card' style='border-left: 5px solid {"#ff4757" if "Aggressive" in profile_name else "#00d4ff" if "Balanced" in profile_name else "#2ed573"}'>
        <h3 style='margin-top: 0; color: {"#ff4757" if "Aggressive" in profile_name else "#00d4ff" if "Balanced" in profile_name else "#2ed573"}'>{profile_name} Strategy (${wing_width} Wings)</h3>
    """, unsafe_allow_html=True)
    
    legs = [
        (s_call, "call", "SELL", "Short Call"),
        (l_call, "call", "BUY", "Long Call"),
        (s_put, "put", "SELL", "Short Put"),
        (l_put, "put", "BUY", "Long Put")
    ]
    
    total_credit = 0.0
    prices = []
    using_estimates = False
    
    # Show a progress indicator
    progress_text = st.empty()
    progress_text.markdown("🔄 Fetching options prices...")
    
    for strike, otype, side, _ in legs:
        price, actual_strike = get_option_price_polygon(symbol, expiry_str, strike, otype)
        prices.append(price)
        if side == "SELL": 
            total_credit += price
        else: 
            total_credit -= price
        
        # Check if we're using estimates
        if price < 0.10 or price > current_price:
            using_estimates = True
    
    progress_text.empty()
    
    # Show warning if using estimates
    if using_estimates:
        st.markdown("""
        <div class='alert-box alert-warning'>
            ⚠️ <strong>Using Estimated Prices:</strong> Live options data may not be available. 
            Prices are calculated using Black-Scholes model. For accurate pricing, verify with your broker.
        </div>
        """, unsafe_allow_html=True)
    
    if total_credit <= 0:
        st.warning(f"⚠️ {profile_name}: This combination results in a net debit or insufficient data. Try different parameters or check during market hours.")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    
    max_profit = total_credit * 100
    max_loss = (wing_width * 100) - max_profit
    
    if max_loss <= 0: max_loss = wing_width * 100 * 0.5
    
    pop = calculate_probability(current_price, s_call, s_put, actual_iv, days_to_expiry)
    risk_reward = max_profit / max_loss if max_loss > 0 else 0
    return_on_risk = (max_profit / max_loss * 100) if max_loss > 0 else 0
    
    upper_breakeven = s_call + total_credit
    lower_breakeven = s_put - total_credit
    breakeven_range = upper_breakeven - lower_breakeven
    breakeven_pct = (breakeven_range / current_price) * 100
    
    st.markdown(create_strategy_visual(current_price, s_call, l_call, s_put, l_put), unsafe_allow_html=True)
    
    st.markdown("<div class='metric-row'>", unsafe_allow_html=True)
    cols = st.columns(6)
    
    with cols[0]:
        st.markdown(f"<div class='metric-box'><div class='metric-label'>💵 Max Profit</div><div class='metric-value'>${max_profit:.0f}</div></div>", unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"<div class='metric-box'><div class='metric-label'>⚠️ Max Loss</div><div class='metric-value negative'>${max_loss:.0f}</div></div>", unsafe_allow_html=True)
    with cols[2]:
        st.markdown(f"<div class='metric-box'><div class='metric-label'>⚖️ Risk/Reward</div><div class='metric-value'>1:{risk_reward:.2f}</div></div>", unsafe_allow_html=True)
    with cols[3]:
        st.markdown(f"<div class='metric-box'><div class='metric-label'>📈 Return on Risk</div><div class='metric-value'>{return_on_risk:.1f}%</div></div>", unsafe_allow_html=True)
    with cols[4]:
        st.markdown(f"<div class='metric-box'><div class='metric-label'>🎲 Win Probability</div><div class='metric-value'>{pop:.1f}%</div></div>", unsafe_allow_html=True)
    with cols[5]:
        st.markdown(f"<div class='metric-box'><div class='metric-label'>🎯 Profit Range</div><div class='metric-value'>{breakeven_pct:.1f}%</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class='alert-box alert-success'>
        <strong>✅ Profit Zone:</strong> ${lower_breakeven:.2f} to ${upper_breakeven:.2f} 
        (±{breakeven_pct/2:.1f}% from current price)<br>
        <strong>📊 Net Credit Collected:</strong> ${total_credit:.2f} per spread
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📋 View Detailed Trade Legs", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class='leg-card call-sell'><div class='leg-header'>🔴 SELL Call (Bear Call Spread)</div><div class='leg-details'><strong>Strike:</strong> ${s_call:.2f}<br><strong>Price:</strong> ${prices[0]:.2f}</div></div>
            <div class='leg-card call-buy'><div class='leg-header'>🟠 BUY Call (Protection)</div><div class='leg-details'><strong>Strike:</strong> ${l_call:.2f}<br><strong>Price:</strong> ${prices[1]:.2f}</div></div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class='leg-card put-sell'><div class='leg-header'>🟢 SELL Put (Bull Put Spread)</div><div class='leg-details'><strong>Strike:</strong> ${s_put:.2f}<br><strong>Price:</strong> ${prices[2]:.2f}</div></div>
            <div class='leg-card put-buy'><div class='leg-header'>🔵 BUY Put (Protection)</div><div class='leg-details'><strong>Strike:</strong> ${l_put:.2f}<br><strong>Price:</strong> ${prices[3]:.2f}</div></div>
            """, unsafe_allow_html=True)
    
    with st.expander("📈 View Profit/Loss Diagram", expanded=False):
        st.plotly_chart(
            plot_payoff(current_price, s_call, l_call, s_put, l_put, total_credit), 
            use_container_width=True, 
            key=f"chart_{profile_key}_{days_offset}"
        )
    
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

def analyze_stock_for_iron_condor(symbol, price_range_category):
    try:
        # Use Polygon for Analysis
        current_price = get_current_price_polygon(symbol)
        if current_price == 0: return None
        
        hist = get_stock_data_polygon(symbol, 60)
        if len(hist) < 30: return None
        
        hist_copy = hist.copy()
        hist_copy['returns'] = hist_copy['close'].pct_change()
        volatility = hist_copy['returns'].tail(30).std() * np.sqrt(252)
        
        if np.isnan(volatility) or volatility == 0:
            volatility = 0.25
        
        news = get_news_data_polygon(symbol)
        news_count = len(news)
        
        price_change_30d = ((current_price - hist_copy['close'].iloc[-30]) / hist_copy['close'].iloc[-30]) * 100
        avg_volume = hist_copy['volume'].mean()
        
        score = 0
        if 0.20 <= volatility <= 0.50: score += 30
        elif 0.15 <= volatility < 0.20 or 0.50 < volatility <= 0.60: score += 20
        else: score += 10
        
        if abs(price_change_30d) < 5: score += 25
        elif abs(price_change_30d) < 10: score += 15
        else: score += 5
        
        if avg_volume > 5000000: score += 20
        elif avg_volume > 1000000: score += 15
        else: score += 5
        
        if news_count >= 2: score += 15
        elif news_count >= 1: score += 10
        else: score += 5
        
        recommendation = "Strong Buy" if score >= 70 else "Consider" if score >= 50 else "Monitor"
        badge_class = "badge-buy" if score >= 70 else "badge-watch" if score >= 50 else "badge-caution"
        
        return {
            'symbol': symbol, 'price': current_price, 'iv': volatility,
            'price_change_30d': price_change_30d, 'volume': avg_volume,
            'news_count': news_count, 'score': score,
            'recommendation': recommendation, 'badge_class': badge_class
        }
    except:
        return None

def scan_stocks_by_price_range(price_range_key, top_n=3):
    stocks = STOCK_UNIVERSE.get(price_range_key, [])
    results = []
    with st.spinner(f"Analyzing {len(stocks)} stocks..."):
        for symbol in stocks:
            analysis = analyze_stock_for_iron_condor(symbol, price_range_key)
            if analysis: results.append(analysis)
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_n]

def display_stock_suggestion(analysis):
    if analysis['iv'] > 0.40:
        heat_class = "suggestion-hot"
        heat_emoji = "🔥"
        heat_text = "HIGH VOLATILITY"
    elif analysis['iv'] > 0.25:
        heat_class = "suggestion-moderate"
        heat_emoji = "📊"
        heat_text = "MODERATE VOLATILITY"
    else:
        heat_class = "suggestion-cool"
        heat_emoji = "❄️"
        heat_text = "LOW VOLATILITY"
    
    st.markdown(f"""
    <div class='suggestion-card {heat_class}'>
        <div class='stock-header'>
            {heat_emoji} ${analysis['symbol']} - ${analysis['price']:.2f}
            <span class='recommendation-badge {analysis['badge_class']}'>{analysis['recommendation']}</span>
        </div>
        <div style='color: #ffd700; font-size: 1.1em; margin: 10px 0;'>
            {heat_text} | IV: {analysis['iv']*100:.1f}%
        </div>
        <div class='stock-metrics'>
            <div class='stock-metric-item'>
                <div style='color: #a0a0a0; font-size: 0.85em;'>30-Day Move</div>
                <div style='color: {"#2ed573" if abs(analysis["price_change_30d"]) < 5 else "#ffa502" if abs(analysis["price_change_30d"]) < 10 else "#ff4757"}; font-size: 1.2em; font-weight: bold;'>
                    {analysis['price_change_30d']:+.2f}%
                </div>
            </div>
            <div class='stock-metric-item'>
                <div style='color: #a0a0a0; font-size: 0.85em;'>Avg Volume</div>
                <div style='color: white; font-size: 1.2em; font-weight: bold;'>
                    {analysis['volume']/1000000:.1f}M
                </div>
            </div>
            <div class='stock-metric-item'>
                <div style='color: #a0a0a0; font-size: 0.85em;'>Score</div>
                <div style='color: {"#2ed573" if analysis["score"] >= 70 else "#ffa502" if analysis["score"] >= 50 else "#ff4757"}; font-size: 1.2em; font-weight: bold;'>
                    {analysis['score']}/100
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def get_market_overview_polygon():
    try:
        spy_price = get_current_price_polygon("SPY")
        if spy_price:
            return {'spy_price': spy_price, 'spy_change': 0.0, 'vix_level': 0.0} # Simplified for speed
        return None
    except:
        return None

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
                <li><strong>Max Profit:</strong> Net credit received</li>
                <li><strong>Max Loss:</strong> Wing width - Net credit</li>
                <li><strong>Breakeven:</strong> Short strikes ± net credit</li>
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
                <li>Want <strong>defined risk</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if symbol:
    try:
        # Use Polygon Data
        df = get_stock_data_polygon(symbol, days_back)
        
        if df.empty:
            st.error("Unable to fetch stock data from Polygon. Please check your API key and ticker.")
            st.stop()
        
        current_price = df.iloc[-1]['close']
        
        # Calculate historical volatility automatically - use copies to avoid modification issues
        df_copy_30 = df.copy()
        df_copy_60 = df.copy()
        historical_iv_30d = calculate_historical_volatility(df_copy_30, window=30)
        historical_iv_60d = calculate_historical_volatility(df_copy_60, window=60)
        
        if iv_mode == "Manual Override":
            active_iv = target_iv
            iv_source = "Manual"
        else:
            active_iv = historical_iv_30d
            iv_source = "30-Day Historical"
        
        # Technical indicators - create fresh copy for each calculation
        df_tech = df.copy()
        df_tech['SMA_50'] = df_tech['close'].rolling(50).mean()
        df_tech['Upper_Band'] = df_tech['close'].rolling(20).mean() + (df_tech['close'].rolling(20).std() * 2)
        df_tech['Lower_Band'] = df_tech['close'].rolling(20).mean() - (df_tech['close'].rolling(20).std() * 2)
        
        # RSI calculation
        delta = df_tech['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_tech['RSI'] = 100 - (100 / (1 + rs))

        # Header Metrics
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"""
            <div class='alert-box alert-info'>
                <strong>📊 {symbol} Current Price:</strong> ${current_price:.2f}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-box'>
                <div class='metric-label'>📉 Active IV ({iv_source})</div>
                <div class='metric-value'>{active_iv*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-box'>
                <div class='metric-label'>📊 60-Day IV</div>
                <div class='metric-value'>{historical_iv_60d*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        # Main tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Strategy Builder", "🔍 Stock Screener", "📈 Technical Chart", "📰 Market News"])

        with tab1:
            st.markdown("### Generate Iron Condor Strategies")
            
            if st.button("🚀 GENERATE ALL TIMEFRAMES", use_container_width=True):
                t1, t2, t3, t4 = st.tabs(["⚡ 3-Day", "📊 7-Day", "📅 15-Day", "📆 30-Day"])
                
                with t1:
                    with st.spinner("Fetching market data..."):
                        generate_scenario_card(symbol, current_price, 3, "3-Day Expiry", active_iv)
                
                with t2:
                    with st.spinner("Fetching market data..."):
                        generate_scenario_card(symbol, current_price, 7, "7-Day Expiry", active_iv)
                
                with t3:
                    with st.spinner("Fetching market data..."):
                        generate_scenario_card(symbol, current_price, 15, "15-Day Expiry", active_iv)
                
                with t4:
                    with st.spinner("Fetching market data..."):
                        generate_scenario_card(symbol, current_price, 30, "30-Day Expiry", active_iv)
            else:
                st.info("👆 Click the button above to generate iron condor strategies")

        with tab2:
            st.markdown("### 🔍 AI-Powered Stock Screener")
            market_data = get_market_overview_polygon()
            if market_data:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"<div class='metric-box'><div class='metric-label'>📊 SPY Price</div><div class='metric-value'>${market_data['spy_price']:.2f}</div></div>", unsafe_allow_html=True)
            
            st.markdown("---")
            
            if st.button("🔎 SCAN FOR OPPORTUNITIES", use_container_width=True, type="primary"):
                price_tabs = st.tabs(["💰 Under $50", "💵 $50-$100", "💎 $100-$500", "🏆 Premium ETFs"])
                
                with price_tabs[0]:
                    suggestions = scan_stocks_by_price_range("under_50", top_n=5)
                    if suggestions:
                        for analysis in suggestions: display_stock_suggestion(analysis)
                    else: st.warning("No suitable candidates found.")
                
                with price_tabs[1]:
                    suggestions = scan_stocks_by_price_range("under_100", top_n=5)
                    if suggestions:
                        for analysis in suggestions: display_stock_suggestion(analysis)
                    else: st.warning("No suitable candidates found.")
                
                with price_tabs[2]:
                    suggestions = scan_stocks_by_price_range("under_500", top_n=5)
                    if suggestions:
                        for analysis in suggestions: display_stock_suggestion(analysis)
                    else: st.warning("No suitable candidates found.")
                
                with price_tabs[3]:
                    suggestions = scan_stocks_by_price_range("premium", top_n=4)
                    if suggestions:
                        for analysis in suggestions: display_stock_suggestion(analysis)
                    else: st.warning("No suitable candidates found.")

        with tab3:
            st.markdown("### Technical Analysis")
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=df_tech['timestamp'], open=df_tech['open'], high=df_tech['high'], low=df_tech['low'], close=df_tech['close'], name='Price'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_tech['timestamp'], y=df_tech['Upper_Band'], line=dict(color='rgba(255, 152, 0, 0.5)', width=1, dash='dot'), name='Upper Band'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_tech['timestamp'], y=df_tech['Lower_Band'], line=dict(color='rgba(255, 152, 0, 0.5)', width=1, dash='dot'), name='Lower Band', fill='tonexty', fillcolor='rgba(255, 152, 0, 0.1)'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_tech['timestamp'], y=df_tech['RSI'], line=dict(color='#A020F0', width=2), name='RSI'), row=2, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
            fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

        with tab4:
            st.markdown("### 📰 Latest Market News")
            news = get_news_data_polygon(symbol)
            if news:
                for article in news:
                    try:
                        title = article.get('title', 'No title')
                        link = article.get('article_url', '#')
                        publisher = article.get('publisher', {}).get('name', 'Unknown')
                        pub_date = article.get('published_utc', 'Unknown date')
                        
                        st.markdown(f"""
                        <div class='info-card'>
                            <h3>📄 {title}</h3>
                            <p style='color: #a0a0a0; font-size: 0.9em;'><strong>Source:</strong> {publisher} | <strong>Date:</strong> {pub_date}</p>
                            <a href='{link}' target='_blank' style='color: #00d4ff;'>Read Full Article →</a>
                        </div>
                        """, unsafe_allow_html=True)
                    except: continue
            else:
                st.markdown("<div class='alert-box alert-warning'>No recent news found.</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
