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
from alpaca.data.enums import DataFeed  # <--- NEW IMPORT
from datetime import datetime, timedelta, date
import math
import pandas as pd

# --- 1. APP CONFIGURATION ---
st.set_page_config(page_title="Iron Condor Master", layout="wide", page_icon="🦅")

st.markdown("""
<style>
    .metric-container { background-color: #1e1e1e; border: 1px solid #333; border-radius: 10px; padding: 15px; text-align: center; }
    .trade-box { background-color: #262730; padding: 20px; border-radius: 10px; border-left: 5px solid #2196F3; }
    .live-data { color: #00E676; font-weight: bold; font-family: monospace; }
    .scenario-table { margin-bottom: 20px; }
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
    st.subheader("⚙️ General Settings")
    wing_width = st.slider("Wing Width ($)", 1, 10, 5)
    target_iv = st.number_input("Implied Volatility (Est.)", 0.1, 1.0, 0.15, step=0.01)

# --- 4. HELPER FUNCTIONS ---
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
        feed=DataFeed.IEX  # <--- THIS FIXES THE "SIP" ERROR
    )
    return stock_client.get_stock_bars(req).df.reset_index()

def get_news_data(sym):
    news_client = NewsClient(API_KEY, SECRET_KEY)
    news_list = []
    try:
        news_req = NewsRequest(symbols=sym, limit=5)
        news_list = news_client.get_news(news_req).news
        if not news_list:
            news_req = NewsRequest(limit=5)
            news_list = news_client.get_news(news_req).news
    except:
        pass
    return news_list

def find_contract(client, underlying, strike, type, expiry):
    try:
        req = GetOptionContractsRequest(
            underlying_symbol=[underlying],
            status=AssetStatus.ACTIVE,
            expiration_date=expiry,
            type=type,
            strike_price_gte=strike - 0.9, 
            strike_price_lte=strike + 0.9,
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
        
        scenarios = {
            "🚀 Aggressive (High Risk)": 0.6,
            "⚖️ Balanced (Standard)": 1.0,
            "🛡️ Conservative (Safe)": 1.4
        }
        
        scenario_data = []
        for name, multiplier in scenarios.items():
            adj_move = base_move * multiplier
            s_call = math.ceil(current_price + adj_move)
            s_put = math.floor(current_price - adj_move)
            prob = calculate_probability(current_price, s_call, s_put, target_iv, days_to_expiry)
            scenario_data.append({
                "Strategy": name,
                "Win Probability": f"{prob:.1f}%",
                "Short Call": f"${s_call}",
                "Short Put": f"${s_put}",
                "Multiplier": multiplier
            })

        # --- TABS ---
        tab1, tab2, tab3 = st.tabs(["📊 Trade Simulator", "📉 Charts", "📰 News"])

        with tab1:
            st.subheader("1. Choose Your Risk Profile")
            st.dataframe(pd.DataFrame(scenario_data).set_index("Strategy"), use_container_width=True)
            
            selected_strat_name = st.selectbox("Select Strategy to Simulate:", list(scenarios.keys()), index=1)
            selected_multiplier = scenarios[selected_strat_name]
            
            final_move = base_move * selected_multiplier
            short_call_strike = math.ceil(current_price + final_move)
            long_call_strike = short_call_strike + wing_width
            short_put_strike = math.floor(current_price - final_move)
            long_put_strike = short_put_strike - wing_width
            
            st.divider()
            st.subheader(f"2. Live Market Data ({selected_strat_name})")
            
            if st.button("🔴 Fetch Live Bid/Ask & Calculate Profit"):
                trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
                
                # Dynamic Expiry Calculation
                today = date.today()
                days_ahead = 4 - today.weekday()
                if days_ahead <= 0: days_ahead += 7
                expiry = today + timedelta(days=days_ahead)
                
                with st.spinner(f"Fetching option chain for {expiry}..."):
                    sc = find_contract(trading_client, symbol, short_call_strike, ContractType.CALL, expiry)
                    lc = find_contract(trading_client, symbol, long_call_strike, ContractType.CALL, expiry)
                    sp = find_contract(trading_client, symbol, short_put_strike, ContractType.PUT, expiry)
                    lp = find_contract(trading_client, symbol, long_put_strike, ContractType.PUT, expiry)
                    
                    legs = [sc, lc, sp, lp]
                    if all(legs):
                        symbols = [l.symbol for l in legs]
                        quotes = get_live_quotes(symbols)
                        
                        table_rows = []
                        net_credit = 0.0
                        
                        for contract in legs:
                            # Handle missing quotes gracefully
                            if contract.symbol in quotes:
                                q = quotes[contract.symbol].latest_quote
                                mid = (q.ask_price + q.bid_price) / 2
                                bid_txt = f"${q.bid_price:.2f}"
                                ask_txt = f"${q.ask_price:.2f}"
                            else:
                                mid = 0.0
                                bid_txt = "-"
                                ask_txt = "-"
                            
                            if contract.strike_price in [short_call_strike, short_put_strike]:
                                side = "SELL"
                                credit = mid
                            else:
                                side = "BUY"
                                credit = -mid
                            
                            net_credit += credit
                            
                            table_rows.append({
                                "Side": side,
                                "Strike": f"${contract.strike_price}",
                                "Type": contract.type.value,
                                "Bid": bid_txt,
                                "Ask": ask_txt,
                                "Mid": f"${mid:.2f}"
                            })
                            
                        st.table(pd.DataFrame(table_rows))
                        
                        max_risk = (wing_width * 100) - (net_credit * 100)
                        max_profit = net_credit * 100
                        roi = (max_profit / max_risk) * 100 if max_risk > 0 else 0
                        
                        st.markdown(f"""
                        <div class='trade-box'>
                        <h3>💰 Projected Profitability</h3>
                        <p><b>Net Credit (Profit):</b> <span class='live-data'>${max_profit:.2f}</span></p>
                        <p><b>Max Risk (Loss):</b> ${max_risk:.2f}</p>
                        <p><b>Return on Risk:</b> {roi:.1f}%</p>
                        <p><b>Win Probability:</b> {calculate_probability(current_price, short_call_strike, short_put_strike, target_iv, 7):.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("Could not find all contracts. The market might be closed or strikes are unavailable.")
            else:
                st.info("Click the button above to pull live pricing from the exchange.")

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
