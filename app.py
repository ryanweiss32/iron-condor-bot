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
            st.error("⚠️ Options data requires a Polygon subscription with options access. Using estimated prices.")
            return estimate_option_price(sym, expiry, strike, option_type), strike
        elif resp.status_code != 200:
            st.warning(f"⚠️ API returned status {resp.status_code}. Using estimated prices.")
            return estimate_option_price(sym, expiry, strike, option_type), strike
        
        data = resp.json()
        
        if data.get("status") == "ERROR":
            st.warning(f"⚠️ Polygon API Error: {data.get('error', 'Unknown')}. Using estimated prices.")
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
        st.warning("⚠️ API request timed out. Using estimated prices.")
        return estimate_option_price(sym, expiry, strike, option_type), strike
    except Exception as e:
        st.warning(f"⚠️ Error fetching option price: {str(e)}. Using estimated prices.")
        return estimate_option_price(sym, expiry, strike, option_type), strike
