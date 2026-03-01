import os
import yfinance as yf
import pandas as pd
import numpy as np

def fetch_empirical_paths(ticker="AAPL", period="max"):
    """
    Downloads historical paths of the underlying asset and the VIX (as a proxy for market variance)
    to serve as the Spot and Variance paths respectively.
    For American Options research, we also pull dividend data.
    """
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(BASE_DIR, "Data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    ticker_clean = ticker.replace("^", "")
    spot_path = os.path.join(data_dir, f"{ticker_clean}_history.csv")
    vix_path = os.path.join(data_dir, "VIX_history.csv")
    div_path = os.path.join(data_dir, f"{ticker_clean}_dividends.csv")
    
    # Download if not present
    if not (os.path.exists(spot_path) and os.path.exists(vix_path)):
        print(f"Downloading {period} of historical {ticker} and VIX data...")
        asset = yf.Ticker(ticker)
        spot_df = asset.history(period=period)['Close']
        vix_df = yf.download("^VIX", period=period, interval="1d", progress=False)['Close']
        
        # Align indices (Force tz-naive to prevent join errors)
        spot_df.index = spot_df.index.tz_localize(None)
        vix_df.index = vix_df.index.tz_localize(None)
        
        df = pd.concat([spot_df, vix_df], axis=1).dropna()
        df.columns = [ticker_clean, 'VIX']
        df[ticker_clean].to_csv(spot_path)
        df['VIX'].to_csv(vix_path)
        
        # Fetch Dividends
        divs = asset.dividends
        divs.to_csv(div_path)
        
    spot_df = pd.read_csv(spot_path, index_col=0, parse_dates=True)
    vix_df = pd.read_csv(vix_path, index_col=0, parse_dates=True)
    div_df = pd.read_csv(div_path, index_col=0, parse_dates=True)
    
    # Format as paths
    S_paths = spot_df.values.T
    V_paths = (vix_df.values / 100.0).T ** 2 
    
    # Calculate annualized Dividend Yield (approximate)
    # dividend_yield = annual_div / current_price
    try:
        div_df.index = pd.to_datetime(div_df.index, utc=True)
        one_year_ago = pd.Timestamp.now(tz='UTC') - pd.DateOffset(years=1)
        recent_divs = div_df.loc[div_df.index >= one_year_ago].sum()
        recent_divs = recent_divs.iloc[0] if isinstance(recent_divs, pd.Series) else float(recent_divs)
    except:
        recent_divs = 0.0
    
    current_price = S_paths[0, -1]
    q_div = recent_divs / current_price if current_price > 0 else 0.0
    
    t = np.linspace(0, len(spot_df)/252.0, len(spot_df)) 
    
    print(f"Loaded {len(t)} trading days for {ticker}. Div Yield: {q_div:.2%}")
    return t, S_paths, V_paths, q_div

if __name__ == "__main__":
    fetch_empirical_paths("AAPL")

if __name__ == "__main__":
    fetch_empirical_paths()
