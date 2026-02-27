import os
import yfinance as yf
import pandas as pd
import numpy as np

def fetch_empirical_paths(period="max"):
    """
    Downloads historical paths of the S&P 500 (^SPX) and the VIX (^VIX) 
    to serve as the Spot and Variance paths respectively.
    This bypasses Monte Carlo simulation, addressing the curse of dimensionality 
    by training strictly on empirically realized market regimes.
    """
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(BASE_DIR, "Data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    spx_path = os.path.join(data_dir, "SPX_history.csv")
    vix_path = os.path.join(data_dir, "VIX_history.csv")
    
    # Download if not present
    if not (os.path.exists(spx_path) and os.path.exists(vix_path)):
        print(f"Downloading {period} of historical SPX and VIX data for Challenge Paths...")
        spx = yf.download("^SPX", period=period, interval="1d", progress=False)['Close']
        vix = yf.download("^VIX", period=period, interval="1d", progress=False)['Close']
        
        # Align indices
        df = pd.concat([spx, vix], axis=1).dropna()
        df.columns = ['SPX', 'VIX']
        df['SPX'].to_csv(spx_path)
        df['VIX'].to_csv(vix_path)
        
    spx_df = pd.read_csv(spx_path, index_col=0, parse_dates=True)
    vix_df = pd.read_csv(vix_path, index_col=0, parse_dates=True)
    
    # Format as paths
    # We shape it as (1, length) to act like a single long trajectory that 
    # the Deep BSDE will chunk into training windows.
    S_paths = spx_df.values.T
    
    # VIX is quoted as annualized standard deviation in %.
    # We convert it to raw variance: V = (VIX / 100)^2
    V_paths = (vix_df.values / 100.0).T ** 2 
    
    # Time vector in years (assuming 252 trading days)
    t = np.linspace(0, len(spx_df)/252.0, len(spx_df)) 
    
    print(f"Loaded {len(t)} empirical historical trading days.")
    return t, S_paths, V_paths

if __name__ == "__main__":
    fetch_empirical_paths()
