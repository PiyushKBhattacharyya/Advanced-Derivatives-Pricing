import numpy as np
import pandas as pd
from scipy.stats import norm
import os

def black_scholes_call(S, K, T, r, sigma):
    """
    Standard Black-Scholes formula for European Call Option.
    Serves as the baseline benchmark against the Deep BSDE.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return price

def get_empirical_dataset():
    """
    Loads the empirical SPX Options Chain dataset downloaded by data_loader.py.
    Provides empirical Spot, Variance, Maturity, and Strike for DL training.
    """
    data_dir = "Data"
    files = [f for f in os.listdir(data_dir) if f.startswith("SPX_options_chain")]
    
    if not files:
        raise FileNotFoundError("No SPX options chain found. Run src/data_loader.py first.")
        
    # Get most recently downloaded chain
    latest_file = sorted(files)[-1]
    df = pd.read_csv(os.path.join(data_dir, latest_file))
    
    # Extract features for Neural Network Training / Benchmarking
    # Using 'impliedVolatility' squared as initial variance proxy xi0
    spots = df['Spot'].values
    strikes = df['strike'].values
    maturities = df['Maturity'].values
    initial_variances = (df['impliedVolatility'].values)**2 
    market_prices = df['lastPrice'].values
    
    return spots, strikes, maturities, initial_variances, market_prices

def extract_challenge_regime(regime='extreme_vol'):
    """
    Slices the historical SPX/VIX dataset (from market_paths.py) to extract 
    specific, empirically challenging market regimes for benchmarking constraint testing.
    """
    vix_path = os.path.join("Data", "VIX_history.csv")
    spx_path = os.path.join("Data", "SPX_history.csv")
    
    if not os.path.exists(vix_path) or not os.path.exists(spx_path):
         raise FileNotFoundError("Historical prices not found. Run src/market_paths.py first.")
         
    vix_df = pd.read_csv(vix_path, index_col=0, parse_dates=True)
    spx_df = pd.read_csv(spx_path, index_col=0, parse_dates=True)
    
    df = pd.concat([spx_df, vix_df], axis=1).dropna()
    df.columns = ['SPX', 'VIX']
    
    # Extract historically extreme volatility crises for robustness testing
    if regime == 'extreme_vol':
        # COVID-19 Crash Window (Feb 2020 - May 2020)
        crisis_df = df.loc['2020-02-15':'2020-05-30']
    elif regime == 'short_maturity':
        # Flash Crash Window or high velocity week (e.g. Aug 2024 VIX spike)
        crisis_df = df.loc['2024-08-01':'2024-08-15']
    else:
        raise ValueError("Unknown empirical regime requested.")
        
    print(f"Extracted {len(crisis_df)} empirical trading days for the '{regime}' challenge.")
    
    # Returns the chunked paths exactly as they happened
    t = np.linspace(0, len(crisis_df)/252.0, len(crisis_df))
    S_paths = crisis_df['SPX'].values.T
    V_paths = (crisis_df['VIX'].values / 100.0).T ** 2 
    
    return t, S_paths, V_paths
