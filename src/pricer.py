import pandas as pd
import numpy as np
import os

def get_empirical_option_price(strike, maturity_days, option_type="call"):
    """
    Retrieves the actual, empirical market price for an option from the 
    live options chain data we downloaded, strictly avoiding any theoretical 
    pricing computation or Monte Carlo methodologies.
    
    Args:
        strike (float): The target strike price.
        maturity_days (float): Approximate maturity in days.
        option_type (str): "call" or "put"
        
    Returns:
        float: The actual last traded market price from the dataset.
    """
    data_dir = "Data"
    files = [f for f in os.listdir(data_dir) if f.startswith("SPX_options_chain")]
    
    if not files:
        raise FileNotFoundError("Empirical options data not found. Run src/data_loader.py")
        
    # Get most recent
    latest_file = sorted(files)[-1]
    df = pd.read_csv(os.path.join(data_dir, latest_file))
    
    # We only have calls downloaded by default in data_loader, 
    # but theoretically we could filter by option_type if we expand it.
    
    # Find the closest match in the empirical dataset
    # Convert maturity from Years to Days
    df['Maturity_Days'] = df['Maturity'] * 365.25
    
    # Calculate Euclidean distance to find the most similar historical option
    # (Since empirical strikes and maturities are discrete)
    df['distance'] = np.sqrt(
        ((df['strike'] - strike) / strike)**2 + 
        ((df['Maturity_Days'] - maturity_days) / maturity_days)**2
    )
    
    closest_option = df.loc[df['distance'].idxmin()]
    
    return closest_option['lastPrice']

def get_empirical_implied_volatility(strike, maturity_days):
    """
    Retrieves the actual, empirical implied volatility for an option from the 
    live options chain data.
    """
    data_dir = "Data"
    files = [f for f in os.listdir(data_dir) if f.startswith("SPX_options_chain")]
    
    if not files:
        raise FileNotFoundError("Empirical options data not found. Run src/data_loader.py")
        
    latest_file = sorted(files)[-1]
    df = pd.read_csv(os.path.join(data_dir, latest_file))
    
    df['Maturity_Days'] = df['Maturity'] * 365.25
    df['distance'] = np.sqrt(
        ((df['strike'] - strike) / strike)**2 + 
        ((df['Maturity_Days'] - maturity_days) / maturity_days)**2
    )
    
    closest_option = df.loc[df['distance'].idxmin()]
    
    return closest_option['impliedVolatility']
