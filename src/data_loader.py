import yfinance as yf
import pandas as pd
from datetime import datetime
import os

def download_spx_options(data_dir="Data"):
    """
    Extracts the live options chain for the S&P 500 (^SPX) index directly from Yahoo Finance API.
    This serves as the foundational empirical options dataset for training and benchmarking 
    the Deep BSDE network, removing any reliance on theoretical models.
    
    Args:
        data_dir (str): Relative directory pathway to store the extracted CSV artifacts.
        
    Returns:
        str: Absolute or relative filepath of the generated CSV dataset.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    print("Extracting ^SPX (S&P 500) options chain...")
    spx = yf.Ticker("^SPX")
    
    expirations = spx.options
    
    if not expirations:
        print("Market data unavailable. Please verify API connection.")
        return None
        
    # Extract end-of-day or prevailing spot price to anchor strikes
    spot_price = spx.history(period="1d")['Close'].iloc[-1]
    
    all_calls = []
    
    for exp in expirations:
        try:
            opt = spx.option_chain(exp)
            calls = opt.calls
            
            # Filter solely for liquidly traded strikes to prevent sparse variance metrics
            calls = calls[(calls['openInterest'] > 0) & (calls['impliedVolatility'] > 0)].copy()
            
            if calls.empty:
                continue
                
            expiration_date = datetime.strptime(exp, '%Y-%m-%d')
            today = datetime.today()
            time_to_maturity = (expiration_date - today).days / 365.25 
            
            # Exclude 0-day expiries (0DTE) to maintain stable volatility surfaces
            if time_to_maturity < 0.005: 
                continue
                
            calls['Maturity'] = time_to_maturity
            calls['Spot'] = spot_price
            all_calls.append(calls)
            
        except Exception as e:
             print(f"Extraction failed for maturity {exp}: {e}")
             
    if not all_calls:
        print("No liquid options chains successfully extracted.")
        return None
        
    final_df = pd.concat(all_calls, ignore_index=True)
    
    timestamp = datetime.now().strftime("%Y%m%d")
    filepath = os.path.join(data_dir, f"SPX_options_chain_{timestamp}.csv")
    final_df.to_csv(filepath, index=False)
    
    print(f"Successfully processed {len(final_df)} empirical SPX option quotes to {filepath}")
    return filepath

if __name__ == "__main__":
    download_spx_options("../Data")
