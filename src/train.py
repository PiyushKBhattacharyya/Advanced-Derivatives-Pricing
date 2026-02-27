import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.models import DeepBSDE_RoughVol
import os
import pandas as pd
from src.baselines import get_empirical_dataset
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# Global scalers to inverse-transform predictions later
price_scaler = StandardScaler()
spot_scaler = StandardScaler()
strike_scaler = StandardScaler()

def prepare_empirical_batches(seq_len=20, batch_size=32):
    """
    Constructs the PyTorch training batches by extracting historical windows exclusively from 
    our extracted Yahoo Finance arrays. 
    """
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(BASE_DIR, "Data")
    # Find the most recent options chain file to extract its temporal anchor date
    files = [f for f in os.listdir(data_dir) if f.startswith("SPX_options_chain")]
    if not files:
        raise FileNotFoundError("Empirical options data not found. Run src/data_loader.py")
    
    latest_file = sorted(files)[-1]
    
    # Extract the 'YYYYMMDD' date string from 'SPX_options_chain_YYYYMMDD.csv'
    date_str = latest_file.split('_')[-1].split('.')[0]
    try:
        anchor_date = datetime.strptime(date_str, "%Y%m%d")
    except ValueError:
        # Fallback if filename format varies
        anchor_date = datetime.today()
        
    spots, strikes, maturities, initial_variances, market_prices = get_empirical_dataset()
    
    # Load the historical dataset to extract the actual trailing paths
    vix_path = os.path.join(data_dir, "VIX_history.csv")
    spx_path = os.path.join(data_dir, "SPX_history.csv")
    
    if not os.path.exists(vix_path) or not os.path.exists(spx_path):
         raise FileNotFoundError("Historical paths missing. Run src/market_paths.py")
         
    vix_df = pd.read_csv(vix_path, index_col=0, parse_dates=True)
    spx_df = pd.read_csv(spx_path, index_col=0, parse_dates=True)
    
    # Perform strict longitudinal matching by Trading Day date.
    # We locate the exact index in the historical table corresponding to our anchor_date.
    historical_spx_slice = spx_df.loc[:anchor_date]
    historical_vix_slice = vix_df.loc[:anchor_date]
    
    if len(historical_spx_slice) < seq_len:
         raise ValueError(f"Not enough historical data before {anchor_date} to construct a {seq_len}-day path.")
         
    trailing_spx = historical_spx_slice.iloc[-seq_len:].values.flatten()
    trailing_vix = (historical_vix_slice.iloc[-seq_len:].values.flatten() / 100.0) ** 2
    
    # NORMALIZATION PHASE: Scale massive financial SPX values to Mean=0, Var=1 for PyTorch stability
    trailing_spx_scaled = spot_scaler.fit_transform(trailing_spx.reshape(-1, 1)).flatten()
    strikes_scaled = strike_scaler.fit_transform(strikes.reshape(-1, 1)).flatten()
    market_prices_scaled = price_scaler.fit_transform(market_prices.reshape(-1, 1)).flatten()
    
    num_samples = len(spots)
    
    # We broadcast this exact realized `seq_len` trajectory across all options in the cross-section
    historical_S = np.tile(trailing_spx_scaled, (num_samples, 1))
    historical_V = np.tile(trailing_vix, (num_samples, 1)) # Variance is already small (e.g. 0.04)
    
    # Stack into LSTM input shape (Batch, Seq_Len, 2)
    paths_tensor = torch.tensor(np.stack([historical_S, historical_V], axis=-1), dtype=torch.float32)
    
    # Contract Details (Batch, 2) -> [Maturity, Strike]
    contract_tensor = torch.tensor(np.stack([maturities, strikes_scaled], axis=-1), dtype=torch.float32)
    
    # Ground Truth Market Price (Batch, 1) to match Neural Net output dimension strictly
    target_tensor = torch.tensor(market_prices_scaled, dtype=torch.float32).unsqueeze(-1)
    
    return paths_tensor, contract_tensor, target_tensor, price_scaler, spot_scaler, strike_scaler

def bsde_empirical_loss(predicted_prices, market_prices, predicted_greeks):
    """
    Calculates the Deep BSDE Loss.
    We strictly penalize deviation from the actual empirical traded option prices.
    A secondary smoothness regularization is applied to the Delta strategy hedging.
    """
    mse_loss = nn.MSELoss()(predicted_prices, market_prices)
    
    # We apply a small L2 regularization to the Greeks (Delta/Vega) to ensure the 
    # Neural Network finds a smooth, realistic hedging strategy rather than massive
    # discontinuous jumps (which would bankrupt a real quantitative trading desk).
    delta_vega_smoothness_penalty = 1e-4 * torch.mean(predicted_greeks**2)
    
    return mse_loss + delta_vega_smoothness_penalty

def train_model(epochs=500, lr=1e-3):
    """
    Executes the training sequence on the actual SPX options chain data.
    """
    print("Initiating Empirical Deep BSDE Training Sequence...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepBSDE_RoughVol().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # Load and move empirical data to Device
    X_paths, X_contract, Y_target, price_scaler, spot_scaler, strike_scaler = prepare_empirical_batches()
    X_paths, X_contract, Y_target = X_paths.to(device), X_contract.to(device), Y_target.to(device)
    
    loss_history = []
    
    # Standard PyTorch Training Loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward Pass
        pred_prices, pred_greeks = model(X_paths, X_contract)
        
        # Calculate Loss explicitly against reality
        loss = bsde_empirical_loss(pred_prices, Y_target, pred_greeks)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | BSDE Empirical Error (MSE + Hedging Penalty): {loss.item():.4f}")
            
    print("Optimization Complete.")
    
    # Save the PyTorch model artifact securely
    torch.save(model.state_dict(), "Data/DeepBSDE_empirical.pth")
    return model, loss_history, price_scaler, spot_scaler, strike_scaler

if __name__ == "__main__":
    train_model(epochs=500)
