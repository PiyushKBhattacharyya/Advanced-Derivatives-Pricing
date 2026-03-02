import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

from src.models import DeepBSDE_RoughVol, AmericanDeepBSDE
import pandas as pd
from src.baselines import get_empirical_dataset
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import pickle

# Global scalers to inverse-transform predictions later
price_scaler = StandardScaler()
spot_scaler = StandardScaler()
strike_scaler = StandardScaler()

SCALER_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data", "scalers.pkl")

# Phase 6: Deep Hedging Frictions
# 0.0002 = 2 basis points per ticket (typical Tier-1 Execution Cost)
EXECUTION_COST = 0.0002 

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
    # If file exists, we can optionally load instead of fit, but for training we fit.
    trailing_spx_scaled = spot_scaler.fit_transform(trailing_spx.reshape(-1, 1)).flatten()
    strikes_scaled = strike_scaler.fit_transform(strikes.reshape(-1, 1)).flatten()
    market_prices_scaled = price_scaler.fit_transform(market_prices.reshape(-1, 1)).flatten()
    
    # Save the fitted scalers for the dashboard
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump({'price': price_scaler, 'spot': spot_scaler, 'strike': strike_scaler}, f)
    
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
    Calculates the Deep Hedging Loss (Friction-Aware BSDE).
    We optimize precisely for:
    1.  Price Convergence (MSE)
    2.  Hedging Stability (Variance Penalty)
    3.  Transaction Cost Minimization (Absolute Greek Penalty)
    """
    mse_loss = nn.MSELoss()(predicted_prices, market_prices)
    
    # 1. Variance Control (Penalize extreme, unstable hedging leverage)
    variance_penalty = 1e-4 * torch.mean(predicted_greeks**2)
    
    # 2. Transaction Cost (Friction) Penalty
    # We penalize the absolute size of the Greeks as a proxy for total 
    # portfolio turnover costs in a discrete-time daily rebalancing regime.
    friction_penalty = EXECUTION_COST * torch.mean(torch.abs(predicted_greeks))
    
    return mse_loss + variance_penalty + friction_penalty

def train_model(epochs=2000, lr=1e-3):
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
    
    # 80/20 Train/Validation Split for Early Stopping
    dataset_size = len(X_paths)
    val_size = int(0.2 * dataset_size)
    train_size = dataset_size - val_size
    
    indices = torch.randperm(dataset_size)
    train_indices, val_indices = indices[:train_size], indices[train_size:]
    
    X_train_paths, X_train_contract, Y_train = X_paths[train_indices], X_contract[train_indices], Y_target[train_indices]
    X_val_paths, X_val_contract, Y_val = X_paths[val_indices], X_contract[val_indices], Y_target[val_indices]
    
    loss_history = []
    
    # Early Stopping Config
    patience = 50
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Standard PyTorch Training Loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward Pass
        pred_prices, pred_greeks = model(X_train_paths, X_train_contract)
        
        # Calculate Loss explicitly against reality
        loss = bsde_empirical_loss(pred_prices, Y_train, pred_greeks)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        # Validation Pass for Early Stopping
        model.eval()
        with torch.no_grad():
            val_pred_prices, val_pred_greeks = model(X_val_paths, X_val_contract)
            val_loss = bsde_empirical_loss(val_pred_prices, Y_val, val_pred_greeks).item()
        model.train()
        
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            # Clone state_dict to avoid reference mutation
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()} 
        else:
            patience_counter += 1
            
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Patience: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            print(f"Early Stopping Triggered at Epoch {epoch+1}. Restoring best weights (Val Loss: {best_loss:.4f})")
            model.load_state_dict(best_model_state)
            break
            
    print("Optimization Complete.")
    
    # Save the PyTorch model artifact securely
    torch.save(model.state_dict(), "Data/DeepBSDE_empirical.pth")
    return model, loss_history, price_scaler, spot_scaler, strike_scaler
def train_american_model(epochs=1000, lr=1e-3):
    """
    Trains the StoppingNetwork (Early Exercise Boundary) for American Options.
    This leverages Transfer Learning by freezing the pre-trained European weights 
    and solely optimizing the stopping probabilities against intrinsic value.
    """
    print("Initiating American Deep BSDE Training Sequence (Optimal Stopping)...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AmericanDeepBSDE(d_model=64).to(device)
    
    # ALWAYS load empirical Base Weights for Pricing/Hedging
    empirical_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data", "DeepBSDE_empirical.pth")
    if os.path.exists(empirical_path):
         # strict=False allows loading shared Transformer weights while ignoring new MLP branches
         model.load_state_dict(torch.load(empirical_path, map_location=device), strict=False)
         print(f"Loaded Pre-Trained Transformer Core from {empirical_path}")
    else:
         print("WARNING: Pre-trained empirical model not found. Training from scratch.")
         
    # Freeze the Core modules to prevent catastrophic forgetting
    for param in model.encoder.parameters(): param.requires_grad = False
    for param in model.pricer.parameters(): param.requires_grad = False
    for param in model.hedger.parameters(): param.requires_grad = False
    
    # We only optimize the stopper network
    optimizer = optim.AdamW(model.stopper.parameters(), lr=lr)
    
    # Load and move empirical data
    X_paths, X_contract, _, price_scaler, spot_scaler, strike_scaler = prepare_empirical_batches()
    X_paths, X_contract = X_paths.to(device), X_contract.to(device)
    
    # For American Calls, the Intrinsic Value is max(S-K, 0)
    # We create a target stopping probability: 1 if S > K (deep in money), else 0
    # True Longstaff-Schwartz compares this to continuation value, but for the surrogate target,
    # we enforce a smooth Sigmoid-based cross entropy around the Strike.
    
    # De-scale Spots and Strikes to compare them in the physical domain
    spots_scaled = X_paths[:, -1, 0].cpu().numpy().reshape(-1, 1)
    strikes_scaled = X_contract[:, 1].cpu().numpy().reshape(-1, 1)
    
    spots_physical = spot_scaler.inverse_transform(spots_scaled)
    strikes_physical = strike_scaler.inverse_transform(strikes_scaled)
    
    # Target Prob = 1 if Spot > Strike * 1.05 (In the money enough to justify exercise)
    target_probs = (spots_physical > (strikes_physical * 1.02)).astype(np.float32)
    target_probs_tensor = torch.tensor(target_probs).to(device)
    
    loss_history = []
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward Pass (we only care about stopping_prob here)
        _, _, stopping_prob = model(X_paths, X_contract)
        
        # Binary Cross Entropy Loss against our theoretical stopping rule
        loss = F.binary_cross_entropy(stopping_prob, target_probs_tensor)
        
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Boundary Classification Error (BCE): {loss.item():.4f}")
            
    print("American Optimization Complete.")
    
    # Save the PyTorch model artifact securely
    torch.save(model.state_dict(), "Data/DeepBSDE_american.pth")
    return model, loss_history

if __name__ == "__main__":
    train_model(epochs=2000)
    train_american_model(epochs=1000)
