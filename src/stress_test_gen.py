import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import sys

# Ensure base directory is in path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

class MarketVAE(nn.Module):
    """
    Variational Autoencoder (VAE) for generating synthetic market regimes.
    Models the joint S&P 500 and VIX dynamics.
    We use a VAE instead of a pure GAN for better training stability on 
    time-series financial data.
    """
    def __init__(self, seq_len=20, latent_dim=32):
        super(MarketVAE, self).__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        
        # Encoder: Time-series trajectories -> Latent Space
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * 2, 128),
            nn.Mish(),
            nn.Linear(128, 64),
            nn.Mish()
        )
        
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        
        # Decoder: Latent Space -> Synthetic Trajectory
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.Mish(),
            nn.Linear(64, 128),
            nn.Mish(),
            nn.Linear(128, seq_len * 2),
            nn.Unflatten(1, (seq_len, 2)) # (Batch, Seq_Len, 2)
        )
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        return self.decoder(z)
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss_function(recon_x, x, mu, logvar):
    MSE = nn.MSELoss()(recon_x, x)
    # KL Divergence: Forces latent space to be unit Gaussian (N(0,1))
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + 0.01 * KLD # Weighted KL to prioritize reconstruction

def train_market_vae(epochs=200):
    """
    Trains the MarketVAE on historical SPX/VIX trajectories.
    """
    print("\n[QUANT ARMOURY] Training MarketVAE for Synthetic Crash Generation...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = MarketVAE().to(device)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    
    # Load data from Data/ directory (assumes market_paths.py has run)
    spx_path = os.path.join(BASE_DIR, "Data", "SPX_history.csv")
    vix_path = os.path.join(BASE_DIR, "Data", "VIX_history.csv")
    
    if not (os.path.exists(spx_path) and os.path.exists(vix_path)):
        print("Error: Historical data missing. Run src/market_paths.py first.")
        return None
        
    spx_df = pd.read_csv(spx_path, index_col=0, parse_dates=True)
    vix_df = pd.read_csv(vix_path, index_col=0, parse_dates=True)
    
    # Preprocessing: Normalize
    s_vals = spx_df.values.flatten()
    v_vals = vix_df.values.flatten() / 100.0 # VIX as decimal
    
    # Log-returns for SPX, raw levels for VIX
    s_returns = np.diff(np.log(s_vals))
    v_levels = v_vals[1:]
    
    # Create windows
    seq_len = 20
    data_windows = []
    for i in range(len(s_returns) - seq_len):
        window = np.stack([s_returns[i:i+seq_len], v_levels[i:i+seq_len]], axis=1)
        data_windows.append(window)
        
    data_tensor = torch.tensor(np.array(data_windows), dtype=torch.float32).to(device)
    
    vae.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        recon_batch, mu, logvar = vae(data_tensor)
        loss = vae_loss_function(recon_batch, data_tensor, mu, logvar)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"VAE Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.6f}")
            
    torch.save(vae.state_dict(), os.path.join(BASE_DIR, "Data", "MarketVAE.pth"))
    print("[SUCCESS] MarketVAE saved to Data/MarketVAE.pth")
    return vae

def generate_synthetic_crashes(num_crashes=1000, seq_len=20):
    """
    Samples from the latent space of the trained VAE to produce 
    out-of-sample "Shadow Crashes".
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = MarketVAE(seq_len=seq_len).to(device)
    model_path = os.path.join(BASE_DIR, "Data", "MarketVAE.pth")
    if not os.path.exists(model_path):
        vae = train_market_vae()
    else:
        vae.load_state_dict(torch.load(model_path, map_location=device))
    
    vae.eval()
    with torch.no_grad():
        # Specifically sample extreme z-values to simulate tail-risk events
        # N(0, 1) * 2.0 pushes samples into the +/- 2 sigma tails
        z = torch.randn(num_crashes, vae.latent_dim).to(device) * 2.5
        synthetic_trajectories = vae.decode(z).cpu().numpy()
        
    # synthetic_trajectories shape: (num_crashes, seq_len, 2)
    # Index 0: log-returns, Index 1: VIX levels
    np.save(os.path.join(BASE_DIR, "Data", "synthetic_crashes.npy"), synthetic_trajectories)
    print(f"[SUCCESS] Generated {num_crashes} synthetic shadow crashes for stress testing.")

if __name__ == "__main__":
    generate_synthetic_crashes()
