import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
import torch.nn.functional as F

class NeuralSDECell(nn.Module):
    """
    SOTA v4.1: Neural Stochastic Differential Equation (NSDE) Layer.
    Models the latent state as a continuous flow: dZ_t = f(Z_t)dt + g(Z_t)dW_t
    This allows the model to 'hallucinate' prices between gaps in market data.
    """
    def __init__(self, latent_dim=128, hidden_dim=64):
        super(NeuralSDECell, self).__init__()
        self.latent_dim = latent_dim
        
        # Drift network: f(Z_t)
        self.drift = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Diffusion network: g(Z_t)
        self.diffusion = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Softplus() # Variance must be positive
        )
        
    def forward(self, z, dt=0.01):
        # Euler-Maruyama discretization step
        f = self.drift(z)
        g = self.diffusion(z)
        
        # Brownian motion sample
        dw = torch.randn_like(z) * torch.sqrt(torch.tensor(dt))
        
        # State update
        z_next = z + f * dt + g * dw
        return z_next, f, g

class RegimeVAE(nn.Module):
    """
    SOTA v4.1: Regime-Aware Variational Autoencoder.
    Compresses the latent state and classifies it into 'Regimes' (e.g., Bull, Bear, Flash Crash).
    The bottleneck 'z_regime' acts as a stabilizer for the main pricer.
    """
    def __init__(self, latent_dim=128, n_regimes=3):
        super(RegimeVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_regimes)
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_regimes, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        
    def forward(self, z):
        # Bottleneck classification
        regime_logits = self.encoder(z)
        regime_probs = F.softmax(regime_logits, dim=-1)
        
        # Reconstruct latent state from regime
        z_recon = self.decoder(regime_probs)
        return regime_probs, z_recon

class TransformerRoughEncoder(nn.Module):
    """
    SOTA v3: Encodes non-Markovian historical roughness using self-attention.
    Captures multi-scale self-similarity of SPX/VIX more effectively than LSTMs.
    """
    def __init__(self, input_dim=2, d_model=64, nhead=4, num_layers=2, seq_len=20):
        super(TransformerRoughEncoder, self).__init__()
        self.d_model = d_model
        
        # Linear projection to model dimension
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional Encoding (Learned for short financial sequences)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*2, 
            dropout=0.1, 
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        # x shape: (B, seq_len, input_dim)
        x = self.input_projection(x) # (B, seq_len, d_model)
        x = x + self.pos_embedding   # Add temporal context
        
        x = self.transformer(x)      # (B, seq_len, d_model)
        
        # Global pooling (mean) to compress into a Markovian latent state
        latent_state = x.mean(dim=1) 
        return latent_state

class PricerMLP(nn.Module):
    """
    Feed-Forward network mapping the Latent State and Maturity to the Option Price V(t).
    """
    def __init__(self, latent_dim=64, hidden_dim=128):
        super(PricerMLP, self).__init__()
        # Input features: Latent state (64) + Time to Maturity (1) + Strike Context (1)
        input_dim = latent_dim + 2
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(), # Mish is often smoother than ReLU for BSDE PDE tracking
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Mish(),
            nn.Linear(hidden_dim // 2, 1) # Output is a single Option Price
        )
        self.dropout = nn.Dropout(p=0.1) # For Bayesian Uncertainty (MC Dropout)
        
    def forward(self, latent_state, contract_terms, mc_dropout=False):
        # Concatenate encoded path memory with current contractual terms
        x = torch.cat([latent_state, contract_terms], dim=1)
        
        # Ensure model is in eval mode if batch size is 1 for BatchNorm
        # EXCEPT when we need dropout active for Bayesian uncertainty
        if x.shape[0] == 1:
            self.net.eval()
            
        x = self.net[0:2](x) # Linear + Mish
        x = self.net[2](x)   # BatchNorm
        
        # MC Dropout: Active if mc_dropout=True, regardless of global train/eval state
        x = F.dropout(x, p=0.1, training=mc_dropout or self.net.training)
        
        x = self.net[3:](x)  # Rest of the network
        return x

class HedgingMLP(nn.Module):
    """
    Feed-Forward network mapping the Latent State to the Hedging Greeks (Delta, Vega).
    """
    def __init__(self, latent_dim=64, hidden_dim=128):
        super(HedgingMLP, self).__init__()
        input_dim = latent_dim + 2
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 2) # Output: [Delta (w.r.t Spot), Vega (w.r.t VIX)]
        )
        
    def forward(self, latent_state, maturity_and_strike):
        x = torch.cat([latent_state, maturity_and_strike], dim=1)
        return self.net(x)

class StoppingNetwork(nn.Module):
    """
    Binary classifier calculating the 'Optimal Stopping Time'.
    Outputs a probability [0, 1] that the contract should be physically exercised
    prematurely based on the current latent state and time-to-maturity.
    """
    def __init__(self, latent_dim=64, hidden_dim=64):
        super(StoppingNetwork, self).__init__()
        input_dim = latent_dim + 2 # Latent Path + Time + Strike
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() # Boundary Classifier
        )
        
    def forward(self, latent_state, maturity_and_strike):
        x = torch.cat([latent_state, maturity_and_strike], dim=1)
        return self.net(x)

class AmericanDeepBSDE(nn.Module):
    """
    Unified Architecture for American Options.
    Combines Path Encoding, Pricing, Hedging, and Optimal Stopping classification.
    """
    def __init__(self, path_input_dim=2, d_model=64, mlp_hidden=128):
        super(AmericanDeepBSDE, self).__init__()
        self.encoder = TransformerRoughEncoder(input_dim=path_input_dim, d_model=d_model)
        self.pricer  = PricerMLP(latent_dim=d_model, hidden_dim=mlp_hidden)
        self.hedger  = HedgingMLP(latent_dim=d_model, hidden_dim=mlp_hidden)
        self.stopper = StoppingNetwork(latent_dim=d_model, hidden_dim=64)
        
    def forward(self, historical_paths, contract_terms):
        """
        Returns:
            price (Batch, 1): Premium
            greeks (Batch, 2): Delta, Vega
            stopping_prob (Batch, 1): Probability of optimal exercise
        """
        latent_state = self.encoder(historical_paths)
        price = self.pricer(latent_state, contract_terms)
        greeks = self.hedger(latent_state, contract_terms)
        stopping_prob = self.stopper(latent_state, contract_terms)
        
        return price, greeks, stopping_prob

class MultiAssetTransformerEncoder(nn.Module):
    """
    SOTA v4.1: Multi-Token Cross-Asset Attention with Lead-Lag Augmentation.
    Implicitly detects temporal delays between assets (e.g., Asset A leading Asset B)
    by expanding the input sequence into Lead and Lag components.
    """
    def __init__(self, n_assets=3, seq_len=20, d_model=128, nhead=8, num_layers=4):
        super(MultiAssetTransformerEncoder, self).__init__()
        self.n_assets = n_assets
        self.d_model = d_model
        
        # Lead-Lag Augmentation: We project (Price, Vol) for both Lead and Lag paths
        # Input features per asset: 2 (Price, Vol) -> Total 4 with Lead-Lag
        self.asset_projection = nn.Linear(2, d_model)
        
        # Identity embeddings: Asset ID + Lead/Lag ID + Temporal ID
        self.asset_embedding = nn.Parameter(torch.randn(1, n_assets, d_model))
        self.lead_lag_embedding = nn.Parameter(torch.randn(1, 2, d_model)) # [Lead, Lag]
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*4, 
            dropout=0.1, 
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        # x shape: (Batch, Seq_Len, N_Assets * 2)
        batch_size, seq_len, _ = x.shape
        
        # 1. Lead-Lag Path Augmentation
        # Lead: Current prices (t)
        # Lag: Shifted prices (t-1)
        x_reshaped = x.view(batch_size, seq_len, self.n_assets, 2)
        
        # Create Lagged path (pad with zero at the start)
        x_lag = torch.zeros_like(x_reshaped)
        x_lag[:, 1:, :, :] = x_reshaped[:, :-1, :, :]
        
        # 2. Project and add embeddings
        # (Batch, Seq_Len, N_Assets, 2) -> (Batch, Seq_Len, N_Assets, d_model)
        feat_lead = self.asset_projection(x_reshaped) + self.lead_lag_embedding[:, 0:1, :]
        feat_lag = self.asset_projection(x_lag) + self.lead_lag_embedding[:, 1:2, :]
        
        # Add Asset and Position identity
        # Broadcast embeddings: (1, 1, N, d) and (1, Seq, 1, d)
        feat_lead = feat_lead + self.asset_embedding.unsqueeze(1) + self.pos_embedding.unsqueeze(2)
        feat_lag = feat_lag + self.asset_embedding.unsqueeze(1) + self.pos_embedding.unsqueeze(2)
        
        # 3. Concatenate Lead and Lag tokens (treat as separate tokens for the transformer)
        # Shape: (Batch, Seq_Len * N_Assets * 2, d_model)
        x_combined = torch.cat([feat_lead, feat_lag], dim=2) # Combine at asset level
        x_flat = x_combined.view(batch_size, seq_len * self.n_assets * 2, self.d_model)
        
        # 4. Global Attention
        x_out = self.transformer(x_flat)
        
        # Global pooling (take mean across all asset-time-lag tokens)
        latent_state = x_out.mean(dim=1)
        return latent_state

class BasketPricerMLP(nn.Module):
    def __init__(self, n_assets=3, latent_dim=128, hidden_dim=256):
        super(BasketPricerMLP, self).__init__()
        # Input: Latent state + Maturity + N_Asset Weights (optional) + Strike
        input_dim = latent_dim + 2 
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, latent_state, contract_terms):
        x = torch.cat([latent_state, contract_terms], dim=1)
        if x.shape[0] == 1:
            self.net.eval()
        return self.net(x)

class BasketHedgingMLP(nn.Module):
    def __init__(self, n_assets=3, latent_dim=128, hidden_dim=256):
        super(BasketHedgingMLP, self).__init__()
        input_dim = latent_dim + 2
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, n_assets + 1) # Output: [Delta1, Delta2, ..., DeltaN, BasketVega]
        )
        
    def forward(self, latent_state, contract_terms):
        x = torch.cat([latent_state, contract_terms], dim=1)
        if x.shape[0] == 1:
            self.net.eval()
        return self.net(x)

class BasketDeepBSDE(nn.Module):
    """
    SOTA v4: Multi-Asset Basket Option Pricing Engine.
    Quotes weights for N assets simultaneously while considering cross-correlation.
    """
    def __init__(self, n_assets=3, d_model=128, mlp_hidden=256):
        super(BasketDeepBSDE, self).__init__()
        self.encoder = MultiAssetTransformerEncoder(n_assets=n_assets, d_model=d_model)
        self.sde = NeuralSDECell(latent_dim=d_model)
        self.pricer = BasketPricerMLP(n_assets=n_assets, latent_dim=d_model, hidden_dim=mlp_hidden)
        self.hedger = BasketHedgingMLP(n_assets=n_assets, latent_dim=d_model, hidden_dim=mlp_hidden)
        
    def forward(self, historical_paths, contract_terms, dt=0.01, mc_dropout=False):
        latent_state = self.encoder(historical_paths)
        
        # NSDE Latent Flow: Models continuous evolution between observations
        latent_state, drift, diff = self.sde(latent_state, dt=dt)
        
        price = self.pricer(latent_state, contract_terms, mc_dropout=mc_dropout)
        greeks = self.hedger(latent_state, contract_terms)
        return price, greeks, (drift, diff)

class DeepBSDE_RoughVol(nn.Module):
    """
    The unified Deep BSDE Architecture for empirical Rough Volatility pricing (European).
    """
    def __init__(self, path_input_dim=2, d_model=64, mlp_hidden=128):
        super(DeepBSDE_RoughVol, self).__init__()
        self.encoder = TransformerRoughEncoder(input_dim=path_input_dim, d_model=d_model)
        self.sde = NeuralSDECell(latent_dim=d_model)
        self.pricer = PricerMLP(latent_dim=d_model, hidden_dim=mlp_hidden)
        self.hedger = HedgingMLP(latent_dim=d_model, hidden_dim=mlp_hidden)
        
    def forward(self, historical_paths, contract_terms, dt=0.01, mc_dropout=False):
        """
        Args:
            historical_paths: Tensor of shape (Batch, Seq_Len, 2) -> Empirical trailing SPX and VIX.
            contract_terms: Tensor of shape (Batch, 2) -> Options Time to Maturity (T-t) and Strike (K).
            
        Returns:
            price (Batch, 1): The predicted Option Premium
            greeks (Batch, 2): The predicted Delta and Vega for the replicating portfolio
        """
        # 1. Compress non-Markovian history into Markovian latent state
        latent_state = self.encoder(historical_paths)
        
        # 2. NSDE Evolution (Stochastic Continuous Flow)
        latent_state, drift, diff = self.sde(latent_state, dt=dt)
        
        # 3. Predict Price (with optional Bayesian MC Dropout)
        price = self.pricer(latent_state, contract_terms, mc_dropout=mc_dropout)
        
        # 4. Predict Greeks
        greeks = self.hedger(latent_state, contract_terms)
        
        return price, greeks, (drift, diff)
