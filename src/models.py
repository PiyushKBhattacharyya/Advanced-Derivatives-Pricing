import torch
import torch.nn as nn

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
        
    def forward(self, latent_state, contract_terms):
        # Concatenate encoded path memory with current contractual terms
        x = torch.cat([latent_state, maturity_and_strike if 'maturity_and_strike' in locals() else contract_terms], dim=1)
        # Ensure model is in eval mode if batch size is 1 for BatchNorm
        if x.shape[0] == 1:
            self.net.eval()
        return self.net(x)

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
    Research v4: Multi-Token Cross-Asset Attention.
    Encodes correlations between multiple assets by treating each asset's 
    recent path as a distinct temporal token.
    """
    def __init__(self, n_assets=3, seq_len=20, d_model=128, nhead=8, num_layers=3):
        super(MultiAssetTransformerEncoder, self).__init__()
        self.n_assets = n_assets
        self.d_model = d_model
        
        # Project each (Price, Vol) pair into d_model
        self.asset_projection = nn.Linear(2, d_model)
        
        # Asset-specific embeddings to help the transformer distinguish between e.g. AAPL and SPX
        self.asset_embedding = nn.Parameter(torch.randn(1, n_assets, d_model))
        # Temporal positional encoding
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
        
        # Reshape to (Batch * Seq_Len, N_Assets, 2) to project assets individually
        x = x.view(batch_size * seq_len, self.n_assets, 2)
        x = self.asset_projection(x) # (Batch * Seq_Len, N_Assets, d_model)
        
        # Reshape back and add asset/temporal context
        # We treat the sequence as Seq_Len * N_Assets tokens for full cross-correlation attention
        x = x.view(batch_size, seq_len, self.n_assets, self.d_model)
        x = x + self.asset_embedding.unsqueeze(1) # Add asset identity
        x = x + self.pos_embedding.unsqueeze(2)   # Add temporal identity
        
        # Flatten into (Batch, Seq_Len * N_Assets, d_model)
        x = x.view(batch_size, seq_len * self.n_assets, self.d_model)
        
        # Full Cross-Asset-Time Attention
        x = self.transformer(x)
        
        # Global pooling across both time and assets
        latent_state = x.mean(dim=1)
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
        self.pricer = BasketPricerMLP(n_assets=n_assets, latent_dim=d_model, hidden_dim=mlp_hidden)
        self.hedger = BasketHedgingMLP(n_assets=n_assets, latent_dim=d_model, hidden_dim=mlp_hidden)
        
    def forward(self, historical_paths, contract_terms):
        latent_state = self.encoder(historical_paths)
        price = self.pricer(latent_state, contract_terms)
        greeks = self.hedger(latent_state, contract_terms)
        return price, greeks

class DeepBSDE_RoughVol(nn.Module):
    """
    The unified Deep BSDE Architecture for empirical Rough Volatility pricing (European).
    """
    def __init__(self, path_input_dim=2, d_model=64, mlp_hidden=128):
        super(DeepBSDE_RoughVol, self).__init__()
        self.encoder = TransformerRoughEncoder(input_dim=path_input_dim, d_model=d_model)
        self.pricer = PricerMLP(latent_dim=d_model, hidden_dim=mlp_hidden)
        self.hedger = HedgingMLP(latent_dim=d_model, hidden_dim=mlp_hidden)
        
    def forward(self, historical_paths, contract_terms):
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
        
        # 2. Predict Price
        price = self.pricer(latent_state, contract_terms)
        
        # 3. Predict Greeks
        greeks = self.hedger(latent_state, contract_terms)
        
        return price, greeks
