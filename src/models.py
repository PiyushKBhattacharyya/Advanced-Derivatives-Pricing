import torch
import torch.nn as nn

class RoughPathEncoder(nn.Module):
    """
    Encodes the non-Markovian historical roughness of the S&P 500 and VIX.
    Takes a sliding window of past (Spot, Variance) pairs and outputs a 
    Markovian latent state summary.
    """
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2):
        super(RoughPathEncoder, self).__init__()
        # Input: (Batch_Size, Sequence_Length, 2 features: Spot and VIX)
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )
        
    def forward(self, x):
        # x shape: (B, seq_len, 2)
        _, (h_n, _) = self.lstm(x)
        # We take the final hidden state of the top LSTM layer (shape: B, hidden_dim)
        latent_state = h_n[-1]
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
        
    def forward(self, latent_state, maturity_and_strike):
        # Concatenate encoded path memory with current contractual terms
        x = torch.cat([latent_state, maturity_and_strike], dim=1)
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

class DeepBSDE_RoughVol(nn.Module):
    """
    The unified Deep BSDE Architecture for empirical Rough Volatility pricing.
    """
    def __init__(self, path_input_dim=2, lstm_hidden=64, mlp_hidden=128):
        super(DeepBSDE_RoughVol, self).__init__()
        self.encoder = RoughPathEncoder(input_dim=path_input_dim, hidden_dim=lstm_hidden)
        self.pricer = PricerMLP(latent_dim=lstm_hidden, hidden_dim=mlp_hidden)
        self.hedger = HedgingMLP(latent_dim=lstm_hidden, hidden_dim=mlp_hidden)
        
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
