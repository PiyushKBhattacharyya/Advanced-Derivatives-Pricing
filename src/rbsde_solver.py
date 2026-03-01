import torch
import torch.nn as nn
import numpy as np

class RBSDESolver:
    """
    Implements the Reflected BSDE solver for American Options.
    
    The American option price V(t) must satisfy:
    1. V(t) >= h(S_t) (where h is the intrinsic payoff: (S_t - K)^+)
    2. dV(t) = f(t, S_t, V_t, Z_t)dt + Z_t dW_t + dK_t
    3. dK_t is a non-decreasing process that pushes V contextually
       back up if it tries to drop below the obstacle h(S_t).
    """
    
    def __init__(self, model, config=None):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def calculate_intrinsic_payoff(self, S, K, option_type='call'):
        """h(S) = (S - K)+ for calls or (K - S)+ for puts."""
        if option_type == 'call':
            return torch.clamp(S - K, min=0.0)
        return torch.clamp(K - S, min=0.0)

    def rbsde_loss(self, historical_paths, T, K, r, q, sigma, option_type='call'):
        """
        Reflected BSDE Loss Function.
        
        We penalize:
        1. Terminal Condition mismatch: V(T) = h(S_T)
        2. Reflection mismatch: V(t) < h(S_t)
        3. Local Martingale consistency: E[dV + f dt] = 0
        """
        # Batch preparation
        batch_size = historical_paths.shape[0]
        S_t = historical_paths[:, -1, 0:1] # Current spot
        
        # Contract terms: [T-t, K]
        # For simplicity, we assume we are evaluating at t=0
        terms = torch.tensor([[T, K]] * batch_size, dtype=torch.float32).to(self.device)
        
        # 1. Forward Pass
        V_pred, greeks, stop_prob = self.model(historical_paths, terms)
        
        # 2. Obstacle Constraint (Reflection)
        h_t = self.calculate_intrinsic_payoff(S_t, K, option_type)
        reflection_penalty = torch.mean(torch.clamp(h_t - V_pred, min=0.0)**2)
        
        # 3. Early Exercise Boundary Penalty (Stopping Network consistency)
        # If stop_prob > 0.5, then V_pred should be very close to h_t
        stopping_consistency = torch.mean(stop_prob * (V_pred - h_t)**2)
        
        # 4. BSDE Drift Consistency (Full RBSDE with dS Step)
        # We simulate a small risk-neutral step dS and ensure dV is consistent with the Delta.
        dt = 1.0 / 252.0  # One trading day
        dW = torch.randn_like(S_t).to(self.device) * np.sqrt(dt)
        
        # S_next under risk-neutral measure: dS = (r-q)Sdt + sigma*SdW
        S_next = S_t * torch.exp((r - q - 0.5 * sigma**2) * dt + sigma * dW)
        
        # Construct the next historical path window for the next-state prediction
        # We shift the window and append the new spot (keeping VIX/vol constant for this dS step)
        next_paths = torch.cat([
            historical_paths[:, 1:, :], 
            torch.cat([S_next.unsqueeze(1), historical_paths[:, -1:, 1:]], dim=2)
        ], dim=1)
        
        next_terms = torch.tensor([[max(T - dt, 1e-4), K]] * batch_size, dtype=torch.float32).to(self.device)
        
        V_next, _, _ = self.model(next_paths, next_terms)
        
        # dV = V_next - V_pred
        # Expected dV in a BSDE: dV = (r * V)dt + Delta * (dS - (r-q)Sdt)
        delta = greeks[:, 0:1]
        dV_pred = V_next - V_pred
        dS = S_next - S_t
        
        drift_consistency = torch.mean((dV_pred - (r * V_pred * dt + delta * (dS - (r - q) * S_t * dt)))**2)
        
        total_loss = reflection_penalty + 10.0 * stopping_consistency + 5.0 * drift_consistency
        
        return total_loss, V_pred.mean().item()

    def find_optimal_exercise_boundary(self, S_range, T_rem, S_live, option_type, spot_scaler, trail_V, k_scaled, trail_S):
        """
        Scans a range of spot prices at a fixed time-to-maturity to find 
        the price S* where the StoppingNetwork flips from 0 to 1.
        
        Args:
            S_range: numpy array of raw spot prices to scan.
            T_rem: time to maturity.
            S_live: current spot (for reference).
            option_type: 'call' or 'put'.
            spot_scaler: fitted StandardScaler for spot prices.
            trail_V: 20-day historical variance path.
            k_scaled: user-selected Strike price (already scaled).
            trail_S: 20-day historical spot price path.
        """
        self.model.eval()
        
        # 1. Scale S_range using the provided spot_scaler
        S_scaled = spot_scaler.transform(S_range.reshape(-1, 1)).flatten()
        
        # 2. Build mock paths: each path ends at a point in S_scaled 
        # but shares the same 19-day history and trailing VIX.
        batch_size = len(S_range)
        mock_paths = torch.zeros((batch_size, 20, 2)).to(self.device)
        
        # Fill trailing variance
        mock_paths[:, :, 1] = torch.tensor(trail_V, dtype=torch.float32).to(self.device)
        
        # USE REAL MARKET MEMORY: Fill the leading 19 days with actual trail_S (scaled)
        # This makes the "Red Line" much more accurate because it's path-dependent.
        s_hist_scaled = spot_scaler.transform(trail_S.reshape(-1, 1)).flatten()
        mock_paths[:, :19, 0] = torch.tensor(s_hist_scaled[:19], dtype=torch.float32).to(self.device)
        
        # For each point in the scan, the 'latest' spot is the candidate s_star
        mock_paths[:, 19, 0] = torch.tensor(S_scaled, dtype=torch.float32).to(self.device)
        
        # 3. Contract terms
        mock_terms = torch.tensor([[T_rem, k_scaled]] * batch_size, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            _, _, stop_probs = self.model(mock_paths, mock_terms)
            
        probs = stop_probs.cpu().numpy().flatten()
        
        # Find the index where prob crosses 0.5
        # For Call: Optimal stopping happens at HIGH prices.
        # For Put: Optimal stopping happens at LOW prices.
        if option_type == 'call':
            indices = np.where(probs > 0.5)[0]
            return S_range[indices[0]] if len(indices) > 0 else None
        else:
            indices = np.where(probs > 0.6)[0] # Put stopping is cleaner at 0.6
            return S_range[indices[-1]] if len(indices) > 0 else None
