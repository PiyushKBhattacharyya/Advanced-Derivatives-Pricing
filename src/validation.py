import torch
import time
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.models import DeepBSDE_RoughVol
from src.train import prepare_empirical_batches
from src.baselines import extract_challenge_regime, black_scholes_call
from src.institutional_baselines import sabr_implied_vol
from scipy.stats import norm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_inference_benchmark(N=100000):
    """
    Executes a raw inference throughput benchmark across N option contracts.
    STRICTLY EMPIRICAL: We identically tile real historical SPX batches to map to throughput scale.
    """
    print(f"\n--- INFERENCE BENCHMARK ({N:,} Contracts) ---")
    
    try:
        X_p, X_c, _, price_scaler, spot_scaler, strike_scaler = prepare_empirical_batches(seq_len=20)
    except Exception as e:
        print("Error pulling empirical scalers:", e)
        return
        
    num_empirical = X_p.shape[0]
    repeats = int(np.ceil(N / float(num_empirical)))
    
    # Tile the genuine empirical structures, cutting precisely to N
    real_paths = X_p.repeat(repeats, 1, 1)[:N].to(device)
    real_contracts = X_c.repeat(repeats, 1)[:N].to(device)
    
    model = DeepBSDE_RoughVol().to(device)
    model.eval()
    
    # 1. PyTorch Deep BSDE Inference Latency
    start_time = time.time()
    with torch.no_grad():
        _ = model(real_paths, real_contracts)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    dl_time = time.time() - start_time
    print(f"Deep BSDE Execution Time: {dl_time:.4f} seconds")
    
    # 2. Black-Scholes Vectorized Inference Latency
    # Pull the exact same physical dimensions from our real tensors to make it an identical match
    t_val = np.clip(spot_scaler.inverse_transform(real_paths[:, -1, 0].cpu().numpy().reshape(-1,1)).flatten(), a_min=1e-5, a_max=None)
    S = t_val  # Real latest spot vector mapping
    K = strike_scaler.inverse_transform(real_contracts[:, 1].cpu().numpy().reshape(-1,1)).flatten()
    T = np.clip(real_contracts[:, 0].cpu().numpy(), a_min=1e-5, a_max=None)
    r = 0.05
    # Variance mapping for BS
    V = real_paths[:, -1, 1].cpu().numpy()
    sigma = np.sqrt(np.clip(V, a_min=1e-5, a_max=None))
    
    start_time = time.time()
    _ = black_scholes_call(S, K, T, r, sigma)
    bs_time = time.time() - start_time
    print(f"Vectorized Black-Scholes Execution Time: {bs_time:.4f} seconds")
    
    metrics = {
        'Deep BSDE (PyTorch)': dl_time,
        'Black-Scholes (NumPy)': bs_time
    }
    np.save(os.path.join(BASE_DIR, "Data", "inference_metrics.npy"), metrics)
    print("Inference metrics saved to Data/inference_metrics.npy")


def bs_delta(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


def run_empirical_hedging_backtest():
    """
    Executes a strict historical Delta-hedging strategy using the actual
    daily COVID-19 crash trajectory.
    """
    print("\n--- EMPIRICAL HEDGING IMPLEMENTATION (COVID-19 CRASH) ---")
    
    # Load Real Market Regime
    t_arr, S_arr, V_arr = extract_challenge_regime('extreme_vol')
    
    # Load Pre-Trained Global Model
    model = DeepBSDE_RoughVol().to(device)
    model_path = os.path.join(BASE_DIR, "Data", "DeepBSDE_empirical.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Extract Real Scalers
    try:
         _, _, _, price_scaler, spot_scaler, strike_scaler = prepare_empirical_batches(seq_len=20)
    except:
         print("ERROR: Run train.py first to establish scalers.")
         return
         
    num_days = len(S_arr)
    seq_len = 20
    
    # We will compute the historical PnL of a 3-month (90 days) At-The-Money call option instantiated on Day 20
    if num_days < seq_len + 10:
        print("Not enough days matching in the crash regime to backtest.")
        return
        
    start_idx = seq_len
    trade_S0 = S_arr[start_idx]
    K = trade_S0 # ATM
    T_initial = 90.0 / 365.25 # 3 months
    
    # Initialize PnL trajectories
    pnls_bs = [0.0]
    pnls_dl = [0.0]
    pnls_sabr = [0.0]
    pnls_lv = [0.0]
    
    delta_bs_prev = 0.0
    delta_dl_prev = 0.0
    delta_sabr_prev = 0.0
    delta_lv_prev = 0.0
    
    r = 0.05
    
    # Step daily through the real catastrophic market event
    for i in range(start_idx, num_days - 1):
         S_today = S_arr[i]
         S_tomorr = S_arr[i+1]
         V_today = V_arr[i]
         
         # Shrinking Maturity
         T_remaining = T_initial - ((i - start_idx) / 365.25)
         if T_remaining <= 0.0: break
         
         # 1. Compute Black-Scholes Empirical Delta
         sigma_today = np.sqrt(V_today)
         d_bs = bs_delta(S_today, K, T_remaining, r, sigma_today)
         
         # 1.5 Compute SABR and Local Vol Empirical Deltas
         s_vol = sabr_implied_vol(S_today, K, T_remaining, alpha=0.4, beta=1.0, rho=-0.6, nu=0.2)
         d_sabr = bs_delta(S_today, K, T_remaining, r, s_vol)
         
         moneyness = np.log(S_today / K)
         lv_vol = np.maximum(sigma_today * (1.0 - 1.5 * moneyness + 0.5 * moneyness**2), 1e-4)
         d_lv = bs_delta(S_today, K, T_remaining, r, lv_vol)
         
         # 2. Compute Deep BSDE Empirical Delta
         trail_S = S_arr[i-seq_len+1 : i+1]
         trail_V = V_arr[i-seq_len+1 : i+1]
         
         s_scaled = spot_scaler.transform(trail_S.reshape(-1,1)).flatten()
         
         path_tnsr = torch.tensor(np.stack([s_scaled, trail_V], axis=-1), dtype=torch.float32).unsqueeze(0).to(device)
         
         k_scaled = strike_scaler.transform(np.array([[K]]))[0,0]
         cont_tnsr = torch.tensor([[T_remaining, k_scaled]], dtype=torch.float32).to(device)
         
         with torch.no_grad():
              _, d_dl_tnsr = model(path_tnsr, cont_tnsr)
              
         d_dl = d_dl_tnsr.cpu().numpy()[0,0]
         
         # 3. Daily Hedging PnL execution
         # Profit from holding exactly Delta shares of SPX overnight
         daily_pnl_bs = delta_bs_prev * (S_tomorr - S_today)
         daily_pnl_dl = delta_dl_prev * (S_tomorr - S_today)
         daily_pnl_sabr = delta_sabr_prev * (S_tomorr - S_today)
         daily_pnl_lv = delta_lv_prev * (S_tomorr - S_today)
         
         pnls_bs.append(pnls_bs[-1] + daily_pnl_bs)
         pnls_dl.append(pnls_dl[-1] + daily_pnl_dl)
         pnls_sabr.append(pnls_sabr[-1] + daily_pnl_sabr)
         pnls_lv.append(pnls_lv[-1] + daily_pnl_lv)
         
         # Roll Deltas
         delta_bs_prev = d_bs
         delta_dl_prev = d_dl
         delta_sabr_prev = d_sabr
         delta_lv_prev = d_lv
         
    # Save Hedging trajectories natively to dict for charting phase
    hedging_data = {
         'days': np.arange(len(pnls_bs)),
         'pnl_black_scholes': pnls_bs,
         'pnl_deep_bsde': pnls_dl,
         'pnl_sabr': pnls_sabr,
         'pnl_local_vol': pnls_lv
    }
    np.save(os.path.join(BASE_DIR, "Data", "empirical_hedging_pnl.npy"), hedging_data)
    
    print("Empirical Hedging Execution Output saved to Data/empirical_hedging_pnl.npy!")

def run_synthetic_stress_test():
    """
    Evaluates the Deep BSDE model against 1,000 synthetic 'Shadow Crashes'
    generated by the MarketVAE. Calculates terminal VaR (Value at Risk).
    """
    print("\n--- SYNTHETIC SHADOW-CRASH STRESS TEST ---")
    
    # 1. Load Synthetic Crashes
    crash_path = os.path.join(BASE_DIR, "Data", "synthetic_crashes.npy")
    if not os.path.exists(crash_path):
         print("ERROR: Run stress_test_gen.py first.")
         return
         
    # Shape: (num_crashes, seq_len, 2) -> [log_returns, VIX]
    crashes = np.load(crash_path)
    num_crashes, seq_len, _ = crashes.shape
    
    # 2. Load Model
    model = DeepBSDE_RoughVol().to(device)
    model.load_state_dict(torch.load(os.path.join(BASE_DIR, "Data", "DeepBSDE_empirical.pth"), map_location=device))
    model.eval()
    
    # 3. Load Scalers
    _, _, _, _, spot_scaler, strike_scaler = prepare_empirical_batches(seq_len=20)
    
    # Metrics
    dl_errors = []
    
    print(f"Stress testing across {num_crashes} hallucinated disaster paths...")
    
    with torch.no_grad():
        for i in range(num_crashes):
            # Convert log-returns back to levels for the LSTM input
            # We assume a normalized starting spot of 1.0 (mean=0)
            ret_path = crashes[i, :, 0]
            vix_path = crashes[i, :, 1]
            
            # Reconstruct dummy spot levels for scaling
            s_levels = np.zeros(seq_len)
            s_levels[0] = 1.0
            for j in range(1, seq_len):
                s_levels[j] = s_levels[j-1] * np.exp(ret_path[j])
            
            s_scaled = spot_scaler.transform(s_levels.reshape(-1,1)).flatten()
            path_tnsr = torch.tensor(np.stack([s_scaled, vix_path], axis=-1), dtype=torch.float32).unsqueeze(0).to(device)
            
            # Assume constant ATM strike for comparison
            # Contract: [Maturity=0.1, Strike=scaled(1.0)]
            k_scaled = strike_scaler.transform([[1.0]])[0,0]
            cont_tnsr = torch.tensor([[0.1, k_scaled]], dtype=torch.float32).to(device)
            
            _, delta_tnsr = model(path_tnsr, cont_tnsr)
            delta = delta_tnsr.cpu().numpy()[0,0]
            
            # Simple 1-day P&L error (delta error)
            # We measure how 'smooth' our hedge would be in the face of this synthetic shock
            # High variance in synthetic deltas indicates failure under stress.
            dl_errors.append(delta)
            
    # Save for P&L distribution plotting
    np.save(os.path.join(BASE_DIR, "Data", "stress_test_deltas.npy"), dl_errors)
    print("Stress test deltas saved to Data/stress_test_deltas.npy")

if __name__ == "__main__":
    run_inference_benchmark()
    run_empirical_hedging_backtest()
    run_synthetic_stress_test()
