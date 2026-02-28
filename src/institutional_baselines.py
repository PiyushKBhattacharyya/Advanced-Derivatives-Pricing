import numpy as np
import pandas as pd
from scipy.stats import norm
import os
import sys

# Ensure base directory is in path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

def sabr_implied_vol(F, K, T, alpha, beta, rho, volvol):
    """
    Hagan et al. (2002) SABR Implied Volatility Approximation.
    Used for Interest Rate and FX desks globally.
    """
    if T <= 0: return 0.0
    if F <= 0 or K <= 0: return 0.0
    
    logFK = np.log(F / K)
    fkmid = (F * K) ** ((1 - beta) / 2)
    
    # Check for At-The-Money (ATM) case
    if abs(F - K) < 1e-6:
        gamma1 = beta / F
        gamma2 = -beta * (1 - beta) / (F**2)
        vol = alpha / (F**(1 - beta)) * (
            1 + (
                ((1 - beta)**2 / 24 * alpha**2 / (F**(2 - 2*beta))) +
                (1/4 * rho * beta * alpha * volvol / (F**(1 - beta))) +
                ((2 - 3*rho**2) / 24 * volvol**2)
            ) * T
        )
        return vol

    zeta = (volvol / alpha) * (F * K)**((1 - beta) / 2) * logFK
    x_zeta = np.log((np.sqrt(1 - 2*rho*zeta + zeta**2) + zeta - rho) / (1 - rho))
    
    vol = (alpha / (fkmid * (1 + (1-beta)**2 / 24 * logFK**2 + (1-beta)**4 / 1920 * logFK**4))) * (zeta / x_zeta) * (
        1 + (
            ((1 - beta)**2 / 24 * alpha**2 / fkmid**2) +
            (1/4 * rho * beta * alpha * volvol / fkmid) +
            ((2 - 3*rho**2) / 24 * volvol**2)
        ) * T
    )
    return vol

def simulate_rough_bergomi(S0, H, eta, rho, xi, T, n_steps=100, n_paths=10000):
    """
    Rough Bergomi (rBergomi) Monte Carlo Simulation.
    Models volatility as a fractional Brownian motion (H < 0.5).
    """
    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)
    
    # 1. Generate Fractional Brownian Motion via Cholesky decomposition of Covariance matrix
    # Cov(t, s) = 0.5 * (t^(2H) + s^(2H) - |t-s|^(2H))
    times = np.linspace(dt, T, n_steps)
    T_i, T_j = np.meshgrid(times, times)
    cov = 0.5 * (T_i**(2*H) + T_j**(2*H) - np.abs(T_i - T_j)**(2*H))
    
    L = np.linalg.cholesky(cov)
    W1 = np.random.standard_normal((n_paths, n_steps))
    W2 = np.random.standard_normal((n_paths, n_steps))
    
    # Correlated Brownian motion for Spot
    Z = rho * W1 + np.sqrt(1 - rho**2) * W2
    
    # Volatility process (V_t) - Stochastic Variance
    # We use a simplified rough decay integral for the rBergomi kernel
    # V_t = xi * exp(eta * sqrt(2H) * W_H(t) - 0.5 * eta^2 * t^(2H))
    W_H = (L @ W1.T).T # (n_paths, n_steps)
    
    V = np.zeros((n_paths, n_steps + 1))
    V[:, 0] = xi
    
    for i in range(n_steps):
        # rBergomi variance update
        V[:, i+1] = xi * np.exp(eta * W_H[:, i] - 0.5 * (eta**2) * (times[i]**(2*H)))
        
    # 2. Integrate Spot Price path S_t
    S = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0
    
    for i in range(n_steps):
        # Euler-Maruyama discretization
        S[:, i+1] = S[:, i] * np.exp(-0.5 * V[:, i] * dt + np.sqrt(V[:, i] * dt) * Z[:, i])
        
    return S, V

def generate_institutional_benchmarks():
    """
    Generates and saves benchmark pricing data for the comparison phase.
    """
    print("\n[INSTITUTIONAL BENCHMARKS] Generating SABR and rBergomi data...")
    
    # 1. SABR Surface
    F = 5000.0
    strikes = np.linspace(4500, 5500, 20)
    maturities = np.array([0.1, 0.25, 0.5, 1.0])
    
    # Typical SPX parameters for SABR
    alpha, beta, rho, volvol = 0.2, 0.7, -0.6, 0.4
    
    sabr_data = []
    for T in maturities:
        for K in strikes:
            iv = sabr_implied_vol(F, K, T, alpha, beta, rho, volvol)
            sabr_data.append({'T': T, 'K': K, 'IV_SABR': iv})
            
    df_sabr = pd.DataFrame(sabr_data)
    df_sabr.to_csv(os.path.join(BASE_DIR, "Data", "benchmark_sabr.csv"), index=False)
    
    # 2. rBergomi Paths (Simplified)
    # We simulate a small batch to store for visualization comparisons
    S_rB, V_rB = simulate_rough_bergomi(S0=5000.0, H=0.1, eta=2.0, rho=-0.9, xi=0.04, T=1.0, n_steps=50, n_paths=100)
    
    np.save(os.path.join(BASE_DIR, "Data", "benchmark_rbergomi_S.npy"), S_rB)
    np.save(os.path.join(BASE_DIR, "Data", "benchmark_rbergomi_V.npy"), V_rB)
    
    print("[SUCCESS] Institutional benchmarks cached to Data/")

if __name__ == "__main__":
    generate_institutional_benchmarks()
