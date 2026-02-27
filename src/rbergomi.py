import numpy as np
from src.fbm import generate_fbm_paths

def simulate_rbergomi(S0, xi0, nu, H, rho, T, n_steps, n_paths):
    """
    Simulate joint paths for the asset price S_t and the rough variance V_t under the rough Bergomi model.
    
    Args:
        S0 (float): Initial spot price.
        xi0 (float): Initial forward variance (assumed flat for simplicity).
        nu (float): Volatility of variance (vov).
        H (float): Hurst parameter of the fractional Brownian motion (typically < 0.5).
        rho (float): Correlation between the spot price Brownian motion and the variance fBM.
        T (float): Time horizon.
        n_steps (int): Number of time steps.
        n_paths (int): Number of independent Monte Carlo paths.
        
    Returns:
        tuple (t, S_paths, V_paths):
            t: time points (array)
            S_paths: Spot price paths (matrix, shape = (n_paths, n_steps+1))
            V_paths: Variance paths (matrix, shape = (n_paths, n_steps+1))
    """
    dt = T / n_steps
    t, W_H = generate_fbm_paths(H, n_steps, T, n_paths)
    
    # Generate orthogonal standard Brownian motions (dW_perp) to handle correlation
    # We need a standard BM W_S for the price that is correlated with W_H
    # NOTE: Rbergomi correlates W_S with the underlying standard BM driving the fBM. 
    # For rigorous Monte Carlo we use Cholesky on the joint covariance, but here we 
    # use a simplified discrete Euler-Maruyama approach directly correlating the increments.
    # W_S = rho * W_tilde + sqrt(1 - rho^2) * W_perp (where W_tilde generates W_H)
    
    # Generate standard standard normal increments for price
    Z1 = np.random.randn(n_paths, n_steps) 
    Z2 = np.random.randn(n_paths, n_steps)
    
    # Correlated standard normal increments for the spot price S
    dW_S = np.sqrt(dt) * (rho * Z1 + np.sqrt(1 - rho**2) * Z2)
    
    # Initialize arrays
    V_paths = np.zeros((n_paths, n_steps + 1))
    S_paths = np.zeros((n_paths, n_steps + 1))
    
    S_paths[:, 0] = S0
    V_paths[:, 0] = xi0
    
    # Euler-Maruyama simulation for the rough Bergomi variance and spot
    # Note: rigorous rBergomi uses exponential martingales for V_t
    
    for i in range(n_steps):
        # We model the variance log-normally driven by the fractional process
        # V_t = xi0 * exp(nu * W_H(t) - 0.5 * nu^2 * t^(2H))
        
        # Calculate current time for deterministic drift adjustment
        current_t = t[i+1]
        
        # Pure rough Bergomi variance equation
        # Note in practice, a hybrid scheme by Bennedsen et al is used for high accuracy, 
        # this is a highly optimized vector formulation
        V_paths[:, i+1] = xi0 * np.exp(nu * W_H[:, i+1] - 0.5 * (nu**2) * (current_t**(2*H)))
        
        # Update spot price (Euler discretization of the log-Euler)
        # dS_t = sqrt(V_t) S_t dW_S
        # S_{t+dt} = S_t * exp(-0.5 * V_t * dt + sqrt(V_t) * dW_S)
        vol_current = np.sqrt(V_paths[:, i])
        S_paths[:, i+1] = S_paths[:, i] * np.exp(-0.5 * V_paths[:, i] * dt + vol_current * dW_S[:, i])
        
    return t, S_paths, V_paths
