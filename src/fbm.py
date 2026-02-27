import numpy as np
from scipy.linalg import cholesky

def generate_fbm(H, n, T):
    """
    Generates fractional Brownian motion (fBM) paths using the Cholesky decomposition of the exact covariance matrix.
    Useful for simulating rough volatility where H < 0.5.

    Args:
        H (float): Hurst parameter (0 < H < 1).
        n (int): Number of time steps.
        T (float): Total time horizon.
    
    Returns:
        tuple: (time vector t, fbm path values)
    """
    if not (0 < H < 1):
        raise ValueError("Hurst parameter H must be between 0 and 1.")
    
    t = np.linspace(0, T, n + 1)
    
    # 1. Build the auto-covariance matrix for fBM
    # Cov(t_i, t_j) = 0.5 * (t_i^(2H) + t_j^(2H) - |t_i - t_j|^(2H))
    # We build a grid of indices to calculate this efficiently avoiding loops
    grid_i, grid_j = np.meshgrid(t[1:], t[1:], indexing='ij')
    cov = 0.5 * (grid_i**(2*H) + grid_j**(2*H) - np.abs(grid_i - grid_j)**(2*H))
    
    # 2. Add small jitter for numerical stability during Cholesky decomposition
    jitter = 1e-8 * np.eye(n)
    
    # 3. Perform Cholesky Decomposition
    L = cholesky(cov + jitter, lower=True)
    
    # 4. Multiply by standard normal vector
    z = np.random.randn(n)
    path_increments = L @ z
    
    # 5. Insert the starting point (at t=0, B_H(t) = 0)
    fbm_path = np.insert(path_increments, 0, 0.0)
    
    return t, fbm_path

def generate_fbm_paths(H, n, T, n_paths):
    """
    Generates multiple independent paths of fractional Brownian Motion.
    """
    paths = np.zeros((n_paths, n + 1))
    t = np.linspace(0, T, n + 1)
    
    # Build covariance matrix once
    grid_i, grid_j = np.meshgrid(t[1:], t[1:], indexing='ij')
    cov = 0.5 * (grid_i**(2*H) + grid_j**(2*H) - np.abs(grid_i - grid_j)**(2*H))
    L = cholesky(cov + 1e-8 * np.eye(n), lower=True)
    
    # Generate all paths efficiently
    z = np.random.randn(n, n_paths) # shape (n_steps, n_paths)
    path_increments = L @ z
    paths[:, 1:] = path_increments.T
    
    return t, paths
