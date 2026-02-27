import numpy as np

def price_european_call(S_paths, K, T, r=0.0):
    """
    Prices a European Call option via Monte Carlo simulation.
    
    Args:
        S_paths (np.ndarray): Simulated spot price paths, shape (n_paths, n_steps + 1)
        K (float): Strike price
        T (float): Time to maturity
        r (float): Risk-free interest rate
        
    Returns:
        float: Estimated option price
        float: Standard error of the estimate
    """
    # Terminal payoffs at time T (the last column of the paths matrix)
    terminal_prices = S_paths[:, -1]
    payoffs = np.maximum(terminal_prices - K, 0)
    
    # Discounted payoffs
    discounted_payoffs = np.exp(-r * T) * payoffs
    
    price = np.mean(discounted_payoffs)
    std_error = np.std(discounted_payoffs) / np.sqrt(len(payoffs))
    
    return price, std_error

def price_asian_call(S_paths, K, T, r=0.0):
    """
    Prices an arithmetic Asian Call option via Monte Carlo simulation.
    """
    # Average price along each path
    average_prices = np.mean(S_paths, axis=1)
    payoffs = np.maximum(average_prices - K, 0)
    
    discounted_payoffs = np.exp(-r * T) * payoffs
    
    price = np.mean(discounted_payoffs)
    std_error = np.std(discounted_payoffs) / np.sqrt(len(payoffs))
    
    return price, std_error

def price_lookback_call(S_paths, K, T, r=0.0):
    """
    Prices a Lookback Call option (floating strike) via Monte Carlo simulation.
    """
    max_prices = np.max(S_paths, axis=1)
    payoffs = np.maximum(max_prices - K, 0)
    
    discounted_payoffs = np.exp(-r * T) * payoffs
    
    price = np.mean(discounted_payoffs)
    std_error = np.std(discounted_payoffs) / np.sqrt(len(payoffs))
    
    return price, std_error
