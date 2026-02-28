import numpy as np
from scipy.stats import norm

def black_scholes_call(S, K, T, r, sigma):
    """Standard Black-Scholes European Call pricing"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_delta(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

def sabr_implied_vol(F, K, T, alpha, beta, rho, nu):
    """
    Hagan's Algebraic 2002 SABR Volatility Approximation.
    Used by Tier-1 trading desks to parametrically map the volatility smile continuously.
    """
    # Prevent divide by zero edge cases mapping strictly to minimums
    eps = 1e-7
    F = np.maximum(F, eps)
    K = np.maximum(K, eps)
    
    if np.abs(F - K) < eps:
        # At-The-Money (ATM) Formula constraints
        term1 = alpha / (F**(1-beta))
        term2 = 1.0 + (((1-beta)**2 / 24.0) * (alpha**2 / (F**(2-2*beta))) +
                       (rho * beta * nu * alpha)/(4.0 * F**(1-beta)) +
                       ((2.0 - 3.0 * rho**2) * nu**2)/24.0) * T
        return term1 * term2
    else:
        # Out-of-The-Money (OTM / ITM) Skew Mapping
        z = (nu / alpha) * ((F * K)**((1-beta)/2.0)) * np.log(F / K)
        x = np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho))
        
        term1 = alpha / ((F*K)**((1-beta)/2.0) * (1 + ((1-beta)**2/24.0)*(np.log(F/K))**2 + ((1-beta)**4/1920.0)*(np.log(F/K))**4))
        term2 = z / np.maximum(x, eps)
        term3 = 1.0 + (((1-beta)**2 / 24.0) * (alpha**2 / ((F*K)**(1-beta))) +
                       (rho * beta * nu * alpha)/(4.0 * (F*K)**((1-beta)/2.0)) +
                       ((2.0 - 3.0 * rho**2) * nu**2)/24.0) * T
        
        return term1 * term2 * term3

def sabr_call_price(F, K, T, r, alpha, beta, rho, nu):
    """
    Synthesizes the deterministic SABR implied volatility matrix identically back 
    into standard Black-Scholes arrays to output exact legacy derivative prices.
    """
    sigma_sabr = sabr_implied_vol(F, K, T, alpha, beta, rho, nu)
    return black_scholes_call(F, K, T, r, sigma_sabr)

def deterministic_local_vol_call(S, K, T, r, sigma_atm, a=-1.5, b=0.5):
    """
    Deterministic Local Volatility (Dupire Proxy).
    Tier-1 banks use this to fit strictly to the current instantaneous smile, 
    but it is famously brittle for forward-prediction dynamics.
    We approximate the implied local topology via a rigid quadratic smile matrix.
    """
    moneyness = np.log(S / K)
    # Quadratic deterministic skew overlay parameter limits
    skewed_vol = sigma_atm * (1.0 + a * moneyness + b * moneyness**2)
    skewed_vol = np.maximum(skewed_vol, 1e-4) # Floor bounds
    
    return black_scholes_call(S, K, T, r, skewed_vol)
