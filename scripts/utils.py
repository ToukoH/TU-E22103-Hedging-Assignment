import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def black_scholes(S, E, T, r, sigma, option_type='call'):
    """
    S: Current price of the underlying asset
    K: Strike price of the option
    T: Time to expiration in years
    r: Risk-free interest rate (annual)
    sigma: Volatility of the underlying asset
    option_type: 'call' or 'put'
    """
    d1 = (np.log(S / E) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - E * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = E * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    return price

def delta(S, E, T, r, sigma, option_type='call'):
    d1 = (np.log(S / E) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return norm.cdf(d1)
    elif option_type == 'put':
        return -norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

def vega(S, E, T, r, sigma):
    d1 = (np.log(S / E) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

def implied_volatility(price, S, E, T, r):
    def objective(sigma):
        return black_scholes(S, E, T, r, sigma) - price

    sigma_min, sigma_max = 0.001, 5.0  # Reasonable bounds for volatility
    try:
        implied_vol = brentq(objective, sigma_min, sigma_max)
    except ValueError:
        print("No solution found within given bounds.")
        return None
    except RuntimeError:
        print("Failed to converge to a solution.")
        return None

    return implied_vol
