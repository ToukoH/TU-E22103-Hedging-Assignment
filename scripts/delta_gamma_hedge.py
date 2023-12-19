from utils import delta, black_scholes, gamma
import numpy as np
import pandas as pd

days = 10
E_msft = 380
E_googl = 138      # Option strike price
T = 39 #24 days to 27th of November 
          # Time to expiration in years

r = 0.05     # Standard risk free interest rate
sigma = 0.2  # Standard Volatility
option_type = "call"

msft_prices = list([378.61, 382.65, 378.85, 378.91, 374.51, 377.85, 369.14, 372.52, 368.8, 374.23])
googl_prices = list([136.41, 137.20, 134.99, 132.53, 131.86, 129.27, 130.99, 130.02, 136.93, 133.29])


msft_option_prices = [black_scholes(underlying_price, E_msft, (T - day)/365, r, sigma, option_type) 
                      for day, underlying_price in enumerate(msft_prices)]
googl_option_prices = [black_scholes(underlying_price, E_googl, (T - day)/365, r, sigma, option_type) 
                      for day, underlying_price in enumerate(googl_prices)]

msft_option_deltas = [delta(underlying_price, E_msft, (T - day)/365, r, sigma, option_type) 
                      for day, underlying_price in enumerate(msft_prices)]
googl_option_deltas = [delta(underlying_price, E_googl, (T - day)/365, r, sigma, option_type) 
                      for day, underlying_price in enumerate(googl_prices)]

msft_option_gammas = [gamma(underlying_price, E_msft, (T - day)/365, r, sigma) 
                      for day, underlying_price in enumerate(msft_prices)]

msft_data = pd.DataFrame({
    'Day': np.arange(1, days + 1),
    'Underlying Price': msft_prices,
    'Call Option Price': msft_option_prices,
    'Option Delta': msft_option_deltas,
})

msft_data['Option Gamma'] = msft_option_gammas

def delta_gamma_hedge(delta_of_option, gamma_of_option, price_of_underlying):
    delta_hedge_amount = -delta_of_option * price_of_underlying
    gamma_hedge_amount = -0.5 * gamma_of_option * price_of_underlying**2
    return delta_hedge_amount, gamma_hedge_amount

msft_data[['Delta Hedge', 'Gamma Hedge']] = msft_data.apply(
    lambda row: delta_gamma_hedge(row['Option Delta'], row['Option Gamma'], row['Underlying Price']), axis=1, result_type='expand')

def calculate_delta_gamma_Ai(Ci_plus_1, Ci, Delta_i, Gamma_i, Si_plus_1, Si):
    delta_component = Delta_i * (Si_plus_1 - Si)
    gamma_component = 0.5 * Gamma_i * (Si_plus_1 - Si)**2
    return (Ci_plus_1 - Ci) - delta_component - gamma_component

msft_data['Call Option Price_next'] = msft_data['Call Option Price'].shift(-1)
msft_data['Underlying Price_next'] = msft_data['Underlying Price'].shift(-1)

msft_data['A_i'] = msft_data.apply(
    lambda row: calculate_delta_gamma_Ai(
        row['Call Option Price_next'], 
        row['Call Option Price'], 
        row['Option Delta'], 
        row['Option Gamma'],
        row['Underlying Price_next'], 
        row['Underlying Price']
    ), axis=1)

msft_data = msft_data.drop(['Call Option Price_next', 'Underlying Price_next'], axis=1)

err_delta_gamma = (msft_data['A_i'] ** 2).mean()

print(msft_data[['Day', 'Underlying Price', 'Option Delta', 'Option Gamma', 'Delta Hedge', 'Gamma Hedge', 'A_i']])
print("Total Mean Squared Error for delta-gamma hedging:", err_delta_gamma)
