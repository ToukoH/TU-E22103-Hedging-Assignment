from utils import delta, black_scholes, vega, implied_volatility
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

msft_option_vegas = [vega(underlying_price, E_msft, (T - day)/365, r, sigma) 
                      for day, underlying_price in enumerate(msft_prices)]
googl_option_vegas = [vega(underlying_price, E_googl, (T - day)/365, r, sigma) 
                      for day, underlying_price in enumerate(googl_prices)]

msft_implied_vols = [implied_volatility(price, underlying_price, E_msft, (T - day)/365, r) 
                     for day, (underlying_price, price) in enumerate(zip(msft_prices, msft_option_prices))]

googl_implied_vols = [implied_volatility(price, underlying_price, E_googl, (T - day)/365, r) 
                      for day, (underlying_price, price) in enumerate(zip(googl_prices, googl_option_prices))]

msft_data_2 = pd.DataFrame({
    'Day': np.arange(1, days + 1),
    'Underlying Price': msft_prices,
    'Call Option Price': msft_option_prices,
    'Option Delta': msft_option_deltas,
    'Option Vega':msft_option_vegas
})

googl_data = pd.DataFrame({
    'Day': np.arange(1, len(googl_prices) + 1),
    'Stock Price': googl_prices,
    'Call Option Price': googl_option_prices,
    'Option Delta': googl_option_deltas,
    'Option Vega': googl_option_vegas
})

def delta_vega_hedge(delta_of_option, vega_of_option, price_of_option):
    delta_hedge_amount = -delta_of_option * price_of_option
    vega_hedge_amount = -vega_of_option
    return delta_hedge_amount, vega_hedge_amount

msft_data_2['Delta Hedge'], msft_data_2['Vega Hedge'] = zip(*msft_data_2.apply(lambda row: delta_vega_hedge(
    row['Option Delta'], 
    row['Option Vega'], 
    row['Call Option Price']
), axis=1))

googl_data['Delta Hedge'], googl_data['Vega Hedge'] = zip(*googl_data.apply(lambda row: delta_vega_hedge(
    row['Option Delta'], 
    row['Option Vega'], 
    row['Call Option Price']
), axis=1))

def calculate_delta_vega_Ai(Ci_plus_1, Ci, Delta_i, Si_plus_1, Si, Vega_i, IV_i, IV_i_plus_1):
    delta_change = Delta_i * (Si_plus_1 - Si)
    vega_change = Vega_i * (IV_i_plus_1 - IV_i)
    total_theoretical_change = delta_change + vega_change
    actual_change = Ci_plus_1 - Ci
    return actual_change - total_theoretical_change

msft_data_2['Implied Volatility'] = msft_implied_vols

msft_data_2['A_i'] = calculate_delta_vega_Ai(
    msft_data_2['Call Option Price'].shift(-1), 
    msft_data_2['Call Option Price'], 
    msft_data_2['Option Delta'], 
    msft_data_2['Underlying Price'].shift(-1), 
    msft_data_2['Underlying Price'],
    msft_data_2['Option Vega'],
    msft_data_2['Implied Volatility'],
    msft_data_2['Implied Volatility'].shift(-1)
)

err_2 = (msft_data_2['A_i'] ** 2).mean()

print(f"MSFT data:\n{msft_data_2}")
print(f"GOOGL data:\n{googl_data}")
print(f"MSE for delta-vega hedging is: {err_2}")
