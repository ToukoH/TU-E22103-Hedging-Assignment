from scripts.utils import vega, black_scholes
import numpy as np
import pandas as pd

days = 5
S = 374.23      # Current stock price
E = 380      # Option strike price
T = 32 #baseline for the time until maturity
          # Time to expiration in years

r = 0.05     # Standard risk free interest rate
sigma = 0.2  # Standard Volatility
option_type = "call"


underlying_prices = list([377.85, 369.14, 372.52, 368.8, 374.23])

call_option_prices = [black_scholes(underlying_price, E, (T - day)/365, r, sigma, option_type) 
                      for day, underlying_price in enumerate(underlying_prices)]

option_deltas = [vega(underlying_price, E, (T - day)/365, r, sigma, option_type) 
                      for day, underlying_price in enumerate(underlying_prices)]


data = pd.DataFrame({
    'Day': np.arange(1, days + 1),
    'Underlying Price': underlying_prices,
    'Call Option Price': call_option_prices,
    'Option Delta': option_deltas
})

def delta_hedge(price_of_underlying, price_of_call_option, delta_of_option):
    """
    Calculate the amount of underlying asset needed to delta hedge a call option position.

    :param price_of_underlying: Current price of the underlying asset.
    :param price_of_call_option: Current price of the call option.
    :param delta_of_option: Delta of the call option.
    :return: Amount of underlying asset to hedge the option position.
    """
    hedge_amount = -delta_of_option * price_of_call_option / price_of_underlying
    return hedge_amount

data['Hedge Amount'] = data.apply(lambda row: delta_hedge(row['Underlying Price'], 
                                                         row['Call Option Price'], 
                                                         row['Option Delta']), axis=1)

print(data)

def calculate_Ai(Ci_plus_1, Ci, Delta_i, Si_plus_1, Si):
    """
    Calculate the difference Ai between the change in value of OP and RE.

    :param Ci_plus_1: Price of the call option at time i+1.
    :param Ci: Price of the call option at time i.
    :param Delta_i: Delta of the option at time i.
    :param Si_plus_1: Price of the underlying asset at time i+1.
    :param Si: Price of the underlying asset at time i.
    :return: Difference Ai.
    """
    return (Ci_plus_1 - Ci) - Delta_i * (Si_plus_1 - Si)

data['A_i'] = calculate_Ai(data['Call Option Price'].shift(-1), 
                          data['Call Option Price'], 
                          data['Option Delta'], 
                          data['Underlying Price'].shift(-1), 
                          data['Underlying Price'])

error = (data['A_i'] ** 2).mean()

print("Total Mean Squared Error:", error)