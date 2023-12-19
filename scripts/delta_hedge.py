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

# Implied volatility is really close to 0.2 so we assume it to be constant when calculating greeks and option prices.
# We use these values to calculate the error so delta and delta-vega don't yield the same A_i.
msft_implied_vols = [implied_volatility(price, underlying_price, E_msft, (T - day)/365, r) 
                     for day, (underlying_price, price) in enumerate(zip(msft_prices, msft_option_prices))]
googl_implied_vols = [implied_volatility(price, underlying_price, E_googl, (T - day)/365, r) 
                      for day, (underlying_price, price) in enumerate(zip(googl_prices, googl_option_prices))]

msft_data = pd.DataFrame({
    'Day': np.arange(1, days + 1),
    'Underlying Price': msft_prices,
    'Call Option Price': msft_option_prices,
    'Option Delta': msft_option_deltas,
    'Option Vega':msft_option_vegas
})

def delta_hedge(price_of_underlying, price_of_call_option, delta_of_option):
    hedge_amount = -delta_of_option * price_of_call_option / price_of_underlying
    return hedge_amount

msft_data['Hedge Amount'] = msft_data.apply(lambda row: delta_hedge(row['Underlying Price'], 
                                                         row['Call Option Price'], 
                                                         row['Option Delta']), axis=1)

# The replicating portfolio is implicitly here.
def calculate_Ai(Ci_plus_1, Ci, Delta_i, Si_plus_1, Si):
    return (Ci_plus_1 - Ci) - Delta_i * (Si_plus_1 - Si)

msft_data['A_i'] = calculate_Ai(msft_data['Call Option Price'].shift(-1), 
                          msft_data['Call Option Price'], 
                          msft_data['Option Delta'], 
                          msft_data['Underlying Price'].shift(-1), 
                          msft_data['Underlying Price'])

err = (msft_data['A_i'] ** 2).mean()

print(msft_data)
print("Total Mean Squared Error when hedging every day:", err)

initial_option_quantity = 10
portfolio_value = []
transactions = []
option_position = initial_option_quantity
underlying_position = 0

print("\nSIMULATING PORTFOLIO OF 10 OPTIONS WITH THE SAME UNDERLYING")
print("_______________________________________________________")
for day in range(days):
    S = msft_prices[day]
    option_price = msft_option_prices[day]
    option_delta = msft_option_deltas[day]

    desired_underlying_position = option_position * option_delta
    underlying_position_change = desired_underlying_position - underlying_position
    transactions.append(underlying_position_change)
    underlying_position = desired_underlying_position

    portfolio_value_today = option_position * option_price + underlying_position * S
    portfolio_value.append(portfolio_value_today)

for day, (value, transaction) in enumerate(zip(portfolio_value, transactions)):
    transaction_type = "Bought" if transaction > 0 else "Sold"
    print(f"Day {day + 1}: Portfolio Value = {value:.2f}, {transaction_type} {abs(transaction):.2f} shares")


print("\nSIMULATING THE SAME PORTFOLIO WITH TRANSACTION COSTS")
print("_______________________________________________________")
initial_option_quantity = 10
portfolio_value = []
transactions = []
option_position = initial_option_quantity
underlying_position = 0
transaction_cost_percentage = 0.05  # 5% transaction cost

for day in range(days):
    S = msft_prices[day]
    option_price = msft_option_prices[day]
    option_delta = msft_option_deltas[day]

    desired_underlying_position = option_position * option_delta
    underlying_position_change = desired_underlying_position - underlying_position

    if underlying_position_change != 0:
        transaction_cost = abs(underlying_position_change * S) * transaction_cost_percentage
    else:
        transaction_cost = 0

    transactions.append(underlying_position_change)
    underlying_position = desired_underlying_position

    portfolio_value_today = option_position * option_price + underlying_position * S - transaction_cost
    portfolio_value.append(portfolio_value_today)

for day, (value, transaction) in enumerate(zip(portfolio_value, transactions)):
    transaction_type = "Bought" if transaction > 0 else "Sold"
    print(f"Day {day + 1}: Portfolio Value = {value:.2f}, {transaction_type} {abs(transaction):.2f} shares")
