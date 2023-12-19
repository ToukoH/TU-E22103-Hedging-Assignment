import numpy as np
import pandas as pd

days = 5

underlying_prices = list([377.85, 369.14, 372.52, 368.8, 374.23])
call_option_prices = list([18.9455, 14.49, 11.65, 13.30, 15.45])
option_deltas = list([0.5072, 0.3418, 0.4026, 0.3280, 0.4146])

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

data.head()

