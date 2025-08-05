# This is a mock implementation of the MetaTrader5 library
# to allow for development on non-Windows platforms.

import pandas as pd
import numpy as np

TIMEFRAME_M5 = 5
TIMEFRAME_M15 = 15
TIMEFRAME_H1 = 16385
TIMEFRAME_H4 = 16388
TIMEFRAME_D1 = 16408

# Order types
ORDER_TYPE_BUY = 0
ORDER_TYPE_SELL = 1

def initialize(login, password, server, path=None):
    print(f"Mock MT5: Initializing with login={login}, server={server}, path={path}")
    return True

def shutdown():
    print("Mock MT5: Shutting down.")

def copy_rates_from_pos(symbol, timeframe, start_pos, count):
    print(f"Mock MT5: Fetching {count} bars of {symbol} on timeframe {timeframe}...")

    # Generate more realistic data with some randomness and trend
    np.random.seed(hash(symbol) % (2**32 - 1))  # Seed with symbol hash for deterministic randomness
    base_price = 1.0655
    trend = np.random.uniform(-0.0001, 0.0001)
    volatility = np.random.uniform(0.0001, 0.0005)

    data = {
        'time': [pd.to_datetime('2023-01-02 00:00:00') + pd.Timedelta(minutes=5*i) for i in range(count)],
        'open': [],
        'high': [],
        'low': [],
        'close': [],
    }

    current_price = base_price
    for i in range(count):
        open_price = current_price
        close_price = open_price + trend + np.random.normal(0, volatility)
        high_price = max(open_price, close_price) + np.random.uniform(0, volatility)
        low_price = min(open_price, close_price) - np.random.uniform(0, volatility)

        data['open'].append(open_price)
        data['high'].append(high_price)
        data['low'].append(low_price)
        data['close'].append(close_price)

        current_price = close_price

    return pd.DataFrame(data).to_records(index=False)

def last_error():
    return "No error"
