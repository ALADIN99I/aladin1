import pandas as pd
import numpy as np

class SyntheticPortfolioManager:
    def __init__(self, portfolio_definition=None):
        """
        Initializes the SyntheticPortfolioManager.

        :param portfolio_definition: A dictionary defining the synthetic portfolio,
                                     e.g., {'buy': ['USD', 'JPY'], 'sell': ['EUR', 'GBP']}
        """
        self.portfolio_definition = portfolio_definition
        self.history = []

    def set_portfolio_definition(self, portfolio_definition):
        """
        Sets or updates the portfolio definition.

        :param portfolio_definition: A dictionary defining the synthetic portfolio.
        """
        self.portfolio_definition = portfolio_definition
        self.history = [] # Reset history when definition changes

    def calculate_portfolio_value(self, ufo_data, timeframe):
        """
        Calculates the current value of the synthetic portfolio based on UFO data.

        :param ufo_data: The UFO data containing currency strength values.
        :param timeframe: The timeframe to use for the calculation.
        :return: The calculated portfolio value, or None if not possible.
        """
        if not self.portfolio_definition or not ufo_data:
            return None

        try:
            # Extract the relevant timeframe data
            if timeframe not in ufo_data['raw_data']:
                return None

            strength_data = ufo_data['raw_data'][timeframe]

            # Get the latest strength values
            latest_strengths = strength_data.iloc[-1]

            buy_strength = 0
            for currency in self.portfolio_definition.get('buy', []):
                if currency in latest_strengths:
                    buy_strength += latest_strengths[currency]

            sell_strength = 0
            for currency in self.portfolio_definition.get('sell', []):
                if currency in latest_strengths:
                    sell_strength += latest_strengths[currency]

            portfolio_value = buy_strength - sell_strength
            return portfolio_value

        except Exception as e:
            print(f"Error calculating portfolio value: {e}")
            return None

    def update_history(self, timestamp, portfolio_value):
        """
        Updates the history of the portfolio's value.

        :param timestamp: The timestamp of the new value.
        :param portfolio_value: The new portfolio value.
        """
        self.history.append({'timestamp': timestamp, 'value': portfolio_value})

    def get_history_df(self):
        """
        Returns the portfolio history as a pandas DataFrame.

        :return: A DataFrame with 'timestamp' and 'value' columns.
        """
        return pd.DataFrame(self.history)

    def analyze_portfolio_trend(self, lookback_period=10):
        """
        Analyzes the trend of the portfolio using a simple moving average.

        :param lookback_period: The number of periods to use for the moving average.
        :return: A dictionary with trend information.
        """
        if len(self.history) < lookback_period:
            return {'trend': 'undetermined', 'ma': None}

        history_df = self.get_history_df()
        history_df['ma'] = history_df['value'].rolling(window=lookback_period).mean()

        latest_ma = history_df['ma'].iloc[-1]
        previous_ma = history_df['ma'].iloc[-2]

        if latest_ma > previous_ma:
            trend = 'up'
        elif latest_ma < previous_ma:
            trend = 'down'
        else:
            trend = 'flat'

        return {'trend': trend, 'ma': latest_ma}
