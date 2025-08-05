import pandas as pd

class SyntheticPortfolioManager:
    def __init__(self):
        self.portfolios = {}

    def group_trades(self, trades):
        """
        Groups trades into synthetic portfolios based on a common currency.
        For example, all trades involving GBP will be grouped into a 'GBP' portfolio.
        """
        self.portfolios = {}
        for trade in trades:
            base_currency, quote_currency = self.get_currencies(trade['symbol'])
            if base_currency not in self.portfolios:
                self.portfolios[base_currency] = []
            self.portfolios[base_currency].append(trade)

            if quote_currency not in self.portfolios:
                self.portfolios[quote_currency] = []
            self.portfolios[quote_currency].append(trade)

    def get_currencies(self, symbol):
        """Extracts the base and quote currencies from a symbol."""
        # A more robust implementation would handle different symbol formats
        return symbol[:3], symbol[3:6]

    def calculate_portfolio_performance(self, portfolio_name):
        """
        Calculates the combined performance of a synthetic portfolio.
        """
        if portfolio_name not in self.portfolios:
            return None

        # This is a simplified implementation. A more robust version would
        # track the performance over time and generate a chart.
        total_pnl = 0
        for trade in self.portfolios[portfolio_name]:
            total_pnl += trade.get('profit', 0.0)

        return total_pnl
