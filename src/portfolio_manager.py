import pandas as pd
try:
    import MetaTrader5 as mt5
except ImportError:
    from . import mock_metatrader5 as mt5

class PortfolioManager:
    def __init__(self, mt5_connection):
        self.mt5_connection = mt5_connection
        self.equity_curve = pd.DataFrame(columns=['time', 'equity'])

    def get_account_info(self):
        """Gets the account information."""
        # Use existing MT5 connection (assumed to be already connected)
        account_info = mt5.account_info()
        return account_info

    def get_positions(self):
        """Gets all open positions."""
        # Use existing MT5 connection (assumed to be already connected)
        positions = mt5.positions_get()
        
        if positions:
            # Convert tuple to DataFrame for easier handling
            positions_df = pd.DataFrame(list(positions), columns=[
                'ticket', 'time', 'time_msc', 'time_update', 'time_update_msc',
                'type', 'magic', 'identifier', 'reason', 'volume', 'price_open',
                'sl', 'tp', 'price_current', 'swap', 'profit', 'symbol', 'comment', 'external_id'
            ])
            return positions_df
        else:
            # Return empty DataFrame
            return pd.DataFrame()

    def get_history(self, start_date, end_date):
        """Gets the trading history for a given period."""
        # Use existing MT5 connection (assumed to be already connected)
        history = mt5.history_deals_get(start_date, end_date)
        return history

    def calculate_equity_curve(self):
        """
        Calculates the equity curve based on the account history.
        This is a simplified implementation. A more robust version would
        track the equity in real-time.
        """
        account_info = self.get_account_info()
        if account_info is None:
            return None

        initial_balance = account_info.balance

        # For this simulation, we'll just use the current equity
        current_equity = account_info.equity

        new_row = pd.DataFrame([{'time': pd.Timestamp.now(), 'equity': current_equity}])
        self.equity_curve = pd.concat([self.equity_curve, new_row], ignore_index=True)

        return self.equity_curve
