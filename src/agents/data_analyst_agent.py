from .base_agent import Agent
from ..data_collector import EconomicCalendarCollector
import pandas as pd

class DataAnalystAgent(Agent):
    def __init__(self, name, mt5_collector):
        super().__init__(name)
        self.mt5_collector = mt5_collector
        self.economic_calendar_collector = EconomicCalendarCollector()

    def execute(self, task):
        """
        Executes a data collection task.
        """
        if task['source'] == 'mt5':
            data = {}
            # Use existing MT5 connection (assumed to be already connected)
            for timeframe in task['timeframes']:
                # Handle different formats of num_bars
                if isinstance(task['num_bars'], dict):
                    num_bars = task['num_bars'].get(timeframe, 100)
                else:
                    num_bars = task['num_bars']
                
                df = self.mt5_collector.get_historical_data(
                    task['symbol'], timeframe, num_bars
                )
                data[timeframe] = df
            
            return data
        elif task['source'] == 'economic_calendar':
            return self.economic_calendar_collector.get_economic_calendar()
        return pd.DataFrame()
