import pandas as pd
import numpy as np
from datetime import datetime, time
import pytz
try:
    import MetaTrader5 as mt5
except ImportError:
    from . import mock_metatrader5 as mt5

from .ufo_trading_engine import UFOTradingEngine

class SimulationUFOTradingEngine(UFOTradingEngine):
    """
    Simulation-aware UFO Trading Engine that uses simulation time instead of real time
    This allows proper testing of historical dates and scenarios
    """
    
    def __init__(self, config, simulation_time=None):
        super().__init__(config)
        self.simulation_time = simulation_time
        
    def set_simulation_time(self, simulation_time):
        """Set the current simulation time"""
        self.simulation_time = simulation_time
    
    def should_trade_now(self):
        """
        Determines if trading should occur based on SIMULATION session timing
        Uses simulation_time instead of real current time
        """
        if self.simulation_time is None:
            # Fallback to parent implementation if no simulation time set
            return super().should_trade_now()
            
        # Use simulation time instead of current time
        sim_time_utc = self.simulation_time.replace(tzinfo=pytz.UTC)
        london_time = sim_time_utc.astimezone(self.session_timezone)
        current_time = london_time.time()
        current_weekday = london_time.weekday()  # 0=Monday, 6=Sunday
        
        # Define trading sessions
        asian_session = time(23, 0) <= current_time or current_time <= time(8, 0)
        london_session = time(8, 0) <= current_time <= time(16, 0)
        ny_session = time(13, 0) <= current_time <= time(22, 0)
        
        # Check weekends - August 1st, 2025 is FRIDAY (weekday=4), so should trade!
        if current_weekday >= 5:  # Saturday (5) or Sunday (6)
            return False, f"Weekend - No trading (Day {current_weekday})"
        
        # Log simulation time for debugging
        sim_date_str = sim_time_utc.strftime('%A, %Y-%m-%d %H:%M GMT')
        
        # Prefer active sessions
        if london_session or ny_session:
            return True, f"Active session on {sim_date_str} - London: {london_session}, NY: {ny_session}"
        elif asian_session:
            return True, f"Asian session on {sim_date_str} - Limited trading"
        else:
            return False, f"Between sessions on {sim_date_str}"
    
    def should_close_for_session_end(self, economic_events=None):
        """
        Determines if positions should be closed due to session ending
        Uses simulation time and actual economic calendar data instead of hardcoded news times
        """
        if self.simulation_time is None:
            return super().should_close_for_session_end()
            
        sim_time_utc = self.simulation_time.replace(tzinfo=pytz.UTC)
        london_time = sim_time_utc.astimezone(self.session_timezone)
        current_time_london = london_time.time()
        current_weekday = london_time.weekday()
        
        # For news periods, use GMT time directly (simulation time is already in GMT)
        current_time_gmt = sim_time_utc.time()
        
        # Close before weekend
        if current_weekday == 4 and current_time_london >= time(21, 0):  # Friday 9 PM London
            return True, "Weekend closure - Friday evening"
            
        # Close at 8 PM GMT (20:00) - UFO methodology end of analysis period
        if current_time_gmt >= time(20, 0):
            return True, "End of UFO analysis period (8 PM GMT)"
            
        # Use actual economic calendar data if provided
        if economic_events is not None and not economic_events.empty:
            # Check for high-impact events in the next 30 minutes
            current_hour = current_time_gmt.hour
            current_minute = current_time_gmt.minute
            
            # Look for high-impact events in current hour or next 30 minutes
            high_impact_events = economic_events[
                (economic_events['impact'] == 'High') &
                (
                    # Events in current hour
                    (economic_events['gmt_hour'] == current_hour) |
                    # Events in next hour if we're in the last 30 minutes
                    ((current_minute >= 30) & (economic_events['gmt_hour'] == (current_hour + 1) % 24))
                )
            ]
            
            if not high_impact_events.empty:
                event_details = []
                for _, event in high_impact_events.iterrows():
                    event_time = f"{event['gmt_hour']:02d}:{event['gmt_minute']:02d}"
                    event_details.append(f"{event_time} GMT: {event['country']} {event['title']}")
                
                return True, f"High-impact economic events approaching: {'; '.join(event_details)}"
        
                
        return False, "Normal trading hours"
