from datetime import datetime, time
from src.simulation_ufo_engine import SimulationUFOTradingEngine
import configparser
import os

# Load config exactly like the simulation
config = configparser.ConfigParser()
config_path = os.path.join('config', 'config.ini')
config.read(config_path)

# Test times from the simulation logs
test_times = [
    datetime(2025, 8, 1, 8, 0),    # 08:00 GMT - Cycle 13
    datetime(2025, 8, 1, 8, 40),   # 08:40 GMT - Cycle 14
    datetime(2025, 8, 1, 9, 20),   # 09:20 GMT - Cycle 15
    datetime(2025, 8, 1, 10, 0),   # 10:00 GMT - Cycle 16
    datetime(2025, 8, 1, 11, 20),  # 11:20 GMT - Cycle 18
]

# Initialize exactly like the simulation
ufo_engine = SimulationUFOTradingEngine(config, datetime(2025, 8, 1))

for test_time in test_times:
    print(f'\n=== Testing {test_time.strftime("%H:%M GMT")} ===')
    ufo_engine.set_simulation_time(test_time)
    should_close, reason = ufo_engine.should_close_for_session_end()
    print(f'should_close_for_session_end(): {should_close} - {reason}')
    
    # Also test the other method used in execution phase
    should_trade, trade_reason = ufo_engine.should_open_new_trades(None, None, None)
    print(f'should_open_new_trades(): {should_trade} - {trade_reason}')
