import configparser
from src.live_trader import LiveTrader

import os

def main():
    config = configparser.ConfigParser()
    # Get the absolute path to the directory containing main.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the absolute path to config.ini
    config_path = os.path.join(script_dir, 'config', 'config.ini')
    config.read(config_path)

    live_trader = LiveTrader(config)
    live_trader.run()

if __name__ == "__main__":
    main()
