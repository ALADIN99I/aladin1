import pandas as pd
try:
    import MetaTrader5 as mt5
except ImportError:
    from . import mock_metatrader5 as mt5

class MT5DataCollector:
    def __init__(self, login, password, server, path):
        self.login = int(login)
        self.password = password
        self.server = server
        self.path = path

    def connect(self):
        """Connects to the MetaTrader 5 terminal."""
        if not mt5.initialize(path=self.path, login=self.login, password=self.password, server=self.server):
            print(f"Failed to initialize MT5: {mt5.last_error()}")
            return False
        print("MT5 initialized successfully.")
        return True

    def disconnect(self):
        """Shuts down the connection to the MetaTrader 5 terminal."""
        mt5.shutdown()
        print("MT5 connection shut down.")

    def get_historical_data(self, symbol, timeframe, num_bars=1000):
        """Gets historical bar data for a given symbol and timeframe."""
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
        if rates is None:
            print(f"Failed to get rates for {symbol}: {mt5.last_error()}")
            return None

        rates_df = pd.DataFrame(rates)
        rates_df['time'] = pd.to_datetime(rates_df['time'], unit='s')
        return rates_df

    def get_live_data(self, symbol, timeframe, prev_time=None):
        """
        Gets the latest bar data for live trading simulation.
        If prev_time is provided, it will only return a new bar if the time has changed.
        """
        latest_rate = mt5.copy_rates_from_pos(symbol, timeframe, 0, 1)
        if latest_rate is None or len(latest_rate) == 0:
            return None

        rates_df = pd.DataFrame(latest_rate)
        rates_df['time'] = pd.to_datetime(rates_df['time'], unit='s')

        if prev_time is not None and rates_df['time'].iloc[0] == prev_time:
            return None  # No new bar yet

        return rates_df
    
    def get_daily_ufo_data(self, symbol, timeframe=None):
        """
        Gets data from 0 GMT today until current time for UFO analysis.
        Returns 240 M5 bars (0 GMT to 8 PM GMT) for proper UFO calculation.
        """
        import datetime
        import pytz
        
        if timeframe is None:
            timeframe = mt5.TIMEFRAME_M5
            
        # Get current UTC time
        now_utc = datetime.datetime.now(pytz.UTC)
        
        # Calculate today's 0 GMT start
        today_start = datetime.datetime.combine(
            now_utc.date(), 
            datetime.time(0, 0, 0)
        ).replace(tzinfo=pytz.UTC)
        
        # Calculate end time (8 PM GMT or current time, whichever is earlier)
        today_8pm = datetime.datetime.combine(
            now_utc.date(), 
            datetime.time(20, 0, 0)
        ).replace(tzinfo=pytz.UTC)
        
        end_time = min(now_utc, today_8pm)
        
        # Convert to MT5 timestamp format
        start_timestamp = int(today_start.timestamp())
        end_timestamp = int(end_time.timestamp())
        
        # Fetch data from start to current time
        rates = mt5.copy_rates_range(symbol, timeframe, start_timestamp, end_timestamp)
        
        if rates is None:
            print(f"Failed to get daily UFO data for {symbol}: {mt5.last_error()}")
            return None
            
        rates_df = pd.DataFrame(rates)
        rates_df['time'] = pd.to_datetime(rates_df['time'], unit='s')
        
        print(f"UFO Daily Data: {len(rates_df)} bars from {today_start.strftime('%H:%M GMT')} to {end_time.strftime('%H:%M GMT')}")
        
        return rates_df

import finnhub

import time

class FinnhubDataCollector:
    def __init__(self, api_key, cache_duration=3600):
        self.api_key = api_key
        self.client = finnhub.Client(api_key=self.api_key)
        self.cache = None
        self.last_cache_time = 0
        self.cache_duration = cache_duration

    def get_economic_calendar(self):
        """
        Gets economic calendar data, using a cache to avoid excessive API calls.
        """
        current_time = time.time()
        if self.cache is not None and (current_time - self.last_cache_time) < self.cache_duration:
            print("Returning cached economic calendar data.")
            return self.cache

        print("Fetching fresh economic calendar data from Finnhub.")
        try:
            # The economic_calendar method has been deprecated.
            # We will use the generic _get method to access the endpoint.
            economic_calendar = self.client._get("/calendar/economic")
            self.cache = pd.DataFrame(economic_calendar.get('economicCalendar', []))
            self.last_cache_time = current_time
            return self.cache
        except finnhub.FinnhubAPIException as e:
            if e.status_code == 403:
                print("Finnhub API key does not have access to the economic calendar. Continuing without economic data.")
                return pd.DataFrame()
            else:
                print(f"Error fetching economic calendar from Finnhub: {e}")
                return pd.DataFrame()
        except Exception as e:
            print(f"An unexpected error occurred while fetching from Finnhub: {e}")
            return pd.DataFrame()

import requests
import datetime
import json
import os
from pathlib import Path

class EconomicCalendarCollector:
    def __init__(self, cache_dir="cache"):
        self.url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "economic_calendar_cache.json"
        self.metadata_file = self.cache_dir / "economic_calendar_metadata.json"

    def _get_current_week_key(self):
        """Returns a unique key for the current week (year-week format)."""
        now = datetime.datetime.now()
        year, week, _ = now.isocalendar()
        return f"{year}-W{week:02d}"

    def _is_cache_valid(self):
        """Checks if the cached data is from the current week."""
        if not self.metadata_file.exists():
            return False
        
        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            cached_week = metadata.get('week_key')
            current_week = self._get_current_week_key()
            
            return cached_week == current_week
        except (json.JSONDecodeError, KeyError, IOError):
            return False

    def _save_cache(self, data):
        """Saves economic calendar data and metadata to cache files."""
        try:
            # Save the data
            with open(self.cache_file, 'w') as f:
                data.to_json(f, orient='records', date_format='iso')
            
            # Save metadata
            metadata = {
                'week_key': self._get_current_week_key(),
                'cached_at': datetime.datetime.now().isoformat(),
                'record_count': len(data)
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Economic calendar cached for week {metadata['week_key']} with {len(data)} records")
        except Exception as e:
            print(f"Error saving economic calendar cache: {e}")

    def _load_cache(self):
        """Loads cached economic calendar data."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                return pd.DataFrame(data)
            return pd.DataFrame()
        except Exception as e:
            print(f"Error loading economic calendar cache: {e}")
            return pd.DataFrame()

    def _cleanup_old_cache(self):
        """Removes old cache files when starting a new week."""
        try:
            if self.cache_file.exists():
                os.remove(self.cache_file)
                print("Removed old economic calendar cache")
            
            if self.metadata_file.exists():
                os.remove(self.metadata_file)
                print("Removed old economic calendar metadata")
        except Exception as e:
            print(f"Error cleaning up old cache: {e}")

    def get_economic_calendar(self):
        """Fetches economic calendar events, using weekly caching to minimize API calls."""
        # Check if we have valid cache for current week
        if self._is_cache_valid():
            print("Returning cached economic calendar data for current week")
            return self._load_cache()
        
        # Cache is either missing or from previous week - clean up old data
        print("Cache invalid or missing - fetching fresh economic calendar data")
        self._cleanup_old_cache()
        
        # Fetch fresh data
        try:
            response = requests.get(self.url, timeout=10)
            response.raise_for_status()
            data = pd.DataFrame(response.json())
            
            # Cache the fresh data for the week
            if not data.empty:
                self._save_cache(data)
            
            return data
        except requests.exceptions.RequestException as e:
            print(f"Error fetching economic calendar: {e}")
            # Try to return any existing cache as fallback
            return self._load_cache()
        except ValueError as e:
            print(f"Error parsing JSON response: {e}")
            return self._load_cache()

    def get_high_impact_events(self, hours_ahead=24):
        """Returns high-impact economic events within specified hours ahead."""
        calendar_data = self.get_economic_calendar()
        
        if calendar_data.empty:
            return pd.DataFrame()
        
        try:
            # Convert date strings to datetime if needed
            if 'date' in calendar_data.columns:
                calendar_data['datetime'] = pd.to_datetime(calendar_data['date'], errors='coerce')
            elif 'time' in calendar_data.columns:
                calendar_data['datetime'] = pd.to_datetime(calendar_data['time'], errors='coerce')
            else:
                print("No date/time column found in economic calendar data")
                return pd.DataFrame()
            
            # Filter for high impact events
            high_impact = calendar_data[calendar_data.get('impact', '').str.upper() == 'HIGH']
            
            # Filter for events within the specified time window
            now = datetime.datetime.now()
            future_cutoff = now + datetime.timedelta(hours=hours_ahead)
            
            upcoming_events = high_impact[
                (high_impact['datetime'] >= now) & 
                (high_impact['datetime'] <= future_cutoff)
            ]
            
            return upcoming_events.sort_values('datetime') if not upcoming_events.empty else pd.DataFrame()
            
        except Exception as e:
            print(f"Error filtering high impact events: {e}")
            return pd.DataFrame()
