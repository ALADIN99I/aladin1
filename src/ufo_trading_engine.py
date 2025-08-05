import pandas as pd
import numpy as np
from datetime import datetime, time
import pytz
try:
    import MetaTrader5 as mt5
except ImportError:
    from . import mock_metatrader5 as mt5

class UFOTradingEngine:
    """
    Core UFO Trading Engine implementing the real UFO methodology:
    - Analysis-based exits (not fixed TP/SL)
    - Session-based timing management
    - Position reinforcement strategy
    - Portfolio-level risk management
    """
    
    def __init__(self, config):
        self.config = config
        # Read from trading section, default to -5.0 if not found
        portfolio_stop_raw = config['trading'].get('portfolio_equity_stop', '-5.0')
        # Handle config values with inline comments like '-5.0 (-3.0)'
        portfolio_stop_clean = portfolio_stop_raw.split(' ')[0].split('(')[0].strip()
        self.portfolio_equity_stop = float(portfolio_stop_clean)  # -5% portfolio stop
        self.session_timezone = pytz.timezone('Europe/London')  # Trading sessions reference
        
        # Intelligent Diversification Parameters (maintains quality-based decisions)
        # Parse config values properly (strip comments and whitespace)
        def parse_config_int(key, default):
            value = config['trading'].get(key, str(default))
            # Remove inline comments and whitespace
            clean_value = value.split('#')[0].strip()
            return int(clean_value)
        
        def parse_config_float(key, default):
            value = config['trading'].get(key, str(default))
            # Remove inline comments and whitespace
            clean_value = value.split('#')[0].strip()
            return float(clean_value)
        
        def parse_config_str(key, default):
            value = config['trading'].get(key, default)
            # Remove inline comments and whitespace
            clean_value = value.split('#')[0].strip()
            return clean_value
        
        self.max_concurrent_positions = parse_config_int('max_concurrent_positions', 9)
        self.target_positions_when_available = parse_config_int('target_positions_when_available', 4)
        self.min_positions_for_session = parse_config_int('min_positions_for_session', 2)
        self.max_correlation_threshold = parse_config_float('max_correlation_threshold', 0.75)
        self.diversification_preference = parse_config_str('diversification_preference', 'balanced')
        
        # Configuration loaded successfully - diversification parameters active
        
        self.current_positions = {}
        self.portfolio_synthetic_value = 0.0
        self.daily_start_balance = 0.0
        
        # Enhanced UFO Methodology Parameters
        self.uncertainty_threshold = 0.4  # Threshold for uncertain market conditions
        self.mean_reversion_sensitivity = 2.0  # Z-score threshold for mean reversion trades
        self.coherence_requirement = 0.6  # Minimum coherence for high-confidence trades
        self.volatility_adjustment_factor = 0.5  # Position scaling in volatile conditions
        
    def should_trade_now(self):
        """
        Determines if trading should occur based on session timing
        Avoids major news and focuses on session-based opportunities
        """
        now_utc = datetime.utcnow().replace(tzinfo=pytz.UTC)
        london_time = now_utc.astimezone(self.session_timezone)
        current_time = london_time.time()
        current_weekday = london_time.weekday()  # 0=Monday, 6=Sunday
        
        # Define trading sessions
        asian_session = time(23, 0) <= current_time or current_time <= time(8, 0)
        london_session = time(8, 0) <= current_time <= time(16, 0)
        ny_session = time(13, 0) <= current_time <= time(22, 0)
        
        # Avoid weekends
        if current_weekday >= 5:  # Saturday or Sunday
            return False, "Weekend - No trading"
            
        # Prefer active sessions
        if london_session or ny_session:
            return True, f"Active session - London: {london_session}, NY: {ny_session}"
        elif asian_session:
            return True, "Asian session - Limited trading"
        else:
            return False, "Between sessions"
    
    def is_active_session(self):
        """
        Returns boolean indicating if currently in an active trading session
        Wrapper for should_trade_now to provide simple boolean interface
        """
        should_trade, _ = self.should_trade_now()
        return should_trade
    
    def should_close_for_session_end(self):
        """
        Determines if positions should be closed due to session ending
        Implements "not to go to bed with positions open" rule
        """
        now_utc = datetime.utcnow().replace(tzinfo=pytz.UTC)
        london_time = now_utc.astimezone(self.session_timezone)
        current_time = london_time.time()
        current_weekday = london_time.weekday()
        
        # Close before weekend
        if current_weekday == 4 and current_time >= time(21, 0):  # Friday 9 PM
            return True, "Weekend closure - Friday evening"
            
        # Close at 8 PM GMT (20:00) - UFO methodology end of analysis period
        if current_time >= time(20, 0):
            return True, "End of UFO analysis period (8 PM GMT)"
            
        # Close during major news times (can be expanded)
        major_news_times = [
            (time(8, 30), time(9, 30)),   # London open + ECB times
            (time(13, 30), time(14, 30)), # NY open + Fed times
        ]
        
        for start_time, end_time in major_news_times:
            if start_time <= current_time <= end_time:
                return True, f"Major news period: {current_time}"
                
        return False, "Normal trading hours"
    
    def analyze_ufo_exit_signals(self, current_ufo_data, previous_ufo_data):
        """
        Analyzes UFO data for exit signals based on currency strength changes
        Core UFO methodology: exit when underlying analysis changes
        """
        exit_signals = []
        
        if previous_ufo_data is None:
            return exit_signals
            
        # Check for currency strength reversals across timeframes
        for timeframe in current_ufo_data.keys():
            if timeframe not in previous_ufo_data:
                continue
                
            current_strengths = current_ufo_data[timeframe]
            previous_strengths = previous_ufo_data[timeframe]
            
            # Detect significant strength changes
            for currency in current_strengths.columns:
                if currency not in previous_strengths.columns:
                    continue
                    
                current_strength = current_strengths[currency].iloc[-1]
                previous_strength = previous_strengths[currency].iloc[-5:]  # Last 5 bars average
                avg_previous = previous_strength.mean()
                
                # Signal strength reversal (threshold can be tuned)
                if abs(current_strength - avg_previous) > 2.0:  # Significant change
                    direction_change = "strengthening" if current_strength > avg_previous else "weakening"
                    exit_signals.append({
                        'currency': currency,
                        'timeframe': timeframe,
                        'change': current_strength - avg_previous,
                        'direction': direction_change,
                        'reason': f"{currency} {direction_change} on {timeframe}"
                    })
        
        return exit_signals
    
    def check_multi_timeframe_coherence(self, ufo_data):
        """
        Checks if currency strength is consistent across timeframes
        Flags positions where coherence is lost
        """
        coherence_issues = []
        
        if len(ufo_data) < 2:
            return coherence_issues
            
        timeframes = list(ufo_data.keys())
        currencies = list(ufo_data[timeframes[0]].columns)
        
        for currency in currencies:
            strengths_by_tf = {}
            
            # Get latest strength for each timeframe
            for tf in timeframes:
                if currency in ufo_data[tf].columns:
                    strengths_by_tf[tf] = ufo_data[tf][currency].iloc[-1]
            
            if len(strengths_by_tf) < 2:
                continue
                
            # Check if all timeframes agree on direction (all positive or all negative)
            values = list(strengths_by_tf.values())
            all_positive = all(v > 0 for v in values)
            all_negative = all(v < 0 for v in values)
            
            if not (all_positive or all_negative):
                # Timeframes disagree - coherence issue
                coherence_issues.append({
                    'currency': currency,
                    'strengths': strengths_by_tf,
                    'issue': 'Timeframe divergence',
                    'recommendation': 'Consider closing positions'
                })
        
        return coherence_issues
    
    def calculate_portfolio_synthetic_value(self, positions, current_prices):
        """
        Calculates the synthetic portfolio value (A+B+C+D)
        Core UFO principle: portfolio should trend toward positive
        """
        total_value = 0.0
        
        for position in positions:
            symbol = position.get('symbol', '')
            direction = position.get('direction', 'BUY')
            lots = position.get('lots', 0.0)
            entry_price = position.get('entry_price', 0.0)
            
            if symbol in current_prices:
                current_price = current_prices[symbol]
                
                if direction == 'BUY':
                    pnl = (current_price - entry_price) * lots * 10  # Simplified PnL calc
                else:
                    pnl = (entry_price - current_price) * lots * 10
                    
                total_value += pnl
        
        return total_value
    
    def detect_early_late_entry(self, position, current_market_data):
        """
        Detects if a position was entered too early or too late based on immediate price movement
        UFO methodology: identify timing errors that cause immediate drawdown
        """
        try:
            symbol = position.get('symbol', '')
            entry_price = position.get('open_price', 0.0)
            direction = position.get('type', 0)  # 0=BUY, 1=SELL
            entry_time = position.get('time', 0)
            current_profit = position.get('profit', 0.0)
            
            # Get current price from market data
            if symbol not in current_market_data:
                return False, "No market data available", 0.0
                
            current_price = current_market_data[symbol]['close']
            
            # Calculate immediate drawdown percentage
            if direction == 0:  # BUY position
                price_movement = (current_price - entry_price) / entry_price * 100
                immediate_drawdown = -price_movement if price_movement < 0 else 0
            else:  # SELL position
                price_movement = (entry_price - current_price) / entry_price * 100
                immediate_drawdown = -price_movement if price_movement < 0 else 0
            
            # Consider entry timing error if:
            # 1. Position shows immediate drawdown > 0.5%
            # 2. Position is relatively new (< 30 minutes)
            import time
            position_age_minutes = (time.time() - entry_time) / 60
            
            is_early_late = (immediate_drawdown > 0.5 and position_age_minutes < 30) or current_profit < -0.8
            
            timing_error = "none"
            if is_early_late:
                if immediate_drawdown > 1.0:
                    timing_error = "too_early" if position_age_minutes < 15 else "too_late"
                else:
                    timing_error = "minor_timing_issue"
            
            return is_early_late, timing_error, immediate_drawdown
            
        except Exception as e:
            return False, f"Error detecting timing: {e}", 0.0
    
    def should_reinforce_position(self, position, current_analysis, current_market_data):
        """
        Determines if a position should be reinforced during retracements
        UFO methodology: add lots if main analysis still holds and compensate for timing errors
        """
        symbol = position.get('symbol', '')
        direction = position.get('type', 0)  # 0=BUY, 1=SELL
        current_profit = position.get('profit', 0.0)
        
        # Check if main currency analysis still holds
        if current_analysis is None:
            return False, "Missing analysis data", {}
            
        # Extract currency pair components (remove suffix)
        clean_symbol = symbol.replace('-ECN', '').replace('-', '')
        if len(clean_symbol) >= 6:
            base_currency = clean_symbol[:3]
            quote_currency = clean_symbol[3:6]
        else:
            return False, "Invalid symbol format", {}
        
        # Get current strength for both currencies
        base_strength = self._get_currency_strength(base_currency, current_analysis)
        quote_strength = self._get_currency_strength(quote_currency, current_analysis)
        
        # Check if original thesis still holds
        if direction == 0:  # BUY position
            original_thesis = base_strength > quote_strength
        else:  # SELL position
            original_thesis = quote_strength > base_strength
        
        # Detect if this is an early/late entry issue
        is_timing_error, timing_type, drawdown_pct = self.detect_early_late_entry(position, current_market_data)
        
        # UFO Logic: Reinforce if analysis holds but position is losing due to timing
        should_reinforce = False
        reinforcement_plan = {}
        
        if original_thesis and current_profit < 0:
            if is_timing_error:
                # Compensate for timing error - UFO methodology
                original_lots = position.get('volume', 0.0)
                
                if timing_type == "too_early":
                    # Add more lots at current better price
                    reinforcement_plan = {
                        'type': 'compensate_early_entry',
                        'additional_lots': min(original_lots * 0.5, original_lots),
                        'reason': f'Compensating early entry - analysis still valid',
                        'timing_error': timing_type,
                        'drawdown': drawdown_pct
                    }
                    should_reinforce = True
                    
                elif timing_type == "too_late":
                    # Add smaller lot to average the position
                    reinforcement_plan = {
                        'type': 'compensate_late_entry', 
                        'additional_lots': min(original_lots * 0.3, original_lots * 0.5),
                        'reason': f'Compensating late entry - averaging position',
                        'timing_error': timing_type,
                        'drawdown': drawdown_pct
                    }
                    should_reinforce = True
                    
            elif current_profit < -1.5:  # Significant loss but analysis holds
                # Standard reinforcement for valid analysis
                original_lots = position.get('volume', 0.0)
                reinforcement_plan = {
                    'type': 'standard_reinforcement',
                    'additional_lots': min(original_lots * 0.4, original_lots),
                    'reason': f'Analysis confirms direction despite drawdown',
                    'drawdown': abs(current_profit)
                }
                should_reinforce = True
        
        if should_reinforce:
            return True, f"Reinforce {symbol}: {reinforcement_plan['reason']}", reinforcement_plan
        elif not original_thesis:
            return False, f"Analysis changed: {base_currency} vs {quote_currency} - close position", {}
        else:
            return False, f"Hold position: analysis valid, no reinforcement needed", {}
    
    def execute_compensation_trade(self, original_position, reinforcement_plan, trade_executor):
        """
        Executes the compensation trade to recover from timing errors
        UFO methodology: add strategic trades to turn losses into profits
        """
        try:
            symbol = original_position.get('symbol', '')
            position_type = original_position.get('type', 0)  # 0=BUY, 1=SELL
            additional_lots = reinforcement_plan.get('additional_lots', 0.0)
            
            if additional_lots <= 0:
                return False, "Invalid lot size for compensation"
            
            # Convert MT5 position type to our trade type
            if position_type == 0:  # BUY position
                trade_type = mt5.ORDER_TYPE_BUY
                direction_text = "BUY"
            else:  # SELL position  
                trade_type = mt5.ORDER_TYPE_SELL
                direction_text = "SELL"
            
            compensation_comment = f"UFO Compensation: {reinforcement_plan.get('type', 'unknown')}"
            
            print(f"🔧 Executing UFO compensation: {direction_text} {additional_lots} lots of {symbol}")
            print(f"📊 Reason: {reinforcement_plan.get('reason', 'N/A')}")
            
            # Execute the compensation trade
            success = trade_executor.execute_ufo_trade(
                symbol=symbol,
                trade_type=trade_type,
                volume=additional_lots,
                comment=compensation_comment
            )
            
            if success:
                print(f"✅ UFO Compensation executed: {direction_text} {additional_lots} lots of {symbol}")
                return True, f"Compensation successful: {reinforcement_plan['reason']}"
            else:
                print(f"❌ UFO Compensation failed: {symbol} {direction_text}")
                return False, "Trade execution failed"
                
        except Exception as e:
            error_msg = f"Error executing compensation trade: {e}"
            print(f"❌ {error_msg}")
            return False, error_msg
    
    def _get_currency_strength(self, currency, ufo_data, primary_tf=mt5.TIMEFRAME_M5):
        """Helper method to get currency strength from UFO analysis"""
        # Handle enhanced UFO data format
        raw_ufo_data = ufo_data.get('raw_data', ufo_data) if isinstance(ufo_data, dict) and 'raw_data' in ufo_data else ufo_data
        
        # Use the primary timeframe (M5 as default)
        if primary_tf in raw_ufo_data:
            strength_data = raw_ufo_data[primary_tf]
            
            # Handle both DataFrame and dict formats
            if hasattr(strength_data, 'columns'):
                # DataFrame format
                if currency in strength_data.columns:
                    return strength_data[currency].iloc[-1]
            else:
                # Dict format
                if currency in strength_data:
                    return strength_data[currency][-1]
        
        # Fallback to any available timeframe
        for tf, data in raw_ufo_data.items():
            if hasattr(data, 'columns'):
                # DataFrame format
                if currency in data.columns:
                    return data[currency].iloc[-1]
            else:
                # Dict format
                if currency in data:
                    return data[currency][-1]
                
        return 0.0  # Neutral if not found
    
    def check_portfolio_equity_stop(self, account_balance, current_equity):
        """
        Checks if portfolio-level stop loss is breached
        UFO methodology: -2% to -3% of total equity, not individual stops
        """
        if account_balance <= 0:
            return False, "Invalid account balance"
            
        current_drawdown = ((current_equity - account_balance) / account_balance) * 100
        
        if current_drawdown <= self.portfolio_equity_stop:
            return True, f"Portfolio stop breached: {current_drawdown:.2f}% (limit: {self.portfolio_equity_stop}%)"
        
        return False, f"Portfolio healthy: {current_drawdown:.2f}% drawdown"
    
    def generate_reinforcement_plan(self, existing_positions, ufo_analysis, account_balance):
        """
        Generates a plan for reinforcing existing positions
        UFO methodology: compensate losses with strategic additions
        """
        reinforcement_plan = []
        
        for position in existing_positions:
            symbol = position.get('symbol', '')
            direction = position.get('direction', '')
            current_pnl = position.get('profit', 0.0)
            
            # Only reinforce losing positions if analysis still supports them
            if current_pnl < 0:
                should_reinforce, reason = self.should_reinforce_position(
                    symbol, direction, ufo_analysis, position.get('entry_analysis')
                )
                
                if should_reinforce:
                    # Calculate reinforcement size (conservative)
                    original_lots = position.get('lots', 0.0)
                    reinforce_lots = min(original_lots * 0.5, original_lots)  # Max 50% of original
                    
                    reinforcement_plan.append({
                        'action': 'reinforce',
                        'symbol': symbol,
                        'direction': direction,
                        'lots': reinforce_lots,
                        'reason': reason,
                        'original_position': position
                    })
        
        return reinforcement_plan
    
    def should_take_profit(self, position, portfolio_value, session_status):
        """
        Determines when to take profits based on UFO methodology
        Not fixed TP levels, but "enough profit" + session timing
        """
        current_pnl = position.get('profit', 0.0)
        lots = position.get('lots', 0.0)
        
        # Calculate profit as percentage of account risk
        profit_percentage = (current_pnl / (lots * 1000)) * 100  # Simplified calc
        
        # Take profit conditions
        enough_profit = profit_percentage > 1.5  # 1.5% profit threshold
        end_of_session = session_status[0] if isinstance(session_status, tuple) else False
        portfolio_positive = portfolio_value > 0
        
        if enough_profit and (end_of_session or portfolio_positive):
            return True, f"Taking profit: {profit_percentage:.2f}% - Session end: {end_of_session}"
        
        return False, f"Holding position: {profit_percentage:.2f}% profit"
    
    def should_open_new_trades(self, current_positions=None, portfolio_status=None, ufo_data=None):
        """
        Determines if new trades can be opened based on UFO methodology with intelligent diversification
        Considers session timing, portfolio status, existing positions, and diversification needs
        """
        # Check if in active trading session
        should_trade, session_reason = self.should_trade_now()
        if not should_trade:
            return False, f"Not trading: {session_reason}"
        
        # Check session end conditions
        should_close, close_reason = self.should_close_for_session_end()
        if should_close:
            return False, f"Session ending: {close_reason}"
        
        # Intelligent diversification logic (maintains quality-based decisions)
        current_position_count = len(current_positions) if current_positions is not None and not (hasattr(current_positions, 'empty') and current_positions.empty) else 0
        
        # Check if we're at maximum diversification limit
        if current_position_count >= self.max_concurrent_positions:
            return False, f"Maximum diversification reached ({current_position_count}/{self.max_concurrent_positions})"
        
        # Encourage diversification if below minimum threshold
        if current_position_count < self.min_positions_for_session:
            return True, f"Building diversification: {current_position_count}/{self.min_positions_for_session} minimum positions"
        
        # Check portfolio health if available
        if portfolio_status:
            account_balance = portfolio_status.get('balance', 0)
            current_equity = portfolio_status.get('equity', 0)
            
            if account_balance > 0:
                stop_breached, stop_reason = self.check_portfolio_equity_stop(account_balance, current_equity)
                if stop_breached:
                    return False, f"Portfolio stop: {stop_reason}"
        
        # Quality-based decision: Allow more positions when quality opportunities exist
        if current_position_count < self.target_positions_when_available:
            return True, f"Quality opportunity available: {current_position_count}/{self.target_positions_when_available} target positions"
        
        # Allow additional positions up to maximum if analysis is strong
        if current_position_count < self.max_concurrent_positions:
            return True, f"Additional diversification possible: {current_position_count}/{self.max_concurrent_positions} max positions"
        
        return True, "Ready for new trades"
