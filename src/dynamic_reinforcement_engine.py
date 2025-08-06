import numpy as np
import pandas as pd
from datetime import datetime, timedelta
try:
    import MetaTrader5 as mt5
except ImportError:
    pass

class DynamicReinforcementEngine:
    """
    Event-driven reinforcement engine that responds to market changes in real-time
    instead of waiting for fixed cycle periods
    """
    
    def __init__(self, config):
        self.config = config
        
        # Helper function to parse config values with comments
        def parse_config_value(value, default):
            if isinstance(value, str):
                # Remove inline comments and extra spaces
                clean_value = value.split('#')[0].split('(')[0].strip()
                try:
                    return float(clean_value) if '.' in clean_value else int(clean_value)
                except ValueError:
                    return default
            return value
        
        # Helper function to parse boolean config values with comments
        def parse_config_bool(value, default):
            if isinstance(value, str):
                clean_value = value.split('#')[0].split('(')[0].strip().lower()
                return clean_value == 'true'
            return default
        
        # Load configuration parameters
        self.enabled = parse_config_bool(config['trading'].get('dynamic_reinforcement_enabled', 'true'), True)
        self.check_frequency_minutes = parse_config_value(config['trading'].get('reinforcement_check_frequency_minutes', '5'), 5)
        self.price_movement_trigger_pips = parse_config_value(config['trading'].get('price_movement_trigger_pips', '20'), 20.0)
        self.volatility_spike_multiplier = parse_config_value(config['trading'].get('volatility_spike_multiplier', '2.0'), 2.0)
        self.max_reinforcements_per_position = parse_config_value(config['trading'].get('max_reinforcements_per_position', '3'), 3)
        self.cooling_period_minutes = parse_config_value(config['trading'].get('reinforcement_cooling_period_minutes', '15'), 15)
        self.adaptive_lot_sizing = parse_config_bool(config['trading'].get('adaptive_lot_sizing', 'true'), True)
        self.session_based_reinforcement = parse_config_bool(config['trading'].get('session_based_reinforcement', 'true'), True)
        self.rapid_loss_threshold_pct = parse_config_value(config['trading'].get('rapid_loss_threshold_pct', '2.0'), 2.0)
        self.momentum_reinforcement_enabled = parse_config_bool(config['trading'].get('momentum_reinforcement_enabled', 'true'), True)
        
        # Tracking variables
        self.position_reinforcement_history = {}
        self.last_reinforcement_check = None
        self.market_volatility_state = 'normal'  # normal, high, extreme
        self.session_multipliers = {
            'asian': 0.7,    # Lower reinforcement during Asian session
            'london': 1.0,   # Normal reinforcement during London session
            'ny': 1.2,       # Higher reinforcement during NY session
            'overlap': 1.5   # Highest during session overlaps
        }
    
    def should_check_reinforcement(self, current_time):
        """Determine if reinforcement check should be performed"""
        if not self.enabled:
            return False
            
        if self.last_reinforcement_check is None:
            return True
            
        time_since_last_check = (current_time - self.last_reinforcement_check).total_seconds() / 60
        
        # Adjust frequency based on market volatility
        required_frequency = self.check_frequency_minutes
        if self.market_volatility_state == 'high':
            required_frequency = self.check_frequency_minutes / self.volatility_spike_multiplier
        elif self.market_volatility_state == 'extreme':
            required_frequency = self.check_frequency_minutes / (self.volatility_spike_multiplier * 2)
        
        return time_since_last_check >= required_frequency
    
    def detect_market_events(self, positions, current_market_data, ufo_data=None):
        """Detect market events that should trigger reinforcement checks"""
        events = []
        
        if not positions or not current_market_data:
            return events
            
        for position in positions:
            symbol = position.get('symbol', '')
            if symbol not in current_market_data:
                continue
                
            # Price movement event detection
            entry_price = position.get('entry_price', position.get('price_open', 0))
            current_price = current_market_data[symbol].get('close', 0)
            
            if entry_price > 0:
                pip_multiplier = 10000 if 'JPY' not in symbol else 100
                price_movement_pips = abs(current_price - entry_price) * pip_multiplier
                
                if price_movement_pips >= self.price_movement_trigger_pips:
                    events.append({
                        'type': 'price_movement',
                        'symbol': symbol,
                        'movement_pips': price_movement_pips,
                        'position': position,
                        'priority': 'high' if price_movement_pips > self.price_movement_trigger_pips * 2 else 'medium'
                    })
            
            # Rapid loss event detection
            current_pnl_pct = 0
            if hasattr(position, 'get'):
                current_pnl = position.get('pnl', position.get('profit', 0))
                initial_balance = 10000  # Could be made configurable
                current_pnl_pct = abs(current_pnl) / initial_balance * 100
                
                if current_pnl < 0 and current_pnl_pct >= self.rapid_loss_threshold_pct:
                    events.append({
                        'type': 'rapid_loss',
                        'symbol': symbol,
                        'loss_pct': current_pnl_pct,
                        'position': position,
                        'priority': 'critical'
                    })
        
        # UFO signal change detection
        if ufo_data and hasattr(self, 'previous_ufo_data'):
            ufo_changes = self._detect_ufo_signal_changes(ufo_data, self.previous_ufo_data)
            events.extend(ufo_changes)
        
        return events
    
    def _detect_ufo_signal_changes(self, current_ufo, previous_ufo):
        """Detect significant changes in UFO signals"""
        changes = []
        
        try:
            current_raw = current_ufo.get('raw_data', current_ufo) if isinstance(current_ufo, dict) else current_ufo
            previous_raw = previous_ufo.get('raw_data', previous_ufo) if isinstance(previous_ufo, dict) else previous_ufo
            
            if not current_raw or not previous_raw:
                return changes
            
            # Check M5 timeframe for quick signal changes
            timeframe = mt5.TIMEFRAME_M5 if 'mt5' in globals() else list(current_raw.keys())[0]
            
            if timeframe in current_raw and timeframe in previous_raw:
                current_strengths = current_raw[timeframe]
                previous_strengths = previous_raw[timeframe]
                
                if hasattr(current_strengths, 'columns') and hasattr(previous_strengths, 'columns'):
                    for currency in current_strengths.columns:
                        if currency in previous_strengths.columns:
                            current_val = current_strengths[currency].iloc[-1]
                            previous_val = previous_strengths[currency].iloc[-3:].mean()  # 3-period average
                            
                            change_magnitude = abs(current_val - previous_val)
                            if change_magnitude > 1.5:  # Significant UFO change
                                changes.append({
                                    'type': 'ufo_signal_change',
                                    'currency': currency,
                                    'change_magnitude': change_magnitude,
                                    'direction': 'strengthening' if current_val > previous_val else 'weakening',
                                    'priority': 'high' if change_magnitude > 2.5 else 'medium'
                                })
        except Exception as e:
            pass  # Silently handle UFO data format issues
            
        return changes
    
    def calculate_dynamic_reinforcement(self, position, market_event, current_market_data, ufo_data=None):
        """Calculate reinforcement parameters based on market conditions"""
        symbol = position.get('symbol', '')
        position_id = position.get('ticket', position.get('id', str(hash(str(position)))))
        
        # Check reinforcement limits
        reinforcement_count = self.position_reinforcement_history.get(position_id, {}).get('count', 0)
        if reinforcement_count >= self.max_reinforcements_per_position:
            return None, "Maximum reinforcements reached"
        
        # Check cooling period
        last_reinforcement = self.position_reinforcement_history.get(position_id, {}).get('last_time')
        if last_reinforcement:
            minutes_since_last = (datetime.now() - last_reinforcement).total_seconds() / 60
            if minutes_since_last < self.cooling_period_minutes:
                return None, f"Cooling period active ({minutes_since_last:.1f} min remaining)"
        
        # Base reinforcement calculation
        original_volume = position.get('volume', position.get('lots', 0.1))
        base_reinforcement_size = original_volume * 0.3  # 30% of original position
        
        # Adjust based on event type and priority
        event_multiplier = {
            'price_movement': {'high': 1.2, 'medium': 1.0, 'low': 0.8},
            'rapid_loss': {'critical': 1.5, 'high': 1.3, 'medium': 1.1},
            'ufo_signal_change': {'high': 1.4, 'medium': 1.1, 'low': 0.9}
        }.get(market_event['type'], {}).get(market_event.get('priority', 'medium'), 1.0)
        
        # Session-based adjustment
        session_multiplier = 1.0
        if self.session_based_reinforcement:
            session_multiplier = self._get_session_multiplier()
        
        # Adaptive lot sizing based on volatility
        volatility_multiplier = 1.0
        if self.adaptive_lot_sizing:
            volatility_multiplier = self._calculate_volatility_adjustment(symbol, current_market_data)
        
        # Final reinforcement size
        final_size = base_reinforcement_size * event_multiplier * session_multiplier * volatility_multiplier
        final_size = max(0.01, min(final_size, original_volume))  # Bounds checking
        
        reinforcement_plan = {
            'type': f'dynamic_{market_event["type"]}',
            'additional_lots': final_size,
            'reason': f'Dynamic reinforcement: {market_event["type"]} ({market_event.get("priority", "medium")} priority)',
            'event_details': market_event,
            'multipliers': {
                'event': event_multiplier,
                'session': session_multiplier,
                'volatility': volatility_multiplier
            }
        }
        
        return reinforcement_plan, "Reinforcement calculated"
    
    def _get_session_multiplier(self):
        """Get current session multiplier"""
        try:
            from datetime import datetime
            import pytz
            
            utc_now = datetime.utcnow().replace(tzinfo=pytz.UTC)
            london_time = utc_now.astimezone(pytz.timezone('Europe/London'))
            hour = london_time.hour
            
            # Determine current session
            if 23 <= hour or hour < 8:  # Asian session
                return self.session_multipliers['asian']
            elif 8 <= hour < 13:  # London session
                return self.session_multipliers['london']  
            elif 13 <= hour < 16:  # London-NY overlap
                return self.session_multipliers['overlap']
            elif 16 <= hour < 22:  # NY session
                return self.session_multipliers['ny']
            else:
                return 1.0
                
        except Exception:
            return 1.0
    
    def _calculate_volatility_adjustment(self, symbol, current_market_data):
        """Calculate volatility-based adjustment"""
        try:
            if symbol in current_market_data:
                spread = current_market_data[symbol].get('spread', 0.0001)
                
                # Use spread as volatility proxy (simplified)
                normal_spread = 0.0001  # 1 pip for major pairs
                volatility_ratio = spread / normal_spread
                
                if volatility_ratio > 3.0:  # High volatility
                    self.market_volatility_state = 'extreme'
                    return 0.7  # Reduce reinforcement size
                elif volatility_ratio > 2.0:
                    self.market_volatility_state = 'high'
                    return 0.85
                else:
                    self.market_volatility_state = 'normal'
                    return 1.0
                    
        except Exception:
            pass
            
        return 1.0
    
    def record_reinforcement(self, position, reinforcement_plan):
        """Record reinforcement execution for tracking"""
        position_id = position.get('ticket', position.get('id', str(hash(str(position)))))
        
        if position_id not in self.position_reinforcement_history:
            self.position_reinforcement_history[position_id] = {
                'count': 0,
                'total_reinforcement_size': 0.0,
                'history': []
            }
        
        record = self.position_reinforcement_history[position_id]
        record['count'] += 1
        record['total_reinforcement_size'] += reinforcement_plan.get('additional_lots', 0)
        record['last_time'] = datetime.now()
        record['history'].append({
            'time': datetime.now(),
            'type': reinforcement_plan.get('type', 'unknown'),
            'size': reinforcement_plan.get('additional_lots', 0),
            'reason': reinforcement_plan.get('reason', 'N/A')
        })
        
        self.last_reinforcement_check = datetime.now()
    
    def get_reinforcement_status(self, position):
        """Get reinforcement status for a position"""
        position_id = position.get('ticket', position.get('id', str(hash(str(position)))))
        
        if position_id not in self.position_reinforcement_history:
            return {
                'reinforcements_used': 0,
                'reinforcements_remaining': self.max_reinforcements_per_position,
                'cooling_period_active': False,
                'total_reinforcement_size': 0.0
            }
        
        record = self.position_reinforcement_history[position_id]
        
        cooling_active = False
        if record.get('last_time'):
            minutes_since_last = (datetime.now() - record['last_time']).total_seconds() / 60
            cooling_active = minutes_since_last < self.cooling_period_minutes
        
        return {
            'reinforcements_used': record['count'],
            'reinforcements_remaining': max(0, self.max_reinforcements_per_position - record['count']),
            'cooling_period_active': cooling_active,
            'total_reinforcement_size': record['total_reinforcement_size'],
            'last_reinforcement': record.get('last_time')
        }
