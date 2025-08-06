import time
import pandas as pd
import numpy as np
import re
import json
from datetime import datetime
import pytz
try:
    import MetaTrader5 as mt5
except ImportError:
    from . import mock_metatrader5 as mt5
from .data_collector import MT5DataCollector
from .agents.data_analyst_agent import DataAnalystAgent
from .agents.market_researcher_agent import MarketResearcherAgent
from .agents.trader_agent import TraderAgent
from .agents.risk_manager_agent import RiskManagerAgent
from .agents.fund_manager_agent import FundManagerAgent
from .communication import CommunicationBus
from .ufo_calculator import UfoCalculator
from .llm.llm_client import LLMClient
from .trade_executor import TradeExecutor
from .ufo_trading_engine import UFOTradingEngine
from .dynamic_reinforcement_engine import DynamicReinforcementEngine

class LiveTrader:
    def __init__(self, config):
        self.config = config
        self.previous_ufo_data = None
        self.trades_executed = []
        self.portfolio_value = 10000.0  # Starting balance
        self.initial_balance = 10000.0
        self.realized_pnl = 0.0  # Track cumulative realized P&L from closed trades
        self.simulation_log = []
        self.cycle_count = 0
        self.open_positions = []  # Track simulated positions
        self.closed_trades = []   # Track completed trades
        
        # Continuous monitoring variables
        self.last_position_update = None
        self.position_update_frequency_minutes = 5  # Update positions every 5 minutes
        self.continuous_monitoring_enabled = True
        self.portfolio_history = []  # Track portfolio value over time

        # Fix config parsing issues
        self.fix_config_values()

        # Initialize components
        self.initialize_components()

    def fix_config_values(self):
        """Fix configuration values that have comments or multiple values"""
        # Parse values with inline comments
        def parse_value(value, default):
            if isinstance(value, str):
                # Remove inline comments and extra spaces
                clean_value = value.split('#')[0].split('(')[0].strip()
                try:
                    return float(clean_value) if '.' in clean_value else int(clean_value)
                except ValueError:
                    return default
            return value
        
        # Fix portfolio equity stop parsing
        portfolio_stop = self.config['trading'].get('portfolio_equity_stop', '-5.0')
        self.portfolio_equity_stop = parse_value(portfolio_stop, -5.0)
        
        # Fix other config values
        self.cycle_period_minutes = parse_value(self.config['trading'].get('cycle_period_minutes', '40'), 40)
        self.max_concurrent_positions = parse_value(self.config['trading'].get('max_concurrent_positions', '11'), 11)
        self.target_positions_when_available = parse_value(self.config['trading'].get('target_positions_when_available', '6'), 6)
        self.min_positions_for_session = parse_value(self.config['trading'].get('min_positions_for_session', '5'), 5)

    def initialize_components(self):
        """Initialize all trading components"""
        # Initialize LLM client
        self.llm_client = LLMClient(api_key=self.config['openrouter']['api_key'])

        # Initialize MT5 data collector and establish persistent connection
        self.mt5_collector = MT5DataCollector(
            login=self.config['mt5']['login'],
            password=self.config['mt5']['password'],
            server=self.config['mt5']['server'],
            path=self.config['mt5']['path']
        )

        # Establish persistent MT5 connection for the simulation
        if not self.mt5_collector.connect():
            self.log_event("⚠️ Warning: MT5 connection failed, using mock data")
        else:
            self.log_event("✅ MT5 persistent connection established")

        # Initialize UFO components
        self.ufo_calculator = UfoCalculator(self.config['trading']['currencies'].split(','))
        self.ufo_engine = UFOTradingEngine(self.config)

        # Initialize agents
        symbols_list = self.config['trading']['symbols'].split(',')
        self.data_analyst = DataAnalystAgent("DataAnalyst", self.mt5_collector)
        self.market_researcher = MarketResearcherAgent("MarketResearcher", self.llm_client)
        self.trader = TraderAgent("Trader", self.llm_client, self.mt5_collector, symbols=symbols_list)
        self.risk_manager = RiskManagerAgent("RiskManager", self.llm_client, self.mt5_collector, self.config)
        self.fund_manager = FundManagerAgent("FundManager", self.llm_client)

        # Initialize trade executor
        self.trade_executor = TradeExecutor(self.mt5_collector, self.config)

        # Initialize dynamic reinforcement engine
        self.dynamic_reinforcement_engine = DynamicReinforcementEngine(self.config)
        if self.dynamic_reinforcement_engine.enabled:
            self.log_event("✅ Dynamic Reinforcement Engine enabled")
        else:
            self.log_event("⚠️ Dynamic Reinforcement Engine disabled")

        self.log_event("Full-day simulation components initialized successfully")

    def check_portfolio_equity_stop(self):
        """Check if portfolio-level stop loss is breached (UFO methodology)"""
        if self.initial_balance <= 0:
            return False, "Invalid initial balance"

        current_drawdown = ((self.portfolio_value - self.initial_balance) / self.initial_balance) * 100

        if current_drawdown <= self.portfolio_equity_stop:
            return True, f"Portfolio stop breached: {current_drawdown:.2f}% (limit: {self.portfolio_equity_stop}%)"

        return False, f"Portfolio healthy: {current_drawdown:.2f}% drawdown"

    def get_pip_value_multiplier(self, symbol):
        """Get correct pip value multiplier for different currency pairs"""
        symbol_clean = symbol.replace('-ECN', '').upper()

        # JPY pairs use 1000 multiplier (pip = 0.01)
        jpy_pairs = ['USDJPY', 'EURJPY', 'GBPJPY', 'AUDJPY', 'NZDJPY', 'CHFJPY', 'CADJPY']
        if any(jpy_pair in symbol_clean for jpy_pair in jpy_pairs):
            return 1000

        # Most other forex pairs use 10000 multiplier (pip = 0.0001)
        # Reduced from 100000 to make P&L more realistic
        return 10000

    def update_portfolio_value(self, force_update=False):
        """Update portfolio value based on open positions P&L using real historical prices"""
        # Skip update if too recent (unless forced)
        if not force_update and self.last_position_update:
            time_since_update = (datetime.now() - self.last_position_update).total_seconds() / 60
            if time_since_update < self.position_update_frequency_minutes:
                return

        total_unrealized_pnl = 0.0
        position_updates = []

        # Calculate P&L for all open positions using real market data
        for position in self.open_positions:
            current_price = self.get_historical_price_for_time(position['symbol'], datetime.now())
            if current_price is None:
                # Use entry price if no historical data available
                current_price = position['entry_price']

            # Get correct pip value multiplier
            pip_multiplier = self.get_pip_value_multiplier(position['symbol'])

            # Calculate P&L based on direction with correct pip values
            price_diff = current_price - position['entry_price']
            if position['direction'] == 'SELL':
                price_diff = -price_diff

            # Calculate P&L: price_difference * volume * pip_multiplier
            pnl = price_diff * position['volume'] * pip_multiplier

            # Track price movement since last update
            previous_price = position.get('current_price', position['entry_price'])
            price_change = current_price - previous_price

            position['current_price'] = current_price
            position['pnl'] = pnl
            position['last_update'] = datetime.now()
            total_unrealized_pnl += pnl

            # Track significant price movements
            if abs(price_change) > 0.0005:  # 0.5 pip movement
                position_updates.append({
                    'symbol': position['symbol'],
                    'price_change': price_change,
                    'pnl_change': price_change * position['volume'] * pip_multiplier * (1 if position['direction'] == 'BUY' else -1),
                    'current_pnl': pnl
                })

        # Update portfolio value: initial_balance + realized_pnl + unrealized_pnl
        previous_portfolio_value = self.portfolio_value
        self.portfolio_value = self.initial_balance + self.realized_pnl + total_unrealized_pnl

        # Track portfolio history for continuous monitoring
        portfolio_change = self.portfolio_value - previous_portfolio_value
        self.portfolio_history.append({
            'timestamp': datetime.now(),
            'portfolio_value': self.portfolio_value,
            'unrealized_pnl': total_unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'change': portfolio_change,
            'position_count': len(self.open_positions)
        })

        # Log significant portfolio changes
        if abs(portfolio_change) > 10 and position_updates:
            self.log_event(f"💰 Portfolio update: ${self.portfolio_value:,.2f} (${portfolio_change:+.2f})")
            for update in position_updates[:3]:  # Log top 3 movers
                self.log_event(f"  📊 {update['symbol']}: {update['price_change']:+.5f} → P&L: ${update['current_pnl']:+.2f}")

        # Close positions based on realistic conditions (stop loss, take profit)
        positions_to_close = []
        for i, position in enumerate(self.open_positions):
            # Enhanced position closing logic with proper thresholds
            close_on_profit = position['pnl'] > 75  # Take profit at +$75
            close_on_loss = position['pnl'] < -50   # Stop loss at -$50

            # Time-based exit: close positions older than 4 hours (240 minutes)
            close_on_time = False
            if 'timestamp' in position:
                position_age = (datetime.now() - position['timestamp']).total_seconds() / 60
                close_on_time = position_age > 240  # 4 hours

            # Trailing stop: close if position has moved against us significantly from peak
            close_on_trailing = False
            if 'peak_pnl' not in position:
                position['peak_pnl'] = position['pnl']
            elif position['pnl'] > position['peak_pnl']:
                position['peak_pnl'] = position['pnl']
            elif position['peak_pnl'] > 30 and position['pnl'] < position['peak_pnl'] * 0.7:
                close_on_trailing = True

            if close_on_profit or close_on_loss or close_on_time or close_on_trailing:
                positions_to_close.append(i)
                if close_on_profit:
                    close_reason = "profit target"
                elif close_on_loss:
                    close_reason = "stop loss"
                elif close_on_time:
                    close_reason = "time-based exit"
                else:
                    close_reason = "trailing stop"

                self.log_event(f"🎯 Marking {position['symbol']} for closure: {close_reason} (P&L: ${position['pnl']:.2f})")

        # Close positions (in reverse order to maintain indices)
        for i in reversed(positions_to_close):
            closed_position = self.open_positions.pop(i)
            # Add realized P&L to running total
            self.realized_pnl += closed_position['pnl']
            self.closed_trades.append(closed_position)
            self.log_event(f"📉 Position closed: {closed_position['symbol']} P&L: ${closed_position['pnl']:.2f}")

        # Update last position update time
        self.last_position_update = datetime.now()

    def log_event(self, message):
        """Log simulation events with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.simulation_log.append(log_entry)
        print(log_entry)

    def get_historical_price_for_time(self, symbol, target_time):
        """Get real historical price for a specific symbol at a specific time"""
        try:
            # Convert target time to MT5 timestamp
            target_timestamp = int(target_time.timestamp())

            # Get historical data around the target time (M5 bars)
            rates = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_M5, target_timestamp, 1)

            if rates is not None and len(rates) > 0:
                self.log_event(f"✅ Using REAL data from MT5 for {symbol}")
                # Return the close price
                return float(rates[0]['close'])
            else:
                # Fallback: get the most recent data if exact time not available
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 1)
                if rates is not None and len(rates) > 0:
                    self.log_event(f"✅ Using REAL data from MT5 for {symbol} (fallback)")
                    return float(rates[0]['close'])
                else:
                    self.log_event(f"⚠️ Using FALLBACK data for {symbol}")
                    # If no historical data is available, return None
                    return None
        except Exception as e:
            self.log_event(f"⚠️ Error getting historical price for {symbol}: {e}")
            return None

    def calculate_ufo_entry_price(self, symbol, direction, ufo_data, current_time):
        """Calculate optimal entry price based on UFO methodology and currency strength"""
        try:
            # Get base historical price
            base_price = self.get_historical_price_for_time(symbol, current_time)
            if base_price is None:
                # Fallback to standard base prices
                base_prices = {
                    'EURUSD-ECN': 1.0850, 'GBPUSD-ECN': 1.2650, 'USDJPY-ECN': 143.50,
                    'AUDUSD-ECN': 0.6720, 'USDCAD-ECN': 1.3580, 'NZDUSD-ECN': 0.6250,
                    'EURJPY-ECN': 155.20, 'GBPJPY-ECN': 180.50, 'AUDJPY-ECN': 96.30,
                    'USDCHF-ECN': 0.9120, 'EURCHF-ECN': 0.9880, 'GBPCHF-ECN': 1.1520,
                    'AUDCAD-ECN': 0.9080, 'NZDJPY-ECN': 89.60, 'CADCHF-ECN': 0.6730,
                    'CHFJPY-ECN': 157.20, 'AUDNZD-ECN': 1.0750, 'EURGBP-ECN': 0.8590,
                    'GBPCAD-ECN': 1.7180, 'XAUUSD-ECN': 1850.00, 'GBPAUD-ECN': 1.8820
                }
                base_price = base_prices.get(symbol, 1.0850)

            # Calculate UFO adjustment if UFO data is available
            if ufo_data:
                # Extract currencies from symbol
                clean_symbol = symbol.replace('-ECN', '')
                if len(clean_symbol) >= 6:
                    base_currency = clean_symbol[:3]
                    quote_currency = clean_symbol[3:6]

                    # Get currency strengths from M5 timeframe (primary trading timeframe)
                    primary_tf = mt5.TIMEFRAME_M5
                    raw_ufo_data = ufo_data.get('raw_data', ufo_data)

                    if primary_tf in raw_ufo_data:
                        strength_data = raw_ufo_data[primary_tf]
                        
                        base_strength = 0.0
                        quote_strength = 0.0

                        # Handle both DataFrame and dict formats
                        if hasattr(strength_data, 'columns'):
                            # DataFrame format
                            if base_currency in strength_data.columns:
                                base_strength = strength_data[base_currency].iloc[-1]
                            if quote_currency in strength_data.columns:
                                quote_strength = strength_data[quote_currency].iloc[-1]
                        else:
                            # Dict format
                            if base_currency in strength_data:
                                base_strength = strength_data[base_currency][-1]
                            if quote_currency in strength_data:
                                quote_strength = strength_data[quote_currency][-1]
                        
                        # Calculate strength differential
                        strength_diff = base_strength - quote_strength
                        
                        # UFO-based price adjustment
                        # Strong differential = better entry timing = slight price improvement
                        if abs(strength_diff) > 1.0:  # Significant strength difference
                            # Apply small adjustment in favorable direction
                            if direction == 'BUY' and strength_diff > 0:  # Strong base currency
                                # Slightly better entry (lower price for BUY)
                                price_adjustment = -base_price * 0.0002  # 2 pip improvement
                            elif direction == 'SELL' and strength_diff < 0:  # Strong quote currency
                                # Slightly better entry (higher price for SELL)
                                price_adjustment = base_price * 0.0002  # 2 pip improvement
                            else:
                                # Weaker signal, use current historical price as-is
                                price_adjustment = 0.0
                        else:
                            # Normal market entry - use historical price as-is
                            price_adjustment = 0.0

                        optimal_price = base_price + price_adjustment
                        return max(optimal_price, base_price * 0.95)  # Safety minimum

            # Fallback: use historical price as-is without any random adjustments
            return base_price

        except Exception as e:
            self.log_event(f"⚠️ Error calculating UFO entry price for {symbol}: {e}")
            # Ultimate fallback
            return 1.0850 if 'EUR' in symbol else 143.50 if 'JPY' in symbol else 1.2650

    def run_single_cycle(self):
        """Simulate a single 40-minute trading cycle"""
        self.cycle_count += 1

        self.log_event(f"\n" + "="*60)
        self.log_event(f"CYCLE {self.cycle_count} - {datetime.now().strftime('%H:%M')} GMT")
        self.log_event("="*60)

        # Check session status
        if not self.ufo_engine.is_active_session():
            self.log_event(f"⏰ Outside trading hours at {datetime.now().strftime('%H:%M')} GMT - Skipping cycle")
            return True

        # 1. Data Collection
        self.log_event("📊 PHASE 1: Data Collection")
        price_data = self.collect_market_data()

        # 2. UFO Analysis
        self.log_event("🛸 PHASE 2: UFO Analysis")
        ufo_data = self.calculate_ufo_indicators(price_data)

        # 3. Economic Calendar
        self.log_event("📅 PHASE 3: Economic Calendar")
        economic_events = self.get_economic_events()

        # 4. Market Research
        self.log_event("🔍 PHASE 4: Market Research")
        research_result = self.market_researcher.execute(ufo_data, economic_events)

        # 5. UFO Portfolio Management (Priority Check)
        self.log_event("💼 PHASE 5: UFO Portfolio Management")

        # UFO METHODOLOGY: Check portfolio-level stop FIRST
        portfolio_stop_breached, stop_reason = self.check_portfolio_equity_stop()
        if portfolio_stop_breached:
            self.log_event(f"🚨 UFO PORTFOLIO STOP TRIGGERED: {stop_reason}")
            self.log_event("🚨 Closing ALL positions - no individual stops needed!")

            # Close all positions and add to realized P&L
            for position in self.open_positions:
                self.realized_pnl += position.get('pnl', 0.0)
                self.closed_trades.append(position)
                self.log_event(f"📉 Emergency close: {position['symbol']} P&L: ${position.get('pnl', 0.0):.2f}")

            self.open_positions.clear()
            self.portfolio_value = self.initial_balance + self.realized_pnl
            self.log_event("🚨 All positions closed. UFO Portfolio Stop engaged.")
            return True

        # UFO: Check session end timing (with updated simulation time and actual economic events)
        should_close, close_reason = self.ufo_engine.should_close_for_session_end()
        if should_close:
            self.log_event(f"🌅 UFO SESSION END: {close_reason}")
            self.log_event("🌅 Closing all positions for session end")

            for position in self.open_positions:
                self.realized_pnl += position.get('pnl', 0.0)
                self.closed_trades.append(position)
                self.log_event(f"📉 Session close: {position['symbol']} P&L: ${position.get('pnl', 0.0):.2f}")

            self.open_positions.clear()
            self.portfolio_value = self.initial_balance + self.realized_pnl
            return True

        # UFO: Analyze exit signals based on currency strength changes
        if hasattr(self, 'previous_ufo_data') and ufo_data:
            exit_signals = self.analyze_ufo_exit_signals(ufo_data, self.previous_ufo_data)
            if exit_signals:
                self.log_event(f"📈 UFO Exit Signals detected: {len(exit_signals)} currency changes")
                for signal in exit_signals:
                    self.log_event(f"⚠️ {signal['reason']} (change: {signal['change']:.2f})")
                
                # Enhanced auto-close on strong signals
                if len(exit_signals) >= 3:
                    self.log_event("🚨 STRONG EXIT SIGNALS detected: Auto-closing positions")
                    positions_closed = self.close_affected_positions(exit_signals)
                    self.log_event(f"🚨 Auto-closed {positions_closed} positions based on strong exit signals")

        # Store UFO data for next cycle comparison
        if ufo_data:
            self.previous_ufo_data = ufo_data

        current_positions = self.assess_portfolio()

        # 6. Trading Decisions
        self.log_event("🎯 PHASE 6: Trading Decisions")
        trade_decisions = self.generate_trade_decisions(research_result, current_positions)

        # 7. Risk Assessment
        self.log_event("⚖️ PHASE 7: Risk Assessment")
        risk_assessment = self.assess_risk(trade_decisions)

        # 8. Fund Manager Authorization
        self.log_event("💰 PHASE 8: Fund Manager Authorization")
        authorization = self.get_fund_authorization(trade_decisions, risk_assessment)

        # 9. Trade Execution
        self.log_event("⚡ PHASE 9: Trade Execution")
        executed_trades = self.execute_approved_trades(authorization, trade_decisions, current_positions, ufo_data)

        # 10. Cycle Summary
        self.log_event("📋 PHASE 10: Cycle Summary")
        self.generate_cycle_summary(executed_trades)

        return True

    def generate_trade_decisions(self, research_result, current_positions):
        """Generate trading decisions using TraderAgent"""
        try:
            diversification_config = {
                'min_positions_for_session': self.min_positions_for_session,
                'target_positions_when_available': self.target_positions_when_available,
                'max_concurrent_positions': self.max_concurrent_positions
            }

            decisions = self.trader.execute(
                research_result['consensus'],
                current_positions,
                diversification_config=diversification_config
            )
            self.log_event("✅ Trading decisions generated")
            return decisions
        except Exception as e:
            self.log_event(f"❌ Trading decision error: {e}")
            return '{"trades": []}'

    def assess_risk(self, trade_decisions):
        """Assess risk of proposed trades"""
        try:
            assessment = self.risk_manager.execute(trade_decisions)
            status = assessment.get('portfolio_risk_status', 'Unknown')
            self.log_event(f"✅ Risk assessment: {status}")
            return assessment
        except Exception as e:
            self.log_event(f"❌ Risk assessment error: {e}")
            return {'trade_risk_assessment': 'Error', 'portfolio_risk_status': 'OK'}

    def get_fund_authorization(self, trade_decisions, risk_assessment):
        """Get Fund Manager authorization"""
        try:
            authorization = self.fund_manager.execute(trade_decisions, risk_assessment)
            decision = "APPROVED" if "APPROVE" in authorization.upper() else "REJECTED"
            self.log_event(f"✅ Fund Manager decision: {decision}")
            return authorization
        except Exception as e:
            self.log_event(f"❌ Fund authorization error: {e}")
            return "REJECT: Authorization error"

    def execute_approved_trades(self, authorization, trade_decisions, current_positions, ufo_data):
        """Execute trades if approved"""
        executed_count = 0

        if "APPROVE" not in authorization.upper():
            self.log_event("❌ Trades not approved - No execution")
            return executed_count

        # Check UFO engine conditions
        try:
            should_trade, reason = self.ufo_engine.should_open_new_trades(
                current_positions=current_positions,
                portfolio_status={'balance': self.portfolio_value, 'equity': self.portfolio_value},
                ufo_data=ufo_data
            )

            if not should_trade:
                self.log_event(f"❌ UFO Engine blocked trades: {reason}")
                return executed_count

            self.log_event(f"✅ UFO Engine approved: {reason}")

            # Simulate trade execution
            try:
                json_match = re.search(r'{.*}', trade_decisions, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    json_str = re.sub(r'//.*?\n', '\n', json_str)
                    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)

                    parsed_data = json.loads(json_str)
                    
                    # Extract trades
                    actions_list = []
                    if 'actions' in parsed_data:
                        actions_list = parsed_data['actions']
                    elif 'trade_plan' in parsed_data:
                        actions_list = parsed_data['trade_plan']
                    elif 'trades' in parsed_data:
                        for trade in parsed_data['trades']:
                            action_type = trade.get('action', 'new_trade')
                            if action_type == 'new_trade':
                                actions_list.append({
                                    'action': 'new_trade',
                                    'currency_pair': trade['currency_pair'],
                                    'direction': trade.get('direction', 'BUY').upper(),
                                    'volume': trade.get('lot_size', 0.1)
                                })
                            elif action_type == 'close_trade':
                                actions_list.append({
                                    'action': 'close_trade',
                                    'trade_id': trade.get('trade_id'),
                                    'currency_pair': trade.get('currency_pair')
                                })

                    # Execute each trade
                    for action in actions_list:
                        if action.get('action') == 'new_trade':
                            symbol = action.get('symbol') or action.get('currency_pair', '')
                            direction = action.get('direction', '').upper()
                            volume = action.get('volume') or action.get('lot_size', 0.1)

                            # Validate and correct currency pair format
                            base_symbol = symbol.replace("/", "")
                            corrected_symbol = self.validate_and_correct_currency_pair(base_symbol)

                            if corrected_symbol is None:
                                self.log_event(f"⚠️ Skipping invalid currency pair: {symbol}")
                                continue  # Skip this trade

                            # Also handle direction inversion if pair was inverted
                            if base_symbol != corrected_symbol and len(base_symbol) >= 6:
                                # Check if we need to invert the direction
                                original_base = base_symbol[:3]
                                corrected_base = corrected_symbol[:3]
                                if original_base != corrected_base:
                                    # Pair was inverted, so invert the direction
                                    direction = 'SELL' if direction == 'BUY' else 'BUY'
                                    self.log_event(f"⚠️ Direction inverted due to pair correction: {direction}")

                            # Add symbol suffix if it doesn't exist
                            suffix = self.config['mt5'].get('symbol_suffix', '')
                            if not corrected_symbol.endswith(suffix):
                                full_symbol = corrected_symbol + suffix
                            else:
                                full_symbol = corrected_symbol

                            # Use UFO-based entry price calculation for more realistic execution
                            entry_price = self.calculate_ufo_entry_price(
                                full_symbol,
                                direction,
                                ufo_data,
                                datetime.now()
                            )

                            # Create position info
                            position_info = {
                                'ticket': np.random.randint(100000, 999999),
                                'symbol': full_symbol,
                                'direction': direction,
                                'volume': volume,
                                'entry_price': entry_price,
                                'current_price': entry_price,
                                'pnl': 0.0,
                                'timestamp': datetime.now(),
                                'comment': f'UFO Cycle {self.cycle_count}'
                            }

                            # Add to both lists
                            self.open_positions.append(position_info)
                            self.trades_executed.append(position_info)
                            executed_count += 1

                            self.log_event(f"🔹 Trade executed: {full_symbol} {direction} {volume} lots @ {position_info['entry_price']:.5f}")

                        elif action.get('action') == 'close_trade':
                            trade_id = action.get('trade_id')
                            if trade_id:
                                position_to_close = next((p for p in self.open_positions if p.get('ticket') == trade_id), None)
                                if position_to_close:
                                    self.open_positions.remove(position_to_close)
                                    self.realized_pnl += position_to_close.get('pnl', 0.0)
                                    self.closed_trades.append(position_to_close)
                                    self.log_event(f"🔹 Trade closed by LLM: {position_to_close['symbol']} P&L: ${position_to_close.get('pnl', 0.0):.2f}")
                                    executed_count += 1
                            continue # Continue to next action

            except Exception as e:
                self.log_event(f"❌ Trade execution error: {e}")
                
        except Exception as e:
            self.log_event(f"❌ UFO engine error: {e}")

        return executed_count

    def generate_cycle_summary(self, executed_trades):
        """Generate summary for this cycle"""
        # Calculate unrealized P&L from open positions
        unrealized_pnl = sum(pos.get('pnl', 0.0) for pos in self.open_positions)
        total_pnl = self.portfolio_value - self.initial_balance

        self.log_event(f"📊 Cycle {self.cycle_count} Summary ({datetime.now().strftime('%H:%M')} GMT):")
        self.log_event(f"   Trades Executed: {executed_trades}")
        self.log_event(f"   Total Trades Today: {len(self.trades_executed)}")
        self.log_event(f"   Open Positions: {len(self.open_positions)}/{self.max_concurrent_positions}")
        self.log_event(f"   Closed Trades: {len(self.closed_trades)}")
        self.log_event(f"   Realized P&L: ${self.realized_pnl:+,.2f}")
        self.log_event(f"   Unrealized P&L: ${unrealized_pnl:+,.2f}")
        self.log_event(f"   Portfolio Value: ${self.portfolio_value:,.2f} (Total P&L: ${total_pnl:+,.2f})")

    def assess_portfolio(self):
        """Assess current portfolio positions"""
        try:
            # Use our realistic position tracking instead of MT5
            positions = self.open_positions
            position_count = len(positions) if positions is not None else 0
            self.log_event(f"✅ Portfolio assessed: {position_count} open positions")
            return positions
        except Exception as e:
            self.log_event(f"❌ Portfolio assessment error: {e}")
            # Return empty dataframe if error
            return pd.DataFrame()

    def run(self):
        """
        Runs the live trading loop with UFO methodology.
        """
        while True:
            try:
                self.run_single_cycle()
                time.sleep(self.cycle_period_minutes * 60)
            except KeyboardInterrupt:
                self.log_event("\nTrading interrupted by user. Exiting...")
                break
            except Exception as e:
                self.log_event(f"Error in trading cycle: {e}")
                self.log_event("Waiting 60 seconds before retrying...")
                time.sleep(60)

    def collect_market_data(self):
        """Collect market data for analysis for all symbols."""
        try:
            symbols = self.config['trading']['symbols'].split(',')
            all_data = {}
            for symbol in symbols:
                timeframes = [mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M15, mt5.TIMEFRAME_H1, mt5.TIMEFRAME_H4, mt5.TIMEFRAME_D1]
                timeframe_bars = {
                    mt5.TIMEFRAME_M5: 240,
                    mt5.TIMEFRAME_M15: 80,
                    mt5.TIMEFRAME_H1: 20,
                    mt5.TIMEFRAME_H4: 120,
                    mt5.TIMEFRAME_D1: 100
                }

                data = self.data_analyst.execute({
                    'source': 'mt5',
                    'symbol': symbol,
                    'timeframes': timeframes,
                    'num_bars': timeframe_bars
                })
                all_data[symbol] = data

            self.log_event(f"✅ Collected data for {len(all_data)} symbols")
            return all_data
        except Exception as e:
            self.log_event(f"❌ Data collection error: {e}")
            return None

    def calculate_ufo_indicators(self, price_data):
        """Calculate UFO indicators from price data with enhanced oscillation and uncertainty analysis"""
        if not price_data:
            return None

        try:
            # Reshape the data for the UfoCalculator
            reshaped_data = {}
            for symbol, timeframe_data in price_data.items():
                for timeframe, df in timeframe_data.items():
                    if timeframe not in reshaped_data:
                        reshaped_data[timeframe] = pd.DataFrame()
                    reshaped_data[timeframe][symbol] = df['close']

            incremental_sums_dict = {}
            for timeframe, price_df in reshaped_data.items():
                variation_data = self.ufo_calculator.calculate_percentage_variation(price_df)
                incremental_sums_dict[timeframe] = self.ufo_calculator.calculate_incremental_sum(variation_data)

            ufo_data = self.ufo_calculator.generate_ufo_data(incremental_sums_dict)

            # ENHANCED UFO ANALYSIS: Apply new oscillation and uncertainty detection
            oscillation_analysis = self.ufo_calculator.detect_oscillations(ufo_data)
            uncertainty_metrics = self.ufo_calculator.analyze_market_uncertainty(ufo_data, oscillation_analysis)
            coherence_analysis = self.ufo_calculator.detect_timeframe_coherence(ufo_data)

            # Store enhanced analysis for decision making
            enhanced_ufo_data = {
                'raw_data': ufo_data,
                'oscillation_analysis': oscillation_analysis,
                'uncertainty_metrics': uncertainty_metrics,
                'coherence_analysis': coherence_analysis
            }

            # Log enhanced analysis results
            self._log_enhanced_analysis(oscillation_analysis, uncertainty_metrics, coherence_analysis)

            self.log_event(f"✅ Enhanced UFO analysis completed for {len(ufo_data)} timeframes")
            return enhanced_ufo_data
        except Exception as e:
            self.log_event(f"❌ UFO calculation error: {e}")
            return None

    def _log_enhanced_analysis(self, oscillation_analysis, uncertainty_metrics, coherence_analysis):
        """Log enhanced UFO analysis results"""
        try:
            # Log market state summary across timeframes
            for timeframe, metrics in uncertainty_metrics.items():
                overall_state = metrics.get('overall_state', 'unknown')
                confidence = metrics.get('confidence_level', 'unknown')
                scaling = metrics.get('recommended_position_scaling', 1.0)

                self.log_event(f"🔍 {timeframe}: {overall_state} (confidence: {confidence}, scaling: {scaling:.2f})")

            # Log coherence insights
            strong_coherence_count = sum(1 for curr_data in coherence_analysis.values()
                                       if curr_data.get('coherence_level') == 'strong')
            total_currencies = len(coherence_analysis)

            if total_currencies > 0:
                coherence_ratio = strong_coherence_count / total_currencies
                self.log_event(f"📊 Timeframe Coherence: {strong_coherence_count}/{total_currencies} currencies show strong coherence ({coherence_ratio:.1%})")

            # Log mean reversion opportunities
            mean_reversion_signals = 0
            for tf_data in oscillation_analysis.values():
                mean_reversion_signals += sum(1 for curr_data in tf_data.values()
                                             if curr_data.get('mean_reversion_signal', False))

            if mean_reversion_signals > 0:
                self.log_event(f"🔄 Mean Reversion Signals: {mean_reversion_signals} detected across timeframes")

        except Exception as e:
            self.log_event(f"⚠️ Error logging enhanced analysis: {e}")

    def get_economic_events(self):
        """Get economic calendar events for simulation"""
        try:
            # Get raw events from cache
            raw_events = self.data_analyst.execute({'source': 'economic_calendar'})

            if raw_events is None or raw_events.empty:
                self.log_event("❌ No economic calendar data available")
                return pd.DataFrame()

            # Process events for simulation date with timezone conversion
            processed_events = self.process_economic_events(raw_events)
            event_count = len(processed_events) if processed_events is not None and not processed_events.empty else 0

            self.log_event(f"✅ Retrieved {event_count} economic events for today")
            return processed_events

        except Exception as e:
            self.log_event(f"❌ Economic calendar error: {e}")
            return pd.DataFrame()

    def process_economic_events(self, raw_events):
        """Process cached economic events for the simulation date with proper timezone handling"""
        try:
            if raw_events.empty:
                return pd.DataFrame()

            # Convert date strings to datetime with timezone awareness
            if 'date' in raw_events.columns:
                raw_events['datetime'] = pd.to_datetime(raw_events['date'], utc=True)
            else:
                self.log_event("❌ No date column found in economic calendar data")
                return pd.DataFrame()

            # Filter events for our simulation date (August 4th, 2025)
            today = datetime.now(pytz.utc).date()

            # Filter events that occur on our simulation date
            daily_events = raw_events[
                raw_events['datetime'].dt.date == today
            ].copy()

            if daily_events.empty:
                self.log_event(f"ℹ️ No economic events found for {today}")
                return pd.DataFrame()

            # Convert to GMT for simulation compatibility
            daily_events['gmt_time'] = daily_events['datetime'].dt.tz_convert('GMT')
            daily_events['gmt_hour'] = daily_events['gmt_time'].dt.hour
            daily_events['gmt_minute'] = daily_events['gmt_time'].dt.minute

            # Add trading impact assessment
            daily_events['trading_significance'] = daily_events['impact'].map({
                'High': 'Major market mover - high volatility expected',
                'Medium': 'Moderate market impact - monitor closely',
                'Low': 'Minor impact - background noise',
                'Holiday': 'Market holiday - reduced liquidity'
            })

            # Sort by GMT time
            daily_events = daily_events.sort_values('gmt_time')

            # Log key events for the trading day
            high_impact_events = daily_events[daily_events['impact'] == 'High']
            if not high_impact_events.empty:
                self.log_event(f"⚠️ {len(high_impact_events)} HIGH IMPACT events scheduled for trading day:")
                for _, event in high_impact_events.iterrows():
                    event_time = f"{event['gmt_hour']:02d}:{event['gmt_minute']:02d}"
                    self.log_event(f"  📅 {event_time} GMT: {event['country']} {event['title']}")

            return daily_events

        except Exception as e:
            self.log_event(f"❌ Error processing simulation economic events: {e}")
            return pd.DataFrame()

    def analyze_ufo_exit_signals(self, current_ufo_data, previous_ufo_data):
        """Analyze UFO data for exit signals based on currency strength changes"""
        exit_signals = []

        if previous_ufo_data is None:
            return exit_signals

        # Extract raw UFO data from enhanced structure
        current_raw_data = current_ufo_data.get('raw_data', current_ufo_data)
        previous_raw_data = previous_ufo_data.get('raw_data', previous_ufo_data)

        # Check for currency strength reversals across timeframes
        for timeframe in current_raw_data.keys():
            if timeframe not in previous_raw_data:
                continue

            current_strengths = current_raw_data[timeframe]
            previous_strengths = previous_raw_data[timeframe]

            # Handle both DataFrame and dict formats
            if hasattr(current_strengths, 'columns'):
                # DataFrame format
                currency_list = current_strengths.columns
            else:
                # Dict format
                currency_list = current_strengths.keys()

            # Detect significant strength changes
            for currency in currency_list:
                if hasattr(previous_strengths, 'columns'):
                    # DataFrame format
                    if currency not in previous_strengths.columns:
                        continue
                    current_strength = current_strengths[currency].iloc[-1]
                    previous_strength = previous_strengths[currency].iloc[-5:]  # Last 5 bars average
                    avg_previous = previous_strength.mean()
                else:
                    # Dict format
                    if currency not in previous_strengths:
                        continue
                    current_strength = current_strengths[currency][-1]
                    previous_strength = previous_strengths[currency][-5:]  # Last 5 bars average
                    avg_previous = sum(previous_strength) / len(previous_strength)

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

    def close_affected_positions(self, exit_signals):
        """Close positions affected by strong exit signals"""
        positions_closed = 0
        currencies_to_close = set()

        # Extract currencies from exit signals
        for signal in exit_signals:
            currencies_to_close.add(signal['currency'])

        open_positions = self.risk_manager.portfolio_manager.get_positions()
        if open_positions is None or open_positions.empty:
            return 0

        # Find positions that involve these currencies
        positions_to_close = []
        for index, position in open_positions.iterrows():
            symbol = position['symbol'].replace('-ECN', '')

            # Extract base and quote currencies
            if len(symbol) >= 6:
                base_currency = symbol[:3]
                quote_currency = symbol[3:6]

                # Check if either currency is affected by exit signals
                if base_currency in currencies_to_close or quote_currency in currencies_to_close:
                    positions_to_close.append(position['ticket'])
                    self.log_event(f"🚨 Marking {symbol} for closure due to {base_currency}/{quote_currency} exit signals")

        # Close positions
        for ticket in positions_to_close:
            self.trade_executor.close_trade(ticket)
            positions_closed += 1

        return positions_closed

    def get_historical_price_for_time(self, symbol, target_time):
        """Get real historical price for a specific symbol at a specific time"""
        try:
            # Convert target time to MT5 timestamp
            target_timestamp = int(target_time.timestamp())

            # Get historical data around the target time (M5 bars)
            rates = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_M5, target_timestamp, 1)

            if rates is not None and len(rates) > 0:
                # Return the close price
                return float(rates[0]['close'])
            else:
                # Fallback: get the most recent data if exact time not available
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 1)
                if rates is not None and len(rates) > 0:
                    return float(rates[0]['close'])
                else:
                    # If no historical data is available, return None
                    return None
        except Exception as e:
            self.log_event(f"⚠️ Error getting historical price for {symbol}: {e}")
            return None

    def validate_and_correct_currency_pair(self, pair):
        """Validate and correct currency pair format

        Handles cases like:
        - CADUSD -> USDCAD (inverted)
        - CHFUSD -> USDCHF (inverted)
        - USDGBP -> GBPUSD (inverted)
        - Invalid pairs return None
        """
        # Define valid currency pairs available in MT5
        valid_pairs = [
            'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF',
            'EURAUD', 'EURCAD', 'EURCHF', 'EURGBP', 'EURJPY', 'EURNZD',
            'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPJPY', 'GBPNZD',
            'AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'AUDGBP',
            'CADCHF', 'CADJPY', 'CHFJPY', 'NZDCAD', 'NZDCHF', 'NZDJPY',
            'NZDUSD'
        ]

        # Clean the pair (remove suffix, slashes, etc.)
        clean_pair = pair.replace('-ECN', '').replace('/', '').upper()

        # If already valid, return it
        if clean_pair in valid_pairs:
            return clean_pair

        # Try to extract base and quote currencies
        if len(clean_pair) >= 6:
            base = clean_pair[:3]
            quote = clean_pair[3:6]

            # Check if inverted pair exists
            inverted = quote + base
            if inverted in valid_pairs:
                self.log_event(f"⚠️ Correcting inverted pair: {clean_pair} -> {inverted}")
                return inverted

            # Special handling for common inversions
            inversion_map = {
                'CADUSD': 'USDCAD',
                'CHFUSD': 'USDCHF',
                'CHFEUR': 'EURCHF',
                'CHFGBP': 'GBPCHF',
                'JPYUSD': 'USDJPY',
                'JPYEUR': 'EURJPY',
                'JPYGBP': 'GBPJPY',
                'JPYAUD': 'AUDJPY',
                'JPYCAD': 'CADJPY',
                'JPYCHF': 'CHFJPY',
                'JPYNZD': 'NZDJPY',
                'NZDEUR': 'EURNZD',
                'NZDGBP': 'GBPNZD',
                'NZDAUD': 'AUDNZD',
                'CADEUR': 'EURCAD',
                'CADGBP': 'GBPCAD',
                'CADAUD': 'AUDCAD',
                'USDEUR': 'EURUSD',
                'USDGBP': 'GBPUSD',
                'USDAUD': 'AUDUSD',
                'USDNZD': 'NZDUSD',
                'GBPEUR': 'EURGBP',
                'AUDEUR': 'EURAUD',
                'AUDGBP': 'GBPAUD'
            }

            if clean_pair in inversion_map:
                corrected = inversion_map[clean_pair]
                self.log_event(f"⚠️ Correcting known inverted pair: {clean_pair} -> {corrected}")
                return corrected

        # If we can't fix it, log error and return None
        self.log_event(f"❌ Invalid currency pair: {pair} (cleaned: {clean_pair})")
        return None

    def calculate_ufo_entry_price(self, symbol, direction, ufo_data, current_time):
        """Calculate optimal entry price based on UFO methodology and currency strength"""
        try:
            # Get base historical price
            base_price = self.get_historical_price_for_time(symbol, current_time)
            if base_price is None:
                # Fallback to standard base prices
                base_prices = {
                    'EURUSD-ECN': 1.0850, 'GBPUSD-ECN': 1.2650, 'USDJPY-ECN': 143.50,
                    'AUDUSD-ECN': 0.6720, 'USDCAD-ECN': 1.3580, 'NZDUSD-ECN': 0.6250,
                    'EURJPY-ECN': 155.20, 'GBPJPY-ECN': 180.50, 'AUDJPY-ECN': 96.30,
                    'USDCHF-ECN': 0.9120, 'EURCHF-ECN': 0.9880, 'GBPCHF-ECN': 1.1520,
                    'AUDCAD-ECN': 0.9080, 'NZDJPY-ECN': 89.60, 'CADCHF-ECN': 0.6730,
                    'CHFJPY-ECN': 157.20, 'AUDNZD-ECN': 1.0750, 'EURGBP-ECN': 0.8590,
                    'GBPCAD-ECN': 1.7180, 'XAUUSD-ECN': 1850.00, 'GBPAUD-ECN': 1.8820
                }
                base_price = base_prices.get(symbol, 1.0850)

            # Calculate UFO adjustment if UFO data is available
            if ufo_data:
                # Extract currencies from symbol
                clean_symbol = symbol.replace('-ECN', '')
                if len(clean_symbol) >= 6:
                    base_currency = clean_symbol[:3]
                    quote_currency = clean_symbol[3:6]

                    # Get currency strengths from M5 timeframe (primary trading timeframe)
                    primary_tf = mt5.TIMEFRAME_M5
                    raw_ufo_data = ufo_data.get('raw_data', ufo_data)

                    if primary_tf in raw_ufo_data:
                        strength_data = raw_ufo_data[primary_tf]

                        base_strength = 0.0
                        quote_strength = 0.0

                        # Handle both DataFrame and dict formats
                        if hasattr(strength_data, 'columns'):
                            # DataFrame format
                            if base_currency in strength_data.columns:
                                base_strength = strength_data[base_currency].iloc[-1]
                            if quote_currency in strength_data.columns:
                                quote_strength = strength_data[quote_currency].iloc[-1]
                        else:
                            # Dict format
                            if base_currency in strength_data:
                                base_strength = strength_data[base_currency][-1]
                            if quote_currency in strength_data:
                                quote_strength = strength_data[quote_currency][-1]

                        # Calculate strength differential
                        strength_diff = base_strength - quote_strength

                        # UFO-based price adjustment
                        # Strong differential = better entry timing = slight price improvement
                        if abs(strength_diff) > 1.0:  # Significant strength difference
                            # Apply small adjustment in favorable direction
                            if direction == 'BUY' and strength_diff > 0:  # Strong base currency
                                # Slightly better entry (lower price for BUY)
                                price_adjustment = -base_price * 0.0002  # 2 pip improvement
                            elif direction == 'SELL' and strength_diff < 0:  # Strong quote currency
                                # Slightly better entry (higher price for SELL)
                                price_adjustment = base_price * 0.0002  # 2 pip improvement
                            else:
                                # Weaker signal, use current historical price as-is
                                price_adjustment = 0.0
                        else:
                            # Normal market entry - use historical price as-is
                            price_adjustment = 0.0

                        optimal_price = base_price + price_adjustment
                        return max(optimal_price, base_price * 0.95)  # Safety minimum

            # Fallback: use historical price as-is without any random adjustments
            return base_price

        except Exception as e:
            self.log_event(f"⚠️ Error calculating UFO entry price for {symbol}: {e}")
            # Ultimate fallback
            return 1.0850 if 'EUR' in symbol else 143.50 if 'JPY' in symbol else 1.2650

    def continuous_position_monitoring(self):
        """Perform continuous position monitoring between trading cycles"""
        try:
            if not self.open_positions:
                return

            # Check if we've already monitored this exact time to prevent loops
            if hasattr(self, '_last_monitoring_time') and self._last_monitoring_time == datetime.now():
                return
            self._last_monitoring_time = datetime.now()

            # Force portfolio value update during continuous monitoring
            self.update_portfolio_value(force_update=True)

            # Check for significant portfolio changes that require immediate action
            if self.portfolio_history:
                recent_changes = self.portfolio_history[-5:]  # Last 5 updates
                if len(recent_changes) >= 2:
                    latest_value = recent_changes[-1]['portfolio_value']
                    previous_value = recent_changes[-2]['portfolio_value']
                    rapid_change = abs(latest_value - previous_value) / previous_value * 100

                    if rapid_change > 1.0:  # 1% rapid change threshold
                        self.log_event(f"⚡ Rapid portfolio change: {rapid_change:.2f}% in {self.position_update_frequency_minutes} min")

                        # Check if portfolio stop is approaching
                        current_drawdown = ((latest_value - self.initial_balance) / self.initial_balance) * 100
                        if current_drawdown < (self.portfolio_equity_stop * 0.8):  # 80% of stop threshold
                            self.log_event(f"⚠️ Approaching portfolio stop: {current_drawdown:.2f}% (threshold: {self.portfolio_equity_stop}%)")

            # Check for positions with excessive unrealized losses during monitoring
            high_risk_positions = []
            for position in self.open_positions:
                if position.get('pnl', 0) < -75:  # High risk threshold
                    high_risk_positions.append(position)

            if high_risk_positions:
                self.log_event(f"🚨 Monitoring alert: {len(high_risk_positions)} positions with high unrealized losses")
                for pos in high_risk_positions[:3]:  # Log top 3
                    self.log_event(f"  ⚠️ {pos['symbol']}: P&L ${pos['pnl']:.2f}")

            # Enhanced Dynamic Reinforcement monitoring
            if self.dynamic_reinforcement_engine.enabled and self.dynamic_reinforcement_engine.should_check_reinforcement(datetime.now()):
                current_market_data = self.get_real_time_market_data_for_positions(self.open_positions)

                # Detect market events that trigger reinforcement
                market_events = self.dynamic_reinforcement_engine.detect_market_events(
                    self.open_positions,
                    current_market_data,
                    getattr(self, 'previous_ufo_data', None)
                )

                if market_events:
                    self.log_event(f"🎯 Dynamic Reinforcement: {len(market_events)} market events detected")

                    # Process each event for reinforcement
                    for event in market_events:
                        position = event.get('position')
                        if position:
                            # Calculate dynamic reinforcement for this event
                            reinforcement_plan, message = self.dynamic_reinforcement_engine.calculate_dynamic_reinforcement(
                                position,
                                event,
                                current_market_data,
                                getattr(self, 'previous_ufo_data', None)
                            )

                            if reinforcement_plan:
                                self.log_event(f"  ⚡ {event['type']}: {position['symbol']} - {message}")
                                self.log_event(f"    📊 Reinforcement: {reinforcement_plan['additional_lots']:.2f} lots")

                                # Execute reinforcement (in simulation)
                                self.execute_dynamic_reinforcement(position, reinforcement_plan, datetime.now())
                            else:
                                self.log_event(f"  ⏸️ {position['symbol']}: {message}")

                # Also check UFO-based reinforcement for compatibility
                if hasattr(self, 'previous_ufo_data'):
                    for position in self.open_positions:
                        # Check if UFO engine also suggests reinforcement
                        should_reinforce, reason, plan = self.ufo_engine.should_reinforce_position(
                            position,
                            self.previous_ufo_data,
                            current_market_data
                        )
                        if should_reinforce and plan:
                            self.log_event(f"  🛸 UFO reinforcement suggestion: {position['symbol']} - {reason}")

        except Exception as e:
            self.log_event(f"❌ Error in continuous position monitoring: {e}")

    def execute_dynamic_reinforcement(self, position, reinforcement_plan, current_time):
        """Execute dynamic reinforcement trade in simulation"""
        try:
            # Create reinforcement position
            reinforcement_position = {
                'ticket': np.random.randint(100000, 999999),
                'symbol': position['symbol'],
                'direction': position['direction'],  # Same direction as original
                'volume': reinforcement_plan['additional_lots'],
                'entry_price': self.get_historical_price_for_time(position['symbol'], current_time) or position['current_price'],
                'current_price': position.get('current_price', position['entry_price']),
                'pnl': 0.0,
                'timestamp': current_time,
                'comment': f'Dynamic {reinforcement_plan["type"]}',
                'original_position': position.get('ticket'),
                'reinforcement_details': reinforcement_plan
            }

            # Add to open positions
            self.open_positions.append(reinforcement_position)

            # Record in dynamic reinforcement engine
            self.dynamic_reinforcement_engine.record_reinforcement(position, reinforcement_plan)

            self.log_event(f"    ✅ Dynamic reinforcement executed: {reinforcement_position['symbol']} "
                          f"{reinforcement_position['direction']} {reinforcement_position['volume']:.2f} lots "
                          f"@ {reinforcement_position['entry_price']:.5f}")

            # Also add to trades executed for tracking
            self.trades_executed.append(reinforcement_position)

        except Exception as e:
            self.log_event(f"    ❌ Failed to execute dynamic reinforcement: {e}")

    def check_portfolio_status(self):
        """
        Checks overall portfolio status using UFO methodology.
        """
        try:
            positions = self.risk_manager.portfolio_manager.get_positions()
            if positions is None or len(positions) == 0:
                return
                
            portfolio_value = self.ufo_engine.calculate_portfolio_synthetic_value()
            self.log_event(f"Portfolio synthetic value: {portfolio_value:.2f}%")
            
            if portfolio_value <= -5.0:  # Portfolio stop loss threshold
                self.log_event("Portfolio stop loss triggered - closing all positions")
                for position in positions:
                    self.trade_executor.close_trade(position.ticket)
                    
        except Exception as e:
            self.log_event(f"Error checking portfolio status: {e}")
