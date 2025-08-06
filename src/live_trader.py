import time
import pandas as pd
import numpy as np
import re
import json
from datetime import datetime
import pytz
import os

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
from .ufo_calculator import UfoCalculator
from .llm.llm_client import LLMClient
from .trade_executor import TradeExecutor
from .ufo_trading_engine import UFOTradingEngine
from .dynamic_reinforcement_engine import DynamicReinforcementEngine
from .portfolio_manager import PortfolioManager

class LiveTrader:
    def __init__(self, config):
        self.config = config
        self.previous_ufo_data = None
        self.simulation_log = []
        self.cycle_count = 0
        self.trades_executed_today = []
        self.portfolio_history = []
        
        # Continuous monitoring variables
        self.position_update_frequency_minutes = 5
        self.continuous_monitoring_enabled = True
        self.last_monitoring_time = None
        self.position_metadata = {} # To store peak_pnl for trailing stops

        # Fix config parsing issues
        self.fix_config_values()

        # Initialize components
        self.initialize_components()

    def fix_config_values(self):
        """Fix configuration values that have comments or multiple values"""
        def parse_value(value, default):
            if isinstance(value, str):
                clean_value = value.split('#')[0].split('(')[0].strip()
                try:
                    return float(clean_value) if '.' in clean_value else int(clean_value)
                except ValueError:
                    return default
            return value
        
        self.portfolio_equity_stop = parse_value(self.config['trading'].get('portfolio_equity_stop', '-5.0'), -5.0)
        self.cycle_period_minutes = parse_value(self.config['trading'].get('cycle_period_minutes', '40'), 40)
        self.max_concurrent_positions = parse_value(self.config['trading'].get('max_concurrent_positions', '11'), 11)
        self.target_positions_when_available = parse_value(self.config['trading'].get('target_positions_when_available', '6'), 6)
        self.min_positions_for_session = parse_value(self.config['trading'].get('min_positions_for_session', '5'), 5)
        self.position_update_frequency_minutes = parse_value(self.config['trading'].get('position_update_frequency_minutes', '5'), 5)

    def initialize_components(self):
        """Initialize all trading components"""
        self.llm_client = LLMClient(api_key=self.config['openrouter']['api_key'])
        self.mt5_collector = MT5DataCollector(
            login=self.config['mt5']['login'],
            password=self.config['mt5']['password'],
            server=self.config['mt5']['server'],
            path=self.config['mt5']['path']
        )

        if not self.mt5_collector.connect():
            self.log_event("⚠️ Warning: MT5 connection failed, using mock data")
        else:
            self.log_event("✅ MT5 persistent connection established")

        self.ufo_calculator = UfoCalculator(self.config['trading']['currencies'].split(','))
        self.ufo_engine = UFOTradingEngine(self.config)
        self.portfolio_manager = PortfolioManager(self.mt5_collector)

        symbols_list = self.config['trading']['symbols'].split(',')
        self.data_analyst = DataAnalystAgent("DataAnalyst", self.mt5_collector)
        self.market_researcher = MarketResearcherAgent("MarketResearcher", self.llm_client)
        self.trader = TraderAgent("Trader", self.llm_client, self.mt5_collector, symbols=symbols_list)
        self.risk_manager = RiskManagerAgent("RiskManager", self.llm_client, self.mt5_collector, self.config)
        self.fund_manager = FundManagerAgent("FundManager", self.llm_client)
        self.trade_executor = TradeExecutor(self.mt5_collector, self.config)

        self.dynamic_reinforcement_engine = DynamicReinforcementEngine(self.config)
        if self.dynamic_reinforcement_engine.enabled:
            self.log_event("✅ Dynamic Reinforcement Engine enabled")
        else:
            self.log_event("⚠️ Dynamic Reinforcement Engine disabled")

        self.log_event("Live Trader components initialized successfully")

    def log_event(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.simulation_log.append(log_entry)
        print(log_entry)

    def run_single_cycle(self):
        """Run a single trading cycle in the live environment."""
        self.cycle_count += 1
        self.log_event(f"\n" + "="*60)
        self.log_event(f"CYCLE {self.cycle_count} - {datetime.now().strftime('%H:%M')} GMT")
        self.log_event("="*60)

        # 0. Portfolio Assessment
        self.log_event("💼 PHASE 0: Portfolio Assessment")
        account_info = self.portfolio_manager.get_account_info()
        if not account_info:
            self.log_event("Could not get account info. Skipping cycle.")
            return True
        current_positions = self.assess_portfolio()

        # 1. Data Collection
        self.log_event("📊 PHASE 1: Data Collection")
        price_data = self.collect_market_data()

        # 2. UFO Analysis
        self.log_event("🛸 PHASE 2: UFO Analysis")
        ufo_data = self.calculate_ufo_indicators(price_data)

        # Coherence Check
        if ufo_data and 'raw_data' in ufo_data:
            coherence_issues = self.check_multi_timeframe_coherence(ufo_data['raw_data'])
            if coherence_issues:
                self.log_event(f"⚠️ UFO Coherence Issues Detected: {len(coherence_issues)}")
                for issue in coherence_issues:
                    self.log_event(f"   - {issue['currency']}: {issue['issue']}")

        # 3. Economic Calendar
        self.log_event("📅 PHASE 3: Economic Calendar")
        economic_events = self.get_economic_events()

        # 4. Market Research
        self.log_event("🔍 PHASE 4: Market Research")
        research_result = self.market_researcher.execute(ufo_data, economic_events)

        # 5. UFO Portfolio Management
        self.log_event("💼 PHASE 5: UFO Portfolio Management")
        portfolio_stop_breached, stop_reason = self.check_portfolio_equity_stop(account_info)
        if portfolio_stop_breached:
            self.log_event(f"🚨 UFO PORTFOLIO STOP TRIGGERED: {stop_reason}")
            self.trade_executor.close_all_trades()
            self.log_event("🚨 All positions closed. UFO Portfolio Stop engaged.")
            return False # Stop trading

        should_close, close_reason = self.ufo_engine.should_close_for_session_end(economic_events)
        if should_close:
            self.log_event(f"🌅 UFO SESSION END: {close_reason}")
            self.trade_executor.close_all_trades()
            return True

        if self.previous_ufo_data and ufo_data:
            exit_signals = self.analyze_ufo_exit_signals(ufo_data, self.previous_ufo_data)
            if exit_signals:
                self.log_event(f"📈 UFO Exit Signals detected: {len(exit_signals)} currency changes")
                if len(exit_signals) >= 3:
                    self.log_event("🚨 STRONG EXIT SIGNALS detected: Auto-closing affected positions")
                    positions_closed = self.close_affected_positions(exit_signals)
                    self.log_event(f"🚨 Auto-closed {positions_closed} positions.")

        self.previous_ufo_data = ufo_data

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
        self.generate_cycle_summary(executed_trades, account_info, current_positions)

        return True

    def run(self):
        """Runs the live trading loop."""
        self.log_event("🚀 Starting Live UFO Trading Bot")
        self.log_event(f"⏰ Cycle Frequency: Every {self.cycle_period_minutes} minutes")
        self.log_event(f"📊 Continuous Monitoring: Every {self.position_update_frequency_minutes} minutes")

        next_cycle_time = datetime.now()

        try:
            while True:
                now = datetime.now()
                
                if not self.ufo_engine.is_active_session():
                    self.log_event(f"⏰ Outside trading hours. Waiting for the next session.")
                    time.sleep(300) # Sleep for 5 minutes before checking again
                    continue

                if now >= next_cycle_time:
                    if not self.run_single_cycle():
                        self.log_event("🛑 Trading stopped due to portfolio stop loss.")
                        break
                    next_cycle_time = now + pd.DateOffset(minutes=self.cycle_period_minutes)

                if self.continuous_monitoring_enabled:
                    self.continuous_position_monitoring()

                time.sleep(60)

        except KeyboardInterrupt:
            self.log_event("\nTrading interrupted by user. Exiting...")
        except Exception as e:
            self.log_event(f"💥 An unexpected error occurred in the main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.generate_final_summary()
            self.save_daily_report()
            self.mt5_collector.disconnect()
            self.log_event("✅ MT5 connection closed. Bot shut down.")

    def assess_portfolio(self):
        """Assess current portfolio positions and trigger position management."""
        try:
            # This method now primarily triggers the update and gets the latest positions
            self.update_portfolio_value()
            positions = self.portfolio_manager.get_positions()
            position_count = len(positions) if not positions.empty else 0
            self.log_event(f"✅ Portfolio assessed: {position_count} open positions")
            return positions
        except Exception as e:
            self.log_event(f"❌ Portfolio assessment error: {e}")
            return pd.DataFrame()

    def update_portfolio_value(self, force_update=False):
        """
        Update portfolio value based on open positions P&L using real-time prices
        and manage positions based on profit/loss, time, and trailing stops.
        """
        now = datetime.now()
        if not force_update and self.last_monitoring_time and \
           (now - self.last_monitoring_time).total_seconds() / 60 < self.position_update_frequency_minutes:
            return

        open_positions = self.portfolio_manager.get_positions()
        if open_positions.empty:
            self.position_metadata.clear()  # Clear metadata if no positions are open
            return

        positions_to_close = []

        # Cleanup metadata for positions that have been manually closed or no longer exist
        open_tickets = set(open_positions['ticket'])
        for ticket in list(self.position_metadata.keys()):
            if ticket not in open_tickets:
                del self.position_metadata[ticket]

        for index, position in open_positions.iterrows():
            ticket = position['ticket']
            pnl = position['profit']

            # Initialize or update peak_pnl for trailing stop logic
            if ticket not in self.position_metadata:
                self.position_metadata[ticket] = {'peak_pnl': pnl}
            elif pnl > self.position_metadata[ticket]['peak_pnl']:
                self.position_metadata[ticket]['peak_pnl'] = pnl

            peak_pnl = self.position_metadata[ticket]['peak_pnl']

            # Enhanced position closing logic from the simulator
            close_on_profit = pnl > 75  # Take profit at +$75
            close_on_loss = pnl < -50   # Stop loss at -$50

            # Time-based exit: close positions older than 4 hours
            position_age_hours = (now.replace(tzinfo=None) - position['time'].replace(tzinfo=None)).total_seconds() / 3600
            close_on_time = position_age_hours > 4

            # Trailing stop: close if position has moved against us significantly from its peak profit
            close_on_trailing = peak_pnl > 30 and pnl < peak_pnl * 0.7

            if close_on_profit or close_on_loss or close_on_time or close_on_trailing:
                positions_to_close.append(ticket)
                if close_on_profit:
                    close_reason = "profit target"
                elif close_on_loss:
                    close_reason = "stop loss"
                elif close_on_time:
                    close_reason = "time-based exit"
                else:
                    close_reason = "trailing stop"
                self.log_event(f"🎯 Marking {position['symbol']} for closure: {close_reason} (P&L: ${pnl:.2f})")

        # Close marked positions
        for ticket in positions_to_close:
            if self.trade_executor.close_trade(ticket):
                if ticket in self.position_metadata:
                    del self.position_metadata[ticket]

        self.last_monitoring_time = now

    def collect_market_data(self):
        try:
            symbols = self.config['trading']['symbols'].split(',')
            all_data = {}
            for symbol in symbols:
                timeframes = [mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M15, mt5.TIMEFRAME_H1, mt5.TIMEFRAME_H4, mt5.TIMEFRAME_D1]
                timeframe_bars = {
                    mt5.TIMEFRAME_M5: 240, mt5.TIMEFRAME_M15: 80, mt5.TIMEFRAME_H1: 20,
                    mt5.TIMEFRAME_H4: 120, mt5.TIMEFRAME_D1: 100
                }
                data = self.data_analyst.execute({
                    'source': 'mt5', 'symbol': symbol,
                    'timeframes': timeframes, 'num_bars': timeframe_bars
                })
                all_data[symbol] = data
            self.log_event(f"✅ Collected data for {len(all_data)} symbols")
            return all_data
        except Exception as e:
            self.log_event(f"❌ Data collection error: {e}")
            return None

    def calculate_ufo_indicators(self, price_data):
        if not price_data: return None
        try:
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

            oscillation_analysis = self.ufo_calculator.detect_oscillations(ufo_data)
            uncertainty_metrics = self.ufo_calculator.analyze_market_uncertainty(ufo_data, oscillation_analysis)
            coherence_analysis = self.ufo_calculator.detect_timeframe_coherence(ufo_data)

            enhanced_ufo_data = {
                'raw_data': ufo_data, 'oscillation_analysis': oscillation_analysis,
                'uncertainty_metrics': uncertainty_metrics, 'coherence_analysis': coherence_analysis
            }
            self._log_enhanced_analysis(oscillation_analysis, uncertainty_metrics, coherence_analysis)
            self.log_event(f"✅ Enhanced UFO analysis completed")
            return enhanced_ufo_data
        except Exception as e:
            self.log_event(f"❌ UFO calculation error: {e}")
            return None

    def _log_enhanced_analysis(self, oscillation_analysis, uncertainty_metrics, coherence_analysis):
        """Log the results of the enhanced UFO analysis."""
        try:
            # Log market state summary
            for timeframe, metrics in uncertainty_metrics.items():
                state = metrics.get('overall_state', 'unknown')
                confidence = metrics.get('confidence_level', 'unknown')
                scaling = metrics.get('recommended_position_scaling', 1.0)
                self.log_event(f"🔍 {timeframe}: {state} (confidence: {confidence}, scaling: {scaling:.2f})")

            # Log coherence insights
            if coherence_analysis:
                strong_coherence_count = sum(1 for data in coherence_analysis.values() if data.get('coherence_level') == 'strong')
                total_currencies = len(coherence_analysis)
                if total_currencies > 0:
                    coherence_ratio = strong_coherence_count / total_currencies
                    self.log_event(f"📊 Timeframe Coherence: {strong_coherence_count}/{total_currencies} currencies show strong coherence ({coherence_ratio:.1%})")

            # Log mean reversion signals
            mean_reversion_signals = 0
            if oscillation_analysis:
                for tf_data in oscillation_analysis.values():
                    mean_reversion_signals += sum(1 for data in tf_data.values() if data.get('mean_reversion_signal', False))
                if mean_reversion_signals > 0:
                    self.log_event(f"🔄 Mean Reversion Signals: {mean_reversion_signals} detected across timeframes")
        except Exception as e:
            self.log_event(f"⚠️ Error logging enhanced analysis: {e}")

    def get_economic_events(self):
        try:
            raw_events = self.data_analyst.execute({'source': 'economic_calendar'})
            if raw_events is None or raw_events.empty:
                return pd.DataFrame()
            processed_events = self.process_economic_events(raw_events)
            self.log_event(f"✅ Retrieved {len(processed_events)} economic events for today")
            return processed_events
        except Exception as e:
            self.log_event(f"❌ Economic calendar error: {e}")
            return pd.DataFrame()

    def process_economic_events(self, raw_events):
        """Process and enrich economic events with trading significance."""
        try:
            if raw_events.empty:
                return pd.DataFrame()

            raw_events['datetime'] = pd.to_datetime(raw_events['date'], utc=True)
            today = datetime.now(pytz.utc).date()
            daily_events = raw_events[raw_events['datetime'].dt.date == today].copy()

            if daily_events.empty:
                return pd.DataFrame()

            # Add trading significance assessment
            daily_events['trading_significance'] = daily_events['impact'].map({
                'High': 'Major market mover - high volatility expected',
                'Medium': 'Moderate market impact - monitor closely',
                'Low': 'Minor impact - background noise',
                'Holiday': 'Market holiday - reduced liquidity'
            }).fillna('Not specified')

            daily_events['gmt_time'] = daily_events['datetime'].dt.tz_convert('GMT')
            daily_events['gmt_hour'] = daily_events['gmt_time'].dt.hour
            daily_events['gmt_minute'] = daily_events['gmt_time'].dt.minute

            # Log high-impact events
            high_impact_events = daily_events[daily_events['impact'] == 'High']
            if not high_impact_events.empty:
                self.log_event(f"⚠️ {len(high_impact_events)} HIGH IMPACT events scheduled for today:")
                for _, event in high_impact_events.iterrows():
                    event_time_str = event['gmt_time'].strftime('%H:%M')
                    self.log_event(f"  📅 {event_time_str} GMT: {event['country']} {event['title']}")

            return daily_events.sort_values('gmt_time')
        except Exception as e:
            self.log_event(f"❌ Error processing economic events: {e}")
            return pd.DataFrame()

    def generate_trade_decisions(self, research_result, current_positions):
        try:
            diversification_config = {
                'min_positions_for_session': self.min_positions_for_session,
                'target_positions_when_available': self.target_positions_when_available,
                'max_concurrent_positions': self.max_concurrent_positions
            }
            decisions = self.trader.execute(
                research_result['consensus'], current_positions,
                diversification_config=diversification_config
            )
            self.log_event("✅ Trading decisions generated")
            return decisions
        except Exception as e:
            self.log_event(f"❌ Trading decision error: {e}")
            return '{"trades": []}'

    def assess_risk(self, trade_decisions):
        try:
            assessment = self.risk_manager.execute(trade_decisions)
            self.log_event(f"✅ Risk assessment: {assessment.get('portfolio_risk_status', 'Unknown')}")
            return assessment
        except Exception as e:
            self.log_event(f"❌ Risk assessment error: {e}")
            return {'portfolio_risk_status': 'Error'}

    def get_fund_authorization(self, trade_decisions, risk_assessment):
        try:
            authorization = self.fund_manager.execute(trade_decisions, risk_assessment)
            decision = "APPROVED" if "APPROVE" in authorization.upper() else "REJECTED"
            self.log_event(f"✅ Fund Manager decision: {decision}")
            return authorization
        except Exception as e:
            self.log_event(f"❌ Fund authorization error: {e}")
            return "REJECT"

    def execute_approved_trades(self, authorization, trade_decisions, current_positions, ufo_data):
        executed_count = 0
        if "APPROVE" not in authorization.upper():
            self.log_event("❌ Trades not approved - No execution")
            return executed_count

        try:
            account_info = self.portfolio_manager.get_account_info()
            should_trade, reason = self.ufo_engine.should_open_new_trades(
                current_positions=current_positions,
                portfolio_status=account_info,
                ufo_data=ufo_data
            )
            if not should_trade:
                self.log_event(f"❌ UFO Engine blocked trades: {reason}")
                return executed_count

            json_match = re.search(r'{.*}', trade_decisions, re.DOTALL)
            if not json_match: return executed_count

            parsed_data = json.loads(re.sub(r',\s*([}\]])', r'\1', re.sub(r'//.*?\n', '\n', json_match.group(0))))
            actions_list = parsed_data.get('actions', parsed_data.get('trade_plan', parsed_data.get('trades', [])))

            for action in actions_list:
                if action.get('action') == 'new_trade':
                    symbol = action.get('symbol') or action.get('currency_pair', '')
                    direction = action.get('direction', '').upper()
                    volume = action.get('volume') or action.get('lot_size', 0.1)

                    base_symbol = symbol.replace("/", "")
                    corrected_symbol = self.validate_and_correct_currency_pair(base_symbol)
                    if corrected_symbol is None: continue

                    if base_symbol != corrected_symbol and len(base_symbol) >= 6:
                        if base_symbol[:3] != corrected_symbol[:3]:
                            direction = 'SELL' if direction == 'BUY' else 'BUY'
                            self.log_event(f"⚠️ Direction inverted to {direction} due to pair correction")

                    suffix = self.config['mt5'].get('symbol_suffix', '')
                    full_symbol = corrected_symbol if corrected_symbol.endswith(suffix) else corrected_symbol + suffix

                    entry_price = self.calculate_ufo_entry_price(full_symbol, direction, ufo_data)

                    trade_type = mt5.ORDER_TYPE_BUY if direction == 'BUY' else mt5.ORDER_TYPE_SELL
                    result = self.trade_executor.execute_ufo_trade(
                        symbol=full_symbol,
                        trade_type=trade_type,
                        volume=volume,
                        price=entry_price,
                        comment=f'UFO Cycle {self.cycle_count}'
                    )
                    if result:
                        executed_count += 1
                        self.trades_executed_today.append(result)
                        self.log_event(f"🔹 Trade executed: {full_symbol} {direction} {volume} lots @ {entry_price:.5f}. Ticket: {result.order}")

                elif action.get('action') == 'close_trade' and action.get('trade_id'):
                    if self.trade_executor.close_trade(action['trade_id']):
                        executed_count += 1
        except Exception as e:
            self.log_event(f"❌ Trade execution error: {e}")
        return executed_count

    def generate_cycle_summary(self, executed_trades, account_info, open_positions):
        unrealized_pnl = open_positions['profit'].sum() if not open_positions.empty else 0.0
        self.log_event(f"📊 Cycle {self.cycle_count} Summary:")
        self.log_event(f"   Trades Executed: {executed_trades}")
        self.log_event(f"   Total Trades Today: {len(self.trades_executed_today)}")
        self.log_event(f"   Open Positions: {len(open_positions)}/{self.max_concurrent_positions}")
        self.log_event(f"   Balance: ${account_info.balance:,.2f}, Equity: ${account_info.equity:,.2f}")
        self.log_event(f"   Unrealized P&L: ${unrealized_pnl:+,.2f}")

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

    def validate_and_correct_currency_pair(self, pair):
        """
        Validate and correct currency pair format.
        Handles inverted pairs and common mistakes.
        """
        valid_pairs = self.config['trading']['symbols'].split(',')
        clean_pair = pair.replace('-ECN', '').replace('/', '').upper()

        if clean_pair in valid_pairs:
            return clean_pair

        if len(clean_pair) >= 6:
            base = clean_pair[:3]
            quote = clean_pair[3:6]

            # Check for direct inversion
            inverted = quote + base
            if inverted in valid_pairs:
                self.log_event(f"⚠️ Correcting inverted pair: {clean_pair} -> {inverted}")
                return inverted

            # Check for common inversions not caught by simple reversal
            inversion_map = {
                'CADUSD': 'USDCAD', 'CHFUSD': 'USDCHF', 'JPYUSD': 'USDJPY',
                'USDEUR': 'EURUSD', 'USDGBP': 'GBPUSD', 'USDAUD': 'AUDUSD'
            }
            if clean_pair in inversion_map:
                corrected = inversion_map[clean_pair]
                self.log_event(f"⚠️ Correcting known inverted pair: {clean_pair} -> {corrected}")
                return corrected

        self.log_event(f"❌ Invalid currency pair: {pair}")
        return None

    def calculate_ufo_entry_price(self, symbol, direction, ufo_data):
        """
        Calculate optimal entry price based on UFO methodology and currency strength.
        """
        try:
            # Get the current market price
            tick_info = mt5.symbol_info_tick(symbol)
            if not tick_info:
                self.log_event(f"⚠️ Could not get tick info for {symbol}, using fallback price logic.")
                # Fallback to historical rates if tick info is not available
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 1)
                if rates is None or len(rates) == 0:
                    return 1.0850 if 'EUR' in symbol else 143.50 if 'JPY' in symbol else 1.2650
                base_price = rates[0]['close']
            else:
                base_price = tick_info.last

            if ufo_data:
                clean_symbol = symbol.replace('-ECN', '')
                if len(clean_symbol) >= 6:
                    base_currency = clean_symbol[:3]
                    quote_currency = clean_symbol[3:6]

                    primary_tf = mt5.TIMEFRAME_M5
                    raw_ufo_data = ufo_data.get('raw_data', ufo_data)

                    if primary_tf in raw_ufo_data:
                        strength_data = raw_ufo_data[primary_tf]
                        base_strength = 0.0
                        quote_strength = 0.0

                        # Handle both DataFrame and dict formats
                        if hasattr(strength_data, 'columns'):
                            if base_currency in strength_data.columns:
                                base_strength = strength_data[base_currency].iloc[-1]
                            if quote_currency in strength_data.columns:
                                quote_strength = strength_data[quote_currency].iloc[-1]
                        else: # Handle dict format
                            if base_currency in strength_data:
                                base_strength = strength_data[base_currency][-1]
                            if quote_currency in strength_data:
                                quote_strength = strength_data[quote_currency][-1]

                        strength_diff = base_strength - quote_strength
                        price_adjustment = 0.0

                        # UFO-based price adjustment: seek better entry if strength is confirmed
                        if abs(strength_diff) > 1.0:
                            # For a BUY, a strong base currency allows for a slightly lower entry
                            if direction == 'BUY' and strength_diff > 0:
                                price_adjustment = -base_price * 0.0002  # 2 pips adjustment
                            # For a SELL, a weak base currency allows for a slightly higher entry
                            elif direction == 'SELL' and strength_diff < 0:
                                price_adjustment = base_price * 0.0002   # 2 pips adjustment

                        optimal_price = base_price + price_adjustment
                        return max(optimal_price, base_price * 0.95) # Safety net

            return base_price

        except Exception as e:
            self.log_event(f"⚠️ Error calculating UFO entry price for {symbol}: {e}")
            return 1.0850 if 'EUR' in symbol else 143.50 if 'JPY' in symbol else 1.2650

    def check_portfolio_equity_stop(self, account_info):
        if not account_info or account_info.balance <= 0: return False, "N/A"
        drawdown = ((account_info.equity - account_info.balance) / account_info.balance) * 100
        if drawdown <= self.portfolio_equity_stop:
            return True, f"Portfolio stop breached: {drawdown:.2f}%"
        return False, f"Portfolio healthy: {drawdown:.2f}% drawdown"

    def check_multi_timeframe_coherence(self, ufo_data):
        """
        Check if currency strength is consistent across timeframes.
        This is a placeholder, as the main logic is now in UfoCalculator.
        This method can be used for additional logging or checks if needed.
        """
        # The core logic is now in `ufo_calculator.detect_timeframe_coherence`
        # This method can be kept for compatibility or extended for more specific live checks.
        return [] # Returning empty list as primary check is done during indicator calculation.

    def analyze_ufo_exit_signals(self, current_ufo_data, previous_ufo_data):
        """Analyze UFO data for exit signals based on significant currency strength changes."""
        exit_signals = []
        if not previous_ufo_data:
            return exit_signals

        current_raw = current_ufo_data.get('raw_data', {})
        previous_raw = previous_ufo_data.get('raw_data', {})

        for timeframe, current_strengths in current_raw.items():
            if timeframe not in previous_raw:
                continue
            previous_strengths = previous_raw[timeframe]

            currency_list = list(current_strengths.columns) if hasattr(current_strengths, 'columns') else list(current_strengths.keys())

            for currency in currency_list:
                if currency not in previous_strengths:
                    continue

                if hasattr(current_strengths, 'iloc'):
                    current_val = current_strengths[currency].iloc[-1]
                    prev_series = previous_strengths[currency].iloc[-5:]
                    avg_previous = prev_series.mean()
                else: # dict format
                    current_val = current_strengths[currency][-1]
                    prev_series = previous_strengths[currency][-5:]
                    avg_previous = sum(prev_series) / len(prev_series)

                # Signal on significant strength reversal
                if abs(current_val - avg_previous) > 2.0:
                    change_dir = "strengthening" if current_val > avg_previous else "weakening"
                    exit_signals.append({
                        'currency': currency,
                        'reason': f"{currency} is rapidly {change_dir} on {timeframe}",
                        'change': current_val - avg_previous
                    })
        return exit_signals

    def close_affected_positions(self, exit_signals):
        """Close positions affected by strong UFO exit signals."""
        positions_closed = 0
        currencies_to_close = {signal['currency'] for signal in exit_signals}

        open_positions = self.portfolio_manager.get_positions()
        if open_positions.empty:
            return 0

        for _, position in open_positions.iterrows():
            symbol = position['symbol'].replace('-ECN', '')
            base, quote = symbol[:3], symbol[3:6]

            if base in currencies_to_close or quote in currencies_to_close:
                self.log_event(f"🚨 Closing {symbol} due to strong exit signals for {base} or {quote}")
                if self.trade_executor.close_trade(position['ticket']):
                    positions_closed += 1
        return positions_closed

    def continuous_position_monitoring(self):
        """Perform continuous monitoring and dynamic reinforcement between trading cycles."""
        now = datetime.now()
        if self.last_monitoring_time and (now - self.last_monitoring_time).total_seconds() / 60 < self.position_update_frequency_minutes:
            return
        self.last_monitoring_time = now

        try:
            open_positions = self.portfolio_manager.get_positions()
            if open_positions.empty:
                return

            # Force portfolio value update and position management
            self.update_portfolio_value(force_update=True)

            current_market_data = self.get_real_time_market_data_for_positions(open_positions)

            # Dynamic Reinforcement Engine checks
            if self.dynamic_reinforcement_engine.enabled and self.dynamic_reinforcement_engine.should_check_reinforcement(now):
                market_events = self.dynamic_reinforcement_engine.detect_market_events(
                    open_positions, current_market_data, self.previous_ufo_data
                )
                if market_events:
                    self.log_event(f"🎯 Dynamic Reinforcement: {len(market_events)} market events detected")
                    for event in market_events:
                        pos = event.get('position')
                        if pos:
                            plan, msg = self.dynamic_reinforcement_engine.calculate_dynamic_reinforcement(
                                pos, event, current_market_data, self.previous_ufo_data
                            )
                            if plan:
                                self.log_event(f"  ⚡ {pos['symbol']}: {msg}")
                                self.execute_dynamic_reinforcement(pos, plan)

            # UFO-Based Reinforcement checks
            if self.previous_ufo_data:
                for _, position_series in open_positions.iterrows():
                    position_dict = position_series.to_dict()
                    should_reinforce, reason, plan = self.ufo_engine.should_reinforce_position(
                        position_dict, self.previous_ufo_data, current_market_data
                    )
                    if should_reinforce and plan:
                        self.log_event(f"  🛸 UFO reinforcement suggestion: {position_dict['symbol']} - {reason}")
                        self.execute_dynamic_reinforcement(position_dict, plan)

        except Exception as e:
            self.log_event(f"❌ Error in continuous position monitoring: {e}")

    def get_real_time_market_data_for_positions(self, open_positions):
        """Collect real-time market data for all open positions."""
        current_market_data = {}
        if open_positions.empty:
            return current_market_data

        symbols_to_fetch = open_positions['symbol'].unique()
        for symbol in symbols_to_fetch:
            tick_info = self.mt5_collector.get_tick_info(symbol)
            if tick_info:
                current_market_data[symbol] = {
                    'close': tick_info.last,
                    'ask': tick_info.ask,
                    'bid': tick_info.bid,
                    'spread': tick_info.ask - tick_info.bid,
                    'timestamp': datetime.now()
                }
            else:
                self.log_event(f"⚠️ Could not get tick info for {symbol}, using fallback price.")
                # Basic fallback for live environment
                fallback_price = 1.0
                current_market_data[symbol] = {
                    'close': fallback_price, 'ask': fallback_price + 0.0001, 'bid': fallback_price,
                    'spread': 0.0001, 'timestamp': datetime.now()
                }
        return current_market_data

    def execute_dynamic_reinforcement(self, position, plan):
        """Execute a dynamic reinforcement trade."""
        try:
            # Determine trade type from the original position
            trade_type = mt5.ORDER_TYPE_BUY if position['type'] == 0 else mt5.ORDER_TYPE_SELL

            result = self.trade_executor.execute_ufo_trade(
                symbol=position['symbol'],
                trade_type=trade_type,
                volume=plan['additional_lots'],
                comment=f"Dynamic {plan['type']} for ticket {position['ticket']}"
            )
            if result:
                self.dynamic_reinforcement_engine.record_reinforcement(position, plan)
                self.log_event(f"    ✅ Dynamic reinforcement executed for {position['symbol']}.")
        except Exception as e:
            self.log_event(f"    ❌ Failed to execute dynamic reinforcement: {e}")

    def generate_final_summary(self):
        self.log_event("\n" + "="*80)
        self.log_event("🎯 TRADING DAY CONCLUDED")
        self.log_event("="*80)
        self.log_event(f"📅 Date: {datetime.now().strftime('%A, %B %d, %Y')}")
        self.log_event(f"⏰ Total Cycles: {self.cycle_count}")
        self.log_event(f"💼 Total Trades Executed: {len(self.trades_executed_today)}")

        if self.trades_executed_today:
            self.log_event("\n📈 EXECUTED TRADES SUMMARY:")
            for i, trade in enumerate(self.trades_executed_today, 1):
                self.log_event(f"  {i}. {trade.symbol} {trade.type} {trade.volume} @ {trade.price_open:.5f} ({trade.comment})")

        account_info = self.portfolio_manager.get_account_info()
        if account_info:
            self.log_event(f"💰 Final Balance: ${account_info.balance:,.2f}")
            self.log_event(f"💰 Final Equity: ${account_info.equity:,.2f}")

    def save_daily_report(self):
        report_filename = f"live_trade_report_{datetime.now().strftime('%Y%m%d')}.txt"
        report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'reports', report_filename)
        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("UFO FOREX AGENT - DAILY TRADING REPORT\n")
            f.write("=" * 60 + "\n")
            for log_entry in self.simulation_log:
                f.write(log_entry + "\n")

        self.log_event(f"\n📁 Daily report saved: {report_path}")
