import time
import pandas as pd
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
        
        # Read cycle period from config (default 40 minutes if not specified)
        cycle_period_raw = config['trading'].get('cycle_period_minutes', '40')
        self.cycle_period_minutes = parse_config_value(cycle_period_raw, 40)
        self.cycle_period_seconds = self.cycle_period_minutes * 60
        
        # Continuous monitoring variables
        self.last_position_update = None
        self.position_update_frequency_seconds = 300  # Update positions every 5 minutes
        self.continuous_monitoring_enabled = True
        self.portfolio_history = []  # Track portfolio value over time
        
        self.llm_client = LLMClient(api_key=config['openrouter']['api_key'])

        self.mt5_collector = MT5DataCollector(
            login=config['mt5']['login'],
            password=config['mt5']['password'],
            server=config['mt5']['server'],
            path=config['mt5']['path']
        )

        self.trade_executor = TradeExecutor(self.mt5_collector, self.config)
        self.ufo_engine = UFOTradingEngine(config)

        self.agents = {
            "data_analyst": DataAnalystAgent("DataAnalyst", self.mt5_collector),
            "researcher": MarketResearcherAgent("MarketResearcher", self.llm_client),
            "trader": TraderAgent("Trader", self.llm_client, self.mt5_collector),
            "risk_manager": RiskManagerAgent("RiskManager", self.llm_client, self.mt5_collector, self.config),
            "fund_manager": FundManagerAgent("FundManager", self.llm_client)
        }

        self.communication_bus = CommunicationBus()
        self.ufo_calculator = UfoCalculator(config['trading']['currencies'].split(','))

        # Initialize Dynamic Reinforcement Engine
        self.dynamic_reinforcement_engine = DynamicReinforcementEngine(config)
        if self.dynamic_reinforcement_engine.enabled:
            print("✅ Dynamic Reinforcement Engine enabled")
        else:
            print("⚠️ Dynamic Reinforcement Engine disabled")

    def run(self):
        """
        Runs the live trading loop with UFO methodology.
        """
        while True:
            try:
                # 1. Check if we're in an active trading session
                if not self.ufo_engine.is_active_session():
                    print(f"Outside active trading session. Waiting 5 minutes...")
                    time.sleep(300)
                    continue

                # 2. Data Collection
                price_data = self.collect_market_data()

                # 3. UFO Calculation
                ufo_data = self.calculate_ufo_indicators(price_data)

                # 4. First Priority: UFO Portfolio Management (NO individual stops!)
                try:
                    open_positions = self.agents['risk_manager'].portfolio_manager.get_positions()
                    if open_positions is not None and len(open_positions) > 0:
                        print(f"\n--- UFO Portfolio Management: {len(open_positions)} positions ---")
                        
                        # 🎯 UFO METHODOLOGY: Check portfolio-level stop FIRST (2-3% of account)
                        account_info = self.mt5_collector.connect() and mt5.account_info()
                        if account_info:
                            portfolio_stop_breached, stop_reason = self.ufo_engine.check_portfolio_equity_stop(
                                account_info.balance, account_info.equity
                            )
                            if portfolio_stop_breached:
                                print(f"🚨 UFO PORTFOLIO STOP TRIGGERED: {stop_reason}")
                                print("🚨 Closing ALL positions - no individual stops needed!")
                                for position in open_positions:
                                    self.trade_executor.close_trade(position.ticket)
                                print("🚨 All positions closed. Waiting 5 minutes before resuming...")
                                time.sleep(300)
                                continue
                        
                        # Check session end timing
                        should_close, close_reason = self.ufo_engine.should_close_for_session_end()
                        if should_close:
                            print(f"🌅 UFO SESSION END: {close_reason}")
                            print("🌅 Closing all positions for session end")
                            for position in open_positions:
                                self.trade_executor.close_trade(position.ticket)
                            time.sleep(300)
                            continue
                        
                        # UFO compensation/reinforcement logic (if portfolio is healthy)
                        current_market_data = self.get_real_time_market_data_for_positions(open_positions)
                        for position in open_positions:
                            should_reinforce, reason, reinforcement_plan = self.ufo_engine.should_reinforce_position(
                                position, ufo_data, current_market_data
                            )
                            
                            if should_reinforce:
                                print(f"🔧 UFO Compensation: {reason}")
                                success, result_msg = self.ufo_engine.execute_compensation_trade(
                                    position, reinforcement_plan, self.trade_executor
                                )
                                if success:
                                    print(f"✅ {result_msg}")
                                else:
                                    print(f"❌ Compensation failed: {result_msg}")
                            elif "close position" in reason:
                                print(f"📊 UFO Analysis: Closing {position.ticket} - {reason}")
                                self.trade_executor.close_trade(position.ticket)
                            else:
                                print(f"📈 Position {position.ticket} - {reason}")
                            
                except Exception as e:
                    print(f"Error managing existing positions: {e}")
                    open_positions = None

                # UFO: Analyze exit signals based on currency strength changes
                if hasattr(self, 'previous_ufo_data') and ufo_data:
                    exit_signals = self.analyze_ufo_exit_signals(ufo_data, self.previous_ufo_data)
                    if exit_signals:
                        print(f"📈 UFO Exit Signals detected: {len(exit_signals)} currency changes")
                        for signal in exit_signals:
                            print(f"⚠️ {signal['reason']} (change: {signal['change']:.2f})")

                        # Enhanced auto-close on strong signals
                        if len(exit_signals) >= 3:
                            print("🚨 STRONG EXIT SIGNALS detected: Auto-closing positions")
                            positions_closed = self.close_affected_positions(exit_signals)
                            print(f"🚨 Auto-closed {positions_closed} positions based on strong exit signals")

                # Store UFO data for next cycle comparison
                if ufo_data:
                    self.previous_ufo_data = ufo_data

                # 5. Agentic Workflow for new trade decisions
                economic_events = self.get_economic_events()
                
                # Get current open positions after management
                try:
                    open_positions = self.agents['risk_manager'].portfolio_manager.get_positions()
                except:
                    open_positions = None
                    
                research_result = self.agents['researcher'].execute(ufo_data, economic_events)
                
                # Pass diversification config to TraderAgent
                diversification_config = {
                    'min_positions_for_session': self.ufo_engine.min_positions_for_session,
                    'target_positions_when_available': self.ufo_engine.target_positions_when_available,
                    'max_concurrent_positions': self.ufo_engine.max_concurrent_positions
                }
                
                trade_decision_str = self.agents['trader'].execute(
                    research_result['consensus'], 
                    open_positions,
                    diversification_config=diversification_config
                )

                risk_assessment = self.agents['risk_manager'].execute(trade_decision_str)

                if risk_assessment['portfolio_risk_status'] == "STOP_LOSS_BREACHED":
                    print("!!! EQUITY STOP LOSS BREACHED. CEASING ALL TRADING. !!!")
                    break

                authorization = self.agents['fund_manager'].execute(trade_decision_str, risk_assessment)

                # 6. Output with Diversification Status
                position_count = len(open_positions) if open_positions is not None and hasattr(open_positions, '__len__') else 0
                diversification_status = f"📊 Portfolio Diversification: {position_count}/{self.ufo_engine.max_concurrent_positions} positions"
                
                if position_count < self.ufo_engine.min_positions_for_session:
                    diversification_status += " ⚠️ Below minimum"
                elif position_count >= self.ufo_engine.target_positions_when_available:
                    diversification_status += " ✅ Well diversified"
                else:
                    diversification_status += " 📈 Building diversification"
                
                print("\n--- Live Trading Cycle ---")
                print(f"Timestamp: {pd.Timestamp.now()}")
                print(diversification_status)
                print(f"Research Consensus: {research_result['consensus']}")
                print(f"Trade Decision: {trade_decision_str}")
                print(f"Risk Assessment: {risk_assessment}")
                print(f"Final Authorization: {authorization}")

                # 7. UFO-based Trade Execution (only if conditions are met)
                should_execute = "APPROVE" in authorization.upper()
                
                # If Fund Manager rejected due to high risk, check if we can auto-scale
                if not should_execute and "REJECT" in authorization.upper():
                    if "risk" in authorization.lower() and "exceed" in authorization.lower():
                        print("🔄 Fund Manager rejected due to high risk - will auto-scale and execute anyway")
                        should_execute = True
                
                if should_execute:
                    # Enhanced UFO engine check with diversification parameters
                    should_trade, trade_reason = self.ufo_engine.should_open_new_trades(
                        current_positions=open_positions, 
                        portfolio_status={'balance': account_info.balance, 'equity': account_info.equity} if account_info else None,
                        ufo_data=ufo_data
                    )
                    
                    if not should_trade:
                        print(f"UFO Engine: {trade_reason}")
                    else:
                        print(f"🎯 UFO Engine: {trade_reason}")
                        try:
                            json_match = re.search(r'{.*}', trade_decision_str, re.DOTALL)
                            if not json_match:
                                print("No JSON object found in the LLM decision.")
                            else:
                                # Clean JSON by removing JavaScript-style comments
                                json_str = json_match.group(0)
                                # Remove single-line comments (// ...)
                                json_str = re.sub(r'//.*?\n', '\n', json_str)
                                # Remove trailing commas before closing brackets
                                json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
                                
                                parsed_data = json.loads(json_str)
                                print(f"Parsed trade data: {parsed_data}")
                                
                                # Handle different JSON formats from LLM
                                actions_list = []
                                if 'actions' in parsed_data:
                                    actions_list = parsed_data['actions']
                                elif 'trade_plan' in parsed_data:
                                    # Handle trade_plan format (already contains actions)
                                    actions_list = parsed_data['trade_plan']
                                elif 'trades' in parsed_data:
                                    # Convert simple trades format to actions format
                                    for trade in parsed_data['trades']:
                                        actions_list.append({
                                            'action': 'new_trade',
                                            'currency_pair': trade['currency_pair'],
                                            'direction': trade['direction'].upper(),
                                            'volume': 0.1,  # Default volume
                                            'symbol': trade['currency_pair']
                                        })
                                
                                # Apply automatic risk scaling to respect 4.5% portfolio limit
                                total_original_risk = 0.0
                                for action in actions_list:
                                    if action.get('action') == 'new_trade':
                                        volume = action.get('volume') or action.get('lot_size', 0.1)
                                        # Estimate risk per trade (simplified: volume * 1% per 0.1 lots)
                                        estimated_risk = (volume / 0.1) * 1.0  # Rough estimate
                                        total_original_risk += estimated_risk
                                
                                # Calculate scaling factor if needed
                                max_portfolio_risk = 4.5  # 4.5% to avoid hitting 5% limit
                                risk_scale_factor = 1.0
                                
                                if total_original_risk > max_portfolio_risk:
                                    risk_scale_factor = max_portfolio_risk / total_original_risk
                                    print(f"⚠️ Scaling down positions: Original risk {total_original_risk:.1f}% → {max_portfolio_risk}%")
                                    print(f"📉 Risk scale factor: {risk_scale_factor:.3f}")
                                
                                for action in actions_list:
                                    if action.get('action') == 'new_trade':
                                        symbol = action.get('symbol') or action.get('currency_pair', '')
                                        direction = action.get('direction', '').upper()
                                        volume = action.get('volume') or action.get('lot_size', 0.1)

                                        # Validate and correct currency pair format
                                        base_symbol = symbol.replace("/", "")
                                        corrected_symbol = self.validate_and_correct_currency_pair(base_symbol)

                                        if corrected_symbol is None:
                                            print(f"⚠️ Skipping invalid currency pair: {symbol}")
                                            continue  # Skip this trade

                                        # Also handle direction inversion if pair was inverted
                                        if base_symbol != corrected_symbol and len(base_symbol) >= 6:
                                            # Check if we need to invert the direction
                                            original_base = base_symbol[:3]
                                            corrected_base = corrected_symbol[:3]
                                            if original_base != corrected_base:
                                                # Pair was inverted, so invert the direction
                                                direction = 'SELL' if direction == 'BUY' else 'BUY'
                                                print(f"⚠️ Direction inverted due to pair correction: {direction}")

                                        # Add symbol suffix if it doesn't exist
                                        suffix = self.config['mt5'].get('symbol_suffix', '')
                                        if not corrected_symbol.endswith(suffix):
                                            full_symbol = corrected_symbol + suffix
                                        else:
                                            full_symbol = corrected_symbol

                                        # Convert direction to MT5 trade type
                                        if direction == 'BUY':
                                            trade_type = mt5.ORDER_TYPE_BUY
                                        elif direction == 'SELL':
                                            trade_type = mt5.ORDER_TYPE_SELL
                                        else:
                                            print(f"Invalid direction: {direction}")
                                            continue

                                        # Get volume and apply scaling
                                        scaled_volume = round(volume * risk_scale_factor, 2)
                                        
                                        # Ensure minimum volume (0.01 lots)
                                        final_volume = max(scaled_volume, 0.01)

                                        # Execute trade without fixed SL/TP (UFO methodology)
                                        success = self.trade_executor.execute_ufo_trade(
                                            symbol=full_symbol,
                                            trade_type=trade_type,
                                            volume=final_volume,
                                            comment=action.get('comment', 'UFO Trade (Auto-Scaled)')
                                        )
                                        
                                        if success:
                                            risk_pct = (final_volume / 0.1) * 1.0  # Rough risk estimate
                                            print(f"✅ UFO Trade executed: {full_symbol} {direction} {final_volume} lots (~{risk_pct:.1f}% risk)")
                                        else:
                                            print(f"❌ Failed to execute: {full_symbol} {direction}")

                                    elif action.get('action') == 'close_trade':
                                        trade_id = action.get('trade_id')
                                        if trade_id:
                                            self.trade_executor.close_trade(trade_id)
                                        continue

                        except Exception as e:
                            print(f"Error during UFO trade execution: {e}")

                # Perform additional position updates between cycles (every 5 minutes)
                next_cycle_time = datetime.now() + pd.Timedelta(minutes=self.cycle_period_minutes)
                monitoring_time = datetime.now() + pd.Timedelta(seconds=self.position_update_frequency_seconds)
                
                while monitoring_time < next_cycle_time:
                    open_positions = self.agents['risk_manager'].portfolio_manager.get_positions()
                    if open_positions is not None and not open_positions.empty:
                        self.continuous_position_monitoring()

                    # Wait for the next monitoring interval
                    time.sleep(self.position_update_frequency_seconds)
                    monitoring_time = datetime.now() + pd.Timedelta(seconds=self.position_update_frequency_seconds)

            except KeyboardInterrupt:
                print("\nTrading interrupted by user. Exiting...")
                break
            except Exception as e:
                print(f"Error in trading cycle: {e}")
                print("Waiting 60 seconds before retrying...")
                time.sleep(60)

    def get_real_time_market_data_for_positions(self, open_positions):
        """
        Collect real-time market data for all open positions
        This replaces the empty current_market_data = {} with actual price data
        """
        current_market_data = {}
        
        if open_positions is None or len(open_positions) == 0:
            return current_market_data
            
        try:
            # Connect to MT5 to get current prices
            if not self.mt5_collector.connect():
                print("⚠️ Failed to connect to MT5 for market data collection")
                return current_market_data
                
            # Extract unique symbols from positions
            symbols_to_fetch = set()
            for _, position in open_positions.iterrows():
                symbols_to_fetch.add(position['symbol'])
            
            # Get current tick data for each symbol
            for symbol in symbols_to_fetch:
                try:
                    tick = mt5.symbol_info_tick(symbol)
                    if tick is not None:
                        current_market_data[symbol] = {
                            'close': tick.bid,  # Use bid for current price
                            'ask': tick.ask,
                            'bid': tick.bid,
                            'spread': tick.ask - tick.bid,
                            'timestamp': pd.Timestamp.now()
                        }
                        print(f"📊 Real-time data: {symbol} @ {tick.bid:.5f} (spread: {(tick.ask - tick.bid):.5f})")
                    else:
                        print(f"⚠️ No tick data available for {symbol}")
                        # Fallback: try to get recent bar data
                        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 1)
                        if rates is not None and len(rates) > 0:
                            current_market_data[symbol] = {
                                'close': rates[0]['close'],
                                'ask': rates[0]['close'] + 0.0001,  # Estimated spread
                                'bid': rates[0]['close'],
                                'spread': 0.0001,
                                'timestamp': pd.Timestamp.now()
                            }
                            print(f"📊 Fallback data: {symbol} @ {rates[0]['close']:.5f} (from M1 bar)")
                        
                except Exception as e:
                    print(f"❌ Error getting market data for {symbol}: {e}")
                    continue
            
            self.mt5_collector.disconnect()
            print(f"✅ Collected real-time market data for {len(current_market_data)} symbols")
            
        except Exception as e:
            print(f"❌ Error in market data collection: {e}")
            
        return current_market_data
    
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

                data = self.agents['data_analyst'].execute({
                    'source': 'mt5',
                    'symbol': symbol,
                    'timeframes': timeframes,
                    'num_bars': timeframe_bars
                })
                all_data[symbol] = data

            print(f"✅ Collected data for {len(all_data)} symbols")
            return all_data
        except Exception as e:
            print(f"❌ Data collection error: {e}")
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

            print(f"✅ Enhanced UFO analysis completed for {len(ufo_data)} timeframes")
            return enhanced_ufo_data
        except Exception as e:
            print(f"❌ UFO calculation error: {e}")
            return None

    def _log_enhanced_analysis(self, oscillation_analysis, uncertainty_metrics, coherence_analysis):
        """Log enhanced UFO analysis results"""
        try:
            # Log market state summary across timeframes
            for timeframe, metrics in uncertainty_metrics.items():
                overall_state = metrics.get('overall_state', 'unknown')
                confidence = metrics.get('confidence_level', 'unknown')
                scaling = metrics.get('recommended_position_scaling', 1.0)

                print(f"🔍 {timeframe}: {overall_state} (confidence: {confidence}, scaling: {scaling:.2f})")

            # Log coherence insights
            strong_coherence_count = sum(1 for curr_data in coherence_analysis.values()
                                       if curr_data.get('coherence_level') == 'strong')
            total_currencies = len(coherence_analysis)

            if total_currencies > 0:
                coherence_ratio = strong_coherence_count / total_currencies
                print(f"📊 Timeframe Coherence: {strong_coherence_count}/{total_currencies} currencies show strong coherence ({coherence_ratio:.1%})")

            # Log mean reversion opportunities
            mean_reversion_signals = 0
            for tf_data in oscillation_analysis.values():
                mean_reversion_signals += sum(1 for curr_data in tf_data.values()
                                             if curr_data.get('mean_reversion_signal', False))

            if mean_reversion_signals > 0:
                print(f"🔄 Mean Reversion Signals: {mean_reversion_signals} detected across timeframes")

        except Exception as e:
            print(f"⚠️ Error logging enhanced analysis: {e}")

    def get_economic_events(self):
        """Get economic calendar events for simulation"""
        try:
            # Get raw events from cache
            raw_events = self.agents['data_analyst'].execute({'source': 'economic_calendar'})

            if raw_events is None or raw_events.empty:
                print("❌ No economic calendar data available")
                return pd.DataFrame()

            # Process events for simulation date with timezone conversion
            processed_events = self.process_economic_events(raw_events)
            event_count = len(processed_events) if processed_events is not None and not processed_events.empty else 0

            print(f"✅ Retrieved {event_count} economic events for today")
            return processed_events

        except Exception as e:
            print(f"❌ Economic calendar error: {e}")
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
                print("❌ No date column found in economic calendar data")
                return pd.DataFrame()

            # Filter events for our simulation date (August 4th, 2025)
            today = datetime.now(pytz.utc).date()

            # Filter events that occur on our simulation date
            daily_events = raw_events[
                raw_events['datetime'].dt.date == today
            ].copy()

            if daily_events.empty:
                print(f"ℹ️ No economic events found for {today}")
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
                print(f"⚠️ {len(high_impact_events)} HIGH IMPACT events scheduled for trading day:")
                for _, event in high_impact_events.iterrows():
                    event_time = f"{event['gmt_hour']:02d}:{event['gmt_minute']:02d}"
                    print(f"  📅 {event_time} GMT: {event['country']} {event['title']}")

            return daily_events

        except Exception as e:
            print(f"❌ Error processing simulation economic events: {e}")
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

        open_positions = self.agents['risk_manager'].portfolio_manager.get_positions()
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
                    print(f"🚨 Marking {symbol} for closure due to {base_currency}/{quote_currency} exit signals")

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
            print(f"⚠️ Error getting historical price for {symbol}: {e}")
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
                print(f"⚠️ Correcting inverted pair: {clean_pair} -> {inverted}")
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
                print(f"⚠️ Correcting known inverted pair: {clean_pair} -> {corrected}")
                return corrected

        # If we can't fix it, log error and return None
        print(f"❌ Invalid currency pair: {pair} (cleaned: {clean_pair})")
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
            print(f"⚠️ Error calculating UFO entry price for {symbol}: {e}")
            # Ultimate fallback
            return 1.0850 if 'EUR' in symbol else 143.50 if 'JPY' in symbol else 1.2650

    def continuous_position_monitoring(self):
        """Perform continuous position monitoring between trading cycles"""
        try:
            open_positions = self.agents['risk_manager'].portfolio_manager.get_positions()
            if open_positions is None or open_positions.empty:
                return

            print(f"--- Continuous Monitoring ({datetime.now().strftime('%H:%M:%S')}) ---")

            # Check for positions with excessive unrealized losses during monitoring
            high_risk_positions = []
            for index, position in open_positions.iterrows():
                if position.profit < -75:  # High risk threshold
                    high_risk_positions.append(position)

            if high_risk_positions:
                print(f"🚨 Monitoring alert: {len(high_risk_positions)} positions with high unrealized losses")
                for pos in high_risk_positions[:3]:  # Log top 3
                    print(f"  ⚠️ {pos['symbol']}: P&L ${pos.profit:.2f}")

            # Enhanced Dynamic Reinforcement monitoring
            if self.dynamic_reinforcement_engine.enabled and self.dynamic_reinforcement_engine.should_check_reinforcement(datetime.now()):
                current_market_data = self.get_real_time_market_data_for_positions(open_positions)

                # Detect market events that trigger reinforcement
                market_events = self.dynamic_reinforcement_engine.detect_market_events(
                    open_positions,
                    current_market_data,
                    getattr(self, 'previous_ufo_data', None)
                )

                if market_events:
                    print(f"🎯 Dynamic Reinforcement: {len(market_events)} market events detected")

                    # Process each event for reinforcement
                    for event in market_events:
                        position = event.get('position')
                        if position is not None:
                            # Calculate dynamic reinforcement for this event
                            reinforcement_plan, message = self.dynamic_reinforcement_engine.calculate_dynamic_reinforcement(
                                position,
                                event,
                                current_market_data,
                                getattr(self, 'previous_ufo_data', None)
                            )

                            if reinforcement_plan:
                                print(f"  ⚡ {event['type']}: {position['symbol']} - {message}")
                                print(f"    📊 Reinforcement: {reinforcement_plan['additional_lots']:.2f} lots")

                                # Execute reinforcement
                                self.execute_dynamic_reinforcement(position, reinforcement_plan)
                            else:
                                print(f"  ⏸️ {position['symbol']}: {message}")

                # Also check UFO-based reinforcement for compatibility
                if hasattr(self, 'previous_ufo_data'):
                    for index, position in open_positions.iterrows():
                        # Check if UFO engine also suggests reinforcement
                        should_reinforce, reason, plan = self.ufo_engine.should_reinforce_position(
                            position,
                            self.previous_ufo_data,
                            current_market_data
                        )
                        if should_reinforce and plan:
                            print(f"  🛸 UFO reinforcement suggestion: {position['symbol']} - {reason}")

        except Exception as e:
            print(f"❌ Error in continuous position monitoring: {e}")

    def execute_dynamic_reinforcement(self, position, reinforcement_plan):
        """Execute dynamic reinforcement trade"""
        try:
            # Convert direction to MT5 trade type
            if position['type'] == 0: # BUY
                trade_type = mt5.ORDER_TYPE_BUY
            else: # SELL
                trade_type = mt5.ORDER_TYPE_SELL

            # Execute the reinforcement trade
            success = self.trade_executor.execute_ufo_trade(
                symbol=position['symbol'],
                trade_type=trade_type,
                volume=reinforcement_plan['additional_lots'],
                comment=f'Dynamic {reinforcement_plan["type"]}'
            )

            if success:
                # Record in dynamic reinforcement engine
                self.dynamic_reinforcement_engine.record_reinforcement(position, reinforcement_plan)
                print(f"    ✅ Dynamic reinforcement executed: {position['symbol']} "
                      f"{position['type']} {reinforcement_plan['additional_lots']:.2f} lots")
            else:
                print(f"    ❌ Failed to execute dynamic reinforcement for {position['symbol']}")

        except Exception as e:
            print(f"    ❌ Failed to execute dynamic reinforcement: {e}")

    def check_portfolio_status(self):
        """
        Checks overall portfolio status using UFO methodology.
        """
        try:
            positions = self.agents['risk_manager'].portfolio_manager.get_positions()
            if positions is None or len(positions) == 0:
                return
                
            portfolio_value = self.ufo_engine.calculate_portfolio_synthetic_value()
            print(f"Portfolio synthetic value: {portfolio_value:.2f}%")
            
            if portfolio_value <= -5.0:  # Portfolio stop loss threshold
                print("Portfolio stop loss triggered - closing all positions")
                for position in positions:
                    self.trade_executor.close_trade(position.ticket)
                    
        except Exception as e:
            print(f"Error checking portfolio status: {e}")
