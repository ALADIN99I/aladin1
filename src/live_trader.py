import time
import pandas as pd
import re
import json
from datetime import datetime
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
                base_symbol = 'EURUSD'
                symbol_suffix = self.config['mt5'].get('symbol_suffix', '')
                symbol_with_suffix = base_symbol + symbol_suffix
                timeframes = [mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M15, mt5.TIMEFRAME_H1, mt5.TIMEFRAME_H4, mt5.TIMEFRAME_D1]

                price_data_dict = {}
                timeframe_bars = {
                    mt5.TIMEFRAME_M5: 240,  # 0 GMT to 8 PM = 20 hours = 240 M5 bars
                    mt5.TIMEFRAME_M15: 80,   # 20 hours = 80 M15 bars
                    mt5.TIMEFRAME_H1: 20,    # 20 hours = 20 H1 bars
                    mt5.TIMEFRAME_H4: 120,   # Keep for multi-day analysis
                    mt5.TIMEFRAME_D1: 100    # Keep for multi-day analysis
                }
                # Collect all timeframes in a single call to reduce connections
                data = self.agents['data_analyst'].execute({
                    'source': 'mt5',
                    'symbol': symbol_with_suffix,
                    'timeframes': list(timeframe_bars.keys()),
                    'num_bars': timeframe_bars
                })
                if data:
                    price_data_dict = data

                if not price_data_dict:
                    print("Could not fetch price data. Retrying in 60 seconds...")
                    time.sleep(60)
                    continue

                # 3. UFO Calculation
                incremental_sums_dict = {}
                for timeframe, price_data in price_data_dict.items():
                    variation_data = self.ufo_calculator.calculate_percentage_variation(price_data)
                    incremental_sums_dict[timeframe] = self.ufo_calculator.calculate_incremental_sum(variation_data)

                ufo_data = self.ufo_calculator.generate_ufo_data(incremental_sums_dict)

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

                # 5. Agentic Workflow for new trade decisions
                economic_events = self.agents['data_analyst'].execute({'source': 'economic_calendar'})
                
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
                                    action_type = action.get('action')
                                    if action_type == 'new_trade':
                                        # Get symbol - try multiple field names
                                        symbol = action.get('symbol') or action.get('currency_pair', '')
                                        if not symbol:
                                            print(f"No symbol found in action: {action}")
                                            continue
                                            
                                        # Convert direction to MT5 trade type
                                        direction = action.get('direction', '').upper()
                                        if direction == 'BUY':
                                            trade_type = mt5.ORDER_TYPE_BUY
                                        elif direction == 'SELL':
                                            trade_type = mt5.ORDER_TYPE_SELL
                                        else:
                                            print(f"Invalid direction: {direction}")
                                            continue
                                        
                                        # Get volume and apply scaling
                                        original_volume = action.get('volume') or action.get('lot_size', 0.1)
                                        scaled_volume = round(original_volume * risk_scale_factor, 2)
                                        
                                        # Ensure minimum volume (0.01 lots)
                                        final_volume = max(scaled_volume, 0.01)
                                        
                                        # Add symbol suffix
                                        base_symbol = symbol.replace("/", "")
                                        full_symbol = base_symbol + symbol_suffix
                                        
                                        if risk_scale_factor < 1.0:
                                            print(f"📊 Scaled trade: {full_symbol} {direction} {original_volume}→{final_volume} lots")
                                        else:
                                            print(f"📊 Executing trade: {full_symbol} {direction} {final_volume} lots")
                                        
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
                                            
                                    elif action_type == 'close_trade':
                                        self.trade_executor.close_trade(action['trade_id'])

                        except Exception as e:
                            print(f"Error during UFO trade execution: {e}")

                print(f"\nWaiting for the next trading cycle ({self.cycle_period_minutes} minutes)...")
                time.sleep(self.cycle_period_seconds)  # Uses the configured cycle period
                
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
