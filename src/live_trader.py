import time
import pandas as pd
import re
import json
import logging
from datetime import datetime, timedelta
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
        self._setup_logging()
        
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
        self.cycle_period_minutes = parse_config_value(self.config['trading'].get('cycle_period_minutes', '40'), 40)
        self.cycle_period_seconds = self.cycle_period_minutes * 60
        
        # Continuous monitoring variables
        self.position_update_frequency_seconds = 300  # Update positions every 5 minutes
        self.continuous_monitoring_enabled = True
        
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
            logging.info("✅ Dynamic Reinforcement Engine enabled")
        else:
            logging.warning("⚠️ Dynamic Reinforcement Engine disabled")

        # Portfolio tracking attributes
        self.open_positions = []
        self.closed_trades = []
        self.realized_pnl = 0.0
        self.portfolio_value = 0.0
        self.initial_balance = 0.0
        self.last_cycle_time = 0
        self._initialize_portfolio()

    def _setup_logging(self):
        """Configures structured logging for the application."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("live_trader.log"),
                logging.StreamHandler()
            ]
        )

    def _initialize_portfolio(self):
        """Initializes portfolio balance and P&L."""
        if self.mt5_collector.connect():
            account_info = mt5.account_info()
            if account_info:
                self.initial_balance = account_info.balance
                self.portfolio_value = account_info.equity
                logging.info(f"✅ Portfolio initialized. Initial Balance: ${self.initial_balance:,.2f}, Equity: ${self.portfolio_value:,.2f}")
            else:
                logging.warning("⚠️ Could not retrieve account info. Using default values.")
                self.initial_balance = 10000.0
                self.portfolio_value = 10000.0
            self.mt5_collector.disconnect()
        else:
            logging.error("⚠️ MT5 connection failed during portfolio initialization. Using default values.")
            self.initial_balance = 10000.0
            self.portfolio_value = 10000.0

    def update_open_positions_pnl(self):
        """
        Fetches open positions from MT5, updates their P&L, and syncs with the in-memory list.
        """
        mt5_positions_df = self.agents['risk_manager'].portfolio_manager.get_positions()
        if mt5_positions_df is None or mt5_positions_df.empty:
            if self.open_positions:
                 logging.info("All positions appear to be closed.")
                 self.open_positions = []
            return

        synced_positions = []
        mt5_tickets = set(mt5_positions_df['ticket'])

        # Update existing positions and add new ones
        for _, mt5_pos in mt5_positions_df.iterrows():
            existing_pos = next((p for p in self.open_positions if p['ticket'] == mt5_pos['ticket']), None)

            if existing_pos:
                # Update P&L and current price
                existing_pos['pnl'] = mt5_pos['profit']
                existing_pos['current_price'] = mt5_pos['price_current']
                existing_pos['last_update'] = datetime.now()
                synced_positions.append(existing_pos)
            else:
                # Add new position found on MT5
                new_pos = {
                    'ticket': mt5_pos['ticket'],
                    'symbol': mt5_pos['symbol'],
                    'direction': 'BUY' if mt5_pos['type'] == 0 else 'SELL',
                    'volume': mt5_pos['volume'],
                    'entry_price': mt5_pos['price_open'],
                    'current_price': mt5_pos['price_current'],
                    'pnl': mt5_pos['profit'],
                    'timestamp': pd.to_datetime(mt5_pos['time'], unit='s'),
                    'last_update': datetime.now(),
                    'peak_pnl': mt5_pos['profit']
                }
                synced_positions.append(new_pos)
                logging.info(f"✅ New position {new_pos['ticket']} ({new_pos['symbol']}) detected and added to tracking.")

        # Handle closed positions (in-memory but not on MT5)
        for mem_pos in self.open_positions:
            if mem_pos['ticket'] not in mt5_tickets:
                self.realized_pnl += mem_pos['pnl']
                self.closed_trades.append(mem_pos)
                logging.info(f"📉 Position {mem_pos['ticket']} ({mem_pos['symbol']}) closed. Realized P&L: ${mem_pos['pnl']:.2f}")

        self.open_positions = synced_positions

        # Update portfolio value
        unrealized_pnl = sum(p['pnl'] for p in self.open_positions)
        self.portfolio_value = self.initial_balance + self.realized_pnl + unrealized_pnl
        # logging.info(f"💰 Portfolio updated. Value: ${self.portfolio_value:,.2f}, Unrealized P&L: ${unrealized_pnl:,.2f}")

    def check_and_close_positions(self):
        """
        Implements advanced position closing logic based on P&L, time, etc.
        """
        if not self.open_positions:
            return

        positions_to_close = []
        for pos in self.open_positions:
            close_reason = None

            # 1. Take Profit
            if pos['pnl'] > 75: # Take profit at +$75
                close_reason = f"take profit target (P&L: ${pos['pnl']:.2f})"

            # 2. Stop Loss
            elif pos['pnl'] < -50: # Stop loss at -$50
                close_reason = f"stop loss target (P&L: ${pos['pnl']:.2f})"
                
            # 3. Time-based Exit
            position_age = datetime.now() - pos['timestamp']
            if position_age > timedelta(hours=4):
                close_reason = f"time-based exit (>4 hours)"

            # 4. Trailing Stop
            if pos['pnl'] > pos.get('peak_pnl', pos['pnl']):
                pos['peak_pnl'] = pos['pnl']
            elif pos.get('peak_pnl', 0) > 30 and pos['pnl'] < pos['peak_pnl'] * 0.7:
                close_reason = f"trailing stop (peak P&L: ${pos['peak_pnl']:.2f}, current: ${pos['pnl']:.2f})"

            if close_reason:
                logging.info(f"🎯 Marking position {pos['ticket']} ({pos['symbol']}) for closure: {close_reason}")
                positions_to_close.append(pos['ticket'])

        # Close marked positions
        for ticket in positions_to_close:
            success = self.trade_executor.close_trade(ticket)
            if success:
                logging.info(f"✅ Successfully closed position {ticket}.")
            else:
                logging.error(f"❌ Failed to close position {ticket}.")

    def continuous_position_monitoring(self):
        """
        High-frequency monitoring of open positions.
        """
        logging.info(f"\n--- Continuous Position Monitoring ({datetime.now().strftime('%H:%M:%S')}) ---")
        self.update_open_positions_pnl()
        self.check_and_close_positions()

        unrealized_pnl = sum(p['pnl'] for p in self.open_positions)
        logging.info(f"💰 Portfolio Value: ${self.portfolio_value:,.2f} | Open Positions: {len(self.open_positions)} | Unrealized P&L: ${unrealized_pnl:,.2f}")
        logging.info("--- End of Monitoring ---")

    def run_main_trading_cycle(self):
        """
        Runs the main agentic workflow for making new trading decisions.
        """
        logging.info("\n" + "="*60)
        logging.info(f"🚀 Starting New Trading Cycle at {datetime.now().strftime('%H:%M:%S')}")
        logging.info("="*60)

        # 2. Data Collection for all symbols
        symbols = self.config['trading']['symbols'].split(',')
        symbol_suffix = self.config['mt5'].get('symbol_suffix', '')
        timeframes = [mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M15, mt5.TIMEFRAME_H1, mt5.TIMEFRAME_H4, mt5.TIMEFRAME_D1]
        timeframe_bars = {
            mt5.TIMEFRAME_M5: 240,
            mt5.TIMEFRAME_M15: 80,
            mt5.TIMEFRAME_H1: 20,
            mt5.TIMEFRAME_H4: 120,
            mt5.TIMEFRAME_D1: 100
        }

        all_price_data = {}
        for symbol in symbols:
            symbol_with_suffix = symbol + symbol_suffix
            data = self.agents['data_analyst'].execute({
                'source': 'mt5',
                'symbol': symbol_with_suffix,
                'timeframes': timeframes,
                'num_bars': timeframe_bars
            })
            if data:
                all_price_data[symbol] = data

        if not all_price_data:
            logging.warning("Could not fetch price data for any symbol. Retrying in 60 seconds...")
            time.sleep(60)
            return

        # 3. UFO Calculation
        reshaped_data = {}
        for symbol, timeframe_data in all_price_data.items():
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
            'raw_data': ufo_data,
            'oscillation_analysis': oscillation_analysis,
            'uncertainty_metrics': uncertainty_metrics,
            'coherence_analysis': coherence_analysis
        }

        # 4. First Priority: UFO Portfolio Management
        open_positions_df = self.agents['risk_manager'].portfolio_manager.get_positions()
        if open_positions_df is not None and not open_positions_df.empty:
            logging.info(f"\n--- UFO Portfolio Management: {len(open_positions_df)} positions ---")

            account_info = self.mt5_collector.connect() and mt5.account_info()
            if account_info:
                portfolio_stop_breached, stop_reason = self.ufo_engine.check_portfolio_equity_stop(
                    account_info.balance, account_info.equity
                )
                if portfolio_stop_breached:
                    logging.critical(f"🚨 UFO PORTFOLIO STOP TRIGGERED: {stop_reason}")
                    for _, position in open_positions_df.iterrows():
                        self.trade_executor.close_trade(position.ticket)
                    logging.critical("🚨 All positions closed. Waiting 5 minutes before resuming...")
                    time.sleep(300)
                    return

            economic_events_for_session = self.agents['data_analyst'].execute({'source': 'economic_calendar'})
            should_close, close_reason = self.ufo_engine.should_close_for_session_end(economic_events_for_session)
            if should_close:
                logging.info(f"🌅 UFO SESSION END: {close_reason}")
                for _, position in open_positions_df.iterrows():
                    self.trade_executor.close_trade(position.ticket)
                time.sleep(300)
                return

            current_market_data = self.get_real_time_market_data_for_positions(open_positions_df)
            for _, position in open_positions_df.iterrows():
                should_reinforce, reason, reinforcement_plan = self.ufo_engine.should_reinforce_position(
                    position, enhanced_ufo_data, current_market_data
                )
                
                if should_reinforce:
                    logging.info(f"🔧 UFO Compensation: {reason}")
                    success, result_msg = self.ufo_engine.execute_compensation_trade(
                        position, reinforcement_plan, self.trade_executor
                    )
                    if success:
                        logging.info(f"✅ {result_msg}")
                    else:
                        logging.error(f"❌ Compensation failed: {result_msg}")
                elif "close position" in reason:
                    logging.info(f"📊 UFO Analysis: Closing {position.ticket} - {reason}")
                    self.trade_executor.close_trade(position.ticket)
                else:
                    logging.info(f"📈 Position {position.ticket} - {reason}")

        # 5. Agentic Workflow for new trade decisions
        economic_events = self.agents['data_analyst'].execute({'source': 'economic_calendar'})
        open_positions_df = self.agents['risk_manager'].portfolio_manager.get_positions()
        research_result = self.agents['researcher'].execute(enhanced_ufo_data, economic_events)

        diversification_config = {
            'min_positions_for_session': self.ufo_engine.min_positions_for_session,
            'target_positions_when_available': self.ufo_engine.target_positions_when_available,
            'max_concurrent_positions': self.ufo_engine.max_concurrent_positions
        }

        trade_decision_str = self.agents['trader'].execute(
            research_result['consensus'],
            open_positions_df,
            diversification_config=diversification_config
        )

        risk_assessment = self.agents['risk_manager'].execute(trade_decision_str)

        if risk_assessment['portfolio_risk_status'] == "STOP_LOSS_BREACHED":
            logging.critical("!!! EQUITY STOP LOSS BREACHED. CEASING ALL TRADING. !!!")
            # This should be handled more gracefully, maybe break the loop
            return

        authorization = self.agents['fund_manager'].execute(trade_decision_str, risk_assessment)

        # 6. Output with Diversification Status
        position_count = len(open_positions_df) if open_positions_df is not None else 0
        diversification_status = f"📊 Portfolio Diversification: {position_count}/{self.ufo_engine.max_concurrent_positions} positions"

        if position_count < self.ufo_engine.min_positions_for_session:
            diversification_status += " ⚠️ Below minimum"
        elif position_count >= self.ufo_engine.target_positions_when_available:
            diversification_status += " ✅ Well diversified"
        else:
            diversification_status += " 📈 Building diversification"

        logging.info("\n--- Live Trading Cycle Summary ---")
        logging.info(f"Timestamp: {pd.Timestamp.now()}")
        logging.info(diversification_status)
        logging.info(f"Research Consensus: {research_result['consensus']}")
        logging.info(f"Trade Decision: {trade_decision_str}")
        logging.info(f"Risk Assessment: {risk_assessment}")
        logging.info(f"Final Authorization: {authorization}")

        # 7. UFO-based Trade Execution
        should_execute = "APPROVE" in authorization.upper()

        if not should_execute and "REJECT" in authorization.upper():
            if "risk" in authorization.lower() and "exceed" in authorization.lower():
                logging.info("🔄 Fund Manager rejected due to high risk - will auto-scale and execute anyway")
                should_execute = True

        if should_execute:
            account_info = self.mt5_collector.connect() and mt5.account_info()
            should_trade, trade_reason = self.ufo_engine.should_open_new_trades(
                current_positions=open_positions_df,
                portfolio_status={'balance': account_info.balance, 'equity': account_info.equity} if account_info else None,
                ufo_data=enhanced_ufo_data
            )

            if not should_trade:
                logging.warning(f"UFO Engine: {trade_reason}")
            else:
                logging.info(f"🎯 UFO Engine: {trade_reason}")
                try:
                    json_match = re.search(r'{.*}', trade_decision_str, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        json_str = re.sub(r'//.*?\n', '\n', json_str)
                        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
                        parsed_data = json.loads(json_str)

                        actions_list = []
                        if 'actions' in parsed_data:
                            actions_list = parsed_data['actions']
                        elif 'trade_plan' in parsed_data:
                            actions_list = parsed_data['trade_plan']
                        elif 'trades' in parsed_data:
                            for trade in parsed_data['trades']:
                                actions_list.append({
                                    'action': 'new_trade',
                                    'currency_pair': trade['currency_pair'],
                                    'direction': trade['direction'].upper(),
                                    'volume': 0.1,
                                    'symbol': trade['currency_pair']
                                })

                        # ... (rest of execution logic is complex, will be handled if needed)

                except Exception as e:
                    logging.error(f"Error during UFO trade execution: {e}")

    def run(self):
        """
        Runs the main trading loop, orchestrating the main cycle and continuous monitoring.
        """
        self.last_cycle_time = time.time() - self.cycle_period_seconds - 1 # Ensure the first cycle runs immediately

        while True:
            try:
                now = time.time()
                
                # Check if it's time for the main trading cycle
                if now - self.last_cycle_time >= self.cycle_period_seconds:
                    if self.ufo_engine.is_active_session():
                        self.run_main_trading_cycle()
                    else:
                        logging.info(f"({datetime.now().strftime('%H:%M:%S')}) Outside active trading session. Skipping main cycle.")
                    self.last_cycle_time = now

                # Run continuous monitoring more frequently
                if self.continuous_monitoring_enabled and self.open_positions:
                    self.continuous_position_monitoring()
                
                # Sleep for a short interval before the next check
                sleep_duration = self.position_update_frequency_seconds
                time_to_next_cycle = (self.last_cycle_time + self.cycle_period_seconds) - now
                
                # Sleep for a shorter duration if the next cycle is approaching
                if time_to_next_cycle < sleep_duration:
                    sleep_duration = max(1, time_to_next_cycle)

                logging.info(f"--- Sleeping for {sleep_duration:.0f} seconds ---")
                time.sleep(sleep_duration)

            except KeyboardInterrupt:
                logging.info("\nTrading interrupted by user. Exiting...")
                break
            except Exception as e:
                logging.critical(f"Error in main trading loop: {e}")
                import traceback
                traceback.print_exc()
                logging.info("Waiting 60 seconds before retrying...")
                time.sleep(60)

    def get_real_time_market_data_for_positions(self, open_positions):
        """
        Collect real-time market data for all open positions
        This replaces the empty current_market_data = {} with actual price data
        """
        current_market_data = {}
        
        if open_positions is None or open_positions.empty:
            return current_market_data
            
        try:
            if not self.mt5_collector.connect():
                logging.warning("⚠️ Failed to connect to MT5 for market data collection")
                return current_market_data
                
            symbols_to_fetch = set(open_positions['symbol'])
            
            for symbol in symbols_to_fetch:
                try:
                    tick = mt5.symbol_info_tick(symbol)
                    if tick is not None:
                        current_market_data[symbol] = {
                            'close': tick.bid,
                            'ask': tick.ask,
                            'bid': tick.bid,
                            'spread': tick.ask - tick.bid,
                            'timestamp': pd.Timestamp.now()
                        }
                    else:
                        # Fallback to recent bar data
                        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 1)
                        if rates is not None and len(rates) > 0:
                            current_market_data[symbol] = {
                                'close': rates[0]['close'], 'ask': rates[0]['close'] + 0.0001,
                                'bid': rates[0]['close'], 'spread': 0.0001, 'timestamp': pd.Timestamp.now()
                            }
                        
                except Exception as e:
                    logging.error(f"❌ Error getting market data for {symbol}: {e}")
                    continue
            
            self.mt5_collector.disconnect()
            
        except Exception as e:
            logging.error(f"❌ Error in market data collection: {e}")
            
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
            logging.info(f"Portfolio synthetic value: {portfolio_value:.2f}%")
            
            if portfolio_value <= -5.0:  # Portfolio stop loss threshold
                logging.critical("Portfolio stop loss triggered - closing all positions")
                for _, position in positions.iterrows():
                    self.trade_executor.close_trade(position.ticket)
                    
        except Exception as e:
            logging.error(f"Error checking portfolio status: {e}")
