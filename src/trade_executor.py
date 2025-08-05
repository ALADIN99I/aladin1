try:
    import MetaTrader5 as mt5
except ImportError:
    from . import mock_metatrader5 as mt5

import datetime
import pytz
from .ufo_trading_engine import UFOTradingEngine

class TradeExecutor:
    def __init__(self, mt5_connection, config=None):
        self.mt5_connection = mt5_connection
        self.ufo_engine = UFOTradingEngine(config) if config else None
        # Read from config, convert to positive percentage (config is negative)
        if config:
            portfolio_stop_raw = config['trading'].get('portfolio_equity_stop', '-5.0')
            # Handle config values with inline comments like '-5.0 (-3.0)'
            portfolio_stop_clean = portfolio_stop_raw.split(' ')[0].split('(')[0].strip()
            portfolio_stop = float(portfolio_stop_clean)
            self.max_portfolio_risk = abs(portfolio_stop) / 100  # Convert -5.0 to 0.05
        else:
            self.max_portfolio_risk = 0.05  # Default 5%

    def execute_trade(self, symbol, trade_type, volume, price, sl=0, tp=0, comment=""):
        """
        Executes a trade on the MT5 terminal.
        UFO Methodology: NO individual stop-losses or take-profits!
        Uses portfolio-level stop management instead.
        """
        if not self.mt5_connection.connect():
            print(f"❌ Failed to connect to MT5 for {symbol}")
            return None

        # Validate symbol exists and is available for trading
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"❌ Symbol {symbol} not found. Trying to add symbol to Market Watch...")
            # Try to add symbol to Market Watch
            if not mt5.symbol_select(symbol, True):
                print(f"❌ Failed to add {symbol} to Market Watch. Available symbols might be different.")
                print(f"💡 Check if the correct symbol suffix is used (e.g., -ECN, .m, etc.)")
                self.mt5_connection.disconnect()
                return None
            # Try to get symbol info again
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                print(f"❌ Still cannot get symbol info for {symbol} after adding to Market Watch")
                self.mt5_connection.disconnect()
                return None
        
        # Check if symbol is available for trading
        if not symbol_info.visible:
            print(f"⚠️ Symbol {symbol} is not visible in Market Watch, attempting to enable...")
            if not mt5.symbol_select(symbol, True):
                print(f"❌ Failed to enable {symbol} in Market Watch")
                self.mt5_connection.disconnect()
                return None
        
        # Get current tick data
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            print(f"❌ Failed to get tick data for {symbol}")
            self.mt5_connection.disconnect()
            return None
        
        # UFO METHODOLOGY: Build trade request WITHOUT individual SL/TP
        # Portfolio-level stop management handles exits
        
        # Limit comment to 31 characters (MT5 limit)
        trade_comment = comment[:25] if len(comment) > 25 else comment
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": trade_type,
            "price": price,
            "deviation": 50,
            "magic": 234000,
            "comment": trade_comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # UFO: NO individual stops! This prevents broker "stop hunting"
        print(f"🎯 UFO Trade: Opening {symbol} {('BUY' if trade_type == mt5.ORDER_TYPE_BUY else 'SELL')} {volume} lots WITHOUT individual stops")

        result = mt5.order_send(request)

        # Check if result is None (connection or other critical error)
        if result is None:
            print(f"Order send failed: mt5.order_send() returned None - check MT5 connection and symbol {symbol}")
            print(f"Last MT5 error: {mt5.last_error()}")
            self.mt5_connection.disconnect()
            return None

        # Check result code
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Order send failed for {symbol}, retcode={result.retcode}")
            print(f"MT5 error description: {mt5.last_error()}")
            self.mt5_connection.disconnect()
            return None

        print(f"✅ Order sent successfully for {symbol}, order ticket: {result.order}")
        self.mt5_connection.disconnect()
        return result

    def execute_portfolio(self, trades):
        """
        Executes a portfolio of trades.
        """
        results = []
        for trade in trades:
            result = self.execute_trade(
                symbol=trade['symbol'],
                trade_type=trade['trade_type'],
                volume=trade['volume'],
                price=trade['price'],
                sl=trade['sl'],
                tp=trade['tp'],
                comment=trade.get('comment', '')
            )
            results.append(result)
        return results

    def close_all_trades(self):
        """
        Closes all open trades.
        """
        if not self.mt5_connection.connect():
            return False

        positions = mt5.positions_get()
        if positions is None:
            print("No positions found.")
            return True

        for position in positions:
            self.close_trade(position.ticket)

        self.mt5_connection.disconnect()
        return True

    def close_trade(self, ticket):
        """
        Closes a trade on the MT5 terminal.
        """
        if not self.mt5_connection.connect():
            return False

        position_info = mt5.positions_get(ticket=ticket)
        if position_info is None or len(position_info) == 0:
            print(f"No position found with ticket {ticket}")
            self.mt5_connection.disconnect()
            return False

        position = position_info[0]

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "position": position.ticket,
            "deviation": 20,
            "magic": 234000,
            "comment": "Closing trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Close order failed, retcode={result.retcode}")
            self.mt5_connection.disconnect()
            return False

        print(f"Position {ticket} closed successfully.")
        self.mt5_connection.disconnect()
        return True

    def manage_open_positions(self, market_data, ufo_data):
        """
        Manages open positions using UFO methodology.
        Returns list of actions to take: 'close', 'reinforce', 'hold'
        """
        if not self.ufo_engine:
            return []
        
        if not self.mt5_connection.connect():
            return []
        
        positions = mt5.positions_get()
        if positions is None:
            self.mt5_connection.disconnect()
            return []
        
        actions = []
        for position in positions:
            symbol = position.symbol
            
            # Check if we should close based on session timing
            if self.ufo_engine.should_close_on_session():
                actions.append({
                    'action': 'close',
                    'ticket': position.ticket,
                    'reason': 'session_close'
                })
                continue
            
            # Check analysis-based exit signals
            if symbol in ufo_data:
                exit_signal = self.ufo_engine.check_exit_signals(ufo_data[symbol], position.type)
                if exit_signal:
                    actions.append({
                        'action': 'close',
                        'ticket': position.ticket,
                        'reason': 'analysis_exit',
                        'details': exit_signal
                    })
                    continue
                
                # Check if we should reinforce position
                if position.profit < 0:  # Only reinforce losing positions
                    reinforce_signal = self.ufo_engine.should_reinforce_position(
                        ufo_data[symbol], position.type
                    )
                    if reinforce_signal:
                        actions.append({
                            'action': 'reinforce',
                            'ticket': position.ticket,
                            'symbol': symbol,
                            'original_volume': position.volume,
                            'reinforcement_volume': position.volume * 0.5,  # 50% of original
                            'reason': 'analysis_reinforce'
                        })
                        continue
            
            # Check portfolio-level stop loss
            account_info = mt5.account_info()
            if account_info and self.ufo_engine.check_portfolio_stop(account_info.equity, account_info.balance):
                actions.append({
                    'action': 'close',
                    'ticket': position.ticket,
                    'reason': 'portfolio_stop'
                })
                continue
            
            # Default: hold position
            actions.append({
                'action': 'hold',
                'ticket': position.ticket
            })
        
        self.mt5_connection.disconnect()
        return actions
    
    def execute_position_actions(self, actions):
        """
        Executes the position management actions from UFO analysis.
        """
        results = []
        
        for action in actions:
            if action['action'] == 'close':
                result = self.close_trade(action['ticket'])
                print(f"Closed position {action['ticket']} - Reason: {action['reason']}")
                results.append({'action': 'close', 'ticket': action['ticket'], 'success': result})
            
            elif action['action'] == 'reinforce':
                # Execute reinforcement trade
                if not self.mt5_connection.connect():
                    continue
                
                # Get current position info to determine trade direction
                position_info = mt5.positions_get(ticket=action['ticket'])
                if position_info and len(position_info) > 0:
                    position = position_info[0]
                    
                    # Get current market price for entry
                    tick = mt5.symbol_info_tick(action['symbol'])
                    if tick:
                        price = tick.ask if position.type == mt5.ORDER_TYPE_BUY else tick.bid
                        
                        # Execute reinforcement trade without SL/TP (managed by analysis)
                        result = self.execute_trade(
                            symbol=action['symbol'],
                            trade_type=position.type,
                            volume=action['reinforcement_volume'],
                            price=price,
                            sl=0,  # No fixed stops - managed by UFO analysis
                            tp=0,
                            comment="UFO Reinforcement"
                        )
                        
                        print(f"Reinforced position for {action['symbol']} with {action['reinforcement_volume']} lots")
                        results.append({
                            'action': 'reinforce', 
                            'ticket': action['ticket'],
                            'new_trade': result.order if result else None,
                            'success': result is not None
                        })
                
                self.mt5_connection.disconnect()
            
            elif action['action'] == 'hold':
                # Just log that we're holding
                results.append({'action': 'hold', 'ticket': action['ticket'], 'success': True})
        
        return results
    
    def execute_ufo_trade(self, symbol, trade_type, volume, comment="UFO Trade"):
        """
        Executes a trade using UFO methodology (no fixed SL/TP).
        """
        if not self.mt5_connection.connect():
            return None
        
        # Get current market price
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            print(f"Failed to get tick data for {symbol}")
            self.mt5_connection.disconnect()
            return None
        
        price = tick.ask if trade_type == mt5.ORDER_TYPE_BUY else tick.bid
        
        # Execute trade without fixed stops - managed by UFO analysis
        result = self.execute_trade(
            symbol=symbol,
            trade_type=trade_type,
            volume=volume,
            price=price,
            sl=0,  # No fixed stops
            tp=0,  # No fixed targets
            comment=comment
        )
        
        return result
    
    def debug_available_symbols(self, search_pattern=""):
        """
        Debug helper to show available symbols in MT5.
        """
        if not self.mt5_connection.connect():
            print("❌ Failed to connect to MT5")
            return []
        
        symbols = mt5.symbols_get()
        if symbols is None:
            print("❌ Failed to get symbols list")
            self.mt5_connection.disconnect()
            return []
        
        available_symbols = []
        for symbol in symbols:
            if search_pattern == "" or search_pattern.upper() in symbol.name.upper():
                available_symbols.append({
                    'name': symbol.name,
                    'visible': symbol.visible,
                    'description': symbol.description
                })
        
        print(f"\n🔍 Found {len(available_symbols)} symbols matching '{search_pattern}':")
        for i, sym in enumerate(available_symbols[:10]):  # Show first 10
            status = "✅" if sym['visible'] else "❌"
            print(f"{status} {sym['name']} - {sym['description']}")
        
        if len(available_symbols) > 10:
            print(f"... and {len(available_symbols) - 10} more")
        
        self.mt5_connection.disconnect()
        return available_symbols
