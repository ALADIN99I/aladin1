from .base_agent import Agent
from ..portfolio_manager import PortfolioManager
from ..synthetic_portfolio_manager import SyntheticPortfolioManager
import json

class TraderAgent(Agent):
    def __init__(self, name, llm_client, mt5_connection, symbols=None):
        super().__init__(name, llm_client)
        self.portfolio_manager = PortfolioManager(mt5_connection)
        self.synthetic_portfolio_manager = SyntheticPortfolioManager()
        self.symbols = symbols if symbols else []

    def formulate_daily_hypothesis(self, ufo_data, economic_events):
        """
        Formulates a daily trading hypothesis and defines a synthetic portfolio.
        """
        if not ufo_data:
            return None

        ufo_summary = self.summarize_ufo_data(ufo_data)
        events_summary = economic_events.to_string() if not economic_events.empty else "No significant economic events today."

        prompt = (
            "You are a senior Forex strategist. Based on the following market analysis, "
            "formulate a trading hypothesis for the day. Identify the currencies that are likely to be strong and weak. "
            "Then, define a synthetic portfolio to express this view. "
            "The output should be a JSON object with 'buy' and 'sell' keys, e.g., "
            "`{'buy': ['USD', 'JPY'], 'sell': ['EUR', 'GBP']}`.\n\n"
            f"UFO Market Analysis:\n{ufo_summary}\n\n"
            f"Economic Events:\n{events_summary}\n\n"
            "Your response should be only the JSON object."
        )

        response = self.llm_client.generate_response(prompt)

        try:
            portfolio_definition = json.loads(response)
            return portfolio_definition
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from LLM response: {response}")
            return None

    def execute(self, research_consensus, open_positions, diversification_config=None, synthetic_portfolio_history=None):
        """
        Makes a trading decision based on the research consensus and open positions using the LLM.
        Enhanced with dynamic diversification awareness and synthetic portfolio analysis.
        """
        account_info = self.portfolio_manager.get_account_info()
        balance = account_info.balance if account_info else 10000  # Default to 10k if info not available

        open_positions_str = open_positions.to_string() if not open_positions.empty else "No open positions."

        portfolio_history_str = synthetic_portfolio_history.to_string() if not synthetic_portfolio_history.empty else "No synthetic portfolio history yet."

        # Calculate diversification context
        position_count = len(open_positions) if not open_positions.empty else 0
        
        # Get diversification parameters from config (with defaults)
        if diversification_config:
            min_positions = diversification_config.get('min_positions_for_session', 2)
            target_positions = diversification_config.get('target_positions_when_available', 4) 
            max_positions = diversification_config.get('max_concurrent_positions', 9)
        else:
            # Fallback defaults if no config provided
            min_positions, target_positions, max_positions = 2, 4, 9
        
        # Dynamic diversification guidance based on config values
        diversification_guidance = ""
        if position_count == 0:
            diversification_guidance = f"\n🎯 DIVERSIFICATION PRIORITY: No open positions - consider opening {min_positions}-{target_positions} quality trades to establish proper diversification."
        elif position_count < min_positions:
            diversification_guidance = f"\n🎯 DIVERSIFICATION PRIORITY: {position_count} position(s) open - strongly consider additional quality trades (minimum: {min_positions}, target: {target_positions} total positions) for better risk distribution."
        elif position_count < target_positions:
            diversification_guidance = f"\n📊 DIVERSIFICATION STATUS: {position_count} positions open - consider additional quality opportunities (target: {target_positions}+ positions) if strong analysis supports them."
        elif position_count >= target_positions and position_count < (max_positions - 2):
            diversification_guidance = f"\n✅ GOOD DIVERSIFICATION: {position_count} positions - well diversified, only add exceptional opportunities (max {max_positions} total)."
        elif position_count >= (max_positions - 2):
            diversification_guidance = f"\n⚠️ HIGH DIVERSIFICATION: {position_count} positions - focus on position management, avoid new trades unless replacing closed ones (max {max_positions})."
        
        if position_count >= max_positions:
            return '{"trades": []}'

        prompt = (
            "You are a professional Forex trader implementing the UFO methodology. "
            "Your primary analysis tool is a 'Synthetic Portfolio' that represents your daily trading hypothesis. "
            "Your task is to analyze the performance of this synthetic portfolio and decide on the appropriate trading actions.\n\n"
            "The trade plan should be a JSON object with a single key 'decision' which can be 'buy_portfolio', 'sell_portfolio', 'close_portfolio', 'reverse_portfolio', or 'hold'. "
            "For example: `{'decision': 'buy_portfolio'}`.\n\n"
            f"Account Balance: ${balance}\n\n"
            f"Synthetic Portfolio History:\n{portfolio_history_str}\n\n"
            f"Research Consensus:\n{research_consensus}\n\n"
            f"Current Open Positions ({position_count} total):\n{open_positions_str}\n\n"
            "YOUR TASK:\n"
            "1.  Analyze the `Synthetic Portfolio History`. Look for trends, support/resistance levels, and potential reversal patterns.\n"
            "2.  Based on your analysis, decide on one of the following actions:\n"
            "    - 'buy_portfolio': If the portfolio is in a confirmed uptrend.\n"
            "    - 'sell_portfolio': If the portfolio is in a confirmed downtrend.\n"
            "    - 'close_portfolio': If the trend is weakening or a profit target has been reached.\n"
            "    - 'reverse_portfolio': If you see a strong reversal signal (e.g., the portfolio has hit a major resistance level and is turning down).\n"
            "    - 'hold': If there is no clear signal.\n"
            "3.  Provide a clear rationale for your decision in your response."
        )

        llm_response = self.llm_client.generate_response(prompt)

        try:
            decision_data = json.loads(llm_response)
            decision = decision_data.get('decision')
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from LLM response: {llm_response}")
            decision = 'hold'

        if decision == 'buy_portfolio':
            return self.construct_portfolio_trades('buy')
        elif decision == 'sell_portfolio':
            return self.construct_portfolio_trades('sell')
        elif decision == 'close_portfolio':
            return self.construct_close_portfolio_trades(open_positions)
        elif decision == 'reverse_portfolio':
            close_trades = self.construct_close_portfolio_trades(open_positions)
            # This is a simplified implementation. A more robust version would determine the new direction.
            open_trades = self.construct_portfolio_trades('buy')

            close_trades_data = json.loads(close_trades)
            open_trades_data = json.loads(open_trades)

            all_trades = close_trades_data.get('trades', []) + open_trades_data.get('trades', [])
            return json.dumps({"trades": all_trades})
        else:
            return '{"trades": []}'

    def calculate_lot_size(self, balance, risk_per_trade_percentage=1.0):
        """
        Calculates the lot size based on account balance and risk percentage.
        This is a simplified version. A real implementation would consider pip values and stop loss levels.
        """
        risk_amount = balance * (risk_per_trade_percentage / 100)
        # Assuming a standard risk of $10 per 0.01 lot
        lot_size = (risk_amount / 10) * 0.01
        return round(lot_size, 2)

    def construct_portfolio_trades(self, direction):
        """
        Constructs a list of trades based on the synthetic portfolio definition.
        """
        trades = []
        if not self.synthetic_portfolio_manager.portfolio_definition:
            return '{"trades": []}'

        account_info = self.portfolio_manager.get_account_info()
        balance = account_info.balance if account_info else 10000

        lot_size = self.calculate_lot_size(balance)

        buy_currencies = self.synthetic_portfolio_manager.portfolio_definition.get('buy', [])
        sell_currencies = self.synthetic_portfolio_manager.portfolio_definition.get('sell', [])

        if direction == 'buy':
            # Buy the "buy" currencies against the "sell" currencies
            for buy_curr in buy_currencies:
                for sell_curr in sell_currencies:
                    # Note: This is a simplified way of forming pairs. A more robust
                    # implementation would check for valid symbols in self.symbols.
                    trades.append({
                        'action': 'new_trade',
                        'currency_pair': f"{buy_curr}{sell_curr}",
                        'direction': 'BUY',
                        'lot_size': lot_size
                    })
        elif direction == 'sell':
            # Sell the "buy" currencies against the "sell" currencies
            for buy_curr in buy_currencies:
                for sell_curr in sell_currencies:
                    trades.append({
                        'action': 'new_trade',
                        'currency_pair': f"{buy_curr}{sell_curr}",
                        'direction': 'SELL',
                        'lot_size': lot_size
                    })

        return json.dumps({"trades": trades})

    def construct_close_portfolio_trades(self, open_positions):
        """
        Constructs a list of actions to close all open trades.
        """
        if open_positions.empty:
            return '{"trades": []}'

        trades = []
        for index, row in open_positions.iterrows():
            trades.append({
                'action': 'close_trade',
                'trade_id': row['ticket']
            })

        return json.dumps({"trades": trades})

    def summarize_ufo_data(self, ufo_data):
        """
        Summarizes the UFO data for the LLM prompt.
        """
        if not ufo_data:
            return "No UFO data available."

        summary = []
        # Correctly access coherence_analysis from ufo_data
        coherence_analysis = ufo_data.get('coherence_analysis', {})
        if not coherence_analysis:
             return "No coherence analysis available in UFO data."

        for currency, values in coherence_analysis.items():
            summary.append(f"Currency {currency}:")
            summary.append(f"  Coherence Level: {values.get('coherence_level', 'N/A')}")
            summary.append(f"  Overall Coherence: {values.get('overall_coherence', 'N/A')}")
            summary.append(f"  Dominant Direction: {values.get('dominant_direction', 'N/A')}")

        return "\n".join(summary)
