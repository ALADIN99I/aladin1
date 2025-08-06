from .base_agent import Agent
from ..portfolio_manager import PortfolioManager

class TraderAgent(Agent):
    def __init__(self, name, llm_client, mt5_connection, symbols=None):
        super().__init__(name, llm_client)
        self.portfolio_manager = PortfolioManager(mt5_connection)
        self.symbols = symbols if symbols else []

    def execute(self, research_consensus, open_positions, diversification_config=None):
        """
        Makes a trading decision based on the research consensus and open positions using the LLM.
        Enhanced with dynamic diversification awareness based on config values.
        """
        account_info = self.portfolio_manager.get_account_info()
        balance = account_info.balance if account_info else 10000  # Default to 10k if info not available

        open_positions_str = open_positions.to_string() if not open_positions.empty else "No open positions."

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
            "You are a professional Forex trader implementing intelligent diversification with UFO methodology. "
            "Based on the following multi-timeframe research consensus and current portfolio, formulate a precise trade plan. "
            "Your decisions should balance QUALITY analysis with DIVERSIFICATION needs for optimal risk management.\n\n"
            "DIVERSIFICATION PRINCIPLES:\n"
            f"• Minimum {min_positions} positions for basic diversification\n"
            f"• Target {target_positions} positions for optimal risk distribution\n"
            f"• Maximum {max_positions} positions to maintain focus\n"
            "• Each trade must have strong analytical support\n"
            "• Prefer different currency pairs/correlations\n\n"
            f"{diversification_guidance}\n\n"
            "The trade plan should be a JSON object with actions having this structure: "
            "`{'action': 'new_trade'/'adjust_trade'/'close_trade', 'trade_id': <optional>, 'currency_pair': 'EURUSD', 'direction': 'BUY/SELL', 'entry_price': 1.0800, 'lot_size': 0.40}`.\n\n"
            f"You MUST use only the following currency pairs for new trades: {self.symbols}\n\n"
            "IMPORTANT: NO individual stop losses or take profits - UFO methodology uses PORTFOLIO-LEVEL risk management only!\n\n"
            f"Account Balance: ${balance} - Risk tolerance: 0.8-1.2% per trade, max 4.5% total portfolio risk.\n\n"
            f"Research Consensus:\n{research_consensus}\n\n"
            f"Current Open Positions ({position_count} total):\n{open_positions_str}\n\n"
            "YOUR TASK:\n"
            "1.  Analyze the `Current Open Positions`. \n"
            "2.  If the number of open positions is at or near the `Maximum positions` limit, you should prioritize closing or adjusting existing positions over opening new ones.\n"
            "3.  Only suggest `new_trade` actions if there is sufficient capacity in the portfolio.\n"
            "4.  If you suggest new trades, ensure they are aligned with the `Research Consensus` and do not excessively increase the portfolio's risk.\n"
            "5.  Provide a clear rationale for each action in your response."
        )

        trade_decision_str = self.llm_client.generate_response(prompt)

        if not isinstance(trade_decision_str, str) or not trade_decision_str.strip():
            trade_decision_str = "{\"trades\": []}"
            print("Warning: TraderAgent LLM did not return a valid trade decision string.")

        print(f"LLM Trade Decision:\n{trade_decision_str}")
        return trade_decision_str
