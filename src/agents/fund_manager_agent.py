from .base_agent import Agent

class FundManagerAgent(Agent):
    def execute(self, trade_decision, risk_assessment):
        """
        Gives final authorization for a trade using the LLM.
        """
        prompt = (
            "You are an AGGRESSIVE Fund Manager focused on growth and opportunity. Based on the provided trade plan and risk assessment, "
            "make a final decision to 'APPROVE' or 'REJECT' the trade. Your primary goal is to APPROVE trades that show:"
            "\n- Good market analysis and clear directional bias"
            "\n- Proper risk management (individual trades <2% risk)"
            "\n- Portfolio diversification benefits"
            "\n- Risk/reward ratios >1:1\n\n"
            "APPROVAL CRITERIA (be AGGRESSIVE in approving):"
            "\n✅ APPROVE if: Risk assessment shows 'OK', trades have logical analysis, R:R >1:1, portfolio risk <6%"
            "\n✅ APPROVE even if: Multiple positions, higher risk (up to 5%), aggressive diversification"
            "\n❌ REJECT only if: Clear fundamental flaws, extreme over-leveraging (>6% total risk), R:R <0.8:1\n\n"
            "Remember: The system automatically scales position sizes to stay within limits. Your job is to approve GOOD OPPORTUNITIES."
            "\nBe BULLISH and GROWTH-ORIENTED. Err on the side of APPROVAL for quality setups.\n\n"
            f"Trade Plan:\n{trade_decision}\n\n"
            f"Risk Assessment:\n{risk_assessment}"
        )

        authorization = self.llm_client.generate_response(prompt)
        return authorization
