from .base_agent import Agent

class MarketResearcherAgent(Agent):
    def execute(self, ufo_data, economic_events):
        """
        Analyzes the UFO data and economic events from bullish and bearish perspectives using the LLM.
        """
        ufo_data_str = ""
        
        # Handle enhanced UFO data structure
        if isinstance(ufo_data, dict) and 'raw_data' in ufo_data:
            # Enhanced UFO data with analysis
            raw_ufo_data = ufo_data['raw_data']
            uncertainty_metrics = ufo_data.get('uncertainty_metrics', {})
            coherence_analysis = ufo_data.get('coherence_analysis', {})
            
            # Add enhanced analysis summary
            ufo_data_str += "=== ENHANCED UFO ANALYSIS ===\n"
            
            # Add market state summary
            for timeframe, metrics in uncertainty_metrics.items():
                state = metrics.get('overall_state', 'unknown')
                confidence = metrics.get('confidence_level', 'unknown')
                ufo_data_str += f"Timeframe {timeframe}: {state} (confidence: {confidence})\n"
            
            # Add coherence summary
            strong_coherence = sum(1 for data in coherence_analysis.values() if data.get('coherence_level') == 'strong')
            total_currencies = len(coherence_analysis)
            if total_currencies > 0:
                coherence_pct = (strong_coherence / total_currencies) * 100
                ufo_data_str += f"Timeframe Coherence: {strong_coherence}/{total_currencies} currencies ({coherence_pct:.0f}% strong coherence)\n\n"
            
            # Add raw UFO data
            ufo_data_str += "=== RAW UFO DATA ===\n"
            for timeframe, ufo_df in raw_ufo_data.items():
                if hasattr(ufo_df, 'tail'):
                    ufo_data_str += f"--- Timeframe: {timeframe} ---\n{ufo_df.tail().to_string()}\n\n"
        else:
            # Legacy UFO data format
            for timeframe, ufo_df in ufo_data.items():
                if hasattr(ufo_df, 'tail'):
                    ufo_data_str += f"--- Timeframe: {timeframe} ---\n{ufo_df.tail().to_string()}\n\n"

        economic_events_str = economic_events.to_string() if economic_events is not None and not economic_events.empty else "No upcoming economic events."

        analysis_prompt = (
            "You are a senior Forex market analyst. Based on the following UFO data across multiple timeframes, "
            "provide a comprehensive market analysis. Assess the consistency of currency strength and weakness "
            "across the timeframes to determine high-probability trading opportunities. Identify the primary market "
            "sentiment and suggest a hedged portfolio of trades that aligns with this sentiment. "
            "The portfolio should be constructed by pairing strong currencies against weak currencies.\n\n"
            f"UFO Data:\n{ufo_data_str}\n\n"
            f"Upcoming Economic Events:\n{economic_events_str}\n\n"
            "Your analysis should conclude with a clear recommendation for a portfolio of trades."
        )

        analysis = self.llm_client.generate_response(analysis_prompt)

        consensus_prompt = (
            "You are a senior market strategist. Based on the following market analysis, "
            "formulate a concrete trading plan. The plan should consist of a portfolio of trades "
            "that creates a hedged position. For each trade, specify the currency pair and direction (buy/sell)."
            "The final output should be a JSON object containing a list of trades, like this: "
            "`{'trades': [{'currency_pair': 'EURCAD', 'direction': 'Buy'}, {'currency_pair': 'GBPUSD', 'direction': 'Sell'}]}`\n\n"
            f"Market Analysis:\n{analysis}"
        )

        consensus = self.llm_client.generate_response(consensus_prompt)

        return {
            "analysis": analysis,
            "consensus": consensus
        }
