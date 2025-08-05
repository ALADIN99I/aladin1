from .base_agent import Agent
from ..portfolio_manager import PortfolioManager
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings

class RiskManagerAgent(Agent):
    def __init__(self, name, llm_client, mt5_connection, config):
        super().__init__(name, llm_client)
        self.portfolio_manager = PortfolioManager(mt5_connection)
        # Read portfolio stop from config, same as UFOTradingEngine
        portfolio_stop_raw = config['trading'].get('portfolio_equity_stop', '-5.0')
        # Handle config values with inline comments like '-5.0 (-3.0)'
        portfolio_stop_clean = portfolio_stop_raw.split(' ')[0].split('(')[0].strip()
        self.portfolio_equity_stop = float(portfolio_stop_clean)
        # Keep legacy parameter for backward compatibility if needed
        self.stop_loss_threshold = self.portfolio_equity_stop
        
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
        
        # Predictive Risk Modeling Components - Configurable via config.ini
        self.config = config
        self.risk_prediction_enabled = config['trading'].get('risk_prediction_enabled', 'true').lower() == 'true'
        self.ufo_trend_history_length = parse_config_value(config['trading'].get('ufo_trend_history_length', '15'), 15)
        self.risk_prediction_window = parse_config_value(config['trading'].get('risk_prediction_window', '8'), 8)
        self.early_warning_threshold = parse_config_value(config['trading'].get('early_warning_threshold', '0.75'), 0.75)
        self.trend_deterioration_threshold = parse_config_value(config['trading'].get('trend_deterioration_threshold', '2.5'), 2.5)
        self.prediction_confidence_threshold = parse_config_value(config['trading'].get('prediction_confidence_threshold', '0.65'), 0.65)
        self.risk_escalation_levels = parse_config_value(config['trading'].get('risk_escalation_levels', '3'), 3)
        self.volatility_risk_multiplier = parse_config_value(config['trading'].get('volatility_risk_multiplier', '1.5'), 1.5)
        
        # Initialize predictive risk tracking
        self.ufo_trend_history = []  # Store UFO data trends for prediction
        self.risk_prediction_cache = {}
        self.last_risk_warnings = []
        self.portfolio_risk_trajectory = []

    def execute(self, trade_decision):
        """
        Assesses the risk of a trade and the overall portfolio using the LLM.
        """
        equity_curve = self.portfolio_manager.calculate_equity_curve()

        prompt = (
            "You are a senior risk analyst. Assess the risk of the following trade plan. "
            "Consider market volatility, the provided equity curve, and the overall risk "
            "profile of the portfolio. Provide a risk score (1-5) and a detailed "
            "justification for your assessment.\n\n"
            f"Trade Plan:\n{trade_decision}\n\n"
            f"Portfolio Equity Curve:\n{equity_curve.to_string() if equity_curve is not None else 'N/A'}"
        )

        risk_assessment = self.llm_client.generate_response(prompt)

        portfolio_risk = "OK"
        if equity_curve is not None and not equity_curve.empty:
            account_info = self.portfolio_manager.get_account_info()
            if account_info:
                current_equity = equity_curve['equity'].iloc[-1]
                initial_balance = account_info.balance
                drawdown = (current_equity - initial_balance) / initial_balance * 100
                if drawdown < self.stop_loss_threshold:
                    portfolio_risk = "STOP_LOSS_BREACHED"
                    self.portfolio_manager.close_all_trades()

        # Enhanced with predictive risk modeling
        predictive_risk_assessment = self.perform_predictive_risk_analysis(equity_curve, trade_decision)
        
        return {
            "trade_risk_assessment": risk_assessment,
            "portfolio_risk_status": portfolio_risk,
            "predictive_risk_analysis": predictive_risk_assessment
        }
    
    def perform_predictive_risk_analysis(self, equity_curve, trade_decision):
        """
        Perform predictive risk analysis using UFO data trends and forward-looking projections
        """
        if not self.risk_prediction_enabled:
            return {
                "prediction_enabled": False,
                "risk_level": "normal",
                "warnings": [],
                "recommendations": []
            }
        
        warnings = []
        recommendations = []
        risk_level = "normal"
        
        try:
            # Current portfolio risk analysis
            current_equity = equity_curve['equity'].iloc[-1] if len(equity_curve) > 0 else 10000
            initial_equity = equity_curve['equity'].iloc[0] if len(equity_curve) > 0 else 10000
            current_drawdown = ((current_equity - initial_equity) / initial_equity) * 100
            
            # Early warning threshold check
            warning_threshold = self.portfolio_equity_stop * self.early_warning_threshold
            if current_drawdown <= warning_threshold:
                risk_level = "high"
                warnings.append(f"Portfolio approaching risk threshold: {current_drawdown:.2f}% (threshold: {warning_threshold:.2f}%)")
                recommendations.append("Consider reducing position sizes or closing losing trades")
            
            # UFO trend deterioration analysis
            if len(self.ufo_trend_history) >= 3:
                recent_trends = self.ufo_trend_history[-3:]
                trend_changes = []
                
                for i in range(1, len(recent_trends)):
                    if 'coherence' in recent_trends[i] and 'coherence' in recent_trends[i-1]:
                        change = recent_trends[i]['coherence'] - recent_trends[i-1]['coherence']
                        trend_changes.append(change)
                
                if trend_changes and abs(sum(trend_changes)) > self.trend_deterioration_threshold:
                    risk_level = "medium" if risk_level == "normal" else risk_level
                    warnings.append("UFO trend deterioration detected")
                    recommendations.append("Monitor positions closely for exit signals")
            
            # Predictive risk projection
            if len(equity_curve) >= self.risk_prediction_window:
                recent_equity = equity_curve['equity'].tail(self.risk_prediction_window)
                equity_trend = recent_equity.pct_change().mean()
                
                if equity_trend < -0.005:  # Declining trend
                    projected_risk = current_drawdown + (equity_trend * 100 * self.risk_prediction_window)
                    if projected_risk <= self.portfolio_equity_stop * 0.8:
                        risk_level = "critical"
                        warnings.append(f"Projected risk breach in {self.risk_prediction_window} periods")
                        recommendations.append("Consider emergency position closure")
            
            # Volatility risk assessment
            if hasattr(self, 'market_volatility_state') and self.market_volatility_state in ['high', 'extreme']:
                risk_multiplier = self.volatility_risk_multiplier if self.market_volatility_state == 'high' else self.volatility_risk_multiplier * 1.5
                adjusted_risk = current_drawdown * risk_multiplier
                
                if adjusted_risk <= warning_threshold:
                    risk_level = "high"
                    warnings.append(f"High volatility increases effective risk to {adjusted_risk:.2f}%")
                    recommendations.append("Reduce position sizes during high volatility period")
            
            return {
                "prediction_enabled": True,
                "risk_level": risk_level,
                "current_drawdown": current_drawdown,
                "warning_threshold": warning_threshold,
                "warnings": warnings,
                "recommendations": recommendations,
                "confidence": self.prediction_confidence_threshold,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "prediction_enabled": True,
                "risk_level": "unknown",
                "error": str(e),
                "warnings": ["Predictive risk analysis failed"],
                "recommendations": ["Manual risk assessment recommended"]
            }
