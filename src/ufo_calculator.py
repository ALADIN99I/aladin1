import pandas as pd
import numpy as np
from scipy import stats

class UfoCalculator:
    def __init__(self, currencies):
        self.currencies = currencies
        # Mean reversion detection parameters
        self.oscillation_lookback = 20  # Bars to analyze for oscillations
        self.mean_reversion_threshold = 2.0  # Standard deviations for mean reversion
        self.trend_coherence_threshold = 0.7  # Correlation threshold for trend coherence

    def calculate_percentage_variation(self, price_data):
        """
        Calculates the percentage variation for each currency cross.
        """
        if 'time' in price_data.columns:
            price_data = price_data.set_index('time')

        variation_data = price_data.pct_change() * 100
        variation_data.fillna(0, inplace=True)
        return variation_data

    def calculate_incremental_sum(self, variation_data):
        """
        Calculates the incremental sum of the percentage variations.
        """
        return variation_data.cumsum()

    def generate_ufo_data(self, incremental_sums_dict):
        """
        Generates the UFO market performance data for multiple timeframes using a rolling window.
        """
        ufo_data_dict = {}
        for timeframe, incremental_sums in incremental_sums_dict.items():
            ufo_data = pd.DataFrame(index=incremental_sums.index)
            for currency in self.currencies:
                currency_performance = pd.Series(0, index=incremental_sums.index)
                for cross in incremental_sums.columns:
                    if currency in cross:
                        base, quote = cross[:3], cross[3:]
                        if currency == base:
                            currency_performance += incremental_sums[cross]
                        else:
                            currency_performance -= incremental_sums[cross]
                # Use a rolling window to calculate the strength
                ufo_data[currency] = currency_performance.rolling(window=20).mean()
            ufo_data.fillna(0, inplace=True)
            ufo_data_dict[timeframe] = ufo_data
        return ufo_data_dict
    
    def detect_oscillations(self, ufo_data_dict):
        """
        Detects short-term oscillations across multiple timeframes for mean reversion opportunities.
        UFO Methodology: Identify when currencies are oscillating within ranges vs trending.
        """
        oscillation_analysis = {}
        
        for timeframe, ufo_data in ufo_data_dict.items():
            if len(ufo_data) < self.oscillation_lookback:
                continue
                
            tf_oscillations = {}
            
            for currency in self.currencies:
                if currency not in ufo_data.columns:
                    continue
                    
                currency_data = ufo_data[currency].iloc[-self.oscillation_lookback:]
                
                # Calculate oscillation metrics
                volatility = currency_data.std()
                mean_value = currency_data.mean()
                current_value = currency_data.iloc[-1]
                
                # Detect mean reversion conditions
                z_score = (current_value - mean_value) / volatility if volatility > 0 else 0
                
                # Check for oscillation pattern (multiple reversals)
                reversals = self._count_direction_changes(currency_data)
                
                # Determine market state
                is_trending = all(currency_data.iloc[-3:] > mean_value) or all(currency_data.iloc[-3:] < mean_value)

                if is_trending and abs(z_score) > 1.5:
                    market_state = 'trending'
                elif abs(z_score) > self.mean_reversion_threshold:
                    market_state = 'mean_reversion_opportunity'
                elif reversals >= 3 and volatility > 0.5:
                    market_state = 'oscillating'
                else:
                    market_state = 'uncertain'
                
                tf_oscillations[currency] = {
                    'z_score': z_score,
                    'volatility': volatility,
                    'reversals': reversals,
                    'market_state': market_state,
                    'mean_reversion_signal': abs(z_score) > self.mean_reversion_threshold,
                    'direction_bias': 'bullish' if current_value > mean_value else 'bearish'
                }
            
            oscillation_analysis[timeframe] = tf_oscillations
        
        return oscillation_analysis
    
    def _count_direction_changes(self, data_series):
        """
        Counts the number of direction changes in a data series to detect oscillations.
        """
        if len(data_series) < 3:
            return 0
            
        changes = data_series.diff().dropna()
        direction_changes = 0
        
        for i in range(1, len(changes)):
            if (changes.iloc[i-1] > 0 and changes.iloc[i] < 0) or (changes.iloc[i-1] < 0 and changes.iloc[i] > 0):
                direction_changes += 1
                
        return direction_changes
    
    def analyze_market_uncertainty(self, ufo_data_dict, oscillation_analysis):
        """
        Analyzes market uncertainty levels across timeframes to guide trading decisions.
        UFO Methodology: Higher uncertainty requires more conservative position sizing.
        """
        uncertainty_metrics = {}
        
        for timeframe in ufo_data_dict.keys():
            if timeframe not in oscillation_analysis:
                continue
                
            tf_oscillations = oscillation_analysis[timeframe]
            
            # Calculate overall uncertainty metrics
            uncertain_currencies = sum(1 for curr_data in tf_oscillations.values() 
                                     if curr_data['market_state'] == 'uncertain')
            oscillating_currencies = sum(1 for curr_data in tf_oscillations.values() 
                                       if curr_data['market_state'] == 'oscillating')
            trending_currencies = sum(1 for curr_data in tf_oscillations.values() 
                                    if curr_data['market_state'] == 'trending')
            
            total_currencies = len(tf_oscillations)
            
            if total_currencies > 0:
                uncertainty_ratio = uncertain_currencies / total_currencies
                oscillation_ratio = oscillating_currencies / total_currencies
                trend_ratio = trending_currencies / total_currencies
                
                # Overall market state determination
                if trend_ratio > 0.6:
                    overall_state = 'trending_market'
                    confidence_level = 'high'
                elif oscillation_ratio > 0.5:
                    overall_state = 'ranging_market'
                    confidence_level = 'medium'
                elif uncertainty_ratio > 0.4:
                    overall_state = 'uncertain_market'
                    confidence_level = 'low'
                else:
                    overall_state = 'mixed_market'
                    confidence_level = 'medium'
                
                uncertainty_metrics[timeframe] = {
                    'uncertainty_ratio': uncertainty_ratio,
                    'oscillation_ratio': oscillation_ratio,
                    'trend_ratio': trend_ratio,
                    'overall_state': overall_state,
                    'confidence_level': confidence_level,
                    'recommended_position_scaling': self._get_position_scaling(confidence_level)
                }
        
        return uncertainty_metrics
    
    def _get_position_scaling(self, confidence_level):
        """
        Returns position scaling factor based on market confidence level.
        UFO Methodology: Scale positions based on market certainty.
        """
        scaling_factors = {
            'high': 1.0,      # Full position sizing in trending markets
            'medium': 0.75,   # Reduced sizing in mixed/ranging markets  
            'low': 0.5        # Conservative sizing in uncertain markets
        }
        return scaling_factors.get(confidence_level, 0.5)
    
    def detect_timeframe_coherence(self, ufo_data_dict):
        """
        Enhanced coherence detection across multiple timeframes.
        UFO Methodology: Strong coherence across timeframes indicates high-probability setups.
        """
        if len(ufo_data_dict) < 2:
            return {}
            
        coherence_analysis = {}
        timeframes = list(ufo_data_dict.keys())
        
        for currency in self.currencies:
            currency_coherence = {}
            
            # Get strength values across all timeframes
            tf_strengths = {}
            for tf in timeframes:
                if currency in ufo_data_dict[tf].columns and len(ufo_data_dict[tf]) > 0:
                    tf_strengths[tf] = ufo_data_dict[tf][currency].iloc[-1]
            
            if len(tf_strengths) < 2:
                continue
                
            # Calculate coherence metrics
            strengths = list(tf_strengths.values())
            
            # Direction coherence (all positive or all negative)
            positive_count = sum(1 for s in strengths if s > 0)
            negative_count = sum(1 for s in strengths if s < 0)
            total_count = len(strengths)
            
            direction_coherence = max(positive_count, negative_count) / total_count
            
            # Magnitude coherence (similar strength levels)
            if len(strengths) > 1:
                # Normalize the strengths to a range of [0, 1]
                min_strength = np.min(strengths)
                max_strength = np.max(strengths)
                if max_strength - min_strength > 0:
                    normalized_strengths = (strengths - min_strength) / (max_strength - min_strength)
                    magnitude_coherence = 1 - np.std(normalized_strengths)
                else:
                    magnitude_coherence = 1.0
                magnitude_coherence = max(0, min(1, magnitude_coherence))  # Clamp to [0,1]
            else:
                magnitude_coherence = 1.0
            
            # Overall coherence score
            overall_coherence = (direction_coherence + magnitude_coherence) / 2
            
            # DEBUG: Log coherence calculation to understand why it's always 0%
            print(f"DEBUG COHERENCE - {currency}: strengths={strengths}, dir_coh={direction_coherence:.3f}, mag_coh={magnitude_coherence:.3f}, overall={overall_coherence:.3f}")
            
            # Coherence classification
            if overall_coherence >= 0.8:
                coherence_level = 'strong'
                print(f"DEBUG: {currency} classified as STRONG coherence ({overall_coherence:.3f})")
            elif overall_coherence >= 0.6:
                coherence_level = 'moderate'
                print(f"DEBUG: {currency} classified as MODERATE coherence ({overall_coherence:.3f})")
            else:
                coherence_level = 'weak'
                print(f"DEBUG: {currency} classified as WEAK coherence ({overall_coherence:.3f})")
            
            currency_coherence = {
                'direction_coherence': direction_coherence,
                'magnitude_coherence': magnitude_coherence,
                'overall_coherence': overall_coherence,
                'coherence_level': coherence_level,
                'timeframe_strengths': tf_strengths,
                'dominant_direction': 'bullish' if positive_count > negative_count else 'bearish'
            }
            
            coherence_analysis[currency] = currency_coherence
        
        return coherence_analysis
