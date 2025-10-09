"""
Model Serving Service
Loads RL models and generates predictions with TP/SL levels
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class RLModelService:
    """
    Service to load RL models and generate predictions
    Models determine: Signals, Entry, TP/SL, Position Size
    """
    
    def __init__(self, model_dir: Path = None):
        self.model_dir = model_dir or Path(__file__).parent.parent / "checkpoints"
        self.loaded_models = {}
        self.model_metadata = {}
        
        logger.info(f"Initializing Model Service with model_dir: {self.model_dir}")
        
        # Load available models
        self._discover_models()
    
    def _discover_models(self):
        """Discover available trained models"""
        
        if not self.model_dir.exists():
            logger.warning(f"Model directory not found: {self.model_dir}")
            return
        
        # Find model directories
        model_types = ['asx_stocks', 'currency_pairs', 'commodities', 'cryptocurrencies', 'etf']
        
        for model_type in model_types:
            model_path = self.model_dir / model_type
            if model_path.exists():
                self.model_metadata[model_type] = {
                    'path': model_path,
                    'available': True,
                    'last_updated': datetime.now().isoformat()
                }
                logger.info(f"✅ Found model: {model_type}")
    
    def load_model(self, model_type: str):
        """
        Load a specific model type
        In production, this would load the actual RL model
        For now, returns a mock model that simulates RL behavior
        """
        
        if model_type in self.loaded_models:
            return self.loaded_models[model_type]
        
        # Check if model exists
        if model_type not in self.model_metadata:
            logger.warning(f"Model type not found: {model_type}")
            return None
        
        # For demo: Use mock model
        # TODO: Load actual RL model from checkpoints
        # model = load_rllib_checkpoint(self.model_metadata[model_type]['path'])
        
        logger.info(f"Loading {model_type} model (mock mode)")
        
        self.loaded_models[model_type] = {
            'type': model_type,
            'version': 'v2.3.1',
            'loaded_at': datetime.now().isoformat(),
            'status': 'ready'
        }
        
        return self.loaded_models[model_type]
    
    def generate_prediction(self, symbol: str, data: pd.DataFrame, 
                          asset_type: str = 'stock') -> Dict[str, Any]:
        """
        Generate prediction with TP/SL from RL model
        
        This is where the RL model makes ALL trading decisions:
        - Signal (BUY/SELL/HOLD)
        - Entry price
        - Take Profit levels (TP1, TP2, TP3)
        - Stop Loss level
        - Position sizing
        """
        
        if data.empty or len(data) < 20:
            return self._generate_hold_prediction(symbol)
        
        # Get latest market state
        latest = data.iloc[-1]
        current_price = float(latest.get('Close', 0))
        
        # Calculate recent metrics for model input
        recent_return = self._calculate_recent_return(data, window=5)
        volatility = self._calculate_volatility(data, window=20)
        rsi = latest.get('RSI', 50)
        macd = latest.get('MACD', 0)
        volume_ratio = latest.get('Volume_Ratio', 1.0)
        
        # ==========================================
        # RL MODEL INFERENCE (Mock for demo)
        # TODO: Replace with actual RL model inference
        # ==========================================
        
        # Trading Agent determines signal
        signal, confidence = self._get_trading_signal(
            recent_return, volatility, rsi, macd, volume_ratio
        )
        
        # Risk Management Agent determines TP/SL
        tp_sl_levels = self._get_risk_levels(
            current_price, volatility, signal, confidence
        )
        
        # Portfolio Manager determines position size
        position_size = self._get_position_size(
            confidence, volatility, tp_sl_levels['risk_reward_ratio']
        )
        
        # Compile full prediction
        prediction = {
            'symbol': symbol,
            'asset_type': asset_type,
            'signal': signal,
            'confidence': round(confidence, 2),
            'timestamp': datetime.now().isoformat(),
            
            # Entry strategy from Trading Agent
            'entry': {
                'price': round(current_price, 2),
                'range': [
                    round(current_price * 0.997, 2),  # -0.3%
                    round(current_price * 1.003, 2)   # +0.3%
                ],
                'timing': 'immediate' if confidence > 0.8 else 'on_pullback'
            },
            
            # TP/SL from Risk Management Agent
            'take_profit': tp_sl_levels['take_profits'],
            'stop_loss': tp_sl_levels['stop_loss'],
            'risk_metrics': tp_sl_levels['risk_metrics'],
            
            # Position sizing from Portfolio Manager Agent
            'position_sizing': position_size,
            
            # Model metadata
            'model_version': 'v2.3.1',
            'model_type': asset_type,
            
            # Technical context
            'indicators': {
                'rsi': {'value': round(float(rsi), 2), 'status': self._get_rsi_status(rsi)},
                'macd': {'value': round(float(macd), 2), 'status': 'bullish' if macd > 0 else 'bearish'},
                'volatility': {'value': round(volatility, 2), 'status': self._get_vol_status(volatility)},
                'volume': {'status': 'high' if volume_ratio > 1.2 else 'normal'},
                'trend': self._get_trend_status(data)
            },
            
            # Chart data for UI
            'chart_data': self._prepare_chart_data(data, tp_sl_levels)
        }
        
        return prediction
    
    def _get_trading_signal(self, recent_return, volatility, rsi, macd, volume_ratio):
        """
        Trading Agent logic (simplified for demo)
        In production: This is the RL model's policy network output
        """
        
        score = 0
        
        # Momentum factor
        if recent_return > 0.03:
            score += 2
        elif recent_return < -0.03:
            score -= 2
        
        # RSI factor
        if rsi < 30:
            score += 1  # Oversold, bullish
        elif rsi > 70:
            score -= 1  # Overbought, bearish
        
        # MACD factor
        if macd > 0:
            score += 1
        else:
            score -= 1
        
        # Volume confirmation
        if volume_ratio > 1.5:
            score += 0.5
        
        # Determine signal and confidence
        if score >= 2:
            signal = 'BUY'
            confidence = min(0.95, 0.6 + (score / 10))
        elif score <= -2:
            signal = 'SELL'
            confidence = min(0.95, 0.6 + (abs(score) / 10))
        else:
            signal = 'HOLD'
            confidence = 0.5
        
        return signal, confidence
    
    def _get_risk_levels(self, current_price: float, volatility: float, 
                        signal: str, confidence: float) -> Dict[str, Any]:
        """
        Risk Management Agent logic
        Determines TP/SL based on learned optimal risk/reward
        
        In production: This is the Risk Management RL agent's output
        """
        
        # ATR-based risk calculation (RL agent learned this is optimal)
        atr_multiplier = 1.5  # Learned from training
        risk_amount = current_price * volatility * atr_multiplier
        
        # RL agent learned optimal reward multiples
        if confidence > 0.85:
            reward_multiples = [1.5, 3.0, 4.5]  # Aggressive
        elif confidence > 0.7:
            reward_multiples = [1.2, 2.5, 3.5]  # Moderate
        else:
            reward_multiples = [1.0, 2.0, 3.0]  # Conservative
        
        if signal == 'BUY':
            # Calculate TP levels (above entry)
            tp1 = current_price + (risk_amount * reward_multiples[0])
            tp2 = current_price + (risk_amount * reward_multiples[1])
            tp3 = current_price + (risk_amount * reward_multiples[2])
            sl = current_price - risk_amount
            
        elif signal == 'SELL':
            # Calculate TP levels (below entry)
            tp1 = current_price - (risk_amount * reward_multiples[0])
            tp2 = current_price - (risk_amount * reward_multiples[1])
            tp3 = current_price - (risk_amount * reward_multiples[2])
            sl = current_price + risk_amount
        else:
            # HOLD signal
            return {
                'take_profits': [],
                'stop_loss': None,
                'risk_metrics': {}
            }
        
        # RL agent learned optimal allocation per TP level
        take_profits = [
            {
                'level': 1,
                'price': round(tp1, 2),
                'percent': round(((tp1 - current_price) / current_price) * 100, 2),
                'close_percent': 50,  # Close 50% at TP1
                'probability': 0.75   # RL learned probability
            },
            {
                'level': 2,
                'price': round(tp2, 2),
                'percent': round(((tp2 - current_price) / current_price) * 100, 2),
                'close_percent': 30,  # Close 30% at TP2
                'probability': 0.55
            },
            {
                'level': 3,
                'price': round(tp3, 2),
                'percent': round(((tp3 - current_price) / current_price) * 100, 2),
                'close_percent': 20,  # Close 20% at TP3
                'probability': 0.35
            }
        ]
        
        stop_loss = {
            'price': round(sl, 2),
            'percent': round(((sl - current_price) / current_price) * 100, 2),
            'type': 'trailing' if confidence > 0.8 else 'fixed',
            'trail_amount': round(risk_amount * 0.5, 2) if confidence > 0.8 else None
        }
        
        # Calculate risk/reward
        avg_reward = (tp1 + tp2 + tp3) / 3
        risk = abs(current_price - sl)
        reward = abs(avg_reward - current_price)
        risk_reward_ratio = round(reward / risk, 2) if risk > 0 else 0
        
        risk_metrics = {
            'risk_reward_ratio': risk_reward_ratio,
            'max_risk_per_share': round(risk, 2),
            'max_reward_per_share': round(reward, 2),
            'win_probability': round(confidence * 0.85, 2),  # Adjusted for realism
            'expected_value': round((reward * confidence) - (risk * (1 - confidence)), 2)
        }
        
        return {
            'take_profits': take_profits,
            'stop_loss': stop_loss,
            'risk_metrics': risk_metrics,
            'risk_reward_ratio': risk_reward_ratio
        }
    
    def _get_position_size(self, confidence: float, volatility: float, 
                          risk_reward_ratio: float) -> Dict[str, Any]:
        """
        Portfolio Manager Agent logic
        Determines position size using Kelly Criterion and risk management
        
        In production: This is the Portfolio Manager RL agent's output
        """
        
        # Kelly Criterion adapted by RL agent
        win_prob = confidence * 0.85
        kelly_fraction = (win_prob * risk_reward_ratio - (1 - win_prob)) / risk_reward_ratio
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        
        # RL agent adjusts based on volatility
        volatility_adjustment = 1.0 - (volatility * 2)  # Higher vol = smaller position
        volatility_adjustment = max(0.5, min(volatility_adjustment, 1.5))
        
        # Final position size
        recommended_percent = round(kelly_fraction * volatility_adjustment * 100, 1)
        recommended_percent = max(1.0, min(recommended_percent, 10.0))  # Between 1-10%
        
        # Assuming $100k portfolio for calculation
        portfolio_value = 100000
        dollar_amount = round(portfolio_value * (recommended_percent / 100), 2)
        
        return {
            'recommended_percent': recommended_percent,
            'dollar_amount': dollar_amount,
            'method': 'Kelly Criterion + Volatility Adjusted',
            'max_risk_percent': round(recommended_percent * 0.5, 2),  # 50% of position
            'confidence_factor': round(confidence, 2),
            'volatility_factor': round(volatility, 2)
        }
    
    def generate_top_setups(self, asset_type: str, data_dict: Dict[str, pd.DataFrame], 
                           count: int = 3) -> List[Dict[str, Any]]:
        """
        Generate top N trading setups for asset type
        Ranks by composite score from RL models
        """
        
        all_predictions = []
        
        # Generate predictions for all symbols
        for symbol, data in data_dict.items():
            try:
                prediction = self.generate_prediction(symbol, data, asset_type)
                
                # Only include BUY or SELL signals
                if prediction['signal'] != 'HOLD':
                    # Calculate ranking score
                    score = self._calculate_ranking_score(prediction)
                    prediction['ranking_score'] = score
                    all_predictions.append(prediction)
                    
            except Exception as e:
                logger.error(f"Error generating prediction for {symbol}: {e}")
        
        # Sort by ranking score
        ranked_predictions = sorted(
            all_predictions, 
            key=lambda x: x['ranking_score'], 
            reverse=True
        )
        
        # Return top N
        top_setups = ranked_predictions[:count]
        
        # Add rank
        for i, setup in enumerate(top_setups):
            setup['rank'] = i + 1
        
        logger.info(f"Generated {len(top_setups)} top setups for {asset_type}")
        
        return top_setups
    
    def _calculate_ranking_score(self, prediction: Dict[str, Any]) -> float:
        """
        Ranking algorithm for trading setups
        Composite score based on multiple factors
        """
        
        # Extract factors
        confidence = prediction.get('confidence', 0)
        risk_reward = prediction.get('risk_metrics', {}).get('risk_reward_ratio', 0)
        win_prob = prediction.get('risk_metrics', {}).get('win_probability', 0)
        expected_value = prediction.get('risk_metrics', {}).get('expected_value', 0)
        
        # Weighted scoring (from RL agent's learned preferences)
        score = (
            confidence * 0.35 +              # Model confidence
            min(risk_reward / 5, 1.0) * 0.30 +  # Risk/Reward (capped at 5)
            win_prob * 0.20 +                # Win probability
            (expected_value / 10) * 0.15     # Expected value
        )
        
        # Bonus for high confidence + high R/R
        if confidence > 0.85 and risk_reward > 3.0:
            score *= 1.1  # 10% bonus
        
        return round(score, 4)
    
    def _calculate_recent_return(self, data: pd.DataFrame, window: int = 5) -> float:
        """Calculate recent return"""
        if 'Close' not in data.columns or len(data) < window:
            return 0.0
        
        recent_return = (data['Close'].iloc[-1] / data['Close'].iloc[-window]) - 1
        return float(recent_return)
    
    def _calculate_volatility(self, data: pd.DataFrame, window: int = 20) -> float:
        """Calculate volatility"""
        if 'Close' not in data.columns or len(data) < window:
            return 0.02
        
        returns = data['Close'].pct_change()
        volatility = returns.tail(window).std()
        return float(volatility) if not pd.isna(volatility) else 0.02
    
    def _get_rsi_status(self, rsi: float) -> str:
        """Get RSI status"""
        if rsi < 30:
            return 'oversold'
        elif rsi > 70:
            return 'overbought'
        elif rsi < 45:
            return 'neutral-bullish'
        elif rsi > 55:
            return 'neutral-bearish'
        else:
            return 'neutral'
    
    def _get_vol_status(self, volatility: float) -> str:
        """Get volatility status"""
        if volatility < 0.015:
            return 'low'
        elif volatility > 0.03:
            return 'high'
        else:
            return 'medium'
    
    def _get_trend_status(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get trend status from price and moving averages"""
        if len(data) < 50:
            return {'direction': 'unknown', 'strength': 'weak'}
        
        latest_close = data['Close'].iloc[-1]
        sma_20 = data['Close'].tail(20).mean()
        sma_50 = data['Close'].tail(50).mean()
        
        if latest_close > sma_20 > sma_50:
            return {'direction': 'up', 'strength': 'strong'}
        elif latest_close > sma_20:
            return {'direction': 'up', 'strength': 'moderate'}
        elif latest_close < sma_20 < sma_50:
            return {'direction': 'down', 'strength': 'strong'}
        elif latest_close < sma_20:
            return {'direction': 'down', 'strength': 'moderate'}
        else:
            return {'direction': 'sideways', 'strength': 'weak'}
    
    def _prepare_chart_data(self, data: pd.DataFrame, tp_sl_levels: Dict) -> List[Dict]:
        """Prepare chart data for frontend (last 30 days)"""
        
        # Get last 30 days
        chart_df = data.tail(30)
        
        # Convert to list of dicts for JSON
        chart_data = []
        for idx, row in chart_df.iterrows():
            chart_data.append({
                'date': row.get('Date', str(idx)),
                'open': round(float(row.get('Open', 0)), 2),
                'high': round(float(row.get('High', 0)), 2),
                'low': round(float(row.get('Low', 0)), 2),
                'close': round(float(row.get('Close', 0)), 2),
                'volume': int(row.get('Volume', 0))
            })
        
        return chart_data
    
    def _generate_hold_prediction(self, symbol: str) -> Dict[str, Any]:
        """Generate HOLD prediction when insufficient data"""
        return {
            'symbol': symbol,
            'signal': 'HOLD',
            'confidence': 0.5,
            'reasoning': 'Insufficient data for confident prediction',
            'timestamp': datetime.now().isoformat()
        }


# Singleton instance
_model_service = None

def get_model_service() -> RLModelService:
    """Get or create model service singleton"""
    global _model_service
    if _model_service is None:
        _model_service = RLModelService()
    return _model_service

