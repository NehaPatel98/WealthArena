"""
Signal Fusion Module for WealthArena

This module combines numerical market data with news sentiment and other
alternative data sources to create comprehensive trading signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn

try:
    from .news_processor import NewsProcessor, NewsConfig
    from .market_data import MarketDataProcessor, TechnicalCalculator
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from data.news_processor import NewsProcessor, NewsConfig
    from data.market_data import MarketDataProcessor, TechnicalCalculator

logger = logging.getLogger(__name__)


@dataclass
class SignalFusionConfig:
    """Configuration for signal fusion"""
    # Data sources
    include_news: bool = True
    include_technical: bool = True
    include_fundamental: bool = True
    include_macro: bool = True
    
    # Fusion parameters
    news_weight: float = 0.3
    technical_weight: float = 0.4
    fundamental_weight: float = 0.2
    macro_weight: float = 0.1
    
    # Feature engineering
    use_pca: bool = True
    pca_components: int = 50
    normalize_features: bool = True
    
    # Model parameters
    fusion_model_type: str = "ensemble"  # ensemble, neural_network, linear
    hidden_layers: List[int] = None
    
    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [128, 64, 32]


class SignalFusion:
    """Signal fusion system for combining multiple data sources"""
    
    def __init__(self, config: SignalFusionConfig = None):
        self.config = config or SignalFusionConfig()
        
        # Initialize components
        self.news_processor = NewsProcessor(NewsConfig())
        self.market_processor = MarketDataProcessor({})
        self.technical_calculator = TechnicalCalculator({})
        
        # Feature engineering
        self.scaler = StandardScaler()
        self.pca = None  # Will be initialized when needed
        
        # Fusion model
        self.fusion_model = None
        self.is_trained = False
        
        logger.info("Signal fusion system initialized")
    
    def extract_technical_signals(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Extract technical analysis signals"""
        if not self.config.include_technical:
            return pd.DataFrame()
        
        try:
            # Calculate technical indicators
            technical_indicators = [
                "sma_20", "sma_50", "ema_12", "ema_26", "rsi", "macd",
                "bollinger_upper", "bollinger_lower", "atr", "obv"
            ]
            
            technical_data = self.technical_calculator.calculate_indicators(
                market_data, technical_indicators
            )
            
            # Add derived technical signals
            technical_signals = self._create_technical_signals(technical_data)
            
            logger.info(f"Extracted {len(technical_signals.columns)} technical signals")
            return technical_signals
            
        except Exception as e:
            logger.error(f"Error extracting technical signals: {e}")
            return pd.DataFrame()
    
    def _create_technical_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create derived technical signals"""
        signals = pd.DataFrame(index=data.index)
        
        # Price momentum signals
        if 'Close' in data.columns:
            signals['price_momentum_5'] = data['Close'].pct_change(5)
            signals['price_momentum_20'] = data['Close'].pct_change(20)
            signals['price_volatility'] = data['Close'].rolling(20).std()
        
        # Moving average signals
        if 'sma_20' in data.columns and 'sma_50' in data.columns:
            signals['sma_cross'] = (data['sma_20'] - data['sma_50']) / data['sma_50']
            signals['price_vs_sma20'] = (data['Close'] - data['sma_20']) / data['sma_20']
        
        # RSI signals
        if 'rsi' in data.columns:
            signals['rsi_oversold'] = (data['rsi'] < 30).astype(int)
            signals['rsi_overbought'] = (data['rsi'] > 70).astype(int)
            signals['rsi_momentum'] = data['rsi'].diff()
        
        # MACD signals
        if 'macd' in data.columns:
            signals['macd_signal'] = data['macd'].diff()
            signals['macd_histogram'] = data['macd'].rolling(3).mean()
        
        # Bollinger Bands signals
        if all(col in data.columns for col in ['bollinger_upper', 'bollinger_lower', 'Close']):
            bb_width = (data['bollinger_upper'] - data['bollinger_lower']) / data['Close']
            signals['bb_width'] = bb_width
            signals['bb_position'] = (data['Close'] - data['bollinger_lower']) / (data['bollinger_upper'] - data['bollinger_lower'])
        
        return signals.fillna(0)
    
    def extract_news_signals(self, symbols: List[str], market_data: pd.DataFrame) -> pd.DataFrame:
        """Extract news sentiment signals"""
        if not self.config.include_news:
            return pd.DataFrame()
        
        try:
            # Get market sentiment
            if self.news_processor is None:
                logger.warning("News processor not initialized, using default sentiment")
                market_sentiment = {symbol: 0.0 for symbol in symbols}
            else:
                market_sentiment = self.news_processor.get_market_sentiment(symbols)
            
            # Create news signals DataFrame
            news_signals = pd.DataFrame(index=market_data.index)
            
            # Add sentiment signals for each symbol
            for symbol in symbols:
                if symbol in market_sentiment:
                    sentiment = market_sentiment[symbol]
                    news_signals[f'{symbol}_sentiment'] = sentiment
                    news_signals[f'{symbol}_sentiment_ma'] = sentiment  # Will be filled with MA later
                else:
                    news_signals[f'{symbol}_sentiment'] = 0.0
                    news_signals[f'{symbol}_sentiment_ma'] = 0.0
            
            # Add overall market sentiment
            if market_sentiment:
                overall_sentiment = np.mean(list(market_sentiment.values()))
                news_signals['overall_sentiment'] = overall_sentiment
                news_signals['sentiment_volatility'] = np.std(list(market_sentiment.values()))
            else:
                news_signals['overall_sentiment'] = 0.0
                news_signals['sentiment_volatility'] = 0.0
            
            # Add moving averages
            for col in news_signals.columns:
                if col.endswith('_sentiment'):
                    ma_col = col.replace('_sentiment', '_sentiment_ma')
                    news_signals[ma_col] = news_signals[col].rolling(5).mean()
            
            logger.info(f"Extracted {len(news_signals.columns)} news signals")
            return news_signals.fillna(0)
            
        except Exception as e:
            logger.error(f"Error extracting news signals: {e}")
            return pd.DataFrame()
    
    def extract_fundamental_signals(self, symbols: List[str], market_data: pd.DataFrame) -> pd.DataFrame:
        """Extract fundamental analysis signals"""
        if not self.config.include_fundamental:
            return pd.DataFrame()
        
        try:
            fundamental_signals = pd.DataFrame(index=market_data.index)
            
            # Simulate fundamental data (in production, fetch from financial APIs)
            for symbol in symbols:
                # Simulate P/E ratio, earnings growth, etc.
                pe_ratio = np.random.normal(20, 5)  # Simulated P/E
                earnings_growth = np.random.normal(0.1, 0.05)  # Simulated growth
                debt_ratio = np.random.normal(0.3, 0.1)  # Simulated debt ratio
                
                fundamental_signals[f'{symbol}_pe_ratio'] = pe_ratio
                fundamental_signals[f'{symbol}_earnings_growth'] = earnings_growth
                fundamental_signals[f'{symbol}_debt_ratio'] = debt_ratio
                
                # Derived fundamental signals
                fundamental_signals[f'{symbol}_pe_momentum'] = 0  # Would calculate from historical P/E
                fundamental_signals[f'{symbol}_growth_momentum'] = 0  # Would calculate from historical growth
            
            logger.info(f"Extracted {len(fundamental_signals.columns)} fundamental signals")
            return fundamental_signals.fillna(0)
            
        except Exception as e:
            logger.error(f"Error extracting fundamental signals: {e}")
            return pd.DataFrame()
    
    def extract_macro_signals(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Extract macroeconomic signals"""
        if not self.config.include_macro:
            return pd.DataFrame()
        
        try:
            macro_signals = pd.DataFrame(index=market_data.index)
            
            # Simulate macro indicators (in production, fetch from FRED, etc.)
            macro_signals['interest_rate'] = np.random.normal(0.05, 0.01)  # Simulated interest rate
            macro_signals['inflation_rate'] = np.random.normal(0.03, 0.005)  # Simulated inflation
            macro_signals['gdp_growth'] = np.random.normal(0.025, 0.01)  # Simulated GDP growth
            macro_signals['unemployment'] = np.random.normal(0.05, 0.01)  # Simulated unemployment
            
            # Add derived macro signals
            macro_signals['yield_curve'] = 0  # Would calculate yield curve slope
            macro_signals['risk_premium'] = 0  # Would calculate risk premium
            
            logger.info(f"Extracted {len(macro_signals.columns)} macro signals")
            return macro_signals.fillna(0)
            
        except Exception as e:
            logger.error(f"Error extracting macro signals: {e}")
            return pd.DataFrame()
    
    def fuse_signals(self, 
                    technical_signals: pd.DataFrame,
                    news_signals: pd.DataFrame,
                    fundamental_signals: pd.DataFrame,
                    macro_signals: pd.DataFrame) -> pd.DataFrame:
        """Fuse all signals into a comprehensive feature set"""
        
        try:
            # Combine all signals
            all_signals = []
            signal_names = []
            
            if not technical_signals.empty:
                all_signals.append(technical_signals)
                signal_names.append("technical")
            
            if not news_signals.empty:
                all_signals.append(news_signals)
                signal_names.append("news")
            
            if not fundamental_signals.empty:
                all_signals.append(fundamental_signals)
                signal_names.append("fundamental")
            
            if not macro_signals.empty:
                all_signals.append(macro_signals)
                signal_names.append("macro")
            
            if not all_signals:
                logger.warning("No signals to fuse")
                return pd.DataFrame()
            
            # Align all DataFrames to common index
            common_index = all_signals[0].index
            for signals in all_signals[1:]:
                common_index = common_index.intersection(signals.index)
            
            if len(common_index) == 0:
                logger.warning("No common index found for signal fusion")
                return pd.DataFrame()
            
            # Align all signals to common index
            aligned_signals = []
            for signals in all_signals:
                aligned_signals.append(signals.loc[common_index])
            
            # Concatenate all signals
            fused_signals = pd.concat(aligned_signals, axis=1)
            
            # Apply weights
            weighted_signals = self._apply_signal_weights(fused_signals, signal_names)
            
            # Normalize features
            if self.config.normalize_features:
                weighted_signals = pd.DataFrame(
                    self.scaler.fit_transform(weighted_signals),
                    index=weighted_signals.index,
                    columns=weighted_signals.columns
                )
            
            # Apply PCA if enabled
            if self.config.use_pca and len(weighted_signals.columns) > 0:
                # Initialize PCA with appropriate number of components
                n_components = min(self.config.pca_components, len(weighted_signals.columns), len(weighted_signals))
                if n_components > 0:
                    self.pca = PCA(n_components=n_components)
                    pca_features = self.pca.fit_transform(weighted_signals)
                    pca_columns = [f"pca_{i}" for i in range(pca_features.shape[1])]
                    fused_signals = pd.DataFrame(
                        pca_features,
                        index=weighted_signals.index,
                        columns=pca_columns
                    )
                else:
                    fused_signals = weighted_signals
            else:
                fused_signals = weighted_signals
            
            logger.info(f"Fused signals: {fused_signals.shape}")
            return fused_signals.fillna(0)
            
        except Exception as e:
            logger.error(f"Error fusing signals: {e}")
            return pd.DataFrame()
    
    def _apply_signal_weights(self, signals: pd.DataFrame, signal_names: List[str]) -> pd.DataFrame:
        """Apply weights to different signal types"""
        weighted_signals = signals.copy()
        
        # Define weights
        weights = {
            "technical": self.config.technical_weight,
            "news": self.config.news_weight,
            "fundamental": self.config.fundamental_weight,
            "macro": self.config.macro_weight
        }
        
        # Apply weights to signal columns
        for signal_type in signal_names:
            if signal_type in weights:
                weight = weights[signal_type]
                # Find columns belonging to this signal type
                type_columns = [col for col in signals.columns if signal_type in col.lower()]
                for col in type_columns:
                    weighted_signals[col] = weighted_signals[col] * weight
        
        return weighted_signals
    
    def create_trading_signals(self, fused_signals: pd.DataFrame) -> pd.DataFrame:
        """Create final trading signals from fused features"""
        
        if fused_signals.empty:
            return pd.DataFrame()
        
        try:
            trading_signals = pd.DataFrame(index=fused_signals.index)
            
            # Create composite signals
            if self.config.fusion_model_type == "ensemble":
                trading_signals = self._create_ensemble_signals(fused_signals)
            elif self.config.fusion_model_type == "neural_network":
                trading_signals = self._create_neural_signals(fused_signals)
            else:  # linear
                trading_signals = self._create_linear_signals(fused_signals)
            
            # Add signal metadata
            trading_signals['signal_strength'] = trading_signals.abs().sum(axis=1)
            trading_signals['signal_confidence'] = self._calculate_signal_confidence(fused_signals)
            
            logger.info(f"Created {len(trading_signals.columns)} trading signals")
            return trading_signals.fillna(0)
            
        except Exception as e:
            logger.error(f"Error creating trading signals: {e}")
            return pd.DataFrame()
    
    def _create_ensemble_signals(self, fused_signals: pd.DataFrame) -> pd.DataFrame:
        """Create ensemble-based trading signals"""
        signals = pd.DataFrame(index=fused_signals.index)
        
        # Simple ensemble of different signal types
        if 'overall_sentiment' in fused_signals.columns:
            signals['sentiment_signal'] = fused_signals['overall_sentiment']
        
        if 'price_momentum_20' in fused_signals.columns:
            signals['momentum_signal'] = fused_signals['price_momentum_20']
        
        if 'rsi' in fused_signals.columns:
            signals['rsi_signal'] = (fused_signals['rsi'] - 50) / 50
        
        # Combine signals
        if len(signals.columns) > 1:
            signals['combined_signal'] = signals.mean(axis=1)
        else:
            signals['combined_signal'] = signals.iloc[:, 0] if len(signals.columns) > 0 else 0
        
        return signals
    
    def _create_neural_signals(self, fused_signals: pd.DataFrame) -> pd.DataFrame:
        """Create neural network-based trading signals"""
        # This would use a trained neural network
        # For now, use simple linear combination
        return self._create_linear_signals(fused_signals)
    
    def _create_linear_signals(self, fused_signals: pd.DataFrame) -> pd.DataFrame:
        """Create linear combination trading signals"""
        signals = pd.DataFrame(index=fused_signals.index)
        
        # Simple linear combination
        signals['linear_signal'] = fused_signals.mean(axis=1)
        
        return signals
    
    def _calculate_signal_confidence(self, fused_signals: pd.DataFrame) -> pd.Series:
        """Calculate confidence score for signals"""
        # Simple confidence based on signal consistency
        if fused_signals.empty:
            return pd.Series(0, index=fused_signals.index)
        
        # Calculate correlation between different signal types
        correlations = fused_signals.corr()
        avg_correlation = correlations.mean().mean()
        
        # Confidence based on correlation (higher correlation = higher confidence)
        confidence = np.clip(avg_correlation, 0, 1)
        
        return pd.Series(confidence, index=fused_signals.index)
    
    def process_market_data_with_signals(self, 
                                       market_data: pd.DataFrame,
                                       symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Process market data and extract all signals"""
        
        try:
            # Extract different signal types
            technical_signals = self.extract_technical_signals(market_data)
            news_signals = self.extract_news_signals(symbols, market_data)
            fundamental_signals = self.extract_fundamental_signals(symbols, market_data)
            macro_signals = self.extract_macro_signals(market_data)
            
            # Fuse all signals
            fused_signals = self.fuse_signals(
                technical_signals, news_signals, 
                fundamental_signals, macro_signals
            )
            
            # Create trading signals
            trading_signals = self.create_trading_signals(fused_signals)
            
            return {
                "technical": technical_signals,
                "news": news_signals,
                "fundamental": fundamental_signals,
                "macro": macro_signals,
                "fused": fused_signals,
                "trading": trading_signals
            }
            
        except Exception as e:
            logger.error(f"Error processing market data with signals: {e}")
            # Return empty DataFrames for each signal type
            return {
                "technical": pd.DataFrame(),
                "news": pd.DataFrame(),
                "fundamental": pd.DataFrame(),
                "macro": pd.DataFrame(),
                "fused": pd.DataFrame(),
                "trading": pd.DataFrame()
            }


def create_signal_fusion(config: SignalFusionConfig = None) -> SignalFusion:
    """Create a signal fusion instance"""
    return SignalFusion(config)


def main():
    """Test the signal fusion system"""
    logging.basicConfig(level=logging.INFO)
    
    print("🔄 Testing Signal Fusion System...")
    print("="*40)
    
    # Create signal fusion
    config = SignalFusionConfig()
    signal_fusion = SignalFusion(config)
    
    # Create sample market data
    dates = pd.date_range(start="2024-01-01", end="2024-01-31", freq='D')
    market_data = pd.DataFrame({
        'Open': np.random.randn(len(dates)) * 100 + 1000,
        'High': np.random.randn(len(dates)) * 100 + 1050,
        'Low': np.random.randn(len(dates)) * 100 + 950,
        'Close': np.random.randn(len(dates)) * 100 + 1000,
        'Volume': np.random.randint(1000000, 5000000, len(dates))
    }, index=dates)
    
    # Test with sample symbols
    symbols = ["AAPL", "MSFT", "GOOGL"]
    
    print(f"\n📊 Processing market data for symbols: {symbols}")
    results = signal_fusion.process_market_data_with_signals(market_data, symbols)
    
    print(f"\n✅ Signal extraction results:")
    for signal_type, data in results.items():
        if not data.empty:
            print(f"  {signal_type}: {data.shape} features")
        else:
            print(f"  {signal_type}: No data")
    
    if 'trading' in results and not results['trading'].empty:
        print(f"\n🎯 Trading signals created:")
        print(f"  Columns: {list(results['trading'].columns)}")
        print(f"  Sample values:")
        print(results['trading'].head())
    
    print("\n✅ Signal fusion test completed!")


if __name__ == "__main__":
    main()
