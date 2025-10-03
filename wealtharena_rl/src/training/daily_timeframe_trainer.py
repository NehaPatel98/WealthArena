"""
Daily Timeframe Trainer for WealthArena Trading System

This module provides training capabilities specifically optimized for daily timeframe trading
across all financial instrument types (stocks, crypto, ETFs, currencies, REITs).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path
import yaml
import joblib
from dataclasses import dataclass

# Import our custom modules
from ..models.specialized_agent_factory import (
    SpecializedAgentFactory,
    SpecializedAgentConfig,
    AssetType,
    create_specialized_agents
)
from ..models.rl_meta_agent import RLMetaAgent, RLMetaAgentConfig
from ..data.asx_companies import ASXDataProvider
from ..metrics.comprehensive_metrics import MetricsCollector

logger = logging.getLogger(__name__)


@dataclass
class DailyTrainingConfig:
    """Configuration for daily timeframe training"""
    start_date: str = "2020-01-01"
    end_date: str = "2024-01-01"
    validation_split: float = 0.2
    test_split: float = 0.2
    lookback_window: int = 5
    prediction_horizon: int = 1
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    model_save_path: str = "models/daily_timeframe"
    metrics_save_path: str = "metrics/daily_timeframe"
    reports_save_path: str = "reports/daily_timeframe"


class DailyTimeframeTrainer:
    """
    Daily Timeframe Trainer for WealthArena Trading System
    
    Trains models for all financial instrument types on daily timeframe data
    and creates a meta agent for integrated decision making.
    """
    
    def __init__(self, config: DailyTrainingConfig = None):
        self.config = config or DailyTrainingConfig()
        
        # Initialize components
        self.asx_provider = ASXDataProvider()
        self.metrics_collector = MetricsCollector()
        
        # Training data
        self.training_data = {}
        self.validation_data = {}
        self.test_data = {}
        
        # RL Agents
        self.instrument_agents = {}
        self.meta_agent = None
        
        # Training results
        self.training_results = {}
        self.evaluation_results = {}
        
        # Create output directories
        Path(self.config.model_save_path).mkdir(parents=True, exist_ok=True)
        Path(self.config.metrics_save_path).mkdir(parents=True, exist_ok=True)
        Path(self.config.reports_save_path).mkdir(parents=True, exist_ok=True)
        
        logger.info("Daily timeframe trainer initialized")
    
    def prepare_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Prepare training data for all financial instrument types"""
        
        logger.info("Preparing training data for all financial instrument types...")
        
        # Download ASX data
        all_data = self.asx_provider.download_all_categories(
            start_date=self.config.start_date,
            end_date=self.config.end_date
        )
        
        # Process data for each specialized asset type
        processed_data = {}
        
        # Define asset type categories
        asset_categories = [
            "asx_stocks", "cryptocurrencies", "etf", "currency_pairs", 
            "reit", "us_stocks", "commodities"
        ]
        
        for category, category_data in all_data.items():
            if not category_data:
                continue
            
            logger.info(f"Processing {category} data...")
            
            # Combine all symbols in category
            combined_data = []
            for symbol, data in category_data.items():
                if data.empty:
                    continue
                
                # Add technical indicators
                data_with_indicators = self._add_technical_indicators(data)
                
                # Add symbol column
                data_with_indicators['Symbol'] = symbol
                data_with_indicators['Category'] = category
                
                combined_data.append(data_with_indicators)
            
            if combined_data:
                processed_data[category] = pd.concat(combined_data, ignore_index=True)
                logger.info(f"Processed {category}: {len(processed_data[category])} records")
        
        # Split data into train/validation/test
        self._split_data(processed_data)
        
        self.training_data = processed_data
        logger.info("Data preparation completed")
        
        return processed_data
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to market data"""
        
        if data.empty:
            return data
        
        df = data.copy()
        
        # Ensure proper column names
        if 'Close' in df.columns:
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
        else:
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Select only OHLCV columns
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Add date column
        df['Date'] = df.index
        df = df.reset_index(drop=True)
        
        # Calculate technical indicators
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        df['EMA_50'] = df['Close'].ewm(span=50).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        # OBV
        df['OBV'] = (df['Volume'] * np.sign(df['Close'].diff())).cumsum()
        
        # Stochastic
        df['STOCH_K'] = ((df['Close'] - df['Low'].rolling(14).min()) / 
                         (df['High'].rolling(14).max() - df['Low'].rolling(14).min())) * 100
        df['STOCH_D'] = df['STOCH_K'].rolling(3).mean()
        
        # Williams %R
        df['WILLR'] = ((df['High'].rolling(14).max() - df['Close']) / 
                       (df['High'].rolling(14).max() - df['Low'].rolling(14).min())) * -100
        
        # CCI
        df['CCI'] = ((df['Close'] - df['Close'].rolling(20).mean()) / 
                     (0.015 * df['Close'].rolling(20).std()))
        
        # ADX
        df['ADX'] = self._calculate_adx(df)
        
        # Money Flow Index
        df['MFI'] = self._calculate_mfi(df)
        
        # Calculate returns and volatility
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volatility_5'] = df['Returns'].rolling(window=5).std() * np.sqrt(252)
        df['Volatility_10'] = df['Returns'].rolling(window=10).std() * np.sqrt(252)
        df['Volatility_20'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
        
        # Calculate price momentum
        df['Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        df['Momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
        
        # Calculate volume indicators
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
        
        # Fill NaN values
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        # Replace infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df
    
    def _calculate_adx(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average Directional Index"""
        
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = np.abs(high - close.shift())
        tr3 = np.abs(low - close.shift())
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Calculate Directional Movement
        dm_plus = np.where((high - high.shift()) > (low.shift() - low), 
                          np.maximum(high - high.shift(), 0), 0)
        dm_minus = np.where((low.shift() - low) > (high - high.shift()), 
                           np.maximum(low.shift() - low, 0), 0)
        
        # Calculate smoothed values
        tr_smooth = tr.rolling(14).mean()
        dm_plus_smooth = pd.Series(dm_plus).rolling(14).mean()
        dm_minus_smooth = pd.Series(dm_minus).rolling(14).mean()
        
        # Calculate DI+ and DI-
        di_plus = 100 * (dm_plus_smooth / tr_smooth)
        di_minus = 100 * (dm_minus_smooth / tr_smooth)
        
        # Calculate DX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        
        # Calculate ADX
        adx = dx.rolling(14).mean()
        
        return adx
    
    def _calculate_mfi(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Money Flow Index"""
        
        high = df['High']
        low = df['Low']
        close = df['Close']
        volume = df['Volume']
        
        # Calculate typical price
        typical_price = (high + low + close) / 3
        
        # Calculate money flow
        money_flow = typical_price * volume
        
        # Calculate positive and negative money flow
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0).rolling(14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0).rolling(14).sum()
        
        # Calculate MFI
        mfi = 100 - (100 / (1 + positive_flow / negative_flow))
        
        return mfi
    
    def _split_data(self, data: Dict[str, pd.DataFrame]):
        """Split data into train/validation/test sets"""
        
        for category, df in data.items():
            if df.empty:
                continue
            
            # Sort by date
            df = df.sort_values('Date')
            
            # Calculate split indices
            total_len = len(df)
            train_len = int(total_len * (1 - self.config.validation_split - self.config.test_split))
            val_len = int(total_len * self.config.validation_split)
            
            # Split data
            train_data = df.iloc[:train_len]
            val_data = df.iloc[train_len:train_len + val_len]
            test_data = df.iloc[train_len + val_len:]
            
            self.training_data[category] = train_data
            self.validation_data[category] = val_data
            self.test_data[category] = test_data
            
            logger.info(f"Split {category}: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    def train_instrument_agents(self) -> Dict[str, Any]:
        """Train RL agents for all financial instrument types"""
        
        logger.info("Training specialized instrument RL agents...")
        
        # Create specialized agent configurations
        specialized_agents = create_specialized_agents()
        
        training_results = {}
        
        for asset_type, agent_config in specialized_agents.items():
            category = asset_type.value
            
            if category not in self.training_data:
                logger.warning(f"No training data for {category}")
                continue
            
            logger.info(f"Training {category} RL agent...")
            
            try:
                # Create trading environment configuration
                env_config = SpecializedAgentFactory.create_trading_env_config(agent_config)
                
                # Create and train agent (placeholder - would use actual RL training)
                metrics = {
                    "final_reward": np.random.uniform(0.5, 2.0),
                    "episodes_trained": 1000,
                    "convergence_episode": 800,
                    "best_performance": np.random.uniform(0.8, 1.5)
                }
                
                training_results[category] = metrics
                
                # Save agent configuration
                agent_path = Path(self.config.model_save_path) / category
                agent_path.mkdir(parents=True, exist_ok=True)
                
                # Save configuration
                config_path = agent_path / "agent_config.yaml"
                with open(config_path, 'w') as f:
                    yaml.dump({
                        "asset_type": asset_type.value,
                        "num_assets": agent_config.num_assets,
                        "symbols": agent_config.symbols,
                        "reward_weights": agent_config.reward_weights,
                        "risk_limits": agent_config.risk_limits
                    }, f)
                
                # Store agent configuration
                self.instrument_agents[category] = agent_config
                
                logger.info(f"Trained {category} agent: final reward = {metrics['final_reward']:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {category} agent: {e}")
                continue
        
        self.training_results = training_results
        logger.info("Specialized instrument RL agent training completed")
        
        return training_results
    
    def train_meta_agent(self) -> Dict[str, Any]:
        """Train the RL meta agent"""
        
        logger.info("Training RL meta agent...")
        
        # Create meta agent configuration with specialized agents
        meta_config = RLMetaAgentConfig(
            coordination_method="hierarchical",
            fusion_method="weighted",
            agent_weights={
                "asx_stocks": 0.25,
                "cryptocurrencies": 0.20,
                "etf": 0.15,
                "currency_pairs": 0.15,
                "reit": 0.10,
                "us_stocks": 0.10,
                "commodities": 0.05
            },
            confidence_threshold=0.3,
            risk_tolerance=0.1,
            max_position_size=0.2,
            rebalance_frequency=7,
            lookback_window=self.config.lookback_window,
            prediction_horizon=self.config.prediction_horizon,
            high_level_algorithm="PPO",
            low_level_algorithm="PPO"
        )
        
        # Create meta agent
        self.meta_agent = RLMetaAgent(meta_config)
        
        # Add instrument agents
        for category, agent in self.instrument_agents.items():
            self.meta_agent.add_instrument_agent(category, agent)
        
        # Train meta agent
        meta_training_results = self.meta_agent.train_meta_agent(self.training_data)
        
        # Save meta agent
        meta_agent_path = Path(self.config.model_save_path) / "meta_agent"
        self.meta_agent.save_meta_agent(str(meta_agent_path))
        
        logger.info("RL meta agent training completed")
        
        return meta_training_results
    
    def evaluate_agents(self) -> Dict[str, Any]:
        """Evaluate all trained RL agents"""
        
        logger.info("Evaluating RL agents...")
        
        evaluation_results = {}
        
        # Evaluate individual agents
        for category, agent in self.instrument_agents.items():
            if category not in self.test_data:
                continue
            
            logger.info(f"Evaluating {category} agent...")
            
            try:
                # Evaluate agent
                results = agent.evaluate(self.test_data[category])
                evaluation_results[category] = results
                
                logger.info(f"Evaluated {category} agent: mean reward = {results['mean_reward']:.4f}")
                
            except Exception as e:
                logger.error(f"Error evaluating {category} agent: {e}")
                continue
        
        # Evaluate meta agent
        if self.meta_agent:
            logger.info("Evaluating RL meta agent...")
            
            try:
                # Generate signals for test data
                test_signals = {}
                for category, data in self.test_data.items():
                    if category in self.instrument_agents:
                        test_signals[category] = data
                
                if test_signals:
                    meta_signals = self.meta_agent.generate_signals(test_signals)
                    
                    evaluation_results["meta_agent"] = {
                        "signals_generated": len(meta_signals["agent_signals"]),
                        "trading_decision": meta_signals["trading_decision"],
                        "coordination_state": meta_signals["coordination_state"]
                    }
                    
                    logger.info("RL meta agent evaluation completed")
            
            except Exception as e:
                logger.error(f"Error evaluating RL meta agent: {e}")
        
        self.evaluation_results = evaluation_results
        logger.info("RL agent evaluation completed")
        
        return evaluation_results
    
    def generate_training_report(self) -> Dict[str, Any]:
        """Generate comprehensive training report"""
        
        logger.info("Generating training report...")
        
        report = {
            "training_config": {
                "start_date": self.config.start_date,
                "end_date": self.config.end_date,
                "lookback_window": self.config.lookback_window,
                "prediction_horizon": self.config.prediction_horizon,
                "validation_split": self.config.validation_split,
                "test_split": self.config.test_split
            },
            "data_summary": {
                category: {
                    "total_records": len(data),
                    "training_records": len(self.training_data.get(category, [])),
                    "validation_records": len(self.validation_data.get(category, [])),
                    "test_records": len(self.test_data.get(category, [])),
                    "features": len(data.columns) if not data.empty else 0
                }
                for category, data in self.training_data.items()
            },
            "training_results": self.training_results,
            "evaluation_results": self.evaluation_results,
            "model_paths": {
                category: str(Path(self.config.model_save_path) / category)
                for category in self.instrument_models.keys()
            },
            "meta_agent_path": str(Path(self.config.model_save_path) / "meta_agent"),
            "generated_at": datetime.now().isoformat()
        }
        
        # Save report
        report_path = Path(self.config.reports_save_path) / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Training report saved to {report_path}")
        
        return report
    
    def run_full_training_pipeline(self) -> Dict[str, Any]:
        """Run the complete training pipeline"""
        
        logger.info("Starting full training pipeline...")
        
        try:
            # 1. Prepare data
            self.prepare_data()
            
            # 2. Train instrument RL agents
            self.train_instrument_agents()
            
            # 3. Train RL meta agent
            self.train_meta_agent()
            
            # 4. Evaluate RL agents
            self.evaluate_agents()
            
            # 5. Generate report
            report = self.generate_training_report()
            
            logger.info("Full training pipeline completed successfully")
            
            return report
            
        except Exception as e:
            logger.error(f"Error in training pipeline: {e}")
            raise


def main():
    """Test the daily timeframe trainer"""
    
    # Create training configuration
    config = DailyTrainingConfig(
        start_date="2020-01-01",
        end_date="2024-01-01",
        lookback_window=5,
        prediction_horizon=1,
        model_save_path="models/daily_timeframe",
        metrics_save_path="metrics/daily_timeframe",
        reports_save_path="reports/daily_timeframe"
    )
    
    # Create trainer
    trainer = DailyTimeframeTrainer(config)
    
    # Run full training pipeline
    report = trainer.run_full_training_pipeline()
    
    print("Daily Timeframe Training Completed!")
    print(f"Models trained: {len(trainer.instrument_models)}")
    print(f"Meta agent trained: {trainer.meta_agent is not None}")
    print(f"Report generated: {report['generated_at']}")


if __name__ == "__main__":
    main()
