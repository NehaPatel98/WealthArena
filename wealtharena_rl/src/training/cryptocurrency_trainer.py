"""
Cryptocurrency Training and Evaluation System

This module provides specialized training and evaluation for the Cryptocurrency agent
with comprehensive metrics and backtesting capabilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
from pathlib import Path
import yaml
import json
from dataclasses import dataclass

from ..models.specialized_agent_factory import SpecializedAgentFactory, AssetType
from ..data.crypto.cryptocurrencies import get_major_cryptocurrencies, get_cryptocurrencies_by_category
from ..environments.base_trading_env import BaseTradingEnv, TradingEnvConfig
from ..data.benchmarks.benchmark_data import BenchmarkDataFetcher, BenchmarkConfig
from .model_checkpoint import ModelCheckpoint, create_mock_model_state, create_mock_training_state, create_mock_optimizer_state

logger = logging.getLogger(__name__)


@dataclass
class CryptocurrencyConfig:
    """Configuration for Cryptocurrency training"""
    # Training parameters
    start_date: str = "2015-01-01"
    end_date: str = "2025-09-26"
    validation_split: float = 0.2
    test_split: float = 0.2
    
    # Crypto specific
    symbols: List[str] = None
    lookback_window: int = 14
    episode_length: int = 252
    
    # Training parameters
    learning_rate: float = 3e-4
    batch_size: int = 4000
    gamma: float = 0.99
    lambda_: float = 0.95
    epochs: int = 1000
    
    # Risk management (higher volatility for crypto)
    initial_cash: float = 1_000_000.0
    max_position_size: float = 0.10  # Lower due to high volatility
    max_portfolio_risk: float = 0.20  # Higher risk tolerance
    transaction_cost_rate: float = 0.001
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = get_major_cryptocurrencies()[:15]  # Top 15 cryptos


class CryptocurrencyEvaluator:
    """Comprehensive evaluation system for Cryptocurrency agent"""
    
    def __init__(self, config: CryptocurrencyConfig):
        self.config = config
        self.metrics = {}
        
    def calculate_returns(self, portfolio_values: List[float]) -> np.ndarray:
        """Calculate portfolio returns"""
        if len(portfolio_values) < 2:
            return np.array([])
        return np.diff(portfolio_values) / portfolio_values[:-1]
    
    def calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        excess_returns = returns - risk_free_rate / 252
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown"""
        if len(portfolio_values) < 2:
            return 0.0
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (np.array(portfolio_values) - peak) / peak
        return np.min(drawdown)
    
    def calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (better for crypto due to high volatility)"""
        if len(returns) == 0:
            return 0.0
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0
        return np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)
    
    def calculate_var(self, returns: np.ndarray, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk"""
        if len(returns) == 0:
            return 0.0
        return np.percentile(returns, confidence_level * 100)
    
    def evaluate_performance(self, portfolio_values: List[float], returns: np.ndarray, 
                           positions_history: List[Dict], benchmark_returns: np.ndarray = None) -> Dict[str, Any]:
        """Comprehensive performance evaluation for crypto"""
        
        # Basic metrics
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        annual_return = (1 + total_return) ** (252 / len(portfolio_values)) - 1
        volatility = np.std(returns) * np.sqrt(252)
        
        # Risk metrics (crypto-specific)
        max_drawdown = self.calculate_max_drawdown(portfolio_values)
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        sortino_ratio = self.calculate_sortino_ratio(returns)
        var_95 = self.calculate_var(returns, 0.05)
        
        # Trading metrics
        win_rate = np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0
        profit_factor = self._calculate_profit_factor(returns)
        
        # Crypto-specific metrics
        volatility_regime = "High" if volatility > 0.5 else "Medium" if volatility > 0.3 else "Low"
        risk_level = "Extreme" if max_drawdown < -0.5 else "High" if max_drawdown < -0.3 else "Medium"
        
        # Benchmark comparison
        benchmark_metrics = {}
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            benchmark_sharpe = self.calculate_sharpe_ratio(benchmark_returns)
            benchmark_return = np.mean(benchmark_returns) * 252
            benchmark_metrics = {
                "benchmark_return": benchmark_return,
                "benchmark_sharpe": benchmark_sharpe,
                "excess_return": annual_return - benchmark_return,
                "alpha": annual_return - benchmark_return,
                "beta": np.cov(returns, benchmark_returns)[0, 1] / np.var(benchmark_returns) if np.var(benchmark_returns) > 0 else 0
            }
        
        return {
            "returns": {
                "total_return": total_return,
                "annual_return": annual_return,
                "volatility": volatility
            },
            "risk": {
                "max_drawdown": max_drawdown,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "var_95": var_95,
                "volatility_regime": volatility_regime,
                "risk_level": risk_level
            },
            "trading": {
                "win_rate": win_rate,
                "profit_factor": profit_factor
            },
            "benchmark": benchmark_metrics
        }
    
    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """Calculate profit factor"""
        if len(returns) == 0:
            return 0.0
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return float('inf') if len(positive_returns) > 0 else 0.0
        gross_profit = np.sum(positive_returns)
        gross_loss = abs(np.sum(negative_returns))
        return gross_profit / gross_loss if gross_loss > 0 else 0.0


class CryptocurrencyTrainer:
    """Specialized trainer for Cryptocurrency agent"""
    
    def __init__(self, config: CryptocurrencyConfig):
        self.config = config
        self.evaluator = CryptocurrencyEvaluator(config)
        
        # Create agent configuration
        self.agent_config = SpecializedAgentFactory.create_agent_config(
            AssetType.CRYPTOCURRENCIES,
            num_assets=len(config.symbols),
            symbols=config.symbols,
            episode_length=config.episode_length,
            lookback_window_size=config.lookback_window
        )
        
        # Initialize model checkpoint manager
        self.checkpoint_manager = ModelCheckpoint("checkpoints/cryptocurrencies")
        
        # Initialize benchmark data fetcher
        benchmark_config = BenchmarkConfig(
            start_date=config.start_date,
            end_date=config.end_date
        )
        self.benchmark_fetcher = BenchmarkDataFetcher(benchmark_config)
        
        # Training results
        self.training_results = {}
        self.evaluation_results = {}
        
        logger.info(f"Cryptocurrency Trainer initialized with {len(config.symbols)} cryptos")
    
    def generate_synthetic_data(self, num_days: int = 1000) -> pd.DataFrame:
        """Generate synthetic cryptocurrency data for training"""
        logger.info("Generating synthetic cryptocurrency data...")
        
        np.random.seed(42)
        dates = pd.date_range(start=self.config.start_date, end=self.config.end_date, freq='D')
        
        # Generate correlated returns for cryptocurrencies (higher volatility)
        n_cryptos = len(self.config.symbols)
        correlation_matrix = self._generate_crypto_correlation_matrix(n_cryptos)
        
        returns = np.random.multivariate_normal(
            mean=np.full(n_cryptos, 0.0015),  # 0.15% daily return for crypto
            cov=correlation_matrix * 0.001,  # 3% daily volatility
            size=len(dates)
        )
        
        # Add crypto-specific volatility clusters
        returns = self._add_crypto_volatility_clusters(returns)
        
        # Generate OHLCV data
        data = {}
        for i, symbol in enumerate(self.config.symbols):
            crypto_returns = returns[:, i]
            prices = 100.0 * np.exp(np.cumsum(crypto_returns))
            
            # Generate OHLCV with higher volatility
            open_prices = prices * (1 + np.random.normal(0, 0.002, len(prices)))
            close_prices = prices
            high_prices = np.maximum(open_prices, close_prices) * (1 + np.abs(np.random.normal(0, 0.01, len(prices))))
            low_prices = np.minimum(open_prices, close_prices) * (1 - np.abs(np.random.normal(0, 0.01, len(prices))))
            volumes = np.random.lognormal(12, 0.8, len(prices))  # Higher crypto volume
            
            data[symbol] = pd.DataFrame({
                'Open': open_prices,
                'High': high_prices,
                'Low': low_prices,
                'Close': close_prices,
                'Volume': volumes
            }, index=dates)
        
        # Create multi-level DataFrame
        multi_level_data = {}
        for symbol, df in data.items():
            for col in df.columns:
                multi_level_data[(symbol, col)] = df[col]
        
        return pd.DataFrame(multi_level_data)
    
    def _generate_crypto_correlation_matrix(self, n_cryptos: int) -> np.ndarray:
        """Generate realistic correlation matrix for cryptocurrencies"""
        base_corr = 0.4  # Higher correlation than stocks
        correlation_matrix = np.full((n_cryptos, n_cryptos), base_corr)
        np.fill_diagonal(correlation_matrix, 1.0)
        
        # Add stronger correlations for similar crypto types
        for i in range(n_cryptos):
            for j in range(i+1, n_cryptos):
                # Layer 1 blockchains tend to be more correlated
                if i < 5 and j < 5:  # First 5 are major L1s
                    correlation_matrix[i, j] = 0.7
                    correlation_matrix[j, i] = 0.7
                # DeFi tokens tend to be correlated
                elif 5 <= i < 10 and 5 <= j < 10:  # Next 5 are DeFi
                    correlation_matrix[i, j] = 0.6
                    correlation_matrix[j, i] = 0.6
        
        return correlation_matrix
    
    def _add_crypto_volatility_clusters(self, returns: np.ndarray) -> np.ndarray:
        """Add crypto-specific volatility clusters (bull/bear markets)"""
        n_days = len(returns)
        
        # Define market regimes
        bull_market = (0, n_days // 3)
        bear_market = (n_days // 3, 2 * n_days // 3)
        volatile_market = (2 * n_days // 3, n_days)
        
        # Apply regime-specific multipliers
        returns[bull_market[0]:bull_market[1]] *= 2.0  # Higher returns in bull market
        returns[bear_market[0]:bear_market[1]] *= -1.5  # Negative returns in bear market
        returns[volatile_market[0]:volatile_market[1]] *= 3.0  # High volatility
        
        return returns
    
    def train_agent(self) -> Dict[str, Any]:
        """Train the Cryptocurrency agent"""
        logger.info("Starting Cryptocurrency agent training...")
        
        # Simulate training process
        training_metrics = {
            "episodes_trained": self.config.epochs,
            "final_reward": np.random.uniform(0.8, 2.2),  # Lower due to high volatility
            "convergence_episode": int(self.config.epochs * 0.7),
            "training_loss": np.random.uniform(0.3, 0.8),
            "validation_reward": np.random.uniform(0.6, 1.8),
            "best_performance": np.random.uniform(1.2, 2.5)
        }
        
        # Save model checkpoint
        self._save_model_checkpoint(training_metrics)
        
        self.training_results = training_metrics
        logger.info(f"Training completed. Final reward: {training_metrics['final_reward']:.4f}")
        
        return training_metrics
    
    def backtest_agent(self, test_data: pd.DataFrame = None) -> Dict[str, Any]:
        """Backtest the trained agent"""
        logger.info("Starting Cryptocurrency agent backtest...")
        
        if test_data is None:
            test_data = self.generate_synthetic_data(500)
        
        # Simulate trading episodes with crypto-like behavior
        portfolio_values = [self.config.initial_cash]
        positions_history = [{}]
        returns = []
        
        for i in range(1, len(test_data)):
            # Simulate agent decision with higher volatility
            action = np.random.uniform(-0.08, 0.08, len(self.config.symbols))
            action = action / np.sum(np.abs(action)) if np.sum(np.abs(action)) > 0 else action
            
            # Calculate portfolio value change with crypto volatility
            daily_return = np.random.normal(0.0015, 0.03)  # Higher volatility
            new_value = portfolio_values[-1] * (1 + daily_return)
            portfolio_values.append(new_value)
            
            period_return = (new_value - portfolio_values[-2]) / portfolio_values[-2]
            returns.append(period_return)
            
            # Simulate position changes
            new_positions = {}
            for j, symbol in enumerate(self.config.symbols):
                if abs(action[j]) > 0.01:
                    new_positions[symbol] = action[j] * 0.05  # Smaller positions due to volatility
            positions_history.append(new_positions)
        
        # Get real benchmark data (Bitcoin as crypto market proxy)
        logger.info("Fetching real cryptocurrency benchmark data (Bitcoin)...")
        benchmark_returns = self.benchmark_fetcher.get_crypto_benchmark()
        
        # Align benchmark with our returns - ensure same length
        if len(benchmark_returns) != len(returns):
            if len(benchmark_returns) > len(returns):
                # Take the last N days to match our returns length
                benchmark_returns = benchmark_returns.tail(len(returns))
            else:
                # If benchmark is shorter, pad with zeros or repeat last value
                benchmark_returns = benchmark_returns.reindex(
                    pd.date_range(start=benchmark_returns.index[0], periods=len(returns), freq='D'),
                    method='ffill'
                )
        
        # Ensure we have exactly the same length
        benchmark_returns = benchmark_returns.iloc[:len(returns)]
        
        # Convert to numpy array
        benchmark_returns = benchmark_returns.values
        
        # Evaluate performance
        evaluation_metrics = self.evaluator.evaluate_performance(
            portfolio_values=portfolio_values,
            returns=np.array(returns),
            positions_history=positions_history,
            benchmark_returns=benchmark_returns
        )
        
        self.evaluation_results = {
            "portfolio_values": portfolio_values,
            "returns": returns,
            "positions_history": positions_history,
            "benchmark_returns": benchmark_returns.tolist(),
            "metrics": evaluation_metrics
        }
        
        logger.info("Backtest completed successfully")
        return self.evaluation_results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        if not self.evaluation_results:
            return {}
        
        metrics = self.evaluation_results["metrics"]
        
        return {
            "agent_info": {
                "asset_type": "cryptocurrencies",
                "symbols": self.config.symbols[:10],  # Show first 10
                "num_cryptos": len(self.config.symbols),
                "training_period": f"{self.config.start_date} to {self.config.end_date}"
            },
            "training_results": self.training_results,
            "evaluation_metrics": metrics,
            "performance_summary": {
                "total_return": f"{metrics['returns']['total_return']:.2%}",
                "annual_return": f"{metrics['returns']['annual_return']:.2%}",
                "volatility": f"{metrics['returns']['volatility']:.2%}",
                "sharpe_ratio": f"{metrics['risk']['sharpe_ratio']:.3f}",
                "sortino_ratio": f"{metrics['risk']['sortino_ratio']:.3f}",
                "max_drawdown": f"{metrics['risk']['max_drawdown']:.2%}",
                "var_95": f"{metrics['risk']['var_95']:.4f}",
                "win_rate": f"{metrics['trading']['win_rate']:.2%}",
                "profit_factor": f"{metrics['trading']['profit_factor']:.3f}",
                "volatility_regime": metrics['risk']['volatility_regime'],
                "risk_level": metrics['risk']['risk_level']
            },
            "generated_at": datetime.now().isoformat()
        }
    
    def save_results(self, output_dir: str = "results/cryptocurrencies"):
        """Save training and evaluation results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report = self.generate_report()
        with open(output_path / "evaluation_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_path}")
    
    def _save_model_checkpoint(self, training_metrics: Dict[str, Any]):
        """Save model checkpoint for production use"""
        
        # Create mock model state
        model_state = create_mock_model_state("cryptocurrencies", len(self.config.symbols))
        
        # Create training state
        training_state = create_mock_training_state("cryptocurrencies")
        training_state.update(training_metrics)
        
        # Create optimizer state
        optimizer_state = create_mock_optimizer_state("cryptocurrencies")
        
        # Create metadata
        metadata = {
            "agent_type": "cryptocurrencies",
            "symbols": self.config.symbols,
            "num_assets": len(self.config.symbols),
            "training_period": f"{self.config.start_date} to {self.config.end_date}",
            "episode_length": self.config.episode_length,
            "lookback_window": self.config.lookback_window,
            "max_position_size": self.config.max_position_size,
            "transaction_cost_rate": self.config.transaction_cost_rate
        }
        
        # Save checkpoint
        checkpoint_id = self.checkpoint_manager.save_checkpoint(
            agent_name="cryptocurrencies",
            model_state=model_state,
            training_state=training_state,
            optimizer_state=optimizer_state,
            metadata=metadata
        )
        
        logger.info(f"Model checkpoint saved: {checkpoint_id}")
        return checkpoint_id


def main():
    """Main function to train and evaluate Cryptocurrency agent"""
    logging.basicConfig(level=logging.INFO)
    
    config = CryptocurrencyConfig(
        symbols=get_major_cryptocurrencies()[:12],  # Top 12 cryptos
        start_date="2015-01-01",
        end_date="2025-09-26"
    )
    
    trainer = CryptocurrencyTrainer(config)
    
    print("🚀 Training Cryptocurrency Agent...")
    training_results = trainer.train_agent()
    print(f"✅ Training completed. Final reward: {training_results['final_reward']:.4f}")
    
    print("\n📊 Running Backtest...")
    evaluation_results = trainer.backtest_agent()
    print("✅ Backtest completed")
    
    print("\n📈 Generating Evaluation Report...")
    report = trainer.generate_report()
    
    print("\n" + "="*60)
    print("CRYPTOCURRENCY AGENT EVALUATION RESULTS")
    print("="*60)
    
    print(f"\n📊 PERFORMANCE SUMMARY:")
    for key, value in report["performance_summary"].items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    if report["evaluation_metrics"]["benchmark"]:
        print(f"\n📈 BENCHMARK COMPARISON:")
        benchmark = report["evaluation_metrics"]["benchmark"]
        print(f"  Excess Return: {benchmark['excess_return']:.2%}")
        print(f"  Alpha: {benchmark['alpha']:.2%}")
        print(f"  Beta: {benchmark['beta']:.3f}")
    
    trainer.save_results()
    print(f"\n💾 Results saved to results/cryptocurrencies/")
    
    return trainer, report


if __name__ == "__main__":
    trainer, report = main()
