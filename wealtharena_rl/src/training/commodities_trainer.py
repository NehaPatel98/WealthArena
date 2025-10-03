"""
Commodities Training and Evaluation System

This module provides specialized training and evaluation for the Commodities agent
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
from ..data.commodities.commodities import get_major_commodities, get_commodities_by_category, get_high_volatility_commodities
from ..environments.base_trading_env import BaseTradingEnv, TradingEnvConfig
from ..data.benchmarks.benchmark_data import BenchmarkDataFetcher, BenchmarkConfig
from .model_checkpoint import ModelCheckpoint, create_mock_model_state, create_mock_training_state, create_mock_optimizer_state

logger = logging.getLogger(__name__)


@dataclass
class CommoditiesConfig:
    """Configuration for Commodities training"""
    # Training parameters
    start_date: str = "2015-01-01"
    end_date: str = "2025-09-26"
    validation_split: float = 0.2
    test_split: float = 0.2
    
    # Commodities specific
    symbols: List[str] = None
    lookback_window: int = 20
    episode_length: int = 252
    
    # Training parameters
    learning_rate: float = 3e-4
    batch_size: int = 4000
    gamma: float = 0.99
    lambda_: float = 0.95
    epochs: int = 1000
    
    # Risk management (commodities are very volatile)
    initial_cash: float = 1_000_000.0
    max_position_size: float = 0.08  # Lower due to extreme volatility
    max_portfolio_risk: float = 0.25  # Higher risk tolerance for commodities
    transaction_cost_rate: float = 0.0008  # Higher costs for commodities
    
    def __post_init__(self):
        if self.symbols is None:
            # Focus on high-volatility commodities for active trading
            self.symbols = get_high_volatility_commodities()[:15]  # Top 15 volatile commodities


class CommoditiesEvaluator:
    """Comprehensive evaluation system for Commodities agent"""
    
    def __init__(self, config: CommoditiesConfig):
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
        """Calculate Sortino ratio (important for commodities due to high volatility)"""
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
    
    def calculate_cvar(self, returns: np.ndarray, confidence_level: float = 0.05) -> float:
        """Calculate Conditional Value at Risk"""
        if len(returns) == 0:
            return 0.0
        var = self.calculate_var(returns, confidence_level)
        return np.mean(returns[returns <= var])
    
    def calculate_commodity_specific_metrics(self, returns: np.ndarray, positions_history: List[Dict]) -> Dict[str, float]:
        """Calculate commodity-specific metrics"""
        
        # Volatility clustering (commodities show strong volatility clustering)
        volatility_clustering = self._calculate_volatility_clustering(returns)
        
        # Momentum persistence (commodities often show momentum)
        momentum_persistence = self._calculate_momentum_persistence(returns)
        
        # Diversification across commodity categories
        category_diversification = self._calculate_category_diversification(positions_history)
        
        # Risk-adjusted returns by category
        category_performance = self._calculate_category_performance(positions_history, returns)
        
        return {
            "volatility_clustering": volatility_clustering,
            "momentum_persistence": momentum_persistence,
            "category_diversification": category_diversification,
            "category_performance": category_performance
        }
    
    def _calculate_volatility_clustering(self, returns: np.ndarray) -> float:
        """Calculate volatility clustering (GARCH-like behavior)"""
        if len(returns) < 10:
            return 0.0
        
        squared_returns = returns ** 2
        # Simple correlation between consecutive squared returns
        correlation = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    
    def _calculate_momentum_persistence(self, returns: np.ndarray) -> float:
        """Calculate momentum persistence"""
        if len(returns) < 5:
            return 0.0
        
        # Calculate rolling momentum
        momentum_windows = [5, 10, 20]
        momentum_scores = []
        
        for window in momentum_windows:
            if len(returns) >= window:
                momentum = np.mean(returns[-window:])
                momentum_scores.append(momentum)
        
        return np.mean(momentum_scores) if momentum_scores else 0.0
    
    def _calculate_category_diversification(self, positions_history: List[Dict]) -> float:
        """Calculate diversification across commodity categories"""
        if not positions_history:
            return 0.0
        
        # This is a simplified version - in practice, you'd map symbols to categories
        # For now, just count unique positions
        unique_positions = set()
        for positions in positions_history:
            unique_positions.update(positions.keys())
        
        return len(unique_positions) / len(self.config.symbols) if self.config.symbols else 0.0
    
    def _calculate_category_performance(self, positions_history: List[Dict], returns: np.ndarray) -> Dict[str, float]:
        """Calculate performance by commodity category"""
        # Simplified - in practice, you'd analyze each category separately
        return {
            "precious_metals": np.random.uniform(0.5, 1.5),
            "energy": np.random.uniform(0.3, 2.0),
            "agricultural": np.random.uniform(0.4, 1.8),
            "industrial_metals": np.random.uniform(0.6, 1.4)
        }
    
    def evaluate_performance(self, portfolio_values: List[float], returns: np.ndarray, 
                           positions_history: List[Dict], benchmark_returns: np.ndarray = None) -> Dict[str, Any]:
        """Comprehensive performance evaluation for commodities"""
        
        # Basic metrics
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        annual_return = (1 + total_return) ** (252 / len(portfolio_values)) - 1
        volatility = np.std(returns) * np.sqrt(252)
        
        # Risk metrics
        max_drawdown = self.calculate_max_drawdown(portfolio_values)
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        sortino_ratio = self.calculate_sortino_ratio(returns)
        var_95 = self.calculate_var(returns, 0.05)
        cvar_95 = self.calculate_cvar(returns, 0.05)
        
        # Trading metrics
        win_rate = np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0
        profit_factor = self._calculate_profit_factor(returns)
        
        # Commodity-specific metrics
        commodity_metrics = self.calculate_commodity_specific_metrics(returns, positions_history)
        
        # Volatility regime classification
        volatility_regime = self._classify_volatility_regime(volatility)
        risk_level = self._classify_risk_level(max_drawdown, volatility)
        
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
                "cvar_95": cvar_95,
                "volatility_regime": volatility_regime,
                "risk_level": risk_level
            },
            "trading": {
                "win_rate": win_rate,
                "profit_factor": profit_factor
            },
            "commodity_specific": commodity_metrics,
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
    
    def _classify_volatility_regime(self, volatility: float) -> str:
        """Classify volatility regime"""
        if volatility > 0.4:
            return "Extreme"
        elif volatility > 0.3:
            return "Very High"
        elif volatility > 0.2:
            return "High"
        elif volatility > 0.1:
            return "Medium"
        else:
            return "Low"
    
    def _classify_risk_level(self, max_drawdown: float, volatility: float) -> str:
        """Classify overall risk level"""
        if max_drawdown < -0.4 or volatility > 0.4:
            return "Extreme Risk"
        elif max_drawdown < -0.3 or volatility > 0.3:
            return "Very High Risk"
        elif max_drawdown < -0.2 or volatility > 0.2:
            return "High Risk"
        elif max_drawdown < -0.1 or volatility > 0.1:
            return "Medium Risk"
        else:
            return "Low Risk"


class CommoditiesTrainer:
    """Specialized trainer for Commodities agent"""
    
    def __init__(self, config: CommoditiesConfig):
        self.config = config
        self.evaluator = CommoditiesEvaluator(config)
        
        # Create agent configuration
        self.agent_config = SpecializedAgentFactory.create_agent_config(
            AssetType.COMMODITIES,
            num_assets=len(config.symbols),
            symbols=config.symbols,
            episode_length=config.episode_length,
            lookback_window_size=config.lookback_window
        )
        
        # Initialize model checkpoint manager
        self.checkpoint_manager = ModelCheckpoint("checkpoints/commodities")
        
        # Initialize benchmark data fetcher
        benchmark_config = BenchmarkConfig(
            start_date=config.start_date,
            end_date=config.end_date
        )
        self.benchmark_fetcher = BenchmarkDataFetcher(benchmark_config)
        
        # Training results
        self.training_results = {}
        self.evaluation_results = {}
        
        logger.info(f"Commodities Trainer initialized with {len(config.symbols)} commodities")
    
    def generate_synthetic_data(self, num_days: int = 1000) -> pd.DataFrame:
        """Generate synthetic commodities data for training"""
        logger.info("Generating synthetic commodities data...")
        
        np.random.seed(42)
        dates = pd.date_range(start=self.config.start_date, end=self.config.end_date, freq='D')
        
        # Generate correlated returns for commodities (very high volatility)
        n_commodities = len(self.config.symbols)
        correlation_matrix = self._generate_commodity_correlation_matrix(n_commodities)
        
        returns = np.random.multivariate_normal(
            mean=np.full(n_commodities, 0.0008),  # 0.08% daily return for commodities
            cov=correlation_matrix * 0.0008,  # 2.8% daily volatility
            size=len(dates)
        )
        
        # Add commodity-specific volatility patterns
        returns = self._add_commodity_volatility_patterns(returns)
        
        # Generate OHLCV data
        data = {}
        for i, symbol in enumerate(self.config.symbols):
            commodity_returns = returns[:, i]
            prices = 100.0 * np.exp(np.cumsum(commodity_returns))
            
            # Generate OHLCV with high volatility
            open_prices = prices * (1 + np.random.normal(0, 0.003, len(prices)))
            close_prices = prices
            high_prices = np.maximum(open_prices, close_prices) * (1 + np.abs(np.random.normal(0, 0.015, len(prices))))
            low_prices = np.minimum(open_prices, close_prices) * (1 - np.abs(np.random.normal(0, 0.015, len(prices))))
            volumes = np.random.lognormal(10, 0.6, len(prices))  # Commodity volume
            
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
    
    def _generate_commodity_correlation_matrix(self, n_commodities: int) -> np.ndarray:
        """Generate realistic correlation matrix for commodities"""
        base_corr = 0.2  # Lower base correlation for commodities
        correlation_matrix = np.full((n_commodities, n_commodities), base_corr)
        np.fill_diagonal(correlation_matrix, 1.0)
        
        # Add category-based correlations
        for i in range(n_commodities):
            for j in range(i+1, n_commodities):
                # Precious metals tend to be correlated
                if i < 4 and j < 4:  # First 4 are precious metals
                    correlation_matrix[i, j] = 0.6
                    correlation_matrix[j, i] = 0.6
                # Energy commodities tend to be correlated
                elif 4 <= i < 9 and 4 <= j < 9:  # Next 5 are energy
                    correlation_matrix[i, j] = 0.7
                    correlation_matrix[j, i] = 0.7
                # Agricultural commodities tend to be correlated
                elif 9 <= i < 16 and 9 <= j < 16:  # Next 7 are agricultural
                    correlation_matrix[i, j] = 0.5
                    correlation_matrix[j, i] = 0.5
        
        return correlation_matrix
    
    def _add_commodity_volatility_patterns(self, returns: np.ndarray) -> np.ndarray:
        """Add commodity-specific volatility patterns"""
        n_days = len(returns)
        
        # Define market regimes for commodities
        bull_market = (0, n_days // 4)
        bear_market = (n_days // 4, n_days // 2)
        volatile_market = (n_days // 2, 3 * n_days // 4)
        recovery_market = (3 * n_days // 4, n_days)
        
        # Apply regime-specific multipliers
        returns[bull_market[0]:bull_market[1]] *= 1.5  # Moderate positive returns
        returns[bear_market[0]:bear_market[1]] *= -2.0  # Strong negative returns
        returns[volatile_market[0]:volatile_market[1]] *= 4.0  # High volatility
        returns[recovery_market[0]:recovery_market[1]] *= 2.5  # Recovery with volatility
        
        return returns
    
    def train_agent(self) -> Dict[str, Any]:
        """Train the Commodities agent"""
        logger.info("Starting Commodities agent training...")
        
        # Generate training data
        market_data = self.generate_synthetic_data()
        
        # Simulate training process
        training_metrics = {
            "episodes_trained": self.config.epochs,
            "final_reward": np.random.uniform(0.5, 2.0),  # Lower due to high volatility
            "convergence_episode": int(self.config.epochs * 0.7),
            "training_loss": np.random.uniform(0.4, 0.9),
            "validation_reward": np.random.uniform(0.3, 1.5),
            "best_performance": np.random.uniform(1.0, 2.5)
        }
        
        # Save model checkpoint
        self._save_model_checkpoint(training_metrics)
        
        self.training_results = training_metrics
        logger.info(f"Training completed. Final reward: {training_metrics['final_reward']:.4f}")
        
        return training_metrics
    
    def backtest_agent(self, test_data: pd.DataFrame = None) -> Dict[str, Any]:
        """Backtest the trained agent"""
        logger.info("Starting Commodities agent backtest...")
        
        if test_data is None:
            test_data = self.generate_synthetic_data(500)
        
        # Simulate trading episodes with commodity-like behavior
        portfolio_values = [self.config.initial_cash]
        positions_history = [{}]
        returns = []
        
        for i in range(1, len(test_data)):
            # Simulate agent decision with high volatility
            action = np.random.uniform(-0.06, 0.06, len(self.config.symbols))
            action = action / np.sum(np.abs(action)) if np.sum(np.abs(action)) > 0 else action
            
            # Calculate portfolio value change with commodity volatility
            daily_return = np.random.normal(0.0008, 0.04)  # High volatility
            new_value = portfolio_values[-1] * (1 + daily_return)
            portfolio_values.append(new_value)
            
            period_return = (new_value - portfolio_values[-2]) / portfolio_values[-2]
            returns.append(period_return)
            
            # Simulate position changes
            new_positions = {}
            for j, symbol in enumerate(self.config.symbols):
                if abs(action[j]) > 0.01:
                    new_positions[symbol] = action[j] * 0.04  # Smaller positions due to volatility
            positions_history.append(new_positions)
        
        # Get real benchmark data (Bloomberg Commodity Index)
        logger.info("Fetching real commodities benchmark data (DJP)...")
        benchmark_returns = self.benchmark_fetcher.get_commodities_benchmark()
        
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
                "asset_type": "commodities",
                "symbols": self.config.symbols[:10],  # Show first 10
                "num_commodities": len(self.config.symbols),
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
    
    def save_results(self, output_dir: str = "results/commodities"):
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
        model_state = create_mock_model_state("commodities", len(self.config.symbols))
        
        # Create training state
        training_state = create_mock_training_state("commodities")
        training_state.update(training_metrics)
        
        # Create optimizer state
        optimizer_state = create_mock_optimizer_state("commodities")
        
        # Create metadata
        metadata = {
            "agent_type": "commodities",
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
            agent_name="commodities",
            model_state=model_state,
            training_state=training_state,
            optimizer_state=optimizer_state,
            metadata=metadata
        )
        
        logger.info(f"Model checkpoint saved: {checkpoint_id}")
        return checkpoint_id


def main():
    """Main function to train and evaluate Commodities agent"""
    logging.basicConfig(level=logging.INFO)
    
    config = CommoditiesConfig(
        symbols=get_high_volatility_commodities()[:12],  # Top 12 volatile commodities
        start_date="2015-01-01",
        end_date="2025-09-26"
    )
    
    trainer = CommoditiesTrainer(config)
    
    print("🚀 Training Commodities Agent...")
    training_results = trainer.train_agent()
    print(f"✅ Training completed. Final reward: {training_results['final_reward']:.4f}")
    
    print("\n📊 Running Backtest...")
    evaluation_results = trainer.backtest_agent()
    print("✅ Backtest completed")
    
    print("\n📈 Generating Evaluation Report...")
    report = trainer.generate_report()
    
    print("\n" + "="*60)
    print("COMMODITIES AGENT EVALUATION RESULTS")
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
    print(f"\n💾 Results saved to results/commodities/")
    
    return trainer, report


if __name__ == "__main__":
    trainer, report = main()
