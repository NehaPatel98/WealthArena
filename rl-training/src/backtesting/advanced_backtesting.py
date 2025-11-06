"""
Advanced Backtesting Module

This module provides comprehensive backtesting capabilities including:
- Vectorized backtesting engine
- Walk-forward testing
- Monte Carlo simulation
- Performance attribution
- Risk metrics calculation
- Transaction cost modeling
- Portfolio optimization backtesting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import warnings
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from collections import defaultdict
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import pickle

logger = logging.getLogger(__name__)

class BacktestType(Enum):
    VECTORIZED = "vectorized"
    WALK_FORWARD = "walk_forward"
    MONTE_CARLO = "monte_carlo"
    EVENT_DRIVEN = "event_driven"

@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    backtest_type: BacktestType
    start_date: datetime
    end_date: datetime
    initial_capital: float = 1000000.0
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    risk_free_rate: float = 0.02
    benchmark_symbol: str = "SPY"
    rebalance_frequency: str = "monthly"  # daily, weekly, monthly, quarterly
    max_positions: int = 10
    position_size_limit: float = 0.1  # Max 10% per position
    enable_short_selling: bool = True
    enable_leverage: bool = False
    max_leverage: float = 1.0

@dataclass
class BacktestResult:
    """Backtesting result container"""
    returns: pd.Series
    portfolio_values: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    performance_metrics: Dict[str, float]
    risk_metrics: Dict[str, float]
    attribution: Dict[str, Any]
    benchmark_comparison: Dict[str, float]

class VectorizedBacktester:
    """High-performance vectorized backtesting engine"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.data = None
        self.benchmark_data = None
        self.portfolio_values = None
        self.positions = None
        self.trades = None
        
    def run_backtest(self, 
                    data: pd.DataFrame,
                    strategy: Callable,
                    benchmark_data: Optional[pd.DataFrame] = None) -> BacktestResult:
        """Run vectorized backtest"""
        self.data = data
        self.benchmark_data = benchmark_data
        
        # Initialize portfolio
        self._initialize_portfolio()
        
        # Generate signals using strategy
        signals = strategy(data)
        
        # Calculate returns
        returns = self._calculate_returns(signals)
        
        # Calculate portfolio values
        self.portfolio_values = self._calculate_portfolio_values(returns)
        
        # Generate trades
        self.trades = self._generate_trades(signals)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(returns)
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(returns)
        
        # Calculate attribution
        attribution = self._calculate_attribution(returns, signals)
        
        # Benchmark comparison
        benchmark_comparison = self._calculate_benchmark_comparison(returns)
        
        return BacktestResult(
            returns=returns,
            portfolio_values=self.portfolio_values,
            positions=self.positions,
            trades=self.trades,
            performance_metrics=performance_metrics,
            risk_metrics=risk_metrics,
            attribution=attribution,
            benchmark_comparison=benchmark_comparison
        )
    
    def _initialize_portfolio(self):
        """Initialize portfolio tracking"""
        n_periods = len(self.data)
        n_assets = len(self.data.columns)
        
        self.portfolio_values = pd.Series(index=self.data.index, dtype=float)
        self.portfolio_values.iloc[0] = self.config.initial_capital
        
        self.positions = pd.DataFrame(
            index=self.data.index,
            columns=self.data.columns,
            dtype=float
        )
        self.positions.iloc[0] = 0.0
    
    def _calculate_returns(self, signals: pd.DataFrame) -> pd.Series:
        """Calculate portfolio returns from signals"""
        # Calculate asset returns
        asset_returns = self.data.pct_change().fillna(0)
        
        # Calculate portfolio returns
        portfolio_returns = (signals * asset_returns).sum(axis=1)
        
        # Apply transaction costs
        portfolio_returns = self._apply_transaction_costs(portfolio_returns, signals)
        
        return portfolio_returns
    
    def _apply_transaction_costs(self, returns: pd.Series, signals: pd.DataFrame) -> pd.Series:
        """Apply transaction costs to returns"""
        # Calculate position changes
        position_changes = signals.diff().abs()
        
        # Calculate transaction costs
        transaction_costs = (position_changes * self.config.commission_rate).sum(axis=1)
        slippage_costs = (position_changes * self.config.slippage_rate).sum(axis=1)
        
        # Apply costs to returns
        adjusted_returns = returns - transaction_costs - slippage_costs
        
        return adjusted_returns
    
    def _calculate_portfolio_values(self, returns: pd.Series) -> pd.Series:
        """Calculate portfolio values over time"""
        portfolio_values = self.config.initial_capital * (1 + returns).cumprod()
        return portfolio_values
    
    def _generate_trades(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Generate trade log from signals"""
        trades = []
        
        for date in signals.index[1:]:
            prev_signals = signals.loc[date - pd.Timedelta(days=1)]
            curr_signals = signals.loc[date]
            
            for asset in signals.columns:
                prev_weight = prev_signals[asset]
                curr_weight = curr_signals[asset]
                
                if abs(curr_weight - prev_weight) > 1e-6:  # Significant change
                    trade = {
                        'date': date,
                        'asset': asset,
                        'prev_weight': prev_weight,
                        'curr_weight': curr_weight,
                        'weight_change': curr_weight - prev_weight,
                        'price': self.data.loc[date, asset]
                    }
                    trades.append(trade)
        
        return pd.DataFrame(trades)
    
    def _calculate_performance_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics"""
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - self.config.risk_free_rate) / volatility if volatility > 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annualized_return - self.config.risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
        
        # Calmar ratio
        max_drawdown = self._calculate_max_drawdown(returns)
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Information ratio
        if self.benchmark_data is not None:
            benchmark_returns = self.benchmark_data.pct_change().fillna(0)
            excess_returns = returns - benchmark_returns
            tracking_error = excess_returns.std() * np.sqrt(252)
            information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
        else:
            information_ratio = 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'information_ratio': information_ratio,
            'max_drawdown': max_drawdown
        }
    
    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate risk metrics"""
        # VaR and CVaR
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        
        # Maximum drawdown
        max_dd = self._calculate_max_drawdown(returns)
        
        # Drawdown duration
        dd_duration = self._calculate_drawdown_duration(returns)
        
        # Skewness and Kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Tail ratio
        tail_ratio = self._calculate_tail_ratio(returns)
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'max_drawdown': max_dd,
            'max_drawdown_duration': dd_duration,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'tail_ratio': tail_ratio
        }
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        return drawdowns.min()
    
    def _calculate_drawdown_duration(self, returns: pd.Series) -> int:
        """Calculate maximum drawdown duration in days"""
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        
        max_duration = 0
        current_duration = 0
        
        for dd in drawdowns:
            if dd < 0:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        return max_duration
    
    def _calculate_tail_ratio(self, returns: pd.Series) -> float:
        """Calculate tail ratio (95th percentile / 5th percentile)"""
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)
        return abs(p95 / p5) if p5 != 0 else 0
    
    def _calculate_attribution(self, returns: pd.Series, signals: pd.DataFrame) -> Dict[str, Any]:
        """Calculate performance attribution"""
        asset_returns = self.data.pct_change().fillna(0)
        
        # Brinson attribution
        allocation_effect = self._calculate_allocation_effect(asset_returns, signals)
        selection_effect = self._calculate_selection_effect(asset_returns, signals)
        interaction_effect = self._calculate_interaction_effect(asset_returns, signals)
        
        # Factor attribution (simplified)
        factor_exposure = self._calculate_factor_exposure(signals)
        
        return {
            'allocation_effect': allocation_effect,
            'selection_effect': selection_effect,
            'interaction_effect': interaction_effect,
            'factor_exposure': factor_exposure
        }
    
    def _calculate_allocation_effect(self, asset_returns: pd.DataFrame, signals: pd.DataFrame) -> pd.Series:
        """Calculate allocation effect"""
        # Simplified allocation effect calculation
        benchmark_weights = signals.mean()  # Use average weights as benchmark
        weight_differences = signals - benchmark_weights
        allocation_effect = (weight_differences * asset_returns).sum(axis=1)
        return allocation_effect
    
    def _calculate_selection_effect(self, asset_returns: pd.DataFrame, signals: pd.DataFrame) -> pd.Series:
        """Calculate selection effect"""
        # Simplified selection effect calculation
        benchmark_returns = asset_returns.mean(axis=1)
        selection_effect = (signals * (asset_returns.sub(benchmark_returns, axis=0))).sum(axis=1)
        return selection_effect
    
    def _calculate_interaction_effect(self, asset_returns: pd.DataFrame, signals: pd.DataFrame) -> pd.Series:
        """Calculate interaction effect"""
        # Interaction effect is the residual
        total_effect = (signals * asset_returns).sum(axis=1)
        allocation_effect = self._calculate_allocation_effect(asset_returns, signals)
        selection_effect = self._calculate_selection_effect(asset_returns, signals)
        interaction_effect = total_effect - allocation_effect - selection_effect
        return interaction_effect
    
    def _calculate_factor_exposure(self, signals: pd.DataFrame) -> Dict[str, float]:
        """Calculate factor exposure"""
        # Simplified factor exposure calculation
        return {
            'market_beta': signals.mean().sum(),  # Total market exposure
            'concentration': (signals**2).sum(axis=1).mean(),  # Herfindahl index
            'turnover': signals.diff().abs().sum(axis=1).mean()  # Average turnover
        }
    
    def _calculate_benchmark_comparison(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate benchmark comparison metrics"""
        if self.benchmark_data is None:
            return {}
        
        benchmark_returns = self.benchmark_data.pct_change().fillna(0)
        
        # Align dates
        common_dates = returns.index.intersection(benchmark_returns.index)
        returns_aligned = returns.loc[common_dates]
        benchmark_aligned = benchmark_returns.loc[common_dates]
        
        # Calculate metrics
        excess_returns = returns_aligned - benchmark_aligned
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
        
        # Beta calculation
        covariance = np.cov(returns_aligned, benchmark_aligned)[0, 1]
        benchmark_variance = np.var(benchmark_aligned)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        # Alpha calculation
        alpha = returns_aligned.mean() * 252 - beta * benchmark_aligned.mean() * 252
        
        return {
            'excess_return': excess_returns.mean() * 252,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'beta': beta,
            'alpha': alpha
        }

class WalkForwardBacktester:
    """Walk-forward backtesting engine"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.vectorized_backtester = VectorizedBacktester(config)
        self.results = []
    
    def run_walk_forward(self, 
                        data: pd.DataFrame,
                        strategy: Callable,
                        train_window: int = 252,  # 1 year
                        test_window: int = 63,   # 3 months
                        step_size: int = 21) -> List[BacktestResult]:
        """Run walk-forward backtest"""
        self.results = []
        
        start_idx = train_window
        end_idx = len(data)
        
        for i in range(start_idx, end_idx, step_size):
            if i + test_window > len(data):
                break
            
            # Define train and test periods
            train_start = i - train_window
            train_end = i
            test_start = i
            test_end = min(i + test_window, len(data))
            
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]
            
            # Train strategy on training data
            strategy_params = self._train_strategy(strategy, train_data)
            
            # Test strategy on test data
            test_strategy = lambda data: strategy(data, **strategy_params)
            result = self.vectorized_backtester.run_backtest(test_data, test_strategy)
            
            # Add metadata
            result.metadata = {
                'train_start': train_data.index[0],
                'train_end': train_data.index[-1],
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1],
                'strategy_params': strategy_params
            }
            
            self.results.append(result)
        
        return self.results
    
    def _train_strategy(self, strategy: Callable, train_data: pd.DataFrame) -> Dict[str, Any]:
        """Train strategy on training data"""
        # This is a placeholder - in practice, you would implement
        # parameter optimization or model training here
        return {}
    
    def get_aggregated_results(self) -> BacktestResult:
        """Get aggregated results from walk-forward backtest"""
        if not self.results:
            return None
        
        # Concatenate all results
        all_returns = pd.concat([result.returns for result in self.results])
        all_portfolio_values = pd.concat([result.portfolio_values for result in self.results])
        all_trades = pd.concat([result.trades for result in self.results])
        
        # Calculate aggregated metrics
        performance_metrics = self._calculate_aggregated_metrics(all_returns)
        risk_metrics = self._calculate_aggregated_risk_metrics(all_returns)
        
        return BacktestResult(
            returns=all_returns,
            portfolio_values=all_portfolio_values,
            positions=None,  # Not meaningful for aggregated results
            trades=all_trades,
            performance_metrics=performance_metrics,
            risk_metrics=risk_metrics,
            attribution={},
            benchmark_comparison={}
        )
    
    def _calculate_aggregated_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate aggregated performance metrics"""
        # Use the same metrics as vectorized backtester
        backtester = VectorizedBacktester(self.config)
        return backtester._calculate_performance_metrics(returns)
    
    def _calculate_aggregated_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate aggregated risk metrics"""
        backtester = VectorizedBacktester(self.config)
        return backtester._calculate_risk_metrics(returns)

class MonteCarloBacktester:
    """Monte Carlo backtesting engine"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.vectorized_backtester = VectorizedBacktester(config)
        self.results = []
    
    def run_monte_carlo(self, 
                       data: pd.DataFrame,
                       strategy: Callable,
                       n_simulations: int = 1000,
                       bootstrap_method: str = "block") -> List[BacktestResult]:
        """Run Monte Carlo backtest"""
        self.results = []
        
        for i in range(n_simulations):
            # Generate bootstrap sample
            if bootstrap_method == "block":
                bootstrap_data = self._block_bootstrap(data)
            elif bootstrap_method == "iid":
                bootstrap_data = self._iid_bootstrap(data)
            else:
                raise ValueError(f"Unknown bootstrap method: {bootstrap_method}")
            
            # Run backtest on bootstrap sample
            result = self.vectorized_backtester.run_backtest(bootstrap_data, strategy)
            result.metadata = {'simulation_id': i}
            self.results.append(result)
        
        return self.results
    
    def _block_bootstrap(self, data: pd.DataFrame, block_size: int = 21) -> pd.DataFrame:
        """Block bootstrap sampling"""
        n_blocks = len(data) // block_size
        bootstrap_indices = []
        
        for _ in range(n_blocks):
            start_idx = np.random.randint(0, len(data) - block_size + 1)
            block_indices = range(start_idx, start_idx + block_size)
            bootstrap_indices.extend(block_indices)
        
        return data.iloc[bootstrap_indices].reset_index(drop=True)
    
    def _iid_bootstrap(self, data: pd.DataFrame) -> pd.DataFrame:
        """IID bootstrap sampling"""
        bootstrap_indices = np.random.choice(len(data), size=len(data), replace=True)
        return data.iloc[bootstrap_indices].reset_index(drop=True)
    
    def get_confidence_intervals(self, metric: str, confidence_level: float = 0.95) -> Tuple[float, float]:
        """Get confidence intervals for a metric"""
        if not self.results:
            return 0, 0
        
        values = []
        for result in self.results:
            if metric in result.performance_metrics:
                values.append(result.performance_metrics[metric])
            elif metric in result.risk_metrics:
                values.append(result.risk_metrics[metric])
        
        if not values:
            return 0, 0
        
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(values, lower_percentile)
        upper_bound = np.percentile(values, upper_percentile)
        
        return lower_bound, upper_bound
    
    def get_risk_metrics_distribution(self) -> Dict[str, Dict[str, float]]:
        """Get distribution of risk metrics across simulations"""
        metrics = ['max_drawdown', 'var_95', 'cvar_95', 'sharpe_ratio']
        distributions = {}
        
        for metric in metrics:
            values = []
            for result in self.results:
                if metric in result.performance_metrics:
                    values.append(result.performance_metrics[metric])
                elif metric in result.risk_metrics:
                    values.append(result.risk_metrics[metric])
            
            if values:
                distributions[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'percentile_5': np.percentile(values, 5),
                    'percentile_95': np.percentile(values, 95)
                }
        
        return distributions

class BacktestAnalyzer:
    """Comprehensive backtest analysis and visualization"""
    
    def __init__(self, results: List[BacktestResult]):
        self.results = results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive backtest report"""
        report = {
            'summary': self._generate_summary(),
            'performance_analysis': self._analyze_performance(),
            'risk_analysis': self._analyze_risk(),
            'attribution_analysis': self._analyze_attribution(),
            'benchmark_analysis': self._analyze_benchmark(),
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate executive summary"""
        if not self.results:
            return {}
        
        # Use first result for summary (or aggregate if multiple)
        result = self.results[0]
        
        return {
            'total_return': result.performance_metrics.get('total_return', 0),
            'annualized_return': result.performance_metrics.get('annualized_return', 0),
            'volatility': result.performance_metrics.get('volatility', 0),
            'sharpe_ratio': result.performance_metrics.get('sharpe_ratio', 0),
            'max_drawdown': result.performance_metrics.get('max_drawdown', 0),
            'total_trades': len(result.trades) if result.trades is not None else 0
        }
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance metrics"""
        if not self.results:
            return {}
        
        result = self.results[0]
        
        return {
            'return_analysis': {
                'total_return': result.performance_metrics.get('total_return', 0),
                'annualized_return': result.performance_metrics.get('annualized_return', 0),
                'monthly_returns': self._calculate_monthly_returns(result.returns),
                'yearly_returns': self._calculate_yearly_returns(result.returns)
            },
            'risk_adjusted_returns': {
                'sharpe_ratio': result.performance_metrics.get('sharpe_ratio', 0),
                'sortino_ratio': result.performance_metrics.get('sortino_ratio', 0),
                'calmar_ratio': result.performance_metrics.get('calmar_ratio', 0),
                'information_ratio': result.performance_metrics.get('information_ratio', 0)
            }
        }
    
    def _analyze_risk(self) -> Dict[str, Any]:
        """Analyze risk metrics"""
        if not self.results:
            return {}
        
        result = self.results[0]
        
        return {
            'drawdown_analysis': {
                'max_drawdown': result.performance_metrics.get('max_drawdown', 0),
                'max_drawdown_duration': result.risk_metrics.get('max_drawdown_duration', 0),
                'drawdown_frequency': self._calculate_drawdown_frequency(result.returns)
            },
            'tail_risk': {
                'var_95': result.risk_metrics.get('var_95', 0),
                'var_99': result.risk_metrics.get('var_99', 0),
                'cvar_95': result.risk_metrics.get('cvar_95', 0),
                'cvar_99': result.risk_metrics.get('cvar_99', 0),
                'tail_ratio': result.risk_metrics.get('tail_ratio', 0)
            },
            'distribution_analysis': {
                'skewness': result.risk_metrics.get('skewness', 0),
                'kurtosis': result.risk_metrics.get('kurtosis', 0),
                'jarque_bera_stat': self._calculate_jarque_bera_stat(result.returns)
            }
        }
    
    def _analyze_attribution(self) -> Dict[str, Any]:
        """Analyze performance attribution"""
        if not self.results:
            return {}
        
        result = self.results[0]
        
        return {
            'brinson_attribution': result.attribution,
            'factor_attribution': result.attribution.get('factor_exposure', {}),
            'sector_attribution': self._calculate_sector_attribution(result),
            'time_attribution': self._calculate_time_attribution(result)
        }
    
    def _analyze_benchmark(self) -> Dict[str, Any]:
        """Analyze benchmark comparison"""
        if not self.results:
            return {}
        
        result = self.results[0]
        
        return {
            'benchmark_comparison': result.benchmark_comparison,
            'relative_performance': self._calculate_relative_performance(result),
            'tracking_error_analysis': self._analyze_tracking_error(result)
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        if not self.results:
            return recommendations
        
        result = self.results[0]
        
        # Performance recommendations
        sharpe_ratio = result.performance_metrics.get('sharpe_ratio', 0)
        if sharpe_ratio < 0.5:
            recommendations.append("Consider improving risk-adjusted returns - Sharpe ratio is below 0.5")
        
        max_drawdown = result.performance_metrics.get('max_drawdown', 0)
        if max_drawdown < -0.2:
            recommendations.append("High maximum drawdown detected - consider risk management improvements")
        
        # Risk recommendations
        volatility = result.performance_metrics.get('volatility', 0)
        if volatility > 0.3:
            recommendations.append("High volatility detected - consider diversification or volatility targeting")
        
        # Trading recommendations
        if result.trades is not None and len(result.trades) > 0:
            avg_trade_size = result.trades['weight_change'].abs().mean()
            if avg_trade_size > 0.1:
                recommendations.append("Large average trade size - consider position sizing optimization")
        
        return recommendations
    
    def _calculate_monthly_returns(self, returns: pd.Series) -> pd.Series:
        """Calculate monthly returns"""
        return returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    
    def _calculate_yearly_returns(self, returns: pd.Series) -> pd.Series:
        """Calculate yearly returns"""
        return returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
    
    def _calculate_drawdown_frequency(self, returns: pd.Series) -> float:
        """Calculate drawdown frequency"""
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        
        # Count periods with drawdown > 1%
        significant_drawdowns = (drawdowns < -0.01).sum()
        return significant_drawdowns / len(returns)
    
    def _calculate_jarque_bera_stat(self, returns: pd.Series) -> float:
        """Calculate Jarque-Bera statistic for normality test"""
        try:
            from scipy.stats import jarque_bera
            return jarque_bera(returns.dropna())[0]
        except:
            return 0
    
    def _calculate_sector_attribution(self, result: BacktestResult) -> Dict[str, float]:
        """Calculate sector attribution (placeholder)"""
        return {}
    
    def _calculate_time_attribution(self, result: BacktestResult) -> Dict[str, float]:
        """Calculate time-based attribution"""
        if result.returns is None:
            return {}
        
        # Calculate returns by time of day, day of week, month, etc.
        returns = result.returns
        
        time_attribution = {
            'monthly_returns': returns.resample('M').apply(lambda x: (1 + x).prod() - 1).to_dict(),
            'quarterly_returns': returns.resample('Q').apply(lambda x: (1 + x).prod() - 1).to_dict()
        }
        
        return time_attribution
    
    def _calculate_relative_performance(self, result: BacktestResult) -> Dict[str, float]:
        """Calculate relative performance metrics"""
        if not result.benchmark_comparison:
            return {}
        
        return {
            'excess_return': result.benchmark_comparison.get('excess_return', 0),
            'tracking_error': result.benchmark_comparison.get('tracking_error', 0),
            'information_ratio': result.benchmark_comparison.get('information_ratio', 0)
        }
    
    def _analyze_tracking_error(self, result: BacktestResult) -> Dict[str, float]:
        """Analyze tracking error characteristics"""
        if not result.benchmark_comparison:
            return {}
        
        tracking_error = result.benchmark_comparison.get('tracking_error', 0)
        
        return {
            'tracking_error': tracking_error,
            'tracking_error_annualized': tracking_error * np.sqrt(252),
            'tracking_error_volatility': tracking_error / result.performance_metrics.get('volatility', 1)
        }
