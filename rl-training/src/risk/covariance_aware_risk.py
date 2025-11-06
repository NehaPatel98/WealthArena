"""
Covariance-Aware Portfolio Risk Management Module

This module provides advanced risk management capabilities including:
- Dynamic covariance estimation
- Portfolio optimization with constraints
- Hedging strategies
- Risk parity and minimum variance portfolios
- Factor model risk decomposition
- Stress testing and scenario analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import warnings
from scipy.optimize import minimize
from scipy.linalg import cholesky, solve_triangular
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf, OAS, ShrunkCovariance
from sklearn.preprocessing import StandardScaler
import cvxpy as cp
from scipy import stats

logger = logging.getLogger(__name__)

class CovarianceMethod(Enum):
    SAMPLE = "sample"
    LEDOIT_WOLF = "ledoit_wolf"
    OAS = "oas"
    SHRUNK = "shrunk"
    FACTOR_MODEL = "factor_model"
    DYNAMIC = "dynamic"

class OptimizationMethod(Enum):
    MEAN_VARIANCE = "mean_variance"
    MINIMUM_VARIANCE = "minimum_variance"
    RISK_PARITY = "risk_parity"
    MAXIMUM_SHARPE = "maximum_sharpe"
    EQUAL_WEIGHT = "equal_weight"

@dataclass
class RiskConfig:
    """Configuration for risk management"""
    covariance_method: CovarianceMethod = CovarianceMethod.LEDOIT_WOLF
    optimization_method: OptimizationMethod = OptimizationMethod.RISK_PARITY
    rebalance_frequency: str = "monthly"
    lookback_period: int = 252
    min_weight: float = 0.0
    max_weight: float = 0.1
    target_volatility: Optional[float] = None
    risk_free_rate: float = 0.02
    transaction_costs: float = 0.001
    enable_short_selling: bool = False
    max_leverage: float = 1.0

class CovarianceEstimator:
    """Advanced covariance matrix estimation"""
    
    def __init__(self, method: CovarianceMethod = CovarianceMethod.LEDOIT_WOLF):
        self.method = method
        self.estimator = None
        self._initialize_estimator()
    
    def _initialize_estimator(self):
        """Initialize covariance estimator"""
        if self.method == CovarianceMethod.LEDOIT_WOLF:
            self.estimator = LedoitWolf()
        elif self.method == CovarianceMethod.OAS:
            self.estimator = OAS()
        elif self.method == CovarianceMethod.SHRUNK:
            self.estimator = ShrunkCovariance()
        else:
            self.estimator = None
    
    def estimate_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Estimate covariance matrix"""
        if self.method == CovarianceMethod.SAMPLE:
            return returns.cov()
        elif self.estimator is not None:
            self.estimator.fit(returns)
            return pd.DataFrame(
                self.estimator.covariance_,
                index=returns.columns,
                columns=returns.columns
            )
        elif self.method == CovarianceMethod.FACTOR_MODEL:
            return self._factor_model_covariance(returns)
        elif self.method == CovarianceMethod.DYNAMIC:
            return self._dynamic_covariance(returns)
        else:
            return returns.cov()
    
    def _factor_model_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Estimate covariance using factor model"""
        # Standardize returns
        scaler = StandardScaler()
        returns_scaled = scaler.fit_transform(returns)
        
        # PCA to extract factors
        pca = PCA(n_components=min(10, len(returns.columns)))
        factors = pca.fit_transform(returns_scaled)
        
        # Factor loadings
        loadings = pca.components_.T
        
        # Factor covariance
        factor_cov = np.cov(factors.T)
        
        # Idiosyncratic variance
        residuals = returns_scaled - factors @ loadings.T
        idiosyncratic_var = np.diag(np.var(residuals, axis=0))
        
        # Reconstruct covariance matrix
        covariance = loadings @ factor_cov @ loadings.T + idiosyncratic_var
        
        return pd.DataFrame(
            covariance,
            index=returns.columns,
            columns=returns.columns
        )
    
    def _dynamic_covariance(self, returns: pd.DataFrame, decay_factor: float = 0.94) -> pd.DataFrame:
        """Estimate dynamic covariance using EWMA"""
        # Initialize with sample covariance
        covariance = returns.cov().values
        
        # EWMA update
        for i in range(1, len(returns)):
            current_return = returns.iloc[i].values.reshape(-1, 1)
            covariance = decay_factor * covariance + (1 - decay_factor) * (current_return @ current_return.T)
        
        return pd.DataFrame(
            covariance,
            index=returns.columns,
            columns=returns.columns
        )

class PortfolioOptimizer:
    """Portfolio optimization with various methods"""
    
    def __init__(self, config: RiskConfig):
        self.config = config
        self.covariance_estimator = CovarianceEstimator(config.covariance_method)
    
    def optimize_portfolio(self, 
                          returns: pd.DataFrame,
                          expected_returns: Optional[pd.Series] = None) -> pd.Series:
        """Optimize portfolio weights"""
        if self.config.optimization_method == OptimizationMethod.MEAN_VARIANCE:
            return self._mean_variance_optimization(returns, expected_returns)
        elif self.config.optimization_method == OptimizationMethod.MINIMUM_VARIANCE:
            return self._minimum_variance_optimization(returns)
        elif self.config.optimization_method == OptimizationMethod.RISK_PARITY:
            return self._risk_parity_optimization(returns)
        elif self.config.optimization_method == OptimizationMethod.MAXIMUM_SHARPE:
            return self._maximum_sharpe_optimization(returns, expected_returns)
        elif self.config.optimization_method == OptimizationMethod.EQUAL_WEIGHT:
            return self._equal_weight_optimization(returns)
        else:
            raise ValueError(f"Unknown optimization method: {self.config.optimization_method}")
    
    def _mean_variance_optimization(self, returns: pd.DataFrame, expected_returns: pd.Series) -> pd.Series:
        """Mean-variance optimization"""
        if expected_returns is None:
            expected_returns = returns.mean() * 252
        
        # Estimate covariance
        covariance = self.covariance_estimator.estimate_covariance(returns)
        
        # Set up optimization problem
        n_assets = len(returns.columns)
        weights = cp.Variable(n_assets)
        
        # Objective: maximize expected return - risk penalty
        risk_penalty = 0.5  # Risk aversion parameter
        objective = cp.Maximize(expected_returns.values @ weights - risk_penalty * cp.quad_form(weights, covariance.values))
        
        # Constraints
        constraints = [
            cp.sum(weights) == 1,  # Weights sum to 1
            weights >= self.config.min_weight,  # Minimum weight
            weights <= self.config.max_weight,  # Maximum weight
        ]
        
        if not self.config.enable_short_selling:
            constraints.append(weights >= 0)  # No short selling
        
        # Solve optimization
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status != cp.OPTIMAL:
            logger.warning("Mean-variance optimization failed, using equal weights")
            return self._equal_weight_optimization(returns)
        
        return pd.Series(weights.value, index=returns.columns)
    
    def _minimum_variance_optimization(self, returns: pd.DataFrame) -> pd.Series:
        """Minimum variance optimization"""
        # Estimate covariance
        covariance = self.covariance_estimator.estimate_covariance(returns)
        
        # Set up optimization problem
        n_assets = len(returns.columns)
        weights = cp.Variable(n_assets)
        
        # Objective: minimize portfolio variance
        objective = cp.Minimize(cp.quad_form(weights, covariance.values))
        
        # Constraints
        constraints = [
            cp.sum(weights) == 1,  # Weights sum to 1
            weights >= self.config.min_weight,  # Minimum weight
            weights <= self.config.max_weight,  # Maximum weight
        ]
        
        if not self.config.enable_short_selling:
            constraints.append(weights >= 0)  # No short selling
        
        # Solve optimization
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status != cp.OPTIMAL:
            logger.warning("Minimum variance optimization failed, using equal weights")
            return self._equal_weight_optimization(returns)
        
        return pd.Series(weights.value, index=returns.columns)
    
    def _risk_parity_optimization(self, returns: pd.DataFrame) -> pd.Series:
        """Risk parity optimization"""
        # Estimate covariance
        covariance = self.covariance_estimator.estimate_covariance(returns)
        
        # Set up optimization problem
        n_assets = len(returns.columns)
        weights = cp.Variable(n_assets)
        
        # Risk contributions
        portfolio_variance = cp.quad_form(weights, covariance.values)
        risk_contributions = []
        
        for i in range(n_assets):
            risk_contrib = weights[i] * (covariance.values[i, :] @ weights)
            risk_contributions.append(risk_contrib)
        
        # Objective: minimize sum of squared differences from equal risk contributions
        target_risk_contrib = portfolio_variance / n_assets
        objective = cp.Minimize(cp.sum_squares(cp.hstack(risk_contributions) - target_risk_contrib))
        
        # Constraints
        constraints = [
            cp.sum(weights) == 1,  # Weights sum to 1
            weights >= self.config.min_weight,  # Minimum weight
            weights <= self.config.max_weight,  # Maximum weight
        ]
        
        if not self.config.enable_short_selling:
            constraints.append(weights >= 0)  # No short selling
        
        # Solve optimization
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status != cp.OPTIMAL:
            logger.warning("Risk parity optimization failed, using equal weights")
            return self._equal_weight_optimization(returns)
        
        return pd.Series(weights.value, index=returns.columns)
    
    def _maximum_sharpe_optimization(self, returns: pd.DataFrame, expected_returns: pd.Series) -> pd.Series:
        """Maximum Sharpe ratio optimization"""
        if expected_returns is None:
            expected_returns = returns.mean() * 252
        
        # Estimate covariance
        covariance = self.covariance_estimator.estimate_covariance(returns)
        
        # Set up optimization problem
        n_assets = len(returns.columns)
        weights = cp.Variable(n_assets)
        
        # Expected excess returns
        excess_returns = expected_returns - self.config.risk_free_rate
        
        # Objective: maximize Sharpe ratio (minimize negative Sharpe ratio)
        portfolio_return = excess_returns.values @ weights
        portfolio_variance = cp.quad_form(weights, covariance.values)
        sharpe_ratio = portfolio_return / cp.sqrt(portfolio_variance)
        objective = cp.Maximize(sharpe_ratio)
        
        # Constraints
        constraints = [
            cp.sum(weights) == 1,  # Weights sum to 1
            weights >= self.config.min_weight,  # Minimum weight
            weights <= self.config.max_weight,  # Maximum weight
        ]
        
        if not self.config.enable_short_selling:
            constraints.append(weights >= 0)  # No short selling
        
        # Solve optimization
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status != cp.OPTIMAL:
            logger.warning("Maximum Sharpe optimization failed, using equal weights")
            return self._equal_weight_optimization(returns)
        
        return pd.Series(weights.value, index=returns.columns)
    
    def _equal_weight_optimization(self, returns: pd.DataFrame) -> pd.Series:
        """Equal weight optimization"""
        n_assets = len(returns.columns)
        equal_weight = 1.0 / n_assets
        return pd.Series(equal_weight, index=returns.columns)

class HedgingStrategy:
    """Advanced hedging strategies"""
    
    def __init__(self, config: RiskConfig):
        self.config = config
        self.covariance_estimator = CovarianceEstimator(config.covariance_method)
    
    def calculate_hedge_ratio(self, 
                            asset_returns: pd.Series,
                            hedge_returns: pd.Series,
                            method: str = "ols") -> float:
        """Calculate optimal hedge ratio"""
        if method == "ols":
            # Ordinary Least Squares
            X = hedge_returns.values.reshape(-1, 1)
            y = asset_returns.values
            hedge_ratio = np.linalg.lstsq(X, y, rcond=None)[0][0]
        elif method == "minimum_variance":
            # Minimum variance hedge ratio
            covariance = np.cov(asset_returns, hedge_returns)[0, 1]
            hedge_variance = np.var(hedge_returns)
            hedge_ratio = covariance / hedge_variance if hedge_variance > 0 else 0
        else:
            raise ValueError(f"Unknown hedge ratio method: {method}")
        
        return hedge_ratio
    
    def construct_hedge_portfolio(self, 
                                portfolio_weights: pd.Series,
                                hedge_assets: List[str],
                                returns: pd.DataFrame) -> pd.Series:
        """Construct hedge portfolio"""
        # Calculate portfolio returns
        portfolio_returns = (portfolio_weights * returns).sum(axis=1)
        
        # Calculate hedge ratios for each hedge asset
        hedge_weights = pd.Series(0.0, index=returns.columns)
        
        for hedge_asset in hedge_assets:
            if hedge_asset in returns.columns:
                hedge_ratio = self.calculate_hedge_ratio(portfolio_returns, returns[hedge_asset])
                hedge_weights[hedge_asset] = -hedge_ratio  # Negative for hedging
        
        return hedge_weights
    
    def dynamic_hedging(self, 
                       portfolio_weights: pd.Series,
                       hedge_assets: List[str],
                       returns: pd.DataFrame,
                       lookback_window: int = 63) -> pd.Series:
        """Dynamic hedging with rolling hedge ratios"""
        if len(returns) < lookback_window:
            return self.construct_hedge_portfolio(portfolio_weights, hedge_assets, returns)
        
        # Use recent data for hedge ratio calculation
        recent_returns = returns.tail(lookback_window)
        recent_portfolio_weights = portfolio_weights  # Assume weights are current
        
        return self.construct_hedge_portfolio(recent_portfolio_weights, hedge_assets, recent_returns)

class RiskDecomposer:
    """Risk decomposition and attribution"""
    
    def __init__(self, config: RiskConfig):
        self.config = config
        self.covariance_estimator = CovarianceEstimator(config.covariance_method)
    
    def decompose_risk(self, 
                      portfolio_weights: pd.Series,
                      returns: pd.DataFrame) -> Dict[str, Any]:
        """Decompose portfolio risk into components"""
        # Estimate covariance
        covariance = self.covariance_estimator.estimate_covariance(returns)
        
        # Portfolio variance
        portfolio_variance = portfolio_weights.T @ covariance @ portfolio_weights
        
        # Risk contributions
        risk_contributions = portfolio_weights * (covariance @ portfolio_weights) / portfolio_variance
        
        # Marginal risk contributions
        marginal_contributions = (covariance @ portfolio_weights) / np.sqrt(portfolio_variance)
        
        # Factor decomposition (simplified)
        factor_decomposition = self._factor_risk_decomposition(portfolio_weights, returns)
        
        return {
            'portfolio_variance': portfolio_variance,
            'portfolio_volatility': np.sqrt(portfolio_variance),
            'risk_contributions': risk_contributions,
            'marginal_contributions': marginal_contributions,
            'factor_decomposition': factor_decomposition
        }
    
    def _factor_risk_decomposition(self, 
                                 portfolio_weights: pd.Series,
                                 returns: pd.DataFrame) -> Dict[str, float]:
        """Decompose risk into factor components"""
        # PCA to extract factors
        pca = PCA(n_components=min(5, len(returns.columns)))
        factors = pca.fit_transform(returns)
        
        # Factor loadings
        loadings = pca.components_.T
        
        # Portfolio factor exposures
        portfolio_factor_exposures = portfolio_weights @ loadings
        
        # Factor variances
        factor_variances = np.var(factors, axis=0)
        
        # Factor risk contributions
        factor_risk_contributions = portfolio_factor_exposures**2 * factor_variances
        
        return {
            'factor_exposures': dict(zip([f'Factor_{i+1}' for i in range(len(portfolio_factor_exposures))], 
                                       portfolio_factor_exposures)),
            'factor_risk_contributions': dict(zip([f'Factor_{i+1}' for i in range(len(factor_risk_contributions))], 
                                                factor_risk_contributions)),
            'explained_variance_ratio': pca.explained_variance_ratio_
        }

class StressTester:
    """Stress testing and scenario analysis"""
    
    def __init__(self, config: RiskConfig):
        self.config = config
        self.covariance_estimator = CovarianceEstimator(config.covariance_method)
    
    def run_stress_tests(self, 
                        portfolio_weights: pd.Series,
                        returns: pd.DataFrame,
                        scenarios: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Any]:
        """Run stress tests on portfolio"""
        if scenarios is None:
            scenarios = self._generate_default_scenarios(returns)
        
        stress_results = {}
        
        for scenario_name, scenario_shocks in scenarios.items():
            # Apply shocks to returns
            shocked_returns = self._apply_scenario_shocks(returns, scenario_shocks)
            
            # Calculate portfolio performance under stress
            portfolio_returns = (portfolio_weights * shocked_returns).sum(axis=1)
            
            # Calculate stress metrics
            stress_metrics = self._calculate_stress_metrics(portfolio_returns)
            stress_results[scenario_name] = stress_metrics
        
        return stress_results
    
    def _generate_default_scenarios(self, returns: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Generate default stress test scenarios"""
        scenarios = {
            'market_crash': {col: -0.2 for col in returns.columns},  # 20% decline
            'volatility_spike': {col: 0.0 for col in returns.columns},  # No return, but higher vol
            'sector_rotation': {},  # Custom sector-specific shocks
            'liquidity_crisis': {col: -0.1 for col in returns.columns},  # 10% decline
            'interest_rate_shock': {col: -0.05 for col in returns.columns},  # 5% decline
        }
        
        return scenarios
    
    def _apply_scenario_shocks(self, 
                             returns: pd.DataFrame,
                             shocks: Dict[str, float]) -> pd.DataFrame:
        """Apply scenario shocks to returns"""
        shocked_returns = returns.copy()
        
        for asset, shock in shocks.items():
            if asset in shocked_returns.columns:
                shocked_returns[asset] += shock
        
        return shocked_returns
    
    def _calculate_stress_metrics(self, portfolio_returns: pd.Series) -> Dict[str, float]:
        """Calculate stress test metrics"""
        return {
            'total_return': (1 + portfolio_returns).prod() - 1,
            'max_drawdown': self._calculate_max_drawdown(portfolio_returns),
            'var_95': np.percentile(portfolio_returns, 5),
            'var_99': np.percentile(portfolio_returns, 1),
            'cvar_95': portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean(),
            'cvar_99': portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 1)].mean(),
            'volatility': portfolio_returns.std() * np.sqrt(252)
        }
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        return drawdowns.min()

class CovarianceAwareRiskManager:
    """Main risk management system"""
    
    def __init__(self, config: RiskConfig):
        self.config = config
        self.optimizer = PortfolioOptimizer(config)
        self.hedging_strategy = HedgingStrategy(config)
        self.risk_decomposer = RiskDecomposer(config)
        self.stress_tester = StressTester(config)
        self.covariance_estimator = CovarianceEstimator(config.covariance_method)
    
    def manage_portfolio_risk(self, 
                            returns: pd.DataFrame,
                            current_weights: pd.Series,
                            expected_returns: Optional[pd.Series] = None,
                            hedge_assets: Optional[List[str]] = None) -> Dict[str, Any]:
        """Comprehensive portfolio risk management"""
        # Optimize portfolio
        optimal_weights = self.optimizer.optimize_portfolio(returns, expected_returns)
        
        # Calculate hedging if requested
        hedge_weights = pd.Series(0.0, index=returns.columns)
        if hedge_assets:
            hedge_weights = self.hedging_strategy.dynamic_hedging(
                optimal_weights, hedge_assets, returns
            )
        
        # Combined portfolio
        combined_weights = optimal_weights + hedge_weights
        
        # Risk decomposition
        risk_decomposition = self.risk_decomposer.decompose_risk(combined_weights, returns)
        
        # Stress testing
        stress_results = self.stress_tester.run_stress_tests(combined_weights, returns)
        
        # Rebalancing recommendations
        rebalancing_recommendations = self._generate_rebalancing_recommendations(
            current_weights, combined_weights
        )
        
        return {
            'optimal_weights': optimal_weights,
            'hedge_weights': hedge_weights,
            'combined_weights': combined_weights,
            'risk_decomposition': risk_decomposition,
            'stress_results': stress_results,
            'rebalancing_recommendations': rebalancing_recommendations
        }
    
    def _generate_rebalancing_recommendations(self, 
                                            current_weights: pd.Series,
                                            target_weights: pd.Series) -> Dict[str, Any]:
        """Generate rebalancing recommendations"""
        weight_changes = target_weights - current_weights
        
        # Calculate rebalancing costs
        rebalancing_cost = abs(weight_changes).sum() * self.config.transaction_costs
        
        # Identify significant changes
        significant_changes = weight_changes[abs(weight_changes) > 0.01]  # > 1% change
        
        # Generate recommendations
        recommendations = []
        
        if rebalancing_cost > 0.01:  # > 1% cost
            recommendations.append(f"High rebalancing cost: {rebalancing_cost:.2%}")
        
        for asset, change in significant_changes.items():
            if change > 0:
                recommendations.append(f"Increase {asset} by {change:.2%}")
            else:
                recommendations.append(f"Decrease {asset} by {abs(change):.2%}")
        
        return {
            'rebalancing_cost': rebalancing_cost,
            'significant_changes': significant_changes.to_dict(),
            'recommendations': recommendations
        }
    
    def calculate_risk_metrics(self, 
                             portfolio_weights: pd.Series,
                             returns: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        # Estimate covariance
        covariance = self.covariance_estimator.estimate_covariance(returns)
        
        # Portfolio variance and volatility
        portfolio_variance = portfolio_weights.T @ covariance @ portfolio_weights
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Portfolio returns
        portfolio_returns = (portfolio_weights * returns).sum(axis=1)
        
        # Risk metrics
        var_95 = np.percentile(portfolio_returns, 5)
        var_99 = np.percentile(portfolio_returns, 1)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        cvar_99 = portfolio_returns[portfolio_returns <= var_99].mean()
        
        max_drawdown = self.stress_tester._calculate_max_drawdown(portfolio_returns)
        
        # Beta calculation (if benchmark available)
        beta = 1.0  # Placeholder - would need benchmark data
        
        return {
            'portfolio_variance': portfolio_variance,
            'portfolio_volatility': portfolio_volatility,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'max_drawdown': max_drawdown,
            'beta': beta
        }
