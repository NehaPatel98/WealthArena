"""
Multi-Objective Optimization Module

This module implements multi-objective optimization for RL agents including:
- Reward shaping with multiple objectives
- Pareto optimization
- Weighted sum methods
- Constraint handling
- Dynamic objective weighting
- Performance metrics tracking
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import defaultdict

logger = logging.getLogger(__name__)

class ObjectiveType(Enum):
    RETURN = "return"
    RISK = "risk"
    SHARPE = "sharpe"
    SORTINO = "sortino"
    MAX_DRAWDOWN = "max_drawdown"
    VOLATILITY = "volatility"
    LIQUIDITY = "liquidity"
    ESG = "esg"
    TURNOVER = "turnover"
    SLIPPAGE = "slippage"

@dataclass
class ObjectiveConfig:
    """Configuration for an objective"""
    objective_type: ObjectiveType
    weight: float = 1.0
    target_value: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    priority: int = 1  # Higher number = higher priority

@dataclass
class MultiObjectiveConfig:
    """Configuration for multi-objective optimization"""
    objectives: List[ObjectiveConfig]
    optimization_method: str = "weighted_sum"  # "weighted_sum", "pareto", "constraint"
    adaptive_weights: bool = True
    weight_update_frequency: int = 100
    pareto_front_size: int = 50

class RewardShaper:
    """Multi-objective reward shaping"""
    
    def __init__(self, config: MultiObjectiveConfig):
        self.config = config
        self.objective_weights = {obj.objective_type: obj.weight for obj in config.objectives}
        self.objective_history = defaultdict(list)
        self.performance_metrics = {}
        
    def calculate_reward(self, 
                        returns: np.ndarray,
                        portfolio_state: Dict[str, Any],
                        market_data: Dict[str, Any],
                        step: int) -> Tuple[float, Dict[str, float]]:
        """Calculate multi-objective reward"""
        objective_values = {}
        total_reward = 0.0
        
        for obj_config in self.config.objectives:
            obj_value = self._calculate_objective_value(
                obj_config, returns, portfolio_state, market_data, step
            )
            objective_values[obj_config.objective_type.value] = obj_value
            
            # Apply constraints
            if obj_config.min_value is not None and obj_value < obj_config.min_value:
                penalty = (obj_config.min_value - obj_value) * 10.0
                total_reward -= penalty
                logger.warning(f"Objective {obj_config.objective_type.value} below minimum: {obj_value:.4f}")
            
            if obj_config.max_value is not None and obj_value > obj_config.max_value:
                penalty = (obj_value - obj_config.max_value) * 10.0
                total_reward -= penalty
                logger.warning(f"Objective {obj_config.objective_type.value} above maximum: {obj_value:.4f}")
            
            # Add weighted objective to total reward
            weight = self._get_adaptive_weight(obj_config, step)
            total_reward += weight * obj_value
            
            # Store objective history
            self.objective_history[obj_config.objective_type].append(obj_value)
        
        # Update adaptive weights if enabled
        if self.config.adaptive_weights and step % self.config.weight_update_frequency == 0:
            self._update_adaptive_weights(step)
        
        return total_reward, objective_values
    
    def _calculate_objective_value(self, 
                                 obj_config: ObjectiveConfig,
                                 returns: np.ndarray,
                                 portfolio_state: Dict[str, Any],
                                 market_data: Dict[str, Any],
                                 step: int) -> float:
        """Calculate value for a specific objective"""
        if obj_config.objective_type == ObjectiveType.RETURN:
            return self._calculate_return_objective(returns, obj_config)
        elif obj_config.objective_type == ObjectiveType.RISK:
            return self._calculate_risk_objective(returns, obj_config)
        elif obj_config.objective_type == ObjectiveType.SHARPE:
            return self._calculate_sharpe_objective(returns, obj_config)
        elif obj_config.objective_type == ObjectiveType.SORTINO:
            return self._calculate_sortino_objective(returns, obj_config)
        elif obj_config.objective_type == ObjectiveType.MAX_DRAWDOWN:
            return self._calculate_max_drawdown_objective(returns, obj_config)
        elif obj_config.objective_type == ObjectiveType.VOLATILITY:
            return self._calculate_volatility_objective(returns, obj_config)
        elif obj_config.objective_type == ObjectiveType.LIQUIDITY:
            return self._calculate_liquidity_objective(portfolio_state, market_data, obj_config)
        elif obj_config.objective_type == ObjectiveType.ESG:
            return self._calculate_esg_objective(portfolio_state, market_data, obj_config)
        elif obj_config.objective_type == ObjectiveType.TURNOVER:
            return self._calculate_turnover_objective(portfolio_state, obj_config)
        elif obj_config.objective_type == ObjectiveType.SLIPPAGE:
            return self._calculate_slippage_objective(portfolio_state, market_data, obj_config)
        else:
            return 0.0
    
    def _calculate_return_objective(self, returns: np.ndarray, config: ObjectiveConfig) -> float:
        """Calculate return objective"""
        if len(returns) == 0:
            return 0.0
        
        total_return = np.prod(1 + returns) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        
        if config.target_value is not None:
            # Reward closeness to target
            return -abs(annualized_return - config.target_value)
        else:
            return annualized_return
    
    def _calculate_risk_objective(self, returns: np.ndarray, config: ObjectiveConfig) -> float:
        """Calculate risk objective (negative volatility)"""
        if len(returns) < 2:
            return 0.0
        
        volatility = np.std(returns) * np.sqrt(252)
        
        if config.target_value is not None:
            # Reward closeness to target volatility
            return -abs(volatility - config.target_value)
        else:
            # Minimize volatility
            return -volatility
    
    def _calculate_sharpe_objective(self, returns: np.ndarray, config: ObjectiveConfig) -> float:
        """Calculate Sharpe ratio objective"""
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns) * 252
        volatility = np.std(returns) * np.sqrt(252)
        
        if volatility == 0:
            return 0.0
        
        sharpe_ratio = mean_return / volatility
        
        if config.target_value is not None:
            # Reward closeness to target Sharpe ratio
            return -abs(sharpe_ratio - config.target_value)
        else:
            return sharpe_ratio
    
    def _calculate_sortino_objective(self, returns: np.ndarray, config: ObjectiveConfig) -> float:
        """Calculate Sortino ratio objective"""
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns) * 252
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return mean_return  # No downside risk
        
        downside_volatility = np.std(downside_returns) * np.sqrt(252)
        
        if downside_volatility == 0:
            return mean_return
        
        sortino_ratio = mean_return / downside_volatility
        
        if config.target_value is not None:
            return -abs(sortino_ratio - config.target_value)
        else:
            return sortino_ratio
    
    def _calculate_max_drawdown_objective(self, returns: np.ndarray, config: ObjectiveConfig) -> float:
        """Calculate max drawdown objective (negative)"""
        if len(returns) == 0:
            return 0.0
        
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        if config.target_value is not None:
            return -abs(max_drawdown - config.target_value)
        else:
            return max_drawdown  # Negative value (minimize)
    
    def _calculate_volatility_objective(self, returns: np.ndarray, config: ObjectiveConfig) -> float:
        """Calculate volatility objective"""
        if len(returns) < 2:
            return 0.0
        
        volatility = np.std(returns) * np.sqrt(252)
        
        if config.target_value is not None:
            return -abs(volatility - config.target_value)
        else:
            return -volatility  # Minimize volatility
    
    def _calculate_liquidity_objective(self, portfolio_state: Dict[str, Any], 
                                     market_data: Dict[str, Any], config: ObjectiveConfig) -> float:
        """Calculate liquidity objective"""
        if 'positions' not in portfolio_state:
            return 0.0
        
        total_liquidity_score = 0.0
        total_weight = 0.0
        
        for asset, position in portfolio_state['positions'].items():
            if asset in market_data and 'volume' in market_data[asset]:
                # Liquidity score based on volume
                volume = market_data[asset]['volume']
                liquidity_score = min(volume / 1000000, 1.0)  # Normalize to [0, 1]
                
                total_liquidity_score += abs(position) * liquidity_score
                total_weight += abs(position)
        
        if total_weight == 0:
            return 0.0
        
        avg_liquidity = total_liquidity_score / total_weight
        
        if config.target_value is not None:
            return -abs(avg_liquidity - config.target_value)
        else:
            return avg_liquidity
    
    def _calculate_esg_objective(self, portfolio_state: Dict[str, Any], 
                               market_data: Dict[str, Any], config: ObjectiveConfig) -> float:
        """Calculate ESG objective"""
        if 'positions' not in portfolio_state:
            return 0.0
        
        total_esg_score = 0.0
        total_weight = 0.0
        
        for asset, position in portfolio_state['positions'].items():
            if asset in market_data and 'esg_score' in market_data[asset]:
                esg_score = market_data[asset]['esg_score']
                total_esg_score += abs(position) * esg_score
                total_weight += abs(position)
        
        if total_weight == 0:
            return 0.0
        
        avg_esg_score = total_esg_score / total_weight
        
        if config.target_value is not None:
            return -abs(avg_esg_score - config.target_value)
        else:
            return avg_esg_score
    
    def _calculate_turnover_objective(self, portfolio_state: Dict[str, Any], config: ObjectiveConfig) -> float:
        """Calculate turnover objective (negative)"""
        if 'previous_weights' not in portfolio_state or 'current_weights' not in portfolio_state:
            return 0.0
        
        prev_weights = np.array(portfolio_state['previous_weights'])
        curr_weights = np.array(portfolio_state['current_weights'])
        
        turnover = np.sum(np.abs(curr_weights - prev_weights))
        
        if config.target_value is not None:
            return -abs(turnover - config.target_value)
        else:
            return -turnover  # Minimize turnover
    
    def _calculate_slippage_objective(self, portfolio_state: Dict[str, Any], 
                                    market_data: Dict[str, Any], config: ObjectiveConfig) -> float:
        """Calculate slippage objective (negative)"""
        if 'slippage' not in portfolio_state:
            return 0.0
        
        slippage = portfolio_state['slippage']
        
        if config.target_value is not None:
            return -abs(slippage - config.target_value)
        else:
            return -slippage  # Minimize slippage
    
    def _get_adaptive_weight(self, obj_config: ObjectiveConfig, step: int) -> float:
        """Get adaptive weight for objective"""
        if not self.config.adaptive_weights:
            return obj_config.weight
        
        # Simple adaptive weighting based on performance
        if len(self.objective_history[obj_config.objective_type]) < 10:
            return obj_config.weight
        
        recent_performance = np.mean(self.objective_history[obj_config.objective_type][-10:])
        
        # Adjust weight based on performance
        if recent_performance < 0:
            # Poor performance, increase weight
            return obj_config.weight * 1.2
        elif recent_performance > 0.1:
            # Good performance, decrease weight
            return obj_config.weight * 0.8
        else:
            return obj_config.weight
    
    def _update_adaptive_weights(self, step: int):
        """Update adaptive weights based on recent performance"""
        for obj_config in self.config.objectives:
            if len(self.objective_history[obj_config.objective_type]) < 20:
                continue
            
            recent_performance = np.mean(self.objective_history[obj_config.objective_type][-20:])
            performance_std = np.std(self.objective_history[obj_config.objective_type][-20:])
            
            # Update weight based on performance and stability
            if performance_std < 0.01:  # Stable performance
                if recent_performance < 0:
                    obj_config.weight *= 1.1
                else:
                    obj_config.weight *= 0.9
            else:  # Unstable performance
                obj_config.weight *= 0.95  # Reduce weight for unstable objectives
        
        # Normalize weights
        total_weight = sum(obj_config.weight for obj_config in self.config.objectives)
        for obj_config in self.config.objectives:
            obj_config.weight /= total_weight

class ParetoOptimizer:
    """Pareto front optimization"""
    
    def __init__(self, config: MultiObjectiveConfig):
        self.config = config
        self.pareto_front = []
        self.pareto_solutions = []
    
    def add_solution(self, objectives: Dict[str, float], solution: Any):
        """Add solution to Pareto front"""
        obj_values = [objectives[obj.objective_type.value] for obj in self.config.objectives]
        
        # Check if solution is dominated
        is_dominated = False
        dominated_indices = []
        
        for i, (front_obj, front_sol) in enumerate(zip(self.pareto_front, self.pareto_solutions)):
            if self._dominates(front_obj, obj_values):
                is_dominated = True
                break
            elif self._dominates(obj_values, front_obj):
                dominated_indices.append(i)
        
        if not is_dominated:
            # Remove dominated solutions
            for i in reversed(dominated_indices):
                self.pareto_front.pop(i)
                self.pareto_solutions.pop(i)
            
            # Add new solution
            self.pareto_front.append(obj_values)
            self.pareto_solutions.append(solution)
            
            # Limit Pareto front size
            if len(self.pareto_front) > self.config.pareto_front_size:
                self._prune_pareto_front()
    
    def _dominates(self, obj1: List[float], obj2: List[float]) -> bool:
        """Check if obj1 dominates obj2"""
        better_in_at_least_one = False
        
        for o1, o2 in zip(obj1, obj2):
            if o1 < o2:  # obj1 is worse in this objective
                return False
            elif o1 > o2:  # obj1 is better in this objective
                better_in_at_least_one = True
        
        return better_in_at_least_one
    
    def _prune_pareto_front(self):
        """Prune Pareto front to maintain size limit"""
        # Simple pruning: remove solutions with lowest hypervolume contribution
        if len(self.pareto_front) <= self.config.pareto_front_size:
            return
        
        # Calculate hypervolume contribution for each solution
        contributions = []
        for i, obj_values in enumerate(self.pareto_front):
            contribution = self._calculate_hypervolume_contribution(i)
            contributions.append((contribution, i))
        
        # Sort by contribution and remove worst solutions
        contributions.sort()
        num_to_remove = len(self.pareto_front) - self.config.pareto_front_size
        
        for _, idx in contributions[:num_to_remove]:
            self.pareto_front.pop(idx)
            self.pareto_solutions.pop(idx)
    
    def _calculate_hypervolume_contribution(self, solution_idx: int) -> float:
        """Calculate hypervolume contribution of a solution"""
        # Simplified hypervolume calculation
        obj_values = self.pareto_front[solution_idx]
        return sum(obj_values)  # Simple sum as approximation
    
    def get_best_solution(self, preference_weights: Optional[Dict[str, float]] = None) -> Any:
        """Get best solution based on preferences"""
        if not self.pareto_solutions:
            return None
        
        if preference_weights is None:
            # Return solution with highest sum of objectives
            best_idx = np.argmax([sum(obj_values) for obj_values in self.pareto_front])
            return self.pareto_solutions[best_idx]
        
        # Weighted sum based on preferences
        best_score = float('-inf')
        best_idx = 0
        
        for i, obj_values in enumerate(self.pareto_front):
            score = 0.0
            for j, obj_config in enumerate(self.config.objectives):
                weight = preference_weights.get(obj_config.objective_type.value, 1.0)
                score += weight * obj_values[j]
            
            if score > best_score:
                best_score = score
                best_idx = i
        
        return self.pareto_solutions[best_idx]

class ConstraintHandler:
    """Handle constraints in multi-objective optimization"""
    
    def __init__(self, constraints: List[Callable]):
        self.constraints = constraints
    
    def check_constraints(self, solution: Any) -> Tuple[bool, List[str]]:
        """Check if solution satisfies constraints"""
        violations = []
        
        for constraint in self.constraints:
            try:
                if not constraint(solution):
                    violations.append(f"Constraint violated: {constraint.__name__}")
            except Exception as e:
                violations.append(f"Constraint error: {str(e)}")
        
        return len(violations) == 0, violations
    
    def apply_penalties(self, reward: float, violations: List[str]) -> float:
        """Apply penalties for constraint violations"""
        penalty = len(violations) * 10.0  # Base penalty per violation
        return reward - penalty

class MultiObjectiveTracker:
    """Track multi-objective performance over time"""
    
    def __init__(self, config: MultiObjectiveConfig):
        self.config = config
        self.history = defaultdict(list)
        self.performance_metrics = {}
    
    def update(self, objectives: Dict[str, float], step: int):
        """Update tracking with new objective values"""
        for obj_type, value in objectives.items():
            self.history[obj_type].append(value)
        
        # Calculate performance metrics
        self._calculate_performance_metrics(step)
    
    def _calculate_performance_metrics(self, step: int):
        """Calculate performance metrics"""
        for obj_type in self.history:
            values = self.history[obj_type]
            
            if len(values) < 2:
                continue
            
            self.performance_metrics[obj_type] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'trend': self._calculate_trend(values),
                'stability': 1.0 / (1.0 + np.std(values))  # Higher is more stable
            }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend slope"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        return slope
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        report = {
            'performance_metrics': self.performance_metrics,
            'objective_correlations': self._calculate_objective_correlations(),
            'convergence_analysis': self._analyze_convergence(),
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _calculate_objective_correlations(self) -> Dict[str, float]:
        """Calculate correlations between objectives"""
        correlations = {}
        obj_types = list(self.history.keys())
        
        for i, obj1 in enumerate(obj_types):
            for obj2 in obj_types[i+1:]:
                if len(self.history[obj1]) == len(self.history[obj2]):
                    corr = np.corrcoef(self.history[obj1], self.history[obj2])[0, 1]
                    correlations[f"{obj1}_{obj2}"] = corr
        
        return correlations
    
    def _analyze_convergence(self) -> Dict[str, Any]:
        """Analyze convergence of objectives"""
        convergence = {}
        
        for obj_type, values in self.history.items():
            if len(values) < 50:
                continue
            
            # Calculate rolling variance
            window_size = min(20, len(values) // 2)
            rolling_var = [np.var(values[i:i+window_size]) for i in range(len(values)-window_size+1)]
            
            # Check if variance is decreasing (converging)
            if len(rolling_var) > 10:
                early_var = np.mean(rolling_var[:len(rolling_var)//2])
                late_var = np.mean(rolling_var[len(rolling_var)//2:])
                convergence[obj_type] = {
                    'converging': late_var < early_var * 0.8,
                    'convergence_ratio': late_var / early_var if early_var > 0 else 1.0
                }
        
        return convergence
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on performance"""
        recommendations = []
        
        for obj_type, metrics in self.performance_metrics.items():
            if metrics['stability'] < 0.5:
                recommendations.append(f"Consider stabilizing {obj_type} objective (stability: {metrics['stability']:.3f})")
            
            if metrics['trend'] < -0.01:
                recommendations.append(f"Monitor declining trend in {obj_type} objective")
            
            if metrics['std'] > abs(metrics['mean']) * 0.5:
                recommendations.append(f"High volatility in {obj_type} objective - consider smoothing")
        
        return recommendations
