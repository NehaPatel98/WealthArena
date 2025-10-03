"""
Evaluation Module for WealthArena Trading System

This module provides comprehensive evaluation capabilities for the trained
multi-agent trading system, including performance metrics, risk analysis,
and comparative evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class TradingEvaluator:
    """
    Trading system evaluator for WealthArena
    
    Provides comprehensive evaluation capabilities including performance metrics,
    risk analysis, and comparative evaluation of different trading strategies.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Evaluation parameters
        self.risk_free_rate = self.config.get("risk_free_rate", 0.02)  # 2% annual
        self.benchmark_return = self.config.get("benchmark_return", 0.08)  # 8% annual
        self.evaluation_periods = self.config.get("evaluation_periods", 252)  # Trading days
        
        # Visualization settings
        self.plot_style = self.config.get("plot_style", "seaborn")
        self.figure_size = self.config.get("figure_size", (12, 8))
        
        # Setup plotting
        plt.style.use(self.plot_style)
        
        logger.info("Trading evaluator initialized")
    
    def evaluate_portfolio(self, 
                          portfolio_data: pd.DataFrame,
                          benchmark_data: pd.DataFrame = None) -> Dict[str, Any]:
        """Evaluate portfolio performance"""
        
        if portfolio_data.empty:
            return {}
        
        try:
            # Calculate returns
            returns = self._calculate_returns(portfolio_data)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(returns)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(returns)
            
            # Calculate drawdown metrics
            drawdown_metrics = self._calculate_drawdown_metrics(portfolio_data)
            
            # Calculate benchmark comparison
            benchmark_metrics = {}
            if benchmark_data is not None:
                benchmark_metrics = self._calculate_benchmark_metrics(returns, benchmark_data)
            
            # Combine all metrics
            evaluation_results = {
                "performance": performance_metrics,
                "risk": risk_metrics,
                "drawdown": drawdown_metrics,
                "benchmark": benchmark_metrics,
                "returns": returns,
                "portfolio_data": portfolio_data
            }
            
            logger.info("Portfolio evaluation completed")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error evaluating portfolio: {e}")
            return {}
    
    def _calculate_returns(self, portfolio_data: pd.DataFrame) -> pd.Series:
        """Calculate portfolio returns"""
        
        if "portfolio_value" in portfolio_data.columns:
            values = portfolio_data["portfolio_value"]
        elif "value" in portfolio_data.columns:
            values = portfolio_data["value"]
        else:
            # Assume first numeric column is portfolio value
            numeric_columns = portfolio_data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                values = portfolio_data[numeric_columns[0]]
            else:
                raise ValueError("No portfolio value column found")
        
        # Calculate returns
        returns = values.pct_change().dropna()
        
        return returns
    
    def _calculate_performance_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics"""
        
        if returns.empty:
            return {}
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (self.evaluation_periods / len(returns)) - 1
        
        # Risk-adjusted metrics
        volatility = returns.std() * np.sqrt(self.evaluation_periods)
        sharpe_ratio = (annualized_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        
        # Additional metrics
        max_return = returns.max()
        min_return = returns.min()
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
        avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
        avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0
        
        profit_factor = abs(positive_returns.sum() / negative_returns.sum()) if negative_returns.sum() != 0 else np.inf
        
        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_return": max_return,
            "min_return": min_return,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor
        }
    
    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate risk metrics"""
        
        if returns.empty:
            return {}
        
        # Value at Risk (VaR)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Conditional Value at Risk (CVaR)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else 0
        cvar_99 = returns[returns <= var_99].mean() if len(returns[returns <= var_99]) > 0 else 0
        
        # Downside deviation
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(self.evaluation_periods) if len(downside_returns) > 0 else 0
        
        # Sortino ratio
        annualized_return = (1 + returns).prod() ** (self.evaluation_periods / len(returns)) - 1
        sortino_ratio = (annualized_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # Calmar ratio (annualized return / max drawdown)
        max_drawdown = self._calculate_max_drawdown(returns)
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            "var_95": var_95,
            "var_99": var_99,
            "cvar_95": cvar_95,
            "cvar_99": cvar_99,
            "downside_deviation": downside_deviation,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio
        }
    
    def _calculate_drawdown_metrics(self, portfolio_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate drawdown metrics"""
        
        if "portfolio_value" in portfolio_data.columns:
            values = portfolio_data["portfolio_value"]
        elif "value" in portfolio_data.columns:
            values = portfolio_data["value"]
        else:
            numeric_columns = portfolio_data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                values = portfolio_data[numeric_columns[0]]
            else:
                return {}
        
        # Calculate drawdown
        peak = values.expanding().max()
        drawdown = (values - peak) / peak
        
        # Drawdown metrics
        max_drawdown = drawdown.min()
        max_drawdown_duration = self._calculate_max_drawdown_duration(drawdown)
        avg_drawdown = drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else 0
        
        # Recovery time
        recovery_time = self._calculate_recovery_time(drawdown)
        
        return {
            "max_drawdown": max_drawdown,
            "max_drawdown_duration": max_drawdown_duration,
            "avg_drawdown": avg_drawdown,
            "recovery_time": recovery_time
        }
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns"""
        
        if returns.empty:
            return 0
        
        # Calculate cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = cumulative_returns.expanding().max()
        
        # Calculate drawdown
        drawdown = (cumulative_returns - running_max) / running_max
        
        return drawdown.min()
    
    def _calculate_max_drawdown_duration(self, drawdown: pd.Series) -> int:
        """Calculate maximum drawdown duration"""
        
        if drawdown.empty:
            return 0
        
        # Find periods where drawdown is negative
        in_drawdown = drawdown < 0
        
        # Calculate consecutive periods
        drawdown_periods = []
        current_period = 0
        
        for is_drawdown in in_drawdown:
            if is_drawdown:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                    current_period = 0
        
        # Add final period if still in drawdown
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        return max(drawdown_periods) if drawdown_periods else 0
    
    def _calculate_recovery_time(self, drawdown: pd.Series) -> float:
        """Calculate average recovery time"""
        
        if drawdown.empty:
            return 0
        
        # Find drawdown periods
        in_drawdown = drawdown < 0
        drawdown_starts = in_drawdown & ~in_drawdown.shift(1).fillna(False)
        drawdown_ends = ~in_drawdown & in_drawdown.shift(1).fillna(False)
        
        # Calculate recovery times
        recovery_times = []
        current_start = None
        
        for i, (is_start, is_end) in enumerate(zip(drawdown_starts, drawdown_ends)):
            if is_start:
                current_start = i
            elif is_end and current_start is not None:
                recovery_times.append(i - current_start)
                current_start = None
        
        return np.mean(recovery_times) if recovery_times else 0
    
    def _calculate_benchmark_metrics(self, 
                                   returns: pd.Series, 
                                   benchmark_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate benchmark comparison metrics"""
        
        if returns.empty or benchmark_data.empty:
            return {}
        
        # Calculate benchmark returns
        if "returns" in benchmark_data.columns:
            benchmark_returns = benchmark_data["returns"]
        else:
            benchmark_returns = self._calculate_returns(benchmark_data)
        
        # Align returns
        common_index = returns.index.intersection(benchmark_returns.index)
        if len(common_index) == 0:
            return {}
        
        returns_aligned = returns.loc[common_index]
        benchmark_aligned = benchmark_returns.loc[common_index]
        
        # Calculate excess returns
        excess_returns = returns_aligned - benchmark_aligned
        
        # Calculate metrics
        alpha = excess_returns.mean() * self.evaluation_periods
        beta = returns_aligned.cov(benchmark_aligned) / benchmark_aligned.var() if benchmark_aligned.var() > 0 else 0
        
        # Information ratio
        tracking_error = excess_returns.std() * np.sqrt(self.evaluation_periods)
        information_ratio = alpha / tracking_error if tracking_error > 0 else 0
        
        # Correlation
        correlation = returns_aligned.corr(benchmark_aligned)
        
        return {
            "alpha": alpha,
            "beta": beta,
            "information_ratio": information_ratio,
            "tracking_error": tracking_error,
            "correlation": correlation
        }
    
    def create_performance_report(self, 
                                evaluation_results: Dict[str, Any],
                                save_path: str = None) -> str:
        """Create comprehensive performance report"""
        
        try:
            # Create report
            report = self._generate_report_text(evaluation_results)
            
            # Save report
            if save_path:
                with open(save_path, 'w') as f:
                    f.write(report)
                logger.info(f"Performance report saved to: {save_path}")
            
            return report
            
        except Exception as e:
            logger.error(f"Error creating performance report: {e}")
            return ""
    
    def _generate_report_text(self, evaluation_results: Dict[str, Any]) -> str:
        """Generate performance report text"""
        
        report = []
        report.append("=" * 80)
        report.append("WEALTHARENA TRADING SYSTEM PERFORMANCE REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Performance metrics
        if "performance" in evaluation_results:
            report.append("PERFORMANCE METRICS")
            report.append("-" * 40)
            performance = evaluation_results["performance"]
            for metric, value in performance.items():
                if isinstance(value, float):
                    report.append(f"{metric.replace('_', ' ').title()}: {value:.4f}")
                else:
                    report.append(f"{metric.replace('_', ' ').title()}: {value}")
            report.append("")
        
        # Risk metrics
        if "risk" in evaluation_results:
            report.append("RISK METRICS")
            report.append("-" * 40)
            risk = evaluation_results["risk"]
            for metric, value in risk.items():
                if isinstance(value, float):
                    report.append(f"{metric.replace('_', ' ').title()}: {value:.4f}")
                else:
                    report.append(f"{metric.replace('_', ' ').title()}: {value}")
            report.append("")
        
        # Drawdown metrics
        if "drawdown" in evaluation_results:
            report.append("DRAWDOWN METRICS")
            report.append("-" * 40)
            drawdown = evaluation_results["drawdown"]
            for metric, value in drawdown.items():
                if isinstance(value, float):
                    report.append(f"{metric.replace('_', ' ').title()}: {value:.4f}")
                else:
                    report.append(f"{metric.replace('_', ' ').title()}: {value}")
            report.append("")
        
        # Benchmark metrics
        if "benchmark" in evaluation_results and evaluation_results["benchmark"]:
            report.append("BENCHMARK COMPARISON")
            report.append("-" * 40)
            benchmark = evaluation_results["benchmark"]
            for metric, value in benchmark.items():
                if isinstance(value, float):
                    report.append(f"{metric.replace('_', ' ').title()}: {value:.4f}")
                else:
                    report.append(f"{metric.replace('_', ' ').title()}: {value}")
            report.append("")
        
        return "\n".join(report)
    
    def create_visualizations(self, 
                            evaluation_results: Dict[str, Any],
                            save_dir: str = None) -> Dict[str, str]:
        """Create performance visualizations"""
        
        try:
            visualizations = {}
            
            # Portfolio value plot
            if "portfolio_data" in evaluation_results:
                portfolio_plot = self._create_portfolio_plot(evaluation_results["portfolio_data"])
                visualizations["portfolio_plot"] = portfolio_plot
            
            # Returns distribution plot
            if "returns" in evaluation_results:
                returns_plot = self._create_returns_plot(evaluation_results["returns"])
                visualizations["returns_plot"] = returns_plot
            
            # Drawdown plot
            if "portfolio_data" in evaluation_results:
                drawdown_plot = self._create_drawdown_plot(evaluation_results["portfolio_data"])
                visualizations["drawdown_plot"] = drawdown_plot
            
            # Risk-return scatter plot
            if "performance" in evaluation_results:
                risk_return_plot = self._create_risk_return_plot(evaluation_results["performance"])
                visualizations["risk_return_plot"] = risk_return_plot
            
            # Save visualizations
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                for name, plot_path in visualizations.items():
                    if plot_path and os.path.exists(plot_path):
                        dest_path = os.path.join(save_dir, f"{name}.png")
                        os.rename(plot_path, dest_path)
                        visualizations[name] = dest_path
            
            logger.info(f"Created {len(visualizations)} visualizations")
            return visualizations
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            return {}
    
    def _create_portfolio_plot(self, portfolio_data: pd.DataFrame) -> str:
        """Create portfolio value plot"""
        
        try:
            plt.figure(figsize=self.figure_size)
            
            if "portfolio_value" in portfolio_data.columns:
                plt.plot(portfolio_data.index, portfolio_data["portfolio_value"], label="Portfolio Value")
            
            plt.title("Portfolio Value Over Time")
            plt.xlabel("Time")
            plt.ylabel("Portfolio Value")
            plt.legend()
            plt.grid(True)
            
            # Save plot
            plot_path = f"portfolio_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return plot_path
            
        except Exception as e:
            logger.error(f"Error creating portfolio plot: {e}")
            return None
    
    def _create_returns_plot(self, returns: pd.Series) -> str:
        """Create returns distribution plot"""
        
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Returns histogram
            ax1.hist(returns, bins=50, alpha=0.7, edgecolor='black')
            ax1.set_title("Returns Distribution")
            ax1.set_xlabel("Returns")
            ax1.set_ylabel("Frequency")
            ax1.grid(True)
            
            # Q-Q plot
            from scipy import stats
            stats.probplot(returns, dist="norm", plot=ax2)
            ax2.set_title("Q-Q Plot")
            ax2.grid(True)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = f"returns_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return plot_path
            
        except Exception as e:
            logger.error(f"Error creating returns plot: {e}")
            return None
    
    def _create_drawdown_plot(self, portfolio_data: pd.DataFrame) -> str:
        """Create drawdown plot"""
        
        try:
            if "portfolio_value" in portfolio_data.columns:
                values = portfolio_data["portfolio_value"]
            else:
                numeric_columns = portfolio_data.select_dtypes(include=[np.number]).columns
                if len(numeric_columns) > 0:
                    values = portfolio_data[numeric_columns[0]]
                else:
                    return None
            
            # Calculate drawdown
            peak = values.expanding().max()
            drawdown = (values - peak) / peak
            
            plt.figure(figsize=self.figure_size)
            plt.fill_between(portfolio_data.index, drawdown, 0, alpha=0.3, color='red')
            plt.plot(portfolio_data.index, drawdown, color='red', linewidth=1)
            plt.title("Portfolio Drawdown")
            plt.xlabel("Time")
            plt.ylabel("Drawdown")
            plt.grid(True)
            
            # Save plot
            plot_path = f"drawdown_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return plot_path
            
        except Exception as e:
            logger.error(f"Error creating drawdown plot: {e}")
            return None
    
    def _create_risk_return_plot(self, performance: Dict[str, float]) -> str:
        """Create risk-return scatter plot"""
        
        try:
            plt.figure(figsize=self.figure_size)
            
            # Extract risk and return metrics
            volatility = performance.get("volatility", 0)
            annualized_return = performance.get("annualized_return", 0)
            sharpe_ratio = performance.get("sharpe_ratio", 0)
            
            # Create scatter plot
            plt.scatter(volatility, annualized_return, s=100, alpha=0.7)
            plt.annotate(f"Sharpe: {sharpe_ratio:.2f}", 
                        (volatility, annualized_return),
                        xytext=(10, 10), textcoords='offset points')
            
            plt.title("Risk-Return Profile")
            plt.xlabel("Volatility")
            plt.ylabel("Annualized Return")
            plt.grid(True)
            
            # Save plot
            plot_path = f"risk_return_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return plot_path
            
        except Exception as e:
            logger.error(f"Error creating risk-return plot: {e}")
            return None


def evaluate_agents(agent_results: Dict[str, Dict[str, Any]], 
                   config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Evaluate multiple agents and compare performance"""
    
    try:
        evaluator = TradingEvaluator(config)
        
        # Evaluate each agent
        agent_evaluations = {}
        for agent_name, results in agent_results.items():
            if "portfolio_data" in results:
                evaluation = evaluator.evaluate_portfolio(results["portfolio_data"])
                agent_evaluations[agent_name] = evaluation
        
        # Create comparison
        comparison = create_agent_comparison(agent_evaluations)
        
        # Create visualizations
        visualizations = create_comparison_visualizations(agent_evaluations)
        
        return {
            "agent_evaluations": agent_evaluations,
            "comparison": comparison,
            "visualizations": visualizations
        }
        
    except Exception as e:
        logger.error(f"Error evaluating agents: {e}")
        return {}


def create_agent_comparison(agent_evaluations: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Create comparison table for multiple agents"""
    
    try:
        comparison_data = []
        
        for agent_name, evaluation in agent_evaluations.items():
            row = {"agent": agent_name}
            
            # Add performance metrics
            if "performance" in evaluation:
                for metric, value in evaluation["performance"].items():
                    row[f"perf_{metric}"] = value
            
            # Add risk metrics
            if "risk" in evaluation:
                for metric, value in evaluation["risk"].items():
                    row[f"risk_{metric}"] = value
            
            # Add drawdown metrics
            if "drawdown" in evaluation:
                for metric, value in evaluation["drawdown"].items():
                    row[f"dd_{metric}"] = value
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
        
    except Exception as e:
        logger.error(f"Error creating agent comparison: {e}")
        return pd.DataFrame()


def create_comparison_visualizations(agent_evaluations: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """Create comparison visualizations for multiple agents"""
    
    try:
        visualizations = {}
        
        # Create comparison plots
        comparison_plot = create_comparison_plot(agent_evaluations)
        if comparison_plot:
            visualizations["comparison_plot"] = comparison_plot
        
        # Create performance radar chart
        radar_plot = create_performance_radar(agent_evaluations)
        if radar_plot:
            visualizations["radar_plot"] = radar_plot
        
        return visualizations
        
    except Exception as e:
        logger.error(f"Error creating comparison visualizations: {e}")
        return {}


def create_comparison_plot(agent_evaluations: Dict[str, Dict[str, Any]]) -> str:
    """Create comparison plot for multiple agents"""
    
    try:
        # Extract metrics for comparison
        metrics = ["annualized_return", "volatility", "sharpe_ratio", "max_drawdown"]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            agents = []
            values = []
            
            for agent_name, evaluation in agent_evaluations.items():
                if "performance" in evaluation and metric in evaluation["performance"]:
                    agents.append(agent_name)
                    values.append(evaluation["performance"][metric])
            
            if agents and values:
                ax.bar(agents, values)
                ax.set_title(f"Agent Comparison - {metric.replace('_', ' ').title()}")
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, axis='y')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = f"agent_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
        
    except Exception as e:
        logger.error(f"Error creating comparison plot: {e}")
        return None


def create_performance_radar(agent_evaluations: Dict[str, Dict[str, Any]]) -> str:
    """Create performance radar chart for multiple agents"""
    
    try:
        # Extract metrics for radar chart
        metrics = ["annualized_return", "sharpe_ratio", "win_rate", "calmar_ratio"]
        
        # Normalize metrics to 0-1 scale
        normalized_data = {}
        for agent_name, evaluation in agent_evaluations.items():
            if "performance" in evaluation:
                values = []
                for metric in metrics:
                    value = evaluation["performance"].get(metric, 0)
                    # Normalize to 0-1 (simple min-max normalization)
                    if metric == "annualized_return":
                        value = max(0, min(1, (value + 0.5) / 1.0))  # Assume -50% to 50% range
                    elif metric == "sharpe_ratio":
                        value = max(0, min(1, (value + 2) / 4))  # Assume -2 to 2 range
                    elif metric == "win_rate":
                        value = value  # Already 0-1
                    elif metric == "calmar_ratio":
                        value = max(0, min(1, (value + 2) / 4))  # Assume -2 to 2 range
                    
                    values.append(value)
                
                normalized_data[agent_name] = values
        
        if not normalized_data:
            return None
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        for agent_name, values in normalized_data.items():
            values += values[:1]  # Complete the circle
            ax.plot(angles, values, 'o-', linewidth=2, label=agent_name)
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title("Agent Performance Radar Chart")
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        # Save plot
        plot_path = f"performance_radar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
        
    except Exception as e:
        logger.error(f"Error creating performance radar: {e}")
        return None


if __name__ == "__main__":
    # Test the evaluator
    import pandas as pd
    import numpy as np
    
    # Create sample data
    dates = pd.date_range("2023-01-01", periods=252, freq="D")
    portfolio_data = pd.DataFrame({
        "portfolio_value": 100000 * (1 + np.random.normal(0.001, 0.02, 252)).cumprod()
    }, index=dates)
    
    # Test evaluator
    evaluator = TradingEvaluator()
    results = evaluator.evaluate_portfolio(portfolio_data)
    
    print("Evaluation Results:")
    print(f"Annualized Return: {results['performance']['annualized_return']:.4f}")
    print(f"Volatility: {results['performance']['volatility']:.4f}")
    print(f"Sharpe Ratio: {results['performance']['sharpe_ratio']:.4f}")
    print(f"Max Drawdown: {results['drawdown']['max_drawdown']:.4f}")
    
    # Create report
    report = evaluator.create_performance_report(results)
    print("\nPerformance Report:")
    print(report)
