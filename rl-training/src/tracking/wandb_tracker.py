"""
Weights & Biases Tracker for WealthArena Trading System

This module provides Weights & Biases integration for experiment tracking,
model versioning, and visualization in the WealthArena trading system.
"""

import wandb
import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class WandbTracker:
    """
    Weights & Biases tracker for WealthArena trading experiments
    
    Provides comprehensive experiment tracking including metrics, parameters,
    artifacts, and visualization for the trading system.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # W&B configuration
        self.project_name = config.get("project_name", "wealtharena-trading")
        self.entity = config.get("entity", None)
        self.tags = config.get("tags", [])
        self.notes = config.get("notes", "")
        
        # Initialize W&B
        self._setup_wandb()
        
        # Current run tracking
        self.current_run = None
        self.run_id = None
        
        logger.info(f"W&B tracker initialized: {self.project_name}")
    
    def _setup_wandb(self):
        """Setup Weights & Biases"""
        
        try:
            # Initialize W&B
            wandb.init(
                project=self.project_name,
                entity=self.entity,
                tags=self.tags,
                notes=self.notes,
                mode="disabled"  # Will be enabled when start_run is called
            )
            
            # Close the initial run
            wandb.finish()
            
        except Exception as e:
            logger.error(f"Error setting up W&B: {e}")
    
    def start_run(self, 
                  run_name: str = None,
                  config: Dict[str, Any] = None,
                  tags: List[str] = None,
                  notes: str = None) -> str:
        """Start a new W&B run"""
        
        try:
            # Prepare run configuration
            run_config = config or {}
            run_tags = tags or self.tags
            run_notes = notes or self.notes
            
            # Start run
            wandb.init(
                project=self.project_name,
                entity=self.entity,
                name=run_name,
                config=run_config,
                tags=run_tags,
                notes=run_notes,
                reinit=True
            )
            
            self.current_run = wandb.run
            self.run_id = wandb.run.id
            
            logger.info(f"Started W&B run: {self.run_id}")
            return self.run_id
            
        except Exception as e:
            logger.error(f"Error starting W&B run: {e}")
            return None
    
    def end_run(self):
        """End the current W&B run"""
        
        if self.current_run is None:
            logger.warning("No active run to end")
            return
        
        try:
            wandb.finish()
            
            logger.info(f"Ended W&B run: {self.run_id}")
            
            # Reset current run
            self.current_run = None
            self.run_id = None
            
        except Exception as e:
            logger.error(f"Error ending W&B run: {e}")
    
    def log_parameters(self, params: Dict[str, Any]):
        """Log parameters to current run"""
        
        if self.current_run is None:
            logger.warning("No active run to log parameters")
            return
        
        try:
            # Convert parameters to appropriate types
            converted_params = {}
            for key, value in params.items():
                if isinstance(value, (int, float, str, bool, list, dict)):
                    converted_params[key] = value
                else:
                    converted_params[key] = str(value)
            
            wandb.config.update(converted_params)
            logger.debug(f"Logged {len(converted_params)} parameters")
            
        except Exception as e:
            logger.error(f"Error logging parameters: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics to current run"""
        
        if self.current_run is None:
            logger.warning("No active run to log metrics")
            return
        
        try:
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)
            
            logger.debug(f"Logged {len(metrics)} metrics")
            
        except Exception as e:
            logger.error(f"Error logging metrics: {e}")
    
    def log_artifacts(self, artifacts: Dict[str, str]):
        """Log artifacts to current run"""
        
        if self.current_run is None:
            logger.warning("No active run to log artifacts")
            return
        
        try:
            for name, path in artifacts.items():
                if os.path.exists(path):
                    if os.path.isfile(path):
                        wandb.log_artifact(path, name=name)
                    elif os.path.isdir(path):
                        wandb.log_artifact(path, name=name)
                    else:
                        logger.warning(f"Path does not exist: {path}")
                else:
                    logger.warning(f"Path does not exist: {path}")
            
            logger.debug(f"Logged {len(artifacts)} artifacts")
            
        except Exception as e:
            logger.error(f"Error logging artifacts: {e}")
    
    def log_model(self, model, model_name: str, model_type: str = "pytorch"):
        """Log model to current run"""
        
        if self.current_run is None:
            logger.warning("No active run to log model")
            return
        
        try:
            if model_type == "pytorch":
                wandb.watch(model, log="all", log_freq=100)
            elif model_type == "tensorflow":
                wandb.watch(model, log="all", log_freq=100)
            else:
                logger.warning(f"Unknown model type: {model_type}")
                return
            
            logger.info(f"Logged {model_type} model: {model_name}")
            
        except Exception as e:
            logger.error(f"Error logging model: {e}")
    
    def log_trading_results(self, results: Dict[str, Any]):
        """Log trading results and performance metrics"""
        
        if self.current_run is None:
            logger.warning("No active run to log trading results")
            return
        
        try:
            # Log performance metrics
            performance_metrics = results.get("performance", {})
            if performance_metrics:
                self.log_metrics(performance_metrics)
            
            # Log trading statistics
            trading_stats = results.get("trading_stats", {})
            if trading_stats:
                self.log_metrics(trading_stats)
            
            # Log risk metrics
            risk_metrics = results.get("risk_metrics", {})
            if risk_metrics:
                self.log_metrics(risk_metrics)
            
            # Log portfolio data
            portfolio_data = results.get("portfolio_data")
            if portfolio_data is not None:
                self._log_portfolio_data(portfolio_data)
            
            # Log trade history
            trade_history = results.get("trade_history")
            if trade_history is not None:
                self._log_trade_history(trade_history)
            
            # Log visualizations
            self._log_trading_visualizations(results)
            
            logger.info("Logged trading results")
            
        except Exception as e:
            logger.error(f"Error logging trading results: {e}")
    
    def _log_portfolio_data(self, portfolio_data: pd.DataFrame):
        """Log portfolio data as table and visualization"""
        
        try:
            # Log as table
            portfolio_table = wandb.Table(
                data=portfolio_data.values.tolist(),
                columns=portfolio_data.columns.tolist()
            )
            wandb.log({"portfolio_data": portfolio_table})
            
            # Create portfolio value plot
            if "portfolio_value" in portfolio_data.columns:
                plt.figure(figsize=(12, 6))
                plt.plot(portfolio_data.index, portfolio_data["portfolio_value"])
                plt.title("Portfolio Value Over Time")
                plt.xlabel("Time")
                plt.ylabel("Portfolio Value")
                plt.grid(True)
                
                wandb.log({"portfolio_value_plot": wandb.Image(plt)})
                plt.close()
            
        except Exception as e:
            logger.error(f"Error logging portfolio data: {e}")
    
    def _log_trade_history(self, trade_history: List[Dict[str, Any]]):
        """Log trade history as table"""
        
        try:
            # Convert to DataFrame
            trade_df = pd.DataFrame(trade_history)
            
            # Log as table
            trade_table = wandb.Table(
                data=trade_df.values.tolist(),
                columns=trade_df.columns.tolist()
            )
            wandb.log({"trade_history": trade_table})
            
        except Exception as e:
            logger.error(f"Error logging trade history: {e}")
    
    def _log_trading_visualizations(self, results: Dict[str, Any]):
        """Log trading visualizations"""
        
        try:
            # Portfolio value plot
            if "portfolio_data" in results:
                portfolio_data = results["portfolio_data"]
                if isinstance(portfolio_data, pd.DataFrame) and "portfolio_value" in portfolio_data.columns:
                    self._create_portfolio_plot(portfolio_data)
            
            # Returns distribution
            if "returns" in results:
                returns = results["returns"]
                if isinstance(returns, (list, np.ndarray)):
                    self._create_returns_plot(returns)
            
            # Risk metrics
            if "risk_metrics" in results:
                risk_metrics = results["risk_metrics"]
                self._create_risk_plots(risk_metrics)
            
        except Exception as e:
            logger.error(f"Error logging trading visualizations: {e}")
    
    def _create_portfolio_plot(self, portfolio_data: pd.DataFrame):
        """Create portfolio value plot"""
        
        try:
            plt.figure(figsize=(12, 8))
            
            # Portfolio value
            plt.subplot(2, 1, 1)
            plt.plot(portfolio_data.index, portfolio_data["portfolio_value"])
            plt.title("Portfolio Value Over Time")
            plt.xlabel("Time")
            plt.ylabel("Portfolio Value")
            plt.grid(True)
            
            # Returns
            if "returns" in portfolio_data.columns:
                plt.subplot(2, 1, 2)
                plt.plot(portfolio_data.index, portfolio_data["returns"])
                plt.title("Portfolio Returns")
                plt.xlabel("Time")
                plt.ylabel("Returns")
                plt.grid(True)
            
            plt.tight_layout()
            wandb.log({"portfolio_analysis": wandb.Image(plt)})
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating portfolio plot: {e}")
    
    def _create_returns_plot(self, returns: Union[List, np.ndarray]):
        """Create returns distribution plot"""
        
        try:
            plt.figure(figsize=(12, 6))
            
            # Returns histogram
            plt.subplot(1, 2, 1)
            plt.hist(returns, bins=50, alpha=0.7, edgecolor='black')
            plt.title("Returns Distribution")
            plt.xlabel("Returns")
            plt.ylabel("Frequency")
            plt.grid(True)
            
            # Q-Q plot
            plt.subplot(1, 2, 2)
            from scipy import stats
            stats.probplot(returns, dist="norm", plot=plt)
            plt.title("Q-Q Plot")
            plt.grid(True)
            
            plt.tight_layout()
            wandb.log({"returns_analysis": wandb.Image(plt)})
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating returns plot: {e}")
    
    def _create_risk_plots(self, risk_metrics: Dict[str, Any]):
        """Create risk metrics plots"""
        
        try:
            plt.figure(figsize=(12, 8))
            
            # Risk metrics bar chart
            metrics_names = list(risk_metrics.keys())
            metrics_values = list(risk_metrics.values())
            
            plt.bar(metrics_names, metrics_values)
            plt.title("Risk Metrics")
            plt.xlabel("Metric")
            plt.ylabel("Value")
            plt.xticks(rotation=45)
            plt.grid(True, axis='y')
            
            plt.tight_layout()
            wandb.log({"risk_metrics": wandb.Image(plt)})
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating risk plots: {e}")
    
    def log_environment_config(self, env_config: Dict[str, Any]):
        """Log environment configuration"""
        
        if self.current_run is None:
            logger.warning("No active run to log environment config")
            return
        
        try:
            # Log as config
            wandb.config.update({"environment": env_config})
            
            # Log as artifact
            config_file = "environment_config.json"
            with open(config_file, 'w') as f:
                json.dump(env_config, f, indent=2)
            
            wandb.log_artifact(config_file, name="environment_config")
            
            # Clean up
            os.remove(config_file)
            
        except Exception as e:
            logger.error(f"Error logging environment config: {e}")
    
    def log_training_config(self, training_config: Dict[str, Any]):
        """Log training configuration"""
        
        if self.current_run is None:
            logger.warning("No active run to log training config")
            return
        
        try:
            # Log as config
            wandb.config.update({"training": training_config})
            
            # Log as artifact
            config_file = "training_config.json"
            with open(config_file, 'w') as f:
                json.dump(training_config, f, indent=2)
            
            wandb.log_artifact(config_file, name="training_config")
            
            # Clean up
            os.remove(config_file)
            
        except Exception as e:
            logger.error(f"Error logging training config: {e}")
    
    def log_code_snapshot(self, code_paths: List[str]):
        """Log code snapshots for reproducibility"""
        
        if self.current_run is None:
            logger.warning("No active run to log code snapshot")
            return
        
        try:
            for code_path in code_paths:
                if os.path.exists(code_path):
                    wandb.log_artifact(code_path, name="code")
                else:
                    logger.warning(f"Code path does not exist: {code_path}")
            
            logger.info(f"Logged code snapshots from {len(code_paths)} paths")
            
        except Exception as e:
            logger.error(f"Error logging code snapshot: {e}")
    
    def log_hyperparameter_sweep(self, sweep_config: Dict[str, Any]):
        """Log hyperparameter sweep configuration"""
        
        if self.current_run is None:
            logger.warning("No active run to log hyperparameter sweep")
            return
        
        try:
            # Log sweep config
            wandb.config.update({"sweep": sweep_config})
            
            # Log as artifact
            config_file = "sweep_config.json"
            with open(config_file, 'w') as f:
                json.dump(sweep_config, f, indent=2)
            
            wandb.log_artifact(config_file, name="sweep_config")
            
            # Clean up
            os.remove(config_file)
            
        except Exception as e:
            logger.error(f"Error logging hyperparameter sweep: {e}")
    
    def log_model_performance(self, 
                            model_name: str,
                            performance_metrics: Dict[str, float],
                            confusion_matrix: np.ndarray = None,
                            feature_importance: Dict[str, float] = None):
        """Log model performance metrics and visualizations"""
        
        if self.current_run is None:
            logger.warning("No active run to log model performance")
            return
        
        try:
            # Log performance metrics
            wandb.log({f"{model_name}_performance": performance_metrics})
            
            # Log confusion matrix
            if confusion_matrix is not None:
                plt.figure(figsize=(8, 6))
                sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
                plt.title(f"{model_name} Confusion Matrix")
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                
                wandb.log({f"{model_name}_confusion_matrix": wandb.Image(plt)})
                plt.close()
            
            # Log feature importance
            if feature_importance is not None:
                plt.figure(figsize=(10, 6))
                features = list(feature_importance.keys())
                importance = list(feature_importance.values())
                
                plt.barh(features, importance)
                plt.title(f"{model_name} Feature Importance")
                plt.xlabel("Importance")
                
                wandb.log({f"{model_name}_feature_importance": wandb.Image(plt)})
                plt.close()
            
        except Exception as e:
            logger.error(f"Error logging model performance: {e}")
    
    def log_agent_comparison(self, agent_results: Dict[str, Dict[str, Any]]):
        """Log comparison between different agents"""
        
        if self.current_run is None:
            logger.warning("No active run to log agent comparison")
            return
        
        try:
            # Create comparison table
            comparison_data = []
            for agent_name, results in agent_results.items():
                row = {"agent": agent_name}
                row.update(results)
                comparison_data.append(row)
            
            comparison_table = wandb.Table(
                data=comparison_data,
                columns=list(comparison_data[0].keys()) if comparison_data else []
            )
            
            wandb.log({"agent_comparison": comparison_table})
            
            # Create comparison plots
            if len(agent_results) > 1:
                self._create_agent_comparison_plots(agent_results)
            
        except Exception as e:
            logger.error(f"Error logging agent comparison: {e}")
    
    def _create_agent_comparison_plots(self, agent_results: Dict[str, Dict[str, Any]]):
        """Create comparison plots for different agents"""
        
        try:
            # Extract common metrics
            common_metrics = set()
            for results in agent_results.values():
                common_metrics.update(results.keys())
            
            # Create plots for each metric
            for metric in common_metrics:
                if all(metric in results for results in agent_results.values()):
                    plt.figure(figsize=(10, 6))
                    
                    agents = list(agent_results.keys())
                    values = [agent_results[agent][metric] for agent in agents]
                    
                    plt.bar(agents, values)
                    plt.title(f"Agent Comparison - {metric}")
                    plt.xlabel("Agent")
                    plt.ylabel(metric)
                    plt.xticks(rotation=45)
                    plt.grid(True, axis='y')
                    
                    wandb.log({f"agent_comparison_{metric}": wandb.Image(plt)})
                    plt.close()
            
        except Exception as e:
            logger.error(f"Error creating agent comparison plots: {e}")
    
    def get_run_url(self) -> str:
        """Get the URL for the current run"""
        
        if self.current_run is None:
            return None
        
        try:
            return self.current_run.url
        except Exception as e:
            logger.error(f"Error getting run URL: {e}")
            return None


# Utility functions
def create_wandb_callback(tracker: WandbTracker):
    """Create W&B callback for RLlib training"""
    
    from ray.air.integrations.wandb import WandbLoggerCallback
    
    return WandbLoggerCallback(
        project=tracker.project_name,
        entity=tracker.entity,
        log_config=True,
        save_checkpoints=True
    )


def log_training_progress(tracker: WandbTracker, 
                         iteration: int,
                         metrics: Dict[str, float],
                         config: Dict[str, Any] = None):
    """Log training progress to W&B"""
    
    if tracker.current_run is None:
        return
    
    # Log metrics
    tracker.log_metrics(metrics, step=iteration)
    
    # Log config if provided
    if config:
        tracker.log_parameters(config)


if __name__ == "__main__":
    # Test the W&B tracker
    config = {
        "project_name": "wealtharena-test",
        "entity": None,
        "tags": ["test", "trading"],
        "notes": "Test run for WealthArena trading system"
    }
    
    tracker = WandbTracker(config)
    
    # Start a run
    run_id = tracker.start_run("test_run", {"learning_rate": 0.001})
    
    if run_id:
        # Log some test data
        tracker.log_parameters({
            "batch_size": 32,
            "num_agents": 3
        })
        
        tracker.log_metrics({
            "episode_reward_mean": 100.0,
            "episode_length_mean": 1000.0
        })
        
        # End the run
        tracker.end_run()
        
        print(f"Test run completed: {run_id}")
    else:
        print("Failed to start test run")
