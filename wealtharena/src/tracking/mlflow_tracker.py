"""
MLflow Tracker for WealthArena Trading System

This module provides MLflow integration for experiment tracking, model versioning,
and artifact management in the WealthArena trading system.
"""

import mlflow
import mlflow.tracking
import mlflow.sklearn
import mlflow.pytorch
import mlflow.tensorflow
from mlflow.tracking import MlflowClient
from mlflow.entities import RunStatus
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME, MLFLOW_RUN_NOTE
import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime
import pickle

logger = logging.getLogger(__name__)


class MLflowTracker:
    """
    MLflow tracker for WealthArena trading experiments
    
    Provides comprehensive experiment tracking including metrics, parameters,
    artifacts, and model versioning for the trading system.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # MLflow configuration
        self.tracking_uri = config.get("tracking_uri", "http://localhost:5000")
        self.experiment_name = config.get("experiment_name", "WealthArena_Trading")
        self.artifact_location = config.get("artifact_location", "./mlruns")
        
        # Initialize MLflow
        self._setup_mlflow()
        
        # Current run tracking
        self.current_run = None
        self.run_id = None
        
        logger.info(f"MLflow tracker initialized: {self.tracking_uri}")
    
    def _setup_mlflow(self):
        """Setup MLflow tracking"""
        
        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Create or get experiment
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    name=self.experiment_name,
                    artifact_location=self.artifact_location
                )
                logger.info(f"Created new experiment: {self.experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {self.experiment_name}")
            
            self.experiment_id = experiment_id
            
        except Exception as e:
            logger.error(f"Error setting up MLflow experiment: {e}")
            self.experiment_id = None
    
    def start_run(self, run_name: str = None, tags: Dict[str, str] = None) -> str:
        """Start a new MLflow run"""
        
        if self.experiment_id is None:
            logger.error("No experiment available")
            return None
        
        try:
            # Start run
            run = mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=run_name,
                tags=tags
            )
            
            self.current_run = run
            self.run_id = run.info.run_id
            
            logger.info(f"Started MLflow run: {self.run_id}")
            return self.run_id
            
        except Exception as e:
            logger.error(f"Error starting MLflow run: {e}")
            return None
    
    def end_run(self, status: str = "FINISHED"):
        """End the current MLflow run"""
        
        if self.current_run is None:
            logger.warning("No active run to end")
            return
        
        try:
            # Set run status
            if status == "FAILED":
                mlflow.end_run(status=RunStatus.FAILED)
            else:
                mlflow.end_run(status=RunStatus.FINISHED)
            
            logger.info(f"Ended MLflow run: {self.run_id}")
            
            # Reset current run
            self.current_run = None
            self.run_id = None
            
        except Exception as e:
            logger.error(f"Error ending MLflow run: {e}")
    
    def log_parameters(self, params: Dict[str, Any]):
        """Log parameters to current run"""
        
        if self.current_run is None:
            logger.warning("No active run to log parameters")
            return
        
        try:
            # Convert parameters to appropriate types
            converted_params = {}
            for key, value in params.items():
                if isinstance(value, (int, float, str, bool)):
                    converted_params[key] = value
                else:
                    converted_params[key] = str(value)
            
            mlflow.log_params(converted_params)
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
                for key, value in metrics.items():
                    mlflow.log_metric(key, value, step=step)
            else:
                mlflow.log_metrics(metrics)
            
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
                        mlflow.log_artifact(path, name)
                    elif os.path.isdir(path):
                        mlflow.log_artifacts(path, name)
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
                mlflow.pytorch.log_model(model, model_name)
            elif model_type == "tensorflow":
                mlflow.tensorflow.log_model(model, model_name)
            elif model_type == "sklearn":
                mlflow.sklearn.log_model(model, model_name)
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
            
            logger.info("Logged trading results")
            
        except Exception as e:
            logger.error(f"Error logging trading results: {e}")
    
    def _log_portfolio_data(self, portfolio_data: pd.DataFrame):
        """Log portfolio data as artifact"""
        
        try:
            # Save portfolio data to CSV
            portfolio_file = "portfolio_data.csv"
            portfolio_data.to_csv(portfolio_file)
            
            # Log as artifact
            mlflow.log_artifact(portfolio_file)
            
            # Clean up
            os.remove(portfolio_file)
            
        except Exception as e:
            logger.error(f"Error logging portfolio data: {e}")
    
    def _log_trade_history(self, trade_history: List[Dict[str, Any]]):
        """Log trade history as artifact"""
        
        try:
            # Convert to DataFrame
            trade_df = pd.DataFrame(trade_history)
            
            # Save trade history to CSV
            trade_file = "trade_history.csv"
            trade_df.to_csv(trade_file, index=False)
            
            # Log as artifact
            mlflow.log_artifact(trade_file)
            
            # Clean up
            os.remove(trade_file)
            
        except Exception as e:
            logger.error(f"Error logging trade history: {e}")
    
    def log_environment_config(self, env_config: Dict[str, Any]):
        """Log environment configuration"""
        
        if self.current_run is None:
            logger.warning("No active run to log environment config")
            return
        
        try:
            # Save config to JSON
            config_file = "environment_config.json"
            with open(config_file, 'w') as f:
                json.dump(env_config, f, indent=2)
            
            # Log as artifact
            mlflow.log_artifact(config_file)
            
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
            # Save config to JSON
            config_file = "training_config.json"
            with open(config_file, 'w') as f:
                json.dump(training_config, f, indent=2)
            
            # Log as artifact
            mlflow.log_artifact(config_file)
            
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
                    mlflow.log_artifact(code_path, "code")
                else:
                    logger.warning(f"Code path does not exist: {code_path}")
            
            logger.info(f"Logged code snapshots from {len(code_paths)} paths")
            
        except Exception as e:
            logger.error(f"Error logging code snapshot: {e}")
    
    def get_run_info(self, run_id: str = None) -> Dict[str, Any]:
        """Get information about a specific run"""
        
        if run_id is None:
            run_id = self.run_id
        
        if run_id is None:
            logger.warning("No run ID provided")
            return {}
        
        try:
            client = MlflowClient()
            run = client.get_run(run_id)
            
            return {
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "parameters": run.data.params,
                "metrics": run.data.metrics,
                "tags": run.data.tags
            }
            
        except Exception as e:
            logger.error(f"Error getting run info: {e}")
            return {}
    
    def search_runs(self, 
                   experiment_id: str = None,
                   filter_string: str = None,
                   max_results: int = 100) -> List[Dict[str, Any]]:
        """Search runs in experiment"""
        
        try:
            client = MlflowClient()
            
            if experiment_id is None:
                experiment_id = self.experiment_id
            
            runs = client.search_runs(
                experiment_ids=[experiment_id],
                filter_string=filter_string,
                max_results=max_results
            )
            
            run_infos = []
            for run in runs:
                run_infos.append({
                    "run_id": run.info.run_id,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "parameters": run.data.params,
                    "metrics": run.data.metrics,
                    "tags": run.data.tags
                })
            
            return run_infos
            
        except Exception as e:
            logger.error(f"Error searching runs: {e}")
            return []
    
    def get_best_run(self, 
                    metric_name: str = "episode_reward_mean",
                    ascending: bool = False) -> Optional[Dict[str, Any]]:
        """Get the best run based on a metric"""
        
        try:
            runs = self.search_runs()
            
            if not runs:
                return None
            
            # Filter runs with the metric
            runs_with_metric = [
                run for run in runs 
                if metric_name in run["metrics"]
            ]
            
            if not runs_with_metric:
                return None
            
            # Sort by metric
            best_run = sorted(
                runs_with_metric,
                key=lambda x: x["metrics"][metric_name],
                reverse=not ascending
            )[0]
            
            return best_run
            
        except Exception as e:
            logger.error(f"Error getting best run: {e}")
            return None
    
    def create_experiment_comparison(self, 
                                   run_ids: List[str],
                                   metrics: List[str] = None) -> pd.DataFrame:
        """Create comparison table for multiple runs"""
        
        try:
            client = MlflowClient()
            
            comparison_data = []
            
            for run_id in run_ids:
                run = client.get_run(run_id)
                
                run_data = {
                    "run_id": run_id,
                    "status": run.info.status,
                    "start_time": run.info.start_time
                }
                
                # Add metrics
                if metrics:
                    for metric in metrics:
                        if metric in run.data.metrics:
                            run_data[metric] = run.data.metrics[metric]
                        else:
                            run_data[metric] = None
                
                comparison_data.append(run_data)
            
            return pd.DataFrame(comparison_data)
            
        except Exception as e:
            logger.error(f"Error creating experiment comparison: {e}")
            return pd.DataFrame()
    
    def export_run_data(self, run_id: str = None, output_dir: str = "./export") -> str:
        """Export all data from a run"""
        
        if run_id is None:
            run_id = self.run_id
        
        if run_id is None:
            logger.warning("No run ID provided")
            return None
        
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Get run info
            run_info = self.get_run_info(run_id)
            
            # Save run info
            with open(os.path.join(output_dir, "run_info.json"), 'w') as f:
                json.dump(run_info, f, indent=2, default=str)
            
            # Download artifacts
            client = MlflowClient()
            artifacts = client.list_artifacts(run_id)
            
            for artifact in artifacts:
                artifact_path = os.path.join(output_dir, artifact.path)
                os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
                
                # Download artifact
                client.download_artifacts(run_id, artifact.path, output_dir)
            
            logger.info(f"Exported run data to: {output_dir}")
            return output_dir
            
        except Exception as e:
            logger.error(f"Error exporting run data: {e}")
            return None


# Utility functions
def create_mlflow_callback(tracker: MLflowTracker):
    """Create MLflow callback for RLlib training"""
    
    from ray.air.integrations.mlflow import MLflowLoggerCallback
    
    return MLflowLoggerCallback(
        tracking_uri=tracker.tracking_uri,
        experiment_name=tracker.experiment_name,
        save_artifact=True
    )


def log_training_progress(tracker: MLflowTracker, 
                         iteration: int,
                         metrics: Dict[str, float],
                         config: Dict[str, Any] = None):
    """Log training progress to MLflow"""
    
    if tracker.current_run is None:
        return
    
    # Log metrics
    tracker.log_metrics(metrics, step=iteration)
    
    # Log config if provided
    if config:
        tracker.log_parameters(config)


if __name__ == "__main__":
    # Test the MLflow tracker
    config = {
        "tracking_uri": "http://localhost:5000",
        "experiment_name": "WealthArena_Test",
        "artifact_location": "./mlruns"
    }
    
    tracker = MLflowTracker(config)
    
    # Start a run
    run_id = tracker.start_run("test_run")
    
    if run_id:
        # Log some test data
        tracker.log_parameters({
            "learning_rate": 0.001,
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
