#!/usr/bin/env python3
"""
WealthArena Trading System - Advanced Agent Training

This is the main training script for the WealthArena multi-agent trading system.
It implements advanced RL algorithms optimized for profit generation and risk management.
"""

import os
import sys
import yaml
import argparse
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Ray and RLlib imports
import ray
from ray import tune, air
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.a2c import A2CConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.policy import Policy
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.air.integrations.mlflow import MLflowLoggerCallback

# Custom imports
from environments.trading_env import WealthArenaTradingEnv
from environments.multi_agent_env import WealthArenaMultiAgentEnv
from models.portfolio_manager import Portfolio, PortfolioManager
from data.data_adapter import DataAdapter
from tracking.mlflow_tracker import MLflowTracker
from tracking.wandb_tracker import WandbTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AdvancedTradingTrainer:
    """
    Advanced Trading Trainer for WealthArena
    
    Implements sophisticated RL training with focus on:
    - Profit maximization
    - Risk management
    - Market adaptation
    - Multi-agent coordination
    """
    
    def __init__(self, config_path: str = "config/production_config.yaml"):
        """Initialize advanced trainer"""
        self.config_path = config_path
        self.config = self._load_config()
        self.setup_directories()
        
        # Performance tracking
        self.performance_metrics = {
            "best_sharpe_ratio": -np.inf,
            "best_profit": -np.inf,
            "best_win_rate": 0.0,
            "best_max_drawdown": np.inf,
            "training_history": []
        }
        
        logger.info("AdvancedTradingTrainer initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load training configuration"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default production config
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default production configuration"""
        return {
            "environment": {
                "num_agents": 5,
                "num_assets": 20,
                "episode_length": 252,
                "initial_cash_per_agent": 1000000,
                "lookback_window_size": 30,
                "transaction_cost_rate": 0.0005,
                "slippage_rate": 0.0002,
                "reward_weights": {
                    "profit": 2.0,
                    "risk": 0.5,
                    "cost": 0.1,
                    "stability": 0.05,
                    "sharpe": 1.0
                }
            },
            "data": {
                "source": "yfinance",
                "symbols": [
                    "AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META", "NFLX", 
                    "AMD", "INTC", "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA", "DIS", 
                    "PYPL", "ADBE"
                ],
                "start_date": "2020-01-01",
                "end_date": "2024-12-31",
                "interval": "1d",
                "cache_enabled": True
            },
            "training": {
                "algorithm": "PPO",
                "learning_rate": 1e-4,
                "gamma": 0.995,
                "gae_lambda": 0.95,
                "entropy_coeff": 0.01,
                "vf_loss_coeff": 0.5,
                "clip_param": 0.2,
                "num_sgd_iter": 20,
                "sgd_minibatch_size": 256,
                "train_batch_size": 8000,
                "max_iterations": 2000,
                "target_reward": 500.0,
                "early_stopping": True,
                "patience": 100
            },
            "risk_management": {
                "max_position_size": 0.15,
                "max_portfolio_risk": 0.12,
                "stop_loss_threshold": 0.08,
                "take_profit_threshold": 0.20,
                "max_drawdown_limit": 0.15,
                "var_confidence": 0.95,
                "correlation_limit": 0.7
            },
            "resources": {
                "num_workers": 4,
                "num_envs_per_worker": 2,
                "num_cpus_per_worker": 2,
                "num_gpus": 0
            },
            "evaluation": {
                "eval_interval": 25,
                "eval_duration": 20,
                "eval_episodes": 50,
                "benchmark_symbols": ["SPY", "QQQ", "IWM"]
            },
            "checkpointing": {
                "checkpoint_freq": 25,
                "keep_checkpoints_num": 10,
                "save_best_only": True
            },
            "experiment_tracking": {
                "wandb": {
                    "enabled": True,
                    "project": "wealtharena-trading",
                    "entity": "wealtharena"
                },
                "mlflow": {
                    "enabled": True,
                    "tracking_uri": "http://localhost:5000",
                    "experiment_name": "WealthArena_Production"
                }
            }
        }
    
    def setup_directories(self):
        """Create necessary directories"""
        directories = ["logs", "checkpoints", "results", "artifacts", "models"]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def prepare_market_data(self) -> bool:
        """Prepare comprehensive market data"""
        logger.info("Preparing market data...")
        
        try:
            # Download data if not exists
            data_dir = Path("data/processed")
            if not data_dir.exists() or len(list(data_dir.glob("*.csv"))) == 0:
                logger.info("Downloading market data...")
                from download_market_data import main as download_main
                download_main()
            
            # Validate data
            symbols = self.config["data"]["symbols"]
            missing_data = []
            
            for symbol in symbols:
                data_file = data_dir / f"{symbol}_processed.csv"
                if not data_file.exists():
                    missing_data.append(symbol)
            
            if missing_data:
                logger.warning(f"Missing data for symbols: {missing_data}")
                return False
            
            logger.info(f"Market data prepared for {len(symbols)} symbols")
            return True
            
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            return False
    
    def create_optimized_environment(self) -> MultiAgentEnv:
        """Create optimized trading environment"""
        logger.info("Creating optimized trading environment...")
        
        env_config = {
            "num_agents": self.config["environment"]["num_agents"],
            "num_assets": len(self.config["data"]["symbols"]),
            "episode_length": self.config["environment"]["episode_length"],
            "initial_cash_per_agent": self.config["environment"]["initial_cash_per_agent"],
            "lookback_window_size": self.config["environment"]["lookback_window_size"],
            "transaction_cost_rate": self.config["environment"]["transaction_cost_rate"],
            "slippage_rate": self.config["environment"]["slippage_rate"],
            "use_real_data": True,
            "data_path": "data/processed/",
            "symbols": self.config["data"]["symbols"],
            "coordination_enabled": True,
            "reward_weights": self.config["environment"]["reward_weights"],
            "risk_management": self.config["risk_management"]
        }
        
        env = WealthArenaMultiAgentEnv(env_config)
        logger.info(f"Environment created: {env.num_agents} agents, {env.num_assets} assets")
        return env
    
    def create_advanced_algorithm_config(self, env: MultiAgentEnv) -> Dict[str, Any]:
        """Create advanced algorithm configuration"""
        algorithm_name = self.config["training"]["algorithm"]
        
        if algorithm_name == "PPO":
            algo_config = PPOConfig()
        elif algorithm_name == "A2C":
            algo_config = A2CConfig()
        elif algorithm_name == "SAC":
            algo_config = SACConfig()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm_name}")
        
        # Environment setup
        algo_config.environment(
            env=WealthArenaMultiAgentEnv,
            env_config=self.config["environment"],
            disable_env_checking=True
        )
        
        # Multi-agent configuration with different agent types
        algo_config.multi_agent(
            policies={
                "conservative_policy": Policy.from_checkpoint(
                    checkpoint_path=None,
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    config={
                        "model": {
                            "custom_model": "trading_lstm",
                            "custom_model_config": {
                                "num_assets": env.num_assets,
                                "lookback_window_size": self.config["environment"]["lookback_window_size"],
                                "hidden_size": 256,
                                "num_layers": 3,
                                "agent_type": "conservative"
                            }
                        }
                    }
                ),
                "aggressive_policy": Policy.from_checkpoint(
                    checkpoint_path=None,
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    config={
                        "model": {
                            "custom_model": "trading_lstm",
                            "custom_model_config": {
                                "num_assets": env.num_assets,
                                "lookback_window_size": self.config["environment"]["lookback_window_size"],
                                "hidden_size": 256,
                                "num_layers": 3,
                                "agent_type": "aggressive"
                            }
                        }
                    }
                ),
                "balanced_policy": Policy.from_checkpoint(
                    checkpoint_path=None,
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    config={
                        "model": {
                            "custom_model": "trading_lstm",
                            "custom_model_config": {
                                "num_assets": env.num_assets,
                                "lookback_window_size": self.config["environment"]["lookback_window_size"],
                                "hidden_size": 256,
                                "num_layers": 3,
                                "agent_type": "balanced"
                            }
                        }
                    }
                )
            },
            policy_mapping_fn=self._policy_mapping_fn,
            policies_to_train=["conservative_policy", "aggressive_policy", "balanced_policy"]
        )
        
        # Advanced training hyperparameters
        training_config = self.config["training"]
        algo_config.training(
            lr=training_config["learning_rate"],
            gamma=training_config["gamma"],
            gae_lambda=training_config["gae_lambda"],
            entropy_coeff=training_config["entropy_coeff"],
            vf_loss_coeff=training_config["vf_loss_coeff"],
            clip_param=training_config["clip_param"],
            num_sgd_iter=training_config["num_sgd_iter"],
            sgd_minibatch_size=training_config["sgd_minibatch_size"],
            train_batch_size=training_config["train_batch_size"],
            # Advanced PPO settings
            kl_coeff=0.2,
            kl_target=0.01,
            use_gae=True,
            use_critic=True,
            use_huber=False,
            huber_threshold=1.0,
            # Learning rate scheduling
            lr_schedule=[[0, training_config["learning_rate"]], 
                        [training_config["max_iterations"], training_config["learning_rate"] * 0.1]]
        )
        
        # Resource configuration
        resources_config = self.config["resources"]
        algo_config.resources(
            num_workers=resources_config["num_workers"],
            num_envs_per_worker=resources_config["num_envs_per_worker"],
            num_cpus_per_worker=resources_config["num_cpus_per_worker"],
            num_gpus=resources_config["num_gpus"],
        )
        
        # Framework
        algo_config.framework("torch")
        
        # Advanced evaluation
        evaluation_config = self.config["evaluation"]
        algo_config.evaluation(
            evaluation_interval=evaluation_config["eval_interval"],
            evaluation_duration=evaluation_config["eval_duration"],
            evaluation_duration_unit="episodes",
            evaluation_num_workers=2,
            evaluation_config={
                "explore": False,
                "num_envs_per_worker": 1,
            }
        )
        
        # Checkpointing
        checkpoint_config = self.config["checkpointing"]
        algo_config.checkpointing(
            checkpoint_freq=checkpoint_config["checkpoint_freq"],
            num_to_keep=checkpoint_config["keep_checkpoints_num"],
            checkpoint_at_end=True
        )
        
        # Logging
        algo_config.debugging(log_level="INFO")
        
        return algo_config.to_dict()
    
    def _policy_mapping_fn(self, agent_id, episode, worker, **kwargs):
        """Map agents to different policies based on their ID"""
        agent_id_int = int(agent_id.split("_")[1])
        
        if agent_id_int % 3 == 0:
            return "conservative_policy"
        elif agent_id_int % 3 == 1:
            return "aggressive_policy"
        else:
            return "balanced_policy"
    
    def setup_experiment_tracking(self) -> List[Any]:
        """Setup advanced experiment tracking"""
        callbacks = []
        
        # Weights & Biases
        if self.config.get("experiment_tracking", {}).get("wandb", {}).get("enabled", False):
            wandb_config = self.config["experiment_tracking"]["wandb"]
            wandb_tracker = WandbTracker(
                project_name=wandb_config.get("project", "wealtharena-trading"),
                entity=wandb_config.get("entity", "wealtharena"),
                config=self.config
            )
            callbacks.append(wandb_tracker.get_rllib_callback())
            logger.info("W&B tracking enabled")
        
        # MLflow
        if self.config.get("experiment_tracking", {}).get("mlflow", {}).get("enabled", False):
            mlflow_config = self.config["experiment_tracking"]["mlflow"]
            mlflow_tracker = MLflowTracker(
                tracking_uri=mlflow_config.get("tracking_uri", "http://localhost:5000"),
                experiment_name=mlflow_config.get("experiment_name", "WealthArena_Production")
            )
            callbacks.append(mlflow_tracker.get_rllib_callback())
            logger.info("MLflow tracking enabled")
        
        return callbacks
    
    def train(self, experiment_name: str = "wealtharena_advanced") -> Dict[str, Any]:
        """Run advanced training"""
        logger.info(f"Starting advanced training: {experiment_name}")
        
        try:
            # Prepare data
            if not self.prepare_market_data():
                raise RuntimeError("Data preparation failed")
            
            # Create environment
            env = self.create_optimized_environment()
            
            # Create algorithm config
            algo_config = self.create_advanced_algorithm_config(env)
            
            # Setup tracking
            callbacks = self.setup_experiment_tracking()
            
            # Advanced stop criteria
            stop_criteria = {
                "training_iteration": self.config["training"]["max_iterations"],
                "episode_reward_mean": self.config["training"]["target_reward"],
                "custom_metrics": {
                    "sharpe_ratio": 2.0,  # Target Sharpe ratio
                    "max_drawdown": 0.1,  # Max 10% drawdown
                    "win_rate": 0.6       # 60% win rate
                }
            }
            
            # Create tuner with advanced configuration
            tuner = tune.Tuner(
                self.config["training"]["algorithm"],
                param_space=algo_config,
                run_config=air.RunConfig(
                    name=experiment_name,
                    stop=stop_criteria,
                    callbacks=callbacks,
                    checkpoint_config=air.CheckpointConfig(
                        checkpoint_frequency=self.config["checkpointing"]["checkpoint_freq"],
                        num_to_keep=self.config["checkpointing"]["keep_checkpoints_num"],
                        checkpoint_at_end=True
                    ),
                    failure_config=air.FailureConfig(max_failures=5),
                    verbose=1,
                    progress_reporter=tune.CLIReporter(
                        metric_columns={
                            "training_iteration": "iter",
                            "episode_reward_mean": "reward_mean",
                            "episode_len_mean": "len_mean",
                            "custom_metrics/sharpe_ratio": "sharpe",
                            "custom_metrics/max_drawdown": "drawdown",
                            "custom_metrics/win_rate": "win_rate",
                            "time_total_s": "time_total_s"
                        },
                        max_progress_rows=15,
                        print_intermediate_tables=True
                    )
                )
            )
            
            # Run training
            logger.info("Starting advanced training run...")
            results = tuner.fit()
            
            # Analyze results
            best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
            
            if best_result:
                logger.info("🎉 Advanced training completed successfully!")
                logger.info(f"Best result: {best_result.metrics}")
                logger.info(f"Best checkpoint: {best_result.checkpoint}")
                
                # Save performance metrics
                self._save_performance_metrics(best_result)
                
                return {
                    "success": True,
                    "best_result": best_result.metrics,
                    "checkpoint": best_result.checkpoint,
                    "experiment_name": experiment_name
                }
            else:
                logger.error("No best result found")
                return {"success": False, "error": "No best result found"}
                
        except Exception as e:
            logger.error(f"Advanced training failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _save_performance_metrics(self, result):
        """Save performance metrics for analysis"""
        metrics = result.metrics
        
        # Update best metrics
        if "custom_metrics/sharpe_ratio" in metrics:
            self.performance_metrics["best_sharpe_ratio"] = max(
                self.performance_metrics["best_sharpe_ratio"],
                metrics["custom_metrics/sharpe_ratio"]
            )
        
        if "episode_reward_mean" in metrics:
            self.performance_metrics["best_profit"] = max(
                self.performance_metrics["best_profit"],
                metrics["episode_reward_mean"]
            )
        
        # Save to file
        import json
        with open("results/performance_metrics.json", "w") as f:
            json.dump(self.performance_metrics, f, indent=2, default=str)
    
    def evaluate_performance(self, checkpoint_path: str) -> Dict[str, Any]:
        """Evaluate model performance against benchmarks"""
        logger.info(f"Evaluating performance: {checkpoint_path}")
        
        try:
            # Load algorithm
            algorithm_name = self.config["training"]["algorithm"]
            if algorithm_name == "PPO":
                from ray.rllib.algorithms.ppo import PPO
                algorithm = PPO.from_checkpoint(checkpoint_path)
            elif algorithm_name == "A2C":
                from ray.rllib.algorithms.a2c import A2C
                algorithm = A2C.from_checkpoint(checkpoint_path)
            elif algorithm_name == "SAC":
                from ray.rllib.algorithms.sac import SAC
                algorithm = SAC.from_checkpoint(checkpoint_path)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm_name}")
            
            # Create environment
            env = self.create_optimized_environment()
            
            # Run evaluation
            evaluation_results = self._run_comprehensive_evaluation(algorithm, env)
            
            # Compare with benchmarks
            benchmark_comparison = self._compare_with_benchmarks(evaluation_results)
            
            # Generate performance report
            performance_report = {
                "evaluation_results": evaluation_results,
                "benchmark_comparison": benchmark_comparison,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save report
            import json
            with open("results/performance_evaluation.json", "w") as f:
                json.dump(performance_report, f, indent=2, default=str)
            
            logger.info("Performance evaluation completed")
            return performance_report
            
        except Exception as e:
            logger.error(f"Performance evaluation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _run_comprehensive_evaluation(self, algorithm, env) -> Dict[str, Any]:
        """Run comprehensive evaluation"""
        results = []
        
        for episode in range(50):  # Evaluate 50 episodes
            obs, info = env.reset()
            episode_reward = 0
            episode_trades = 0
            episode_returns = []
            done = False
            
            while not done:
                actions = {}
                for agent_id in env.agent_ids:
                    action = algorithm.compute_single_action(obs[agent_id])
                    actions[agent_id] = action
                
                obs, rewards, terminateds, truncateds, infos = env.step(actions)
                episode_reward += sum(rewards.values())
                episode_trades += sum(1 for info in infos.values() if info.get("num_trades_in_step", 0) > 0)
                
                # Calculate returns
                for info in infos.values():
                    if "portfolio_value" in info:
                        episode_returns.append(info["portfolio_value"])
                
                done = terminateds.get("__all__", False)
            
            # Calculate episode metrics
            if episode_returns:
                returns = np.array(episode_returns)
                episode_return = (returns[-1] / returns[0] - 1) * 100 if len(returns) > 1 else 0
                episode_volatility = np.std(np.diff(returns) / returns[:-1]) * np.sqrt(252) if len(returns) > 1 else 0
                episode_sharpe = episode_return / episode_volatility if episode_volatility > 0 else 0
                episode_max_drawdown = self._calculate_max_drawdown(returns)
            else:
                episode_return = episode_volatility = episode_sharpe = episode_max_drawdown = 0
            
            results.append({
                "episode": episode,
                "reward": episode_reward,
                "return": episode_return,
                "volatility": episode_volatility,
                "sharpe_ratio": episode_sharpe,
                "max_drawdown": episode_max_drawdown,
                "trades": episode_trades
            })
        
        # Calculate aggregate metrics
        returns = [r["return"] for r in results]
        sharpe_ratios = [r["sharpe_ratio"] for r in results]
        max_drawdowns = [r["max_drawdown"] for r in results]
        
        return {
            "episodes": results,
            "aggregate_metrics": {
                "mean_return": np.mean(returns),
                "std_return": np.std(returns),
                "mean_sharpe_ratio": np.mean(sharpe_ratios),
                "mean_max_drawdown": np.mean(max_drawdowns),
                "win_rate": len([r for r in returns if r > 0]) / len(returns),
                "total_episodes": len(results)
            }
        }
    
    def _calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown"""
        peak = np.maximum.accumulate(returns)
        drawdown = (returns - peak) / peak
        return np.min(drawdown) * 100
    
    def _compare_with_benchmarks(self, evaluation_results) -> Dict[str, Any]:
        """Compare performance with market benchmarks"""
        # This would typically fetch real benchmark data
        # For now, we'll use simulated benchmarks
        
        benchmark_returns = {
            "SPY": 0.12,  # 12% annual return
            "QQQ": 0.15,  # 15% annual return
            "IWM": 0.10   # 10% annual return
        }
        
        our_return = evaluation_results["aggregate_metrics"]["mean_return"]
        our_sharpe = evaluation_results["aggregate_metrics"]["mean_sharpe_ratio"]
        
        comparison = {
            "our_performance": {
                "annual_return": our_return,
                "sharpe_ratio": our_sharpe
            },
            "benchmarks": benchmark_returns,
            "outperformance": {
                symbol: our_return - benchmark_return
                for symbol, benchmark_return in benchmark_returns.items()
            }
        }
        
        return comparison


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="WealthArena Advanced Trading Training")
    parser.add_argument("--config", type=str, default="config/production_config.yaml", help="Config file")
    parser.add_argument("--experiment", type=str, default="wealtharena_advanced", help="Experiment name")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate"], default="train", help="Mode")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path for evaluation")
    parser.add_argument("--local", action="store_true", help="Run locally")
    
    args = parser.parse_args()
    
    # Initialize Ray
    if args.local:
        ray.init(local_mode=True, num_cpus=8)
    else:
        ray.init(address="auto")
    
    # Create trainer
    trainer = AdvancedTradingTrainer(args.config)
    
    try:
        if args.mode == "train":
            # Run training
            results = trainer.train(args.experiment)
            
            if results["success"]:
                print("✅ Advanced training completed successfully!")
                print(f"Best checkpoint: {results['checkpoint']}")
            else:
                print(f"❌ Training failed: {results['error']}")
                sys.exit(1)
        
        elif args.mode == "evaluate":
            if not args.checkpoint:
                print("❌ Checkpoint path required for evaluation")
                sys.exit(1)
            
            # Run evaluation
            results = trainer.evaluate_performance(args.checkpoint)
            
            if "error" not in results:
                print("✅ Performance evaluation completed!")
                print(f"Results saved to: results/performance_evaluation.json")
            else:
                print(f"❌ Evaluation failed: {results['error']}")
                sys.exit(1)
    
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()