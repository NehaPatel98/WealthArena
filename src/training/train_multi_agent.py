"""
Multi-Agent Training Script for WealthArena Trading System

This module provides training capabilities for the multi-agent trading system
using RLlib with comprehensive experiment tracking and evaluation.
"""

import ray
from ray import tune, air
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.a2c import A2C
from ray.rllib.algorithms.sac import SAC
from ray.rllib.algorithms.td3 import TD3
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.air.integrations.mlflow import MLflowLoggerCallback
import yaml
import argparse
import os
import sys
from pathlib import Path
import logging
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from environments.multi_agent_env import WealthArenaMultiAgentEnv
from tracking.mlflow_tracker import MLflowTracker
from tracking.wandb_tracker import WandbTracker

logger = logging.getLogger(__name__)


class MultiAgentTrainer:
    """
    Multi-agent trainer for WealthArena trading system
    
    Handles training of multiple trading agents using RLlib with comprehensive
    experiment tracking and evaluation capabilities.
    """
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize trackers
        self.trackers = self._initialize_trackers()
        
        # Training state
        self.algorithm = None
        self.training_results = None
        
        logger.info(f"Multi-agent trainer initialized with config: {config_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load training configuration from YAML file"""
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            logger.info("Configuration loaded successfully")
            return config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def _initialize_trackers(self) -> Dict[str, Any]:
        """Initialize experiment tracking"""
        
        trackers = {}
        
        # MLflow tracker
        if self.config.get("experiment_tracking", {}).get("mlflow", {}).get("enabled", False):
            try:
                mlflow_config = self.config["experiment_tracking"]["mlflow"]
                trackers["mlflow"] = MLflowTracker(mlflow_config)
                logger.info("MLflow tracker initialized")
            except Exception as e:
                logger.error(f"Error initializing MLflow tracker: {e}")
        
        # W&B tracker
        if self.config.get("experiment_tracking", {}).get("wandb", {}).get("enabled", False):
            try:
                wandb_config = self.config["experiment_tracking"]["wandb"]
                trackers["wandb"] = WandbTracker(wandb_config)
                logger.info("W&B tracker initialized")
            except Exception as e:
                logger.error(f"Error initializing W&B tracker: {e}")
        
        return trackers
    
    def _get_algorithm_class(self, algorithm_name: str):
        """Get algorithm class by name"""
        
        algorithms = {
            "PPO": PPO,
            "A2C": A2C,
            "SAC": SAC,
            "TD3": TD3
        }
        
        if algorithm_name not in algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
        return algorithms[algorithm_name]
    
    def _create_training_config(self) -> Dict[str, Any]:
        """Create RLlib training configuration"""
        
        env_config = self.config["environment"]
        training_config = self.config["training"]
        resources_config = self.config["resources"]
        
        # Get algorithm class
        algorithm_class = self._get_algorithm_class(training_config["algorithm"])
        
        # Base training configuration
        rllib_config = {
            # Environment
            "env": WealthArenaMultiAgentEnv,
            "env_config": env_config,
            
            # Multi-agent configuration
            "multiagent": training_config["multiagent"],
            
            # Algorithm-specific parameters
            "lr": training_config["learning_rate"],
            "gamma": training_config["gamma"],
            "lambda": training_config["gae_lambda"],
            "entropy_coeff": training_config["entropy_coeff"],
            "vf_loss_coeff": training_config["vf_loss_coeff"],
            "clip_param": training_config["clip_param"],
            "num_sgd_iter": training_config["num_sgd_iter"],
            "sgd_minibatch_size": training_config["sgd_minibatch_size"],
            "train_batch_size": training_config["train_batch_size"],
            
            # Resource configuration
            "num_workers": resources_config["num_workers"],
            "num_envs_per_worker": resources_config["num_envs_per_worker"],
            "num_cpus_per_worker": resources_config["num_cpus_per_worker"],
            "num_gpus": resources_config["num_gpus"],
            "local_mode": resources_config["local_mode"],
            
            # Framework
            "framework": "torch",
            
            # Evaluation
            "evaluation_interval": self.config["evaluation"]["eval_interval"],
            "evaluation_duration": self.config["evaluation"]["eval_duration"],
            "evaluation_config": {
                "explore": False,
                "num_envs_per_worker": 1
            },
            
            # Checkpointing
            "checkpoint_freq": self.config["checkpointing"]["checkpoint_freq"],
            "keep_checkpoints_num": self.config["checkpointing"]["keep_checkpoints_num"],
            
            # Logging
            "log_level": self.config["logging"]["log_level"]
        }
        
        return rllib_config
    
    def _create_callbacks(self) -> List[Any]:
        """Create training callbacks"""
        
        callbacks = []
        
        # MLflow callback
        if "mlflow" in self.trackers:
            try:
                from ray.air.integrations.mlflow import MLflowLoggerCallback
                mlflow_config = self.config["experiment_tracking"]["mlflow"]
                callbacks.append(
                    MLflowLoggerCallback(
                        tracking_uri=mlflow_config["tracking_uri"],
                        experiment_name=mlflow_config["experiment_name"],
                        save_artifact=True
                    )
                )
            except Exception as e:
                logger.error(f"Error creating MLflow callback: {e}")
        
        # W&B callback
        if "wandb" in self.trackers:
            try:
                from ray.air.integrations.wandb import WandbLoggerCallback
                wandb_config = self.config["experiment_tracking"]["wandb"]
                callbacks.append(
                    WandbLoggerCallback(
                        project=wandb_config["project_name"],
                        entity=wandb_config.get("entity"),
                        log_config=True,
                        save_checkpoints=True
                    )
                )
            except Exception as e:
                logger.error(f"Error creating W&B callback: {e}")
        
        return callbacks
    
    def train(self, 
              max_iterations: int = None,
              target_reward: float = None,
              smoke_test: bool = False) -> Dict[str, Any]:
        """Train the multi-agent system"""
        
        try:
            # Start tracking
            self._start_tracking()
            
            # Create training configuration
            training_config = self._create_training_config()
            
            # Override parameters if provided
            if max_iterations is not None:
                training_config["max_iterations"] = max_iterations
            if target_reward is not None:
                training_config["target_reward"] = target_reward
            
            # Create callbacks
            callbacks = self._create_callbacks()
            
            # Setup stop criteria
            if smoke_test:
                stop_criteria = {"training_iteration": 2}
            else:
                stop_criteria = {
                    "training_iteration": training_config.get("max_iterations", 1000),
                    "episode_reward_mean": training_config.get("target_reward", 100.0)
                }
            
            # Create tuner
            tuner = tune.Tuner(
                training_config["algorithm"],
                param_space=training_config,
                run_config=air.RunConfig(
                    name=f"wealtharena_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    stop=stop_criteria,
                    callbacks=callbacks,
                    checkpoint_config=air.CheckpointConfig(
                        checkpoint_frequency=training_config["checkpoint_freq"],
                        num_to_keep=training_config["keep_checkpoints_num"]
                    ),
                    failure_config=air.FailureConfig(max_failures=3),
                    progress_reporter=tune.CLIReporter(
                        metric_columns={
                            "training_iteration": "iter",
                            "episode_reward_mean": "reward_mean",
                            "episode_len_mean": "len_mean",
                            "time_total_s": "time_total_s"
                        }
                    )
                )
            )
            
            # Run training
            logger.info("Starting training...")
            results = tuner.fit()
            
            # Get best result
            best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
            
            # Store results
            self.training_results = {
                "best_result": best_result,
                "all_results": results,
                "config": training_config
            }
            
            # Log results
            self._log_training_results(best_result)
            
            logger.info(f"Training completed. Best reward: {best_result.metrics['episode_reward_mean']}")
            
            return self.training_results
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
        finally:
            # End tracking
            self._end_tracking()
    
    def _start_tracking(self):
        """Start experiment tracking"""
        
        for tracker_name, tracker in self.trackers.items():
            try:
                run_name = f"wealtharena_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                run_id = tracker.start_run(run_name)
                
                if run_id:
                    # Log configuration
                    tracker.log_parameters(self.config)
                    logger.info(f"Started {tracker_name} tracking: {run_id}")
                
            except Exception as e:
                logger.error(f"Error starting {tracker_name} tracking: {e}")
    
    def _end_tracking(self):
        """End experiment tracking"""
        
        for tracker_name, tracker in self.trackers.items():
            try:
                tracker.end_run()
                logger.info(f"Ended {tracker_name} tracking")
            except Exception as e:
                logger.error(f"Error ending {tracker_name} tracking: {e}")
    
    def _log_training_results(self, best_result):
        """Log training results to trackers"""
        
        for tracker_name, tracker in self.trackers.items():
            try:
                # Log final metrics
                final_metrics = {
                    "final_episode_reward_mean": best_result.metrics.get("episode_reward_mean", 0),
                    "final_episode_length_mean": best_result.metrics.get("episode_len_mean", 0),
                    "final_training_iteration": best_result.metrics.get("training_iteration", 0),
                    "final_time_total_s": best_result.metrics.get("time_total_s", 0)
                }
                
                tracker.log_metrics(final_metrics)
                
                # Log checkpoint path
                if hasattr(best_result, 'checkpoint'):
                    tracker.log_artifacts({"best_checkpoint": str(best_result.checkpoint)})
                
                logger.info(f"Logged training results to {tracker_name}")
                
            except Exception as e:
                logger.error(f"Error logging results to {tracker_name}: {e}")
    
    def evaluate(self, checkpoint_path: str = None) -> Dict[str, Any]:
        """Evaluate the trained model"""
        
        try:
            if checkpoint_path is None and self.training_results is not None:
                checkpoint_path = str(self.training_results["best_result"].checkpoint)
            
            if checkpoint_path is None:
                raise ValueError("No checkpoint path provided")
            
            # Create evaluation environment
            env_config = self.config["environment"]
            eval_env = WealthArenaMultiAgentEnv(env_config)
            
            # Load algorithm
            training_config = self._create_training_config()
            algorithm_class = self._get_algorithm_class(training_config["algorithm"])
            
            # Create algorithm instance
            algorithm = algorithm_class(training_config)
            
            # Load checkpoint
            algorithm.restore(checkpoint_path)
            
            # Run evaluation
            eval_results = self._run_evaluation(algorithm, eval_env)
            
            # Log evaluation results
            self._log_evaluation_results(eval_results)
            
            return eval_results
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise
    
    def _run_evaluation(self, algorithm, env, num_episodes: int = 10) -> Dict[str, Any]:
        """Run evaluation episodes"""
        
        eval_results = {
            "episode_rewards": [],
            "episode_lengths": [],
            "portfolio_values": [],
            "sharpe_ratios": [],
            "max_drawdowns": []
        }
        
        for episode in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0
            episode_length = 0
            portfolio_values = []
            
            while True:
                # Get actions from all agents
                actions = {}
                for agent_id in env.agent_ids:
                    if agent_id in obs:
                        action = algorithm.compute_single_action(obs[agent_id])
                        actions[agent_id] = action
                
                # Step environment
                obs, rewards, terminateds, truncateds, infos = env.step(actions)
                
                # Update episode statistics
                episode_reward += sum(rewards.values())
                episode_length += 1
                
                # Track portfolio values
                for agent_id in env.agent_ids:
                    if agent_id in infos:
                        portfolio_values.append(infos[agent_id].get("portfolio_value", 0))
                
                # Check if episode is done
                if terminateds.get("__all__", False) or truncateds.get("__all__", False):
                    break
            
            # Calculate episode metrics
            eval_results["episode_rewards"].append(episode_reward)
            eval_results["episode_lengths"].append(episode_length)
            
            if portfolio_values:
                eval_results["portfolio_values"].append(np.mean(portfolio_values))
                
                # Calculate Sharpe ratio
                returns = np.diff(portfolio_values)
                if len(returns) > 1 and np.std(returns) > 0:
                    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
                    eval_results["sharpe_ratios"].append(sharpe_ratio)
                else:
                    eval_results["sharpe_ratios"].append(0)
                
                # Calculate max drawdown
                peak = np.maximum.accumulate(portfolio_values)
                drawdown = (peak - np.array(portfolio_values)) / peak
                max_drawdown = np.max(drawdown)
                eval_results["max_drawdowns"].append(max_drawdown)
            else:
                eval_results["portfolio_values"].append(0)
                eval_results["sharpe_ratios"].append(0)
                eval_results["max_drawdowns"].append(0)
        
        # Calculate summary statistics
        eval_results["mean_episode_reward"] = np.mean(eval_results["episode_rewards"])
        eval_results["std_episode_reward"] = np.std(eval_results["episode_rewards"])
        eval_results["mean_episode_length"] = np.mean(eval_results["episode_lengths"])
        eval_results["mean_portfolio_value"] = np.mean(eval_results["portfolio_values"])
        eval_results["mean_sharpe_ratio"] = np.mean(eval_results["sharpe_ratios"])
        eval_results["mean_max_drawdown"] = np.mean(eval_results["max_drawdowns"])
        
        return eval_results
    
    def _log_evaluation_results(self, eval_results: Dict[str, Any]):
        """Log evaluation results to trackers"""
        
        for tracker_name, tracker in self.trackers.items():
            try:
                # Log evaluation metrics
                eval_metrics = {
                    "eval_mean_episode_reward": eval_results["mean_episode_reward"],
                    "eval_std_episode_reward": eval_results["std_episode_reward"],
                    "eval_mean_episode_length": eval_results["mean_episode_length"],
                    "eval_mean_portfolio_value": eval_results["mean_portfolio_value"],
                    "eval_mean_sharpe_ratio": eval_results["mean_sharpe_ratio"],
                    "eval_mean_max_drawdown": eval_results["mean_max_drawdown"]
                }
                
                tracker.log_metrics(eval_metrics)
                
                logger.info(f"Logged evaluation results to {tracker_name}")
                
            except Exception as e:
                logger.error(f"Error logging evaluation results to {tracker_name}: {e}")
    
    def save_model(self, model_path: str = None):
        """Save the trained model"""
        
        if self.training_results is None:
            logger.warning("No training results to save")
            return
        
        try:
            if model_path is None:
                model_path = self.config["model"]["model_dir"]
            
            # Create model directory
            os.makedirs(model_path, exist_ok=True)
            
            # Save best checkpoint
            best_checkpoint = self.training_results["best_result"].checkpoint
            checkpoint_path = os.path.join(model_path, "best_checkpoint")
            
            # Copy checkpoint files
            import shutil
            shutil.copytree(best_checkpoint, checkpoint_path, dirs_exist_ok=True)
            
            # Save configuration
            config_path = os.path.join(model_path, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            
            # Save training results
            results_path = os.path.join(model_path, "training_results.json")
            import json
            with open(results_path, 'w') as f:
                json.dump({
                    "best_reward": self.training_results["best_result"].metrics.get("episode_reward_mean", 0),
                    "best_iteration": self.training_results["best_result"].metrics.get("training_iteration", 0),
                    "checkpoint_path": str(best_checkpoint)
                }, f, indent=2)
            
            logger.info(f"Model saved to: {model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")


def train_multi_agent(config_path: str, 
                     max_iterations: int = None,
                     target_reward: float = None,
                     smoke_test: bool = False) -> Dict[str, Any]:
    """Train multi-agent trading system"""
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
    
    try:
        # Create trainer
        trainer = MultiAgentTrainer(config_path)
        
        # Train
        results = trainer.train(
            max_iterations=max_iterations,
            target_reward=target_reward,
            smoke_test=smoke_test
        )
        
        # Evaluate
        eval_results = trainer.evaluate()
        results["evaluation"] = eval_results
        
        # Save model
        trainer.save_model()
        
        return results
        
    except Exception as e:
        logger.error(f"Error in training: {e}")
        raise
    finally:
        # Shutdown Ray
        ray.shutdown()


def main():
    """Main training function"""
    
    parser = argparse.ArgumentParser(description="Train WealthArena Multi-Agent Trading AI")
    parser.add_argument("--config", type=str, default="config/training_config.yaml", 
                       help="Path to config file")
    parser.add_argument("--max-iterations", type=int, default=None,
                       help="Maximum training iterations")
    parser.add_argument("--target-reward", type=float, default=None,
                       help="Target reward for early stopping")
    parser.add_argument("--smoke-test", action="store_true",
                       help="Run quick smoke test")
    parser.add_argument("--local", action="store_true",
                       help="Run in local mode")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize Ray
    if args.local:
        ray.init(local_mode=True)
    else:
        ray.init()
    
    try:
        # Train
        results = train_multi_agent(
            config_path=args.config,
            max_iterations=args.max_iterations,
            target_reward=args.target_reward,
            smoke_test=args.smoke_test
        )
        
        print("Training completed successfully!")
        print(f"Best reward: {results['best_result'].metrics.get('episode_reward_mean', 0)}")
        print(f"Evaluation results: {results['evaluation']}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
