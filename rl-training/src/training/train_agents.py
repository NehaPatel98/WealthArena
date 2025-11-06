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

# Add src to path for custom imports
# Don't add parent directory here to avoid namespace conflicts with collect_training_data.py
# Parent directory will be added only when collect_training_data is needed
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set environment variables BEFORE importing Ray/PyTorch to prevent libuv usage on Windows
# This must be done before any imports that might trigger PyTorch distributed initialization
if sys.platform == "win32":
    os.environ.setdefault("RAY_DISABLE_IMPORT_WARNING", "1")
    # CRITICAL: Disable libuv usage in PyTorch (Windows doesn't support libuv)
    os.environ["USE_LIBUV"] = "0"  # This prevents PyTorch from trying to use libuv
    os.environ.setdefault("TORCH_NCCL_DISABLE", "1")
    os.environ.setdefault("TORCH_DISTRIBUTED_BACKEND", "gloo")
    os.environ.setdefault("TORCH_DISTRIBUTED_DETAIL", "OFF")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("RAY_TRAIN_ENABLE_WORKER_SPREAD", "0")
    os.environ.setdefault("RAY_TRAIN_ENABLE_WORKER_CACHING", "0")
    os.environ.setdefault("RAY_TRAIN_ENABLE_NCCL", "0")

# Ray and RLlib imports
import ray
from ray import tune
from ray import air
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.sac import SAC, SACConfig
# A2C removed: deprecated and no longer available in Ray 2.50.1
# Use PPO as replacement for on-policy algorithms
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.policy import Policy
# from ray.tune.logger import WandbLoggerCallback
# from ray.tune.logger import MLflowLoggerCallback

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
                # Lazy import with explicit path to avoid namespace collision
                import sys
                import os
                # Save current sys.argv to restore later
                original_argv = sys.argv.copy()
                # Store whether parent_dir was in sys.path originally
                parent_dir = Path(__file__).parent.parent.parent
                parent_dir_str = str(parent_dir)
                path_was_inserted = parent_dir_str not in sys.path
                
                try:
                    # Add parent directory to path to find collect_training_data.py
                    if path_was_inserted:
                        sys.path.insert(0, parent_dir_str)
                    
                    # Temporarily set sys.argv to arguments for collect_training_data
                    # Use cached/processed data if available, otherwise fallback to synthetic
                    # This respects existing processed data while providing fallback when needed
                    sys.argv = ['collect_training_data.py', '--stocks', 
                                '--max-stocks-symbols', '20']  # 20 symbols max
                    
                    # Import at runtime to prevent module-level namespace issues
                    import collect_training_data
                    collect_training_data.main()
                finally:
                    # Always restore original sys.argv
                    sys.argv = original_argv
                    # Remove parent_dir from sys.path if we inserted it and it wasn't there originally
                    if path_was_inserted and parent_dir_str in sys.path:
                        try:
                            sys.path.remove(parent_dir_str)
                        except ValueError:
                            # Path might have been removed already or modified, ignore
                            pass
            else:
                logger.info(f"Market data already exists in {data_dir}")
            
            # After data collection and cache checks, update config symbols to match processed CSVs
            # Check for processed data files (including synthetic data)
            processed_files = list(data_dir.glob("*_processed.csv"))
            # Also check stocks subdirectory
            stocks_dir = data_dir / "stocks"
            if stocks_dir.exists():
                processed_files.extend(list(stocks_dir.glob("*_processed.csv")))
            
            if processed_files:
                # Extract symbols from file names
                available_symbols = [f.stem.replace("_processed", "") for f in processed_files]
                
                # Remove duplicates and sort
                available_symbols = sorted(list(set(available_symbols)))
                # Verify that corresponding processed files exist for each symbol
                verified_symbols = []
                for symbol in available_symbols:
                    # Check in stocks subdirectory first
                    symbol_file = data_dir / "stocks" / f"{symbol}_processed.csv"
                    if not symbol_file.exists():
                        # Check flat structure
                        symbol_file = data_dir / f"{symbol}_processed.csv"
                        if not symbol_file.exists():
                            logger.warning(f"Symbol {symbol} in available list but file not found, skipping")
                            continue
                    verified_symbols.append(symbol)
                
                if not verified_symbols:
                    logger.error("No verified processed files found for any symbols")
                    # Try to use cached verified symbols as fallback
                    try:
                        from src.data.asx.asx_symbols import get_verified_symbols
                        verified_symbols = get_verified_symbols(limit=10)
                        logger.warning(f"Falling back to verified symbols: {len(verified_symbols)} symbols")
                        # Check which verified symbols have processed files
                        available_verified = []
                        for symbol in verified_symbols:
                            symbol_file = data_dir / "stocks" / f"{symbol}_processed.csv"
                            if not symbol_file.exists():
                                symbol_file = data_dir / f"{symbol}_processed.csv"
                            if symbol_file.exists():
                                available_verified.append(symbol)
                        if available_verified:
                            verified_symbols = available_verified
                        else:
                            logger.error("No processed data exists for verified symbols either. Forcing data collection re-run.")
                            return False
                    except Exception as e:
                        logger.error(f"Failed to get verified symbols: {e}")
                        return False
                
                # Optionally reduce to configured maximum count
                max_symbols = self.config.get("data", {}).get("max_symbols", None)
                if max_symbols and len(verified_symbols) > max_symbols:
                    verified_symbols = verified_symbols[:max_symbols]
                    logger.info(f"Reduced symbol list to configured maximum: {max_symbols}")
                
                # Update config with verified symbols that have processed files
                self.config["data"]["symbols"] = verified_symbols
                logger.info(f"Updated config symbols to {len(verified_symbols)} verified symbols with processed files: {verified_symbols}")
                
                # Update num_assets to match verified symbols count (always equals final symbol list length)
                self.config["environment"]["num_assets"] = len(verified_symbols)
                logger.info(f"Updated num_assets to {len(verified_symbols)} to match verified symbols count")
            else:
                # No processed data found - check if synthetic data was generated
                logger.warning("No processed CSVs found. Checking for synthetic data...")
                
                # Check for synthetic data metadata
                synthetic_metadata_path = data_dir / "synthetic_data_metadata.json"
                if synthetic_metadata_path.exists():
                    try:
                        import json
                        with open(synthetic_metadata_path, 'r') as f:
                            metadata = json.load(f)
                            synthetic_symbols = metadata.get('symbols', [])
                            if synthetic_symbols:
                                logger.info(f"Found {len(synthetic_symbols)} synthetic symbols in metadata")
                                # Synthetic data might be generated but not saved yet
                                # Try to use symbols from metadata
                                self.config["data"]["symbols"] = synthetic_symbols
                                self.config["environment"]["num_assets"] = len(synthetic_symbols)
                                logger.info(f"Using {len(synthetic_symbols)} synthetic symbols from metadata")
                                return True
                    except Exception as e:
                        logger.warning(f"Failed to read synthetic metadata: {e}")
                
                # If still no data, return False to trigger synthetic data generation
                logger.error("No processed data found. Synthetic data generation should have been triggered.")
                return False
            
            logger.info(f"Market data prepared for {len(self.config['data']['symbols'])} symbols")
            return True
            
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            
            # Enhanced error logging with actionable messages
            error_msg = str(e).lower()
            
            # Check for specific exception types
            if isinstance(e, ConnectionError) or "connection" in error_msg:
                logger.error("Connection error detected. This suggests:")
                logger.error("  - Yahoo Finance API is inaccessible")
                logger.error("  - Network connectivity issues")
                logger.error("  - Firewall or proxy blocking requests")
                logger.error("Recommendation: Check internet connection and try again in 1 hour")
            
            elif isinstance(e, TimeoutError) or "timeout" in error_msg:
                logger.error("Timeout error detected. This suggests:")
                logger.error("  - Yahoo Finance API is slow or overloaded")
                logger.error("  - Network latency is high")
                logger.error("Recommendation: Try again later or reduce --max-stocks-symbols")
            
            elif "json" in error_msg or "parse" in error_msg:
                logger.error("JSON parsing error detected. This suggests:")
                logger.error("  - Yahoo Finance API returned invalid data")
                logger.error("  - API rate limiting (Expecting value: line 1 column 1)")
                logger.error("Recommendation: Use synthetic data instead - run with --use-synthetic-data flag")
                logger.error("Or wait 1 hour and try again")
            
            else:
                logger.error(f"Unknown error: {e}")
                logger.error("Recommendation: Check logs for details and verify data collection settings")
            
            # Save failed symbols list for manual review
            try:
                failed_symbols_file = Path("logs/failed_symbols.txt")
                failed_symbols_file.parent.mkdir(parents=True, exist_ok=True)
                with open(failed_symbols_file, 'w') as f:
                    f.write(f"Data collection failed at {datetime.now().isoformat()}\n")
                    f.write(f"Error: {str(e)}\n")
                    f.write(f"Symbols requested: {self.config.get('data', {}).get('symbols', [])}\n")
                logger.info(f"Failed symbols list saved to: {failed_symbols_file}")
            except Exception as save_err:
                logger.warning(f"Failed to save failed symbols list: {save_err}")
            
            return False
    
    def create_optimized_environment(self) -> MultiAgentEnv:
        """Create optimized trading environment"""
        logger.info("Creating optimized trading environment...")
        
        # Ensure num_assets matches actual symbol count
        num_assets = len(self.config["data"]["symbols"])
        self.config["environment"]["num_assets"] = num_assets
        
        # Handle both initial_cash and initial_cash_per_agent for compatibility
        initial_cash_per_agent = self.config["environment"].get(
            "initial_cash_per_agent",
            self.config["environment"].get("initial_cash", 1000000)
        )
        
        env_config = {
            "num_agents": self.config["environment"]["num_agents"],
            "num_assets": num_assets,
            "episode_length": self.config["environment"]["episode_length"],
            "initial_cash_per_agent": initial_cash_per_agent,
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
    
    def create_advanced_algorithm_config(self, env: MultiAgentEnv):
        """Create advanced algorithm configuration"""
        algorithm_name = self.config["training"]["algorithm"]
        
        if algorithm_name == "PPO":
            algo_config = PPOConfig()
        elif algorithm_name == "SAC":
            algo_config = SACConfig()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm_name}. Only PPO and SAC are supported (A2C was deprecated in Ray 2.50.1)")
        
        # Environment setup - build complete env_config mirroring create_optimized_environment()
        # Ensure env_config includes all keys used by WealthArenaMultiAgentEnv
        num_assets = len(self.config["data"]["symbols"])
        
        # Handle both initial_cash and initial_cash_per_agent for compatibility
        initial_cash_per_agent = self.config["environment"].get(
            "initial_cash_per_agent",
            self.config["environment"].get("initial_cash", 1000000)
        )
        
        env_config = {
            "num_agents": self.config["environment"]["num_agents"],
            "num_assets": num_assets,
            "episode_length": self.config["environment"]["episode_length"],
            "initial_cash_per_agent": initial_cash_per_agent,
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
        
        algo_config.environment(
            env=WealthArenaMultiAgentEnv,
            env_config=env_config,
            disable_env_checking=True
        )
        
        # Multi-agent configuration with different agent types
        # Policies are defined as tuples: (policy_class, obs_space, action_space, config_dict)
        # None for spaces means RLlib will infer them from the environment automatically
        # This avoids issues with Box spaces not being iterable in newer RLlib versions
        algo_config.multi_agent(
            policies={
                "conservative_policy": (
                    None,  # Use default policy class
                    None,  # RLlib will infer obs_space from environment
                    None,  # RLlib will infer action_space from environment
                    {
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
                "aggressive_policy": (
                    None,  # Use default policy class
                    None,  # RLlib will infer obs_space from environment
                    None,  # RLlib will infer action_space from environment
                    {
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
                "balanced_policy": (
                    None,  # Use default policy class
                    None,  # RLlib will infer obs_space from environment
                    None,  # RLlib will infer action_space from environment
                    {
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
            lambda_=training_config["gae_lambda"],  # lambda_ is the correct parameter name in RLlib
            entropy_coeff=training_config["entropy_coeff"],
            vf_loss_coeff=training_config["vf_loss_coeff"],
            clip_param=training_config["clip_param"],
            num_sgd_iter=training_config["num_sgd_iter"],
            minibatch_size=training_config["sgd_minibatch_size"],  # Changed from sgd_minibatch_size to minibatch_size
            train_batch_size=training_config["train_batch_size"],
            # Removed invalid parameters: kl_coeff, kl_target, use_gae, use_critic, use_huber, huber_threshold, lr_schedule
            # These are either defaults or need to be set via other methods in RLlib
        )
        
        # Environment runners configuration (replaces deprecated rollouts())
        resources_config = self.config["resources"]
        # On Windows, avoid distributed training to prevent libuv requirement
        # Use single worker (0 env runners = use main process only)
        is_windows = sys.platform == "win32"
        if is_windows:
            logger.info("Windows detected: Using single-worker configuration to avoid libuv requirement")
            num_env_runners = 0  # Use main process only, no distributed workers
        else:
            num_env_runners = resources_config["num_workers"]
        
        algo_config.env_runners(
            num_env_runners=num_env_runners,
            num_envs_per_env_runner=resources_config["num_envs_per_worker"] if num_env_runners > 0 else 1,
            num_cpus_per_env_runner=resources_config["num_cpus_per_worker"] if num_env_runners > 0 else 1
        )
        
        # Learner configuration (GPU allocation)
        # On Windows, ensure single learner to avoid distributed setup
        algo_config.learners(
            num_learners=1,  # Always use single learner to avoid libuv requirement on Windows
            num_gpus_per_learner=resources_config["num_gpus"] if resources_config["num_gpus"] > 0 else 0
        )
        
        # Framework
        algo_config.framework("torch")
        
        # Advanced evaluation
        evaluation_config = self.config["evaluation"]
        algo_config.evaluation(
            evaluation_interval=evaluation_config["eval_interval"],
            evaluation_duration=evaluation_config["eval_duration"],
            evaluation_duration_unit="episodes",
            evaluation_num_env_runners=2,  # Changed from evaluation_num_workers
            evaluation_config={
                "explore": False,
                "num_envs_per_env_runner": 1,  # Changed from num_envs_per_worker
            }
        )
        
        # Checkpointing configuration
        # Note: Checkpoint frequency is handled via air.CheckpointConfig in RunConfig (see line 639)
        # The checkpointing() method in AlgorithmConfig only handles export-related settings if needed
        # We'll handle checkpoint frequency in the training loop via air.CheckpointConfig
        
        # Logging
        algo_config.debugging(log_level="INFO")
        
        return algo_config
    
    def _policy_mapping_fn(self, agent_id, episode, worker=None, **kwargs):
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
        
        # Weights & Biases (optional, non-blocking)
        if self.config.get("experiment_tracking", {}).get("wandb", {}).get("enabled", False):
            try:
                # Check if WandB API key is available before attempting to use WandB
                import os
                import wandb
                api_key = os.environ.get("WANDB_API_KEY") or wandb.api.api_key
                
                if not api_key:
                    # Try to check if wandb is logged in
                    try:
                        wandb.login(relogin=False)
                        api_key = wandb.api.api_key
                    except:
                        pass
                
                if not api_key:
                    logger.warning("WandB API key not found. WandB tracking will be disabled.")
                    logger.warning("To use W&B: set WANDB_API_KEY environment variable or run 'wandb login'")
                else:
                    try:
                        wandb_config = self.config["experiment_tracking"]["wandb"]
                        # WandbTracker expects a config dict, not keyword arguments
                        wandb_tracker_config = {
                            "project_name": wandb_config.get("project", "wealtharena-trading"),
                            "entity": wandb_config.get("entity", "wealtharena"),
                            "tags": wandb_config.get("tags", []),
                            "notes": wandb_config.get("notes", "")
                        }
                        wandb_tracker = WandbTracker(wandb_tracker_config)
                        # Use the utility function to create RLlib callback
                        from tracking.wandb_tracker import create_wandb_callback
                        wandb_callback = create_wandb_callback(wandb_tracker)
                        callbacks.append(wandb_callback)
                        logger.info("W&B tracking enabled")
                    except Exception as callback_error:
                        logger.warning(f"Failed to create WandB callback (training will continue without it): {callback_error}")
            except Exception as e:
                logger.warning(f"Failed to initialize W&B tracking (training will continue without it): {e}")
                logger.warning("To use W&B: set WANDB_API_KEY environment variable or run 'wandb login'")
        
        # MLflow (optional, non-blocking)
        if self.config.get("experiment_tracking", {}).get("mlflow", {}).get("enabled", False):
            try:
                mlflow_config = self.config["experiment_tracking"]["mlflow"]
                # MLflowTracker expects a config dict, not keyword arguments
                # Use local file tracking if server is not available
                mlflow_tracker_config = {
                    "tracking_uri": mlflow_config.get("tracking_uri", "file:./mlruns"),
                    "experiment_name": mlflow_config.get("experiment_name", "WealthArena_Production"),
                    "artifact_location": mlflow_config.get("artifact_location", "./mlruns")
                }
                # Try to connect to server, fallback to local file storage if it fails
                tracking_uri = mlflow_config.get("tracking_uri", "http://localhost:5000")
                if tracking_uri.startswith("http://") or tracking_uri.startswith("https://"):
                    # Try server connection first
                    try:
                        import mlflow
                        mlflow.set_tracking_uri(tracking_uri)
                        # Test connection by trying to list experiments
                        mlflow.search_experiments(max_results=1)
                    except Exception as server_error:
                        logger.warning(f"MLflow server at {tracking_uri} is not available: {server_error}")
                        logger.info("Falling back to local file storage for MLflow tracking")
                        mlflow_tracker_config["tracking_uri"] = "file:./mlruns"
                
                mlflow_tracker = MLflowTracker(mlflow_tracker_config)
                # Use the utility function to create RLlib callback
                from tracking.mlflow_tracker import create_mlflow_callback
                callbacks.append(create_mlflow_callback(mlflow_tracker))
                logger.info("MLflow tracking enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize MLflow tracking (training will continue without it): {e}")
        
        return callbacks
    
    def train(self, experiment_name: str = "wealtharena_advanced") -> Dict[str, Any]:
        """Run advanced training"""
        logger.info(f"Starting advanced training: {experiment_name}")
        
        try:
            # Initialize Ray if not already initialized
            # On Windows, ensure local mode to avoid libuv requirement and Kubernetes connections
            is_windows = sys.platform == "win32"
            if not ray.is_initialized():
                # CRITICAL: Clear any cluster/Kubernetes addresses BEFORE Ray init
                os.environ["RAY_ADDRESS"] = ""  # Prevent cluster auto-discovery
                os.environ["RAY_HEAD_NODE_IP"] = ""  # Clear head node IP
                os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"
                
                if is_windows:
                    logger.info("Windows detected: Initializing Ray in local mode to avoid libuv requirement and Kubernetes")
                    # CRITICAL: Disable libuv usage in PyTorch (Windows doesn't support libuv)
                    os.environ["USE_LIBUV"] = "0"  # This prevents PyTorch from trying to use libuv
                    # Disable PyTorch distributed to prevent libuv requirement and Kubernetes connections
                    os.environ["TORCH_NCCL_DISABLE"] = "1"  # Disable NCCL
                    os.environ["TORCH_DISTRIBUTED_BACKEND"] = "gloo"  # Use Gloo backend instead of libuv
                    os.environ["TORCH_DISTRIBUTED_DETAIL"] = "OFF"  # Disable distributed logging
                    # Set single-node distributed config (prevents libuv initialization and Kubernetes)
                    os.environ["MASTER_ADDR"] = "127.0.0.1"
                    os.environ["MASTER_PORT"] = "29500"
                    os.environ["WORLD_SIZE"] = "1"  # Single node
                    os.environ["RANK"] = "0"  # Master rank
                    # Prevent Ray Train from using distributed setup
                    os.environ["RAY_TRAIN_ENABLE_WORKER_SPREAD"] = "0"
                    os.environ["RAY_TRAIN_ENABLE_WORKER_CACHING"] = "0"
                    # Disable Ray Train distributed backend
                    os.environ["RAY_TRAIN_ENABLE_NCCL"] = "0"
                    # Disable PyTorch c10d (collective communications) to prevent Kubernetes connections
                    os.environ["TORCH_DISTRIBUTED_INIT_METHOD"] = "tcp://127.0.0.1:29500"
                    os.environ["TORCH_DISTRIBUTED_WORLD_SIZE"] = "1"
                    os.environ["TORCH_DISTRIBUTED_RANK"] = "0"
                    ray.init(local_mode=True, ignore_reinit_error=True, num_cpus=8, include_dashboard=False)
                    logger.info("Ray initialized in local mode (no Kubernetes, no distributed training)")
                else:
                    # On non-Windows, still default to local mode unless explicitly configured
                    ray.init(local_mode=True, ignore_reinit_error=True, include_dashboard=False)
                    logger.info("Ray initialized in local mode")
            
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
            
            # On Windows, ALWAYS use algorithm directly to avoid Ray Train/libuv issues
            # tune.Tuner uses Ray Train internally, which tries to initialize distributed training
            # even with 0 workers, causing libuv errors on Windows
            if is_windows:
                logger.info(f"Windows detected (sys.platform={sys.platform}): Using algorithm directly (bypassing tune.Tuner to avoid libuv)")
                logger.info("This avoids Ray Train's distributed initialization which requires libuv")
                try:
                    return self._train_direct(env, algo_config, callbacks, stop_criteria, experiment_name)
                except Exception as direct_error:
                    logger.error(f"Direct training failed: {direct_error}")
                    import traceback
                    traceback.print_exc()
                    raise
            
            # Non-Windows: Create tuner with advanced configuration
            logger.info(f"Non-Windows platform ({sys.platform}): Using tune.Tuner for advanced features")
            try:
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
                    logger.info("[SUCCESS] Advanced training completed successfully!")
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
                    
            except RuntimeError as e:
                # Catch libuv errors and fallback to direct training
                if "libuv" in str(e).lower() or "use_libuv" in str(e):
                    logger.warning(f"Ray Train failed with libuv error: {e}")
                    logger.info("Falling back to direct algorithm training (bypassing Ray Train)")
                    return self._train_direct(env, algo_config, callbacks, stop_criteria, experiment_name)
                else:
                    raise
                
        except Exception as e:
            logger.error(f"Advanced training failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _train_direct(self, env, algo_config, callbacks, stop_criteria, experiment_name) -> Dict[str, Any]:
        """Train algorithm directly without tune.Tuner (Windows fallback to avoid libuv)"""
        logger.info("Starting direct algorithm training...")
        
        try:
            # Build algorithm
            algorithm = algo_config.build()
            
            # Setup checkpoint directory (use absolute path for Ray compatibility)
            base_dir = Path.cwd()
            checkpoint_dir = base_dir / "checkpoints" / experiment_name
            checkpoint_dir = checkpoint_dir.resolve()  # Convert to absolute path
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Training parameters
            max_iterations = stop_criteria.get("training_iteration", self.config["training"]["max_iterations"])
            target_reward = stop_criteria.get("episode_reward_mean")
            checkpoint_freq = self.config["checkpointing"]["checkpoint_freq"]
            
            # Training state
            best_reward = -np.inf
            best_checkpoint = None
            training_metrics = []
            
            logger.info(f"Training for up to {max_iterations} iterations...")
            
            # Training loop
            for i in range(max_iterations):
                # Train one iteration
                result = algorithm.train()
                
                # Extract metrics
                episode_reward_mean = result.get("episode_reward_mean", 0.0)
                episode_len_mean = result.get("episode_len_mean", 0.0)
                custom_metrics = result.get("custom_metrics", {})
                
                # Update best reward
                if episode_reward_mean > best_reward:
                    best_reward = episode_reward_mean
                
                # Call callbacks manually (simulate tune callback interface)
                callback_metrics = {
                    "training_iteration": i + 1,
                    "episode_reward_mean": episode_reward_mean,
                    "episode_len_mean": episode_len_mean,
                    "custom_metrics": custom_metrics,
                    "time_total_s": result.get("time_total_s", 0)
                }
                
                # Call callbacks
                for callback in callbacks:
                    try:
                        # Simulate callback interface
                        if hasattr(callback, "on_trial_result"):
                            from ray.tune import Result
                            tune_result = Result(
                                metric_analysis={},
                                training_iteration=i + 1,
                                episode_reward_mean=episode_reward_mean,
                                episode_len_mean=episode_len_mean,
                                custom_metrics=custom_metrics,
                                time_total_s=result.get("time_total_s", 0)
                            )
                            callback.on_trial_result(trial=None, result=tune_result)
                    except Exception as callback_error:
                        logger.warning(f"Callback error (continuing): {callback_error}")
                
                # Log progress
                if (i + 1) % 10 == 0:
                    logger.info(f"Iteration {i+1}/{max_iterations}: "
                              f"reward={episode_reward_mean:.2f}, "
                              f"len={episode_len_mean:.2f}, "
                              f"best={best_reward:.2f}")
                
                # Save checkpoint periodically
                if (i + 1) % checkpoint_freq == 0:
                    checkpoint_name = f"checkpoint_{i+1}"
                    checkpoint_full_path = (checkpoint_dir / checkpoint_name).resolve()
                    checkpoint_path = algorithm.save(str(checkpoint_full_path))
                    logger.info(f"Saved checkpoint at iteration {i+1}: {checkpoint_path}")
                
                # Early stopping
                if target_reward and episode_reward_mean >= target_reward:
                    logger.info(f"Target reward achieved: {episode_reward_mean:.2f} >= {target_reward:.2f}")
                    break
                
                # Store metrics
                training_metrics.append(callback_metrics)
            
            # Final checkpoint
            final_checkpoint_path = checkpoint_dir / "checkpoint_final"
            final_checkpoint = algorithm.save(str(final_checkpoint_path.resolve()))
            logger.info(f"Saved final checkpoint: {final_checkpoint}")
            
            # Use metrics from last training iteration
            if training_metrics:
                final_metrics = training_metrics[-1].copy()
            else:
                # Fallback if no training occurred
                final_metrics = {
                    "training_iteration": 0,
                    "episode_reward_mean": 0.0,
                    "episode_len_mean": 0.0,
                    "custom_metrics": {},
                    "time_total_s": 0
                }
            
            logger.info("[SUCCESS] Direct algorithm training completed successfully!")
            logger.info(f"Best reward: {best_reward:.2f}")
            logger.info(f"Final checkpoint: {final_checkpoint}")
            
            # Save performance metrics
            try:
                self._save_performance_metrics_direct(final_metrics)
            except Exception as e:
                logger.warning(f"Failed to save performance metrics: {e}")
            
            return {
                "success": True,
                "best_result": final_metrics,
                "checkpoint": final_checkpoint,
                "experiment_name": experiment_name,
                "best_reward": best_reward,
                "iterations": i + 1
            }
            
        except Exception as e:
            logger.error(f"Direct training failed: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    def _save_performance_metrics_direct(self, metrics):
        """Save performance metrics for direct training"""
        # Update best metrics
        if "custom_metrics" in metrics:
            custom = metrics["custom_metrics"]
            if isinstance(custom, dict) and "sharpe_ratio" in custom:
                self.performance_metrics["best_sharpe_ratio"] = max(
                    self.performance_metrics["best_sharpe_ratio"],
                    custom["sharpe_ratio"]
                )
        
        if "episode_reward_mean" in metrics:
            self.performance_metrics["best_profit"] = max(
                self.performance_metrics["best_profit"],
                metrics["episode_reward_mean"]
            )
        
        # Save to file
        import json
        Path("results").mkdir(parents=True, exist_ok=True)
        with open("results/performance_metrics.json", "w") as f:
            json.dump(self.performance_metrics, f, indent=2, default=str)
    
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
            elif algorithm_name == "SAC":
                from ray.rllib.algorithms.sac import SAC
                algorithm = SAC.from_checkpoint(checkpoint_path)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm_name}. Only PPO and SAC are supported (A2C was deprecated in Ray 2.50.1)")
            
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
    # Create argument parser with prog name to avoid confusion with collect_training_data.py
    parser = argparse.ArgumentParser(
        description="WealthArena Advanced Trading Training",
        prog='train.py',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=str, default="config/production_config.yaml", help="Config file")
    parser.add_argument("--experiment", type=str, default="wealtharena_advanced", help="Experiment name")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate"], default="train", help="Mode")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path for evaluation")
    parser.add_argument("--local", action="store_true", help="Run locally")
    parser.add_argument("--max-iterations", type=int, help="Maximum training iterations")
    parser.add_argument("--target-reward", type=float, help="Target reward for early stopping")
    parser.add_argument("--num-workers", type=int, help="Number of workers")
    parser.add_argument("--address", type=str, help="Ray cluster address (e.g., 'auto', 'localhost:6379')")
    
    # Parse arguments explicitly, ensuring we use sys.argv from train.py execution
    args = parser.parse_args()
    
    # Log startup information
    logger.info("="*80)
    logger.info("WealthArena RL Training Started")
    logger.info(f"Config file: {args.config}")
    logger.info(f"Experiment name: {args.experiment}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Local mode: {args.local}")
    logger.info("="*80)
    
    # Initialize Ray
    try:
        # On Windows, ALWAYS use local mode (no libuv support)
        # On other platforms, default to local mode unless cluster address is explicitly provided
        is_windows = sys.platform == "win32"
        
        if is_windows or args.local or not hasattr(args, 'address') or not args.address:
            # Local mode: sequential execution, no distributed features
            num_cpus = getattr(args, 'num_workers', None) or 8
            logger.info(f"Initializing Ray in LOCAL mode with {num_cpus} CPUs...")
            logger.info("Local mode: No cluster connection, no Kubernetes, fully isolated")
            
            # Set environment variables to prevent any cluster auto-discovery
            os.environ["RAY_ADDRESS"] = ""  # Clear any existing cluster address
            os.environ["RAY_HEAD_NODE_IP"] = ""  # Clear head node IP
            os.environ["RAY_IGNORE_UNHANDLED_ERRORS"] = "1"
            
            ray.init(
                local_mode=True,
                num_cpus=num_cpus,
                ignore_reinit_error=True,
                _redis_password=None,  # Explicitly no Redis
                include_dashboard=False,  # Disable dashboard to avoid port conflicts
                dashboard_host="127.0.0.1"  # Only bind to localhost if dashboard is used
            )
            logger.info("Ray initialized successfully in local mode (no cluster, no Kubernetes)")
        else:
            # Cluster mode: Only if explicitly requested with --address
            cluster_address = args.address if hasattr(args, 'address') and args.address else None
            if cluster_address and cluster_address != "auto":
                logger.info(f"Initializing Ray with explicit cluster address: {cluster_address}")
                ray.init(address=cluster_address, ignore_reinit_error=True)
            else:
                # Default to local mode if address="auto" or invalid
                logger.warning("Cluster mode requested but no valid address provided. Using local mode instead.")
                logger.info("Initializing Ray in local mode with 8 CPUs...")
                ray.init(local_mode=True, num_cpus=8, ignore_reinit_error=True)
                logger.info("Ray initialized successfully in local mode")
    except Exception as e:
        logger.error(f"Failed to initialize Ray: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Create trainer
    try:
        logger.info("Creating AdvancedTradingTrainer instance...")
        trainer = AdvancedTradingTrainer(args.config)
        
        # Apply CLI overrides to config if provided
        if args.max_iterations is not None:
            trainer.config["training"]["max_iterations"] = args.max_iterations
        if args.target_reward is not None:
            trainer.config["training"]["target_reward"] = args.target_reward
        if args.num_workers is not None:
            trainer.config["resources"]["num_workers"] = args.num_workers
        
        logger.info("Trainer created successfully")
    except Exception as e:
        logger.error(f"Failed to create trainer (configuration issue?): {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    try:
        if args.mode == "train":
            # Run training
            logger.info("Starting training mode...")
            results = trainer.train(args.experiment)
            
            if results["success"]:
                logger.info("[OK] Advanced training completed successfully!")
                logger.info(f"Best checkpoint: {results['checkpoint']}")
                print("[OK] Advanced training completed successfully!")
                print(f"Best checkpoint: {results['checkpoint']}")
            else:
                logger.error(f"[FAILED] Training failed: {results.get('error', 'Unknown error')}")
                print(f"[FAILED] Training failed: {results['error']}")
                sys.exit(1)
        
        elif args.mode == "evaluate":
            if not args.checkpoint:
                logger.error("Checkpoint path required for evaluation")
                print("[FAILED] Checkpoint path required for evaluation")
                sys.exit(1)
            
            # Run evaluation
            logger.info("Starting evaluation mode...")
            results = trainer.evaluate_performance(args.checkpoint)
            
            if "error" not in results:
                logger.info("[OK] Performance evaluation completed!")
                logger.info("Results saved to: results/performance_evaluation.json")
                print("[OK] Performance evaluation completed!")
                print(f"Results saved to: results/performance_evaluation.json")
            else:
                logger.error(f"[FAILED] Evaluation failed: {results['error']}")
                print(f"[FAILED] Evaluation failed: {results['error']}")
                sys.exit(1)
    
    except Exception as e:
        logger.error(f"Unexpected error during training/evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        logger.info("Shutting down Ray...")
        ray.shutdown()
        logger.info("Ray shutdown complete")
    
    logger.info("="*80)
    logger.info("WealthArena RL Training Completed Successfully")
    logger.info("="*80)
    sys.exit(0)


if __name__ == "__main__":
    main()