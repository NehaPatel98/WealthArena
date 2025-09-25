#!/usr/bin/env python3
"""
WealthArena Trading System - Environment Setup

This script sets up the complete Python environment for the WealthArena
multi-agent trading system with all necessary dependencies.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnvironmentSetup:
    """Setup and configure the WealthArena trading environment"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.platform = platform.system().lower()
        
        # Create necessary directories
        self.directories = [
            "data/raw",
            "data/processed", 
            "logs",
            "checkpoints",
            "results",
            "artifacts",
            "models"
        ]
        
        logger.info(f"Setting up WealthArena environment for Python {self.python_version} on {self.platform}")
    
    def create_directories(self):
        """Create necessary directories"""
        logger.info("Creating project directories...")
        
        for directory in self.directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def install_dependencies(self):
        """Install all required dependencies"""
        logger.info("Installing dependencies...")
        
        # Core dependencies
        core_packages = [
            "ray[rllib]==2.8.0",
            "torch>=1.13.0",
            "numpy>=1.21.0",
            "pandas>=1.5.0",
            "gymnasium>=0.29.0",
            "pyyaml>=6.0",
            "scikit-learn>=1.3.0",
            "matplotlib>=3.6.0",
            "seaborn>=0.12.0",
            "yfinance>=0.2.18",
            "aiohttp>=3.8.0",
            "websockets>=11.0.0",
            "scipy>=1.9.0",
            "redis>=4.5.0",
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0"
        ]
        
        # Experiment tracking
        tracking_packages = [
            "wandb>=0.15.0",
            "mlflow>=2.7.0"
        ]
        
        # Development tools
        dev_packages = [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=5.0.0",
            "jupyter>=1.0.0"
        ]
        
        all_packages = core_packages + tracking_packages + dev_packages
        
        for package in all_packages:
            try:
                logger.info(f"Installing {package}...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", package, "--upgrade"
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                logger.info(f"✅ Installed {package}")
            except subprocess.CalledProcessError as e:
                logger.warning(f"⚠️  Failed to install {package}: {e}")
        
        # Try to install TA-Lib (optional but recommended)
        try:
            logger.info("Installing TA-Lib (optional)...")
            if self.platform == "windows":
                # For Windows, try to install from a wheel
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "TA-Lib"
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                # For Linux/Mac, install from source
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "TA-Lib"
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info("✅ Installed TA-Lib")
        except subprocess.CalledProcessError:
            logger.warning("⚠️  TA-Lib installation failed (optional dependency)")
    
    def create_configuration(self):
        """Create production configuration files"""
        logger.info("Creating configuration files...")
        
        # Main configuration
        config = {
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
                "num_gpus": 1 if self.platform == "linux" else 0
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
        
        # Save main config
        import yaml
        with open("config/production_config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        logger.info("✅ Created production configuration")
    
    def create_startup_scripts(self):
        """Create startup and utility scripts"""
        logger.info("Creating startup scripts...")
        
        # Main training script
        train_script = '''#!/usr/bin/env python3
"""
WealthArena Trading System - Main Training Script
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from training.train_agents import main

if __name__ == "__main__":
    main()
'''
        
        with open("train.py", "w") as f:
            f.write(train_script)
        
        # Data download script
        download_script = '''#!/usr/bin/env python3
"""
WealthArena Trading System - Data Download Script
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from download_market_data import main

if __name__ == "__main__":
    main()
'''
        
        with open("download_data.py", "w") as f:
            f.write(download_script)
        
        # Quick test script
        test_script = '''#!/usr/bin/env python3
"""
WealthArena Trading System - Quick Test Script
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

def test_basic_functionality():
    """Test basic system functionality"""
    print("🧪 Testing WealthArena Trading System...")
    
    try:
        # Test imports
        from environments.trading_env import WealthArenaTradingEnv
        from models.portfolio_manager import Portfolio
        from data.data_adapter import DataAdapter
        print("✅ Core imports successful")
        
        # Test portfolio
        portfolio = Portfolio(100000)
        portfolio.execute_trade("AAPL", 0.1, 150.0)
        print("✅ Portfolio management working")
        
        # Test environment
        env_config = {
            "num_assets": 5,
            "initial_cash": 100000,
            "episode_length": 10,
            "use_real_data": False
        }
        env = WealthArenaTradingEnv(env_config)
        obs, info = env.reset()
        print(f"✅ Environment working: obs shape {obs.shape}")
        
        print("🎉 All tests passed! System is ready.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_basic_functionality()
'''
        
        with open("test_system.py", "w") as f:
            f.write(test_script)
        
        logger.info("✅ Created startup scripts")
    
    def setup_logging(self):
        """Setup logging configuration"""
        logger.info("Setting up logging...")
        
        log_config = '''version: 1
disable_existing_loggers: false

formatters:
  standard:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  detailed:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: standard
    stream: ext://sys.stdout
  
  file:
    class: logging.handlers.RotatingFileHandler
    level: DEBUG
    formatter: detailed
    filename: logs/wealtharena.log
    maxBytes: 10485760
    backupCount: 5
  
  training:
    class: logging.handlers.RotatingFileHandler
    level: INFO
    formatter: detailed
    filename: logs/training.log
    maxBytes: 10485760
    backupCount: 10

loggers:
  wealtharena:
    level: DEBUG
    handlers: [console, file]
    propagate: false
  
  training:
    level: INFO
    handlers: [console, training]
    propagate: false

root:
  level: INFO
  handlers: [console]
'''
        
        with open("logging_config.yaml", "w") as f:
            f.write(log_config)
        
        logger.info("✅ Logging configuration created")
    
    def run_setup(self):
        """Run complete environment setup"""
        logger.info("🚀 Starting WealthArena environment setup...")
        
        try:
            self.create_directories()
            self.install_dependencies()
            self.create_configuration()
            self.create_startup_scripts()
            self.setup_logging()
            
            logger.info("🎉 Environment setup completed successfully!")
            logger.info("\nNext steps:")
            logger.info("1. Run: python test_system.py")
            logger.info("2. Run: python download_data.py")
            logger.info("3. Run: python train.py")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            return False


def main():
    """Main setup function"""
    setup = EnvironmentSetup()
    success = setup.run_setup()
    
    if success:
        print("\n🎉 WealthArena Trading System setup completed!")
        print("The system is ready for high-performance trading.")
    else:
        print("\n❌ Setup failed. Please check the logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
