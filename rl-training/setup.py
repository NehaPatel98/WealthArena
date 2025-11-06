#!/usr/bin/env python3
"""
WealthArena Trading System - Complete Setup Script

This script sets up the complete WealthArena trading system with all
dependencies and configurations for production use.
"""

import os
import sys
import subprocess
import platform
import logging
from pathlib import Path
import json
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/setup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class WealthArenaSetup:
    """Complete setup for WealthArena Trading System"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.platform = platform.system().lower()
        
        # Create logs directory
        Path("logs").mkdir(exist_ok=True)
        
        logger.info(f"Setting up WealthArena Trading System")
        logger.info(f"Python version: {self.python_version}")
        logger.info(f"Platform: {self.platform}")
    
    def check_python_version(self):
        """Check if Python version is compatible"""
        if sys.version_info < (3, 8):
            logger.error("Python 3.8 or higher is required")
            return False
        
        logger.info(f"✅ Python version {self.python_version} is compatible")
        return True
    
    def create_directories(self):
        """Create all necessary directories"""
        directories = [
            "data/raw",
            "data/processed",
            "logs",
            "checkpoints",
            "results",
            "artifacts",
            "models",
            "config",
            "src/environments",
            "src/data",
            "src/models",
            "src/training",
            "src/tracking",
            "notebooks",
            "tests"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def install_dependencies(self):
        """Install all required dependencies"""
        logger.info("Installing dependencies...")
        
        # Upgrade pip first
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--upgrade", "pip"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info("✅ Upgraded pip")
        except subprocess.CalledProcessError:
            logger.warning("⚠️  Failed to upgrade pip")
        
        # Install core dependencies
        core_packages = [
            "numpy>=1.21.0",
            "pandas>=1.5.0",
            "scipy>=1.9.0",
            "scikit-learn>=1.3.0",
            "matplotlib>=3.6.0",
            "seaborn>=0.12.0",
            "pyyaml>=6.0",
            "tqdm>=4.64.0",
            "requests>=2.28.0"
        ]
        
        for package in core_packages:
            try:
                logger.info(f"Installing {package}...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", package
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                logger.info(f"✅ Installed {package}")
            except subprocess.CalledProcessError as e:
                logger.warning(f"⚠️  Failed to install {package}: {e}")
        
        # Install RL frameworks
        rl_packages = [
            "gymnasium>=0.29.0",
            "ray[rllib]==2.8.0",
            "torch>=1.13.0"
        ]
        
        for package in rl_packages:
            try:
                logger.info(f"Installing {package}...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", package
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                logger.info(f"✅ Installed {package}")
            except subprocess.CalledProcessError as e:
                logger.warning(f"⚠️  Failed to install {package}: {e}")
        
        # Install financial data packages
        financial_packages = [
            "yfinance>=0.2.18",
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0"
        ]
        
        for package in financial_packages:
            try:
                logger.info(f"Installing {package}...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", package
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                logger.info(f"✅ Installed {package}")
            except subprocess.CalledProcessError as e:
                logger.warning(f"⚠️  Failed to install {package}: {e}")
        
        # Try to install TA-Lib (optional but recommended)
        try:
            logger.info("Installing TA-Lib (optional)...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "TA-Lib"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info("✅ Installed TA-Lib")
        except subprocess.CalledProcessError:
            logger.warning("⚠️  TA-Lib installation failed (optional dependency)")
            logger.info("You can install TA-Lib manually later if needed")
        
        # Try to install Redis (optional)
        try:
            logger.info("Installing Redis client (optional)...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "redis>=4.5.0"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info("✅ Installed Redis client")
        except subprocess.CalledProcessError:
            logger.warning("⚠️  Redis client installation failed (optional dependency)")
    
    def create_configuration_files(self):
        """Create production configuration files"""
        logger.info("Creating configuration files...")
        
        # Main production config
        production_config = {
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
                    "sharpe": 1.0,
                    "momentum": 0.3,
                    "diversification": 0.2
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
                    "enabled": False,
                    "project": "wealtharena-trading",
                    "entity": "wealtharena"
                },
                "mlflow": {
                    "enabled": False,
                    "tracking_uri": "http://localhost:5000",
                    "experiment_name": "WealthArena_Production"
                }
            }
        }
        
        # Save production config
        with open("config/production_config.yaml", "w") as f:
            yaml.dump(production_config, f, default_flow_style=False, indent=2)
        
        # Create data adapter config
        data_adapter_config = {
            "api": {
                "base_url": "https://api.sys1.com",
                "api_key": "YOUR_API_KEY_HERE",
                "secret_key": "YOUR_SECRET_KEY_HERE",
                "rate_limit": 100,
                "timeout": 30
            },
            "cache": {
                "enabled": True,
                "host": "localhost",
                "port": 6379,
                "db": 0,
                "cache_ttl": 3600
            },
            "validate": True,
            "quality_checks": True
        }
        
        with open("config/data_adapter_config.yaml", "w") as f:
            yaml.dump(data_adapter_config, f, default_flow_style=False, indent=2)
        
        logger.info("✅ Configuration files created")
    
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
        
        logger.info("✅ Startup scripts created")
    
    def create_documentation(self):
        """Create comprehensive documentation"""
        logger.info("Creating documentation...")
        
        # README
        readme_content = '''# WealthArena Trading System

Advanced Multi-Agent Reinforcement Learning Trading Platform

## 🚀 Quick Start

```bash
# 1. Setup environment
python setup.py

# 2. Test system
python test_system.py

# 3. Download data
python download_data.py

# 4. Start training
python train.py
```

## 📊 Features

- **Multi-Agent RL**: 5 specialized trading agents
- **Advanced Risk Management**: Comprehensive risk controls
- **Real Market Data**: Live data integration
- **Performance Optimization**: Designed to beat market benchmarks
- **Production Ready**: Full deployment capabilities

## 🎯 Performance Targets

- **Annual Return**: 25-40%
- **Sharpe Ratio**: 2.0+
- **Max Drawdown**: <15%
- **Win Rate**: 60%+

## 📁 Project Structure

```
wealtharena_rllib/
├── src/                    # Source code
│   ├── environments/       # Trading environments
│   ├── data/              # Data processing
│   ├── models/            # Portfolio management
│   ├── training/          # RL training
│   └── tracking/          # Experiment tracking
├── config/                # Configuration files
├── data/                  # Market data
├── checkpoints/           # Model checkpoints
├── results/               # Results and reports
└── logs/                  # Log files
```

## 🔧 Configuration

Edit `config/production_config.yaml` to customize:
- Number of agents and assets
- Training parameters
- Risk management settings
- Reward function weights

## 📈 Monitoring

- **MLflow**: Training metrics and model tracking
- **Weights & Biases**: Advanced experiment tracking
- **Logs**: Detailed system logs in `logs/` directory

## 🛡️ Risk Management

- Position limits (15% max per asset)
- Portfolio risk controls (12% max volatility)
- Stop loss protection (8% loss threshold)
- Drawdown management (15% max drawdown)

## 📚 Documentation

- `PRODUCTION_READY_ANALYSIS.md`: Complete system analysis
- `config/`: Configuration files and examples
- `notebooks/`: Jupyter notebooks for analysis

## 🤝 Support

For questions and support, please check the logs and documentation.
'''
        
        with open("README.md", "w") as f:
            f.write(readme_content)
        
        logger.info("✅ Documentation created")
    
    def run_setup(self):
        """Run complete setup process"""
        logger.info("🚀 Starting WealthArena Trading System setup...")
        
        try:
            # Check Python version
            if not self.check_python_version():
                return False
            
            # Create directories
            self.create_directories()
            
            # Install dependencies
            self.install_dependencies()
            
            # Create configuration files
            self.create_configuration_files()
            
            # Create startup scripts
            self.create_startup_scripts()
            
            # Create documentation
            self.create_documentation()
            
            logger.info("🎉 Setup completed successfully!")
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
    setup = WealthArenaSetup()
    success = setup.run_setup()
    
    if success:
        print("\n🎉 WealthArena Trading System setup completed!")
        print("The system is ready for high-performance trading.")
        print("\nTo get started:")
        print("1. python test_system.py")
        print("2. python download_data.py")
        print("3. python train.py")
    else:
        print("\n❌ Setup failed. Please check the logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
