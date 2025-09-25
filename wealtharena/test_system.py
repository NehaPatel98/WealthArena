#!/usr/bin/env python3
"""
WealthArena Trading System - Comprehensive Test

This script tests the complete WealthArena trading system to ensure
all components work correctly and the system is ready for production.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import traceback

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/system_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SystemTester:
    """Comprehensive system tester for WealthArena"""
    
    def __init__(self):
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "overall_status": "unknown",
            "summary": {}
        }
        
        # Create logs directory
        Path("logs").mkdir(exist_ok=True)
        
        logger.info("SystemTester initialized")
    
    def test_imports(self) -> dict:
        """Test all critical imports"""
        logger.info("Testing imports...")
        
        import_tests = [
            ("numpy", "import numpy as np"),
            ("pandas", "import pandas as pd"),
            ("gymnasium", "import gymnasium as gym"),
            ("ray", "import ray"),
            ("yfinance", "import yfinance as yf"),
            ("sklearn", "from sklearn.preprocessing import StandardScaler"),
            ("matplotlib", "import matplotlib.pyplot as plt"),
            ("seaborn", "import seaborn as sns")
        ]
        
        results = []
        missing_critical = []
        
        for module_name, import_statement in import_tests:
            try:
                exec(import_statement)
                results.append({"module": module_name, "status": "available"})
                logger.info(f"✅ {module_name} imported successfully")
            except ImportError as e:
                results.append({"module": module_name, "status": "missing", "error": str(e)})
                if module_name in ["numpy", "pandas", "gymnasium", "ray"]:
                    missing_critical.append(module_name)
                logger.warning(f"⚠️  {module_name} not available: {e}")
        
        return {
            "status": "pass" if len(missing_critical) == 0 else "fail",
            "total_modules": len(import_tests),
            "available_modules": len([r for r in results if r["status"] == "available"]),
            "missing_critical": missing_critical,
            "results": results
        }
    
    def test_environment_creation(self) -> dict:
        """Test trading environment creation"""
        logger.info("Testing environment creation...")
        
        try:
            from environments.trading_env import WealthArenaTradingEnv
            
            # Test with minimal config
            config = {
                "num_assets": 5,
                "initial_cash": 100000,
                "episode_length": 50,
                "use_real_data": False
            }
            
            env = WealthArenaTradingEnv(config)
            
            # Test reset
            obs, info = env.reset()
            
            # Test step
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Test properties
            assert obs.shape == env.observation_space.shape
            assert isinstance(reward, (int, float))
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)
            
            logger.info("✅ Environment creation test passed")
            return {
                "status": "pass",
                "obs_shape": obs.shape,
                "action_space": str(env.action_space),
                "observation_space": str(env.observation_space)
            }
            
        except Exception as e:
            logger.error(f"❌ Environment creation test failed: {e}")
            return {
                "status": "fail",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def test_portfolio_management(self) -> dict:
        """Test portfolio management system"""
        logger.info("Testing portfolio management...")
        
        try:
            from models.portfolio_manager import Portfolio, PortfolioManager
            
            # Test portfolio creation
            portfolio = Portfolio(100000)
            
            # Test trade execution
            success = portfolio.execute_trade("AAPL", 0.1, 150.0)
            assert success, "Trade execution failed"
            
            # Test portfolio value calculation
            current_prices = {"AAPL": 155.0}
            value = portfolio.get_portfolio_value(current_prices)
            assert value > 0, "Portfolio value should be positive"
            
            # Test performance metrics
            portfolio.update_performance(current_prices)
            metrics = portfolio.calculate_performance_metrics()
            assert isinstance(metrics, dict), "Performance metrics should be a dict"
            
            # Test portfolio manager
            manager = PortfolioManager(3, 100000)
            assert len(manager.portfolios) == 3, "Should have 3 portfolios"
            
            logger.info("✅ Portfolio management test passed")
            return {
                "status": "pass",
                "portfolio_value": value,
                "metrics_keys": list(metrics.keys())
            }
            
        except Exception as e:
            logger.error(f"❌ Portfolio management test failed: {e}")
            return {
                "status": "fail",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def test_data_adapter(self) -> dict:
        """Test data adapter functionality"""
        logger.info("Testing data adapter...")
        
        try:
            from data.data_adapter import DataAdapter
            
            # Test data adapter creation
            config = {
                "api": {"base_url": "test", "api_key": "test"},
                "cache": {"enabled": False}
            }
            
            adapter = DataAdapter(config)
            
            # Test yfinance fallback
            import yfinance as yf
            ticker = yf.Ticker("AAPL")
            data = ticker.history(period="5d")
            
            assert not data.empty, "Should get data from yfinance"
            
            logger.info("✅ Data adapter test passed")
            return {
                "status": "pass",
                "data_shape": data.shape,
                "data_columns": list(data.columns)
            }
            
        except Exception as e:
            logger.error(f"❌ Data adapter test failed: {e}")
            return {
                "status": "fail",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def test_multi_agent_environment(self) -> dict:
        """Test multi-agent environment"""
        logger.info("Testing multi-agent environment...")
        
        try:
            from environments.multi_agent_env import WealthArenaMultiAgentEnv
            
            # Test multi-agent environment creation
            config = {
                "num_agents": 3,
                "num_assets": 5,
                "episode_length": 50,
                "use_real_data": False
            }
            
            env = WealthArenaMultiAgentEnv(config)
            
            # Test reset
            obs, info = env.reset()
            assert len(obs) == 3, "Should have 3 agent observations"
            
            # Test step
            actions = {agent_id: env.action_space.sample() for agent_id in env.agent_ids}
            obs, rewards, terminateds, truncateds, infos = env.step(actions)
            
            assert len(rewards) == 3, "Should have 3 agent rewards"
            assert len(terminateds) == 4, "Should have 3 agent + 1 global termination"
            
            logger.info("✅ Multi-agent environment test passed")
            return {
                "status": "pass",
                "num_agents": len(obs),
                "agent_ids": env.agent_ids
            }
            
        except Exception as e:
            logger.error(f"❌ Multi-agent environment test failed: {e}")
            return {
                "status": "fail",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def test_training_pipeline(self) -> dict:
        """Test training pipeline components"""
        logger.info("Testing training pipeline...")
        
        try:
            from training.train_agents import AdvancedTradingTrainer
            
            # Test trainer creation
            trainer = AdvancedTradingTrainer()
            
            # Test configuration loading
            config = trainer.config
            assert "environment" in config, "Should have environment config"
            assert "training" in config, "Should have training config"
            
            # Test environment creation
            env = trainer.create_optimized_environment()
            assert env is not None, "Should create environment"
            
            logger.info("✅ Training pipeline test passed")
            return {
                "status": "pass",
                "config_sections": list(config.keys()),
                "environment_created": env is not None
            }
            
        except Exception as e:
            logger.error(f"❌ Training pipeline test failed: {e}")
            return {
                "status": "fail",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def test_data_download(self) -> dict:
        """Test data download functionality"""
        logger.info("Testing data download...")
        
        try:
            # Test if data download script exists and is importable
            from download_market_data import DataDownloader
            
            # Test data downloader creation
            config = {
                "symbols": ["AAPL", "GOOGL"],
                "start_date": "2023-01-01",
                "end_date": "2023-12-31"
            }
            
            downloader = DataDownloader(config)
            
            # Test data download (small test)
            all_data = downloader.download_all_data()
            
            assert len(all_data) > 0, "Should download some data"
            
            logger.info("✅ Data download test passed")
            return {
                "status": "pass",
                "downloaded_symbols": len(all_data),
                "symbols": list(all_data.keys())
            }
            
        except Exception as e:
            logger.error(f"❌ Data download test failed: {e}")
            return {
                "status": "fail",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def test_performance_optimization(self) -> dict:
        """Test performance optimization features"""
        logger.info("Testing performance optimization...")
        
        try:
            from environments.trading_env import WealthArenaTradingEnv
            
            # Test with performance-focused config
            config = {
                "num_assets": 10,
                "initial_cash": 1000000,
                "episode_length": 100,
                "use_real_data": False,
                "reward_weights": {
                    "profit": 2.0,
                    "risk": 0.5,
                    "cost": 0.1,
                    "stability": 0.05,
                    "sharpe": 1.0,
                    "momentum": 0.3,
                    "diversification": 0.2
                }
            }
            
            env = WealthArenaTradingEnv(config)
            
            # Test multiple episodes for performance
            total_rewards = []
            for episode in range(5):
                obs, info = env.reset()
                episode_reward = 0
                
                for step in range(20):
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    
                    if terminated or truncated:
                        break
                
                total_rewards.append(episode_reward)
            
            avg_reward = np.mean(total_rewards)
            
            logger.info("✅ Performance optimization test passed")
            return {
                "status": "pass",
                "avg_reward": avg_reward,
                "episodes_tested": len(total_rewards),
                "rewards": total_rewards
            }
            
        except Exception as e:
            logger.error(f"❌ Performance optimization test failed: {e}")
            return {
                "status": "fail",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def run_all_tests(self) -> dict:
        """Run all system tests"""
        logger.info("🚀 Starting comprehensive system tests...")
        
        # Define test functions
        tests = {
            "imports": self.test_imports,
            "environment_creation": self.test_environment_creation,
            "portfolio_management": self.test_portfolio_management,
            "data_adapter": self.test_data_adapter,
            "multi_agent_environment": self.test_multi_agent_environment,
            "training_pipeline": self.test_training_pipeline,
            "data_download": self.test_data_download,
            "performance_optimization": self.test_performance_optimization
        }
        
        # Run tests
        for test_name, test_func in tests.items():
            logger.info(f"Running {test_name} test...")
            try:
                result = test_func()
                self.test_results["tests"][test_name] = result
            except Exception as e:
                logger.error(f"Test {test_name} crashed: {e}")
                self.test_results["tests"][test_name] = {
                    "status": "error",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
        
        # Calculate overall status
        passed_tests = len([t for t in self.test_results["tests"].values() if t.get("status") == "pass"])
        total_tests = len(self.test_results["tests"])
        
        if passed_tests == total_tests:
            self.test_results["overall_status"] = "success"
        elif passed_tests >= total_tests * 0.8:
            self.test_results["overall_status"] = "partial"
        else:
            self.test_results["overall_status"] = "failed"
        
        # Create summary
        self.test_results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": (passed_tests / total_tests) * 100
        }
        
        return self.test_results
    
    def generate_report(self) -> str:
        """Generate comprehensive test report"""
        report = []
        report.append("=" * 80)
        report.append("WEALTHARENA TRADING SYSTEM - COMPREHENSIVE TEST REPORT")
        report.append("=" * 80)
        report.append(f"Timestamp: {self.test_results['timestamp']}")
        report.append(f"Overall Status: {self.test_results['overall_status'].upper()}")
        report.append("")
        
        # Summary
        summary = self.test_results["summary"]
        report.append("SUMMARY:")
        report.append(f"  Total Tests: {summary['total_tests']}")
        report.append(f"  Passed: {summary['passed_tests']}")
        report.append(f"  Failed: {summary['failed_tests']}")
        report.append(f"  Success Rate: {summary['success_rate']:.1f}%")
        report.append("")
        
        # Detailed results
        report.append("DETAILED RESULTS:")
        for test_name, result in self.test_results["tests"].items():
            status_icon = "✅" if result.get("status") == "pass" else "❌" if result.get("status") == "fail" else "⚠️"
            report.append(f"  {status_icon} {test_name.replace('_', ' ').title()}: {result.get('status', 'unknown')}")
            
            if result.get("status") == "fail" and "error" in result:
                report.append(f"    Error: {result['error']}")
        
        report.append("")
        
        # Recommendations
        if self.test_results["overall_status"] == "success":
            report.append("🎉 ALL TESTS PASSED!")
            report.append("The WealthArena Trading System is ready for production use.")
            report.append("")
            report.append("Next steps:")
            report.append("1. Run: python download_data.py")
            report.append("2. Run: python train.py")
            report.append("3. Monitor performance and optimize as needed")
        elif self.test_results["overall_status"] == "partial":
            report.append("⚠️  MOSTLY SUCCESSFUL")
            report.append("The system is mostly ready but has some issues to address.")
            report.append("Check the failed tests above and fix any issues.")
        else:
            report.append("❌ MULTIPLE FAILURES")
            report.append("The system needs significant work before it's ready.")
            report.append("Please address all failed tests before proceeding.")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """Main test function"""
    print("🚀 Starting WealthArena Trading System Tests...")
    
    # Create tester
    tester = SystemTester()
    
    try:
        # Run all tests
        results = tester.run_all_tests()
        
        # Generate report
        report = tester.generate_report()
        
        # Print report
        print("\n" + report)
        
        # Save results
        import json
        with open("test_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        with open("test_report.txt", "w") as f:
            f.write(report)
        
        print(f"\n📋 Test results saved to:")
        print(f"  - test_results.json")
        print(f"  - test_report.txt")
        print(f"  - logs/system_test.log")
        
        # Return appropriate exit code
        if results["overall_status"] == "success":
            print("\n🎉 All tests passed! System is ready for production!")
            return 0
        elif results["overall_status"] == "partial":
            print("\n⚠️  Some tests failed, but system is mostly ready")
            return 1
        else:
            print("\n❌ Multiple tests failed, system needs work")
            return 2
            
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        print(traceback.format_exc())
        return 3


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
