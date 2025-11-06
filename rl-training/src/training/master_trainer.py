"""
Master Trainer for All Financial Instrument Agents

This module provides a comprehensive training and evaluation system for all
specialized agents with performance comparison and reporting.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import logging
from datetime import datetime
from pathlib import Path
import json
import yaml
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import shutil
import ray

# Import all specialized trainers
from .asx_stocks_trainer import ASXStocksTrainer, ASXStocksConfig
from .currency_pairs_trainer import CurrencyPairsTrainer, CurrencyPairsConfig
from .cryptocurrency_trainer import CryptocurrencyTrainer, CryptocurrencyConfig
from .etf_trainer import ETFTrainer, ETFConfig
from .commodities_trainer import CommoditiesTrainer, CommoditiesConfig
from .train_agents import AdvancedTradingTrainer

# Import data sources
from ..data.asx.asx_symbols import get_asx_200_symbols
from ..data.currencies.currency_pairs import get_currency_pairs_by_category
from ..data.crypto.cryptocurrencies import get_major_cryptocurrencies
from ..data.commodities.commodities import get_high_volatility_commodities
from ..utils.data_loader import load_processed_data, prepare_training_data

logger = logging.getLogger(__name__)


@dataclass
class MasterConfig:
    """Configuration for master trainer"""
    start_date: str = "2015-01-01"
    end_date: str = "2025-09-26"
    output_dir: str = "results/master_comparison"
    config_file: str = None
    use_real_data: bool = True
    target_win_rate: float = 0.70
    quick_mode: bool = False
    
    # Agent configurations
    asx_stocks_count: int = 30
    currency_pairs_count: int = 10
    crypto_count: int = 12
    etf_count: int = 20
    commodities_count: int = 15


class MasterTrainer:
    """Master trainer for all financial instrument agents"""
    
    def __init__(self, config: MasterConfig):
        self.config = config
        self.trainers = {}
        self.results = {}
        self.comparison_metrics = {}
        self.training_config = None
        
        # Load training config from file if provided
        if config.config_file:
            self._load_training_config(config.config_file)
        
        # Initialize all trainers
        self._initialize_trainers()
        
        logger.info("Master Trainer initialized with all specialized agents")
    
    def _load_training_config(self, config_path: str):
        """Load training configuration from YAML file"""
        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return
        
        with open(config_file, 'r') as f:
            self.training_config = yaml.safe_load(f)
        
        logger.info(f"Loaded training config from {config_path}")
    
    def _initialize_trainers(self):
        """Initialize all specialized trainers"""
        
        # ASX Stocks Trainer
        asx_config = ASXStocksConfig(
            symbols=get_asx_200_symbols()[:self.config.asx_stocks_count],
            start_date=self.config.start_date,
            end_date=self.config.end_date
        )
        self.trainers['asx_stocks'] = ASXStocksTrainer(asx_config)
        
        # Currency Pairs Trainer
        currency_config = CurrencyPairsConfig(
            currency_pairs=get_currency_pairs_by_category("Major_Pairs")[:self.config.currency_pairs_count],
            start_date=self.config.start_date,
            end_date=self.config.end_date
        )
        self.trainers['currency_pairs'] = CurrencyPairsTrainer(currency_config)
        
        # Cryptocurrency Trainer
        crypto_config = CryptocurrencyConfig(
            symbols=get_major_cryptocurrencies()[:self.config.crypto_count],
            start_date=self.config.start_date,
            end_date=self.config.end_date
        )
        self.trainers['cryptocurrencies'] = CryptocurrencyTrainer(crypto_config)
        
        # ETF Trainer
        etf_config = ETFConfig(
            symbols=[
                "SPY", "QQQ", "IWM", "VTI", "VEA", "VWO", "BND", "TLT", "GLD", "SLV",
                "XLF", "XLK", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLB"
            ][:self.config.etf_count],
            start_date=self.config.start_date,
            end_date=self.config.end_date
        )
        self.trainers['etf'] = ETFTrainer(etf_config)
        
        # Commodities Trainer
        commodities_config = CommoditiesConfig(
            symbols=get_high_volatility_commodities()[:self.config.commodities_count],
            start_date=self.config.start_date,
            end_date=self.config.end_date
        )
        self.trainers['commodities'] = CommoditiesTrainer(commodities_config)
    
    def train_all_agents(self) -> Dict[str, Any]:
        """Train all specialized agents"""
        logger.info("Starting training for all agents...")
        
        training_results = {}
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            local_mode = getattr(self.config, 'local_mode', False)
            num_cpus = getattr(self.config, 'num_cpus', 8)
            ray.init(local_mode=local_mode, num_cpus=num_cpus, ignore_reinit_error=True)
            logger.info(f"Ray initialized with local_mode={local_mode}, num_cpus={num_cpus}")
        
        for agent_name, trainer in self.trainers.items():
            print(f"\n[STARTING] Training {agent_name.replace('_', ' ').title()} Agent...")
            
            try:
                # Pass training config if available
                if self.training_config:
                    result = trainer.train_agent(training_config=self.training_config)
                else:
                    result = trainer.train_agent()
                
                training_results[agent_name] = result
                win_rate = result.get('win_rate', 0.0)
                final_reward = result.get('final_reward', 0.0)
                
                print(f"[OK] {agent_name} training completed. Final reward: {final_reward:.4f}, Win rate: {win_rate:.2%}")
                
                # Validate win rate
                if win_rate < self.config.target_win_rate:
                    logger.warning(f"{agent_name} win rate {win_rate:.2%} below target {self.config.target_win_rate:.2%}")
                
            except Exception as e:
                logger.error(f"Error training {agent_name}: {e}", exc_info=True)
                training_results[agent_name] = {"error": str(e)}
                print(f"[FAILED] {agent_name} training failed: {e}")
        
        self.results['training'] = training_results
        logger.info("All agents training completed")
        
        return training_results
    
    def backtest_all_agents(self) -> Dict[str, Any]:
        """Backtest all specialized agents"""
        logger.info("Starting backtest for all agents...")
        
        backtest_results = {}
        
        for agent_name, trainer in self.trainers.items():
            print(f"\n[SUMMARY] Running backtest for {agent_name.replace('_', ' ').title()} Agent...")
            
            try:
                result = trainer.backtest_agent()
                backtest_results[agent_name] = result
                print(f"[OK] {agent_name} backtest completed")
                
            except Exception as e:
                logger.error(f"Error backtesting {agent_name}: {e}")
                backtest_results[agent_name] = {"error": str(e)}
                print(f"[FAILED] {agent_name} backtest failed: {e}")
        
        self.results['backtest'] = backtest_results
        logger.info("All agents backtest completed")
        
        return backtest_results
    
    def generate_comparison_report(self) -> Dict[str, Any]:
        """Generate comprehensive comparison report"""
        logger.info("Generating comparison report...")
        
        if not self.results.get('backtest'):
            logger.warning("No backtest results available. Run backtest_all_agents() first.")
            return {}
        
        # Extract metrics for comparison
        comparison_data = {}
        
        for agent_name, backtest_result in self.results['backtest'].items():
            if 'error' in backtest_result:
                continue
                
            metrics = backtest_result.get('metrics', {})
            
            comparison_data[agent_name] = {
                'total_return': metrics.get('returns', {}).get('total_return', 0),
                'annual_return': metrics.get('returns', {}).get('annual_return', 0),
                'volatility': metrics.get('returns', {}).get('volatility', 0),
                'sharpe_ratio': metrics.get('risk', {}).get('sharpe_ratio', 0),
                'max_drawdown': metrics.get('risk', {}).get('max_drawdown', 0),
                'win_rate': metrics.get('trading', {}).get('win_rate', 0),
                'profit_factor': metrics.get('trading', {}).get('profit_factor', 0)
            }
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data).T
        
        # Calculate rankings
        rankings = {}
        for metric in ['annual_return', 'sharpe_ratio', 'win_rate', 'profit_factor']:
            rankings[metric] = comparison_df[metric].rank(ascending=False).to_dict()
        
        # Calculate composite score
        composite_scores = {}
        for agent_name in comparison_data.keys():
            score = 0
            score += rankings['annual_return'][agent_name] * 0.3
            score += rankings['sharpe_ratio'][agent_name] * 0.3
            score += rankings['win_rate'][agent_name] * 0.2
            score += rankings['profit_factor'][agent_name] * 0.2
            composite_scores[agent_name] = score
        
        # Sort by composite score
        sorted_agents = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Generate report
        report = {
            "comparison_summary": {
                "total_agents": len(comparison_data),
                "successful_agents": len([r for r in self.results['backtest'].values() if 'error' not in r]),
                "failed_agents": len([r for r in self.results['backtest'].values() if 'error' in r])
            },
            "performance_comparison": comparison_data,
            "rankings": rankings,
            "composite_scores": composite_scores,
            "agent_rankings": sorted_agents,
            "best_performing_agent": sorted_agents[0][0] if sorted_agents else None,
            "worst_performing_agent": sorted_agents[-1][0] if sorted_agents else None,
            "generated_at": datetime.now().isoformat()
        }
        
        self.comparison_metrics = report
        return report
    
    def print_comparison_summary(self):
        """Print a formatted comparison summary"""
        if not self.comparison_metrics:
            print("No comparison data available. Run generate_comparison_report() first.")
            return
        
        print("\n" + "="*80)
        print("MASTER AGENT COMPARISON RESULTS")
        print("="*80)
        
        # Summary
        summary = self.comparison_metrics['comparison_summary']
        print(f"\n[SUMMARY] SUMMARY:")
        print(f"  Total Agents: {summary['total_agents']}")
        print(f"  Successful: {summary['successful_agents']}")
        print(f"  Failed: {summary['failed_agents']}")
        
        # Rankings
        print(f"\n[RANKINGS] AGENT RANKINGS (by composite score):")
        for i, (agent_name, score) in enumerate(self.comparison_metrics['agent_rankings'], 1):
            print(f"  {i}. {agent_name.replace('_', ' ').title()}: {score:.2f}")
        
        # Performance comparison
        print(f"\n[BREAKDOWN] PERFORMANCE COMPARISON:")
        comparison = self.comparison_metrics['performance_comparison']
        
        # Create a formatted table
        print(f"{'Agent':<20} {'Return':<10} {'Sharpe':<8} {'Drawdown':<10} {'Win Rate':<10}")
        print("-" * 70)
        
        for agent_name, metrics in comparison.items():
            print(f"{agent_name.replace('_', ' ').title():<20} "
                  f"{metrics['annual_return']:<10.2%} "
                  f"{metrics['sharpe_ratio']:<8.3f} "
                  f"{metrics['max_drawdown']:<10.2%} "
                  f"{metrics['win_rate']:<10.2%}")
        
        # Best and worst
        best = self.comparison_metrics['best_performing_agent']
        worst = self.comparison_metrics['worst_performing_agent']
        
        if best:
            print(f"\n🥇 BEST PERFORMING AGENT: {best.replace('_', ' ').title()}")
        if worst:
            print(f"🥉 WORST PERFORMING AGENT: {worst.replace('_', ' ').title()}")
    
    def save_all_results(self):
        """Save all results and comparison data"""
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save individual agent results
        for agent_name, trainer in self.trainers.items():
            agent_dir = output_path / agent_name
            agent_dir.mkdir(exist_ok=True)
            
            # Save trainer results
            trainer.save_results(str(agent_dir))
        
        # Save comparison report
        if self.comparison_metrics:
            with open(output_path / "comparison_report.json", "w") as f:
                json.dump(self.comparison_metrics, f, indent=2, default=str)
        
        # Save master summary
        master_summary = {
            "config": {
                "start_date": self.config.start_date,
                "end_date": self.config.end_date,
                "agents_trained": list(self.trainers.keys())
            },
            "results": self.results,
            "comparison": self.comparison_metrics,
            "generated_at": datetime.now().isoformat()
        }
        
        with open(output_path / "master_summary.json", "w") as f:
            json.dump(master_summary, f, indent=2, default=str)
        
        logger.info(f"All results saved to {output_path}")
    
    def run_full_pipeline(self):
        """Run the complete training and evaluation pipeline"""
        from tqdm import tqdm
        
        print("[STARTING] Starting Master Training Pipeline...")
        print("="*60)
        
        # Step 1: Train all agents
        print("\n[STEP 1] STEP 1: Training All Agents")
        training_results = self.train_all_agents()
        
        # Step 2: Backtest all agents
        print("\n[SUMMARY] STEP 2: Backtesting All Agents")
        backtest_results = self.backtest_all_agents()
        
        # Step 3: Validate win rates
        print("\n[OK] STEP 3: Validating Win Rates")
        validation_results = self.validate_win_rates()
        
        # Step 4: Generate comparison report
        print("\n[BREAKDOWN] STEP 4: Generating Comparison Report")
        comparison_report = self.generate_comparison_report()
        
        # Step 5: Print summary
        print("\n[STEP 5] STEP 5: Results Summary")
        self.print_comparison_summary()
        
        # Step 6: Save all results
        print("\n[SAVE] STEP 6: Saving Results")
        self.save_all_results()
        
        print("\n[OK] Master Training Pipeline Completed!")
        print(f"[PATH] Results saved to: {self.config.output_dir}")
        
        return {
            "training_results": training_results,
            "backtest_results": backtest_results,
            "validation_results": validation_results,
            "comparison_report": comparison_report
        }
    
    def validate_win_rates(self) -> Dict[str, bool]:
        """Validate that all agents meet win rate target"""
        validation_results = {}
        
        for agent_name, backtest_result in self.results.get('backtest', {}).items():
            if 'error' in backtest_result:
                validation_results[agent_name] = False
                continue
            
            metrics = backtest_result.get('metrics', {})
            win_rate = metrics.get('trading', {}).get('win_rate', 0.0)
            
            meets_target = win_rate >= self.config.target_win_rate
            validation_results[agent_name] = meets_target
            
            if meets_target:
                print(f"[OK] {agent_name}: Win rate {win_rate:.2%} >= target {self.config.target_win_rate:.2%}")
            else:
                print(f"[WARNING] {agent_name}: Win rate {win_rate:.2%} < target {self.config.target_win_rate:.2%}")
        
        return validation_results
    
    def deploy_best_checkpoints(self, target_dir: str = "../services/rl-service/models/latest"):
        """Copy best RLlib checkpoints to inference service directory"""
        import json
        
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)
        
        for agent_name in self.trainers.keys():
            checkpoint_dir = Path(f"checkpoints/{agent_name}")
            if not checkpoint_dir.exists():
                logger.warning(f"No checkpoint directory for {agent_name}")
                continue
            
            # Check for latest_checkpoint.json metadata file first
            metadata_file = checkpoint_dir / "latest_checkpoint.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    rllib_checkpoint_path = metadata.get("checkpoint_path")
                    
                    if rllib_checkpoint_path and Path(rllib_checkpoint_path).exists():
                        # Copy RLlib checkpoint directory
                        checkpoint_source = Path(rllib_checkpoint_path)
                        target_checkpoint_dir = target_path / f"{agent_name}_checkpoint"
                        
                        if target_checkpoint_dir.exists():
                            shutil.rmtree(target_checkpoint_dir)
                        shutil.copytree(checkpoint_source, target_checkpoint_dir)
                        
                        logger.info(f"Deployed {agent_name} RLlib checkpoint from {rllib_checkpoint_path} to {target_checkpoint_dir}")
                        continue
                except Exception as e:
                    logger.warning(f"Error reading metadata for {agent_name}: {e}")
            
            # Fallback: Look for RLlib checkpoint directories directly
            checkpoint_dirs = list(checkpoint_dir.glob("**/checkpoint_*"))
            checkpoint_dirs.extend(list(checkpoint_dir.glob("**/*_final")))
            
            if checkpoint_dirs:
                # Sort by modification time - get the most recent directory
                latest_checkpoint_dir = max(checkpoint_dirs, key=lambda p: p.stat().st_mtime if p.is_dir() else 0)
                
                if latest_checkpoint_dir.is_dir():
                    # Copy RLlib checkpoint directory
                    target_checkpoint_dir = target_path / f"{agent_name}_checkpoint"
                    
                    if target_checkpoint_dir.exists():
                        shutil.rmtree(target_checkpoint_dir)
                    shutil.copytree(latest_checkpoint_dir, target_checkpoint_dir)
                    
                    logger.info(f"Deployed {agent_name} RLlib checkpoint from {latest_checkpoint_dir} to {target_checkpoint_dir}")
                else:
                    logger.warning(f"Expected directory but found file: {latest_checkpoint_dir}")
            else:
                logger.warning(f"No RLlib checkpoint found for {agent_name}")


def main():
    """Main function to run master training pipeline with CLI support"""
    parser = argparse.ArgumentParser(description="WealthArena Master Trainer")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "evaluate"],
                       help="Mode: train or evaluate")
    parser.add_argument("--local", action="store_true",
                       help="Run in local mode (sequential execution)")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to training config YAML file")
    parser.add_argument("--asset-class", type=str, default="all",
                       choices=["all", "asx_stocks", "cryptocurrencies", "currency_pairs", "commodities", "etf"],
                       help="Asset class to train (default: all)")
    parser.add_argument("--output-dir", type=str, default="results/master_comparison",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/master_training.log'),
            logging.StreamHandler()
        ]
    )
    
    # Create master configuration
    config = MasterConfig(
        start_date="2022-01-01",
        end_date="2025-01-01",
        output_dir=args.output_dir,
        config_file=args.config,
        use_real_data=True,
        target_win_rate=0.70,
        quick_mode=(args.config and "quick" in args.config.lower())
    )
    
    # Set local mode flag
    config.local_mode = args.local
    config.num_cpus = 8 if not args.local else 1
    
    # Create master trainer
    master_trainer = MasterTrainer(config)
    
    # Run based on mode
    if args.mode == "train":
        if args.asset_class == "all":
            # Run full pipeline
            results = master_trainer.run_full_pipeline()
        else:
            # Train single asset class
            if args.asset_class in master_trainer.trainers:
                trainer = master_trainer.trainers[args.asset_class]
                if master_trainer.training_config:
                    result = trainer.train_agent(training_config=master_trainer.training_config)
                else:
                    result = trainer.train_agent()
                print(f"[OK] Training completed for {args.asset_class}")
            else:
                print(f"[FAILED] Unknown asset class: {args.asset_class}")
    
    elif args.mode == "evaluate":
        # Run backtesting and validation
        backtest_results = master_trainer.backtest_all_agents()
        validation_results = master_trainer.validate_win_rates()
        comparison_report = master_trainer.generate_comparison_report()
        master_trainer.print_comparison_summary()
        master_trainer.save_all_results()
        results = {
            "backtest_results": backtest_results,
            "validation_results": validation_results,
            "comparison_report": comparison_report
        }
    
    # Cleanup Ray if initialized
    if ray.is_initialized():
        ray.shutdown()
    
    return master_trainer, results


if __name__ == "__main__":
    master_trainer, results = main()
