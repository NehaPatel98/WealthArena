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

# Import all specialized trainers
from .asx_stocks_trainer import ASXStocksTrainer, ASXStocksConfig
from .currency_pairs_trainer import CurrencyPairsTrainer, CurrencyPairsConfig
from .cryptocurrency_trainer import CryptocurrencyTrainer, CryptocurrencyConfig
from .etf_trainer import ETFTrainer, ETFConfig
from .commodities_trainer import CommoditiesTrainer, CommoditiesConfig

# Import data sources
from ..data.asx.asx_symbols import get_asx_200_symbols
from ..data.currencies.currency_pairs import get_currency_pairs_by_category
from ..data.crypto.cryptocurrencies import get_major_cryptocurrencies
from ..data.commodities.commodities import get_high_volatility_commodities

logger = logging.getLogger(__name__)


@dataclass
class MasterConfig:
    """Configuration for master trainer"""
    start_date: str = "2015-01-01"
    end_date: str = "2025-09-26"
    output_dir: str = "results/master_comparison"
    
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
        
        # Initialize all trainers
        self._initialize_trainers()
        
        logger.info("Master Trainer initialized with all specialized agents")
    
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
        
        for agent_name, trainer in self.trainers.items():
            print(f"\n🚀 Training {agent_name.replace('_', ' ').title()} Agent...")
            
            try:
                result = trainer.train_agent()
                training_results[agent_name] = result
                print(f"✅ {agent_name} training completed. Final reward: {result['final_reward']:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {agent_name}: {e}")
                training_results[agent_name] = {"error": str(e)}
                print(f"❌ {agent_name} training failed: {e}")
        
        self.results['training'] = training_results
        logger.info("All agents training completed")
        
        return training_results
    
    def backtest_all_agents(self) -> Dict[str, Any]:
        """Backtest all specialized agents"""
        logger.info("Starting backtest for all agents...")
        
        backtest_results = {}
        
        for agent_name, trainer in self.trainers.items():
            print(f"\n📊 Running backtest for {agent_name.replace('_', ' ').title()} Agent...")
            
            try:
                result = trainer.backtest_agent()
                backtest_results[agent_name] = result
                print(f"✅ {agent_name} backtest completed")
                
            except Exception as e:
                logger.error(f"Error backtesting {agent_name}: {e}")
                backtest_results[agent_name] = {"error": str(e)}
                print(f"❌ {agent_name} backtest failed: {e}")
        
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
        print(f"\n📊 SUMMARY:")
        print(f"  Total Agents: {summary['total_agents']}")
        print(f"  Successful: {summary['successful_agents']}")
        print(f"  Failed: {summary['failed_agents']}")
        
        # Rankings
        print(f"\n🏆 AGENT RANKINGS (by composite score):")
        for i, (agent_name, score) in enumerate(self.comparison_metrics['agent_rankings'], 1):
            print(f"  {i}. {agent_name.replace('_', ' ').title()}: {score:.2f}")
        
        # Performance comparison
        print(f"\n📈 PERFORMANCE COMPARISON:")
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
        print("🚀 Starting Master Training Pipeline...")
        print("="*60)
        
        # Step 1: Train all agents
        print("\n📚 STEP 1: Training All Agents")
        training_results = self.train_all_agents()
        
        # Step 2: Backtest all agents
        print("\n📊 STEP 2: Backtesting All Agents")
        backtest_results = self.backtest_all_agents()
        
        # Step 3: Generate comparison report
        print("\n📈 STEP 3: Generating Comparison Report")
        comparison_report = self.generate_comparison_report()
        
        # Step 4: Print summary
        print("\n📋 STEP 4: Results Summary")
        self.print_comparison_summary()
        
        # Step 5: Save all results
        print("\n💾 STEP 5: Saving Results")
        self.save_all_results()
        
        print("\n✅ Master Training Pipeline Completed!")
        print(f"📁 Results saved to: {self.config.output_dir}")
        
        return {
            "training_results": training_results,
            "backtest_results": backtest_results,
            "comparison_report": comparison_report
        }


def main():
    """Main function to run master training pipeline"""
    logging.basicConfig(level=logging.INFO)
    
    # Create master configuration
    config = MasterConfig(
        start_date="2020-01-01",
        end_date="2024-01-01",
        output_dir="results/master_comparison"
    )
    
    # Create master trainer
    master_trainer = MasterTrainer(config)
    
    # Run full pipeline
    results = master_trainer.run_full_pipeline()
    
    return master_trainer, results


if __name__ == "__main__":
    master_trainer, results = main()
