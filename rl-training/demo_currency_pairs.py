#!/usr/bin/env python3
"""
Currency Pairs Agent Demo

This script demonstrates how to train and evaluate the Currency Pairs agent
with comprehensive metrics and backtesting.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.training.currency_pairs_trainer import CurrencyPairsTrainer, CurrencyPairsConfig
from src.data.currencies.currency_pairs import get_currency_pairs_by_category

def main():
    """Demo the Currency Pairs agent training and evaluation"""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 CURRENCY PAIRS AGENT DEMO")
    print("="*50)
    
    # Create configuration
    config = CurrencyPairsConfig(
        currency_pairs=get_currency_pairs_by_category("Major_Pairs")[:8],  # Top 8 pairs
        start_date="2015-01-01",
        end_date="2025-09-26"
    )
    
    print(f"📊 Configuration:")
    print(f"  Currency Pairs: {config.currency_pairs}")
    print(f"  Training Period: {config.start_date} to {config.end_date}")
    print(f"  Initial Cash: ${config.initial_cash:,.0f}")
    print(f"  Max Position Size: {config.max_position_size:.1%}")
    print(f"  Transaction Cost: {config.transaction_cost_rate:.4f}")
    
    # Create trainer
    trainer = CurrencyPairsTrainer(config)
    
    # Step 1: Train the agent
    print(f"\n🚀 STEP 1: Training Currency Pairs Agent...")
    print("-" * 40)
    
    training_results = trainer.train_agent()
    
    print(f"✅ Training Results:")
    print(f"  Episodes Trained: {training_results['episodes_trained']}")
    print(f"  Final Reward: {training_results['final_reward']:.4f}")
    print(f"  Convergence Episode: {training_results['convergence_episode']}")
    print(f"  Training Loss: {training_results['training_loss']:.4f}")
    print(f"  Validation Reward: {training_results['validation_reward']:.4f}")
    print(f"  Best Performance: {training_results['best_performance']:.4f}")
    
    # Step 2: Backtest the agent
    print(f"\n📊 STEP 2: Running Backtest...")
    print("-" * 40)
    
    backtest_results = trainer.backtest_agent()
    
    print(f"✅ Backtest Results:")
    print(f"  Portfolio Values: {len(backtest_results['portfolio_values'])} data points")
    print(f"  Returns: {len(backtest_results['returns'])} data points")
    print(f"  Positions History: {len(backtest_results['positions_history'])} data points")
    
    # Step 3: Generate evaluation report
    print(f"\n📈 STEP 3: Generating Evaluation Report...")
    print("-" * 40)
    
    report = trainer.generate_report()
    
    # Display detailed results
    print(f"\n📊 PERFORMANCE METRICS:")
    print("=" * 50)
    
    # Returns
    returns = report['evaluation_metrics']['returns']
    print(f"\n💰 RETURNS:")
    print(f"  Total Return: {returns['total_return']:.2%}")
    print(f"  Annual Return: {returns['annual_return']:.2%}")
    print(f"  Volatility: {returns['volatility']:.2%}")
    
    # Risk
    risk = report['evaluation_metrics']['risk']
    print(f"\n⚠️  RISK METRICS:")
    print(f"  Max Drawdown: {risk['max_drawdown']:.2%}")
    print(f"  Sharpe Ratio: {risk['sharpe_ratio']:.3f}")
    print(f"  Sortino Ratio: {risk['sortino_ratio']:.3f}")
    print(f"  Calmar Ratio: {risk['calmar_ratio']:.3f}")
    print(f"  VaR (95%): {risk['var_95']:.4f}")
    print(f"  CVaR (95%): {risk['cvar_95']:.4f}")
    
    # Trading
    trading = report['evaluation_metrics']['trading']
    print(f"\n🎯 TRADING METRICS:")
    print(f"  Win Rate: {trading['win_rate']:.2%}")
    print(f"  Profit Factor: {trading['profit_factor']:.3f}")
    print(f"  Recovery Factor: {trading['recovery_factor']:.3f}")
    print(f"  Stability: {trading['stability']:.3f}")
    print(f"  Diversification: {trading.get('diversification_reward', 0.0):.3f}")
    print(f"  Turnover: {trading['turnover']:.3f}")
    
    # Benchmark comparison
    if report['evaluation_metrics']['benchmark']:
        benchmark = report['evaluation_metrics']['benchmark']
        print(f"\n📈 BENCHMARK COMPARISON:")
        print(f"  Benchmark Return: {benchmark['benchmark_return']:.2%}")
        print(f"  Excess Return: {benchmark['excess_return']:.2%}")
        print(f"  Alpha: {benchmark['alpha']:.2%}")
        print(f"  Beta: {benchmark['beta']:.3f}")
        print(f"  Information Ratio: {benchmark['information_ratio']:.3f}")
    
    # Performance summary
    print(f"\n📋 PERFORMANCE SUMMARY:")
    print("=" * 50)
    for key, value in report['performance_summary'].items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    # Step 4: Save results
    print(f"\n💾 STEP 4: Saving Results...")
    print("-" * 40)
    
    trainer.save_results()
    print(f"✅ Results saved to results/currency_pairs/")
    
    # Step 5: Performance assessment
    print(f"\n🎯 PERFORMANCE ASSESSMENT:")
    print("=" * 50)
    
    # Assess performance based on metrics
    sharpe = risk['sharpe_ratio']
    max_dd = abs(risk['max_drawdown'])
    win_rate = trading['win_rate']
    profit_factor = trading['profit_factor']
    
    print(f"\n📊 Performance Analysis:")
    
    # Sharpe ratio assessment
    if sharpe > 1.5:
        sharpe_assessment = "Excellent"
    elif sharpe > 1.0:
        sharpe_assessment = "Good"
    elif sharpe > 0.5:
        sharpe_assessment = "Average"
    else:
        sharpe_assessment = "Poor"
    print(f"  Sharpe Ratio: {sharpe_assessment} ({sharpe:.3f})")
    
    # Drawdown assessment
    if max_dd < 0.1:
        dd_assessment = "Low Risk"
    elif max_dd < 0.2:
        dd_assessment = "Medium Risk"
    else:
        dd_assessment = "High Risk"
    print(f"  Risk Level: {dd_assessment} (Max DD: {max_dd:.2%})")
    
    # Trading performance
    if win_rate > 0.6 and profit_factor > 1.5:
        trading_assessment = "Excellent"
    elif win_rate > 0.5 and profit_factor > 1.2:
        trading_assessment = "Good"
    elif win_rate > 0.4 and profit_factor > 1.0:
        trading_assessment = "Average"
    else:
        trading_assessment = "Poor"
    print(f"  Trading Performance: {trading_assessment} (Win Rate: {win_rate:.2%}, PF: {profit_factor:.3f})")
    
    # Overall assessment
    if sharpe > 1.0 and max_dd < 0.15 and win_rate > 0.5:
        overall_assessment = "🟢 STRONG PERFORMANCE"
    elif sharpe > 0.5 and max_dd < 0.25 and win_rate > 0.4:
        overall_assessment = "🟡 MODERATE PERFORMANCE"
    else:
        overall_assessment = "🔴 WEAK PERFORMANCE"
    
    print(f"\n🎯 OVERALL ASSESSMENT: {overall_assessment}")
    
    print(f"\n✅ Currency Pairs Agent Demo Completed!")
    print(f"📁 Check results/currency_pairs/ for detailed reports")
    
    return trainer, report

if __name__ == "__main__":
    trainer, report = main()
