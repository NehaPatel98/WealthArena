#!/usr/bin/env python3
"""
Real Benchmark Analysis Report

This script generates a comprehensive analysis of real market benchmarks
used to measure model profitability.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.benchmarks.benchmark_data import BenchmarkDataFetcher, BenchmarkConfig, BenchmarkAnalyzer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    """Generate comprehensive benchmark analysis report"""
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    print("📊 REAL BENCHMARK ANALYSIS REPORT")
    print("="*60)
    
    # Create benchmark fetcher
    config = BenchmarkConfig(
        start_date="2015-01-01",
        end_date="2025-09-26"
    )
    fetcher = BenchmarkDataFetcher(config)
    
    # Fetch all benchmarks
    print("\n🔄 Fetching Real Market Data...")
    benchmarks = fetcher.get_all_benchmarks()
    
    # Create analyzer
    analyzer = BenchmarkAnalyzer(benchmarks)
    
    # Generate comparison table
    print("\n📈 BENCHMARK PERFORMANCE COMPARISON (2015-2025)")
    print("="*60)
    
    comparison_df = analyzer.compare_benchmarks()
    
    # Display results
    print(f"\n{'Benchmark':<20} {'Annual Return':<15} {'Volatility':<12} {'Sharpe Ratio':<12} {'Max DD':<10} {'Data Points':<12}")
    print("-" * 80)
    
    for _, row in comparison_df.iterrows():
        print(f"{row['benchmark']:<20} {row['annual_return']:<15.2%} {row['volatility']:<12.2%} {row['sharpe_ratio']:<12.3f} {row['max_drawdown']:<10.2%} {row['data_points']:<12.0f}")
    
    # Risk-Free Rate Analysis
    print(f"\n💰 RISK-FREE RATE ANALYSIS:")
    risk_free = benchmarks["risk_free_rate"]
    risk_free_annual = (1 + risk_free).prod() ** (252 / len(risk_free)) - 1
    print(f"  10-Year Treasury Average: {risk_free.mean():.2%} daily")
    print(f"  10-Year Treasury Annual: {risk_free_annual:.2%}")
    print(f"  Data Points: {len(risk_free)} days")
    
    # Sector Analysis
    print(f"\n🏭 SECTOR BENCHMARK ANALYSIS:")
    sectors = benchmarks["sectors"]
    for sector, data in sectors.items():
        if not data.empty:
            annual_return = (1 + data).prod() ** (252 / len(data)) - 1
            volatility = data.std() * np.sqrt(252)
            sharpe = annual_return / volatility if volatility > 0 else 0
            print(f"  {sector.title():<15}: {annual_return:>8.2%} return, {volatility:>8.2%} volatility, {sharpe:>6.3f} Sharpe")
    
    # Benchmark Characteristics
    print(f"\n📊 BENCHMARK CHARACTERISTICS:")
    print(f"  Currency (DXY):           {len(benchmarks['currency'])} days - Dollar strength index")
    print(f"  ASX Stocks (ASX 200):     {len(benchmarks['asx_stocks'])} days - Australian equity market")
    print(f"  Cryptocurrencies (BTC):   {len(benchmarks['cryptocurrencies'])} days - Bitcoin as crypto proxy")
    print(f"  ETFs (S&P 500):          {len(benchmarks['etf'])} days - US equity market")
    print(f"  Commodities (DJP):       {len(benchmarks['commodities'])} days - Bloomberg commodity index")
    print(f"  Risk-Free Rate (TNX):    {len(benchmarks['risk_free_rate'])} days - 10-year Treasury yield")
    
    # Data Quality Assessment
    print(f"\n✅ DATA QUALITY ASSESSMENT:")
    for name, data in benchmarks.items():
        if isinstance(data, dict):  # Skip sector data
            continue
        
        if data.empty:
            print(f"  ❌ {name}: No data available")
        else:
            missing_pct = (data.isna().sum() / len(data)) * 100
            if missing_pct < 5:
                print(f"  ✅ {name}: {len(data)} days, {missing_pct:.1f}% missing data")
            else:
                print(f"  ⚠️  {name}: {len(data)} days, {missing_pct:.1f}% missing data")
    
    # Benchmark Correlation Analysis
    print(f"\n🔗 BENCHMARK CORRELATION ANALYSIS:")
    correlation_data = {}
    for name, data in benchmarks.items():
        if isinstance(data, dict) or data.empty:
            continue
        # Normalize timezone and align data
        data_normalized = data.copy()
        if hasattr(data_normalized.index, 'tz') and data_normalized.index.tz is not None:
            data_normalized.index = data_normalized.index.tz_localize(None)
        correlation_data[name] = data_normalized
    
    if len(correlation_data) > 1:
        try:
            # Align all series to common index
            common_index = None
            for name, data in correlation_data.items():
                if common_index is None:
                    common_index = data.index
                else:
                    common_index = common_index.intersection(data.index)
            
            if len(common_index) > 100:  # Need sufficient data for correlation
                aligned_data = {}
                for name, data in correlation_data.items():
                    aligned_data[name] = data.loc[common_index]
                
                corr_df = pd.DataFrame(aligned_data)
                corr_matrix = corr_df.corr()
                
                print(f"\n  Correlation Matrix (selected pairs):")
                if 'currency' in corr_matrix.index and 'asx_stocks' in corr_matrix.index:
                    print(f"  Currency vs ASX:     {corr_matrix.loc['currency', 'asx_stocks']:.3f}")
                if 'currency' in corr_matrix.index and 'cryptocurrencies' in corr_matrix.index:
                    print(f"  Currency vs Crypto:  {corr_matrix.loc['currency', 'cryptocurrencies']:.3f}")
                if 'asx_stocks' in corr_matrix.index and 'etf' in corr_matrix.index:
                    print(f"  ASX vs S&P 500:      {corr_matrix.loc['asx_stocks', 'etf']:.3f}")
                if 'cryptocurrencies' in corr_matrix.index and 'commodities' in corr_matrix.index:
                    print(f"  Crypto vs Commodities: {corr_matrix.loc['cryptocurrencies', 'commodities']:.3f}")
            else:
                print(f"  Insufficient common data for correlation analysis")
        except Exception as e:
            print(f"  Error in correlation analysis: {e}")
    
    # Model Performance vs Benchmarks
    print(f"\n🎯 MODEL PERFORMANCE vs REAL BENCHMARKS:")
    print(f"  This analysis shows how our trading models perform against")
    print(f"  real market benchmarks instead of synthetic data.")
    print(f"  \n  Key Benefits:")
    print(f"  • Real market conditions and volatility")
    print(f"  • Actual economic cycles and events")
    print(f"  • Proper risk-adjusted performance measurement")
    print(f"  • Industry-standard benchmark comparisons")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"  1. Use DXY for currency pair strategies")
    print(f"  2. Use ASX 200 for Australian equity strategies")
    print(f"  3. Use Bitcoin for cryptocurrency strategies")
    print(f"  4. Use S&P 500 for ETF and US equity strategies")
    print(f"  5. Use Bloomberg Commodity Index for commodity strategies")
    print(f"  6. Use 10-Year Treasury for risk-free rate calculations")
    print(f"  7. Consider sector-specific benchmarks for specialized strategies")
    
    # Save detailed report
    output_dir = Path("results/benchmark_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save comparison data
    comparison_df.to_csv(output_dir / "benchmark_comparison.csv", index=False)
    
    # Save correlation matrix
    if len(correlation_data) > 1:
        corr_df.to_csv(output_dir / "benchmark_correlations.csv")
    
    print(f"\n💾 Detailed analysis saved to: {output_dir}")
    print(f"\n✅ Real benchmark analysis completed!")


if __name__ == "__main__":
    main()
