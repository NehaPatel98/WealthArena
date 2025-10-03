"""
Real Benchmark Data Fetcher

This module provides real market benchmark data fetching using yfinance and other APIs
to replace synthetic benchmarks with actual market data.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import requests
import json
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark data fetching"""
    start_date: str = "2015-01-01"
    end_date: str = "2025-09-26"
    risk_free_rate_symbol: str = "^TNX"  # 10-year Treasury
    cache_duration_hours: int = 24
    fallback_to_synthetic: bool = True


class BenchmarkDataFetcher:
    """Fetches real market benchmark data"""
    
    def __init__(self, config: BenchmarkConfig = None):
        self.config = config or BenchmarkConfig()
        self.cache = {}
        self.cache_timestamps = {}
        
    def get_risk_free_rate(self) -> pd.Series:
        """Get 10-year Treasury yield as risk-free rate"""
        try:
            logger.info("Fetching 10-year Treasury yield...")
            treasury = yf.Ticker(self.config.risk_free_rate_symbol)
            data = treasury.history(
                start=self.config.start_date,
                end=self.config.end_date,
                interval="1d"
            )
            
            if data.empty:
                logger.warning("No Treasury data found, using default 2%")
                return self._create_default_risk_free_rate()
            
            # Use adjusted close as the yield
            risk_free_rate = data['Close'] / 100  # Convert percentage to decimal
            risk_free_rate = risk_free_rate.fillna(method='ffill').fillna(0.02)
            
            logger.info(f"Fetched Treasury data: {len(risk_free_rate)} days")
            return risk_free_rate
            
        except Exception as e:
            logger.error(f"Error fetching Treasury data: {e}")
            return self._create_default_risk_free_rate()
    
    def _create_default_risk_free_rate(self) -> pd.Series:
        """Create default risk-free rate series"""
        dates = pd.date_range(start=self.config.start_date, end=self.config.end_date, freq='D')
        return pd.Series(0.02, index=dates)  # 2% default
    
    def get_currency_benchmark(self) -> pd.Series:
        """Get DXY (Dollar Index) as currency benchmark"""
        try:
            logger.info("Fetching DXY (Dollar Index)...")
            dxy = yf.Ticker("DX-Y.NYB")
            data = dxy.history(
                start=self.config.start_date,
                end=self.config.end_date,
                interval="1d"
            )
            
            if data.empty:
                logger.warning("No DXY data found, using synthetic benchmark")
                return self._create_synthetic_currency_benchmark()
            
            # Calculate daily returns
            returns = data['Close'].pct_change().dropna()
            returns = returns.fillna(0)
            
            logger.info(f"Fetched DXY data: {len(returns)} days")
            return returns
            
        except Exception as e:
            logger.error(f"Error fetching DXY data: {e}")
            return self._create_synthetic_currency_benchmark()
    
    def _create_synthetic_currency_benchmark(self) -> pd.Series:
        """Create synthetic currency benchmark"""
        dates = pd.date_range(start=self.config.start_date, end=self.config.end_date, freq='D')
        returns = np.random.normal(0.0003, 0.008, len(dates))
        return pd.Series(returns, index=dates)
    
    def get_asx_benchmark(self) -> pd.Series:
        """Get ASX 200 index as Australian stocks benchmark"""
        try:
            logger.info("Fetching ASX 200 index...")
            asx200 = yf.Ticker("^AXJO")
            data = asx200.history(
                start=self.config.start_date,
                end=self.config.end_date,
                interval="1d"
            )
            
            if data.empty:
                logger.warning("No ASX 200 data found, using synthetic benchmark")
                return self._create_synthetic_asx_benchmark()
            
            # Calculate daily returns
            returns = data['Close'].pct_change().dropna()
            returns = returns.fillna(0)
            
            logger.info(f"Fetched ASX 200 data: {len(returns)} days")
            return returns
            
        except Exception as e:
            logger.error(f"Error fetching ASX 200 data: {e}")
            return self._create_synthetic_asx_benchmark()
    
    def _create_synthetic_asx_benchmark(self) -> pd.Series:
        """Create synthetic ASX benchmark"""
        dates = pd.date_range(start=self.config.start_date, end=self.config.end_date, freq='D')
        returns = np.random.normal(0.0006, 0.012, len(dates))
        return pd.Series(returns, index=dates)
    
    def get_crypto_benchmark(self) -> pd.Series:
        """Get crypto market cap as cryptocurrency benchmark"""
        try:
            logger.info("Fetching crypto market cap data...")
            # Use Bitcoin as proxy for crypto market
            btc = yf.Ticker("BTC-USD")
            data = btc.history(
                start=self.config.start_date,
                end=self.config.end_date,
                interval="1d"
            )
            
            if data.empty:
                logger.warning("No Bitcoin data found, using synthetic benchmark")
                return self._create_synthetic_crypto_benchmark()
            
            # Calculate daily returns
            returns = data['Close'].pct_change().dropna()
            returns = returns.fillna(0)
            
            logger.info(f"Fetched Bitcoin data: {len(returns)} days")
            return returns
            
        except Exception as e:
            logger.error(f"Error fetching Bitcoin data: {e}")
            return self._create_synthetic_crypto_benchmark()
    
    def _create_synthetic_crypto_benchmark(self) -> pd.Series:
        """Create synthetic crypto benchmark"""
        dates = pd.date_range(start=self.config.start_date, end=self.config.end_date, freq='D')
        returns = np.random.normal(0.0012, 0.025, len(dates))
        return pd.Series(returns, index=dates)
    
    def get_etf_benchmark(self) -> pd.Series:
        """Get S&P 500 as ETF benchmark"""
        try:
            logger.info("Fetching S&P 500 index...")
            sp500 = yf.Ticker("^GSPC")
            data = sp500.history(
                start=self.config.start_date,
                end=self.config.end_date,
                interval="1d"
            )
            
            if data.empty:
                logger.warning("No S&P 500 data found, using synthetic benchmark")
                return self._create_synthetic_etf_benchmark()
            
            # Calculate daily returns
            returns = data['Close'].pct_change().dropna()
            returns = returns.fillna(0)
            
            logger.info(f"Fetched S&P 500 data: {len(returns)} days")
            return returns
            
        except Exception as e:
            logger.error(f"Error fetching S&P 500 data: {e}")
            return self._create_synthetic_etf_benchmark()
    
    def _create_synthetic_etf_benchmark(self) -> pd.Series:
        """Create synthetic ETF benchmark"""
        dates = pd.date_range(start=self.config.start_date, end=self.config.end_date, freq='D')
        returns = np.random.normal(0.0005, 0.006, len(dates))
        return pd.Series(returns, index=dates)
    
    def get_commodities_benchmark(self) -> pd.Series:
        """Get Bloomberg Commodity Index as commodities benchmark"""
        try:
            logger.info("Fetching Bloomberg Commodity Index...")
            # Use DJP (iPath Bloomberg Commodity Index) as proxy
            djp = yf.Ticker("DJP")
            data = djp.history(
                start=self.config.start_date,
                end=self.config.end_date,
                interval="1d"
            )
            
            if data.empty:
                logger.warning("No commodity index data found, using synthetic benchmark")
                return self._create_synthetic_commodities_benchmark()
            
            # Calculate daily returns
            returns = data['Close'].pct_change().dropna()
            returns = returns.fillna(0)
            
            logger.info(f"Fetched commodity index data: {len(returns)} days")
            return returns
            
        except Exception as e:
            logger.error(f"Error fetching commodity index data: {e}")
            return self._create_synthetic_commodities_benchmark()
    
    def _create_synthetic_commodities_benchmark(self) -> pd.Series:
        """Create synthetic commodities benchmark"""
        dates = pd.date_range(start=self.config.start_date, end=self.config.end_date, freq='D')
        returns = np.random.normal(0.0006, 0.035, len(dates))
        return pd.Series(returns, index=dates)
    
    def get_sector_benchmarks(self) -> Dict[str, pd.Series]:
        """Get sector-specific benchmarks"""
        sector_benchmarks = {}
        
        # Technology sector
        try:
            logger.info("Fetching technology sector benchmark...")
            qqq = yf.Ticker("QQQ")
            data = qqq.history(start=self.config.start_date, end=self.config.end_date, interval="1d")
            if not data.empty:
                sector_benchmarks["technology"] = data['Close'].pct_change().dropna().fillna(0)
        except Exception as e:
            logger.error(f"Error fetching technology benchmark: {e}")
        
        # Financial sector
        try:
            logger.info("Fetching financial sector benchmark...")
            xlf = yf.Ticker("XLF")
            data = xlf.history(start=self.config.start_date, end=self.config.end_date, interval="1d")
            if not data.empty:
                sector_benchmarks["financial"] = data['Close'].pct_change().dropna().fillna(0)
        except Exception as e:
            logger.error(f"Error fetching financial benchmark: {e}")
        
        # Energy sector
        try:
            logger.info("Fetching energy sector benchmark...")
            xle = yf.Ticker("XLE")
            data = xle.history(start=self.config.start_date, end=self.config.end_date, interval="1d")
            if not data.empty:
                sector_benchmarks["energy"] = data['Close'].pct_change().dropna().fillna(0)
        except Exception as e:
            logger.error(f"Error fetching energy benchmark: {e}")
        
        # Healthcare sector
        try:
            logger.info("Fetching healthcare sector benchmark...")
            xlv = yf.Ticker("XLV")
            data = xlv.history(start=self.config.start_date, end=self.config.end_date, interval="1d")
            if not data.empty:
                sector_benchmarks["healthcare"] = data['Close'].pct_change().dropna().fillna(0)
        except Exception as e:
            logger.error(f"Error fetching healthcare benchmark: {e}")
        
        return sector_benchmarks
    
    def get_composite_benchmark(self, weights: Dict[str, float] = None) -> pd.Series:
        """Create custom composite benchmark for multi-asset strategies"""
        if weights is None:
            weights = {
                "equity": 0.4,      # S&P 500
                "bonds": 0.3,       # 10-year Treasury
                "commodities": 0.2,  # Commodity Index
                "currency": 0.1     # DXY
            }
        
        try:
            # Get individual benchmarks
            equity_bench = self.get_etf_benchmark()
            bonds_bench = self.get_risk_free_rate() / 252  # Convert annual to daily
            commodities_bench = self.get_commodities_benchmark()
            currency_bench = self.get_currency_benchmark()
            
            # Align all series to common index
            common_index = equity_bench.index.intersection(bonds_bench.index).intersection(
                commodities_bench.index).intersection(currency_bench.index)
            
            if len(common_index) == 0:
                logger.warning("No common index found for composite benchmark")
                return self._create_synthetic_composite_benchmark()
            
            # Create weighted composite
            composite = (
                weights["equity"] * equity_bench.loc[common_index] +
                weights["bonds"] * bonds_bench.loc[common_index] +
                weights["commodities"] * commodities_bench.loc[common_index] +
                weights["currency"] * currency_bench.loc[common_index]
            )
            
            logger.info(f"Created composite benchmark: {len(composite)} days")
            return composite
            
        except Exception as e:
            logger.error(f"Error creating composite benchmark: {e}")
            return self._create_synthetic_composite_benchmark()
    
    def _create_synthetic_composite_benchmark(self) -> pd.Series:
        """Create synthetic composite benchmark"""
        dates = pd.date_range(start=self.config.start_date, end=self.config.end_date, freq='D')
        returns = np.random.normal(0.0006, 0.015, len(dates))
        return pd.Series(returns, index=dates)
    
    def get_all_benchmarks(self) -> Dict[str, pd.Series]:
        """Get all available benchmarks"""
        benchmarks = {
            "risk_free_rate": self.get_risk_free_rate(),
            "currency": self.get_currency_benchmark(),
            "asx_stocks": self.get_asx_benchmark(),
            "cryptocurrencies": self.get_crypto_benchmark(),
            "etf": self.get_etf_benchmark(),
            "commodities": self.get_commodities_benchmark(),
            "sectors": self.get_sector_benchmarks(),
            "composite": self.get_composite_benchmark()
        }
        
        logger.info(f"Fetched {len(benchmarks)} benchmark categories")
        return benchmarks


class BenchmarkAnalyzer:
    """Analyzes benchmark performance and characteristics"""
    
    def __init__(self, benchmarks: Dict[str, pd.Series]):
        self.benchmarks = benchmarks
    
    def calculate_benchmark_metrics(self, benchmark_name: str) -> Dict[str, float]:
        """Calculate performance metrics for a benchmark"""
        if benchmark_name not in self.benchmarks:
            return {}
        
        returns = self.benchmarks[benchmark_name]
        if returns.empty:
            return {}
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Risk metrics
        max_drawdown = self._calculate_max_drawdown(returns)
        
        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "data_points": len(returns)
        }
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()
    
    def compare_benchmarks(self) -> pd.DataFrame:
        """Compare all benchmarks"""
        comparison_data = []
        
        for name, returns in self.benchmarks.items():
            if isinstance(returns, dict):  # Skip sector benchmarks for now
                continue
            
            metrics = self.calculate_benchmark_metrics(name)
            if metrics:
                comparison_data.append({
                    "benchmark": name,
                    **metrics
                })
        
        return pd.DataFrame(comparison_data)


def create_benchmark_fetcher(start_date: str = "2015-01-01", 
                           end_date: str = "2025-09-26") -> BenchmarkDataFetcher:
    """Create a benchmark data fetcher with specified date range"""
    config = BenchmarkConfig(
        start_date=start_date,
        end_date=end_date
    )
    return BenchmarkDataFetcher(config)


def main():
    """Test the benchmark data fetcher"""
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 Testing Real Benchmark Data Fetcher...")
    print("="*50)
    
    # Create fetcher
    fetcher = create_benchmark_fetcher()
    
    # Test individual benchmarks
    print("\n📊 Testing Individual Benchmarks:")
    
    benchmarks = {
        "Risk-Free Rate (10Y Treasury)": fetcher.get_risk_free_rate(),
        "Currency (DXY)": fetcher.get_currency_benchmark(),
        "ASX Stocks (ASX 200)": fetcher.get_asx_benchmark(),
        "Cryptocurrencies (BTC)": fetcher.get_crypto_benchmark(),
        "ETFs (S&P 500)": fetcher.get_etf_benchmark(),
        "Commodities (DJP)": fetcher.get_commodities_benchmark()
    }
    
    for name, data in benchmarks.items():
        if not data.empty:
            annual_return = (1 + data).prod() ** (252 / len(data)) - 1
            volatility = data.std() * np.sqrt(252)
            print(f"  ✅ {name}: {len(data)} days, {annual_return:.2%} annual return, {volatility:.2%} volatility")
        else:
            print(f"  ❌ {name}: No data available")
    
    # Test composite benchmark
    print("\n📈 Testing Composite Benchmark:")
    composite = fetcher.get_composite_benchmark()
    if not composite.empty:
        annual_return = (1 + composite).prod() ** (252 / len(composite)) - 1
        volatility = composite.std() * np.sqrt(252)
        print(f"  ✅ Composite: {len(composite)} days, {annual_return:.2%} annual return, {volatility:.2%} volatility")
    
    # Test sector benchmarks
    print("\n🏭 Testing Sector Benchmarks:")
    sectors = fetcher.get_sector_benchmarks()
    for sector, data in sectors.items():
        if not data.empty:
            annual_return = (1 + data).prod() ** (252 / len(data)) - 1
            print(f"  ✅ {sector.title()}: {len(data)} days, {annual_return:.2%} annual return")
    
    print("\n✅ Benchmark data fetcher test completed!")


if __name__ == "__main__":
    main()
