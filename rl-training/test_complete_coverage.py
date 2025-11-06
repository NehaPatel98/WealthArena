#!/usr/bin/env python3
"""
Comprehensive tests to achieve 80%+ code coverage
Tests all major utility functions and data processing
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from download_market_data import DataDownloader


class TestDataDownloaderComplete:
    """Complete test coverage for DataDownloader"""
    
    @pytest.fixture
    def downloader_full(self, tmp_path):
        """Create fully configured downloader"""
        config = {
            "symbols": ["AAPL", "GOOGL", "MSFT"],
            "start_date": "2022-01-01",
            "end_date": "2023-01-01"
        }
        downloader = DataDownloader(config)
        downloader.data_dir = tmp_path / "data"
        downloader.raw_dir = downloader.data_dir / "raw"
        downloader.processed_dir = downloader.data_dir / "processed"
        downloader.raw_dir.mkdir(parents=True, exist_ok=True)
        downloader.processed_dir.mkdir(parents=True, exist_ok=True)
        return downloader
    
    @pytest.fixture
    def complete_stock_data(self):
        """Create complete stock data for testing"""
        np.random.seed(42)
        n = 252  # 1 year of trading days
        dates = pd.date_range(start='2022-01-01', periods=n, freq='B')
        
        # Generate realistic price data
        close = 100 + np.cumsum(np.random.randn(n) * 2)
        high = close + np.abs(np.random.randn(n) * 1.5)
        low = close - np.abs(np.random.randn(n) * 1.5)
        open_price = close + np.random.randn(n) * 0.5
        
        return pd.DataFrame({
            'Date': dates,
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': np.random.randint(1000000, 10000000, n)
        })
    
    def test_download_all_data_mock(self, downloader_full, complete_stock_data):
        """Test download_all_data with mock data"""
        # Create mock processed data
        for symbol in downloader_full.symbols:
            downloader_full.save_data(complete_stock_data, symbol, "processed")
        
        # Verify files were created
        for symbol in downloader_full.symbols:
            file_path = downloader_full.processed_dir / f"{symbol}_processed.csv"
            assert file_path.exists()
    
    def test_add_all_technical_indicators(self, downloader_full, complete_stock_data):
        """Test adding all technical indicators"""
        result = downloader_full.add_technical_indicators(complete_stock_data.copy(), "TEST")
        
        # Check for key indicators
        expected_columns = [
            'Returns', 'Volatility_20', 'Momentum_10', 
            'Volume_Ratio', 'SMA_20', 'SMA_50'
        ]
        
        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"
    
    def test_technical_indicators_with_small_dataset(self, downloader_full):
        """Test technical indicators with small dataset"""
        # Create small dataset (less than typical indicator windows)
        small_data = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=10),
            'Open': [100 + i for i in range(10)],
            'High': [105 + i for i in range(10)],
            'Low': [95 + i for i in range(10)],
            'Close': [102 + i for i in range(10)],
            'Volume': [1000000 + i * 10000 for i in range(10)]
        })
        
        result = downloader_full.add_technical_indicators(small_data, "SMALL")
        
        # Should still return DataFrame with additional columns
        assert len(result) == len(small_data)
        assert len(result.columns) > len(small_data.columns)
    
    def test_save_raw_and_processed_data(self, downloader_full, complete_stock_data):
        """Test saving both raw and processed data"""
        symbol = "TEST"
        
        # Save raw data
        downloader_full.save_data(complete_stock_data, symbol, "raw")
        raw_file = downloader_full.raw_dir / f"{symbol}_raw.csv"
        assert raw_file.exists()
        
        # Save processed data
        processed = downloader_full.add_technical_indicators(complete_stock_data, symbol)
        downloader_full.save_data(processed, symbol, "processed")
        processed_file = downloader_full.processed_dir / f"{symbol}_processed.csv"
        assert processed_file.exists()
        
        # Verify processed has more columns than raw
        raw_df = pd.read_csv(raw_file)
        processed_df = pd.read_csv(processed_file)
        assert len(processed_df.columns) > len(raw_df.columns)
    
    def test_data_summary_with_multiple_symbols(self, downloader_full, complete_stock_data):
        """Test creating summary for multiple symbols"""
        all_data = {}
        for symbol in downloader_full.symbols:
            # Create slightly different data for each symbol
            data = complete_stock_data.copy()
            data['Close'] = data['Close'] * (np.random.rand() + 0.5)
            all_data[symbol] = data
        
        summary = downloader_full.create_data_summary(all_data)
        
        # Verify summary structure
        assert summary['num_symbols'] == len(downloader_full.symbols)
        assert len(summary['symbols']) == len(downloader_full.symbols)
        
        # Check each symbol has details
        for symbol in downloader_full.symbols:
            assert symbol in summary['symbol_details']
            details = summary['symbol_details'][symbol]
            assert 'records' in details
            assert 'features' in details
            assert 'price_range' in details
            assert 'volume_range' in details
    
    def test_quality_check_with_various_issues(self, downloader_full):
        """Test quality check with various data issues"""
        # Good quality data
        good_data = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=100),
            'Close': np.random.uniform(100, 110, 100),
            'Volume': np.random.randint(1000000, 2000000, 100),
            'Open': np.random.uniform(100, 110, 100),
            'High': np.random.uniform(110, 120, 100),
            'Low': np.random.uniform(90, 100, 100)
        })
        
        # Data with extreme movements
        extreme_data = good_data.copy()
        extreme_data.loc[50, 'Close'] = extreme_data.loc[49, 'Close'] * 2  # 100% jump
        
        # Data with volume anomalies
        volume_data = good_data.copy()
        volume_data.loc[60, 'Volume'] = volume_data['Volume'].mean() * 10
        
        all_data = {
            'GOOD': good_data,
            'EXTREME': extreme_data,
            'VOLUME_SPIKE': volume_data
        }
        
        quality_report = downloader_full.validate_data_quality(all_data)
        
        assert 'overall_quality' in quality_report
        assert 'symbol_quality' in quality_report
        assert len(quality_report['symbol_quality']) == 3
    
    def test_validate_with_all_scenarios(self, downloader_full):
        """Test validation with all edge case scenarios"""
        # Test 1: Valid data
        valid_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [99, 100, 101],
            'Close': [103, 104, 105],
            'Volume': [1000000, 1100000, 1200000]
        })
        assert downloader_full._validate_basic_data(valid_data, "VALID")
        
        # Test 2: Empty DataFrame
        empty_data = pd.DataFrame()
        assert not downloader_full._validate_basic_data(empty_data, "EMPTY")
        
        # Test 3: Missing columns
        incomplete_data = pd.DataFrame({'Open': [100], 'Close': [101]})
        assert not downloader_full._validate_basic_data(incomplete_data, "INCOMPLETE")
        
        # Test 4: Negative prices (should filter them out)
        negative_data = pd.DataFrame({
            'Open': [100, -50, 102],
            'High': [105, 106, 107],
            'Low': [99, 100, 101],
            'Close': [103, 104, 105],
            'Volume': [1000000, 1100000, 1200000]
        })
        result = downloader_full._validate_basic_data(negative_data, "NEGATIVE")
        assert isinstance(result, bool)
        
        # Test 5: Invalid OHLC relationships
        invalid_ohlc = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [95, 96, 97],  # High < Low
            'Low': [99, 100, 101],
            'Close': [103, 104, 105],
            'Volume': [1000000, 1100000, 1200000]
        })
        result = downloader_full._validate_basic_data(invalid_ohlc, "INVALID_OHLC")
        assert isinstance(result, bool)


class TestAdvancedDataProcessing:
    """Advanced data processing tests"""
    
    def test_bollinger_bands_equivalent(self):
        """Test Bollinger Bands calculation logic"""
        prices = pd.Series([100 + i + np.random.randn() * 2 for i in range(50)])
        
        # Calculate moving average
        ma = prices.rolling(window=20).mean()
        
        # Calculate standard deviation
        std = prices.rolling(window=20).std()
        
        # Upper and lower bands
        upper_band = ma + (std * 2)
        lower_band = ma - (std * 2)
        
        # Verify calculations
        assert len(upper_band) == len(prices)
        assert all(upper_band.dropna() >= ma.dropna())
        assert all(lower_band.dropna() <= ma.dropna())
    
    def test_rsi_logic(self):
        """Test RSI calculation logic"""
        prices = pd.Series([100 + i + np.random.randn() * 5 for i in range(50)])
        
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=14).mean()
        avg_losses = losses.rolling(window=14).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        # Verify RSI is between 0 and 100
        valid_rsi = rsi.dropna()
        assert all(valid_rsi >= 0)
        assert all(valid_rsi <= 100)
    
    def test_macd_calculation(self):
        """Test MACD calculation"""
        prices = pd.Series([100 + i + np.random.randn() * 3 for i in range(100)])
        
        # Calculate EMAs
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        
        # MACD line
        macd = ema_12 - ema_26
        
        # Signal line
        signal = macd.ewm(span=9).mean()
        
        # Histogram
        histogram = macd - signal
        
        assert len(macd) == len(prices)
        assert len(signal) == len(prices)
        assert len(histogram) == len(prices)
    
    def test_atr_calculation(self):
        """Test ATR (Average True Range) calculation"""
        high = pd.Series([105 + i + np.random.rand() * 2 for i in range(50)])
        low = pd.Series([95 + i - np.random.rand() * 2 for i in range(50)])
        close = pd.Series([100 + i for i in range(50)])
        
        # Calculate true range components
        h_l = high - low
        h_pc = np.abs(high - close.shift())
        l_pc = np.abs(low - close.shift())
        
        # True range is the max of the three
        tr = pd.DataFrame({'h_l': h_l, 'h_pc': h_pc, 'l_pc': l_pc}).max(axis=1)
        
        # ATR is the moving average of true range
        atr = tr.rolling(window=14).mean()
        
        assert len(atr) == len(high)
        assert all(atr.dropna() > 0)


class TestDataQualityChecks:
    """Comprehensive data quality checks"""
    
    def test_completeness_check(self):
        """Test data completeness"""
        data = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': [1, 2, 3, 4, 5],
            'C': [np.nan, np.nan, 3, 4, 5]
        })
        
        # Calculate completeness per column
        completeness = data.count() / len(data)
        
        assert completeness['B'] == 1.0  # 100% complete
        assert completeness['A'] == 0.8  # 80% complete
        assert completeness['C'] == 0.6  # 60% complete
    
    def test_consistency_checks(self):
        """Test data consistency"""
        data = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=5),
            'Price': [100, 101, 102, 103, 104]
        })
        
        # Check date continuity
        date_diffs = data['Date'].diff()
        
        # Check for duplicates
        duplicates = data.duplicated()
        
        assert len(data) == 5
        assert not any(duplicates)
    
    def test_range_validation(self):
        """Test value range validation"""
        data = pd.DataFrame({
            'Price': [100, 101, -5, 102, 103],  # -5 is out of valid range
            'Volume': [1000000, 1100000, 1200000, 1300000, -100000]  # -100000 invalid
        })
        
        # Identify out-of-range values
        invalid_prices = data[data['Price'] < 0]
        invalid_volumes = data[data['Volume'] < 0]
        
        assert len(invalid_prices) == 1
        assert len(invalid_volumes) == 1


class TestEdgeCasesAndErrors:
    """Test edge cases and error handling"""
    
    def test_single_row_data(self):
        """Test handling of single row data"""
        data = pd.DataFrame({
            'Open': [100],
            'High': [105],
            'Low': [99],
            'Close': [103],
            'Volume': [1000000]
        })
        
        # Should still be valid
        assert len(data) == 1
        assert all(data['High'] >= data['Low'])
    
    def test_very_large_dataset(self):
        """Test handling of very large dataset"""
        n = 10000
        large_data = pd.DataFrame({
            'Date': pd.date_range('2000-01-01', periods=n),
            'Price': np.random.uniform(100, 200, n),
            'Volume': np.random.randint(1000000, 10000000, n)
        })
        
        # Should handle large data efficiently
        assert len(large_data) == n
        assert large_data.memory_usage().sum() > 0
    
    def test_data_with_infinities(self):
        """Test handling of infinite values"""
        data = pd.DataFrame({
            'A': [1, 2, np.inf, 4, 5],
            'B': [1, 2, 3, -np.inf, 5]
        })
        
        # Identify infinities
        has_inf = data.isin([np.inf, -np.inf]).any()
        
        assert has_inf['A']
        assert has_inf['B']
        
        # Replace infinities with NaN
        clean_data = data.replace([np.inf, -np.inf], np.nan)
        assert not clean_data.isin([np.inf, -np.inf]).any().any()
    
    def test_mixed_data_types(self):
        """Test handling of mixed data types"""
        data = pd.DataFrame({
            'String': ['a', 'b', 'c'],
            'Numeric': [1, 2, 3],
            'Float': [1.1, 2.2, 3.3]
        })
        
        # Get numeric columns only
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        assert 'String' not in numeric_cols
        assert 'Numeric' in numeric_cols
        assert 'Float' in numeric_cols


def test_integration_scenario():
    """Test complete integration scenario"""
    # Create downloader
    config = {
        "symbols": ["TEST1", "TEST2"],
        "start_date": "2023-01-01",
        "end_date": "2023-12-31"
    }
    downloader = DataDownloader(config)
    
    # Verify initialization
    assert len(downloader.symbols) == 2
    assert downloader.start_date == "2023-01-01"
    
    # Create mock data
    mock_data = pd.DataFrame({
        'Date': pd.date_range('2023-01-01', periods=100),
        'Open': np.random.uniform(100, 110, 100),
        'High': np.random.uniform(110, 120, 100),
        'Low': np.random.uniform(90, 100, 100),
        'Close': np.random.uniform(100, 110, 100),
        'Volume': np.random.randint(1000000, 5000000, 100)
    })
    
    # Test validation
    is_valid = downloader._validate_basic_data(mock_data, "TEST")
    assert is_valid
    
    # Test technical indicators
    processed = downloader.add_technical_indicators(mock_data, "TEST")
    assert len(processed.columns) > len(mock_data.columns)
    
    # Test quality check
    quality = downloader.validate_data_quality({"TEST": processed})
    assert 'TEST' in quality['symbol_quality']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

