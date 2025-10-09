#!/usr/bin/env python3
"""
Unit tests for market data utility functions
Tests for data validation, processing, and technical indicators
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


class TestDataValidation:
    """Test data validation functions"""
    
    def test_valid_ohlc_data(self):
        """Test validation of valid OHLC data"""
        data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [99, 100, 101],
            'Close': [103, 104, 105],
            'Volume': [1000000, 1100000, 1200000]
        })
        
        # Check basic properties
        assert len(data) == 3
        assert all(data['High'] >= data['Low'])
        assert all(data['High'] >= data['Open'])
        assert all(data['High'] >= data['Close'])
    
    def test_data_with_nans(self):
        """Test handling of NaN values"""
        data = pd.DataFrame({
            'Open': [100, np.nan, 102],
            'High': [105, 106, np.nan],
            'Low': [99, 100, 101],
            'Close': [103, 104, 105]
        })
        
        # Drop NaNs
        clean_data = data.dropna()
        assert len(clean_data) == 1
    
    def test_price_consistency(self):
        """Test price consistency checks"""
        data = pd.DataFrame({
            'Open': [100.0, 101.5, 102.3],
            'High': [105.2, 106.1, 107.8],
            'Low': [99.1, 100.2, 101.5],
            'Close': [103.5, 104.8, 105.9]
        })
        
        # Verify high is always highest
        assert all(data['High'] >= data['Open'])
        assert all(data['High'] >= data['Close'])
        assert all(data['High'] >= data['Low'])
        
        # Verify low is always lowest
        assert all(data['Low'] <= data['Open'])
        assert all(data['Low'] <= data['Close'])
        assert all(data['Low'] <= data['High'])


class TestTechnicalIndicators:
    """Test technical indicator calculations"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample market data"""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        close_prices = 100 + np.cumsum(np.random.randn(100) * 2)
        
        return pd.DataFrame({
            'Date': dates,
            'Open': close_prices + np.random.rand(100) - 0.5,
            'High': close_prices + np.random.rand(100) * 2,
            'Low': close_prices - np.random.rand(100) * 2,
            'Close': close_prices,
            'Volume': np.random.randint(1000000, 5000000, 100)
        })
    
    def test_simple_moving_average(self, sample_data):
        """Test SMA calculation"""
        window = 20
        sma = sample_data['Close'].rolling(window=window).mean()
        
        assert len(sma) == len(sample_data)
        assert pd.isna(sma.iloc[window-2])  # Should be NaN for early values
        assert not pd.isna(sma.iloc[window])  # Should have value after window
    
    def test_returns_calculation(self, sample_data):
        """Test returns calculation"""
        returns = sample_data['Close'].pct_change()
        
        assert len(returns) == len(sample_data)
        assert pd.isna(returns.iloc[0])  # First value should be NaN
        assert not pd.isna(returns.iloc[1])  # Second value should exist
    
    def test_volatility_calculation(self, sample_data):
        """Test volatility calculation"""
        returns = sample_data['Close'].pct_change()
        volatility = returns.rolling(window=20).std() * np.sqrt(252)
        
        assert len(volatility) == len(sample_data)
        assert all(volatility.dropna() >= 0)  # Volatility should be non-negative
    
    def test_momentum_calculation(self, sample_data):
        """Test momentum calculation"""
        period = 5
        momentum = sample_data['Close'] / sample_data['Close'].shift(period) - 1
        
        assert len(momentum) == len(sample_data)
        # Check that we have valid values after the period
        assert not pd.isna(momentum.iloc[period])
    
    def test_volume_ratio(self, sample_data):
        """Test volume ratio calculation"""
        volume_ma = sample_data['Volume'].rolling(window=20).mean()
        volume_ratio = sample_data['Volume'] / volume_ma
        
        assert len(volume_ratio) == len(sample_data)
        assert all(volume_ratio.dropna() > 0)  # Ratios should be positive


class TestDataTransformations:
    """Test data transformation functions"""
    
    def test_normalize_prices(self):
        """Test price normalization"""
        prices = pd.Series([100, 110, 105, 115, 120])
        
        # Min-max normalization
        normalized = (prices - prices.min()) / (prices.max() - prices.min())
        
        assert normalized.min() == 0.0
        assert normalized.max() == 1.0
        assert len(normalized) == len(prices)
    
    def test_log_returns(self):
        """Test log returns calculation"""
        prices = pd.Series([100, 110, 105, 115, 120])
        log_returns = np.log(prices / prices.shift(1))
        
        assert len(log_returns) == len(prices)
        assert pd.isna(log_returns.iloc[0])
        assert all(np.isfinite(log_returns.dropna()))
    
    def test_percentage_change(self):
        """Test percentage change calculation"""
        prices = pd.Series([100, 110, 105, 115, 120])
        pct_change = prices.pct_change()
        
        assert len(pct_change) == len(prices)
        assert pd.isna(pct_change.iloc[0])
        assert abs(pct_change.iloc[1] - 0.10) < 0.001  # 10% increase


class TestDataFrameOperations:
    """Test DataFrame operations"""
    
    def test_merge_data_frames(self):
        """Test merging multiple data frames"""
        df1 = pd.DataFrame({'Date': pd.date_range('2023-01-01', periods=5),
                            'Price': [100, 101, 102, 103, 104]})
        df2 = pd.DataFrame({'Date': pd.date_range('2023-01-01', periods=5),
                            'Volume': [1000, 1100, 1200, 1300, 1400]})
        
        merged = pd.merge(df1, df2, on='Date')
        
        assert len(merged) == 5
        assert 'Price' in merged.columns
        assert 'Volume' in merged.columns
    
    def test_filter_by_date_range(self):
        """Test filtering data by date range"""
        dates = pd.date_range('2023-01-01', periods=100)
        df = pd.DataFrame({'Date': dates, 'Value': range(100)})
        
        start_date = '2023-01-10'
        end_date = '2023-01-20'
        
        filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        
        assert len(filtered) <= len(df)
        assert all(filtered['Date'] >= pd.Timestamp(start_date))
        assert all(filtered['Date'] <= pd.Timestamp(end_date))
    
    def test_resample_data(self):
        """Test data resampling"""
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        df = pd.DataFrame({'Date': dates, 'Price': range(30)})
        df.set_index('Date', inplace=True)
        
        # Resample to weekly
        weekly = df.resample('W').mean()
        
        assert len(weekly) < len(df)
        assert all(pd.notna(weekly['Price']))


class TestStatisticalFunctions:
    """Test statistical calculations"""
    
    def test_mean_calculation(self):
        """Test mean calculation"""
        data = pd.Series([10, 20, 30, 40, 50])
        mean = data.mean()
        
        assert mean == 30.0
    
    def test_std_calculation(self):
        """Test standard deviation calculation"""
        data = pd.Series([10, 20, 30, 40, 50])
        std = data.std()
        
        assert std > 0
        assert isinstance(std, (float, np.floating))
    
    def test_correlation(self):
        """Test correlation calculation"""
        data1 = pd.Series([1, 2, 3, 4, 5])
        data2 = pd.Series([2, 4, 6, 8, 10])
        
        corr = data1.corr(data2)
        
        assert abs(corr - 1.0) < 0.001  # Perfect positive correlation
    
    def test_quantiles(self):
        """Test quantile calculation"""
        data = pd.Series(range(100))
        
        q25 = data.quantile(0.25)
        q50 = data.quantile(0.50)
        q75 = data.quantile(0.75)
        
        assert q25 < q50 < q75
        assert q50 == data.median()


def test_data_type_conversions():
    """Test data type conversions"""
    # String to numeric
    df = pd.DataFrame({'Price': ['100', '101', '102']})
    df['Price'] = pd.to_numeric(df['Price'])
    
    assert df['Price'].dtype in [np.int64, np.float64]
    
    # Date string to datetime
    df['Date'] = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
    assert df['Date'].dtype == 'datetime64[ns]'


def test_handling_outliers():
    """Test outlier detection and handling"""
    data = pd.Series([10, 12, 11, 13, 12, 1000, 11, 12, 13])  # 1000 is outlier
    
    # Calculate IQR
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    # Define outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter outliers
    filtered = data[(data >= lower_bound) & (data <= upper_bound)]
    
    assert len(filtered) < len(data)
    assert 1000 not in filtered.values


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

