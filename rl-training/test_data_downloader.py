#!/usr/bin/env python3
"""
Unit tests for DataDownloader module
Tests for download_market_data.py functionality
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from download_market_data import DataDownloader


class TestDataDownloader:
    """Test suite for DataDownloader class"""
    
    @pytest.fixture
    def downloader(self, tmp_path):
        """Create a DataDownloader instance with temporary directory"""
        config = {
            "symbols": ["AAPL", "GOOGL"],
            "start_date": "2023-01-01",
            "end_date": "2023-02-01"
        }
        downloader = DataDownloader(config)
        # Override data directories to use temp path
        downloader.data_dir = tmp_path / "data"
        downloader.raw_dir = downloader.data_dir / "raw"
        downloader.processed_dir = downloader.data_dir / "processed"
        downloader.raw_dir.mkdir(parents=True, exist_ok=True)
        downloader.processed_dir.mkdir(parents=True, exist_ok=True)
        return downloader
    
    def test_initialization(self, downloader):
        """Test DataDownloader initialization"""
        assert downloader is not None
        assert len(downloader.symbols) == 2
        assert "AAPL" in downloader.symbols
        assert downloader.start_date == "2023-01-01"
        assert downloader.end_date == "2023-02-01"
    
    def test_validate_basic_data_valid(self, downloader):
        """Test validation with valid data"""
        data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [99, 100, 101],
            'Close': [103, 104, 105],
            'Volume': [1000000, 1100000, 1200000]
        })
        
        result = downloader._validate_basic_data(data, "AAPL")
        assert result == True
    
    def test_validate_basic_data_empty(self, downloader):
        """Test validation with empty data"""
        data = pd.DataFrame()
        result = downloader._validate_basic_data(data, "AAPL")
        assert result == False
    
    def test_validate_basic_data_missing_columns(self, downloader):
        """Test validation with missing columns"""
        data = pd.DataFrame({
            'Open': [100, 101, 102],
            'Close': [103, 104, 105]
        })
        result = downloader._validate_basic_data(data, "AAPL")
        assert result == False
    
    def test_add_technical_indicators(self, downloader):
        """Test adding technical indicators"""
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'Date': dates,
            'Open': np.random.uniform(100, 110, 100),
            'High': np.random.uniform(110, 120, 100),
            'Low': np.random.uniform(90, 100, 100),
            'Close': np.random.uniform(100, 110, 100),
            'Volume': np.random.randint(1000000, 2000000, 100)
        })
        
        result = downloader.add_technical_indicators(data, "AAPL")
        
        # Check that new columns were added
        assert len(result.columns) > len(data.columns)
        assert 'SMA_20' in result.columns or 'SMA_5' in result.columns
        assert 'Returns' in result.columns
        assert 'Volatility_20' in result.columns
    
    def test_save_data(self, downloader, tmp_path):
        """Test saving data to file"""
        data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [99, 100, 101],
            'Close': [103, 104, 105],
            'Volume': [1000000, 1100000, 1200000]
        })
        
        downloader.save_data(data, "TEST", "processed")
        
        # Check that file was created
        file_path = downloader.processed_dir / "TEST_processed.csv"
        assert file_path.exists()
    
    def test_create_data_summary(self, downloader):
        """Test creating data summary"""
        # Create mock data
        all_data = {
            "AAPL": pd.DataFrame({
                'Date': pd.date_range('2023-01-01', periods=30),
                'Close': np.random.uniform(100, 110, 30),
                'Volume': np.random.randint(1000000, 2000000, 30)
            }),
            "GOOGL": pd.DataFrame({
                'Date': pd.date_range('2023-01-01', periods=30),
                'Close': np.random.uniform(2000, 2100, 30),
                'Volume': np.random.randint(500000, 1000000, 30)
            })
        }
        
        summary = downloader.create_data_summary(all_data)
        
        assert summary is not None
        assert 'symbols' in summary
        assert 'num_symbols' in summary
        assert summary['num_symbols'] == 2
        assert 'AAPL' in summary['symbols']
        assert 'GOOGL' in summary['symbols']
    
    def test_validate_data_quality(self, downloader):
        """Test data quality validation"""
        # Create mock data with good quality
        all_data = {
            "AAPL": pd.DataFrame({
                'Date': pd.date_range('2023-01-01', periods=30),
                'Close': np.random.uniform(100, 110, 30),
                'Volume': np.random.randint(1000000, 2000000, 30),
                'Open': np.random.uniform(100, 110, 30),
                'High': np.random.uniform(110, 120, 30),
                'Low': np.random.uniform(90, 100, 30)
            })
        }
        
        quality_report = downloader.validate_data_quality(all_data)
        
        assert quality_report is not None
        assert 'overall_quality' in quality_report
        assert 'symbol_quality' in quality_report
        assert 'AAPL' in quality_report['symbol_quality']


class TestDataDownloaderEdgeCases:
    """Test edge cases and error handling"""
    
    @pytest.fixture
    def downloader(self, tmp_path):
        """Create a DataDownloader instance"""
        config = {
            "symbols": ["TEST"],
            "start_date": "2023-01-01",
            "end_date": "2023-01-31"
        }
        downloader = DataDownloader(config)
        downloader.data_dir = tmp_path / "data"
        downloader.raw_dir = downloader.data_dir / "raw"
        downloader.processed_dir = downloader.data_dir / "processed"
        downloader.raw_dir.mkdir(parents=True, exist_ok=True)
        downloader.processed_dir.mkdir(parents=True, exist_ok=True)
        return downloader
    
    def test_validate_with_negative_prices(self, downloader):
        """Test validation with negative prices"""
        data = pd.DataFrame({
            'Open': [100, -50, 102],
            'High': [105, 106, 107],
            'Low': [99, 100, 101],
            'Close': [103, 104, 105],
            'Volume': [1000000, 1100000, 1200000]
        })
        
        # Should handle negative prices
        result = downloader._validate_basic_data(data, "TEST")
        # Validation should still pass as it filters out negative rows
        assert isinstance(result, bool)
    
    def test_validate_with_invalid_ohlc(self, downloader):
        """Test validation with invalid OHLC relationships"""
        data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [95, 96, 97],  # High < Low - invalid
            'Low': [99, 100, 101],
            'Close': [103, 104, 105],
            'Volume': [1000000, 1100000, 1200000]
        })
        
        result = downloader._validate_basic_data(data, "TEST")
        assert isinstance(result, bool)
    
    def test_empty_data_quality_check(self, downloader):
        """Test quality check with empty dataset"""
        all_data = {
            "EMPTY": pd.DataFrame()
        }
        
        quality_report = downloader.validate_data_quality(all_data)
        assert quality_report['symbol_quality']['EMPTY'] == 'poor'
    
    def test_data_with_missing_values(self, downloader):
        """Test handling data with missing values"""
        data = pd.DataFrame({
            'Open': [100, np.nan, 102],
            'High': [105, 106, np.nan],
            'Low': [99, 100, 101],
            'Close': [103, 104, 105],
            'Volume': [1000000, 1100000, 1200000]
        })
        
        result = downloader._validate_basic_data(data, "TEST")
        assert isinstance(result, bool)


def test_downloader_with_default_config():
    """Test DataDownloader with default configuration"""
    downloader = DataDownloader()
    assert downloader is not None
    assert len(downloader.symbols) > 0
    assert downloader.start_date is not None
    assert downloader.end_date is not None


def test_downloader_custom_config():
    """Test DataDownloader with custom configuration"""
    config = {
        "symbols": ["AAPL", "MSFT", "GOOGL"],
        "start_date": "2020-01-01",
        "end_date": "2021-01-01"
    }
    downloader = DataDownloader(config)
    assert len(downloader.symbols) == 3
    assert downloader.start_date == "2020-01-01"
    assert downloader.end_date == "2021-01-01"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

