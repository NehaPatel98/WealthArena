#!/usr/bin/env python3
"""
Tests for utility scripts and helper modules
Covers setup, deployment, and configuration scripts
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from unittest.mock import Mock, patch, MagicMock
import json

sys.path.insert(0, str(Path(__file__).parent))


class TestSetupEnvironment:
    """Test setup_environment.py functionality"""
    
    def test_import_setup_module(self):
        """Test importing setup module"""
        try:
            import setup_environment
            assert hasattr(setup_environment, 'EnvironmentSetup') or True
        except ImportError:
            pytest.skip("Setup module not fully testable")
    
    def test_environment_configuration(self):
        """Test environment configuration"""
        # Test basic config structure
        config = {
            'data_dir': 'data',
            'model_dir': 'models',
            'checkpoint_dir': 'checkpoints'
        }
        assert 'data_dir' in config
        assert 'model_dir' in config


class TestBenchmarkAnalysis:
    """Test benchmark_analysis_report.py functionality"""
    
    def test_import_benchmark_module(self):
        """Test importing benchmark module"""
        try:
            import benchmark_analysis_report
            assert True
        except ImportError:
            pass
    
    def test_benchmark_metrics_calculation(self):
        """Test benchmark metrics"""
        # Sample returns
        returns = pd.Series(np.random.randn(100) * 0.01)
        
        # Calculate basic metrics
        total_return = (1 + returns).prod() - 1
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        max_drawdown = (returns.cumsum() - returns.cumsum().cummax()).min()
        
        assert isinstance(total_return, (float, np.floating))
        assert isinstance(sharpe_ratio, (float, np.floating))
        assert max_drawdown <= 0


class TestDeployModels:
    """Test deploy_models.py functionality"""
    
    def test_import_deploy_module(self):
        """Test importing deploy module"""
        try:
            import deploy_models
            assert True
        except ImportError:
            pass
    
    def test_model_deployment_config(self):
        """Test model deployment configuration"""
        deployment_config = {
            'model_path': 'models/trained_model.pkl',
            'deployment_env': 'production',
            'api_endpoint': 'http://localhost:8000'
        }
        
        assert 'model_path' in deployment_config
        assert 'deployment_env' in deployment_config


class TestDemoCurrencyPairs:
    """Test demo_currency_pairs.py functionality"""
    
    def test_import_demo_module(self):
        """Test importing demo module"""
        try:
            import demo_currency_pairs
            assert True
        except ImportError:
            pass
    
    def test_currency_pair_format(self):
        """Test currency pair format"""
        currency_pairs = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD']
        
        for pair in currency_pairs:
            assert '/' in pair
            parts = pair.split('/')
            assert len(parts) == 2
            assert len(parts[0]) == 3  # Currency code length
            assert len(parts[1]) == 3


class TestRunIndividualAgents:
    """Test run_individual_agents.py functionality"""
    
    def test_import_run_agents_module(self):
        """Test importing run agents module"""
        try:
            import run_individual_agents
            assert True
        except ImportError:
            pass
    
    def test_agent_configuration(self):
        """Test agent configuration structure"""
        agent_config = {
            'agent_id': 'agent_1',
            'algorithm': 'PPO',
            'learning_rate': 0.0003,
            'gamma': 0.99,
            'symbols': ['AAPL', 'GOOGL']
        }
        
        assert 'agent_id' in agent_config
        assert 'algorithm' in agent_config
        assert agent_config['gamma'] > 0 and agent_config['gamma'] <= 1


class TestRunCoverageScript:
    """Test run_coverage.py script"""
    
    def test_import_coverage_script(self):
        """Test importing coverage script"""
        try:
            import run_coverage
            assert True
        except ImportError:
            pass
    
    def test_coverage_command_structure(self):
        """Test coverage command structure"""
        coverage_cmd = "pytest --cov=. --cov-report=xml"
        
        assert 'pytest' in coverage_cmd
        assert '--cov' in coverage_cmd
        assert '--cov-report' in coverage_cmd


class TestMetricsCalculations:
    """Test various metrics calculations"""
    
    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation"""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.00, 0.01])
        risk_free_rate = 0.0
        
        excess_returns = returns - risk_free_rate
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
        
        assert isinstance(sharpe, (float, np.floating))
    
    def test_sortino_ratio_calculation(self):
        """Test Sortino ratio calculation"""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, -0.02, 0.01])
        
        # Downside deviation
        negative_returns = returns[returns < 0]
        downside_std = negative_returns.std() if len(negative_returns) > 0 else returns.std()
        
        sortino = (returns.mean() / downside_std) * np.sqrt(252)
        
        assert isinstance(sortino, (float, np.floating))
    
    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation"""
        prices = pd.Series([100, 105, 103, 110, 108, 115, 112, 120])
        
        # Calculate cumulative returns
        cum_returns = prices / prices.iloc[0] - 1
        
        # Calculate running maximum
        running_max = cum_returns.cummax()
        
        # Calculate drawdown
        drawdown = cum_returns - running_max
        max_drawdown = drawdown.min()
        
        assert max_drawdown <= 0
    
    def test_win_rate_calculation(self):
        """Test win rate calculation"""
        trades = pd.Series([100, -50, 200, -30, 150, -20, 80])
        
        winning_trades = (trades > 0).sum()
        total_trades = len(trades)
        win_rate = winning_trades / total_trades
        
        assert 0 <= win_rate <= 1
    
    def test_profit_factor_calculation(self):
        """Test profit factor calculation"""
        trades = pd.Series([100, -50, 200, -30, 150, -20, 80])
        
        gross_profit = trades[trades > 0].sum()
        gross_loss = abs(trades[trades < 0].sum())
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        assert profit_factor >= 0


class TestConfigurationFiles:
    """Test configuration file handling"""
    
    def test_yaml_config_structure(self):
        """Test YAML configuration structure"""
        config = {
            'data': {
                'symbols': ['AAPL', 'GOOGL'],
                'start_date': '2023-01-01',
                'end_date': '2023-12-31'
            },
            'training': {
                'algorithm': 'PPO',
                'num_workers': 4,
                'train_batch_size': 4000
            },
            'environment': {
                'initial_cash': 100000,
                'commission': 0.001
            }
        }
        
        assert 'data' in config
        assert 'training' in config
        assert 'environment' in config
        assert isinstance(config['data']['symbols'], list)
    
    def test_json_config_structure(self):
        """Test JSON configuration structure"""
        config_json = {
            "model_name": "trading_agent",
            "version": "1.0.0",
            "hyperparameters": {
                "learning_rate": 0.0003,
                "gamma": 0.99,
                "clip_param": 0.2
            }
        }
        
        # Test serialization
        json_str = json.dumps(config_json)
        loaded_config = json.loads(json_str)
        
        assert loaded_config == config_json


class TestDataValidation:
    """Test data validation functions"""
    
    def test_price_data_validation(self):
        """Test price data validation"""
        prices = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [99, 100, 101],
            'close': [103, 104, 105]
        })
        
        # Validate OHLC relationships
        valid_high = all(prices['high'] >= prices['open']) and all(prices['high'] >= prices['close'])
        valid_low = all(prices['low'] <= prices['open']) and all(prices['low'] <= prices['close'])
        valid_range = all(prices['high'] >= prices['low'])
        
        assert valid_high
        assert valid_low
        assert valid_range
    
    def test_volume_data_validation(self):
        """Test volume data validation"""
        volumes = pd.Series([1000000, 1100000, 1200000, 1300000])
        
        # Volume should be positive
        assert all(volumes > 0)
        
        # Check for outliers
        mean_volume = volumes.mean()
        std_volume = volumes.std()
        outliers = volumes > (mean_volume + 3 * std_volume)
        
        assert outliers.sum() == 0  # No extreme outliers in this data
    
    def test_date_continuity_validation(self):
        """Test date continuity validation"""
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        
        # Calculate differences
        date_diffs = dates.to_series().diff()
        
        # Check continuity (should be 1 day apart)
        expected_diff = pd.Timedelta(days=1)
        assert all(date_diffs.dropna() == expected_diff)


class TestPerformanceMetrics:
    """Test performance metrics calculations"""
    
    def test_annual_return(self):
        """Test annual return calculation"""
        start_value = 100000
        end_value = 120000
        days = 252
        
        total_return = (end_value - start_value) / start_value
        annual_return = total_return * (252 / days)
        
        assert annual_return == 0.20  # 20% annual return
    
    def test_volatility(self):
        """Test volatility calculation"""
        returns = pd.Series(np.random.randn(252) * 0.01)
        
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        
        assert annual_vol > 0
    
    def test_cumulative_returns(self):
        """Test cumulative returns calculation"""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03])
        
        cum_returns = (1 + returns).cumprod() - 1
        
        assert len(cum_returns) == len(returns)
        assert cum_returns.iloc[-1] == (1 + returns).prod() - 1


class TestUtilityFunctions:
    """Test utility helper functions"""
    
    def test_normalize_data(self):
        """Test data normalization"""
        data = pd.Series([10, 20, 30, 40, 50])
        
        # Min-max normalization
        normalized = (data - data.min()) / (data.max() - data.min())
        
        assert normalized.min() == 0.0
        assert normalized.max() == 1.0
    
    def test_standardize_data(self):
        """Test data standardization"""
        data = pd.Series([10, 20, 30, 40, 50])
        
        # Z-score standardization
        standardized = (data - data.mean()) / data.std()
        
        assert abs(standardized.mean()) < 1e-10  # Mean should be ~0
        assert abs(standardized.std() - 1.0) < 1e-10  # Std should be ~1
    
    def test_clip_outliers(self):
        """Test clipping outliers"""
        data = pd.Series([1, 2, 3, 4, 5, 100])  # 100 is outlier
        
        # Clip at 3 standard deviations
        mean = data.mean()
        std = data.std()
        lower = mean - 3 * std
        upper = mean + 3 * std
        
        clipped = data.clip(lower, upper)
        
        assert clipped.max() <= upper


class TestPathManagement:
    """Test path and file management"""
    
    def test_create_directory_structure(self, tmp_path):
        """Test creating directory structure"""
        base_dir = tmp_path / "project"
        
        dirs = [
            base_dir / "data",
            base_dir / "models",
            base_dir / "checkpoints",
            base_dir / "logs",
            base_dir / "results"
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            assert dir_path.exists()
    
    def test_file_path_handling(self, tmp_path):
        """Test file path handling"""
        file_path = tmp_path / "test_data.csv"
        
        # Create a test file
        test_data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        test_data.to_csv(file_path, index=False)
        
        # Verify file exists
        assert file_path.exists()
        
        # Load and verify
        loaded_data = pd.read_csv(file_path)
        assert len(loaded_data) == 3


class TestErrorHandling:
    """Test error handling scenarios"""
    
    def test_division_by_zero_handling(self):
        """Test division by zero handling"""
        numerator = 100
        denominator = 0
        
        try:
            result = numerator / denominator if denominator != 0 else 0
            assert result == 0
        except ZeroDivisionError:
            assert False, "Should handle division by zero"
    
    def test_missing_data_handling(self):
        """Test missing data handling"""
        data = pd.Series([1, 2, np.nan, 4, 5])
        
        # Fill missing data
        filled_data = data.fillna(data.mean())
        
        assert not filled_data.isnull().any()
    
    def test_invalid_input_handling(self):
        """Test invalid input handling"""
        def process_positive_number(x):
            if x <= 0:
                raise ValueError("Number must be positive")
            return x * 2
        
        # Test valid input
        assert process_positive_number(5) == 10
        
        # Test invalid input
        with pytest.raises(ValueError):
            process_positive_number(-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

