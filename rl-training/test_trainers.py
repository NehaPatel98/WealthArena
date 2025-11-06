#!/usr/bin/env python3
"""
Test script for all trainers

This script tests that all trainers can be imported and initialized correctly.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all trainers can be imported"""
    print("🧪 Testing imports...")
    
    try:
        from src.training.asx_stocks_trainer import ASXStocksTrainer, ASXStocksConfig
        print("✅ ASX Stocks Trainer imported successfully")
    except Exception as e:
        print(f"❌ ASX Stocks Trainer import failed: {e}")
        return False
    
    try:
        from src.training.currency_pairs_trainer import CurrencyPairsTrainer, CurrencyPairsConfig
        print("✅ Currency Pairs Trainer imported successfully")
    except Exception as e:
        print(f"❌ Currency Pairs Trainer import failed: {e}")
        return False
    
    try:
        from src.training.cryptocurrency_trainer import CryptocurrencyTrainer, CryptocurrencyConfig
        print("✅ Cryptocurrency Trainer imported successfully")
    except Exception as e:
        print(f"❌ Cryptocurrency Trainer import failed: {e}")
        return False
    
    try:
        from src.training.etf_trainer import ETFTrainer, ETFConfig
        print("✅ ETF Trainer imported successfully")
    except Exception as e:
        print(f"❌ ETF Trainer import failed: {e}")
        return False
    
    try:
        from src.training.master_trainer import MasterTrainer, MasterConfig
        print("✅ Master Trainer imported successfully")
    except Exception as e:
        print(f"❌ Master Trainer import failed: {e}")
        return False
    
    return True

def test_initialization():
    """Test that all trainers can be initialized"""
    print("\n🧪 Testing initialization...")
    
    try:
        from src.training.asx_stocks_trainer import ASXStocksTrainer, ASXStocksConfig
        from src.data.asx.asx_symbols import get_asx_200_symbols
        
        config = ASXStocksConfig(symbols=get_asx_200_symbols()[:5])
        trainer = ASXStocksTrainer(config)
        print("✅ ASX Stocks Trainer initialized successfully")
    except Exception as e:
        print(f"❌ ASX Stocks Trainer initialization failed: {e}")
        return False
    
    try:
        from src.training.currency_pairs_trainer import CurrencyPairsTrainer, CurrencyPairsConfig
        from src.data.currencies.currency_pairs import get_currency_pairs_by_category
        
        config = CurrencyPairsConfig(currency_pairs=get_currency_pairs_by_category("Major_Pairs")[:5])
        trainer = CurrencyPairsTrainer(config)
        print("✅ Currency Pairs Trainer initialized successfully")
    except Exception as e:
        print(f"❌ Currency Pairs Trainer initialization failed: {e}")
        return False
    
    try:
        from src.training.cryptocurrency_trainer import CryptocurrencyTrainer, CryptocurrencyConfig
        from src.data.crypto.cryptocurrencies import get_major_cryptocurrencies
        
        config = CryptocurrencyConfig(symbols=get_major_cryptocurrencies()[:5])
        trainer = CryptocurrencyTrainer(config)
        print("✅ Cryptocurrency Trainer initialized successfully")
    except Exception as e:
        print(f"❌ Cryptocurrency Trainer initialization failed: {e}")
        return False
    
    try:
        from src.training.etf_trainer import ETFTrainer, ETFConfig
        
        config = ETFConfig(symbols=["SPY", "QQQ", "IWM", "VTI", "VEA"])
        trainer = ETFTrainer(config)
        print("✅ ETF Trainer initialized successfully")
    except Exception as e:
        print(f"❌ ETF Trainer initialization failed: {e}")
        return False
    
    try:
        from src.training.master_trainer import MasterTrainer, MasterConfig
        
        config = MasterConfig()
        trainer = MasterTrainer(config)
        print("✅ Master Trainer initialized successfully")
    except Exception as e:
        print(f"❌ Master Trainer initialization failed: {e}")
        return False
    
    return True

def test_data_generation():
    """Test that data generation works"""
    print("\n🧪 Testing data generation...")
    
    try:
        from src.training.currency_pairs_trainer import CurrencyPairsTrainer, CurrencyPairsConfig
        from src.data.currencies.currency_pairs import get_currency_pairs_by_category
        
        config = CurrencyPairsConfig(currency_pairs=get_currency_pairs_by_category("Major_Pairs")[:3])
        trainer = CurrencyPairsTrainer(config)
        
        data = trainer.generate_synthetic_data(100)  # Small dataset
        print(f"✅ Data generation successful: {data.shape} shape")
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        return False
    
    return True

def test_training_simulation():
    """Test that training simulation works"""
    print("\n🧪 Testing training simulation...")
    
    try:
        from src.training.currency_pairs_trainer import CurrencyPairsTrainer, CurrencyPairsConfig
        from src.data.currencies.currency_pairs import get_currency_pairs_by_category
        
        config = CurrencyPairsConfig(currency_pairs=get_currency_pairs_by_category("Major_Pairs")[:3])
        trainer = CurrencyPairsTrainer(config)
        
        results = trainer.train_agent()
        print(f"✅ Training simulation successful: {results['final_reward']:.4f} final reward")
    except Exception as e:
        print(f"❌ Training simulation failed: {e}")
        return False
    
    return True

def test_backtesting():
    """Test that backtesting works"""
    print("\n🧪 Testing backtesting...")
    
    try:
        from src.training.currency_pairs_trainer import CurrencyPairsTrainer, CurrencyPairsConfig
        from src.data.currencies.currency_pairs import get_currency_pairs_by_category
        
        config = CurrencyPairsConfig(currency_pairs=get_currency_pairs_by_category("Major_Pairs")[:3])
        trainer = CurrencyPairsTrainer(config)
        
        results = trainer.backtest_agent()
        print(f"✅ Backtesting successful: {len(results['portfolio_values'])} portfolio values")
    except Exception as e:
        print(f"❌ Backtesting failed: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🚀 WEALTHARENA TRAINER TEST SUITE")
    print("=" * 50)
    
    # Configure logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise
    
    tests = [
        ("Import Test", test_imports),
        ("Initialization Test", test_initialization),
        ("Data Generation Test", test_data_generation),
        ("Training Simulation Test", test_training_simulation),
        ("Backtesting Test", test_backtesting)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print(f"\n📊 TEST RESULTS:")
    print(f"  Passed: {passed}/{total}")
    print(f"  Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Trainers are ready to use.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
