#!/usr/bin/env python3
"""
Week 2 Data Download and Preparation Script

This script downloads and prepares sample market data for Week 2 testing.
It includes data validation, technical indicator calculation, and data quality checks.
"""

import os
import sys
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
from typing import Dict, List, Any, Optional
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataDownloader:
    """
    Data downloader and preparer for Week 2
    
    Downloads market data from yfinance, calculates technical indicators,
    and prepares data for training and testing.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize data downloader"""
        self.config = config or {}
        self.symbols = self.config.get("symbols", [
            "AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", 
            "NVDA", "META", "NFLX", "AMD", "INTC"
        ])
        self.start_date = self.config.get("start_date", "2022-01-01")
        self.end_date = self.config.get("end_date", "2024-01-01")
        self.data_dir = Path("data")
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        
        logger.info(f"DataDownloader initialized for {len(self.symbols)} symbols")
    
    def download_symbol_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Download data for a single symbol"""
        try:
            logger.info(f"Downloading data for {symbol}...")
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=self.start_date,
                end=self.end_date,
                interval="1d",
                auto_adjust=True,
                back_adjust=True
            )
            
            if data.empty:
                logger.warning(f"No data available for {symbol}")
                return None
            
            # Ensure proper column names
            data.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Add date column
            data['Date'] = data.index
            data = data.reset_index(drop=True)
            
            # Basic data validation
            if not self._validate_basic_data(data, symbol):
                return None
            
            logger.info(f"Downloaded {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error downloading {symbol}: {e}")
            return None
    
    def _validate_basic_data(self, data: pd.DataFrame, symbol: str) -> bool:
        """Validate basic data quality"""
        if data.empty:
            logger.error(f"Empty data for {symbol}")
            return False
        
        # Check for required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            logger.error(f"Missing columns for {symbol}: {missing_cols}")
            return False
        
        # Check for negative prices
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if (data[col] <= 0).any():
                logger.warning(f"Negative/zero prices in {col} for {symbol}")
                data = data[data[col] > 0]
        
        # Check for invalid OHLC relationships
        invalid_ohlc = (
            (data["High"] < data["Low"]) |
            (data["High"] < data["Open"]) |
            (data["High"] < data["Close"]) |
            (data["Low"] > data["Open"]) |
            (data["Low"] > data["Close"])
        )
        
        if invalid_ohlc.any():
            logger.warning(f"Invalid OHLC relationships in {invalid_ohlc.sum()} records for {symbol}")
            data = data[~invalid_ohlc]
        
        # Check for missing values
        if data.isnull().any().any():
            logger.warning(f"Missing values found in {symbol}")
            data = data.dropna()
        
        return len(data) > 0
    
    def add_technical_indicators(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add technical indicators to market data"""
        if data.empty:
            return data
        
        df = data.copy()
        logger.info(f"Adding technical indicators for {symbol}...")
        
        try:
            # Try to use TA-Lib for advanced indicators
            import talib
            
            # Moving averages
            df['SMA_5'] = talib.SMA(df['Close'], timeperiod=5)
            df['SMA_10'] = talib.SMA(df['Close'], timeperiod=10)
            df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
            df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
            df['SMA_200'] = talib.SMA(df['Close'], timeperiod=200)
            
            # Exponential moving averages
            df['EMA_12'] = talib.EMA(df['Close'], timeperiod=12)
            df['EMA_26'] = talib.EMA(df['Close'], timeperiod=26)
            df['EMA_50'] = talib.EMA(df['Close'], timeperiod=50)
            
            # RSI
            df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
            df['RSI_6'] = talib.RSI(df['Close'], timeperiod=6)
            df['RSI_21'] = talib.RSI(df['Close'], timeperiod=21)
            
            # MACD
            df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['Close'])
            df['MACD_fast'], df['MACD_slow'], df['MACD_hist_fast'] = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
            
            # Bollinger Bands
            df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2)
            df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
            df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
            
            # ATR
            df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
            df['ATR_5'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=5)
            
            # OBV
            df['OBV'] = talib.OBV(df['Close'], df['Volume'])
            
            # Stochastic
            df['STOCH_K'], df['STOCH_D'] = talib.STOCH(df['High'], df['Low'], df['Close'])
            df['STOCH_RSI_K'], df['STOCH_RSI_D'] = talib.STOCHRSI(df['Close'])
            
            # Williams %R
            df['WILLR'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)
            
            # CCI
            df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)
            
            # ADX
            df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
            df['PLUS_DI'] = talib.PLUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
            df['MINUS_DI'] = talib.MINUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
            
            # Aroon
            df['AROON_UP'], df['AROON_DOWN'] = talib.AROON(df['High'], df['Low'], timeperiod=14)
            df['AROONOSC'] = talib.AROONOSC(df['High'], df['Low'], timeperiod=14)
            
            # Money Flow Index
            df['MFI'] = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=14)
            
            # Ultimate Oscillator
            df['ULTOSC'] = talib.ULTOSC(df['High'], df['Low'], df['Close'])
            
            # Commodity Channel Index
            df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)
            
            logger.info(f"Added TA-Lib indicators for {symbol}")
            
        except ImportError:
            logger.warning("TA-Lib not available, using simplified indicators")
            # Simplified indicators without TA-Lib
            df['SMA_5'] = df['Close'].rolling(window=5).mean()
            df['SMA_10'] = df['Close'].rolling(window=10).mean()
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            
            df['EMA_12'] = df['Close'].ewm(span=12).mean()
            df['EMA_26'] = df['Close'].ewm(span=26).mean()
            df['EMA_50'] = df['Close'].ewm(span=50).mean()
            
            # Simple RSI calculation
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Simple MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_hist'] = df['MACD'] - df['MACD_signal']
            
            # Simple Bollinger Bands
            df['BB_middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
            df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
            df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
            df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
            
            # Simple ATR
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df['ATR'] = true_range.rolling(14).mean()
            
            # Simple OBV
            df['OBV'] = (df['Volume'] * np.sign(df['Close'].diff())).cumsum()
            
            logger.info(f"Added simplified indicators for {symbol}")
        
        # Calculate returns and volatility
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volatility_5'] = df['Returns'].rolling(window=5).std() * np.sqrt(252)
        df['Volatility_10'] = df['Returns'].rolling(window=10).std() * np.sqrt(252)
        df['Volatility_20'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
        df['Volatility_50'] = df['Returns'].rolling(window=50).std() * np.sqrt(252)
        
        # Calculate price momentum
        df['Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        df['Momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
        
        # Calculate volume indicators
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
        df['Volume_Price_Trend'] = (df['Volume'] * df['Returns']).cumsum()
        
        # Calculate support and resistance levels
        df['Resistance_20'] = df['High'].rolling(window=20).max()
        df['Support_20'] = df['Low'].rolling(window=20).min()
        df['Price_Position'] = (df['Close'] - df['Support_20']) / (df['Resistance_20'] - df['Support_20'])
        
        # Calculate trend indicators
        df['Trend_5'] = np.where(df['Close'] > df['SMA_5'], 1, -1)
        df['Trend_10'] = np.where(df['Close'] > df['SMA_10'], 1, -1)
        df['Trend_20'] = np.where(df['Close'] > df['SMA_20'], 1, -1)
        df['Trend_50'] = np.where(df['Close'] > df['SMA_50'], 1, -1)
        
        # Calculate crossover signals
        df['SMA_5_10_Cross'] = np.where(df['SMA_5'] > df['SMA_10'], 1, -1)
        df['SMA_10_20_Cross'] = np.where(df['SMA_10'] > df['SMA_20'], 1, -1)
        df['EMA_12_26_Cross'] = np.where(df['EMA_12'] > df['EMA_26'], 1, -1)
        
        # Fill NaN values
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        # Replace infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        logger.info(f"Technical indicators added for {symbol}: {len(df.columns)} total columns")
        return df
    
    def save_data(self, data: pd.DataFrame, symbol: str, data_type: str = "processed"):
        """Save data to file"""
        if data_type == "raw":
            file_path = self.raw_dir / f"{symbol}_raw.csv"
        else:
            file_path = self.processed_dir / f"{symbol}_processed.csv"
        
        data.to_csv(file_path, index=False)
        logger.info(f"Saved {data_type} data for {symbol}: {file_path}")
    
    def download_all_data(self) -> Dict[str, pd.DataFrame]:
        """Download and process data for all symbols"""
        logger.info(f"Starting data download for {len(self.symbols)} symbols")
        
        all_data = {}
        successful_downloads = 0
        
        for symbol in self.symbols:
            try:
                # Download raw data
                raw_data = self.download_symbol_data(symbol)
                if raw_data is None:
                    continue
                
                # Save raw data
                self.save_data(raw_data, symbol, "raw")
                
                # Add technical indicators
                processed_data = self.add_technical_indicators(raw_data, symbol)
                
                # Save processed data
                self.save_data(processed_data, symbol, "processed")
                
                all_data[symbol] = processed_data
                successful_downloads += 1
                
                logger.info(f"✅ {symbol}: {len(processed_data)} records, {len(processed_data.columns)} features")
                
            except Exception as e:
                logger.error(f"❌ {symbol}: {e}")
                continue
        
        logger.info(f"Data download completed: {successful_downloads}/{len(self.symbols)} symbols successful")
        return all_data
    
    def create_data_summary(self, all_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Create summary of downloaded data"""
        summary = {
            "download_date": datetime.now().isoformat(),
            "symbols": list(all_data.keys()),
            "num_symbols": len(all_data),
            "date_range": {
                "start": self.start_date,
                "end": self.end_date
            },
            "symbol_details": {}
        }
        
        for symbol, data in all_data.items():
            if not data.empty:
                summary["symbol_details"][symbol] = {
                    "records": len(data),
                    "features": len(data.columns),
                    "date_range": {
                        "start": data['Date'].min().strftime('%Y-%m-%d'),
                        "end": data['Date'].max().strftime('%Y-%m-%d')
                    },
                    "price_range": {
                        "min": float(data['Close'].min()),
                        "max": float(data['Close'].max()),
                        "mean": float(data['Close'].mean())
                    },
                    "volume_range": {
                        "min": float(data['Volume'].min()),
                        "max": float(data['Volume'].max()),
                        "mean": float(data['Volume'].mean())
                    }
                }
        
        return summary
    
    def validate_data_quality(self, all_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Validate data quality across all symbols"""
        quality_report = {
            "overall_quality": "good",
            "issues": [],
            "symbol_quality": {}
        }
        
        for symbol, data in all_data.items():
            symbol_issues = []
            
            # Check data completeness
            if data.empty:
                symbol_issues.append("Empty dataset")
                quality_report["symbol_quality"][symbol] = "poor"
                continue
            
            # Check for missing values
            missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100
            if missing_pct > 5:
                symbol_issues.append(f"High missing data: {missing_pct:.1f}%")
            
            # Check for extreme price movements
            returns = data['Close'].pct_change()
            extreme_moves = (returns.abs() > 0.5).sum()
            if extreme_moves > 0:
                symbol_issues.append(f"Extreme price movements: {extreme_moves}")
            
            # Check for volume anomalies
            volume_mean = data['Volume'].mean()
            volume_std = data['Volume'].std()
            extreme_volume = (data['Volume'] > volume_mean + 3 * volume_std).sum()
            if extreme_volume > 0:
                symbol_issues.append(f"Volume anomalies: {extreme_volume}")
            
            # Check for data gaps
            if len(data) > 1:
                time_diff = pd.to_datetime(data['Date']).diff()
                large_gaps = (time_diff > pd.Timedelta(days=7)).sum()
                if large_gaps > 0:
                    symbol_issues.append(f"Large time gaps: {large_gaps}")
            
            # Determine quality level
            if len(symbol_issues) == 0:
                quality_report["symbol_quality"][symbol] = "excellent"
            elif len(symbol_issues) <= 2:
                quality_report["symbol_quality"][symbol] = "good"
            else:
                quality_report["symbol_quality"][symbol] = "poor"
                quality_report["issues"].extend([f"{symbol}: {issue}" for issue in symbol_issues])
        
        # Overall quality assessment
        poor_quality_count = sum(1 for q in quality_report["symbol_quality"].values() if q == "poor")
        if poor_quality_count > len(all_data) * 0.3:
            quality_report["overall_quality"] = "poor"
        elif poor_quality_count > 0:
            quality_report["overall_quality"] = "fair"
        
        return quality_report


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and prepare market data for Week 2")
    parser.add_argument("--symbols", nargs="+", default=[
        "AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", 
        "NVDA", "META", "NFLX", "AMD", "INTC"
    ], help="Symbols to download")
    parser.add_argument("--start-date", default="2022-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2024-01-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--config", default="config/week2_config.yaml", help="Config file path")
    
    args = parser.parse_args()
    
    # Load config if available
    config = {}
    if os.path.exists(args.config):
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Override with command line arguments
    config.update({
        "symbols": args.symbols,
        "start_date": args.start_date,
        "end_date": args.end_date
    })
    
    # Create downloader
    downloader = DataDownloader(config)
    
    try:
        # Download all data
        all_data = downloader.download_all_data()
        
        if not all_data:
            logger.error("No data downloaded successfully")
            sys.exit(1)
        
        # Create summary
        summary = downloader.create_data_summary(all_data)
        
        # Validate data quality
        quality_report = downloader.validate_data_quality(all_data)
        
        # Save reports
        with open("data_download_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        with open("data_quality_report.json", "w") as f:
            json.dump(quality_report, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "="*60)
        print("📊 DATA DOWNLOAD SUMMARY")
        print("="*60)
        print(f"✅ Successfully downloaded: {len(all_data)} symbols")
        print(f"📅 Date range: {args.start_date} to {args.end_date}")
        print(f"📁 Data saved to: data/processed/")
        print(f"📋 Summary saved to: data_download_summary.json")
        print(f"🔍 Quality report saved to: data_quality_report.json")
        
        print(f"\n📈 SYMBOL DETAILS:")
        for symbol, details in summary["symbol_details"].items():
            print(f"  {symbol}: {details['records']} records, {details['features']} features")
        
        print(f"\n🔍 DATA QUALITY: {quality_report['overall_quality'].upper()}")
        if quality_report["issues"]:
            print("⚠️  Issues found:")
            for issue in quality_report["issues"][:5]:  # Show first 5 issues
                print(f"    - {issue}")
            if len(quality_report["issues"]) > 5:
                print(f"    ... and {len(quality_report['issues']) - 5} more issues")
        
        print("\n🎉 Data download completed successfully!")
        print("\nNext steps:")
        print("1. Review the data quality report")
        print("2. Run: python quick_start_week2.py")
        print("3. Start training: python src/training/train_week2.py")
        
    except Exception as e:
        logger.error(f"Data download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
