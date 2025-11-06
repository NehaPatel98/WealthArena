"""
Data Loader Utility for WealthArena RL Training

This module provides utilities to load processed CSV files from Phase 2
and convert them to the format expected by the trading environment.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def load_processed_data(
    asset_class: str,
    symbols: List[str] = None,
    start_date: str = None,
    end_date: str = None
) -> pd.DataFrame:
    """
    Load processed CSVs from Phase 2 and convert to multi-level DataFrame format.
    
    Args:
        asset_class: Asset class directory name (e.g., 'stocks', 'crypto', 'forex')
        symbols: Optional list of symbols to filter (if None, loads all)
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
    
    Returns:
        Multi-level DataFrame with (symbol, column) structure
    """
    data_dir = Path("data/processed") / asset_class
    
    if not data_dir.exists():
        raise FileNotFoundError(f"No data directory for {asset_class}: {data_dir}")
    
    # Get all CSV files
    csv_files = list(data_dir.glob("*_processed.csv"))
    
    if not csv_files:
        logger.warning(f"No processed CSV files found in {data_dir}")
        return pd.DataFrame()
    
    # Filter by symbols if provided - use exact matching
    if symbols:
        symbol_set = set(symbols)
        filtered_files = []
        for f in csv_files:
            # Extract symbol from filename (e.g., "BHP.AX_processed.csv" -> "BHP.AX")
            stem_symbol = f.stem.replace("_processed", "")
            if stem_symbol in symbol_set:
                filtered_files.append(f)
        csv_files = filtered_files
    
    if not csv_files:
        logger.warning(f"No CSV files found matching symbols: {symbols}")
        return pd.DataFrame()
    
    # Load each CSV
    symbol_data = {}
    for csv_file in csv_files:
        try:
            # Extract symbol name from filename (e.g., "BHP.AX_processed.csv" -> "BHP.AX")
            symbol = csv_file.stem.replace("_processed", "")
            
            # Read CSV
            df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            
            # Normalize: ensure Date column exists if index is datetime
            # If index is datetime but Date column is missing, add it
            if isinstance(df.index, pd.DatetimeIndex) and 'Date' not in df.columns:
                df = df.reset_index()
                df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
                logger.debug(f"Normalized processed data for {symbol}: added Date column from index")
            elif not isinstance(df.index, pd.DatetimeIndex) and 'Date' not in df.columns:
                # If neither index nor Date column exists, create Date from row number
                logger.warning(f"No date information found for {symbol}, using row numbers")
                df['Date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
            
            # Filter by date range if provided
            # Use Date column if available, otherwise use index
            date_col = df.index if isinstance(df.index, pd.DatetimeIndex) else df.get('Date')
            if date_col is not None:
                if start_date:
                    if isinstance(df.index, pd.DatetimeIndex):
                        df = df[df.index >= start_date]
                    elif 'Date' in df.columns:
                        df = df[df['Date'] >= start_date]
                if end_date:
                    if isinstance(df.index, pd.DatetimeIndex):
                        df = df[df.index <= end_date]
                    elif 'Date' in df.columns:
                        df = df[df['Date'] <= end_date]
            
            if len(df) > 0:
                symbol_data[symbol] = df
                logger.debug(f"Loaded {len(df)} records for {symbol}")
            else:
                logger.warning(f"No data for {symbol} after date filtering")
                
        except Exception as e:
            logger.error(f"Error loading {csv_file}: {e}")
            continue
    
    if not symbol_data:
        logger.warning(f"No data loaded for {asset_class}")
        return pd.DataFrame()
    
    # Convert to multi-level DataFrame
    multi_level_data = create_multi_level_dataframe(symbol_data)
    
    logger.info(f"Loaded {len(multi_level_data)} records for {len(symbol_data)} symbols from {asset_class}")
    
    return multi_level_data


def get_available_symbols(asset_class: str) -> List[str]:
    """
    Scan data directory and extract available symbol names.
    
    Args:
        asset_class: Asset class directory name
    
    Returns:
        Sorted list of available symbols
    """
    data_dir = Path("data/processed") / asset_class
    
    if not data_dir.exists():
        return []
    
    # Get all CSV files
    csv_files = list(data_dir.glob("*_processed.csv"))
    
    # Extract symbols from filenames
    symbols = []
    for csv_file in csv_files:
        symbol = csv_file.stem.replace("_processed", "")
        symbols.append(symbol)
    
    return sorted(symbols)


def validate_data_quality(
    data: pd.DataFrame,
    min_rows: int = 100
) -> Tuple[bool, List[str]]:
    """
    Validate data quality and completeness.
    
    Args:
        data: Multi-level DataFrame to validate
        min_rows: Minimum number of rows required per symbol
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    if data.empty:
        issues.append("DataFrame is empty")
        return False, issues
    
    # Check for minimum rows
    if len(data) < min_rows:
        issues.append(f"Insufficient rows: {len(data)} < {min_rows}")
    
    # Check for null values in critical columns
    if isinstance(data.columns, pd.MultiIndex):
        # Multi-level columns
        for level0 in data.columns.get_level_values(0).unique():
            for col in ['Close', 'Volume']:
                if (level0, col) in data.columns:
                    null_count = data[(level0, col)].isnull().sum()
                    if null_count > 0:
                        issues.append(f"{level0}.{col} has {null_count} null values")
    else:
        # Single-level columns
        for col in ['Close', 'Volume']:
            if col in data.columns:
                null_count = data[col].isnull().sum()
                if null_count > 0:
                    issues.append(f"{col} has {null_count} null values")
    
    # Check date range coverage
    if hasattr(data.index, 'min') and hasattr(data.index, 'max'):
        date_range = data.index.max() - data.index.min()
        if date_range.days < 30:
            issues.append(f"Date range too short: {date_range.days} days")
    
    is_valid = len(issues) == 0
    return is_valid, issues


def prepare_training_data(
    asset_class: str,
    config: Dict
) -> Dict[str, pd.DataFrame]:
    """
    Load data for asset class and split into train/validation/test sets.
    
    Args:
        asset_class: Asset class name
        config: Configuration dict with symbols, dates, etc.
    
    Returns:
        Dict with 'train', 'validation', 'test' DataFrames
    """
    # Load all data
    full_data = load_processed_data(
        asset_class=asset_class,
        symbols=config.get('symbols'),
        start_date=config.get('start_date'),
        end_date=config.get('end_date')
    )
    
    if full_data.empty:
        logger.warning(f"No data loaded for {asset_class}")
        return {
            'train': pd.DataFrame(),
            'validation': pd.DataFrame(),
            'test': pd.DataFrame()
        }
    
    # Temporal split (not random to avoid data leakage)
    n = len(full_data)
    train_end = int(n * config.get('train_split', 0.8))
    val_end = int(n * (config.get('train_split', 0.8) + config.get('val_split', 0.1)))
    
    train_data = full_data.iloc[:train_end]
    val_data = full_data.iloc[train_end:val_end] if val_end > train_end else pd.DataFrame()
    test_data = full_data.iloc[val_end:] if val_end < n else pd.DataFrame()
    
    logger.info(f"Split {asset_class} data: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    
    return {
        'train': train_data,
        'validation': val_data,
        'test': test_data
    }


def create_multi_level_dataframe(
    symbol_data_dict: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Convert dict of {symbol: DataFrame} to multi-level DataFrame.
    
    Args:
        symbol_data_dict: Dictionary mapping symbol names to DataFrames
    
    Returns:
        Multi-level DataFrame with columns (symbol, column_name)
    """
    if not symbol_data_dict:
        return pd.DataFrame()
    
    # Align all DataFrames by index (date)
    aligned_data = {}
    
    for symbol, df in symbol_data_dict.items():
        for col in df.columns:
            aligned_data[(symbol, col)] = df[col]
    
    # Create multi-level DataFrame
    multi_level_df = pd.DataFrame(aligned_data)
    
    # Align by date index
    if isinstance(multi_level_df.index, pd.DatetimeIndex):
        multi_level_df = multi_level_df.sort_index()
    
    return multi_level_df


def load_cached_market_data(
    symbols: List[str],
    max_age_days: int = 7,
    data_dir: str = "data/processed"
) -> Dict[str, pd.DataFrame]:
    """
    Load cached market data from processed directory.
    
    Args:
        symbols: List of symbols to load (e.g., ['BHP.AX', 'CBA.AX'])
        max_age_days: Maximum age of cache in days (default 7)
        data_dir: Base directory for processed data (default 'data/processed')
    
    Returns:
        Dictionary mapping symbols to DataFrames
    """
    from datetime import timedelta
    from pathlib import Path
    
    cached_data = {}
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.warning(f"Data directory does not exist: {data_path}")
        return cached_data
    
    cutoff_date = datetime.now() - timedelta(days=max_age_days)
    
    for symbol in symbols:
        # Check in flat structure first
        symbol_file = data_path / f"{symbol}_processed.csv"
        if symbol_file.exists():
            # Check file age
            mtime = datetime.fromtimestamp(symbol_file.stat().st_mtime)
            if mtime >= cutoff_date:
                try:
                    df = pd.read_csv(symbol_file, index_col=0, parse_dates=True)
                    if not df.empty:
                        cached_data[symbol] = df
                        logger.info(f"Loaded cached data for {symbol} ({len(df)} rows, age: {datetime.now() - mtime})")
                        continue
                except Exception as e:
                    logger.warning(f"Failed to load cached file {symbol_file}: {e}")
        
        # Check in subdirectories (stocks/, etc.)
        for subdir in data_path.iterdir():
            if subdir.is_dir():
                symbol_file = subdir / f"{symbol}_processed.csv"
                if symbol_file.exists():
                    mtime = datetime.fromtimestamp(symbol_file.stat().st_mtime)
                    if mtime >= cutoff_date:
                        try:
                            df = pd.read_csv(symbol_file, index_col=0, parse_dates=True)
                            if not df.empty:
                                cached_data[symbol] = df
                                logger.info(f"Loaded cached data for {symbol} from {subdir.name} ({len(df)} rows, age: {datetime.now() - mtime})")
                                break
                        except Exception as e:
                            logger.warning(f"Failed to load cached file {symbol_file}: {e}")
    
    logger.info(f"Loaded {len(cached_data)}/{len(symbols)} symbols from cache")
    return cached_data


def check_data_availability(
    symbols: List[str],
    data_dir: str = "data/processed"
) -> Dict[str, Dict[str, Any]]:
    """
    Check availability of cached data for symbols.
    
    Args:
        symbols: List of symbols to check
        data_dir: Base directory for processed data (default 'data/processed')
    
    Returns:
        Dictionary mapping symbols to availability reports
    """
    from datetime import timedelta
    from pathlib import Path
    
    availability_report = {}
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.warning(f"Data directory does not exist: {data_path}")
        return {symbol: {"available": False, "reason": "Directory does not exist"} for symbol in symbols}
    
    for symbol in symbols:
        report = {
            "symbol": symbol,
            "available": False,
            "location": None,
            "age_days": None,
            "rows": None,
            "columns": None,
            "quality": "unknown"
        }
        
        # Check in flat structure
        symbol_file = data_path / f"{symbol}_processed.csv"
        if symbol_file.exists():
            try:
                mtime = datetime.fromtimestamp(symbol_file.stat().st_mtime)
                age = (datetime.now() - mtime).days
                
                df = pd.read_csv(symbol_file, index_col=0, parse_dates=True, nrows=1)
                
                report["available"] = True
                report["location"] = str(symbol_file)
                report["age_days"] = age
                report["rows"] = len(pd.read_csv(symbol_file, index_col=0))
                report["columns"] = len(df.columns)
                
                # Assess quality based on age
                if age < 7:
                    report["quality"] = "excellent"
                elif age < 30:
                    report["quality"] = "good"
                elif age < 90:
                    report["quality"] = "fair"
                else:
                    report["quality"] = "stale"
                
                availability_report[symbol] = report
                continue
            except Exception as e:
                logger.warning(f"Error checking {symbol_file}: {e}")
        
        # Check in subdirectories
        for subdir in data_path.iterdir():
            if subdir.is_dir():
                symbol_file = subdir / f"{symbol}_processed.csv"
                if symbol_file.exists():
                    try:
                        mtime = datetime.fromtimestamp(symbol_file.stat().st_mtime)
                        age = (datetime.now() - mtime).days
                        
                        df = pd.read_csv(symbol_file, index_col=0, parse_dates=True, nrows=1)
                        
                        report["available"] = True
                        report["location"] = str(symbol_file)
                        report["age_days"] = age
                        report["rows"] = len(pd.read_csv(symbol_file, index_col=0))
                        report["columns"] = len(df.columns)
                        
                        if age < 7:
                            report["quality"] = "excellent"
                        elif age < 30:
                            report["quality"] = "good"
                        elif age < 90:
                            report["quality"] = "fair"
                        else:
                            report["quality"] = "stale"
                        
                        availability_report[symbol] = report
                        break
                    except Exception as e:
                        logger.warning(f"Error checking {symbol_file}: {e}")
        
        if not report["available"]:
            report["reason"] = "File not found"
            availability_report[symbol] = report
    
    # Generate summary recommendations
    available_count = sum(1 for r in availability_report.values() if r["available"])
    logger.info(f"Data availability: {available_count}/{len(symbols)} symbols available")
    
    # Provide recommendations
    if available_count < len(symbols):
        missing_symbols = [s for s, r in availability_report.items() if not r["available"]]
        logger.info(f"Missing symbols: {missing_symbols[:10]}...")  # Show first 10
        logger.info("Recommendation: Run data collection to fetch missing symbols")
    
    # Check cache freshness
    stale_count = sum(1 for r in availability_report.values() if r.get("quality") == "stale")
    if stale_count > 0:
        logger.warning(f"{stale_count} symbols have stale cache (>90 days old)")
        logger.info("Recommendation: Refresh stale data for better training results")
    
    return availability_report