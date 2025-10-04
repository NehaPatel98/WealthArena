"""
Airflow DAG: Preprocess Market Data
Runs after data fetch to calculate technical indicators and prepare features
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.data.market_data import MarketDataProcessor, create_rolling_features, create_lag_features
    PROCESSOR_AVAILABLE = True
except ImportError:
    PROCESSOR_AVAILABLE = False

import pandas as pd
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)


def load_raw_data(**context):
    """Load raw data from previous DAG"""
    
    data_dir = Path(project_root) / 'data' / 'processed'
    
    # Get list of symbol files
    symbol_files = list(data_dir.glob("*_processed.csv"))
    
    if not symbol_files:
        raise ValueError("No data files found to process")
    
    # Load all data
    all_data = {}
    for file_path in symbol_files:
        symbol = file_path.stem.replace('_processed', '')
        try:
            df = pd.read_csv(file_path)
            all_data[symbol] = df
            logger.info(f"✅ Loaded {symbol}: {len(df)} records")
        except Exception as e:
            logger.error(f"❌ Error loading {symbol}: {e}")
    
    # Push to XCom
    summary = {
        'symbols': list(all_data.keys()),
        'total_records': sum(len(df) for df in all_data.values()),
        'timestamp': datetime.now().isoformat()
    }
    
    context['task_instance'].xcom_push(key='load_summary', value=summary)
    
    logger.info(f"📊 Loaded {len(all_data)} symbols with {summary['total_records']} total records")
    
    return summary


def calculate_features(**context):
    """Calculate additional features and indicators"""
    
    data_dir = Path(project_root) / 'data' / 'processed'
    feature_dir = Path(project_root) / 'data' / 'features'
    feature_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    load_summary = context['task_instance'].xcom_pull(
        task_ids='load_raw_data',
        key='load_summary'
    )
    
    processed_count = 0
    feature_stats = {}
    
    for symbol in load_summary['symbols']:
        try:
            # Load processed data
            file_path = data_dir / f"{symbol}_processed.csv"
            df = pd.read_csv(file_path)
            
            # Calculate additional features
            if len(df) > 0:
                # Returns if not already present
                if 'Returns' not in df.columns and 'Close' in df.columns:
                    df['Returns'] = df['Close'].pct_change()
                
                # Volatility
                if 'Close' in df.columns:
                    df['Volatility_5'] = df['Close'].pct_change().rolling(5).std() * np.sqrt(252)
                    df['Volatility_20'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(252)
                
                # Price momentum
                if 'Close' in df.columns:
                    df['Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
                    df['Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
                
                # Volume features
                if 'Volume' in df.columns:
                    df['Volume_MA_20'] = df['Volume'].rolling(20).mean()
                    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
                
                # Fill NaN values
                df = df.fillna(method='bfill').fillna(method='ffill')
                
                # Save with features
                output_path = feature_dir / f"{symbol}_features.csv"
                df.to_csv(output_path, index=False)
                
                feature_stats[symbol] = {
                    'num_records': len(df),
                    'num_features': len(df.columns),
                    'date_range': {
                        'start': df['Date'].min() if 'Date' in df.columns else None,
                        'end': df['Date'].max() if 'Date' in df.columns else None
                    }
                }
                
                processed_count += 1
                logger.info(f"✅ {symbol}: Added features → {len(df.columns)} total features")
            
        except Exception as e:
            logger.error(f"❌ Error processing {symbol}: {e}")
    
    # Save feature stats
    stats_path = feature_dir / 'feature_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(feature_stats, f, indent=2)
    
    summary = {
        'processed_symbols': processed_count,
        'feature_stats': feature_stats,
        'timestamp': datetime.now().isoformat()
    }
    
    context['task_instance'].xcom_push(key='feature_summary', value=summary)
    
    logger.info(f"✅ Feature calculation completed for {processed_count} symbols")
    
    return summary


def validate_features(**context):
    """Validate calculated features"""
    
    feature_summary = context['task_instance'].xcom_pull(
        task_ids='calculate_features',
        key='feature_summary'
    )
    
    feature_dir = Path(project_root) / 'data' / 'features'
    
    validation_results = {
        'timestamp': datetime.now().isoformat(),
        'all_valid': True,
        'issues': []
    }
    
    for symbol, stats in feature_summary['feature_stats'].items():
        file_path = feature_dir / f"{symbol}_features.csv"
        
        if not file_path.exists():
            validation_results['all_valid'] = False
            validation_results['issues'].append(f"Missing feature file for {symbol}")
        else:
            try:
                df = pd.read_csv(file_path)
                
                # Check for NaN values
                nan_count = df.isnull().sum().sum()
                if nan_count > 0:
                    logger.warning(f"⚠️ {symbol}: {nan_count} NaN values found")
                
                # Check for infinite values
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                inf_count = np.isinf(df[numeric_cols]).sum().sum()
                if inf_count > 0:
                    validation_results['all_valid'] = False
                    validation_results['issues'].append(f"{symbol}: {inf_count} infinite values")
                
                logger.info(f"✅ {symbol}: {len(df)} records, {len(df.columns)} features validated")
                
            except Exception as e:
                validation_results['all_valid'] = False
                validation_results['issues'].append(f"Error validating {symbol}: {str(e)}")
    
    if not validation_results['all_valid']:
        logger.warning(f"⚠️ Validation issues: {validation_results['issues']}")
    else:
        logger.info("✅ All features validated successfully")
    
    return validation_results


def notify_completion(**context):
    """Notify that preprocessing is complete"""
    
    feature_summary = context['task_instance'].xcom_pull(
        task_ids='calculate_features',
        key='feature_summary'
    )
    
    logger.info("=" * 60)
    logger.info("🔧 DATA PREPROCESSING COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Processed Symbols: {feature_summary['processed_symbols']}")
    logger.info(f"Timestamp: {feature_summary['timestamp']}")
    
    for symbol, stats in feature_summary['feature_stats'].items():
        logger.info(f"  {symbol}: {stats['num_records']} records, {stats['num_features']} features")
    
    logger.info("=" * 60)
    
    return "success"


# Define DAG
default_args = {
    'owner': 'wealtharena',
    'depends_on_past': False,
    'email': ['admin@wealtharena.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'preprocess_market_data',
    default_args=default_args,
    description='Preprocess market data and calculate features',
    schedule_interval='0 19 * * 1-5',  # Run at 7 PM on weekdays (after fetch DAG)
    start_date=days_ago(1),
    catchup=False,
    tags=['preprocessing', 'features', 'data-pipeline'],
) as dag:
    
    # Wait for fetch DAG to complete
    wait_for_fetch = ExternalTaskSensor(
        task_id='wait_for_fetch',
        external_dag_id='fetch_market_data',
        external_task_id='notify_completion',
        timeout=3600,
        mode='reschedule',
    )
    
    # Task 1: Load raw data
    load_task = PythonOperator(
        task_id='load_raw_data',
        python_callable=load_raw_data,
        provide_context=True,
    )
    
    # Task 2: Calculate features
    calculate_task = PythonOperator(
        task_id='calculate_features',
        python_callable=calculate_features,
        provide_context=True,
    )
    
    # Task 3: Validate features
    validate_task = PythonOperator(
        task_id='validate_features',
        python_callable=validate_features,
        provide_context=True,
    )
    
    # Task 4: Notify completion
    notify_task = PythonOperator(
        task_id='notify_completion',
        python_callable=notify_completion,
        provide_context=True,
    )
    
    # Define task dependencies
    wait_for_fetch >> load_task >> calculate_task >> validate_task >> notify_task

