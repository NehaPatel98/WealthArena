"""
Airflow DAG: Fetch Market Data from yfinance
Runs daily to fetch latest market data
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from download_market_data import DataDownloader
import pandas as pd
import json
import logging

logger = logging.getLogger(__name__)


def fetch_data(**context):
    """Fetch market data from yfinance"""
    
    # Configuration
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    
    # Get date range (last 90 days for demo)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    
    config = {
        'symbols': symbols,
        'start_date': start_date,
        'end_date': end_date
    }
    
    logger.info(f"Fetching data for {symbols} from {start_date} to {end_date}")
    
    # Create downloader
    downloader = DataDownloader(config)
    
    # Download data
    all_data = downloader.download_all_data()
    
    # Save summary
    summary = {
        'execution_date': context['execution_date'].isoformat(),
        'symbols': list(all_data.keys()),
        'num_symbols': len(all_data),
        'date_range': {
            'start': start_date,
            'end': end_date
        }
    }
    
    # Push to XCom for next task
    context['task_instance'].xcom_push(key='data_summary', value=summary)
    
    logger.info(f"✅ Successfully fetched data for {len(all_data)} symbols")
    
    return summary


def validate_data(**context):
    """Validate fetched data quality"""
    
    # Pull summary from previous task
    summary = context['task_instance'].xcom_pull(
        task_ids='fetch_data', 
        key='data_summary'
    )
    
    logger.info(f"Validating data for {summary['num_symbols']} symbols")
    
    # Check data directory
    data_dir = Path(project_root) / 'data' / 'processed'
    
    validation_results = {
        'timestamp': datetime.now().isoformat(),
        'symbols_validated': summary['symbols'],
        'all_valid': True,
        'issues': []
    }
    
    for symbol in summary['symbols']:
        file_path = data_dir / f"{symbol}_processed.csv"
        
        if not file_path.exists():
            validation_results['all_valid'] = False
            validation_results['issues'].append(f"Missing data file for {symbol}")
        else:
            # Check file is not empty
            try:
                df = pd.read_csv(file_path)
                if len(df) == 0:
                    validation_results['all_valid'] = False
                    validation_results['issues'].append(f"Empty data file for {symbol}")
                else:
                    logger.info(f"✅ {symbol}: {len(df)} records validated")
            except Exception as e:
                validation_results['all_valid'] = False
                validation_results['issues'].append(f"Error reading {symbol}: {str(e)}")
    
    # Save validation report
    report_path = data_dir.parent / 'validation_report.json'
    with open(report_path, 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    if not validation_results['all_valid']:
        raise ValueError(f"Data validation failed: {validation_results['issues']}")
    
    logger.info("✅ All data validated successfully")
    
    return validation_results


def notify_completion(**context):
    """Notify that data fetch is complete"""
    
    summary = context['task_instance'].xcom_pull(
        task_ids='fetch_data',
        key='data_summary'
    )
    
    logger.info("=" * 60)
    logger.info("📊 DATA FETCH COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Symbols: {', '.join(summary['symbols'])}")
    logger.info(f"Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    logger.info(f"Execution Date: {summary['execution_date']}")
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
    'fetch_market_data',
    default_args=default_args,
    description='Fetch market data from yfinance',
    schedule_interval='0 18 * * 1-5',  # Run at 6 PM on weekdays
    start_date=days_ago(1),
    catchup=False,
    tags=['market-data', 'yfinance', 'data-ingestion'],
) as dag:
    
    # Task 1: Fetch data
    fetch_task = PythonOperator(
        task_id='fetch_data',
        python_callable=fetch_data,
        provide_context=True,
    )
    
    # Task 2: Validate data
    validate_task = PythonOperator(
        task_id='validate_data',
        python_callable=validate_data,
        provide_context=True,
    )
    
    # Task 3: Notify completion
    notify_task = PythonOperator(
        task_id='notify_completion',
        python_callable=notify_completion,
        provide_context=True,
    )
    
    # Define task dependencies
    fetch_task >> validate_task >> notify_task

