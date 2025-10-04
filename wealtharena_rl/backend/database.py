"""
Database Service
Handles data storage - works with local files now, ready for AzureSQL
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import json
from datetime import datetime
import os

logger = logging.getLogger(__name__)


class DatabaseService:
    """
    Database abstraction layer
    Currently uses local CSV/JSON files
    Ready to switch to AzureSQL when available
    """
    
    def __init__(self, connection_string: str = None):
        self.connection_string = connection_string
        self.use_azure = connection_string is not None
        
        # Local data directories (fallback)
        self.data_dir = Path(__file__).parent.parent / "data"
        self.processed_dir = self.data_dir / "processed"
        self.features_dir = self.data_dir / "features"
        self.cache_dir = self.data_dir / "cache"
        
        # Create directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if self.use_azure:
            self._init_azure_connection()
        else:
            logger.info("Using local file storage (AzureSQL not configured)")
    
    def _init_azure_connection(self):
        """Initialize AzureSQL connection"""
        try:
            from sqlalchemy import create_engine
            self.engine = create_engine(self.connection_string)
            logger.info("✅ Connected to AzureSQL")
        except Exception as e:
            logger.error(f"Failed to connect to AzureSQL: {e}")
            logger.info("Falling back to local storage")
            self.use_azure = False
    
    # ==================== RAW MARKET DATA ====================
    
    def get_raw_market_data(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """Get raw market data for symbol"""
        
        if self.use_azure:
            return self._get_from_azure('raw_market_data', symbol, days)
        else:
            return self._get_from_local('processed', symbol, days)
    
    def save_raw_market_data(self, symbol: str, data: pd.DataFrame, asset_type: str):
        """Save raw market data"""
        
        if self.use_azure:
            self._save_to_azure('raw_market_data', symbol, data, asset_type)
        else:
            self._save_to_local('processed', symbol, data)
    
    # ==================== PROCESSED FEATURES ====================
    
    def get_processed_features(self, symbol: str, days: int = None) -> Optional[pd.DataFrame]:
        """Get processed features for symbol"""
        
        if self.use_azure:
            return self._get_from_azure('processed_features', symbol, days)
        else:
            return self._get_from_local('features', symbol, days)
    
    def save_processed_features(self, symbol: str, data: pd.DataFrame):
        """Save processed features"""
        
        if self.use_azure:
            self._save_to_azure('processed_features', symbol, data)
        else:
            self._save_to_local('features', symbol, data)
    
    # ==================== MODEL PREDICTIONS ====================
    
    def get_latest_predictions(self, asset_type: str = None) -> List[Dict[str, Any]]:
        """Get latest model predictions"""
        
        if self.use_azure:
            return self._get_predictions_from_azure(asset_type)
        else:
            return self._get_predictions_from_local(asset_type)
    
    def save_prediction(self, prediction: Dict[str, Any]):
        """Save model prediction"""
        
        if self.use_azure:
            self._save_prediction_to_azure(prediction)
        else:
            self._save_prediction_to_local(prediction)
    
    # ==================== PORTFOLIO STATE ====================
    
    def get_portfolio(self, user_id: str = 'demo_user') -> Dict[str, Any]:
        """Get user portfolio"""
        
        if self.use_azure:
            return self._get_portfolio_from_azure(user_id)
        else:
            return self._get_portfolio_from_local(user_id)
    
    def update_portfolio(self, user_id: str, portfolio: Dict[str, Any]):
        """Update portfolio state"""
        
        if self.use_azure:
            self._update_portfolio_azure(user_id, portfolio)
        else:
            self._update_portfolio_local(user_id, portfolio)
    
    # ==================== LOCAL STORAGE METHODS ====================
    
    def _get_from_local(self, data_type: str, symbol: str, days: int = None) -> Optional[pd.DataFrame]:
        """Get data from local CSV files"""
        
        if data_type == 'processed':
            file_path = self.processed_dir / f"{symbol}_processed.csv"
        elif data_type == 'features':
            file_path = self.features_dir / f"{symbol}_features.csv"
        else:
            return None
        
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return None
        
        try:
            df = pd.read_csv(file_path)
            
            if days and len(df) > days:
                df = df.tail(days)
            
            return df
        
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None
    
    def _save_to_local(self, data_type: str, symbol: str, data: pd.DataFrame):
        """Save data to local CSV files"""
        
        if data_type == 'processed':
            file_path = self.processed_dir / f"{symbol}_processed.csv"
        elif data_type == 'features':
            file_path = self.features_dir / f"{symbol}_features.csv"
        else:
            return
        
        data.to_csv(file_path, index=False)
        logger.info(f"Saved {symbol} to {file_path}")
    
    def _get_predictions_from_local(self, asset_type: str = None) -> List[Dict[str, Any]]:
        """Get predictions from local cache"""
        
        cache_file = self.cache_dir / "latest_predictions.json"
        
        if not cache_file.exists():
            return []
        
        try:
            with open(cache_file, 'r') as f:
                all_predictions = json.load(f)
            
            if asset_type:
                return [p for p in all_predictions if p.get('asset_type') == asset_type]
            return all_predictions
        
        except Exception as e:
            logger.error(f"Error loading predictions: {e}")
            return []
    
    def _save_prediction_to_local(self, prediction: Dict[str, Any]):
        """Save prediction to local cache"""
        
        cache_file = self.cache_dir / "latest_predictions.json"
        
        # Load existing
        predictions = []
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                predictions = json.load(f)
        
        # Add new prediction
        predictions.append(prediction)
        
        # Keep only latest per symbol
        symbol_map = {}
        for pred in predictions:
            symbol_map[pred['symbol']] = pred
        
        predictions = list(symbol_map.values())
        
        # Save
        with open(cache_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        logger.info(f"Saved prediction for {prediction['symbol']}")
    
    def _get_portfolio_from_local(self, user_id: str) -> Dict[str, Any]:
        """Get portfolio from local storage"""
        
        portfolio_file = self.cache_dir / f"portfolio_{user_id}.json"
        
        if not portfolio_file.exists():
            # Return default empty portfolio
            return {
                'user_id': user_id,
                'cash': 100000,
                'holdings': [],
                'total_value': 100000,
                'pnl': 0
            }
        
        with open(portfolio_file, 'r') as f:
            return json.load(f)
    
    def _update_portfolio_local(self, user_id: str, portfolio: Dict[str, Any]):
        """Update portfolio in local storage"""
        
        portfolio_file = self.cache_dir / f"portfolio_{user_id}.json"
        
        with open(portfolio_file, 'w') as f:
            json.dump(portfolio, f, indent=2)
        
        logger.info(f"Updated portfolio for {user_id}")
    
    # ==================== AZURE SQL METHODS (Placeholder) ====================
    
    def _get_from_azure(self, table: str, symbol: str, days: int = None) -> Optional[pd.DataFrame]:
        """
        Get data from AzureSQL
        TODO: Implement when AzureSQL is available
        """
        # query = f"SELECT * FROM {table} WHERE symbol = '{symbol}'"
        # if days:
        #     query += f" ORDER BY date DESC LIMIT {days}"
        # return pd.read_sql(query, self.engine)
        
        # For now, fall back to local
        return self._get_from_local('features' if table == 'processed_features' else 'processed', symbol, days)
    
    def _save_to_azure(self, table: str, symbol: str, data: pd.DataFrame, asset_type: str = None):
        """
        Save data to AzureSQL
        TODO: Implement when AzureSQL is available
        """
        # data.to_sql(table, self.engine, if_exists='append', index=False)
        pass
    
    def _get_predictions_from_azure(self, asset_type: str = None) -> List[Dict[str, Any]]:
        """
        Get predictions from AzureSQL
        TODO: Implement when AzureSQL is available
        """
        # query = "SELECT * FROM model_predictions WHERE prediction_date > NOW() - INTERVAL '1 day'"
        # if asset_type:
        #     query += f" AND asset_type = '{asset_type}'"
        # df = pd.read_sql(query, self.engine)
        # return df.to_dict('records')
        
        return self._get_predictions_from_local(asset_type)
    
    def _save_prediction_to_azure(self, prediction: Dict[str, Any]):
        """
        Save prediction to AzureSQL
        TODO: Implement when AzureSQL is available
        """
        # df = pd.DataFrame([prediction])
        # df.to_sql('model_predictions', self.engine, if_exists='append', index=False)
        pass
    
    def _get_portfolio_from_azure(self, user_id: str) -> Dict[str, Any]:
        """
        Get portfolio from AzureSQL
        TODO: Implement when AzureSQL is available
        """
        return self._get_portfolio_from_local(user_id)
    
    def _update_portfolio_azure(self, user_id: str, portfolio: Dict[str, Any]):
        """
        Update portfolio in AzureSQL
        TODO: Implement when AzureSQL is available
        """
        pass


# Singleton instance
_db_service = None

def get_database() -> DatabaseService:
    """Get or create database service singleton"""
    global _db_service
    if _db_service is None:
        # Try to get AzureSQL connection string from environment
        connection_string = os.getenv('AZURE_SQL_CONNECTION_STRING')
        _db_service = DatabaseService(connection_string)
    return _db_service

