"""
Specialized Agent Factory for WealthArena Trading System

This module creates specialized RL agents for different asset types with
optimized configurations and reward functions.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging

from ..environments.base_trading_env import BaseTradingEnv, TradingEnvConfig
from ..data.asx.asx_symbols import get_all_symbols, get_asx_200_symbols
from ..data.currencies.currency_pairs import get_currency_pairs_by_category, get_major_currencies
from ..data.crypto.cryptocurrencies import get_major_cryptocurrencies, get_cryptocurrencies_by_category
from ..data.commodities.commodities import get_major_commodities, get_high_volatility_commodities

logger = logging.getLogger(__name__)


class AssetType(Enum):
    """Asset type enumeration"""
    ASX_STOCKS = "asx_stocks"
    ETF = "etf"
    REIT = "reit"
    CURRENCY_PAIRS = "currency_pairs"
    US_STOCKS = "us_stocks"
    COMMODITIES = "commodities"
    CRYPTOCURRENCIES = "cryptocurrencies"


@dataclass
class SpecializedAgentConfig:
    """Configuration for specialized agents"""
    asset_type: AssetType
    num_assets: int
    episode_length: int = 252
    lookback_window_size: int = 30
    initial_cash: float = 1_000_000.0
    
    # Asset-specific parameters
    symbols: List[str] = None
    reward_weights: Dict[str, float] = None
    risk_limits: Dict[str, float] = None
    
    # Training parameters
    learning_rate: float = 3e-4
    batch_size: int = 4000
    gamma: float = 0.99
    lambda_: float = 0.95
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = self._get_default_symbols()
        
        if self.reward_weights is None:
            self.reward_weights = self._get_default_reward_weights()
        
        if self.risk_limits is None:
            self.risk_limits = self._get_default_risk_limits()
    
    def _get_default_symbols(self) -> List[str]:
        """Get default symbols for asset type"""
        if self.asset_type == AssetType.ASX_STOCKS:
            return get_asx_200_symbols()[:self.num_assets]
        elif self.asset_type == AssetType.CURRENCY_PAIRS:
            return get_currency_pairs_by_category("Major_Pairs")[:self.num_assets]
        elif self.asset_type == AssetType.CRYPTOCURRENCIES:
            return get_major_cryptocurrencies()[:self.num_assets]
        elif self.asset_type == AssetType.ETF:
            return [f"ETF_{i}" for i in range(self.num_assets)]
        elif self.asset_type == AssetType.REIT:
            return [f"REIT_{i}" for i in range(self.num_assets)]
        elif self.asset_type == AssetType.US_STOCKS:
            return [f"US_{i}" for i in range(self.num_assets)]
        elif self.asset_type == AssetType.COMMODITIES:
            return [f"COMM_{i}" for i in range(self.num_assets)]
        else:
            return [f"ASSET_{i}" for i in range(self.num_assets)]
    
    def _get_default_reward_weights(self) -> Dict[str, float]:
        """Get default reward weights for asset type"""
        base_weights = {
            "profit": 2.0,
            "risk": 0.5,
            "cost": 0.1,
            "stability": 0.05,
            "sharpe": 1.0,
            "momentum": 0.3,
            "diversification": 0.2
        }
        
        if self.asset_type == AssetType.ASX_STOCKS:
            return base_weights
        elif self.asset_type == AssetType.CURRENCY_PAIRS:
            return {**base_weights, "momentum": 0.5, "stability": 0.03}
        elif self.asset_type == AssetType.CRYPTOCURRENCIES:
            return {**base_weights, "momentum": 0.6, "risk": 0.3}
        elif self.asset_type == AssetType.ETF:
            return {**base_weights, "stability": 0.1, "diversification": 0.3}
        elif self.asset_type == AssetType.REIT:
            return {**base_weights, "diversification": 0.3, "stability": 0.08}
        elif self.asset_type == AssetType.US_STOCKS:
            return base_weights
        elif self.asset_type == AssetType.COMMODITIES:
            return {**base_weights, "momentum": 0.4, "risk": 0.3}
        else:
            return base_weights
    
    def _get_default_risk_limits(self) -> Dict[str, float]:
        """Get default risk limits for asset type"""
        base_limits = {
            "max_position_size": 0.15,
            "max_portfolio_risk": 0.12,
            "max_drawdown_limit": 0.15,
            "var_confidence": 0.95,
            "correlation_limit": 0.7
        }
        
        if self.asset_type == AssetType.ASX_STOCKS:
            return base_limits
        elif self.asset_type == AssetType.CURRENCY_PAIRS:
            return {**base_limits, "max_position_size": 0.20, "max_portfolio_risk": 0.15}
        elif self.asset_type == AssetType.CRYPTOCURRENCIES:
            return {**base_limits, "max_position_size": 0.10, "max_portfolio_risk": 0.20}
        elif self.asset_type == AssetType.ETF:
            return {**base_limits, "max_position_size": 0.12, "max_portfolio_risk": 0.10}
        elif self.asset_type == AssetType.REIT:
            return {**base_limits, "max_position_size": 0.18, "max_portfolio_risk": 0.14}
        elif self.asset_type == AssetType.US_STOCKS:
            return base_limits
        elif self.asset_type == AssetType.COMMODITIES:
            return {**base_limits, "max_position_size": 0.20, "max_portfolio_risk": 0.18}
        else:
            return base_limits


class SpecializedAgentFactory:
    """Factory for creating specialized trading agents"""
    
    @staticmethod
    def create_agent_config(asset_type: AssetType, num_assets: int = None, **kwargs) -> SpecializedAgentConfig:
        """Create agent configuration for specific asset type"""
        
        # Set default number of assets based on asset type
        if num_assets is None:
            if asset_type == AssetType.ASX_STOCKS:
                num_assets = 200  # ASX 200
            elif asset_type == AssetType.CURRENCY_PAIRS:
                num_assets = 28   # 8 currencies = 28 pairs
            elif asset_type == AssetType.CRYPTOCURRENCIES:
                num_assets = 20   # Top 20 cryptos
            elif asset_type == AssetType.ETF:
                num_assets = 50   # Major ETFs
            elif asset_type == AssetType.REIT:
                num_assets = 30   # Major REITs
            elif asset_type == AssetType.US_STOCKS:
                num_assets = 50   # Major US stocks
            elif asset_type == AssetType.COMMODITIES:
                num_assets = 20   # Major commodities
            else:
                num_assets = 20
        
        return SpecializedAgentConfig(
            asset_type=asset_type,
            num_assets=num_assets,
            **kwargs
        )
    
    @staticmethod
    def create_trading_env_config(agent_config: SpecializedAgentConfig) -> TradingEnvConfig:
        """Create trading environment configuration from agent config"""
        
        return TradingEnvConfig(
            num_assets=agent_config.num_assets,
            initial_cash=agent_config.initial_cash,
            episode_length=agent_config.episode_length,
            lookback_window_size=agent_config.lookback_window_size,
            reward_weights=agent_config.reward_weights,
            max_position_size=agent_config.risk_limits["max_position_size"],
            max_portfolio_risk=agent_config.risk_limits["max_portfolio_risk"],
            max_drawdown_limit=agent_config.risk_limits["max_drawdown_limit"]
        )
    
    @staticmethod
    def create_agent_configs_for_all_asset_types() -> Dict[AssetType, SpecializedAgentConfig]:
        """Create agent configurations for all asset types"""
        
        configs = {}
        
        for asset_type in AssetType:
            configs[asset_type] = SpecializedAgentFactory.create_agent_config(asset_type)
        
        return configs
    
    @staticmethod
    def get_asset_type_info(asset_type: AssetType) -> Dict[str, Any]:
        """Get information about an asset type"""
        
        info = {
            AssetType.ASX_STOCKS: {
                "name": "ASX Stocks",
                "description": "Australian Securities Exchange listed companies",
                "symbols": get_asx_200_symbols(),
                "volatility": "Medium",
                "liquidity": "High",
                "market_hours": "9:30 AM - 4:00 PM AEST"
            },
            AssetType.CURRENCY_PAIRS: {
                "name": "Currency Pairs",
                "description": "Major forex currency pairs",
                "symbols": get_currency_pairs_by_category("Major_Pairs"),
                "volatility": "Medium-High",
                "liquidity": "Very High",
                "market_hours": "24/7"
            },
            AssetType.CRYPTOCURRENCIES: {
                "name": "Cryptocurrencies",
                "description": "Major global cryptocurrencies",
                "symbols": get_major_cryptocurrencies(),
                "volatility": "Very High",
                "liquidity": "High",
                "market_hours": "24/7"
            },
            AssetType.ETF: {
                "name": "Exchange-Traded Funds",
                "description": "Diversified investment funds",
                "symbols": ["SPY", "QQQ", "IWM", "VTI", "VEA", "VWO"],
                "volatility": "Low-Medium",
                "liquidity": "High",
                "market_hours": "9:30 AM - 4:00 PM EST"
            },
            AssetType.REIT: {
                "name": "Real Estate Investment Trusts",
                "description": "Real estate investment vehicles",
                "symbols": ["VNQ", "SCHH", "IYR", "XLRE", "FREL"],
                "volatility": "Medium",
                "liquidity": "Medium-High",
                "market_hours": "9:30 AM - 4:00 PM EST"
            },
            AssetType.US_STOCKS: {
                "name": "US Stocks",
                "description": "Major US market influencers",
                "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META"],
                "volatility": "Medium-High",
                "liquidity": "Very High",
                "market_hours": "9:30 AM - 4:00 PM EST"
            },
            AssetType.COMMODITIES: {
                "name": "Commodities",
                "description": "Major tradable commodities",
                "symbols": ["GOLD", "SILVER", "OIL", "GAS", "WHEAT", "CORN"],
                "volatility": "High",
                "liquidity": "Medium-High",
                "market_hours": "24/7"
            }
        }
        
        return info.get(asset_type, {})


def create_specialized_agents() -> Dict[AssetType, SpecializedAgentConfig]:
    """Create all specialized agent configurations"""
    
    logger.info("Creating specialized agent configurations for all asset types...")
    
    agents = {}
    
    # ASX Stocks Agent
    agents[AssetType.ASX_STOCKS] = SpecializedAgentFactory.create_agent_config(
        AssetType.ASX_STOCKS,
        num_assets=200,
        episode_length=252,
        lookback_window_size=30
    )
    
    # Currency Pairs Agent
    agents[AssetType.CURRENCY_PAIRS] = SpecializedAgentFactory.create_agent_config(
        AssetType.CURRENCY_PAIRS,
        num_assets=28,
        episode_length=252,
        lookback_window_size=14
    )
    
    # Cryptocurrency Agent
    agents[AssetType.CRYPTOCURRENCIES] = SpecializedAgentFactory.create_agent_config(
        AssetType.CRYPTOCURRENCIES,
        num_assets=20,
        episode_length=252,
        lookback_window_size=14
    )
    
    # ETF Agent
    agents[AssetType.ETF] = SpecializedAgentFactory.create_agent_config(
        AssetType.ETF,
        num_assets=50,
        episode_length=252,
        lookback_window_size=20
    )
    
    # REIT Agent
    agents[AssetType.REIT] = SpecializedAgentFactory.create_agent_config(
        AssetType.REIT,
        num_assets=30,
        episode_length=252,
        lookback_window_size=20
    )
    
    # US Stocks Agent
    agents[AssetType.US_STOCKS] = SpecializedAgentFactory.create_agent_config(
        AssetType.US_STOCKS,
        num_assets=50,
        episode_length=252,
        lookback_window_size=30
    )
    
    # Commodities Agent
    agents[AssetType.COMMODITIES] = SpecializedAgentFactory.create_agent_config(
        AssetType.COMMODITIES,
        num_assets=15,
        symbols=get_high_volatility_commodities()[:15],
        episode_length=252,
        lookback_window_size=20
    )
    
    logger.info(f"Created {len(agents)} specialized agent configurations")
    
    return agents


if __name__ == "__main__":
    # Test the factory
    agents = create_specialized_agents()
    
    for asset_type, config in agents.items():
        print(f"\n{asset_type.value.upper()}:")
        print(f"  Assets: {config.num_assets}")
        print(f"  Symbols: {config.symbols[:5]}...")
        print(f"  Reward weights: {config.reward_weights}")
        print(f"  Risk limits: {config.risk_limits}")
