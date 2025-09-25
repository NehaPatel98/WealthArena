"""
Portfolio Management System for WealthArena Trading

This module provides comprehensive portfolio management capabilities including
position tracking, risk management, and performance metrics calculation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TradeType(Enum):
    """Trade type enumeration"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class Trade:
    """Trade record data class"""
    timestamp: datetime
    symbol: str
    trade_type: TradeType
    shares: float
    price: float
    value: float
    commission: float
    portfolio_value: float


@dataclass
class RiskMetrics:
    """Risk metrics data class"""
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    calmar_ratio: float
    sortino_ratio: float


class Portfolio:
    """
    Portfolio management class for WealthArena trading system
    
    Handles position tracking, trade execution, risk management,
    and performance metrics calculation.
    """
    
    def __init__(self, 
                 initial_cash: float = 100000,
                 commission_rate: float = 0.001,
                 risk_free_rate: float = 0.02):
        """
        Initialize portfolio
        
        Args:
            initial_cash: Initial cash amount
            commission_rate: Commission rate per trade (0.1% default)
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.commission_rate = commission_rate
        self.risk_free_rate = risk_free_rate
        
        # Position tracking
        self.positions = {}  # {symbol: shares}
        self.avg_cost = {}   # {symbol: average_cost_per_share}
        
        # Trade history
        self.trade_history = []
        self.portfolio_history = []
        
        # Performance tracking
        self.daily_returns = []
        self.performance_metrics = {}
        
        # Risk management
        self.max_position_size = 0.3  # 30% max position per asset
        self.max_portfolio_risk = 0.15  # 15% max portfolio risk
        self.stop_loss_threshold = 0.1  # 10% stop loss
        
        logger.info(f"Portfolio initialized with ${initial_cash:,.2f}")
    
    def execute_trade(self, 
                     symbol: str, 
                     action: float, 
                     price: float, 
                     timestamp: datetime = None) -> bool:
        """
        Execute a trade
        
        Args:
            symbol: Asset symbol
            action: Action value (-1 to 1, negative=sell, positive=buy)
            price: Current price of the asset
            timestamp: Trade timestamp
            
        Returns:
            bool: True if trade was executed successfully
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        if abs(action) < 0.01:  # Minimum trade threshold
            return False
        
        # Check risk constraints
        if not self._check_risk_constraints(symbol, action, price):
            logger.warning(f"Trade rejected due to risk constraints: {symbol}")
            return False
        
        # Execute trade
        if action > 0:  # Buy
            return self._execute_buy(symbol, action, price, timestamp)
        else:  # Sell
            return self._execute_sell(symbol, abs(action), price, timestamp)
    
    def _execute_buy(self, symbol: str, action: float, price: float, timestamp: datetime) -> bool:
        """Execute buy order"""
        # Calculate trade amount
        trade_value = self.cash * action
        shares = trade_value / price
        commission = trade_value * self.commission_rate
        total_cost = trade_value + commission
        
        # Check if we have enough cash
        if total_cost > self.cash:
            logger.warning(f"Insufficient cash for buy order: {symbol}")
            return False
        
        # Execute trade
        self.cash -= total_cost
        self.positions[symbol] = self.positions.get(symbol, 0) + shares
        
        # Update average cost
        if symbol in self.avg_cost:
            total_shares = self.positions[symbol]
            old_value = (total_shares - shares) * self.avg_cost[symbol]
            new_value = old_value + trade_value
            self.avg_cost[symbol] = new_value / total_shares
        else:
            self.avg_cost[symbol] = price
        
        # Record trade
        self._record_trade(symbol, TradeType.BUY, shares, price, trade_value, commission, timestamp)
        
        logger.debug(f"Buy executed: {shares:.2f} shares of {symbol} at ${price:.2f}")
        return True
    
    def _execute_sell(self, symbol: str, action: float, price: float, timestamp: datetime) -> bool:
        """Execute sell order"""
        # Check if we have position
        if symbol not in self.positions or self.positions[symbol] <= 0:
            logger.warning(f"No position to sell: {symbol}")
            return False
        
        # Calculate shares to sell
        shares_to_sell = self.positions[symbol] * action
        trade_value = shares_to_sell * price
        commission = trade_value * self.commission_rate
        net_proceeds = trade_value - commission
        
        # Execute trade
        self.cash += net_proceeds
        self.positions[symbol] -= shares_to_sell
        
        # Remove from avg_cost if position is closed
        if self.positions[symbol] <= 0:
            self.positions[symbol] = 0
            if symbol in self.avg_cost:
                del self.avg_cost[symbol]
        
        # Record trade
        self._record_trade(symbol, TradeType.SELL, shares_to_sell, price, trade_value, commission, timestamp)
        
        logger.debug(f"Sell executed: {shares_to_sell:.2f} shares of {symbol} at ${price:.2f}")
        return True
    
    def _check_risk_constraints(self, symbol: str, action: float, price: float) -> bool:
        """Check risk management constraints"""
        # Check position size limit
        if action > 0:  # Buy action
            trade_value = self.cash * action
            portfolio_value = self.get_portfolio_value({symbol: price})
            position_ratio = trade_value / portfolio_value
            
            if position_ratio > self.max_position_size:
                logger.warning(f"Position size limit exceeded: {position_ratio:.2%} > {self.max_position_size:.2%}")
                return False
        
        # Check portfolio risk
        if self._calculate_portfolio_risk() > self.max_portfolio_risk:
            logger.warning(f"Portfolio risk limit exceeded")
            return False
        
        return True
    
    def _calculate_portfolio_risk(self) -> float:
        """Calculate current portfolio risk"""
        if len(self.daily_returns) < 20:
            return 0.0
        
        # Calculate portfolio volatility
        returns = np.array(self.daily_returns[-20:])
        volatility = np.std(returns) * np.sqrt(252)
        
        return volatility
    
    def _record_trade(self, 
                     symbol: str, 
                     trade_type: TradeType, 
                     shares: float, 
                     price: float, 
                     value: float, 
                     commission: float, 
                     timestamp: datetime):
        """Record trade in history"""
        portfolio_value = self.get_portfolio_value({symbol: price})
        
        trade = Trade(
            timestamp=timestamp,
            symbol=symbol,
            trade_type=trade_type,
            shares=shares,
            price=price,
            value=value,
            commission=commission,
            portfolio_value=portfolio_value
        )
        
        self.trade_history.append(trade)
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate current portfolio value"""
        asset_value = sum(
            shares * current_prices.get(symbol, 0) 
            for symbol, shares in self.positions.items()
        )
        return self.cash + asset_value
    
    def get_position_weights(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """Get position weights as percentage of portfolio"""
        portfolio_value = self.get_portfolio_value(current_prices)
        
        if portfolio_value <= 0:
            return {}
        
        weights = {}
        for symbol, shares in self.positions.items():
            if shares > 0:
                position_value = shares * current_prices.get(symbol, 0)
                weights[symbol] = position_value / portfolio_value
        
        return weights
    
    def update_performance(self, current_prices: Dict[str, float]):
        """Update performance metrics"""
        current_value = self.get_portfolio_value(current_prices)
        self.portfolio_history.append(current_value)
        
        # Calculate daily return
        if len(self.portfolio_history) > 1:
            daily_return = (current_value - self.portfolio_history[-2]) / self.portfolio_history[-2]
            self.daily_returns.append(daily_return)
        
        # Update performance metrics
        self.performance_metrics = self.calculate_performance_metrics()
    
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        if len(self.portfolio_history) < 2:
            return {}
        
        # Basic metrics
        total_return = (self.portfolio_history[-1] / self.initial_cash - 1) * 100
        annualized_return = (self.portfolio_history[-1] / self.initial_cash) ** (252 / len(self.portfolio_history)) - 1
        
        # Risk metrics
        if len(self.daily_returns) > 1:
            returns = np.array(self.daily_returns)
            volatility = np.std(returns) * np.sqrt(252)
            sharpe_ratio = (np.mean(returns) * 252 - self.risk_free_rate) / volatility if volatility > 0 else 0
            
            # Drawdown metrics
            peak = np.maximum.accumulate(self.portfolio_history)
            drawdown = (np.array(self.portfolio_history) - peak) / peak
            max_drawdown = np.min(drawdown) * 100
            
            # VaR and CVaR
            var_95 = np.percentile(returns, 5) * 100
            var_99 = np.percentile(returns, 1) * 100
            cvar_95 = np.mean(returns[returns <= np.percentile(returns, 5)]) * 100
            cvar_99 = np.mean(returns[returns <= np.percentile(returns, 1)]) * 100
            
            # Additional metrics
            calmar_ratio = annualized_return / abs(max_drawdown / 100) if max_drawdown != 0 else 0
            
            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = (np.mean(returns) * 252 - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
            
        else:
            volatility = sharpe_ratio = max_drawdown = 0
            var_95 = var_99 = cvar_95 = cvar_99 = 0
            calmar_ratio = sortino_ratio = 0
        
        return {
            "total_return": total_return,
            "annualized_return": annualized_return * 100,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "var_95": var_95,
            "var_99": var_99,
            "cvar_95": cvar_95,
            "cvar_99": cvar_99,
            "calmar_ratio": calmar_ratio,
            "sortino_ratio": sortino_ratio,
            "num_trades": len(self.trade_history),
            "total_commission": sum(trade.commission for trade in self.trade_history)
        }
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Get comprehensive risk metrics"""
        if len(self.daily_returns) < 2:
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        returns = np.array(self.daily_returns)
        volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = (np.mean(returns) * 252 - self.risk_free_rate) / volatility if volatility > 0 else 0
        
        # Drawdown
        peak = np.maximum.accumulate(self.portfolio_history)
        drawdown = (np.array(self.portfolio_history) - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # VaR and CVaR
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        cvar_95 = np.mean(returns[returns <= var_95])
        cvar_99 = np.mean(returns[returns <= var_99])
        
        # Calmar ratio
        annualized_return = (self.portfolio_history[-1] / self.initial_cash) ** (252 / len(self.portfolio_history)) - 1
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (np.mean(returns) * 252 - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        return RiskMetrics(
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio
        )
    
    def get_trade_summary(self) -> Dict[str, Any]:
        """Get summary of all trades"""
        if not self.trade_history:
            return {}
        
        # Group by symbol
        symbol_trades = {}
        for trade in self.trade_history:
            if trade.symbol not in symbol_trades:
                symbol_trades[trade.symbol] = []
            symbol_trades[trade.symbol].append(trade)
        
        # Calculate summary for each symbol
        summary = {}
        for symbol, trades in symbol_trades.items():
            buy_trades = [t for t in trades if t.trade_type == TradeType.BUY]
            sell_trades = [t for t in trades if t.trade_type == TradeType.SELL]
            
            total_bought = sum(t.shares for t in buy_trades)
            total_sold = sum(t.shares for t in sell_trades)
            total_commission = sum(t.commission for t in trades)
            
            summary[symbol] = {
                "total_trades": len(trades),
                "buy_trades": len(buy_trades),
                "sell_trades": len(sell_trades),
                "total_bought": total_bought,
                "total_sold": total_sold,
                "net_position": total_bought - total_sold,
                "total_commission": total_commission
            }
        
        return summary
    
    def reset(self):
        """Reset portfolio to initial state"""
        self.cash = self.initial_cash
        self.positions = {}
        self.avg_cost = {}
        self.trade_history = []
        self.portfolio_history = [self.initial_cash]
        self.daily_returns = []
        self.performance_metrics = {}
        
        logger.info("Portfolio reset to initial state")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current portfolio status"""
        return {
            "cash": self.cash,
            "positions": self.positions.copy(),
            "avg_cost": self.avg_cost.copy(),
            "num_trades": len(self.trade_history),
            "portfolio_value": self.portfolio_history[-1] if self.portfolio_history else self.initial_cash,
            "performance_metrics": self.performance_metrics.copy()
        }


class PortfolioManager:
    """
    Portfolio manager for multiple portfolios
    
    Manages multiple portfolios and provides portfolio-level analytics.
    """
    
    def __init__(self, num_portfolios: int = 1, initial_cash: float = 100000):
        """
        Initialize portfolio manager
        
        Args:
            num_portfolios: Number of portfolios to manage
            initial_cash: Initial cash per portfolio
        """
        self.portfolios = [Portfolio(initial_cash) for _ in range(num_portfolios)]
        self.num_portfolios = num_portfolios
        
        logger.info(f"Portfolio manager initialized with {num_portfolios} portfolios")
    
    def get_portfolio(self, index: int) -> Portfolio:
        """Get portfolio by index"""
        if 0 <= index < self.num_portfolios:
            return self.portfolios[index]
        else:
            raise IndexError(f"Portfolio index {index} out of range")
    
    def execute_trade(self, 
                     portfolio_index: int, 
                     symbol: str, 
                     action: float, 
                     price: float, 
                     timestamp: datetime = None) -> bool:
        """Execute trade for specific portfolio"""
        portfolio = self.get_portfolio(portfolio_index)
        return portfolio.execute_trade(symbol, action, price, timestamp)
    
    def update_all_performance(self, current_prices: Dict[str, float]):
        """Update performance for all portfolios"""
        for portfolio in self.portfolios:
            portfolio.update_performance(current_prices)
    
    def get_aggregate_metrics(self) -> Dict[str, float]:
        """Get aggregate performance metrics across all portfolios"""
        if not self.portfolios:
            return {}
        
        # Calculate aggregate metrics
        total_values = [p.portfolio_history[-1] if p.portfolio_history else p.initial_cash for p in self.portfolios]
        total_initial = sum(p.initial_cash for p in self.portfolios)
        
        aggregate_return = (sum(total_values) / total_initial - 1) * 100
        
        # Calculate portfolio-weighted metrics
        all_returns = []
        for portfolio in self.portfolios:
            if portfolio.daily_returns:
                all_returns.extend(portfolio.daily_returns)
        
        if all_returns:
            returns_array = np.array(all_returns)
            volatility = np.std(returns_array) * np.sqrt(252)
            sharpe_ratio = (np.mean(returns_array) * 252 - 0.02) / volatility if volatility > 0 else 0
        else:
            volatility = sharpe_ratio = 0
        
        return {
            "aggregate_return": aggregate_return,
            "aggregate_volatility": volatility,
            "aggregate_sharpe_ratio": sharpe_ratio,
            "num_portfolios": self.num_portfolios,
            "total_value": sum(total_values),
            "total_initial": total_initial
        }


if __name__ == "__main__":
    # Test the portfolio manager
    portfolio = Portfolio(100000)
    
    # Test trades
    portfolio.execute_trade("AAPL", 0.1, 150.0)  # Buy 10% of portfolio
    portfolio.execute_trade("GOOGL", 0.05, 2500.0)  # Buy 5% of portfolio
    
    # Update performance
    current_prices = {"AAPL": 155.0, "GOOGL": 2600.0}
    portfolio.update_performance(current_prices)
    
    # Print status
    print("Portfolio Status:")
    print(f"Portfolio Value: ${portfolio.get_portfolio_value(current_prices):,.2f}")
    print(f"Positions: {portfolio.positions}")
    print(f"Performance Metrics: {portfolio.performance_metrics}")
    
    # Test risk metrics
    risk_metrics = portfolio.get_risk_metrics()
    print(f"Risk Metrics: {risk_metrics}")
