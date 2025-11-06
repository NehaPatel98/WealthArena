"""
Market Microstructure Simulation Module

This module provides realistic market microstructure simulation including:
- Order book simulation
- Transaction cost models
- Latency simulation
- Market impact modeling
- Slippage calculation
- Fill simulation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import random
from collections import deque
import math

logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

@dataclass
class Order:
    """Order representation"""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: datetime = None
    time_in_force: str = "DAY"
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class Fill:
    """Fill representation"""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    commission: float = 0.0
    slippage: float = 0.0
    market_impact: float = 0.0

class OrderBook:
    """Order book implementation"""
    
    def __init__(self, symbol: str, tick_size: float = 0.01):
        self.symbol = symbol
        self.tick_size = tick_size
        self.bids = {}  # price -> quantity
        self.asks = {}  # price -> quantity
        self.best_bid = 0.0
        self.best_ask = float('inf')
        self.mid_price = 0.0
        self.spread = 0.0
        
    def add_order(self, order: Order) -> List[Fill]:
        """Add order to book and return fills"""
        fills = []
        
        if order.side == OrderSide.BUY:
            fills = self._process_buy_order(order)
        else:
            fills = self._process_sell_order(order)
        
        self._update_best_prices()
        return fills
    
    def _process_buy_order(self, order: Order) -> List[Fill]:
        """Process buy order"""
        fills = []
        remaining_qty = order.quantity
        
        if order.order_type == OrderType.MARKET:
            # Market order - fill at best ask prices
            for price in sorted(self.asks.keys()):
                if remaining_qty <= 0:
                    break
                
                available_qty = self.asks[price]
                fill_qty = min(remaining_qty, available_qty)
                
                if fill_qty > 0:
                    fill = Fill(
                        order_id=order.id,
                        symbol=order.symbol,
                        side=order.side,
                        quantity=fill_qty,
                        price=price,
                        timestamp=datetime.now()
                    )
                    fills.append(fill)
                    
                    self.asks[price] -= fill_qty
                    if self.asks[price] <= 0:
                        del self.asks[price]
                    
                    remaining_qty -= fill_qty
        
        elif order.order_type == OrderType.LIMIT:
            # Limit order - add to book if not immediately fillable
            if order.price >= self.best_ask:
                # Can be filled immediately
                for price in sorted(self.asks.keys()):
                    if remaining_qty <= 0 or price > order.price:
                        break
                    
                    available_qty = self.asks[price]
                    fill_qty = min(remaining_qty, available_qty)
                    
                    if fill_qty > 0:
                        fill = Fill(
                            order_id=order.id,
                            symbol=order.symbol,
                            side=order.side,
                            quantity=fill_qty,
                            price=price,
                            timestamp=datetime.now()
                        )
                        fills.append(fill)
                        
                        self.asks[price] -= fill_qty
                        if self.asks[price] <= 0:
                            del self.asks[price]
                        
                        remaining_qty -= fill_qty
                
                # Add remaining quantity to book
                if remaining_qty > 0:
                    self.bids[order.price] = self.bids.get(order.price, 0) + remaining_qty
            else:
                # Add to book
                self.bids[order.price] = self.bids.get(order.price, 0) + remaining_qty
        
        return fills
    
    def _process_sell_order(self, order: Order) -> List[Fill]:
        """Process sell order"""
        fills = []
        remaining_qty = order.quantity
        
        if order.order_type == OrderType.MARKET:
            # Market order - fill at best bid prices
            for price in sorted(self.bids.keys(), reverse=True):
                if remaining_qty <= 0:
                    break
                
                available_qty = self.bids[price]
                fill_qty = min(remaining_qty, available_qty)
                
                if fill_qty > 0:
                    fill = Fill(
                        order_id=order.id,
                        symbol=order.symbol,
                        side=order.side,
                        quantity=fill_qty,
                        price=price,
                        timestamp=datetime.now()
                    )
                    fills.append(fill)
                    
                    self.bids[price] -= fill_qty
                    if self.bids[price] <= 0:
                        del self.bids[price]
                    
                    remaining_qty -= fill_qty
        
        elif order.order_type == OrderType.LIMIT:
            # Limit order - add to book if not immediately fillable
            if order.price <= self.best_bid:
                # Can be filled immediately
                for price in sorted(self.bids.keys(), reverse=True):
                    if remaining_qty <= 0 or price < order.price:
                        break
                    
                    available_qty = self.bids[price]
                    fill_qty = min(remaining_qty, available_qty)
                    
                    if fill_qty > 0:
                        fill = Fill(
                            order_id=order.id,
                            symbol=order.symbol,
                            side=order.side,
                            quantity=fill_qty,
                            price=price,
                            timestamp=datetime.now()
                        )
                        fills.append(fill)
                        
                        self.bids[price] -= fill_qty
                        if self.bids[price] <= 0:
                            del self.bids[price]
                        
                        remaining_qty -= fill_qty
                
                # Add remaining quantity to book
                if remaining_qty > 0:
                    self.asks[order.price] = self.asks.get(order.price, 0) + remaining_qty
            else:
                # Add to book
                self.asks[order.price] = self.asks.get(order.price, 0) + remaining_qty
        
        return fills
    
    def _update_best_prices(self):
        """Update best bid/ask and mid price"""
        self.best_bid = max(self.bids.keys()) if self.bids else 0.0
        self.best_ask = min(self.asks.keys()) if self.asks else float('inf')
        self.mid_price = (self.best_bid + self.best_ask) / 2 if self.best_ask != float('inf') else self.best_bid
        self.spread = self.best_ask - self.best_bid if self.best_ask != float('inf') else 0.0

class TransactionCostModel:
    """Transaction cost modeling"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.commission_rates = config.get('commission_rates', {})
        self.spread_models = config.get('spread_models', {})
        self.impact_models = config.get('impact_models', {})
    
    def calculate_commission(self, fill: Fill, symbol: str) -> float:
        """Calculate commission for a fill"""
        rate = self.commission_rates.get(symbol, self.commission_rates.get('default', 0.001))
        return fill.quantity * fill.price * rate
    
    def calculate_spread_cost(self, fill: Fill, market_data: Dict[str, Any]) -> float:
        """Calculate spread cost"""
        if 'spread' in market_data:
            spread = market_data['spread']
            return fill.quantity * spread / 2
        return 0.0
    
    def calculate_market_impact(self, fill: Fill, symbol: str, market_data: Dict[str, Any]) -> float:
        """Calculate market impact using Kyle model"""
        if 'volume' not in market_data or 'volatility' not in market_data:
            return 0.0
        
        volume = market_data['volume']
        volatility = market_data['volatility']
        
        # Kyle model: impact = lambda * quantity
        # lambda = sqrt(pi/2) * sigma / (2 * V)
        lambda_param = math.sqrt(math.pi / 2) * volatility / (2 * volume)
        impact = lambda_param * fill.quantity
        
        return impact * fill.price
    
    def calculate_slippage(self, fill: Fill, market_data: Dict[str, Any]) -> float:
        """Calculate slippage based on market conditions"""
        if 'volatility' not in market_data:
            return 0.0
        
        volatility = market_data['volatility']
        
        # Slippage increases with volatility and order size
        base_slippage = volatility * 0.1  # 10% of volatility
        size_impact = min(fill.quantity / 10000, 1.0)  # Cap at 1.0
        
        slippage = base_slippage * (1 + size_impact)
        return slippage * fill.price

class LatencySimulator:
    """Latency simulation for realistic trading"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_latency = config.get('base_latency_ms', 1.0)
        self.jitter = config.get('jitter_ms', 0.5)
        self.network_latency = config.get('network_latency_ms', 5.0)
        self.exchange_latency = config.get('exchange_latency_ms', 2.0)
    
    def simulate_latency(self, order: Order) -> float:
        """Simulate order processing latency"""
        # Base processing latency
        base = random.gauss(self.base_latency, self.jitter)
        
        # Network latency
        network = random.gauss(self.network_latency, self.jitter)
        
        # Exchange processing latency
        exchange = random.gauss(self.exchange_latency, self.jitter)
        
        # Total latency in milliseconds
        total_latency = max(0, base + network + exchange)
        
        return total_latency / 1000.0  # Convert to seconds

class MarketMicrostructureSimulator:
    """Main market microstructure simulator"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.order_books = {}
        self.transaction_cost_model = TransactionCostModel(config.get('transaction_costs', {}))
        self.latency_simulator = LatencySimulator(config.get('latency', {}))
        self.order_counter = 0
        
    def initialize_symbol(self, symbol: str, initial_price: float, tick_size: float = 0.01):
        """Initialize order book for a symbol"""
        self.order_books[symbol] = OrderBook(symbol, tick_size)
        
        # Add some initial liquidity
        book = self.order_books[symbol]
        spread = initial_price * 0.001  # 0.1% spread
        
        # Add some bid orders
        for i in range(5):
            price = initial_price - spread * (i + 1)
            quantity = random.uniform(100, 1000)
            book.bids[price] = quantity
        
        # Add some ask orders
        for i in range(5):
            price = initial_price + spread * (i + 1)
            quantity = random.uniform(100, 1000)
            book.asks[price] = quantity
        
        book._update_best_prices()
    
    def submit_order(self, order: Order, market_data: Dict[str, Any]) -> List[Fill]:
        """Submit order and return fills"""
        if order.symbol not in self.order_books:
            self.initialize_symbol(order.symbol, market_data.get('price', 100.0))
        
        # Simulate latency
        latency = self.latency_simulator.simulate_latency(order)
        
        # Process order
        book = self.order_books[order.symbol]
        fills = book.add_order(order)
        
        # Calculate costs for each fill
        for fill in fills:
            fill.commission = self.transaction_cost_model.calculate_commission(fill, order.symbol)
            fill.slippage = self.transaction_cost_model.calculate_slippage(fill, market_data)
            fill.market_impact = self.transaction_cost_model.calculate_market_impact(fill, order.symbol, market_data)
        
        return fills
    
    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get current market data for a symbol"""
        if symbol not in self.order_books:
            return {}
        
        book = self.order_books[symbol]
        return {
            'symbol': symbol,
            'best_bid': book.best_bid,
            'best_ask': book.best_ask,
            'mid_price': book.mid_price,
            'spread': book.spread,
            'bid_size': sum(book.bids.values()),
            'ask_size': sum(book.asks.values()),
            'timestamp': datetime.now()
        }
    
    def simulate_trading_session(self, orders: List[Order], market_data_history: List[Dict[str, Any]]) -> List[Fill]:
        """Simulate a trading session with multiple orders"""
        all_fills = []
        
        for i, order in enumerate(orders):
            # Get market data for this timestamp
            market_data = market_data_history[i] if i < len(market_data_history) else {}
            
            # Submit order
            fills = self.submit_order(order, market_data)
            all_fills.extend(fills)
        
        return all_fills
    
    def calculate_performance_metrics(self, fills: List[Fill]) -> Dict[str, Any]:
        """Calculate trading performance metrics"""
        if not fills:
            return {}
        
        total_volume = sum(fill.quantity * fill.price for fill in fills)
        total_commission = sum(fill.commission for fill in fills)
        total_slippage = sum(fill.slippage for fill in fills)
        total_impact = sum(fill.market_impact for fill in fills)
        
        # Calculate VWAP
        vwap = total_volume / sum(fill.quantity for fill in fills) if fills else 0
        
        # Calculate average fill price
        avg_fill_price = np.mean([fill.price for fill in fills])
        
        # Calculate cost breakdown
        cost_breakdown = {
            'commission': total_commission,
            'slippage': total_slippage,
            'market_impact': total_impact,
            'total_cost': total_commission + total_slippage + total_impact
        }
        
        return {
            'total_volume': total_volume,
            'total_fills': len(fills),
            'avg_fill_price': avg_fill_price,
            'vwap': vwap,
            'cost_breakdown': cost_breakdown,
            'cost_per_share': cost_breakdown['total_cost'] / sum(fill.quantity for fill in fills) if fills else 0
        }

# Example usage and testing
def create_sample_orders(symbol: str, num_orders: int = 10) -> List[Order]:
    """Create sample orders for testing"""
    orders = []
    
    for i in range(num_orders):
        side = OrderSide.BUY if i % 2 == 0 else OrderSide.SELL
        order_type = OrderType.MARKET if i % 3 == 0 else OrderType.LIMIT
        
        order = Order(
            id=f"order_{i}",
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=random.uniform(100, 1000),
            price=random.uniform(95, 105) if order_type == OrderType.LIMIT else None
        )
        orders.append(order)
    
    return orders

def create_sample_market_data(symbol: str, num_points: int = 10) -> List[Dict[str, Any]]:
    """Create sample market data for testing"""
    market_data = []
    base_price = 100.0
    
    for i in range(num_points):
        # Simulate price movement
        price_change = random.gauss(0, 0.02)  # 2% volatility
        price = base_price * (1 + price_change)
        base_price = price
        
        data = {
            'symbol': symbol,
            'price': price,
            'volume': random.uniform(10000, 100000),
            'volatility': random.uniform(0.1, 0.3),
            'spread': price * random.uniform(0.001, 0.005)
        }
        market_data.append(data)
    
    return market_data
