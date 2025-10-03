"""
Historical Fast-Forward Game for WealthArena

This module implements the historical simulation game where users can compete
against RL agents using historical market episodes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import random
from enum import Enum

try:
    from ..data.benchmarks.benchmark_data import BenchmarkDataFetcher
    from ..data.signal_fusion import SignalFusion, SignalFusionConfig
    from ..training.model_checkpoint import ProductionModelManager
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from data.benchmarks.benchmark_data import BenchmarkDataFetcher
    from data.signal_fusion import SignalFusion, SignalFusionConfig
    from training.model_checkpoint import ProductionModelManager

logger = logging.getLogger(__name__)


class GameStatus(Enum):
    """Game status enumeration"""
    WAITING = "waiting"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class PlayerType(Enum):
    """Player type enumeration"""
    HUMAN = "human"
    AGENT = "agent"
    BENCHMARK = "benchmark"


@dataclass
class GameConfig:
    """Configuration for historical game"""
    # Episode settings
    episode_duration_days: int = 90  # 3 months
    lookback_days: int = 30  # 1 month of historical data
    
    # Trading settings
    initial_capital: float = 100000.0
    max_positions: int = 10
    position_size_limit: float = 0.2  # 20% per position
    
    # Game settings
    max_players: int = 10
    min_players: int = 2
    time_limit_per_turn: int = 300  # 5 minutes
    
    # Available instruments
    available_instruments: List[str] = None
    
    def __post_init__(self):
        if self.available_instruments is None:
            self.available_instruments = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA",
                "SPY", "QQQ", "IWM", "VTI", "GLD", "SLV",
                "BTC-USD", "ETH-USD", "ADA-USD",
                "EURUSD=X", "GBPUSD=X", "USDJPY=X",
                "GC=F", "CL=F", "NG=F"
            ]


@dataclass
class Player:
    """Player in the game"""
    player_id: str
    player_type: PlayerType
    name: str
    portfolio_value: float
    positions: Dict[str, float]
    cash: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    trades_count: int
    is_active: bool = True


@dataclass
class GameEpisode:
    """Historical game episode"""
    episode_id: str
    start_date: datetime
    end_date: datetime
    market_data: pd.DataFrame
    instruments: List[str]
    benchmark_data: pd.DataFrame
    created_at: datetime


class HistoricalGame:
    """Historical fast-forward game implementation"""
    
    def __init__(self, config: GameConfig = None):
        self.config = config or GameConfig()
        
        # Initialize components with error handling
        try:
            self.benchmark_fetcher = BenchmarkDataFetcher()
        except Exception as e:
            logger.warning(f"Failed to initialize benchmark fetcher: {e}")
            self.benchmark_fetcher = None
        
        try:
            self.signal_fusion = SignalFusion(SignalFusionConfig())
        except Exception as e:
            logger.warning(f"Failed to initialize signal fusion: {e}")
            self.signal_fusion = None
        
        try:
            self.model_manager = ProductionModelManager()
        except Exception as e:
            logger.warning(f"Failed to initialize model manager: {e}")
            self.model_manager = None
        
        # Game state
        self.game_id = None
        self.status = GameStatus.WAITING
        self.players = {}
        self.episode = None
        self.current_date = None
        self.turn_count = 0
        self.leaderboard = []
        
        # Game history
        self.trade_history = []
        self.portfolio_history = []
        self.performance_history = []
        
        logger.info("Historical game initialized")
    
    def create_episode(self, 
                      start_date: str,
                      end_date: str,
                      instruments: List[str] = None) -> GameEpisode:
        """Create a new historical episode"""
        
        try:
            episode_id = f"episode_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Generate market data for the episode
            market_data = self._generate_episode_data(start_date, end_date, instruments)
            
            # Get benchmark data
            benchmark_data = self._get_benchmark_data(start_date, end_date)
            
            episode = GameEpisode(
                episode_id=episode_id,
                start_date=datetime.strptime(start_date, "%Y-%m-%d"),
                end_date=datetime.strptime(end_date, "%Y-%m-%d"),
                market_data=market_data,
                instruments=instruments or self.config.available_instruments[:10],
                benchmark_data=benchmark_data,
                created_at=datetime.now()
            )
            
            logger.info(f"Created episode {episode_id} with {len(market_data)} days of data")
            return episode
            
        except Exception as e:
            logger.error(f"Error creating episode: {e}")
            raise
    
    def _generate_episode_data(self, start_date: str, end_date: str, instruments: List[str]) -> pd.DataFrame:
        """Generate historical market data for episode"""
        
        # Create date range
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate synthetic market data (in production, use real historical data)
        market_data = {}
        
        for instrument in instruments:
            # Generate realistic price movements
            returns = np.random.normal(0.0005, 0.02, len(dates))  # 0.05% daily return, 2% volatility
            prices = 100 * np.exp(np.cumsum(returns))
            
            # Generate OHLCV data
            open_prices = prices * (1 + np.random.normal(0, 0.001, len(dates)))
            close_prices = prices
            high_prices = np.maximum(open_prices, close_prices) * (1 + np.abs(np.random.normal(0, 0.01, len(dates))))
            low_prices = np.minimum(open_prices, close_prices) * (1 - np.abs(np.random.normal(0, 0.01, len(dates))))
            volumes = np.random.lognormal(10, 0.5, len(dates))
            
            market_data[instrument] = pd.DataFrame({
                'Open': open_prices,
                'High': high_prices,
                'Low': low_prices,
                'Close': close_prices,
                'Volume': volumes
            }, index=dates)
        
        # Combine all instruments
        multi_level_data = {}
        for instrument, df in market_data.items():
            for col in df.columns:
                multi_level_data[(instrument, col)] = df[col]
        
        return pd.DataFrame(multi_level_data)
    
    def _get_benchmark_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get benchmark data for the episode period"""
        
        try:
            if self.benchmark_fetcher is None:
                logger.warning("Benchmark fetcher not available, using synthetic data")
                # Return synthetic benchmark
                dates = pd.date_range(start=start_date, end=end_date, freq='D')
                returns = np.random.normal(0.0005, 0.015, len(dates))
                return pd.DataFrame({'benchmark_return': returns}, index=dates)
            
            # Get S&P 500 as benchmark
            benchmark_returns = self.benchmark_fetcher.get_etf_benchmark()
            
            # Filter to episode period
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            
            episode_benchmark = benchmark_returns[
                (benchmark_returns.index >= start_dt) & 
                (benchmark_returns.index <= end_dt)
            ]
            
            return episode_benchmark.to_frame(name='benchmark_return')
            
        except Exception as e:
            logger.error(f"Error getting benchmark data: {e}")
            # Return synthetic benchmark
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            returns = np.random.normal(0.0005, 0.015, len(dates))
            return pd.DataFrame({'benchmark_return': returns}, index=dates)
    
    def start_game(self, episode: GameEpisode, players: List[Dict[str, Any]]) -> str:
        """Start a new game with the given episode and players"""
        
        try:
            self.game_id = f"game_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.episode = episode
            self.current_date = episode.start_date
            self.status = GameStatus.ACTIVE
            self.turn_count = 0
            
            # Initialize players
            self.players = {}
            for i, player_data in enumerate(players):
                player_id = player_data.get('player_id', f"player_{i}")
                player_type = PlayerType(player_data.get('player_type', 'human'))
                
                player = Player(
                    player_id=player_id,
                    player_type=player_type,
                    name=player_data.get('name', f"Player {i+1}"),
                    portfolio_value=self.config.initial_capital,
                    positions={},
                    cash=self.config.initial_capital,
                    total_return=0.0,
                    sharpe_ratio=0.0,
                    max_drawdown=0.0,
                    trades_count=0
                )
                
                self.players[player_id] = player
            
            # Initialize game history
            self.trade_history = []
            self.portfolio_history = []
            self.performance_history = []
            
            logger.info(f"Started game {self.game_id} with {len(self.players)} players")
            return self.game_id
            
        except Exception as e:
            logger.error(f"Error starting game: {e}")
            raise
    
    def execute_turn(self, player_id: str, actions: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a turn for a player"""
        
        if self.status != GameStatus.ACTIVE:
            raise ValueError("Game is not active")
        
        if player_id not in self.players:
            raise ValueError(f"Player {player_id} not found")
        
        player = self.players[player_id]
        if not player.is_active:
            raise ValueError(f"Player {player_id} is not active")
        
        try:
            # Get current market data
            current_data = self._get_current_market_data()
            
            # Execute trades
            trade_results = self._execute_trades(player, actions, current_data)
            
            # Update portfolio
            self._update_portfolio(player, trade_results)
            
            # Record trade history
            self.trade_history.append({
                'game_id': self.game_id,
                'player_id': player_id,
                'turn': self.turn_count,
                'date': self.current_date,
                'actions': actions,
                'trade_results': trade_results,
                'portfolio_value': player.portfolio_value,
                'cash': player.cash
            })
            
            # Advance to next day
            self.current_date += timedelta(days=1)
            self.turn_count += 1
            
            # Check if episode is complete
            if self.current_date >= self.episode.end_date:
                self._complete_game()
            
            return {
                'success': True,
                'trade_results': trade_results,
                'portfolio_value': player.portfolio_value,
                'cash': player.cash,
                'next_date': self.current_date,
                'game_complete': self.status == GameStatus.COMPLETED
            }
            
        except Exception as e:
            logger.error(f"Error executing turn for player {player_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'portfolio_value': player.portfolio_value,
                'cash': player.cash
            }
    
    def _get_current_market_data(self) -> Dict[str, float]:
        """Get current market prices"""
        
        if self.current_date not in self.episode.market_data.index:
            # Use last available data
            current_data = self.episode.market_data.iloc[-1]
        else:
            current_data = self.episode.market_data.loc[self.current_date]
        
        prices = {}
        for instrument in self.episode.instruments:
            if (instrument, 'Close') in current_data.index:
                prices[instrument] = current_data[(instrument, 'Close')]
        
        return prices
    
    def _execute_trades(self, player: Player, actions: Dict[str, Any], current_prices: Dict[str, float]) -> Dict[str, Any]:
        """Execute trades for a player"""
        
        trade_results = {
            'trades_executed': [],
            'total_cost': 0.0,
            'success': True
        }
        
        try:
            # Process buy orders
            if 'buy' in actions:
                for instrument, quantity in actions['buy'].items():
                    if instrument in current_prices and quantity > 0:
                        price = current_prices[instrument]
                        cost = price * quantity
                        
                        if cost <= player.cash:
                            # Execute buy
                            if instrument in player.positions:
                                player.positions[instrument] += quantity
                            else:
                                player.positions[instrument] = quantity
                            
                            player.cash -= cost
                            player.trades_count += 1
                            
                            trade_results['trades_executed'].append({
                                'type': 'buy',
                                'instrument': instrument,
                                'quantity': quantity,
                                'price': price,
                                'cost': cost
                            })
                            
                            trade_results['total_cost'] += cost
            
            # Process sell orders
            if 'sell' in actions:
                for instrument, quantity in actions['sell'].items():
                    if instrument in current_prices and quantity > 0:
                        if instrument in player.positions and player.positions[instrument] >= quantity:
                            price = current_prices[instrument]
                            proceeds = price * quantity
                            
                            # Execute sell
                            player.positions[instrument] -= quantity
                            if player.positions[instrument] <= 0:
                                del player.positions[instrument]
                            
                            player.cash += proceeds
                            player.trades_count += 1
                            
                            trade_results['trades_executed'].append({
                                'type': 'sell',
                                'instrument': instrument,
                                'quantity': quantity,
                                'price': price,
                                'proceeds': proceeds
                            })
            
            return trade_results
            
        except Exception as e:
            logger.error(f"Error executing trades: {e}")
            trade_results['success'] = False
            trade_results['error'] = str(e)
            return trade_results
    
    def _update_portfolio(self, player: Player, trade_results: Dict[str, Any]):
        """Update player portfolio value"""
        
        # Calculate current portfolio value
        current_prices = self._get_current_market_data()
        portfolio_value = player.cash
        
        for instrument, quantity in player.positions.items():
            if instrument in current_prices:
                portfolio_value += quantity * current_prices[instrument]
        
        # Update player
        old_value = player.portfolio_value
        player.portfolio_value = portfolio_value
        player.total_return = (portfolio_value - self.config.initial_capital) / self.config.initial_capital
        
        # Update max drawdown
        if portfolio_value < old_value:
            drawdown = (old_value - portfolio_value) / old_value
            player.max_drawdown = max(player.max_drawdown, drawdown)
        
        # Record portfolio history
        self.portfolio_history.append({
            'game_id': self.game_id,
            'player_id': player.player_id,
            'date': self.current_date,
            'portfolio_value': portfolio_value,
            'cash': player.cash,
            'positions': player.positions.copy(),
            'total_return': player.total_return
        })
    
    def _complete_game(self):
        """Complete the game and calculate final results"""
        
        self.status = GameStatus.COMPLETED
        
        # Calculate final performance metrics
        for player in self.players.values():
            if player.portfolio_value > 0:
                # Calculate Sharpe ratio (simplified)
                returns = [h['total_return'] for h in self.portfolio_history if h['player_id'] == player.player_id]
                if len(returns) > 1:
                    player.sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        
        # Create leaderboard
        self.leaderboard = sorted(
            self.players.values(),
            key=lambda p: p.total_return,
            reverse=True
        )
        
        logger.info(f"Game {self.game_id} completed. Winner: {self.leaderboard[0].name}")
    
    def get_game_state(self) -> Dict[str, Any]:
        """Get current game state"""
        
        return {
            'game_id': self.game_id,
            'status': self.status.value,
            'current_date': self.current_date.isoformat() if self.current_date else None,
            'turn_count': self.turn_count,
            'players': {
                pid: {
                    'name': p.name,
                    'player_type': p.player_type.value,
                    'portfolio_value': p.portfolio_value,
                    'total_return': p.total_return,
                    'sharpe_ratio': p.sharpe_ratio,
                    'max_drawdown': p.max_drawdown,
                    'trades_count': p.trades_count,
                    'is_active': p.is_active
                }
                for pid, p in self.players.items()
            },
            'leaderboard': [
                {
                    'rank': i + 1,
                    'player_id': p.player_id,
                    'name': p.name,
                    'total_return': p.total_return,
                    'sharpe_ratio': p.sharpe_ratio
                }
                for i, p in enumerate(self.leaderboard)
            ],
            'episode': {
                'episode_id': self.episode.episode_id if self.episode else None,
                'start_date': self.episode.start_date.isoformat() if self.episode else None,
                'end_date': self.episode.end_date.isoformat() if self.episode else None,
                'instruments': self.episode.instruments if self.episode else []
            }
        }
    
    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """Get current leaderboard"""
        
        return [
            {
                'rank': i + 1,
                'player_id': p.player_id,
                'name': p.name,
                'player_type': p.player_type.value,
                'portfolio_value': p.portfolio_value,
                'total_return': p.total_return,
                'sharpe_ratio': p.sharpe_ratio,
                'max_drawdown': p.max_drawdown,
                'trades_count': p.trades_count
            }
            for i, p in enumerate(self.leaderboard)
        ]
    
    def pause_game(self):
        """Pause the game"""
        if self.status == GameStatus.ACTIVE:
            self.status = GameStatus.PAUSED
            logger.info(f"Game {self.game_id} paused")
    
    def resume_game(self):
        """Resume the game"""
        if self.status == GameStatus.PAUSED:
            self.status = GameStatus.ACTIVE
            logger.info(f"Game {self.game_id} resumed")
    
    def end_game(self):
        """End the game"""
        self.status = GameStatus.CANCELLED
        logger.info(f"Game {self.game_id} ended")


def create_historical_game(config: GameConfig = None) -> HistoricalGame:
    """Create a historical game instance"""
    return HistoricalGame(config)


def main():
    """Test the historical game system"""
    logging.basicConfig(level=logging.INFO)
    
    print("🎮 Testing Historical Fast-Forward Game...")
    print("="*50)
    
    # Create game
    config = GameConfig(
        episode_duration_days=30,
        initial_capital=100000.0,
        available_instruments=["AAPL", "MSFT", "GOOGL", "SPY", "QQQ"]
    )
    game = HistoricalGame(config)
    
    # Create episode
    print("\n📅 Creating historical episode...")
    episode = game.create_episode(
        start_date="2024-01-01",
        end_date="2024-01-31",
        instruments=["AAPL", "MSFT", "GOOGL", "SPY", "QQQ"]
    )
    print(f"✅ Episode created: {episode.episode_id}")
    print(f"   Period: {episode.start_date.date()} to {episode.end_date.date()}")
    print(f"   Instruments: {episode.instruments}")
    
    # Create players
    players = [
        {'player_id': 'human_1', 'player_type': 'human', 'name': 'Human Player'},
        {'player_id': 'agent_1', 'player_type': 'agent', 'name': 'RL Agent'},
        {'player_id': 'benchmark_1', 'player_type': 'benchmark', 'name': 'S&P 500 Benchmark'}
    ]
    
    # Start game
    print(f"\n🚀 Starting game with {len(players)} players...")
    game_id = game.start_game(episode, players)
    print(f"✅ Game started: {game_id}")
    
    # Simulate a few turns
    print(f"\n🎯 Simulating game turns...")
    for turn in range(5):
        print(f"\n--- Turn {turn + 1} ---")
        
        # Human player action
        human_actions = {
            'buy': {'AAPL': 10, 'MSFT': 5},
            'sell': {}
        }
        result = game.execute_turn('human_1', human_actions)
        print(f"Human player: Portfolio value = ${result['portfolio_value']:,.2f}")
        
        # Agent player action (simplified)
        agent_actions = {
            'buy': {'GOOGL': 8, 'SPY': 20},
            'sell': {}
        }
        result = game.execute_turn('agent_1', agent_actions)
        print(f"RL Agent: Portfolio value = ${result['portfolio_value']:,.2f}")
        
        # Benchmark player (buy and hold)
        if turn == 0:  # Only buy once
            benchmark_actions = {
                'buy': {'SPY': 100},
                'sell': {}
            }
            result = game.execute_turn('benchmark_1', benchmark_actions)
            print(f"Benchmark: Portfolio value = ${result['portfolio_value']:,.2f}")
        else:
            # No action for benchmark (buy and hold)
            result = game.execute_turn('benchmark_1', {'buy': {}, 'sell': {}})
            print(f"Benchmark: Portfolio value = ${result['portfolio_value']:,.2f}")
    
    # Get final results
    print(f"\n🏆 Final Results:")
    leaderboard = game.get_leaderboard()
    for player in leaderboard:
        print(f"  {player['rank']}. {player['name']}: {player['total_return']:.2%} return")
    
    print(f"\n✅ Historical game test completed!")


if __name__ == "__main__":
    main()
