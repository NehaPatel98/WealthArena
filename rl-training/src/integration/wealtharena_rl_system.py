"""
WealthArena RL System Integration

This module integrates all components of the WealthArena RL system:
- Data collection and preprocessing
- Signal engineering
- RL agent training and deployment
- Risk management
- Backtesting
- LLM integration
- Performance monitoring
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import asyncio
import json
import yaml
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.advanced_signal_engineering import AdvancedSignalEngine, SignalConfig, SignalType
from simulation.market_microstructure import MarketMicrostructureSimulator, Order, OrderSide, OrderType
from agents.hierarchical_rl_agents import HierarchicalRLSystem, AgentConfig, AgentType
from agents.offline_rl_pretraining import OfflineRLTrainer, OfflineConfig, OfflineAlgorithm
from optimization.multi_objective_optimization import RewardShaper, MultiObjectiveConfig, ObjectiveConfig, ObjectiveType
from backtesting.advanced_backtesting import VectorizedBacktester, WalkForwardBacktester, MonteCarloBacktester, BacktestConfig, BacktestType
from risk.covariance_aware_risk import CovarianceAwareRiskManager, RiskConfig, CovarianceMethod, OptimizationMethod
from .chatbot_integration import get_chatbot_integration

logger = logging.getLogger(__name__)

@dataclass
class SystemConfig:
    """Main system configuration"""
    # Data collection
    data_sources: List[str] = None
    asset_types: List[str] = None
    exchanges: List[str] = None
    start_date: datetime = None
    end_date: datetime = None
    
    # RL agents
    agent_configs: Dict[str, Any] = None
    offline_pretraining: bool = True
    online_training: bool = True
    
    # Risk management
    risk_config: Dict[str, Any] = None
    
    # Backtesting
    backtest_config: Dict[str, Any] = None
    
    # LLM integration
    llm_config: Dict[str, Any] = None
    
    # System settings
    log_level: str = "INFO"
    data_dir: str = "data"
    models_dir: str = "models"
    results_dir: str = "results"

class WealthArenaRLSystem:
    """Main WealthArena RL system"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.setup_logging()
        self.setup_directories()
        
        # Initialize components
        self.data_collector = None
        self.signal_engine = None
        self.market_simulator = None
        self.rl_system = None
        self.risk_manager = None
        self.backtester = None
        self.llm_manager = None
        
        # System state
        self.market_data = {}
        self.signals = {}
        self.portfolio_state = {}
        self.performance_metrics = {}
        
        logger.info("WealthArena RL System initialized")
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('wealtharena_rl.log'),
                logging.StreamHandler()
            ]
        )
    
    def setup_directories(self):
        """Setup required directories"""
        directories = [self.config.data_dir, self.config.models_dir, self.config.results_dir]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def initialize_components(self):
        """Initialize all system components"""
        logger.info("Initializing system components...")
        
        # Initialize signal engine
        signal_config = {
            'lookback_period': 252,
            'update_frequency': 'daily'
        }
        self.signal_engine = AdvancedSignalEngine(signal_config)
        
        # Initialize market simulator
        market_config = {
            'transaction_costs': {'default': 0.001},
            'latency': {'base_latency_ms': 1.0, 'jitter_ms': 0.5}
        }
        self.market_simulator = MarketMicrostructureSimulator(market_config)
        
        # Initialize RL system
        rl_config = {
            'allocator_state_dim': 50,
            'executor_state_dim': 20,
            'executor_action_dim': 5,
            'num_assets': 10,
            'assets': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NVDA', 'NFLX', 'AMD', 'INTC']
        }
        self.rl_system = HierarchicalRLSystem(rl_config)
        
        # Initialize risk manager
        risk_config = RiskConfig(
            covariance_method=CovarianceMethod.LEDOIT_WOLF,
            optimization_method=OptimizationMethod.RISK_PARITY,
            rebalance_frequency='monthly',
            min_weight=0.0,
            max_weight=0.1
        )
        self.risk_manager = CovarianceAwareRiskManager(risk_config)
        
        # Initialize backtester
        backtest_config = BacktestConfig(
            backtest_type=BacktestType.VECTORIZED,
            start_date=self.config.start_date or datetime.now() - timedelta(days=365),
            end_date=self.config.end_date or datetime.now(),
            initial_capital=1000000.0
        )
        self.backtester = VectorizedBacktester(backtest_config)
        
        # Initialize LLM manager
        # Initialize chatbot integration (connects to existing wealtharena_chatbot service)
        self.chatbot_integration = get_chatbot_integration()
        
        logger.info("All components initialized successfully")
    
    async def collect_market_data(self) -> Dict[str, pd.DataFrame]:
        """Collect market data from various sources"""
        logger.info("Collecting market data...")
        
        # This would integrate with the data collection system
        # For now, we'll use mock data
        market_data = {}
        
        # Generate mock data for demonstration
        dates = pd.date_range(start=self.config.start_date, end=self.config.end_date, freq='D')
        assets = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        
        for asset in assets:
            # Generate realistic price data
            np.random.seed(hash(asset) % 2**32)
            returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
            prices = 100 * np.cumprod(1 + returns)
            
            market_data[asset] = pd.DataFrame({
                'Date': dates,
                'Open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
                'High': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
                'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
                'Close': prices,
                'Volume': np.random.randint(1000000, 10000000, len(dates))
            })
        
        self.market_data = market_data
        logger.info(f"Collected market data for {len(assets)} assets")
        return market_data
    
    def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Generate trading signals using advanced signal engineering"""
        logger.info("Generating trading signals...")
        
        signals = {}
        
        for asset, data in market_data.items():
            # Generate signals for each asset
            asset_signals = self.signal_engine.generate_all_signals(data, asset)
            signals[asset] = asset_signals
        
        self.signals = signals
        logger.info(f"Generated signals for {len(signals)} assets")
        return signals
    
    def train_rl_agents(self, market_data: Dict[str, pd.DataFrame], signals: Dict[str, pd.DataFrame]):
        """Train RL agents using historical data"""
        logger.info("Training RL agents...")
        
        if self.config.offline_pretraining:
            # Offline pretraining
            self._offline_pretraining(market_data, signals)
        
        if self.config.online_training:
            # Online training
            self._online_training(market_data, signals)
        
        logger.info("RL agent training completed")
    
    def _offline_pretraining(self, market_data: Dict[str, pd.DataFrame], signals: Dict[str, pd.DataFrame]):
        """Perform offline pretraining"""
        logger.info("Starting offline pretraining...")
        
        # Prepare offline dataset
        offline_data = self._prepare_offline_dataset(market_data, signals)
        
        # Train offline RL agents
        for algorithm in [OfflineAlgorithm.CQL, OfflineAlgorithm.BCQ, OfflineAlgorithm.BC]:
            config = OfflineConfig(
                algorithm=algorithm,
                dataset_path=f"{self.config.data_dir}/offline_dataset.pkl",
                state_dim=50,
                action_dim=5,
                hidden_dims=[256, 128, 64],
                num_epochs=100
            )
            
            trainer = OfflineRLTrainer(config)
            training_history = trainer.train()
            
            # Save trained agent
            trainer.save_agent(f"{self.config.models_dir}/offline_{algorithm.value}_agent.pkl")
            
            logger.info(f"Offline pretraining completed for {algorithm.value}")
    
    def _online_training(self, market_data: Dict[str, pd.DataFrame], signals: Dict[str, pd.DataFrame]):
        """Perform online training"""
        logger.info("Starting online training...")
        
        # This would implement online RL training
        # For now, we'll use the hierarchical RL system
        for date in pd.date_range(start=self.config.start_date, end=self.config.end_date, freq='D'):
            # Prepare market state for this date
            market_state = self._prepare_market_state(market_data, signals, date)
            
            # Get RL system decision
            decision = self.rl_system.step(market_state)
            
            # Update portfolio state
            self.portfolio_state = decision['portfolio_state']
        
        logger.info("Online training completed")
    
    def _prepare_offline_dataset(self, market_data: Dict[str, pd.DataFrame], signals: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """Prepare offline dataset for RL training"""
        # Combine market data and signals
        all_data = []
        all_actions = []
        all_rewards = []
        
        for asset, data in market_data.items():
            if asset in signals:
                # Use signals as actions (simplified)
                signal_data = signals[asset]
                
                # Extract features
                features = signal_data.select_dtypes(include=[np.number]).values
                actions = np.random.randn(len(features), 5)  # Mock actions
                rewards = np.random.randn(len(features))  # Mock rewards
                
                all_data.append(features)
                all_actions.append(actions)
                all_rewards.append(rewards)
        
        # Combine all data
        states = np.vstack(all_data)
        actions = np.vstack(all_actions)
        rewards = np.concatenate(all_rewards)
        
        # Create next states
        next_states = np.roll(states, -1, axis=0)
        dones = np.zeros(len(states), dtype=bool)
        dones[-1] = True
        
        return {
            'states': states[:-1],
            'actions': actions[:-1],
            'rewards': rewards[:-1],
            'next_states': next_states[:-1],
            'dones': dones[:-1]
        }
    
    def _prepare_market_state(self, market_data: Dict[str, pd.DataFrame], signals: Dict[str, pd.DataFrame], date: datetime) -> Dict[str, Any]:
        """Prepare market state for RL system"""
        market_state = {
            'prices': [],
            'returns': [],
            'volatilities': [],
            'signals': {}
        }
        
        for asset, data in market_data.items():
            if date in data.index:
                market_state['prices'].append(data.loc[date, 'Close'])
                market_state['returns'].append(data.loc[date, 'Close'] / data.loc[date, 'Open'] - 1)
                market_state['volatilities'].append(0.02)  # Mock volatility
                
                if asset in signals and date in signals[asset].index:
                    market_state['signals'][asset] = signals[asset].loc[date].to_dict()
        
        return market_state
    
    def run_backtesting(self, market_data: Dict[str, pd.DataFrame], signals: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Run comprehensive backtesting"""
        logger.info("Running backtesting...")
        
        # Prepare data for backtesting
        combined_data = self._prepare_backtest_data(market_data, signals)
        
        # Run vectorized backtest
        result = self.backtester.run_backtest(combined_data, self._strategy_function)
        
        # Run walk-forward backtest
        walk_forward_backtester = WalkForwardBacktester(self.backtester.config)
        walk_forward_results = walk_forward_backtester.run_walk_forward(combined_data, self._strategy_function)
        
        # Run Monte Carlo backtest
        monte_carlo_backtester = MonteCarloBacktester(self.backtester.config)
        monte_carlo_results = monte_carlo_backtester.run_monte_carlo(combined_data, self._strategy_function)
        
        backtest_results = {
            'vectorized': result,
            'walk_forward': walk_forward_results,
            'monte_carlo': monte_carlo_results
        }
        
        logger.info("Backtesting completed")
        return backtest_results
    
    def _prepare_backtest_data(self, market_data: Dict[str, pd.DataFrame], signals: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Prepare data for backtesting"""
        # Combine all market data into a single DataFrame
        combined_data = pd.DataFrame()
        
        for asset, data in market_data.items():
            if asset in signals:
                # Use signals as portfolio weights
                signal_data = signals[asset]
                combined_data[asset] = signal_data['Close'] if 'Close' in signal_data.columns else signal_data.iloc[:, 0]
        
        return combined_data
    
    def _strategy_function(self, data: pd.DataFrame) -> pd.DataFrame:
        """Strategy function for backtesting"""
        # Simple equal-weight strategy
        n_assets = len(data.columns)
        weights = pd.DataFrame(index=data.index, columns=data.columns)
        weights[:] = 1.0 / n_assets
        return weights
    
    def manage_risk(self, portfolio_weights: pd.Series, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Manage portfolio risk using covariance-aware methods"""
        logger.info("Managing portfolio risk...")
        
        # Prepare returns data
        returns_data = pd.DataFrame()
        for asset, data in market_data.items():
            returns_data[asset] = data['Close'].pct_change().fillna(0)
        
        # Risk management
        risk_results = self.risk_manager.manage_portfolio_risk(
            returns_data, portfolio_weights
        )
        
        logger.info("Risk management completed")
        return risk_results
    
    async def integrate_llm_insights(self, market_data: Dict[str, pd.DataFrame], news_data: List[str]) -> Dict[str, Any]:
        """Integrate LLM insights using existing chatbot service"""
        logger.info("Integrating LLM insights via chatbot service...")
        
        try:
            # Convert news data to proper format
            news_articles = []
            for i, news_item in enumerate(news_data):
                news_articles.append({
                    "title": f"Market News {i+1}",
                    "summary": news_item,
                    "url": f"https://example.com/news/{i+1}"
                })
            
            # Generate news insights using chatbot service
            news_insights = await self.chatbot_integration.generate_news_insights(news_articles)
            
            # Generate trade rationales for each symbol
            trade_rationales = {}
            for symbol, data in market_data.items():
                if not data.empty:
                    # Get latest indicators
                    latest_data = data.iloc[-1]
                    indicators = {
                        "price": latest_data.get("Close", 0),
                        "volume": latest_data.get("Volume", 0),
                        "rsi": latest_data.get("RSI", 50),
                        "sma_20": latest_data.get("SMA_20", latest_data.get("Close", 0))
                    }
                    
                    # Get explainable rationale
                    rationale = await self.chatbot_integration.explain_trade_rationale(symbol, indicators)
                    trade_rationales[symbol] = rationale
            
            llm_results = {
                "news_insights": news_insights,
                "trade_rationales": trade_rationales,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info("LLM integration completed via chatbot service")
            return llm_results
            
        except Exception as e:
            logger.error(f"LLM integration failed: {e}")
            return {
                "news_insights": {"error": "Unable to analyze news"},
                "trade_rationales": {},
                "timestamp": datetime.now().isoformat()
            }
    
    async def run_full_system(self) -> Dict[str, Any]:
        """Run the complete WealthArena RL system"""
        logger.info("Starting full system run...")
        
        try:
            # Initialize components
            self.initialize_components()
            
            # Collect market data
            market_data = await self.collect_market_data()
            
            # Generate signals
            signals = self.generate_signals(market_data)
            
            # Train RL agents
            self.train_rl_agents(market_data, signals)
            
            # Run backtesting
            backtest_results = self.run_backtesting(market_data, signals)
            
            # Manage risk
            portfolio_weights = pd.Series(0.1, index=list(market_data.keys()))
            risk_results = self.manage_risk(portfolio_weights, market_data)
            
            # Integrate LLM insights
            news_data = ["Market shows strong performance", "Tech stocks rally continues"]
            llm_results = await self.integrate_llm_insights(market_data, news_data)
            
            # Compile results
            system_results = {
                'market_data': market_data,
                'signals': signals,
                'backtest_results': backtest_results,
                'risk_results': risk_results,
                'llm_results': llm_results,
                'portfolio_state': self.portfolio_state,
                'performance_metrics': self.performance_metrics
            }
            
            logger.info("Full system run completed successfully")
            return system_results
            
        except Exception as e:
            logger.error(f"System run failed: {e}")
            raise
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save system results"""
        if filename is None:
            filename = f"wealtharena_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = Path(self.config.results_dir) / filename
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_serializable(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filepath}")
    
    def _make_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-serializable format"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj

def load_config(config_path: str) -> SystemConfig:
    """Load system configuration from file"""
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    return SystemConfig(**config_data)

async def main():
    """Main entry point"""
    # Load configuration
    config = load_config('config/wealtharena_config.yaml')
    
    # Create system
    system = WealthArenaRLSystem(config)
    
    # Run full system
    results = await system.run_full_system()
    
    # Save results
    system.save_results(results)
    
    print("WealthArena RL System completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
