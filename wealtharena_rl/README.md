# WealthArena RL System

A comprehensive reinforcement learning platform for multi-asset portfolio management, featuring advanced signal engineering, hierarchical RL agents, risk management, and LLM integration.

## 🚀 Features

### Core Capabilities
- **Multi-Asset Data Collection**: Dynamic data fetching from exchanges (ASX, NYSE, NASDAQ) and APIs (Yahoo Finance, Alpha Vantage, FRED, CoinGecko)
- **Advanced Signal Engineering**: 100+ technical indicators, volatility estimators, and fundamental signals
- **Hierarchical RL Agents**: High-level allocator + low-level execution policies with PPO and SAC algorithms
- **Offline RL Pretraining**: Conservative Q-Learning (CQL), Batch-Constrained Q-Learning (BCQ), and Behavior Cloning
- **Multi-Objective Optimization**: Reward shaping balancing return, risk, liquidity, and ESG goals
- **Covariance-Aware Risk Management**: Dynamic hedging, portfolio optimization, and stress testing
- **Advanced Backtesting**: Vectorized, walk-forward, and Monte Carlo backtesting engines
- **LLM Integration**: Event extraction, sentiment analysis, and explainable trade rationales

### Technical Highlights
- **Market Microstructure Simulation**: Order book simulation, transaction costs, and latency modeling
- **Real-time Processing**: Asynchronous data processing and signal generation
- **Production-Ready**: Docker containerization, Kubernetes deployment, and Azure integration
- **Comprehensive Monitoring**: Performance metrics, risk attribution, and system health monitoring

## 📁 Project Structure

```
wealtharena_rl/
├── src/
│   ├── data/
│   │   └── advanced_signal_engineering.py    # Signal generation and processing
│   ├── simulation/
│   │   └── market_microstructure.py          # Market simulation and order book
│   ├── agents/
│   │   ├── hierarchical_rl_agents.py         # Hierarchical RL system
│   │   └── offline_rl_pretraining.py         # Offline RL algorithms
│   ├── optimization/
│   │   └── multi_objective_optimization.py   # Multi-objective reward shaping
│   ├── backtesting/
│   │   └── advanced_backtesting.py           # Backtesting engines
│   ├── risk/
│   │   └── covariance_aware_risk.py          # Risk management and hedging
│   ├── llm/
│   │   └── llm_integration.py                # LLM integration and NLP
│   └── integration/
│       └── wealtharena_rl_system.py          # Main system integration
├── config/
│   └── wealtharena_config.yaml              # System configuration
├── data_collection_for_training.py          # Enhanced data collection
├── requirements.txt                         # Python dependencies
└── README.md                               # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, for accelerated training)
- 8GB+ RAM recommended

### Quick Start

1. **Clone the repository**
```bash
git clone <repository-url>
cd wealtharena_rl
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure the system**
```bash
cp config/wealtharena_config.yaml.example config/wealtharena_config.yaml
# Edit the configuration file with your API keys and preferences
```

4. **Run the system**
```bash
python src/integration/wealtharena_rl_system.py
```

### Docker Deployment

1. **Build the container**
```bash
docker build -t wealtharena-rl .
```

2. **Run the container**
```bash
docker run -p 8000:8000 wealtharena-rl
```

## 🔧 Configuration

The system is configured via `config/wealtharena_config.yaml`. Key configuration options:

### Data Sources
```yaml
data_sources:
  - yahoo_finance
  - alpha_vantage
  - fred
  - coingecko

asset_types:
  - stocks
  - etfs
  - crypto
  - forex
  - commodities

exchanges:
  - ASX
  - NYSE
  - NASDAQ
```

### RL Agents
```yaml
agent_configs:
  allocator:
    state_dim: 50
    action_dim: 10
    hidden_dims: [256, 128, 64]
    learning_rate: 0.0001
```

### Risk Management
```yaml
risk_config:
  covariance_method: "ledoit_wolf"
  optimization_method: "risk_parity"
  rebalance_frequency: "monthly"
  min_weight: 0.0
  max_weight: 0.1
```

## 📊 Usage Examples

### Data Collection
```python
from data_collection_for_training import MultiAssetDataCollector

# Collect data for all asset types from all exchanges
collector = MultiAssetDataCollector({
    "asset_types": ["stocks", "crypto", "forex"],
    "exchanges": ["ASX", "NYSE", "NASDAQ"],
    "start_date": "2020-01-01",
    "end_date": "2024-12-31"
})

# Run data collection
collector.collect_all_data()
```

### Signal Engineering
```python
from src.data.advanced_signal_engineering import AdvancedSignalEngine

# Initialize signal engine
signal_engine = AdvancedSignalEngine({"lookback_period": 252})

# Generate signals for market data
signals = signal_engine.generate_all_signals(market_data, "AAPL")
```

### RL Agent Training
```python
from src.agents.hierarchical_rl_agents import HierarchicalRLSystem

# Initialize RL system
rl_system = HierarchicalRLSystem({
    "allocator_state_dim": 50,
    "executor_state_dim": 20,
    "assets": ["AAPL", "GOOGL", "MSFT"]
})

# Train agents
rl_system.train_agents(market_data, signals)
```

### Risk Management
```python
from src.risk.covariance_aware_risk import CovarianceAwareRiskManager

# Initialize risk manager
risk_manager = CovarianceAwareRiskManager(risk_config)

# Manage portfolio risk
risk_results = risk_manager.manage_portfolio_risk(returns_data, portfolio_weights)
```

### Backtesting
```python
from src.backtesting.advanced_backtesting import VectorizedBacktester

# Initialize backtester
backtester = VectorizedBacktester(backtest_config)

# Run backtest
results = backtester.run_backtest(market_data, strategy_function)
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

Run specific test categories:
```bash
pytest tests/test_agents.py          # RL agent tests
pytest tests/test_backtesting.py     # Backtesting tests
pytest tests/test_risk_management.py # Risk management tests
```

## 📈 Performance Monitoring

The system provides comprehensive performance monitoring:

- **Real-time Metrics**: Portfolio performance, risk metrics, and agent performance
- **Risk Attribution**: Factor decomposition and risk contribution analysis
- **Backtesting Results**: Historical performance analysis with confidence intervals
- **LLM Insights**: Sentiment analysis and event impact assessment

## 🔒 Security & Compliance

- **Data Encryption**: All sensitive data encrypted at rest and in transit
- **API Key Management**: Secure storage and rotation of API keys
- **Audit Logging**: Comprehensive logging for compliance and debugging
- **Access Control**: Role-based access control for different user types

## 🚀 Deployment

### Azure Deployment

1. **Create Azure resources**
```bash
az group create --name wealtharena-rg --location eastus
az acr create --resource-group wealtharena-rg --name wealtharenaregistry --sku Basic
```

2. **Deploy with Kubernetes**
```bash
kubectl apply -f k8s/
```

### Local Development

1. **Start development environment**
```bash
docker-compose up -d
```

2. **Access services**
- API: http://localhost:8000
- Monitoring: http://localhost:3000
- Database: localhost:5432

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Stable Baselines3** for RL algorithms
- **TA-Lib** for technical analysis
- **Hugging Face** for transformer models
- **OpenAI** for LLM integration
- **Yahoo Finance** for market data

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Email: support@wealtharena.com
- Documentation: https://docs.wealtharena.com

## 🔮 Roadmap

### Phase 1 (Completed)
- ✅ Multi-asset data collection
- ✅ Advanced signal engineering
- ✅ Hierarchical RL agents
- ✅ Offline RL pretraining
- ✅ Multi-objective optimization
- ✅ Risk management
- ✅ Advanced backtesting
- ✅ LLM integration

### Phase 2 (In Progress)
- 🔄 Production deployment
- 🔄 Real-time streaming
- 🔄 Web dashboard
- 🔄 User authentication
- 🔄 Tournament system

### Phase 3 (Planned)
- 📋 Mobile app
- 📋 Advanced analytics
- 📋 Social trading features
- 📋 Regulatory compliance tools

---

**WealthArena RL System** - Empowering intelligent portfolio management through reinforcement learning.