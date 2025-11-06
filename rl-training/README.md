# WealthArena RL System

A comprehensive reinforcement learning platform for multi-asset portfolio management, featuring advanced signal engineering, hierarchical RL agents, risk management, and LLM integration.

## ⚠️ EMERGENCY: Yahoo Finance Blocked

**If you're seeing "Expecting value: line 1 column 1 (char 0)" errors**, Yahoo Finance API is completely blocked or unavailable. This is a **CRITICAL SITUATION** that prevents all live data downloads.

### Immediate Solution: Use Synthetic Data

**The system now automatically uses synthetic data when Yahoo Finance is blocked.** However, you can force it explicitly:

```bash
python collect_training_data.py --stocks --use-synthetic-data --max-stocks-symbols 20
```

This will:
- **Bypass Yahoo Finance completely** - No API calls attempted
- **Generate realistic synthetic OHLCV data** - Uses geometric Brownian motion with correlations
- **Save data to `data/processed/`** - Can be reused for training
- **Complete in 5-10 minutes** - Instead of 45+ minutes of failures

### Why Synthetic Data Works

Synthetic data is **sufficient for training and testing** RL agents because:
- Uses realistic price movements and correlations between stocks
- Includes proper OHLCV data with technical indicators
- Generated quickly without API dependencies
- Can be customized with parameters (volatility, correlations, etc.)

### Quick Start with Synthetic Data

1. **Generate synthetic data:**
   ```bash
   python collect_training_data.py --stocks --use-synthetic-data --max-stocks-symbols 20
   ```

2. **Verify data:**
   Check `data/processed/` directory for CSV files

3. **Start training:**
   ```bash
   python train.py --config config/production_config.yaml
   ```

4. **Monitor progress:**
   Check `logs/training.log`

**Total time: 5-10 minutes** instead of hours of failures.

### Understanding the Error

The "Expecting value: line 1 column 1" error means:
- Yahoo Finance API is **rate limiting your IP address completely**
- **Automated access is blocked**
- **Service outage** or maintenance
- **Authentication/headers required** that aren't being sent

This is **NOT about specific symbols** - it's a complete API blockage.

### When to Use Real Data

Once API access is restored:
- Remove `--use-synthetic-data` flag
- Use `--force-live-download` to override cache
- Real data will be downloaded automatically

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

## 📊 Synthetic Data Generation

### Overview

Synthetic data generation provides a **reliable fallback** when Yahoo Finance or other data sources are unavailable. It uses **geometric Brownian motion** with realistic correlations to generate OHLCV stock data.

### Parameters

The synthetic data generator uses these default parameters:
- **Daily return mean**: 0.0005 (0.05% per day, ~13% annual)
- **Daily volatility**: 0.02 (2% per day, ~32% annual)
- **Correlation between stocks**: 0.3 (moderate correlation)
- **Starting price**: $100 per stock
- **Volume**: Log-normal distribution with mean 1M shares

### Customizing Parameters

You can customize synthetic data parameters in the configuration:

```yaml
synthetic_data:
  enabled: true
  num_symbols: 20
  start_date: "2020-01-01"
  end_date: "2024-12-31"
  daily_return_mean: 0.0005
  daily_volatility: 0.02
  correlation: 0.3
  starting_price: 100.0
  volume_mean: 1000000
```

### Usage

**Generate synthetic data:**
```bash
python collect_training_data.py --stocks --use-synthetic-data --max-stocks-symbols 20
```

**Training with synthetic data:**
The training script automatically uses synthetic data when Yahoo Finance is blocked. No additional configuration needed.

### Verifying Data Quality

After generating synthetic data:
1. Check `data/processed/synthetic_data_metadata.json` for generation parameters
2. Verify CSV files in `data/processed/` directory
3. Check logs for generation statistics

### Caching

Generated synthetic data is automatically saved to `data/processed/` and can be reused across training runs. The cache is checked before regenerating.

## 🔧 Troubleshooting Data Collection Issues

### Common Errors and Solutions

#### "No timezone found, symbol may be delisted"
This error indicates that Yahoo Finance cannot find data for a symbol, typically because:
- The symbol is delisted or no longer trades
- The symbol format is incorrect (e.g., missing .AX suffix for ASX stocks)
- The symbol has been renamed or merged with another company

**Solutions:**
- Use verified symbol lists (enable `--use-verified-symbols` flag)
- Check symbol format matches exchange conventions
- Filter symbols before download using `validate_symbols_before_download()`

#### "Expecting value: line 1 column 1 (char 0)"
This JSON parsing error typically indicates:
- Yahoo Finance API rate limiting
- Network connectivity issues
- API service outage

**Solutions:**
- Reduce batch size (use `--batch-size 5` or lower)
- Increase sleep time between batches (use `--sleep-between 8.0` or higher)
- Reduce number of workers (use `--max-workers 2` or lower)
- Wait 1 hour and try again (rate limits are temporary)
- Use cached data if available (`allow_cached_data: true`)

#### High Failure Rate (>50%)
If most symbols fail to download:
- **API Rate Limiting**: Reduce batch size and increase sleep times
- **Many Delisted Symbols**: Use verified symbol lists only
- **Network Issues**: Check internet connection
- **API Outage**: Try again later

**Recommended Settings for Slow/Unreliable Networks:**
```bash
python collect_training_data.py \
  --stocks \
  --exchanges ASX \
  --max-stocks-symbols 10 \
  --batch-size 2 \
  --sleep-between 15.0 \
  --max-workers 1
```

### Fallback Mechanisms

The system includes multiple fallback mechanisms:

1. **Cached Data**: Automatically uses cached data from previous runs (< 7 days old)
2. **Verified Symbols**: Falls back to curated list of known-good symbols
3. **Minimal Dataset**: Generates minimal dataset if all else fails (training may have reduced quality)

To enable fallback:
```yaml
data:
  allow_cached_data: true
  
data_collection:
  fallback_strategy: "cached_then_minimal"
  min_symbols_required: 5
```

### Manual Symbol Validation

Before running training, validate symbols:
```python
from src.data.asx.asx_symbols import get_verified_symbols, is_symbol_likely_valid

# Get verified symbols
verified = get_verified_symbols(limit=20)

# Check individual symbol
is_valid = is_symbol_likely_valid("BHP.AX")
```

## 📚 Best Practices

### Data Collection

1. **Start Small**: Begin with 10-20 symbols for initial testing
   ```bash
   --max-stocks-symbols 20
   ```

2. **Use Verified Lists**: Enable verified symbols for production training
   ```yaml
   data:
     asx:
       use_verified_symbols_only: true
   ```

3. **Enable Caching**: Cache data to avoid repeated downloads
   ```yaml
   data:
     cache_enabled: true
     allow_cached_data: true
   ```

4. **Monitor API Status**: Check Yahoo Finance API status before large downloads
   - Use off-peak hours when possible
   - Monitor error logs for rate limiting patterns

5. **Conservative Settings**: Use conservative batch sizes and sleep times
   ```yaml
   data:
     asx:
       batch_size: 5
       sleep_between: 8.0
       max_workers: 2
   ```

### Symbol Selection

1. **Preferred Sources** (in order):
   - Verified symbol lists (`get_verified_symbols()`)
   - ASX 200 symbols (highly liquid)
   - Custom curated lists

2. **Avoid**:
   - Recently delisted symbols
   - Symbols with low liquidity
   - Symbols with unusual naming patterns

### Error Handling

1. **Enable Detailed Logging**:
   ```python
   logging.basicConfig(level=logging.INFO)
   ```

2. **Save Failed Symbols**: Check `logs/failed_symbols.txt` after failures

3. **Review Logs**: Check `logs/data_download.log` for detailed error information

4. **Incremental Testing**: Test with small symbol counts before full runs

### Training with Partial Data

The system can train with partial data:
- Minimum 5 symbols required (configurable via `min_symbols_required`)
- Training quality may be reduced with fewer symbols
- Use cached data to supplement missing symbols

---

## Troubleshooting Data Collection Issues

### Common Errors and Solutions

#### "No timezone found, symbol may be delisted"
**Cause**: The symbol is no longer actively traded on the exchange.  
**Solution**: 
- The system automatically skips delisted symbols
- Use the `--use-verified-asx` flag to filter to known active stocks
- Check the symbol list against current exchange listings

#### "Expecting value: line 1 column 1 (char 0)"
**Cause**: Yahoo Finance API rate limiting or service outage.  
**Solution**:
- Reduce `--batch-size` to 3 or less
- Increase `--sleep-between` to 10+ seconds
- Set `--max-workers` to 1 for sequential downloads
- Wait a few minutes and retry

#### "'dict' object has no attribute 'empty'"
**Cause**: Type mismatch bug (fixed in current version).  
**Solution**: 
- Update to latest version of the code
- If issue persists, check that all dependencies are updated
- Report as bug with full traceback

#### "Connection timeout" or "Read timed out"
**Cause**: Network issues or API overload.  
**Solution**:
- Check internet connection stability
- Increase timeout values in configuration
- Use cached data from previous runs
- Try again during off-peak hours

### Recovery Procedures

#### Step 1: Use Conservative Settings
```bash
python collect_training_data.py --stocks --exchanges ASX \
  --max-stocks-symbols 20 \
  --batch-size 3 \
  --sleep-between 10.0 \
  --max-workers 1
```

#### Step 2: Use Verified Symbols (if available)
```bash
# If --use-verified-asx flag is supported
python collect_training_data.py --stocks --exchanges ASX --use-verified-asx \
  --max-stocks-symbols 20 --batch-size 3 --sleep-between 10.0 --max-workers 1
```

#### Step 3: Use Cached Data
If live collection fails, the system will automatically fallback to cached data from previous runs stored in `data/processed/`.

#### Step 4: Minimal Fallback
If no cache exists, the system will attempt to download a minimal verified symbol set (top 10 ASX stocks).

---

## Quick Start for Reliable Training

### Recommended Settings

For reliable data collection, use these conservative settings:

```yaml
data:
  symbols: []  # Empty list triggers ASX collection
  max_symbols: 20  # Small number for reliability
  cache_enabled: true
  allow_cached_data: true
  
asx:
  use_verified_symbols_only: true
  max_symbols: 20
  batch_size: 3
  sleep_between: 10.0
  max_workers: 1

data_collection:
  fallback_strategy: "cached_then_minimal"
  min_symbols_required: 5
  use_verified_symbols_only: true
  max_consecutive_failures: 10
  circuit_breaker_cooldown_seconds: 60
  enable_symbol_validation: true
  validation_sample_size: 10
```

### Example Command

```bash
# Start with minimal configuration
python train.py --config config/production_config.yaml --experiment wealtharena_production --local

# The system will automatically:
# 1. Try to collect ASX stock data with conservative settings
# 2. Fallback to cached data if live collection fails
# 3. Use minimal verified symbol set if no cache exists
# 4. Proceed with training if at least 5 symbols are available
```

### Fallback Mechanisms

The system uses a multi-layer fallback strategy:

1. **Layer 1**: Try live data collection with validated symbols
2. **Layer 2**: If live collection fails, use cached data from `data/processed/`
3. **Layer 3**: If no cache, use minimal verified symbol set (top 10 ASX stocks)
4. **Layer 4**: If all else fails, log error and suggest checking logs

Check the logs to see which layer was used:
- `[FALLBACK] Loaded X symbols from cache` - Layer 2 active
- `[FALLBACK] Using X verified symbols` - Layer 3 active
- Error messages with recommendations - Manual intervention needed

---

**WealthArena RL System** - Empowering intelligent portfolio management through reinforcement learning.