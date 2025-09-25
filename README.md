# WealthArena Multi-Agent Trading RL Pipeline

## Project Overview

This repository contains the implementation of a multi-agent reinforcement learning trading system for WealthArena using RLlib. The system is designed to handle multiple trading agents operating in a shared market environment with sophisticated observation spaces, action spaces, and reward functions.

## Architecture

### Week 1: Foundation & Architecture Setup
- [x] Agent API specification (observation, action, reward spaces)
- [x] Technical specification documentation
- [x] TradingEnv skeleton (Gym-compatible)
- [x] Integration points for data and risk systems
- [x] Experiment tracking hooks (MLflow/W&B)
- [x] Audit artefact checklist

### Week 2: Data Infrastructure & Core Services
- [ ] Full TradingEnv implementation with Gym interface
- [ ] Market data pipeline integration (SYS1 API)
- [ ] Reward function and portfolio tracking
- [ ] Agent training loop with RLlib

## Directory Structure

```
wealtharena_rllib/
├── README.md
├── requirements.txt
├── config/
│   ├── training_config.yaml
│   ├── deployment_config.yaml
│   └── hyperparameters.yaml
├── src/
│   ├── environments/
│   │   ├── __init__.py
│   │   ├── trading_env.py
│   │   ├── multi_agent_env.py
│   │   └── market_simulator.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── trading_networks.py
│   │   └── custom_policies.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_multi_agent.py
│   │   └── evaluation.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_adapter.py
│   │   └── market_data.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── visualization.py
│   └── tracking/
│       ├── __init__.py
│       ├── mlflow_tracker.py
│       └── wandb_tracker.py
├── experiments/
│   ├── logs/
│   ├── checkpoints/
│   └── results/
├── tests/
│   ├── test_environments.py
│   ├── test_models.py
│   └── test_training.py
├── notebooks/
│   ├── agent_api_specification.ipynb
│   ├── data_exploration.ipynb
│   └── performance_analysis.ipynb
└── docs/
    ├── agent_api_specification.md
    ├── technical_architecture.md
    └── integration_guide.md
```

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run training:
```bash
python src/training/train_multi_agent.py --config config/training_config.yaml
```

3. View results in MLflow:
```bash
mlflow ui
```

## Key Features

- **Multi-Agent RL**: Multiple trading agents with different strategies
- **Sophisticated Observation Space**: OHLCV data, technical indicators, portfolio state
- **Flexible Action Space**: Discrete and continuous trading actions
- **Advanced Reward Function**: Profit, risk, and cost considerations
- **Experiment Tracking**: MLflow and Weights & Biases integration
- **Production Ready**: Docker containerization and deployment scripts

## Documentation

- [Agent API Specification](docs/agent_api_specification.md)
- [Technical Architecture](docs/technical_architecture.md)
- [Integration Guide](docs/integration_guide.md)
