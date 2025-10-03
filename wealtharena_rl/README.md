# WealthArena Multi-Agent RL Trading System

## Project Overview

WealthArena is a **multi-agent reinforcement learning trading system** that coordinates multiple RL agents across different financial instrument types using hierarchical RL and advanced coordination mechanisms. The system uses RLlib for distributed reinforcement learning and implements sophisticated reward functions, risk management, and signal fusion.

## Architecture

### RL Agent Types
- [x] **Stock RL Agent (PPO)**: ASX stock trading with momentum strategies
- [x] **Crypto RL Agent (SAC)**: High-frequency crypto trading with volatility management
- [x] **ETF RL Agent (A2C)**: Sector rotation and diversification strategies
- [x] **Currency RL Agent (DQN)**: Forex trading with carry strategies
- [x] **REIT RL Agent (PPO)**: Real estate investment trust trading

### Multi-Agent RL Environment
- [x] **Hierarchical RL Coordination**: High-level allocator + low-level execution
- [x] **Signal Fusion**: Weighted, voting, and neural fusion methods
- [x] **Risk Management**: Integrated position sizing and risk controls
- [x] **Reward Shaping**: Multi-objective optimization with coordination rewards

### Training Pipeline
- [x] **Individual Agent Training**: RL agents for each instrument type
- [x] **Multi-Agent Environment Training**: Coordinated training across agents
- [x] **Meta Agent Training**: Hierarchical coordination training
- [x] **Daily Timeframe Optimization**: Optimized for daily trading decisions

## Directory Structure

```
wealtharena_rl/
├── README.md
├── requirements.txt
├── config/
│   └── training_config.yaml
├── docs/
│   ├── rl_architecture.md
│   └── rl_api_specification.md
├── src/
│   ├── environments/
│   │   ├── __init__.py
│   │   ├── trading_env.py
│   │   ├── multi_agent_env.py
│   │   ├── multi_agent_rl_env.py
│   │   └── market_simulator.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── rl_agents.py
│   │   ├── rl_meta_agent.py
│   │   ├── trading_networks.py
│   │   └── custom_policies.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── daily_timeframe_trainer.py
│   │   ├── train_multi_agent.py
│   │   └── evaluation.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── asx_companies.py
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

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Individual RL Agents
```python
from src.models.rl_agents import RLAgentFactory

# Create and train RL agents
for instrument_type in ["stocks", "crypto", "etf", "currencies", "reits"]:
    agent = RLAgentFactory.create_agent(instrument_type)
    results = agent.train(training_data, num_iterations=1000)
    print(f"{instrument_type} agent trained: final reward = {results['final_reward']:.4f}")
```

### 3. Multi-Agent RL Environment
```python
from src.environments.multi_agent_rl_env import WealthArenaMultiAgentRLEnv

# Create multi-agent environment
env = WealthArenaMultiAgentRLEnv(config)

# Train agents in environment
obs, info = env.reset()
for step in range(1000):
    actions = {agent_id: env.action_space.sample() for agent_id in env.agent_ids}
    obs, rewards, terminateds, truncateds, infos = env.step(actions)
```

### 4. RL Meta Agent
```python
from src.models.rl_meta_agent import RLMetaAgent, RLMetaAgentConfig

# Create meta agent
config = RLMetaAgentConfig(
    coordination_method="hierarchical",
    fusion_method="weighted",
    agent_weights={"stocks": 0.3, "crypto": 0.2, "etf": 0.2, "currencies": 0.2, "reits": 0.1}
)
meta_agent = RLMetaAgent(config)

# Train meta agent
results = meta_agent.train_meta_agent(training_data)

# Generate trading signals
signals = meta_agent.generate_signals(market_data)
print(f"Trading decision: {signals['trading_decision']}")
```

### 5. Full Training Pipeline
```bash
# Train RL agents
python train_wealtharena_system.py --train-models-only

# Run full RL system
python train_wealtharena_system.py --config config/training_config.yaml
```

## Key Features

- **Multi-Agent RL**: Multiple RL agents with different strategies (PPO, SAC, A2C, DQN)
- **Hierarchical RL**: High-level allocator + low-level execution policies
- **Signal Fusion**: Weighted, voting, and neural fusion methods
- **Risk Management**: Integrated position sizing and risk controls
- **Reward Shaping**: Multi-objective optimization with coordination rewards
- **Daily Timeframe**: Optimized for daily trading decisions
- **ASX Integration**: Comprehensive ASX stock data and company information
- **Experiment Tracking**: MLflow and Weights & Biases integration

## RL Algorithms

- **PPO (Proximal Policy Optimization)**: Stock and REIT agents
- **SAC (Soft Actor-Critic)**: Crypto agent for continuous actions
- **A2C (Advantage Actor-Critic)**: ETF agent for faster convergence
- **DQN (Deep Q-Network)**: Currency agent for discrete actions

## Documentation

- [RL Architecture](docs/rl_architecture.md)
- [RL API Specification](docs/rl_api_specification.md)
