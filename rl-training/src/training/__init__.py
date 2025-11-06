"""
WealthArena Training Module

This module contains training scripts and utilities for the WealthArena trading system,
including multi-agent training with RLlib and hyperparameter tuning.
"""

from .train_multi_agent import MultiAgentTrainer, train_multi_agent
from .evaluation import TradingEvaluator, evaluate_agents

__all__ = [
    "MultiAgentTrainer",
    "train_multi_agent",
    "TradingEvaluator", 
    "evaluate_agents"
]
