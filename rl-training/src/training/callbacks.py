"""
Ray RLlib Callbacks for WealthArena Training

Custom callbacks for tracking win rate and other custom metrics.
"""

from typing import Dict, Any
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import RolloutWorker, Episode
from ray.rllib.policy import Policy


class WinRateCallback(DefaultCallbacks):
    """Callback to track win rate from environment info"""
    
    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        """Called at the end of each episode"""
        # Get win rate from episode info
        episode_info = episode.last_info_for()
        if episode_info:
            win_rate = episode_info.get('win_rate', 0.0)
            # Set custom metric for Ray to track
            episode.custom_metrics['win_rate'] = win_rate

