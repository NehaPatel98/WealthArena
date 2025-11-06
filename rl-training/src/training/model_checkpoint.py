"""
Model Checkpoint Management

This module provides functionality to save and load model checkpoints,
including neural network weights, training state, and optimizer state.
"""

import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import json
import yaml

logger = logging.getLogger(__name__)


class ModelCheckpoint:
    """Manages model checkpoints for production use"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, 
                       agent_name: str,
                       model_state: Dict[str, Any],
                       training_state: Dict[str, Any],
                       optimizer_state: Optional[Dict[str, Any]] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save complete model checkpoint"""
        
        # Create agent-specific directory
        agent_dir = self.checkpoint_dir / agent_name
        agent_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate checkpoint timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_id = f"{agent_name}_{timestamp}"
        
        # Save model weights (.pt file)
        model_path = agent_dir / f"{checkpoint_id}_model.pt"
        torch.save(model_state, model_path)
        
        # Save training state (.pkl file)
        training_path = agent_dir / f"{checkpoint_id}_training.pkl"
        with open(training_path, 'wb') as f:
            pickle.dump(training_state, f)
        
        # Save optimizer state if provided
        if optimizer_state:
            optimizer_path = agent_dir / f"{checkpoint_id}_optimizer.pkl"
            with open(optimizer_path, 'wb') as f:
                pickle.dump(optimizer_state, f)
        
        # Save metadata
        metadata_path = agent_dir / f"{checkpoint_id}_metadata.json"
        checkpoint_metadata = {
            "checkpoint_id": checkpoint_id,
            "agent_name": agent_name,
            "created_at": datetime.now().isoformat(),
            "model_path": str(model_path),
            "training_path": str(training_path),
            "optimizer_path": str(optimizer_path) if optimizer_state else None,
            "metadata": metadata or {}
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(checkpoint_metadata, f, indent=2)
        
        # Save latest checkpoint reference
        latest_path = agent_dir / "latest_checkpoint.json"
        with open(latest_path, 'w') as f:
            json.dump(checkpoint_metadata, f, indent=2)
        
        logger.info(f"Checkpoint saved: {checkpoint_id}")
        return checkpoint_id
    
    def load_checkpoint(self, 
                       agent_name: str, 
                       checkpoint_id: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]], Dict[str, Any]]:
        """Load model checkpoint"""
        
        agent_dir = self.checkpoint_dir / agent_name
        
        if checkpoint_id is None:
            # Load latest checkpoint - check both direct and nested directories
            latest_paths = [
                agent_dir / "latest_checkpoint.json",
                agent_dir / agent_name / "latest_checkpoint.json"
            ]
            
            latest_path = None
            for path in latest_paths:
                if path.exists():
                    latest_path = path
                    break
            
            if latest_path is None:
                raise FileNotFoundError(f"No checkpoints found for {agent_name}")
            
            with open(latest_path, 'r') as f:
                checkpoint_metadata = json.load(f)
            checkpoint_id = checkpoint_metadata["checkpoint_id"]
        else:
            # Load specific checkpoint - check both directories
            metadata_paths = [
                agent_dir / f"{checkpoint_id}_metadata.json",
                agent_dir / agent_name / f"{checkpoint_id}_metadata.json"
            ]
            
            metadata_path = None
            for path in metadata_paths:
                if path.exists():
                    metadata_path = path
                    break
            
            if metadata_path is None:
                raise FileNotFoundError(f"Checkpoint {checkpoint_id} not found")
            
            with open(metadata_path, 'r') as f:
                checkpoint_metadata = json.load(f)
        
        # Load model weights
        model_path = Path(checkpoint_metadata["model_path"])
        model_state = torch.load(model_path, map_location='cpu')
        
        # Load training state
        training_path = Path(checkpoint_metadata["training_path"])
        with open(training_path, 'rb') as f:
            training_state = pickle.load(f)
        
        # Load optimizer state if exists
        optimizer_state = None
        if checkpoint_metadata.get("optimizer_path"):
            optimizer_path = Path(checkpoint_metadata["optimizer_path"])
            if optimizer_path.exists():
                with open(optimizer_path, 'rb') as f:
                    optimizer_state = pickle.load(f)
        
        logger.info(f"Checkpoint loaded: {checkpoint_id}")
        return model_state, training_state, optimizer_state, checkpoint_metadata
    
    def list_checkpoints(self, agent_name: str) -> list:
        """List all available checkpoints for an agent"""
        
        agent_dir = self.checkpoint_dir / agent_name
        if not agent_dir.exists():
            return []
        
        checkpoints = []
        # Check both direct directory and nested directory
        search_dirs = [agent_dir, agent_dir / agent_name]
        
        for search_dir in search_dirs:
            if search_dir.exists():
                for metadata_file in search_dir.glob("*_metadata.json"):
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        checkpoints.append(metadata)
        
        # Sort by creation time
        checkpoints.sort(key=lambda x: x["created_at"], reverse=True)
        return checkpoints
    
    def delete_checkpoint(self, agent_name: str, checkpoint_id: str) -> bool:
        """Delete a specific checkpoint"""
        
        agent_dir = self.checkpoint_dir / agent_name
        metadata_path = agent_dir / f"{checkpoint_id}_metadata.json"
        
        if not metadata_path.exists():
            return False
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Delete all checkpoint files
        files_to_delete = [
            metadata["model_path"],
            metadata["training_path"]
        ]
        
        if metadata.get("optimizer_path"):
            files_to_delete.append(metadata["optimizer_path"])
        
        for file_path in files_to_delete:
            Path(file_path).unlink(missing_ok=True)
        
        # Delete metadata file
        metadata_path.unlink()
        
        logger.info(f"Checkpoint deleted: {checkpoint_id}")
        return True


class ProductionModelManager:
    """Manages models for production deployment"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_manager = ModelCheckpoint(checkpoint_dir)
        self.loaded_models = {}
    
    def load_agent_model(self, agent_name: str, checkpoint_id: Optional[str] = None):
        """Load agent model for production use"""
        
        try:
            model_state, training_state, optimizer_state, metadata = self.checkpoint_manager.load_checkpoint(
                agent_name, checkpoint_id
            )
            
            # Store in memory for quick access
            self.loaded_models[agent_name] = {
                "model_state": model_state,
                "training_state": training_state,
                "optimizer_state": optimizer_state,
                "metadata": metadata
            }
            
            logger.info(f"Agent {agent_name} loaded for production")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load agent {agent_name}: {e}")
            return False
    
    def get_model_prediction(self, agent_name: str, input_data: np.ndarray) -> np.ndarray:
        """Get prediction from loaded model"""
        
        if agent_name not in self.loaded_models:
            raise ValueError(f"Agent {agent_name} not loaded")
        
        # This is a placeholder - in real implementation, you would:
        # 1. Reconstruct the neural network architecture
        # 2. Load the model state
        # 3. Run inference on input_data
        
        # For now, return a mock prediction
        return np.random.uniform(-1, 1, size=(input_data.shape[0], 7))  # 7 currency pairs
    
    def get_available_agents(self) -> list:
        """Get list of available agents"""
        return list(self.loaded_models.keys())
    
    def unload_agent(self, agent_name: str):
        """Unload agent from memory"""
        if agent_name in self.loaded_models:
            del self.loaded_models[agent_name]
            logger.info(f"Agent {agent_name} unloaded")


def create_mock_model_state(agent_name: str, num_assets: int) -> Dict[str, Any]:
    """Create mock model state for demonstration"""
    
    # Mock neural network weights
    model_state = {
        "policy_network": {
            "layer1.weight": torch.randn(64, num_assets * 20),  # Input features
            "layer1.bias": torch.randn(64),
            "layer2.weight": torch.randn(32, 64),
            "layer2.bias": torch.randn(32),
            "output.weight": torch.randn(num_assets, 32),
            "output.bias": torch.randn(num_assets)
        },
        "value_network": {
            "layer1.weight": torch.randn(64, num_assets * 20),
            "layer1.bias": torch.randn(64),
            "layer2.weight": torch.randn(32, 64),
            "layer2.bias": torch.randn(32),
            "output.weight": torch.randn(1, 32),
            "output.bias": torch.randn(1)
        }
    }
    
    return model_state


def create_mock_training_state(agent_name: str) -> Dict[str, Any]:
    """Create mock training state for demonstration"""
    
    return {
        "episode": 1000,
        "total_reward": 2.5,
        "best_reward": 3.2,
        "convergence_episode": 800,
        "training_loss": 0.3,
        "validation_reward": 2.1,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "epsilon": 0.1,
        "training_history": {
            "episode_rewards": np.random.uniform(0, 3, 1000).tolist(),
            "losses": np.random.uniform(0, 1, 1000).tolist(),
            "validation_rewards": np.random.uniform(0, 3, 100).tolist()
        }
    }


def create_mock_optimizer_state(agent_name: str) -> Dict[str, Any]:
    """Create mock optimizer state for demonstration"""
    
    return {
        "state": {
            "param_groups": [
                {
                    "params": [0, 1, 2, 3, 4, 5],  # Parameter indices
                    "lr": 3e-4,
                    "weight_decay": 1e-5
                }
            ],
            "step": 1000,
            "exp_avg": np.random.randn(6).tolist(),
            "exp_avg_sq": np.random.randn(6).tolist()
        }
    }
