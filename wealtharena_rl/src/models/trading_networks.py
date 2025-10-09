"""
Trading Neural Networks for WealthArena

This module contains custom neural network architectures for the WealthArena
trading system, including LSTM-based and Transformer-based models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TradingLSTMModel(nn.Module):
    """
    LSTM-based trading model for WealthArena
    
    A custom neural network model that uses LSTM layers to process sequential
    market data and generate trading actions. Designed for RLlib integration.
    """
    
    def __init__(self, 
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 **kwargs):
        super().__init__()
        
        self.obs_space = obs_space
        self.action_space = action_space
        self.num_outputs = num_outputs
        self.model_config = model_config
        self.name = name
        
        # Model configuration
        self.hidden_size = model_config.get("hidden_size", 128)
        self.num_layers = model_config.get("num_layers", 2)
        self.dropout = model_config.get("dropout", 0.1)
        self.strategy = model_config.get("strategy", "balanced")
        
        # Input dimensions
        self.input_size = obs_space.shape[0]
        self.output_size = action_space.shape[0] if hasattr(action_space, 'shape') else num_outputs
        
        # Build network
        self._build_network()
        
        logger.info(f"TradingLSTMModel initialized: {self.input_size} -> {self.output_size}")
    
    def _build_network(self):
        """Build the LSTM network architecture"""
        
        # Input projection layer
        self.input_projection = nn.Linear(self.input_size, self.hidden_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Output layers
        self.actor_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size // 2, self.output_size),
            nn.Tanh()  # Output in [-1, 1] for continuous actions
        )
        
        self.critic_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size // 2, 1)
        )
        
        # Strategy-specific layers
        if self.strategy == "conservative":
            self.strategy_layer = nn.Linear(self.hidden_size, self.hidden_size // 4)
        elif self.strategy == "aggressive":
            self.strategy_layer = nn.Linear(self.hidden_size, self.hidden_size // 4)
        else:  # balanced
            self.strategy_layer = nn.Linear(self.hidden_size, self.hidden_size // 2)
    
    def forward(self, input_dict, state, seq_lens):
        """Forward pass through the network"""
        
        # Extract observations
        if isinstance(input_dict, dict):
            obs = input_dict["obs"]
        else:
            obs = input_dict
        
        # Handle sequence lengths
        if seq_lens is not None:
            # Pack sequence for LSTM
            packed_obs = nn.utils.rnn.pack_padded_sequence(
                obs, seq_lens, batch_first=True, enforce_sorted=False
            )
        else:
            # Add sequence dimension if not present
            if len(obs.shape) == 2:
                obs = obs.unsqueeze(1)  # Add time dimension
            packed_obs = obs
        
        # Input projection
        projected_obs = self.input_projection(obs)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(projected_obs)
        
        # Apply attention
        if len(lstm_out.shape) == 3:  # [batch, seq, hidden]
            attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            # Use the last timestep
            lstm_out = attended_out[:, -1, :]
        else:
            # Single timestep
            attended_out, _ = self.attention(
                lstm_out.unsqueeze(1), 
                lstm_out.unsqueeze(1), 
                lstm_out.unsqueeze(1)
            )
            lstm_out = attended_out.squeeze(1)
        
        # Apply strategy-specific processing
        strategy_out = self.strategy_layer(lstm_out)
        
        # Generate outputs
        actor_out = self.actor_head(strategy_out)
        critic_out = self.critic_head(strategy_out)
        
        # Apply strategy-specific modifications
        if self.strategy == "conservative":
            # Reduce action magnitude for conservative strategy
            actor_out = actor_out * 0.5
        elif self.strategy == "aggressive":
            # Increase action magnitude for aggressive strategy
            actor_out = actor_out * 1.5
        
        # Clamp actions to valid range
        actor_out = torch.clamp(actor_out, -1.0, 1.0)
        
        return actor_out, state
    
    def value_function(self):
        """Get the value function output"""
        return self.critic_out
    
    def get_initial_state(self):
        """Get initial hidden state for LSTM"""
        return [
            torch.zeros(self.num_layers, 1, self.hidden_size),
            torch.zeros(self.num_layers, 1, self.hidden_size)
        ]


class TradingTransformerModel(nn.Module):
    """
    Transformer-based trading model for WealthArena
    
    A custom neural network model that uses Transformer layers to process
    sequential market data and generate trading actions.
    """
    
    def __init__(self, 
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 **kwargs):
        super().__init__()
        
        self.obs_space = obs_space
        self.action_space = action_space
        self.num_outputs = num_outputs
        self.model_config = model_config
        self.name = name
        
        # Model configuration
        self.hidden_size = model_config.get("hidden_size", 128)
        self.num_heads = model_config.get("num_heads", 8)
        self.num_layers = model_config.get("num_layers", 4)
        self.dropout = model_config.get("dropout", 0.1)
        self.max_seq_length = model_config.get("max_seq_length", 100)
        
        # Input dimensions
        self.input_size = obs_space.shape[0]
        self.output_size = action_space.shape[0] if hasattr(action_space, 'shape') else num_outputs
        
        # Build network
        self._build_network()
        
        logger.info(f"TradingTransformerModel initialized: {self.input_size} -> {self.output_size}")
    
    def _build_network(self):
        """Build the Transformer network architecture"""
        
        # Input embedding
        self.input_embedding = nn.Linear(self.input_size, self.hidden_size)
        self.positional_encoding = self._create_positional_encoding()
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_size * 4,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
        
        # Output layers
        self.actor_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size // 2, self.output_size),
            nn.Tanh()
        )
        
        self.critic_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size // 2, 1)
        )
    
    def _create_positional_encoding(self):
        """Create positional encoding for Transformer"""
        
        pe = torch.zeros(self.max_seq_length, self.hidden_size)
        position = torch.arange(0, self.max_seq_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, self.hidden_size, 2).float() * 
                           (-np.log(10000.0) / self.hidden_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def forward(self, input_dict, state, seq_lens):
        """Forward pass through the network"""
        
        # Extract observations
        if isinstance(input_dict, dict):
            obs = input_dict["obs"]
        else:
            obs = input_dict
        
        # Add sequence dimension if not present
        if len(obs.shape) == 2:
            obs = obs.unsqueeze(1)  # Add time dimension
        
        batch_size, seq_len, _ = obs.shape
        
        # Input embedding
        embedded = self.input_embedding(obs)
        
        # Add positional encoding
        if seq_len <= self.max_seq_length:
            pos_encoding = self.positional_encoding[:, :seq_len, :]
            embedded = embedded + pos_encoding
        
        # Create attention mask for padding
        if seq_lens is not None:
            attention_mask = self._create_attention_mask(seq_lens, seq_len)
        else:
            attention_mask = None
        
        # Transformer forward pass
        transformer_out = self.transformer(
            embedded,
            src_key_padding_mask=attention_mask
        )
        
        # Use the last timestep
        last_output = transformer_out[:, -1, :]
        
        # Generate outputs
        actor_out = self.actor_head(last_output)
        critic_out = self.critic_head(last_output)
        
        # Clamp actions to valid range
        actor_out = torch.clamp(actor_out, -1.0, 1.0)
        
        return actor_out, state
    
    def _create_attention_mask(self, seq_lens, max_len):
        """Create attention mask for padded sequences"""
        
        batch_size = len(seq_lens)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        
        for i, seq_len in enumerate(seq_lens):
            if seq_len < max_len:
                mask[i, seq_len:] = True
        
        return mask
    
    def value_function(self):
        """Get the value function output"""
        return self.critic_out


class TradingCNNModel(nn.Module):
    """
    CNN-based trading model for WealthArena
    
    A custom neural network model that uses convolutional layers to process
    market data as images and generate trading actions.
    """
    
    def __init__(self, 
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 **kwargs):
        super().__init__()
        
        self.obs_space = obs_space
        self.action_space = action_space
        self.num_outputs = num_outputs
        self.model_config = model_config
        self.name = name
        
        # Model configuration
        self.hidden_size = model_config.get("hidden_size", 128)
        self.num_filters = model_config.get("num_filters", 64)
        self.kernel_size = model_config.get("kernel_size", 3)
        self.dropout = model_config.get("dropout", 0.1)
        
        # Input dimensions
        self.input_size = obs_space.shape[0]
        self.output_size = action_space.shape[0] if hasattr(action_space, 'shape') else num_outputs
        
        # Build network
        self._build_network()
        
        logger.info(f"TradingCNNModel initialized: {self.input_size} -> {self.output_size}")
    
    def _build_network(self):
        """Build the CNN network architecture"""
        
        # Reshape input to 2D for CNN
        self.input_reshape = nn.Linear(self.input_size, 64)  # Reshape to 8x8
        
        # CNN layers
        self.conv1 = nn.Conv2d(1, self.num_filters, kernel_size=self.kernel_size, padding=1)
        self.conv2 = nn.Conv2d(self.num_filters, self.num_filters * 2, kernel_size=self.kernel_size, padding=1)
        self.conv3 = nn.Conv2d(self.num_filters * 2, self.num_filters * 4, kernel_size=self.kernel_size, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate flattened size
        self.flattened_size = self.num_filters * 4 * 2 * 2  # After 3 pooling operations
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size // 2)
        
        # Output layers
        self.actor_head = nn.Sequential(
            nn.Linear(self.hidden_size // 2, self.output_size),
            nn.Tanh()
        )
        
        self.critic_head = nn.Sequential(
            nn.Linear(self.hidden_size // 2, 1)
        )
        
        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)
    
    def forward(self, input_dict, state, seq_lens):
        """Forward pass through the network"""
        
        # Extract observations
        if isinstance(input_dict, dict):
            obs = input_dict["obs"]
        else:
            obs = input_dict
        
        # Reshape input
        x = self.input_reshape(obs)
        x = x.view(x.size(0), 1, 8, 8)  # Reshape to 8x8 image
        
        # CNN forward pass
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout_layer(x)
        x = F.relu(self.fc2(x))
        x = self.dropout_layer(x)
        
        # Generate outputs
        actor_out = self.actor_head(x)
        critic_out = self.critic_head(x)
        
        # Clamp actions to valid range
        actor_out = torch.clamp(actor_out, -1.0, 1.0)
        
        return actor_out, state
    
    def value_function(self):
        """Get the value function output"""
        return self.critic_out


class TradingEnsembleModel(nn.Module):
    """
    Ensemble trading model for WealthArena
    
    A custom neural network model that combines multiple architectures
    (LSTM, Transformer, CNN) for robust trading decisions.
    """
    
    def __init__(self, 
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 **kwargs):
        super().__init__()
        
        self.obs_space = obs_space
        self.action_space = action_space
        self.num_outputs = num_outputs
        self.model_config = model_config
        self.name = name
        
        # Model configuration
        self.hidden_size = model_config.get("hidden_size", 128)
        self.dropout = model_config.get("dropout", 0.1)
        self.ensemble_weights = model_config.get("ensemble_weights", [0.4, 0.3, 0.3])
        
        # Input dimensions
        self.input_size = obs_space.shape[0]
        self.output_size = action_space.shape[0] if hasattr(action_space, 'shape') else num_outputs
        
        # Build ensemble models
        self._build_ensemble()
        
        logger.info(f"TradingEnsembleModel initialized: {self.input_size} -> {self.output_size}")
    
    def _build_ensemble(self):
        """Build ensemble of different model architectures"""
        
        # LSTM model
        self.lstm_model = TradingLSTMModel(
            self.obs_space, self.action_space, self.num_outputs,
            self.model_config, f"{self.name}_lstm"
        )
        
        # Transformer model
        self.transformer_model = TradingTransformerModel(
            self.obs_space, self.action_space, self.num_outputs,
            self.model_config, f"{self.name}_transformer"
        )
        
        # CNN model
        self.cnn_model = TradingCNNModel(
            self.obs_space, self.action_space, self.num_outputs,
            self.model_config, f"{self.name}_cnn"
        )
        
        # Ensemble fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.output_size * 3, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.output_size),
            nn.Tanh()
        )
        
        # Value function fusion
        self.value_fusion = nn.Sequential(
            nn.Linear(3, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 1)
        )
    
    def forward(self, input_dict, state, seq_lens):
        """Forward pass through the ensemble"""
        
        # Get predictions from each model
        lstm_out, _ = self.lstm_model(input_dict, state, seq_lens)
        transformer_out, _ = self.transformer_model(input_dict, state, seq_lens)
        cnn_out, _ = self.cnn_model(input_dict, state, seq_lens)
        
        # Combine outputs
        combined_out = torch.cat([lstm_out, transformer_out, cnn_out], dim=-1)
        ensemble_out = self.fusion_layer(combined_out)
        
        # Combine value functions
        lstm_value = self.lstm_model.value_function()
        transformer_value = self.transformer_model.value_function()
        cnn_value = self.cnn_model.value_function()
        
        combined_values = torch.cat([lstm_value, transformer_value, cnn_value], dim=-1)
        ensemble_value = self.value_fusion(combined_values)
        
        return ensemble_out, state
    
    def value_function(self):
        """Get the ensemble value function output"""
        return self.ensemble_value


# Model registration functions
def register_trading_models():
    """Register custom models with RLlib"""
    
    try:
        from ray.rllib.models import ModelCatalog
        
        # Register LSTM model
        ModelCatalog.register_custom_model("trading_lstm", TradingLSTMModel)
        
        # Register Transformer model
        ModelCatalog.register_custom_model("trading_transformer", TradingTransformerModel)
        
        # Register CNN model
        ModelCatalog.register_custom_model("trading_cnn", TradingCNNModel)
        
        # Register Ensemble model
        ModelCatalog.register_custom_model("trading_ensemble", TradingEnsembleModel)
        
        logger.info("Custom trading models registered with RLlib")
        
    except Exception as e:
        logger.error(f"Error registering custom models: {e}")


if __name__ == "__main__":
    # Test the models
    import torch
    
    # Create dummy spaces
    obs_space = type('obs_space', (), {'shape': (100,)})
    action_space = type('action_space', (), {'shape': (10,)})
    
    # Test LSTM model
    lstm_model = TradingLSTMModel(
        obs_space, action_space, 10,
        {"hidden_size": 64, "num_layers": 2, "strategy": "balanced"},
        "test_lstm"
    )
    
    # Test forward pass
    dummy_input = torch.randn(32, 100)
    output, state = lstm_model({"obs": dummy_input}, None, None)
    
    print(f"LSTM Model Output Shape: {output.shape}")
    print(f"LSTM Model Value Shape: {lstm_model.value_function().shape}")
    
    # Test Transformer model
    transformer_model = TradingTransformerModel(
        obs_space, action_space, 10,
        {"hidden_size": 64, "num_heads": 4, "num_layers": 2},
        "test_transformer"
    )
    
    output, state = transformer_model({"obs": dummy_input}, None, None)
    print(f"Transformer Model Output Shape: {output.shape}")
    
    # Test Ensemble model
    ensemble_model = TradingEnsembleModel(
        obs_space, action_space, 10,
        {"hidden_size": 64, "ensemble_weights": [0.4, 0.3, 0.3]},
        "test_ensemble"
    )
    
    output, state = ensemble_model({"obs": dummy_input}, None, None)
    print(f"Ensemble Model Output Shape: {output.shape}")
    
    print("All models tested successfully!")
