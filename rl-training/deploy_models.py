#!/usr/bin/env python3
"""
Model Deployment Script

This script demonstrates how to load and use trained models for production.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.training.model_checkpoint import ProductionModelManager, ModelCheckpoint

def main():
    """Deploy models for production use"""
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 WEALTHARENA MODEL DEPLOYMENT")
    print("="*50)
    
    # Initialize production model manager
    model_manager = ProductionModelManager("checkpoints")
    
    # List available agents
    print("\n📋 Available Agents:")
    checkpoint_manager = ModelCheckpoint("checkpoints")
    
    agents = ["currency_pairs", "asx_stocks", "cryptocurrencies", "etf", "commodities"]
    
    for agent_name in agents:
        checkpoints = checkpoint_manager.list_checkpoints(agent_name)
        if checkpoints:
            latest = checkpoints[0]
            print(f"  ✅ {agent_name}: {latest['checkpoint_id']} ({latest['created_at']})")
        else:
            print(f"  ❌ {agent_name}: No checkpoints found")
    
    # Load models for production
    print("\n🔄 Loading Models for Production...")
    
    loaded_agents = []
    for agent_name in agents:
        if checkpoint_manager.list_checkpoints(agent_name):
            success = model_manager.load_agent_model(agent_name)
            if success:
                loaded_agents.append(agent_name)
                print(f"  ✅ {agent_name} loaded successfully")
            else:
                print(f"  ❌ Failed to load {agent_name}")
        else:
            print(f"  ⏭️  Skipping {agent_name} (no checkpoints)")
    
    # Demonstrate model usage
    if loaded_agents:
        print(f"\n🎯 Production Models Ready: {', '.join(loaded_agents)}")
        
        # Mock input data (in real usage, this would be live market data)
        import numpy as np
        mock_input = np.random.randn(1, 140)  # 7 assets * 20 features
        
        print(f"\n📊 Testing Model Predictions:")
        for agent_name in loaded_agents:
            try:
                prediction = model_manager.get_model_prediction(agent_name, mock_input)
                print(f"  {agent_name}: {prediction.shape} output shape")
            except Exception as e:
                print(f"  {agent_name}: Error - {e}")
        
        print(f"\n💾 Model Files Location:")
        print(f"  Checkpoints: {Path('checkpoints').absolute()}")
        print(f"  Results: {Path('results').absolute()}")
        
        print(f"\n📋 Model Sharing Instructions:")
        print(f"  1. Copy the entire 'checkpoints' folder to your team")
        print(f"  2. Copy the 'results' folder for evaluation reports")
        print(f"  3. Use ProductionModelManager to load models in production")
        print(f"  4. Each model includes:")
        print(f"     - Neural network weights (.pt files)")
        print(f"     - Training state (.pkl files)")
        print(f"     - Optimizer state (.pkl files)")
        print(f"     - Metadata (.json files)")
        
    else:
        print("\n❌ No models available for deployment")
        print("   Run training first: python -m src.training.master_trainer")
    
    return model_manager, loaded_agents


if __name__ == "__main__":
    model_manager, loaded_agents = main()
