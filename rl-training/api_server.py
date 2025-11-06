#!/usr/bin/env python3
"""
WealthArena AI Models API Server
This server hosts all 5 trained models and provides REST API endpoints
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# FastAPI imports
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import our model management
from src.training.model_checkpoint import ProductionModelManager, ModelCheckpoint

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="WealthArena AI Trading Models API",
    description="API for WealthArena AI Trading Models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model management
model_manager = None
checkpoint_manager = None
available_agents = ["asx_stocks", "currency_pairs", "cryptocurrencies", "etf", "commodities"]

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    agent_name: str
    input_data: List[List[float]]  # 2D array of features
    symbol: str = None

class PredictionResponse(BaseModel):
    agent_name: str
    prediction: List[float]
    confidence: float
    timestamp: str
    symbol: str = None

class HealthResponse(BaseModel):
    status: str
    available_agents: List[str]
    timestamp: str

class ModelInfo(BaseModel):
    agent_name: str
    checkpoint_id: str
    created_at: str
    status: str

# Initialize models on startup
@app.on_event("startup")
async def startup_event():
    """Initialize all models when the server starts"""
    global model_manager, checkpoint_manager
    
    try:
        logger.info("🚀 Starting WealthArena AI Models API Server...")
        
        # Initialize model managers
        checkpoint_manager = ModelCheckpoint("checkpoints")
        model_manager = ProductionModelManager("checkpoints")
        
        # Load all available models
        loaded_agents = []
        for agent_name in available_agents:
            try:
                success = model_manager.load_agent_model(agent_name)
                if success:
                    loaded_agents.append(agent_name)
                    logger.info(f"✅ {agent_name} model loaded successfully")
                else:
                    logger.warning(f"⚠️ Failed to load {agent_name} model")
            except Exception as e:
                logger.error(f"❌ Error loading {agent_name}: {e}")
        
        logger.info(f"🎯 {len(loaded_agents)} models loaded and ready: {loaded_agents}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize models: {e}")
        raise

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API and models are healthy"""
    try:
        loaded_agents = []
        if model_manager:
            for agent_name in available_agents:
                try:
                    # Test if model is loaded
                    test_input = np.random.randn(1, 140)  # 7 assets * 20 features
                    _ = model_manager.get_model_prediction(agent_name, test_input)
                    loaded_agents.append(agent_name)
                except:
                    pass
        
        return HealthResponse(
            status="healthy" if loaded_agents else "unhealthy",
            available_agents=loaded_agents,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            available_agents=[],
            timestamp=datetime.now().isoformat()
        )

# Get model information
@app.get("/models", response_model=List[ModelInfo])
async def get_models():
    """Get information about all available models"""
    models = []
    
    if not checkpoint_manager:
        raise HTTPException(status_code=500, detail="Model manager not initialized")
    
    for agent_name in available_agents:
        try:
            checkpoints = checkpoint_manager.list_checkpoints(agent_name)
            if checkpoints:
                latest = checkpoints[0]
                models.append(ModelInfo(
                    agent_name=agent_name,
                    checkpoint_id=latest["checkpoint_id"],
                    created_at=latest["created_at"],
                    status="loaded" if agent_name in model_manager._loaded_models else "available"
                ))
            else:
                models.append(ModelInfo(
                    agent_name=agent_name,
                    checkpoint_id="none",
                    created_at="unknown",
                    status="not_found"
                ))
        except Exception as e:
            models.append(ModelInfo(
                agent_name=agent_name,
                checkpoint_id="error",
                created_at="unknown",
                status=f"error: {str(e)}"
            ))
    
    return models

# Get prediction from a specific model
@app.post("/predict", response_model=PredictionResponse)
async def get_prediction(request: PredictionRequest):
    """Get prediction from a specific AI agent"""
    
    if not model_manager:
        raise HTTPException(status_code=500, detail="Model manager not initialized")
    
    if request.agent_name not in available_agents:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid agent name. Available agents: {available_agents}"
        )
    
    try:
        # Convert input data to numpy array
        input_array = np.array(request.input_data, dtype=np.float32)
        
        # Get prediction from the model
        prediction = model_manager.get_model_prediction(request.agent_name, input_array)
        
        # Calculate confidence (simple heuristic)
        confidence = float(np.mean(np.abs(prediction)))
        
        return PredictionResponse(
            agent_name=request.agent_name,
            prediction=prediction.tolist(),
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
            symbol=request.symbol
        )
        
    except Exception as e:
        logger.error(f"Prediction error for {request.agent_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Get prediction from all models
@app.post("/predict-all", response_model=List[PredictionResponse])
async def get_all_predictions(request: PredictionRequest):
    """Get predictions from all available models"""
    
    if not model_manager:
        raise HTTPException(status_code=500, detail="Model manager not initialized")
    
    predictions = []
    
    for agent_name in available_agents:
        try:
            # Convert input data to numpy array
            input_array = np.array(request.input_data, dtype=np.float32)
            
            # Get prediction from the model
            prediction = model_manager.get_model_prediction(agent_name, input_array)
            
            # Calculate confidence
            confidence = float(np.mean(np.abs(prediction)))
            
            predictions.append(PredictionResponse(
                agent_name=agent_name,
                prediction=prediction.tolist(),
                confidence=confidence,
                timestamp=datetime.now().isoformat(),
                symbol=request.symbol
            ))
            
        except Exception as e:
            logger.warning(f"Failed to get prediction from {agent_name}: {e}")
            # Continue with other models
    
    return predictions

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "WealthArena AI Trading Models API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "models": "/models",
            "predict": "/predict",
            "predict-all": "/predict-all",
            "docs": "/docs"
        },
        "available_agents": available_agents
    }

# Main function to run the server
if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
