"""
WealthArena Backend API
FastAPI backend for serving market data, model predictions, and game state
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import json
import logging
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="WealthArena API",
    description="Backend API for WealthArena RL Trading Platform",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data directory paths
DATA_DIR = Path(__file__).parent.parent / "data"
FEATURES_DIR = DATA_DIR / "features"
PROCESSED_DIR = DATA_DIR / "processed"


# ==================== Data Models ====================

class MarketDataRequest(BaseModel):
    symbols: List[str]
    days: int = 30


class PredictionRequest(BaseModel):
    symbol: str
    horizon: int = 1  # days


class PortfolioRequest(BaseModel):
    symbols: List[str]
    weights: List[float]


class ChatRequest(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None


# ==================== Helper Functions ====================

def load_symbol_data(symbol: str, use_features: bool = True) -> Optional[pd.DataFrame]:
    """Load data for a symbol"""
    try:
        if use_features:
            file_path = FEATURES_DIR / f"{symbol}_features.csv"
            if not file_path.exists():
                file_path = PROCESSED_DIR / f"{symbol}_processed.csv"
        else:
            file_path = PROCESSED_DIR / f"{symbol}_processed.csv"
        
        if not file_path.exists():
            return None
        
        df = pd.read_csv(file_path)
        return df
    
    except Exception as e:
        logger.error(f"Error loading data for {symbol}: {e}")
        return None


def generate_mock_prediction(symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
    """Generate mock prediction (replace with actual model in production)"""
    
    # Get latest data
    latest = data.iloc[-1]
    
    # Simple momentum-based mock prediction
    if 'Close' in data.columns and len(data) > 5:
        recent_return = (data['Close'].iloc[-1] / data['Close'].iloc[-6]) - 1
        
        if recent_return > 0.05:
            signal = "BUY"
            confidence = min(0.9, 0.6 + abs(recent_return))
        elif recent_return < -0.05:
            signal = "SELL"
            confidence = min(0.9, 0.6 + abs(recent_return))
        else:
            signal = "HOLD"
            confidence = 0.5
    else:
        signal = "HOLD"
        confidence = 0.5
    
    prediction = {
        'symbol': symbol,
        'signal': signal,
        'confidence': round(confidence, 2),
        'current_price': float(latest.get('Close', 0)),
        'predicted_return_1d': round(np.random.randn() * 0.01, 4),
        'timestamp': datetime.now().isoformat(),
        'features_used': list(data.columns[:10])  # Top 10 features
    }
    
    return prediction


def calculate_portfolio_metrics(portfolio_data: Dict[str, pd.DataFrame], weights: List[float]) -> Dict[str, Any]:
    """Calculate portfolio performance metrics"""
    
    # Calculate returns for each asset
    returns = []
    for symbol, df in portfolio_data.items():
        if 'Returns' in df.columns:
            returns.append(df['Returns'].tail(30))
        elif 'Close' in df.columns:
            returns.append(df['Close'].pct_change().tail(30))
    
    if not returns:
        return {}
    
    # Create returns dataframe
    returns_df = pd.concat(returns, axis=1, keys=portfolio_data.keys())
    
    # Portfolio return
    weights_array = np.array(weights)
    portfolio_returns = (returns_df * weights_array).sum(axis=1)
    
    # Calculate metrics
    total_return = (1 + portfolio_returns).prod() - 1
    annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
    volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = (annualized_return / volatility) if volatility > 0 else 0
    
    # Max drawdown
    cumulative = (1 + portfolio_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    metrics = {
        'total_return': round(float(total_return * 100), 2),  # percentage
        'annualized_return': round(float(annualized_return * 100), 2),
        'volatility': round(float(volatility * 100), 2),
        'sharpe_ratio': round(float(sharpe_ratio), 2),
        'max_drawdown': round(float(max_drawdown * 100), 2),
        'period_days': len(portfolio_returns),
        'timestamp': datetime.now().isoformat()
    }
    
    return metrics


# ==================== API Endpoints ====================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "WealthArena API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "market_data": "/api/market-data",
            "predictions": "/api/predictions",
            "portfolio": "/api/portfolio",
            "chat": "/api/chat",
            "game": "/api/game"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "data_dir_exists": DATA_DIR.exists(),
        "features_dir_exists": FEATURES_DIR.exists()
    }


@app.post("/api/market-data")
async def get_market_data(request: MarketDataRequest):
    """Get market data for specified symbols"""
    
    results = {}
    
    for symbol in request.symbols:
        df = load_symbol_data(symbol, use_features=False)
        
        if df is None:
            results[symbol] = {"error": "Data not available"}
            continue
        
        # Get last N days
        df_recent = df.tail(request.days)
        
        # Convert to dict
        results[symbol] = {
            "data": df_recent.to_dict('records'),
            "latest_price": float(df_recent.iloc[-1].get('Close', 0)),
            "records": len(df_recent),
            "columns": list(df_recent.columns)
        }
    
    return {
        "symbols": request.symbols,
        "days": request.days,
        "data": results,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/predictions")
async def get_predictions(request: PredictionRequest):
    """Get model predictions for a symbol"""
    
    # Load data
    df = load_symbol_data(request.symbol, use_features=True)
    
    if df is None:
        raise HTTPException(status_code=404, detail=f"Data not found for {request.symbol}")
    
    # Generate prediction
    prediction = generate_mock_prediction(request.symbol, df)
    
    # Add explanation
    prediction['explanation'] = {
        'reason': f"Based on recent price momentum and technical indicators",
        'key_factors': [
            "Recent price trend",
            "Relative strength index (RSI)",
            "Moving average convergence"
        ],
        'risk_level': 'Medium',
        'suggested_action': f"{prediction['signal']} with {prediction['confidence']*100:.0f}% confidence"
    }
    
    return prediction


@app.post("/api/portfolio")
async def analyze_portfolio(request: PortfolioRequest):
    """Analyze portfolio performance and risk"""
    
    # Validate weights
    if len(request.symbols) != len(request.weights):
        raise HTTPException(status_code=400, detail="Number of symbols must match number of weights")
    
    if not np.isclose(sum(request.weights), 1.0):
        raise HTTPException(status_code=400, detail="Weights must sum to 1.0")
    
    # Load data for all symbols
    portfolio_data = {}
    for symbol in request.symbols:
        df = load_symbol_data(symbol, use_features=True)
        if df is not None:
            portfolio_data[symbol] = df
    
    if not portfolio_data:
        raise HTTPException(status_code=404, detail="No data found for portfolio symbols")
    
    # Calculate metrics
    metrics = calculate_portfolio_metrics(portfolio_data, request.weights)
    
    # Add holdings info
    holdings = []
    for symbol, weight in zip(request.symbols, request.weights):
        if symbol in portfolio_data:
            df = portfolio_data[symbol]
            holdings.append({
                'symbol': symbol,
                'weight': round(weight * 100, 2),
                'current_price': float(df.iloc[-1].get('Close', 0)),
                'allocation': f"{weight * 100:.1f}%"
            })
    
    return {
        'portfolio': {
            'symbols': request.symbols,
            'weights': request.weights,
            'holdings': holdings
        },
        'metrics': metrics,
        'recommendations': {
            'rebalance_needed': False,
            'risk_level': 'Moderate',
            'diversification_score': round(len(request.symbols) / 10 * 100, 0)
        },
        'timestamp': datetime.now().isoformat()
    }


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Chatbot endpoint for answering questions"""
    
    # Mock chatbot response (replace with RAG model in production)
    message = request.message.lower()
    
    if 'buy' in message or 'purchase' in message:
        response = {
            'answer': "Based on current market analysis, our RL models suggest considering stocks with strong momentum and positive technical indicators. Always ensure proper diversification and risk management.",
            'confidence': 0.85,
            'sources': ['Technical Analysis', 'Model Predictions', 'Risk Assessment']
        }
    elif 'sell' in message:
        response = {
            'answer': "Selling decisions should be based on your risk tolerance, profit targets, and stop-loss levels. Our models can help identify when momentum is weakening or risk is increasing.",
            'confidence': 0.80,
            'sources': ['Risk Management', 'Portfolio Analysis']
        }
    elif 'risk' in message:
        response = {
            'answer': "Risk management is crucial. We recommend: 1) Diversification across assets, 2) Position sizing based on volatility, 3) Setting stop-losses, 4) Regular portfolio rebalancing.",
            'confidence': 0.90,
            'sources': ['Risk Management Framework', 'Portfolio Theory']
        }
    else:
        response = {
            'answer': "I'm here to help you understand market dynamics and model recommendations. You can ask about specific stocks, risk management, portfolio strategies, or how our RL models make decisions.",
            'confidence': 0.75,
            'sources': ['General Knowledge']
        }
    
    return {
        'query': request.message,
        'response': response,
        'timestamp': datetime.now().isoformat(),
        'model': 'WealthArena RAG v1.0'
    }


@app.get("/api/game/leaderboard")
async def get_leaderboard():
    """Get game leaderboard"""
    
    # Mock leaderboard data
    leaderboard = [
        {'rank': 1, 'username': 'TradingPro', 'score': 25.4, 'sharpe': 2.1, 'games': 15},
        {'rank': 2, 'username': 'RL_Master', 'score': 22.8, 'sharpe': 1.9, 'games': 12},
        {'rank': 3, 'username': 'QuanTitan', 'score': 21.5, 'sharpe': 1.8, 'games': 20},
        {'rank': 4, 'username': 'MarketGuru', 'score': 19.2, 'sharpe': 1.6, 'games': 8},
        {'rank': 5, 'username': 'AITrader', 'score': 18.7, 'sharpe': 1.5, 'games': 10},
    ]
    
    return {
        'leaderboard': leaderboard,
        'period': 'all_time',
        'last_updated': datetime.now().isoformat()
    }


@app.get("/api/metrics/summary")
async def get_metrics_summary():
    """Get overall system metrics"""
    
    # Check available symbols
    symbols_with_data = []
    if FEATURES_DIR.exists():
        symbols_with_data = [f.stem.replace('_features', '') 
                            for f in FEATURES_DIR.glob("*_features.csv")]
    
    return {
        'system': {
            'status': 'operational',
            'uptime': '99.9%',
            'last_data_update': datetime.now().isoformat()
        },
        'data': {
            'symbols_tracked': len(symbols_with_data),
            'symbols': symbols_with_data,
            'last_fetch': datetime.now().isoformat()
        },
        'models': {
            'active_models': 5,
            'avg_confidence': 0.78,
            'predictions_today': 150
        },
        'users': {
            'active_users': 42,
            'games_in_progress': 8
        }
    }


# ==================== Main ====================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting WealthArena Backend API...")
    logger.info(f"📁 Data Directory: {DATA_DIR}")
    logger.info(f"📊 Features Directory: {FEATURES_DIR}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True
    )

