"""
WealthArena API v1 - Mobile Integration Backend
Versioned API endpoints for mobile SDKs
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from .app import (
    trading_state, 
    chat_sessions, 
    data_gen, 
    generate_chat_response,
    search_knowledge_base
)

# Initialize v1 router
router = APIRouter(prefix="/v1", tags=["v1"])

# Security
security = HTTPBearer(auto_error=False)

# Pydantic models for v1 API
class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    symbol: Optional[str] = Field(None, description="Optional symbol for context")
    mode: Optional[str] = Field(None, description="Chat mode: teach, analyze, risk")

class ChatResponse(BaseModel):
    reply: str
    sources: List[str]
    suggestions: List[str]
    timestamp: str

class AnalyzeRequest(BaseModel):
    symbol: str = Field(..., description="Symbol to analyze")

class AnalyzeResponse(BaseModel):
    symbol: str
    current_price: float
    indicators: Dict[str, Any]
    signals: List[Dict[str, Any]]
    timestamp: str

class TradeRequest(BaseModel):
    action: str = Field(..., description="buy or sell")
    symbol: str = Field(..., description="Trading symbol")
    quantity: float = Field(..., description="Quantity to trade")
    price: Optional[float] = Field(None, description="Optional price (uses current if not provided)")

class TradeResponse(BaseModel):
    success: bool
    message: str
    new_balance: float
    position: Optional[Dict[str, Any]] = None
    timestamp: str

class StateResponse(BaseModel):
    balance: float
    positions: Dict[str, Any]
    trades: List[Dict[str, Any]]
    total_pnl: float
    timestamp: str

class LearnResponse(BaseModel):
    lessons: List[Dict[str, Any]]
    quizzes: List[Dict[str, Any]]
    timestamp: str

class EventSchema(BaseModel):
    type: str = Field(..., description="Event type: TradePlaced, SignalGenerated, LessonCompleted")
    data: Dict[str, Any] = Field(..., description="Event data")
    timestamp: str

# Auth dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Optional authentication - only required if AUTH_REQUIRED=true"""
    auth_required = os.getenv("AUTH_REQUIRED", "false").lower() == "true"
    
    if not auth_required:
        return None
    
    if not credentials:
        raise HTTPException(status_code=401, detail="Authorization header required")
    
    # Simple token validation (in production, use proper JWT validation)
    expected_token = os.getenv("API_TOKEN", "wealtharena-mobile-token")
    if credentials.credentials != expected_token:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return {"user_id": "mobile_user"}

# CORS headers for mobile
@router.middleware("http")
async def add_cors_headers(request, call_next):
    response = await call_next(request)
    
    # Allow mobile emulator origins
    allowed_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000", 
        "http://10.0.2.2:8000",  # Android emulator
        "http://127.0.0.1:8000",
        "http://localhost:8080",
        "http://localhost:19006",  # Expo
    ]
    
    origin = request.headers.get("origin")
    if origin in allowed_origins:
        response.headers["Access-Control-Allow-Origin"] = origin
    
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    
    return response

# API Endpoints
@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    user: Optional[Dict] = Depends(get_current_user)
):
    """Chat with the educational trading bot."""
    try:
        # Search knowledge base
        relevant_sections = search_knowledge_base(request.message, top_k=3)
        
        # Generate response
        reply, sources, suggestions = await generate_chat_response(
            request.message,
            request.symbol,
            request.mode,
            relevant_sections
        )
        
        return ChatResponse(
            reply=reply,
            sources=sources,
            suggestions=suggestions,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_asset(
    request: AnalyzeRequest,
    user: Optional[Dict] = Depends(get_current_user)
):
    """Analyze an asset with technical indicators."""
    symbol = request.symbol.upper()
    
    if symbol not in data_gen.base_prices:
        raise HTTPException(status_code=404, detail="Symbol not supported")
    
    # Generate mock data
    ohlcv_data = data_gen.generate_ohlcv(symbol, 30)
    closes = [candle["close"] for candle in ohlcv_data]
    
    # Calculate indicators
    sma_20 = data_gen.calculate_sma(closes, 20)
    ema_12 = data_gen.calculate_ema(closes, 12)
    rsi = data_gen.calculate_rsi(closes, 14)
    
    # Generate signals
    signals = []
    current_price = closes[-1]
    current_rsi = rsi[-1] if rsi[-1] is not None else 50
    
    # RSI signals
    if current_rsi < 30:
        signals.append({
            "type": "buy",
            "indicator": "RSI",
            "message": f"RSI at {current_rsi:.1f} indicates oversold conditions",
            "explanation": "RSI below 30 suggests the asset may be oversold and could bounce back."
        })
    elif current_rsi > 70:
        signals.append({
            "type": "sell", 
            "indicator": "RSI",
            "message": f"RSI at {current_rsi:.1f} indicates overbought conditions",
            "explanation": "RSI above 70 suggests the asset may be overbought and could see a pullback."
        })
    
    # Moving average signals
    if sma_20[-1] and ema_12[-1]:
        if ema_12[-1] > sma_20[-1] and ema_12[-2] <= sma_20[-2]:
            signals.append({
                "type": "buy",
                "indicator": "MA Cross",
                "message": "EMA(12) crossed above SMA(20) - bullish signal",
                "explanation": "When the faster EMA crosses above the slower SMA, it often indicates a potential uptrend beginning."
            })
        elif ema_12[-1] < sma_20[-1] and ema_12[-2] >= sma_20[-2]:
            signals.append({
                "type": "sell",
                "indicator": "MA Cross", 
                "message": "EMA(12) crossed below SMA(20) - bearish signal",
                "explanation": "When the faster EMA crosses below the slower SMA, it often indicates a potential downtrend beginning."
            })
    
    return AnalyzeResponse(
        symbol=symbol,
        current_price=current_price,
        indicators={
            "sma_20": sma_20[-10:],
            "ema_12": ema_12[-10:],
            "rsi": rsi[-10:]
        },
        signals=signals,
        timestamp=datetime.now().isoformat()
    )

@router.get("/state", response_model=StateResponse)
async def get_trading_state(
    user: Optional[Dict] = Depends(get_current_user)
):
    """Get current trading state."""
    # Calculate unrealized P&L
    total_pnl = 0
    for symbol, position in trading_state["positions"].items():
        current_price = data_gen.base_prices.get(symbol, 100.0)
        unrealized_pnl = (current_price - position["avg_price"]) * position["quantity"]
        total_pnl += unrealized_pnl
    
    return StateResponse(
        balance=trading_state["balance"],
        positions=trading_state["positions"],
        trades=trading_state["trades"][-10:],  # Last 10 trades
        total_pnl=round(total_pnl, 2),
        timestamp=datetime.now().isoformat()
    )

@router.post("/papertrade", response_model=TradeResponse)
async def paper_trade(
    request: TradeRequest,
    user: Optional[Dict] = Depends(get_current_user)
):
    """Execute a paper trade."""
    symbol = request.symbol.upper()
    action = request.action.lower()
    quantity = request.quantity
    
    if action not in ["buy", "sell"]:
        return TradeResponse(
            success=False,
            message="Invalid action. Use 'buy' or 'sell'",
            new_balance=trading_state["balance"],
            timestamp=datetime.now().isoformat()
        )
    
    # Get current price (mock)
    current_price = data_gen.base_prices.get(symbol, 100.0)
    trade_price = request.price or current_price
    total_cost = quantity * trade_price
    
    if action == "buy":
        if total_cost > trading_state["balance"]:
            return TradeResponse(
                success=False,
                message="Insufficient balance for this trade",
                new_balance=trading_state["balance"],
                timestamp=datetime.now().isoformat()
            )
        
        # Execute buy
        trading_state["balance"] -= total_cost
        if symbol in trading_state["positions"]:
            trading_state["positions"][symbol]["quantity"] += quantity
            trading_state["positions"][symbol]["avg_price"] = (
                (trading_state["positions"][symbol]["avg_price"] * 
                 trading_state["positions"][symbol]["quantity"] + total_cost) /
                (trading_state["positions"][symbol]["quantity"])
            )
        else:
            trading_state["positions"][symbol] = {
                "quantity": quantity,
                "avg_price": trade_price
            }
        
        # Record trade
        trade_record = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "action": "buy",
            "quantity": quantity,
            "price": trade_price,
            "total": total_cost
        }
        trading_state["trades"].append(trade_record)
        
        return TradeResponse(
            success=True,
            message=f"Bought {quantity} shares of {symbol} at ${trade_price:.2f}",
            new_balance=trading_state["balance"],
            position=trading_state["positions"][symbol],
            timestamp=datetime.now().isoformat()
        )
    
    else:  # sell
        if symbol not in trading_state["positions"] or trading_state["positions"][symbol]["quantity"] < quantity:
            return TradeResponse(
                success=False,
                message="Insufficient shares to sell",
                new_balance=trading_state["balance"],
                timestamp=datetime.now().isoformat()
            )
        
        # Execute sell
        trading_state["balance"] += total_cost
        trading_state["positions"][symbol]["quantity"] -= quantity
        
        if trading_state["positions"][symbol]["quantity"] == 0:
            del trading_state["positions"][symbol]
        
        # Record trade
        trade_record = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "action": "sell",
            "quantity": quantity,
            "price": trade_price,
            "total": total_cost
        }
        trading_state["trades"].append(trade_record)
        
        return TradeResponse(
            success=True,
            message=f"Sold {quantity} shares of {symbol} at ${trade_price:.2f}",
            new_balance=trading_state["balance"],
            position=trading_state["positions"].get(symbol),
            timestamp=datetime.now().isoformat()
        )

@router.get("/learn", response_model=LearnResponse)
async def get_learning_content(
    user: Optional[Dict] = Depends(get_current_user)
):
    """Get educational content and quizzes."""
    content = {
        "lessons": [
            {
                "id": "basics",
                "title": "Trading Basics",
                "content": "Learn the fundamentals of trading, including key terms and concepts.",
                "topics": ["What is trading?", "Market types", "Order types", "Risk management"]
            },
            {
                "id": "indicators",
                "title": "Technical Indicators", 
                "content": "Understand how to use technical analysis tools.",
                "topics": ["Moving averages", "RSI", "Support and resistance", "Trend analysis"]
            },
            {
                "id": "risk",
                "title": "Risk Management",
                "content": "Learn how to protect your capital and manage risk.",
                "topics": ["Position sizing", "Stop losses", "Diversification", "Risk-reward ratios"]
            }
        ],
        "quizzes": [
            {
                "id": "quiz1",
                "question": "What does RSI below 30 typically indicate?",
                "options": ["Overbought", "Oversold", "Neutral", "Strong uptrend"],
                "correct": 1,
                "explanation": "RSI below 30 indicates oversold conditions, suggesting a potential buying opportunity."
            },
            {
                "id": "quiz2", 
                "question": "What is the purpose of a stop-loss order?",
                "options": ["To guarantee profits", "To limit losses", "To increase position size", "To time the market"],
                "correct": 1,
                "explanation": "Stop-loss orders help limit potential losses by automatically selling when price reaches a predetermined level."
            }
        ]
    }
    
    return LearnResponse(
        lessons=content["lessons"],
        quizzes=content["quizzes"],
        timestamp=datetime.now().isoformat()
    )

@router.get("/healthz")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "wealtharena-mobile-api",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }
