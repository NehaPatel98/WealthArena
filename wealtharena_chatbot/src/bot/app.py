"""
WealthArena Trading Bot Dashboard
Educational trading platform with paper trading capabilities.
"""

import json
import random
import asyncio
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import knowledge base
from .kb import search_knowledge_base, get_system_prompt
from .api_v1 import router as v1_router

# Initialize FastAPI app
app = FastAPI(title="WealthArena Trading Bot", version="1.0.0")

# Add CORS middleware for mobile
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://10.0.2.2:8000",  # Android emulator
        "http://127.0.0.1:8000",
        "http://localhost:8080",
        "http://localhost:19006",  # Expo
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include v1 router
app.include_router(v1_router)

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global state for paper trading
trading_state = {
    "balance": 10000.0,
    "positions": {},
    "trades": [],
    "watchlist": ["AAPL", "TSLA", "BTC-USD", "ETH-USD"],
    "alerts": []
}

# Chat sessions storage
chat_sessions = {}

# Mock data generator
class MockDataGenerator:
    """Generates realistic mock OHLCV data and technical indicators."""
    
    def __init__(self):
        self.base_prices = {
            "AAPL": 150.0,
            "TSLA": 200.0,
            "BTC-USD": 45000.0,
            "ETH-USD": 3000.0
        }
    
    def generate_ohlcv(self, symbol: str, days: int = 30) -> List[Dict]:
        """Generate mock OHLCV data for a symbol."""
        base_price = self.base_prices.get(symbol, 100.0)
        data = []
        
        for i in range(days):
            date = datetime.now() - timedelta(days=days-i)
            
            # Generate realistic price movement
            volatility = 0.02 if "USD" in symbol else 0.01  # Crypto more volatile
            change = random.uniform(-volatility, volatility)
            price = base_price * (1 + change)
            base_price = price
            
            # Generate OHLC from base price
            high = price * random.uniform(1.001, 1.02)
            low = price * random.uniform(0.98, 0.999)
            open_price = price * random.uniform(0.995, 1.005)
            close = price
            
            volume = random.randint(1000000, 10000000)
            
            data.append({
                "date": date.strftime("%Y-%m-%d"),
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close, 2),
                "volume": volume
            })
        
        return data
    
    def calculate_sma(self, prices: List[float], period: int) -> List[Optional[float]]:
        """Calculate Simple Moving Average."""
        sma = []
        for i in range(len(prices)):
            if i < period - 1:
                sma.append(None)
            else:
                avg = sum(prices[i-period+1:i+1]) / period
                sma.append(round(avg, 2))
        return sma
    
    def calculate_ema(self, prices: List[float], period: int) -> List[Optional[float]]:
        """Calculate Exponential Moving Average."""
        if not prices:
            return []
        
        ema = [None] * (period - 1)
        multiplier = 2 / (period + 1)
        
        # First EMA value is SMA
        first_ema = sum(prices[:period]) / period
        ema.append(round(first_ema, 2))
        
        for i in range(period, len(prices)):
            ema_value = (prices[i] * multiplier) + (ema[-1] * (1 - multiplier))
            ema.append(round(ema_value, 2))
        
        return ema
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> List[Optional[float]]:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return [None] * len(prices)
        
        rsi = [None] * period
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            gains.append(max(change, 0))
            losses.append(max(-change, 0))
        
        for i in range(period, len(prices)):
            period_gains = gains[i-period:i]
            period_losses = losses[i-period:i]
            
            avg_gain = sum(period_gains) / period
            avg_loss = sum(period_losses) / period
            
            if avg_loss == 0:
                rsi.append(100)
            else:
                rs = avg_gain / avg_loss
                rsi_value = 100 - (100 / (1 + rs))
                rsi.append(round(rsi_value, 2))
        
        return rsi

# Initialize data generator
data_gen = MockDataGenerator()

# Pydantic models
class TradeRequest(BaseModel):
    symbol: str
    action: str  # "buy" or "sell"
    quantity: float
    price: Optional[float] = None

class TradeResponse(BaseModel):
    success: bool
    message: str
    new_balance: float
    position: Optional[Dict] = None

class ChatRequest(BaseModel):
    message: str
    symbol: Optional[str] = None
    mode: Optional[str] = None  # "teach", "analyze", "risk"

class ChatResponse(BaseModel):
    reply: str
    sources: List[str]
    suggestions: List[str]

# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page."""
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "balance": trading_state["balance"],
        "positions": trading_state["positions"],
        "watchlist": trading_state["watchlist"]
    })

@app.get("/healthz")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/learn")
async def get_learning_content():
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
    return content

@app.get("/api/analyze")
async def analyze_asset(symbol: str):
    """Analyze an asset with technical indicators and signals."""
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
            "explanation": "RSI below 30 suggests the asset may be oversold and could bounce back. Consider a potential entry point, but always use proper risk management."
        })
    elif current_rsi > 70:
        signals.append({
            "type": "sell",
            "indicator": "RSI",
            "message": f"RSI at {current_rsi:.1f} indicates overbought conditions",
            "explanation": "RSI above 70 suggests the asset may be overbought and could see a pullback. Consider taking profits or reducing position size."
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
    
    return {
        "symbol": symbol,
        "current_price": current_price,
        "ohlcv": ohlcv_data[-10:],  # Last 10 days
        "indicators": {
            "sma_20": sma_20[-10:],
            "ema_12": ema_12[-10:],
            "rsi": rsi[-10:]
        },
        "signals": signals,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/papertrade", response_model=TradeResponse)
async def paper_trade(trade: TradeRequest):
    """Execute a paper trade."""
    symbol = trade.symbol.upper()
    action = trade.action.lower()
    quantity = trade.quantity
    
    if action not in ["buy", "sell"]:
        return TradeResponse(
            success=False,
            message="Invalid action. Use 'buy' or 'sell'",
            new_balance=trading_state["balance"]
        )
    
    # Get current price (mock)
    current_price = data_gen.base_prices.get(symbol, 100.0)
    trade_price = trade.price or current_price
    total_cost = quantity * trade_price
    
    if action == "buy":
        if total_cost > trading_state["balance"]:
            return TradeResponse(
                success=False,
                message="Insufficient balance for this trade",
                new_balance=trading_state["balance"]
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
        trading_state["trades"].append({
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "action": "buy",
            "quantity": quantity,
            "price": trade_price,
            "total": total_cost
        })
        
        return TradeResponse(
            success=True,
            message=f"Bought {quantity} shares of {symbol} at ${trade_price:.2f}",
            new_balance=trading_state["balance"],
            position=trading_state["positions"][symbol]
        )
    
    else:  # sell
        if symbol not in trading_state["positions"] or trading_state["positions"][symbol]["quantity"] < quantity:
            return TradeResponse(
                success=False,
                message="Insufficient shares to sell",
                new_balance=trading_state["balance"]
            )
        
        # Execute sell
        trading_state["balance"] += total_cost
        trading_state["positions"][symbol]["quantity"] -= quantity
        
        if trading_state["positions"][symbol]["quantity"] == 0:
            del trading_state["positions"][symbol]
        
        # Record trade
        trading_state["trades"].append({
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "action": "sell",
            "quantity": quantity,
            "price": trade_price,
            "total": total_cost
        })
        
        return TradeResponse(
            success=True,
            message=f"Sold {quantity} shares of {symbol} at ${trade_price:.2f}",
            new_balance=trading_state["balance"],
            position=trading_state["positions"].get(symbol)
        )

@app.get("/api/state")
async def get_trading_state():
    """Get current trading state."""
    # Calculate unrealized P&L
    total_pnl = 0
    for symbol, position in trading_state["positions"].items():
        current_price = data_gen.base_prices.get(symbol, 100.0)
        unrealized_pnl = (current_price - position["avg_price"]) * position["quantity"]
        total_pnl += unrealized_pnl
    
    return {
        "balance": trading_state["balance"],
        "positions": trading_state["positions"],
        "trades": trading_state["trades"][-10:],  # Last 10 trades
        "watchlist": trading_state["watchlist"],
        "alerts": trading_state["alerts"],
        "total_pnl": round(total_pnl, 2),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/reset")
async def reset_trading_state():
    """Reset trading state to initial values."""
    global trading_state
    trading_state = {
        "balance": 10000.0,
        "positions": {},
        "trades": [],
        "watchlist": ["AAPL", "TSLA", "BTC-USD", "ETH-USD"],
        "alerts": []
    }
    return {"message": "Trading state reset successfully"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, http_request: Request):
    """Chat endpoint for educational trading guidance."""
    try:
        # Get or create client session
        client_id = http_request.cookies.get("client_id")
        if not client_id:
            client_id = str(uuid.uuid4())
        
        # Initialize session if new
        if client_id not in chat_sessions:
            chat_sessions[client_id] = {
                "messages": [],
                "created_at": datetime.now()
            }
        
        # Search knowledge base
        relevant_sections = search_knowledge_base(request.message, top_k=3)
        
        # Generate response using LLM or fallback
        reply, sources, suggestions = await generate_chat_response(
            request.message, 
            request.symbol, 
            request.mode, 
            relevant_sections
        )
        
        # Store in session (keep last 20 messages)
        chat_sessions[client_id]["messages"].append({
            "timestamp": datetime.now().isoformat(),
            "user_message": request.message,
            "bot_reply": reply,
            "sources": sources
        })
        
        # Keep only last 20 messages
        if len(chat_sessions[client_id]["messages"]) > 20:
            chat_sessions[client_id]["messages"] = chat_sessions[client_id]["messages"][-20:]
        
        return ChatResponse(
            reply=reply,
            sources=sources,
            suggestions=suggestions
        )
        
    except Exception as e:
        return ChatResponse(
            reply="I apologize, but I encountered an error. Please try again with a different question about trading education.",
            sources=["Error handling"],
            suggestions=["Try asking about RSI", "Ask about risk management", "Learn about moving averages"]
        )

async def generate_chat_response(message: str, symbol: Optional[str], mode: Optional[str], relevant_sections: List[Dict]) -> tuple[str, List[str], List[str]]:
    """Generate chat response using LLM or fallback rules."""
    
    # Check for API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    # Try LLM providers if available
    if openai_key:
        return await call_openai(message, symbol, mode, relevant_sections)
    elif groq_key:
        return await call_groq(message, symbol, mode, relevant_sections)
    else:
        return generate_fallback_response(message, symbol, mode, relevant_sections)


async def call_groq(message: str, symbol: Optional[str], mode: Optional[str], relevant_sections: List[Dict]) -> tuple[str, List[str], List[str]]:
    """Call Groq API for chat response."""
    try:
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # Build context from relevant sections
        context = "\n\n".join([section["content"] for section in relevant_sections])
        
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": f"Context: {context}\n\nUser question: {message}"}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        reply = response.choices[0].message.content
        sources = [section["title"] for section in relevant_sections]
        suggestions = ["Explain RSI", "Position sizing 1% rule", "What is a stop-loss?", "Analyze BTC now"]
        
        return reply, sources, suggestions
        
    except Exception as e:
        return generate_fallback_response(message, symbol, mode, relevant_sections)

def generate_fallback_response(message: str, symbol: Optional[str], mode: Optional[str], relevant_sections: List[Dict]) -> tuple[str, List[str], List[str]]:
    """Generate fallback response using rules and knowledge base."""
    message_lower = message.lower()
    
    # Safety checks
    if any(word in message_lower for word in ["buy", "sell", "trade", "invest", "money", "profit"]):
        reply = "I can only provide educational information about trading concepts. For actual trading decisions, please consult with a qualified financial advisor. Remember to always practice with paper trading first!"
        sources = ["Safety Guidelines"]
        suggestions = ["Learn about risk management", "Understand technical indicators", "Practice with paper trading"]
        return reply, sources, suggestions
    
    # RSI explanations
    if "rsi" in message_lower:
        reply = """RSI (Relative Strength Index) is a momentum oscillator that measures the speed and change of price movements on a scale of 0-100.

Key RSI levels:
- RSI < 30: Oversold conditions - potential buying opportunity (but always use proper risk management)
- RSI > 70: Overbought conditions - potential selling opportunity
- RSI 40-60: Neutral zone

Remember: RSI is just one indicator. Always combine it with other analysis and never risk more than you can afford to lose. Practice with paper trading first!"""
        sources = ["Technical Indicators"]
        suggestions = ["Learn about SMA", "Understand risk management", "Practice with paper trading"]
        return reply, sources, suggestions
    
    # Moving averages
    if any(word in message_lower for word in ["sma", "ema", "moving average"]):
        reply = """Moving averages help smooth out price data to identify trends:

SMA (Simple Moving Average): Average price over a specific period
EMA (Exponential Moving Average): Gives more weight to recent prices

Common strategies:
- SMA(20) crossing above SMA(50): Potential uptrend
- EMA(12) crossing above SMA(20): Potential momentum shift

Remember: These are educational concepts. Always practice with paper trading and never risk more than 1-2% of your account per trade!"""
        sources = ["Technical Indicators"]
        suggestions = ["Learn about RSI", "Understand risk management", "Practice with paper trading"]
        return reply, sources, suggestions
    
    # Risk management
    if any(word in message_lower for word in ["risk", "position", "sizing", "stop", "loss"]):
        reply = """Risk management is crucial for successful trading:

Key principles:
- Never risk more than 1-2% of your account per trade
- Always use stop losses to limit potential losses
- Diversify your portfolio across different assets
- Maintain proper risk-reward ratios (aim for 2:1 or better)

Remember: This is educational content only. Always practice with paper trading first and consult with financial professionals before making real trading decisions!"""
        sources = ["Risk Management"]
        suggestions = ["Learn about technical indicators", "Understand position sizing", "Practice with paper trading"]
        return reply, sources, suggestions
    
    # General trading education
    if any(word in message_lower for word in ["learn", "teach", "explain", "what", "how"]):
        reply = f"""I'm here to help you learn about trading concepts! Based on your question, here are some key educational points:

{relevant_sections[0]['content'] if relevant_sections else 'Trading involves buying and selling financial instruments with the goal of making a profit, but it comes with significant risks.'}

Remember: This is educational content only. Always practice with paper trading first and never risk more than you can afford to lose!"""
        sources = [section["title"] for section in relevant_sections] if relevant_sections else ["General Education"]
        suggestions = ["Learn about RSI", "Understand risk management", "Practice with paper trading", "Ask about technical indicators"]
        return reply, sources, suggestions
    
    # Default response
    reply = """I'm here to help you learn about trading concepts! I can explain technical indicators like RSI, moving averages, risk management principles, and more.

Remember: This is educational content only. Always practice with paper trading first and never risk more than you can afford to lose!"""
    sources = ["General Education"]
    suggestions = ["Explain RSI", "Position sizing 1% rule", "What is a stop-loss?", "Learn about moving averages"]
    return reply, sources, suggestions

def find_free_port():
    """Find a free port to run the server on."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

# Note: Server startup is handled by run.py
# To run directly: python src/bot/app.py
if __name__ == "__main__":
    import socket
    
    port = find_free_port()
    print(f"\n🚀 WealthArena Trading Bot Dashboard")
    print(f"📊 Educational Trading Platform")
    print(f"🌐 Starting server on: http://localhost:{port}")
    print(f"💡 Features: Learn, Analyze, Paper Trade")
    print(f"⚠️  Educational only - Not financial advice")
    print(f"\nPress Ctrl+C to stop the server\n")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port, 
        log_level="info",
        reload=False
    )
